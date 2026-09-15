"""Compact public runner for the principal SPFRIME experiments.

This script is intentionally smaller than the internal development harness. It keeps the
paper-level configurations needed to reproduce the principal SPFRIME curves/tables and
writes CSV summaries that can be regenerated without any external data files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from spfrime import steering, trap_dml_rime
from experiments.paper_protocol import (
    CONDITIONAL_BUDGET,
    EVIDENCE_STEP_DEG,
    MONTE_CARLO_TRIALS,
    NOMINAL_PHI,
    NOMINAL_THETA,
    RESOLUTION_ANCHOR,
    RESOLUTION_SNRS_DB,
    RESOLUTION_SNAPSHOTS,
    RESOLUTION_SPACINGS_DEG,
    angle_grids,
    uca_positions,
)


def spherical_error_deg(a, b):
    a = np.deg2rad(np.asarray(a, float))
    b = np.deg2rad(np.asarray(b, float))
    ua = np.column_stack((np.cos(a[:, 1]) * np.cos(a[:, 0]),
                          np.cos(a[:, 1]) * np.sin(a[:, 0]),
                          np.sin(a[:, 1])))
    ub = np.column_stack((np.cos(b[:, 1]) * np.cos(b[:, 0]),
                          np.cos(b[:, 1]) * np.sin(b[:, 0]),
                          np.sin(b[:, 1])))
    c = np.clip(ua @ ub.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


def assigned_errors(true_doa, est_doa):
    C = spherical_error_deg(true_doa, est_doa)
    r, c = linear_sum_assignment(C)
    return C[r, c]


def simulate(true_doa, snapshots, snr_db, rng, wavelength=1.0, per_source_snr=False):
    pos = uca_positions(wavelength)
    A = steering(true_doa, pos, wavelength)
    k = len(true_doa)
    # Deterministic narrowband source sequences; exact code-level frequencies are not
    # scientifically material to the paper and can be changed without changing the protocol.
    n = np.arange(snapshots)
    freq = np.linspace(0.11, 0.31, k, endpoint=True)
    S = np.exp(1j * 2 * np.pi * freq[:, None] * n[None, :])
    clean = A @ S
    if per_source_snr:
        source_power = 1.0
        noise_var = source_power * 10.0 ** (-snr_db / 10.0)
    else:
        noise_var = (np.linalg.norm(clean, "fro") ** 2 / (pos.shape[0] * snapshots)) * 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_var / 2) * (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape))
    return clean + noise, pos


def estimate(X, pos, k, seed):
    tg, pg = angle_grids()
    th, ph = trap_dml_rime(
        X, pos, 1.0, tg, pg, k,
        evidence_step=EVIDENCE_STEP_DEG,
        conditional_budget=CONDITIONAL_BUDGET,
        seed=seed,
    )
    return np.column_stack((th, ph))


def second_source_same_elevation(anchor, delta_deg):
    """Solve azimuth displacement giving exact great-circle spacing at fixed elevation."""
    th0, ph0 = np.deg2rad(anchor)
    d = np.deg2rad(delta_deg)
    # cos(d)=sin^2(phi)+cos^2(phi) cos(dtheta)
    q = (np.cos(d) - np.sin(ph0) ** 2) / max(np.cos(ph0) ** 2, 1e-15)
    dth = np.arccos(np.clip(q, -1.0, 1.0))
    return np.array([(np.rad2deg(th0 + dth)) % 360.0, np.rad2deg(ph0)])


def run_main_snr(trials, out_dir):
    rows = []
    true_doa = np.column_stack((NOMINAL_THETA, NOMINAL_PHI))
    for snr in (-5, -3, -1, 1, 3, 5):
        sq = []
        for r in range(trials):
            rng = np.random.default_rng(100000 + 1000 * int(snr + 10) + r)
            X, pos = simulate(true_doa, snapshots=100, snr_db=float(snr), rng=rng)
            est = estimate(X, pos, 3, seed=900000 + r)
            e = assigned_errors(true_doa, est)
            sq.extend((e ** 2).tolist())
        rows.append((snr, float(np.sqrt(np.mean(sq)))))
    _write(out_dir / "main_snr_spfrime.csv", ["snr_db", "pooled_rmse_deg"], rows)


def run_resolution(trials, out_dir):
    rows = []
    anchor = np.array(RESOLUTION_ANCHOR, float)
    for snr in RESOLUTION_SNRS_DB:
        for delta in RESOLUTION_SPACINGS_DEG:
            success = 0
            for r in range(trials):
                other = second_source_same_elevation(anchor, float(delta))
                truth = np.vstack((anchor, other))
                rng = np.random.default_rng(2000000 + int(100 * snr) + int(100 * delta) + r)
                X, pos = simulate(truth, RESOLUTION_SNAPSHOTS, float(snr), rng, per_source_snr=True)
                est = estimate(X, pos, 2, seed=3000000 + r)
                e = assigned_errors(truth, est)
                est_sep = float(spherical_error_deg(est[:1], est[1:2])[0, 0])
                strict = bool(np.all(e < 0.49 * delta) and est_sep >= 0.5 * delta)
                success += int(strict)
            rows.append((snr, delta, success / trials))
    _write(out_dir / "resolution_strict_probability.csv",
           ["snr_db", "spacing_deg", "strict_probability"], rows)


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=MONTE_CARLO_TRIALS)
    ap.add_argument("--experiment", choices=("main-snr", "resolution", "all"), default="all")
    ap.add_argument("--out", type=Path, default=Path("results/public_reproduction"))
    args = ap.parse_args()
    if args.experiment in ("main-snr", "all"):
        run_main_snr(args.trials, args.out)
    if args.experiment in ("resolution", "all"):
        run_resolution(args.trials, args.out)
    print(f"Results written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
