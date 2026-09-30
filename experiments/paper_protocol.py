"""Frozen paper-level settings for the IRIME manuscript."""
from __future__ import annotations

import numpy as np

M = 12
RADIUS_OVER_WAVELENGTH = 1.0 / (4.0 * np.sin(np.pi / M))
H_DEG = 2.5
CONDITIONAL_BUDGET = 896
CYCLIC_SWEEPS = 2
MONTE_CARLO_TRIALS = 500
THETA_RANGE = (0.0, 360.0)
PHI_RANGE = (0.0, 90.0)

NOMINAL_THETA = np.array([30.42, 120.27, 240.51], dtype=float)
NOMINAL_PHI = np.array([60.39, 29.42, 45.55], dtype=float)
SNR_SWEEP_DB = np.array([-5, -3, -1, 1, 3, 5], dtype=float)
SNAPSHOT_SWEEP = np.array([50, 100, 200, 300, 500], dtype=int)

SOURCE_COUNT_K = tuple(range(2, 10))
SOURCE_COUNT_SNAPSHOTS = 300
SOURCE_COUNT_PER_SOURCE_SNR_DB = 10.0
GOLDEN_FRACTION = (np.sqrt(5.0) - 1.0) / 2.0

RESOLUTION_ANCHOR = (83.17, 35.29)
RESOLUTION_SPACINGS_DEG = np.array(
    [0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
     3.0, 3.25, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0], dtype=float)
RESOLUTION_SNRS_DB = (10.0, 20.0)
RESOLUTION_SNAPSHOTS = 1024


def uca_positions(wavelength: float = 1.0, m: int = M) -> np.ndarray:
    """Horizontal UCA with adjacent chord length wavelength/2."""
    radius = float(wavelength) / (4.0 * np.sin(np.pi / int(m)))
    alpha = 2.0 * np.pi * np.arange(m) / m
    return np.column_stack((radius * np.cos(alpha), radius * np.sin(alpha), np.zeros(m)))


def source_count_geometry(k: int, alpha_deg: float, beta: float) -> np.ndarray:
    """Frozen K-source geometry used by the source-count protocol."""
    idx = np.arange(int(k), dtype=float)
    az = np.mod(float(alpha_deg) + idx * 360.0 / int(k), 360.0)
    el = 20.0 + 50.0 * np.mod(float(beta) + idx * GOLDEN_FRACTION, 1.0)
    D = np.column_stack((az, el))
    return D[np.argsort(D[:, 0])]


def resolution_pair(delta_deg: float) -> np.ndarray:
    """Equal-elevation pair with exact great-circle separation delta_deg."""
    th1, ph = RESOLUTION_ANCHOR
    d = np.deg2rad(float(delta_deg))
    p = np.deg2rad(ph)
    c = (np.cos(d) - np.sin(p) ** 2) / max(np.cos(p) ** 2, np.finfo(float).eps)
    c = np.clip(c, -1.0, 1.0)
    dth = np.rad2deg(np.arccos(c))
    return np.array([[th1, ph], [np.mod(th1 + dth, 360.0), ph]], dtype=float)
