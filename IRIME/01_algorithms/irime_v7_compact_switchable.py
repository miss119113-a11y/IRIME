"""Compact IRIME candidate for paper-facing evaluation.

The implementation is organized as three modules rather than five separate
"upgrades":

1. Rank-stabilized recursive initialization.  A known-rank spiked covariance
   model suppresses finite-sample noise-eigenvalue splitting.  Recursive
   projected-subspace selection retains the paper's selective (K>=3) quadratic
   subgrid proposal with exact projected-score verification.
2. Structured finite-budget search.  Differential Subspace Correction (DSC;
   historically "Decoupling"/DSD) supplies a phase-invariant, sphere-aware
   close-pair proposal, followed by the canonical two cyclic conditional RIME
   searches.  DSC is implemented once in ``alg/dsc.py`` and imported here and by
   the paper variant, so the ablation and the paper path run bit-identical DSC
   (bounds guards + azimuth wrap + |cos phi_c| floor included).
3. Observation-driven terminal refinement.  At high array occupancy, below-noise
   evidence repair and stochastic-ML polishing are used together; otherwise a
   weak/close-mode covariance refinement or the standard bounded DML polish is
   selected from observed spectrum/DSC evidence.

Compared with IRIME-V4, this version removes the universal K=2 subgrid-fusion
extension and the duplicate pair-geometry gate, fixes the quadratic Hessian
stencil, and makes the DSC tangent direction invariant to eigenvector phase and
normalized in the local spherical metric.  No ground-truth DOA, true spacing, or
true SNR is used by any branch.

Ablation switches (Sec. 4.6):
  enable_rank_stabilization  -> mechanism I
  enable_dsd                 -> mechanism D  (shared canonical DSC)
  enable_terminal_controller -> mechanism J
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from alg.rime_music import _optimize_rime
from alg.single_opt_common import _joint_seed_population
from irime_v3_clean import (
    ConditionalDMLObjective,
    _orthogonal_complement,
    _rime_shape,
    _projected_signal_basis,
    _steering,
    _validate_bounds,
    _grid_axis,
    _spiked_covariance_projection,
    _nonnegative_normal_equations,
    _high_occupancy_evidence_swap_repair,
    _joint_stochastic_ml_polish,
    _joint_uncorrelated_covariance_polish,
    _joint_dml_polish,
    _trace_projection_gain,
    EPS,
    _CANONICAL_RIME_W,
    _AP_SWEEPS,
)
from alg.dsc import differential_subspace_decouple


def irime_v7_switchable(
    observation: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_range: np.ndarray,
    phi_range: np.ndarray,
    source_num: int,
    *,
    seed: int = 42,
    coarse_evidence_step_deg: float = 2.5,
    continuous_search_budget_per_coordinate: int = 896,
    return_diagnostics: bool = False,
    enable_rank_stabilization: bool = True,
    enable_dsd: bool = True,
    enable_terminal_controller: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray]:
    """Compact IRIME candidate: original selective fusion plus stabilized search and merged structural refinement."""
    k_sources = int(source_num)
    sensor_count = int(array_pos.shape[0])
    snapshots = int(observation.shape[1])
    sample_cov = (observation @ observation.conj().T) / snapshots
    sample_cov = 0.5 * (sample_cov + sample_cov.conj().T)
    
    # Module 1: rank-stabilized recursive initialization
    sample_eigenvalues = np.linalg.eigvalsh(sample_cov)[::-1]
    if enable_rank_stabilization:
        covariance, spiked_diag = _spiked_covariance_projection(sample_cov, k_sources)
        estimated_noise_variance = spiked_diag.get(
            "estimated_noise_variance",
            float(np.mean(np.linalg.eigvalsh(sample_cov)[:max(1, sensor_count - k_sources)])),
        )
    else:
        covariance = sample_cov.copy()
        noise_tail = np.linalg.eigvalsh(sample_cov)[:max(1, sensor_count - k_sources)]
        estimated_noise_variance = float(np.mean(noise_tail))
        spiked_diag = {"enabled": False, "estimated_noise_variance": estimated_noise_variance}
    covariance_trace = float(np.real(np.trace(covariance)))
    
    theta_bounds = _validate_bounds(theta_range, "theta_range")
    phi_bounds = _validate_bounds(phi_range, "phi_range")
    step = float(coarse_evidence_step_deg)
    
    theta_axis = _grid_axis(theta_bounds[0], theta_bounds[1], step, periodic=True)
    phi_axis = _grid_axis(phi_bounds[0], phi_bounds[1], step, periodic=False)
    grid_points = np.column_stack([np.repeat(theta_axis, len(phi_axis)), np.tile(phi_axis, len(theta_axis))])
    all_steering = _steering(grid_points, array_pos, wavelength)
    
    # Recursive projected-subspace evidence grid with selective subgrid proposals
    selected = []
    evidence_candidates = []
    high_occupancy = bool(2 * k_sources >= sensor_count)
    
    for q in range(k_sources) if enable_rank_stabilization else range(0):
        prior = np.asarray(selected, dtype=float) if selected else np.empty((0, 2), dtype=float)
        prior_steering = _steering(prior, array_pos, wavelength) if len(prior) else np.empty((sensor_count, 0), dtype=complex)
        complement, _ = _orthogonal_complement(prior_steering, sensor_count)
        
        projected_cov = complement @ covariance @ complement
        vals, vecs = np.linalg.eigh(projected_cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
        
        rem_rank = max(1, k_sources - q)
        basis = vecs[:, :rem_rank]
        
        qa = complement @ all_steering
        den = np.sum(np.abs(qa)**2, axis=0)
        num = np.sum(np.abs(basis.conj().T @ qa)**2, axis=0)
        score = np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 128.0 * EPS * sensor_count)
        best_idx = int(np.argmax(score))
        
        surface = score.reshape(len(theta_axis), len(phi_axis))
        local_max = (surface == maximum_filter(surface, size=3, mode=["wrap", "nearest"]))
        peak_indices = np.argwhere(local_max)
        
        if high_occupancy and len(peak_indices) > 0:
            evidence_candidates.extend(np.column_stack((theta_axis[peak_indices[:, 0]], phi_axis[peak_indices[:, 1]])))
            
        # Quadratic subgrid proposal; retained only for K>=3 below
        predictions = []
        for ti, pi in peak_indices:
            if pi == 0 or pi == len(phi_axis) - 1:
                continue
            tm, tp = (ti - 1) % len(theta_axis), (ti + 1) % len(theta_axis)
            stencil = np.asarray([
                surface[ti, pi], surface[tm, pi], surface[tp, pi],
                surface[ti, pi-1], surface[ti, pi+1],
                surface[tm, pi-1], surface[tm, pi+1],
                surface[tp, pi-1], surface[tp, pi+1],
            ], dtype=float)
            if not np.isfinite(stencil).all():
                continue
            f = surface[ti, pi]
            gradient = np.array([(surface[tp, pi] - surface[tm, pi]) / (2 * step), (surface[ti, pi+1] - surface[ti, pi-1]) / (2 * step)])
            cross = (surface[tp, pi+1] - surface[tp, pi-1] - surface[tm, pi+1] + surface[tm, pi-1]) / (4 * step**2)
            hessian = np.array([[(surface[tp, pi] - 2*f + surface[tm, pi]) / step**2, cross], [cross, (surface[ti, pi+1] - 2*f + surface[ti, pi-1]) / step**2]])
            if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
                continue
            eigen = np.linalg.eigvalsh(hessian)
            if eigen[-1] >= -EPS * max(1.0, abs(eigen[0])):
                continue
            offset = np.clip(-np.linalg.solve(hessian, gradient), -step, step)
            point = np.array([theta_axis[ti], phi_axis[pi]]) + offset
            point[0] = (point[0] - theta_bounds[0]) % (theta_bounds[1] - theta_bounds[0]) + theta_bounds[0]
            point[1] = np.clip(point[1], *phi_bounds)
            predictions.append(point)

        # Preserve the paper's selective-fusion boundary.  K=2 close pairs are
        # handled by the dedicated differential branch instead of a universal
        # interpolation layer.
        if k_sources < 3:
            predictions = []
            
        center = grid_points[best_idx].copy()
        selected_score = float(score[best_idx])
        if predictions:
            pred_pts = np.asarray(predictions)
            qa_pred = complement @ _steering(pred_pts, array_pos, wavelength)
            den_p = np.sum(np.abs(qa_pred)**2, axis=0)
            num_p = np.sum(np.abs(basis.conj().T @ qa_pred)**2, axis=0)
            actual_p = np.divide(num_p, den_p, out=np.full_like(num_p, -np.inf), where=den_p > 128.0 * EPS * sensor_count)
            best_p = int(np.argmax(actual_p))
            if actual_p[best_p] > selected_score + 64.0 * EPS:
                center = pred_pts[best_p].copy()
                selected_score = float(actual_p[best_p])
                
        selected.append(center.copy())
        
    if not enable_rank_stabilization:
        # Paper -I ablation: genuinely replace recursive I by 2.5-degree MUSIC peaks.
        # Compute spectrum on SAME observation; no rank projection or recursion.
        _, ev = np.linalg.eigh(sample_cov)
        en = ev[:, :sensor_count-k_sources]
        ps = en @ en.conj().T
        denom = np.real(np.sum(np.conj(all_steering) * (ps @ all_steering), axis=0))
        music_surface = (1.0/np.maximum(denom,1e-12)).reshape(len(theta_axis),len(phi_axis))
        peak_mask = music_surface == maximum_filter(music_surface, size=3, mode=["wrap","nearest"])
        peak_indices = np.flatnonzero(peak_mask.ravel())
        # MUSIC spectral local maxima, then fallback to highest remaining bins
        order = sorted(peak_indices, key=lambda i: -music_surface.ravel()[i])
        peak_set = set(peak_indices.tolist())
        order.extend(i for i in np.argsort(denom) if i not in peak_set)
        selected = []
        for idx in order:
            candidate = grid_points[idx]
            if all(np.linalg.norm([((candidate[0]-p[0]+180)%360)-180,candidate[1]-p[1]])>=step for p in selected):
                selected.append(candidate.copy())
                if len(selected)==k_sources: break
        if len(selected)!=k_sources: raise RuntimeError('MUSIC initializer did not find K peaks')
    estimates = np.asarray(selected, dtype=float)
    
    # Module 2a: analytic Differential Subspace Decoupling (DSD) before RIME
    if enable_dsd:
        estimates, dsd_diag = differential_subspace_decouple(
            estimates, covariance, array_pos, wavelength, theta_bounds, phi_bounds
        )
    else:
        dsd_diag = {"applied": False, "reason": "ablation_disabled"}
    
    # Module 2b: canonical two cyclic conditional RIME sweeps
    pop_size, iters, _ = _rime_shape(sensor_count, int(continuous_search_budget_per_coordinate))
    rng = np.random.default_rng(seed)
    
    for sweep in range(_AP_SWEEPS):
        for s_idx in range(k_sources):
            other = np.delete(estimates, s_idx, axis=0)
            other_steering = _steering(other, array_pos, wavelength) if len(other) else np.empty((sensor_count, 0), dtype=complex)
            complement, _ = _orthogonal_complement(other_steering, sensor_count)
            conditional = ConditionalDMLObjective(covariance, array_pos, wavelength, complement, covariance_trace)
            before_point = estimates[s_idx].copy()
            population = _joint_seed_population(
                before_point, pop_size, theta_bounds, phi_bounds, rng,
                theta_jitter=max(1.0, 2.0 * step / 3.0), phi_jitter=max(0.75, step / 2.0)
            )
            result = _optimize_rime(
                conditional, theta_bounds, phi_bounds, pop_size, iters,
                population_init=population, rng=rng, W=_CANONICAL_RIME_W,
                use_opposition_init=True, use_differential_move=False, use_adaptive_restart=False
            )
            before_loss = float(conditional.first_value)
            proposed = np.asarray(result["best_pos"], dtype=float)
            proposed_loss = float(result["best_obj"])
            if proposed_loss <= before_loss + 64.0 * EPS:
                estimates[s_idx] = proposed
                
    # Module 3a: high-occupancy below-noise evidence repair
    evidence_repair_diag = {"triggered": False}
    if enable_terminal_controller and high_occupancy and evidence_candidates:
        evidence_array = np.asarray(evidence_candidates, dtype=float).reshape(-1, 2)
        estimates, evidence_repair_diag = _high_occupancy_evidence_swap_repair(
            estimates, evidence_array, sample_cov, array_pos, wavelength
        )
        
    # Module 3b: observation-driven terminal likelihood refinement
    weakest_signal_eigenvalue = float(sample_eigenvalues[k_sources - 1])
    weak_mode_energy_bound = float((sensor_count / k_sources) * estimated_noise_variance)
    weak_mode_gate = bool(weakest_signal_eigenvalue <= weak_mode_energy_bound)

    # One terminal refinement controller.  The close-pair branch reuses the
    # DSD decision instead of introducing a second manifold-geometry gate.
    # High occupancy keeps the SML branch because the projector-DML criterion
    # becomes increasingly permissive as the signal-subspace dimension grows.
    if not enable_terminal_controller:
        joint_polish = None
        rank_regime = "terminal_refinement_disabled"
    elif high_occupancy:
        joint_polish = _joint_stochastic_ml_polish
        rank_regime = "high_occupancy_stochastic_ml"
    elif weak_mode_gate or bool(dsd_diag.get("applied", False)):
        joint_polish = _joint_uncorrelated_covariance_polish
        rank_regime = "weak_or_close_structured_covariance_fit"
    else:
        joint_polish = _joint_dml_polish
        rank_regime = "standard_dml"
        
    if joint_polish is None:
        polish_diag = {"skipped": True, "eval_count": 0}
    else:
        estimates, polish_diag = joint_polish(
            estimates, covariance, array_pos, wavelength,
            theta_bounds, phi_bounds, step,
            int(continuous_search_budget_per_coordinate),
        )
    
    order = np.argsort(estimates[:, 0])
    estimates = estimates[order]
    theta_est = estimates[:, 0]
    phi_est = estimates[:, 1]
    
    total_eval_count = int(
        k_sources * len(grid_points)
        + _AP_SWEEPS * k_sources * pop_size * iters
        + polish_diag.get("eval_count", 60)
        + (25 if dsd_diag.get("applied", False) else 0)
    )

    if return_diagnostics:
        return theta_est, phi_est, {
            "method": "IRIME-V7-Switchable",
            "rank_stabilization_enabled": bool(enable_rank_stabilization),
            "dsd_enabled": bool(enable_dsd),
            "terminal_controller_enabled": bool(enable_terminal_controller),
            "rank_regime": rank_regime,
            "dsd_diagnostics": dsd_diag,
            "evidence_repair_diagnostics": evidence_repair_diag,
            "polish_diagnostics": polish_diag,
            "objective_eval_count": total_eval_count,
            "objective_eval_count_type": "legacy_estimate_not_for_exact_cost_comparison",
            "objective_eval_count_exact": False,
        }
    return theta_est, phi_est
