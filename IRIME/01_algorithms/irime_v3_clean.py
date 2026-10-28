"""Clean, parameter-light IRIME candidate for the replacement gate.

The paper IRIME path is retained: rank-safe TRAP initialization, the published
fusion boundary, and two cyclic conditional-DML RIME sweeps.  V3 replaces the
last local polish with one rank-balanced structural rule.  When ``2K >= M``, it
first replaces any below-noise false direction from the already computed TRAP
local maxima, then flattens only the noise eigenvalues and minimizes the
concentrated stochastic-ML likelihood.  At lower occupancy it invokes an
uncorrelated covariance fit only when the weakest sample mode fails an
energy-balance test, or when a two-source manifold is geometrically
ill-conditioned; otherwise it preserves the paper DML polish.  Every boundary
is derived from ``K``, ``M``, the sample spectrum, and the array manifold rather
than tuned to an SNR or angular spacing.

There is no SNR switch, spacing threshold, collision barrier, damped
pseudo-projector, extra sweep, or truth-dependent branch.  The only public
numerical controls remain the evidence-grid step ``h`` and continuous-search
budget ``B``.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter


_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from alg.rime_music import _optimize_rime
from alg.single_opt_common import _joint_seed_population, build_music_objective


EPS = np.finfo(float).eps
_CANONICAL_RIME_W = 5.0
_AP_SWEEPS = 2


def _spiked_covariance_projection(
    covariance: np.ndarray,
    k_sources: int,
) -> tuple[np.ndarray, dict]:
    """Project a sample covariance onto the known-rank spiked model.

    The K leading sample eigenpairs are retained.  Every noise-subspace
    eigenvalue is replaced by their arithmetic mean, the maximum-likelihood
    isotropic-noise estimate for the known signal rank.  No shrinkage strength,
    threshold, or scene-dependent control is introduced.
    """
    sample = np.asarray(covariance, dtype=complex)
    if sample.ndim != 2 or sample.shape[0] != sample.shape[1]:
        raise ValueError("covariance must be square")
    sensor_count = int(sample.shape[0])
    k_sources = int(k_sources)
    if not 1 <= k_sources < sensor_count:
        raise ValueError("k_sources must satisfy 1 <= K < sensor count")
    hermitian = 0.5 * (sample + sample.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    order = np.argsort(eigenvalues)[::-1]
    values = np.real(eigenvalues[order])
    vectors = eigenvectors[:, order]
    noise_variance = float(np.mean(values[k_sources:]))
    projected_values = values.copy()
    projected_values[k_sources:] = noise_variance
    projected = (vectors * projected_values[None, :]) @ vectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    return projected, {
        "applied": True,
        "rule": "retain_K_signal_eigenvalues_and_average_noise_eigenvalues",
        "known_signal_rank": k_sources,
        "estimated_noise_variance": noise_variance,
        "noise_eigenvalue_std_before": float(np.std(values[k_sources:])),
        "noise_eigenvalue_std_after": 0.0,
        "user_selected_parameters": [],
    }


def _nonnegative_normal_equations(
    gram: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve the tiny non-negative normal equations by monotone active removal."""
    active = np.ones(len(rhs), dtype=bool)
    coefficients = np.zeros(len(rhs), dtype=float)
    for _ in range(len(rhs)):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        solution = np.linalg.lstsq(
            gram[np.ix_(indices, indices)], rhs[indices], rcond=None
        )[0]
        if np.all(solution >= 0.0):
            coefficients[indices] = solution
            break
        active[indices[solution < 0.0]] = False
    return coefficients


def _uncorrelated_covariance_set_fit(
    points: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
) -> tuple[float, int, np.ndarray]:
    """Return normalized covariance residual, manifold rank, and powers/noise."""
    candidate = np.asarray(points, dtype=float)
    covariance = 0.5 * (
        np.asarray(covariance, dtype=complex)
        + np.asarray(covariance, dtype=complex).conj().T
    )
    k_sources = int(len(candidate))
    sensor_count = int(covariance.shape[0])
    steering = _steering(candidate, array_pos, wavelength)
    singular = np.linalg.svd(steering, compute_uv=False)
    tolerance = EPS * float(max(steering.shape)) * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    if rank < k_sources:
        return float(1.0 + k_sources - rank), rank, np.zeros(k_sources + 1)
    manifold_gram = steering.conj().T @ steering
    model_gram = np.empty((k_sources + 1, k_sources + 1), dtype=float)
    model_gram[:k_sources, :k_sources] = np.abs(manifold_gram) ** 2
    model_gram[:k_sources, -1] = sensor_count
    model_gram[-1, :k_sources] = sensor_count
    model_gram[-1, -1] = sensor_count
    rhs = np.empty(k_sources + 1, dtype=float)
    rhs[:k_sources] = np.real(
        np.sum(np.conj(steering) * (covariance @ steering), axis=0)
    )
    rhs[-1] = float(np.real(np.trace(covariance)))
    coefficients = _nonnegative_normal_equations(model_gram, rhs)
    covariance_norm_sq = float(np.real(np.vdot(covariance, covariance)))
    residual_sq = float(
        covariance_norm_sq
        - 2.0 * coefficients @ rhs
        + coefficients @ model_gram @ coefficients
    )
    normalized = max(0.0, residual_sq) / max(covariance_norm_sq, EPS)
    return float(normalized), rank, coefficients


def _high_occupancy_evidence_swap_repair(
    estimates: np.ndarray,
    evidence_candidates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
) -> tuple[np.ndarray, dict]:
    """Replace one below-noise direction using computed TRAP local maxima.

    A swap is attempted only when the weakest fitted source power does not
    clear the simultaneously fitted white-noise variance.  This is a physical
    identifiability boundary, not a tuned ratio.  Every retained swap strictly
    lowers the same parameter-free covariance residual.  The pass uses no
    angular exclusion, candidate-count cutoff, or user threshold.
    """
    current = np.asarray(estimates, dtype=float).copy()
    candidates = np.asarray(evidence_candidates, dtype=float)
    if candidates.size == 0:
        candidates = np.empty((0, 2), dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("evidence candidates must have shape (n, 2)")
    current_loss, current_rank, coefficients = _uncorrelated_covariance_set_fit(
        current, covariance, array_pos, wavelength
    )
    initial_loss = float(current_loss)
    initial_zero_indices = np.flatnonzero(coefficients[:-1] == 0.0)
    initial_weakest_index = int(np.argmin(coefficients[:-1]))
    initial_weakest_power = float(coefficients[initial_weakest_index])
    initial_noise_variance = float(coefficients[-1])
    initial_below_noise = bool(initial_weakest_power <= initial_noise_variance)
    set_calls = 1
    swaps: list[dict] = []
    for _ in range(len(current)):
        weakest_index = int(np.argmin(coefficients[:-1]))
        weakest_power = float(coefficients[weakest_index])
        noise_variance = float(coefficients[-1])
        if weakest_power > noise_variance or len(candidates) == 0:
            break
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for candidate in candidates:
            proposed = current.copy()
            proposed[weakest_index] = candidate
            loss, rank, proposed_coefficients = _uncorrelated_covariance_set_fit(
                proposed, covariance, array_pos, wavelength
            )
            set_calls += 1
            if rank < len(current):
                continue
            if best is None or loss < best[0]:
                best = (
                    float(loss),
                    weakest_index,
                    proposed,
                    proposed_coefficients,
                )
        if best is None or best[0] >= current_loss - 64.0 * EPS:
            break
        old_direction = current[best[1]].copy()
        current_loss, source_index, current, coefficients = best
        swaps.append(
            {
                "source_index": int(source_index),
                "old_direction": old_direction.tolist(),
                "new_direction": current[source_index].tolist(),
                "covariance_fit_loss": float(current_loss),
            }
        )
    return current, {
        "method": "below_noise_power_TRAP_evidence_swap_repair",
        "power_gate_rule": "weakest_fitted_source_power <= fitted_noise_variance",
        "triggered": initial_below_noise,
        "initial_zero_power_source_indices": initial_zero_indices.tolist(),
        "initial_weakest_source_index": initial_weakest_index,
        "initial_weakest_source_power": initial_weakest_power,
        "initial_fitted_noise_variance": initial_noise_variance,
        "initial_weakest_power_below_noise": initial_below_noise,
        "candidate_count": int(len(candidates)),
        "initial_loss": initial_loss,
        "final_loss": float(current_loss),
        "accepted_swap_count": int(len(swaps)),
        "swaps": swaps,
        "joint_set_objective_calls": int(set_calls),
        "equivalent_angle_evaluations": int(set_calls * len(current)),
        "initial_manifold_rank": int(current_rank),
        "user_selected_parameters": [],
    }


def _validate_bounds(values: np.ndarray, name: str) -> tuple[float, float]:
    axis = np.asarray(values, dtype=float).reshape(-1)
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain at least two finite values")
    lower = float(np.min(axis))
    upper = float(np.max(axis))
    if upper <= lower:
        raise ValueError(f"{name} must span a positive interval")
    return lower, upper


def _grid_axis(lower: float, upper: float, step: float, *, periodic: bool) -> np.ndarray:
    """Build a bounded axis and omit the duplicate periodic endpoint."""
    lower = float(lower)
    upper = float(upper)
    step = float(step)
    if not np.isfinite([lower, upper, step]).all() or upper <= lower or step <= 0.0:
        raise ValueError("grid bounds and step must be finite, with upper > lower and step > 0")
    tol = 64.0 * EPS * max(1.0, abs(lower), abs(upper), abs(step))
    count = int(np.floor((upper - lower) / step + tol))
    values = lower + step * np.arange(count + 1, dtype=float)
    values = values[values <= upper + tol]
    if periodic:
        values = values[values < upper - tol]
    elif values.size == 0 or not np.isclose(values[-1], upper, atol=tol, rtol=0.0):
        values = np.r_[values, upper]
    else:
        values[-1] = upper
    if values.size == 0:
        raise ValueError("coarse evidence grid is empty")
    return values


def _steering(points: np.ndarray, array_pos: np.ndarray, wavelength: float) -> np.ndarray:
    """Return the M-by-N steering matrix for degree-valued (theta, phi) rows."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.all(np.isfinite(pts)):
        raise ValueError("angle points must have shape (n, 2) and contain finite values")
    theta = np.deg2rad(pts[:, 0])
    phi = np.deg2rad(pts[:, 1])
    directions = np.column_stack(
        (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi))
    )
    phase = -2.0 * np.pi / float(wavelength) * (np.asarray(array_pos) @ directions.T)
    return np.exp(1j * phase)


def _orthogonal_complement(columns: np.ndarray, sensor_count: int) -> tuple[np.ndarray, int]:
    """Return I-P_columns using the standard machine-precision SVD rank."""
    matrix = np.asarray(columns, dtype=complex)
    identity = np.eye(int(sensor_count), dtype=complex)
    if matrix.size == 0:
        return identity, 0
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or float(singular[0]) <= 0.0:
        return identity, 0
    tolerance = EPS * float(max(matrix.shape)) * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    if rank <= 0:
        return identity, 0
    basis = left[:, :rank]
    projector = identity - basis @ basis.conj().T
    return 0.5 * (projector + projector.conj().T), rank


def _projected_signal_basis(
    signal_vectors: np.ndarray,
    complement: np.ndarray,
    remaining_rank: int,
) -> tuple[np.ndarray, int]:
    projected = complement @ np.asarray(signal_vectors, dtype=complex)
    left, singular, _ = np.linalg.svd(projected, full_matrices=False)
    if singular.size == 0 or float(singular[0]) <= 0.0:
        return np.empty((projected.shape[0], 0), dtype=complex), 0
    tolerance = EPS * float(max(projected.shape)) * float(singular[0])
    numerical_rank = int(np.sum(singular > tolerance))
    rank = min(int(remaining_rank), numerical_rank)
    return left[:, :rank], rank


def _trace_projection_gain(
    points: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
) -> tuple[float, int]:
    steering = _steering(points, array_pos, wavelength)
    complement, rank = _orthogonal_complement(steering, steering.shape[0])
    projector = np.eye(steering.shape[0], dtype=complex) - complement
    return float(np.real(np.trace(projector @ covariance))), int(rank)


@dataclass
class ConditionalDMLObjective:
    """Continuous conditional DML loss for one source coordinate."""

    covariance: np.ndarray
    array_pos: np.ndarray
    wavelength: float
    complement: np.ndarray
    covariance_trace: float
    eval_count: int = 0
    batch_calls: int = 0
    first_value: float | None = None

    def objective_batch(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 2 or not np.all(np.isfinite(pts)):
            raise ValueError("conditional DML points must have shape (n, 2) and be finite")
        steering = _steering(pts, self.array_pos, self.wavelength)
        projected = self.complement @ steering
        denominator = np.sum(np.abs(projected) ** 2, axis=0).real
        covariance_projected = self.covariance @ projected
        numerator = np.sum(np.conj(projected) * covariance_projected, axis=0).real
        gain = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, -np.inf, dtype=float),
            where=denominator > 128.0 * EPS * max(1, projected.shape[0]),
        )
        # Positive normalization preserves the DML maximizer and makes the
        # standard minimization-oriented RIME kernel numerically well scaled.
        loss = 1.0 - gain / max(float(self.covariance_trace), EPS)
        loss = np.where(np.isfinite(loss), np.maximum(loss, EPS), np.inf)
        if self.first_value is None and len(loss):
            self.first_value = float(loss[0])
        self.eval_count += int(len(pts))
        self.batch_calls += 1
        return loss.astype(float)

    def objective(self, point: np.ndarray) -> float:
        return float(self.objective_batch(np.asarray(point, dtype=float))[0])


def _rime_shape(sensor_count: int, budget_per_coordinate: int) -> tuple[int, int, int]:
    """Derive the population from one budget; iteration depth follows array size."""
    # Opposition initialization evaluates 3P points and each iteration P.
    # ceil(log2(M)) gives four refinement rounds for the paper's M=12 UCA.
    iterations = max(2, int(np.ceil(np.log2(max(2, int(sensor_count))))))
    coefficient = iterations + 3
    population = int(budget_per_coordinate) // coefficient
    if population < 8:
        raise ValueError(
            "continuous_search_budget_per_coordinate is too small for the derived RIME schedule"
        )
    used = population * coefficient
    return population, iterations, used




def _joint_dml_polish(
    estimates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    local_radius_deg: float,
    evaluation_budget_per_coordinate: int,
) -> tuple[np.ndarray, dict]:
    """Run the paper's bounded joint deterministic-ML polish unchanged."""
    start = np.asarray(estimates, dtype=float).copy()
    k_sources = int(len(start))
    covariance_trace = float(np.real(np.trace(covariance)))
    theta_span = float(theta_bounds[1] - theta_bounds[0])
    counters = {"joint_set_calls": 0, "equivalent_angle_evaluations": 0}

    def points_from_offsets(offsets: np.ndarray) -> np.ndarray:
        points = start + np.asarray(offsets, dtype=float).reshape(k_sources, 2)
        points[:, 0] = (
            (points[:, 0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        )
        points[:, 1] = np.clip(points[:, 1], phi_bounds[0], phi_bounds[1])
        return points

    def loss(offsets: np.ndarray) -> float:
        points = points_from_offsets(offsets)
        steering = _steering(points, array_pos, wavelength)
        left, singular, _ = np.linalg.svd(steering, full_matrices=False)
        tolerance = EPS * float(max(steering.shape)) * float(singular[0])
        rank = int(np.sum(singular > tolerance))
        counters["joint_set_calls"] += 1
        counters["equivalent_angle_evaluations"] += k_sources
        if rank < k_sources:
            return float(1.0 + k_sources - rank)
        basis = left[:, :k_sources]
        captured = float(np.real(np.trace(basis.conj().T @ covariance @ basis)))
        return float(max(EPS, 1.0 - captured / max(covariance_trace, EPS)))

    radius = float(local_radius_deg)
    bounds: list[tuple[float, float]] = []
    for point in start:
        bounds.extend(
            [
                (-radius, radius),
                (
                    max(-radius, phi_bounds[0] - float(point[1])),
                    min(radius, phi_bounds[1] - float(point[1])),
                ),
            ]
        )
    initial_offsets = np.zeros(2 * k_sources, dtype=float)
    initial_loss = loss(initial_offsets)
    result = minimize(
        loss,
        initial_offsets,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxfun": int(evaluation_budget_per_coordinate) * k_sources,
            "maxiter": int(evaluation_budget_per_coordinate),
        },
    )
    proposed = points_from_offsets(result.x)
    proposed_loss = float(result.fun)
    accepted = bool(
        np.isfinite(proposed_loss) and proposed_loss <= initial_loss + 64.0 * EPS
    )
    final = proposed if accepted else start
    final_loss = proposed_loss if accepted else initial_loss
    return final, {
        "method": "bounded_joint_DML_L-BFGS-B",
        "local_radius_rule": "one_coarse_evidence_cell",
        "local_radius_deg": radius,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "accepted": accepted,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "joint_set_objective_calls": int(counters["joint_set_calls"]),
        "equivalent_angle_evaluations": int(counters["equivalent_angle_evaluations"]),
    }


def _joint_uncorrelated_covariance_polish(
    estimates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    local_radius_deg: float,
    evaluation_budget_per_coordinate: int,
) -> tuple[np.ndarray, dict]:
    """Fit an uncorrelated-source covariance model in one bounded local step.

    For each angle set, source powers and white-noise variance are eliminated by
    a small non-negative least-squares active set.  The model therefore adds no
    power, noise, balance, or penalty parameter.  All angle bounds remain the
    same one-cell bounds as the paper joint polish.
    """
    start = np.asarray(estimates, dtype=float).copy()
    covariance = 0.5 * (
        np.asarray(covariance, dtype=complex)
        + np.asarray(covariance, dtype=complex).conj().T
    )
    k_sources = int(len(start))
    sensor_count = int(covariance.shape[0])
    theta_span = float(theta_bounds[1] - theta_bounds[0])
    counters = {"joint_set_calls": 0, "equivalent_angle_evaluations": 0}

    def points_from_offsets(offsets: np.ndarray) -> np.ndarray:
        points = start + np.asarray(offsets, dtype=float).reshape(k_sources, 2)
        points[:, 0] = (
            (points[:, 0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        )
        points[:, 1] = np.clip(points[:, 1], phi_bounds[0], phi_bounds[1])
        return points

    def set_loss(points: np.ndarray) -> tuple[float, int, np.ndarray]:
        return _uncorrelated_covariance_set_fit(
            points, covariance, array_pos, wavelength
        )

    def loss(offsets: np.ndarray) -> float:
        counters["joint_set_calls"] += 1
        counters["equivalent_angle_evaluations"] += k_sources
        value, _, _ = set_loss(points_from_offsets(offsets))
        return value

    radius = float(local_radius_deg)
    bounds: list[tuple[float, float]] = []
    for point in start:
        bounds.extend(
            [
                (-radius, radius),
                (
                    max(-radius, phi_bounds[0] - float(point[1])),
                    min(radius, phi_bounds[1] - float(point[1])),
                ),
            ]
        )
    initial_offsets = np.zeros(2 * k_sources, dtype=float)
    initial_loss = loss(initial_offsets)
    _, initial_rank, initial_coefficients = set_loss(start)
    local_iteration_cap = max(2, int(np.ceil(np.log2(sensor_count))))
    result = minimize(
        loss,
        initial_offsets,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxfun": int(evaluation_budget_per_coordinate) * k_sources,
            "maxiter": local_iteration_cap,
        },
    )
    proposed = points_from_offsets(result.x)
    proposed_loss, proposed_rank, proposed_coefficients = set_loss(proposed)
    accepted = bool(
        np.isfinite(proposed_loss)
        and proposed_rank == k_sources
        and proposed_loss <= initial_loss + 64.0 * EPS
    )
    final = proposed if accepted else start
    final_loss = proposed_loss if accepted else initial_loss
    coefficients = proposed_coefficients if accepted else initial_coefficients
    return final, {
        "method": "bounded_joint_uncorrelated_covariance_fit_L-BFGS-B",
        "inner_rule": "nonnegative_source_powers_plus_white_noise_variance",
        "user_selected_parameters": [],
        "local_radius_rule": "one_coarse_evidence_cell",
        "local_radius_deg": radius,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "accepted": accepted,
        "fitted_source_powers": coefficients[:k_sources].tolist(),
        "fitted_noise_variance": float(coefficients[-1]),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "optimizer_iteration_cap_rule": "ceil(log2(M))",
        "optimizer_iteration_cap": int(local_iteration_cap),
        "joint_set_objective_calls": int(counters["joint_set_calls"]),
        "equivalent_angle_evaluations": int(counters["equivalent_angle_evaluations"]),
        "initial_manifold_rank": int(initial_rank),
        "final_manifold_rank": int(proposed_rank if accepted else initial_rank),
    }


def _joint_stochastic_ml_polish(
    estimates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    local_radius_deg: float,
    evaluation_budget_per_coordinate: int,
) -> tuple[np.ndarray, dict]:
    """Bounded concentrated stochastic-ML polish with no tuning coefficient."""
    start = np.asarray(estimates, dtype=float).copy()
    covariance = 0.5 * (
        np.asarray(covariance, dtype=complex)
        + np.asarray(covariance, dtype=complex).conj().T
    )
    k_sources = int(len(start))
    sensor_count = int(covariance.shape[0])
    noise_dimension = sensor_count - k_sources
    theta_span = float(theta_bounds[1] - theta_bounds[0])
    covariance_trace = float(np.real(np.trace(covariance)))
    counters = {"joint_set_calls": 0, "equivalent_angle_evaluations": 0}

    def points_from_offsets(offsets: np.ndarray) -> np.ndarray:
        points = start + np.asarray(offsets, dtype=float).reshape(k_sources, 2)
        points[:, 0] = (
            (points[:, 0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        )
        points[:, 1] = np.clip(points[:, 1], phi_bounds[0], phi_bounds[1])
        return points

    def set_loss(points: np.ndarray) -> tuple[float, int, float]:
        steering = _steering(points, array_pos, wavelength)
        left, singular, _ = np.linalg.svd(steering, full_matrices=False)
        tolerance = EPS * float(max(steering.shape)) * float(singular[0])
        rank = int(np.sum(singular > tolerance))
        if rank < k_sources:
            return float(np.inf), rank, float("nan")
        basis = left[:, :k_sources]
        signal_block = basis.conj().T @ covariance @ basis
        signal_block = 0.5 * (signal_block + signal_block.conj().T)
        sign, logdet = np.linalg.slogdet(signal_block)
        captured = float(np.real(np.trace(signal_block)))
        noise_variance = max(
            (covariance_trace - captured) / noise_dimension,
            EPS * max(1.0, covariance_trace / sensor_count),
        )
        if sign <= 0 or not np.isfinite(logdet):
            return float(np.inf), rank, noise_variance
        return float(logdet + noise_dimension * np.log(noise_variance)), rank, noise_variance

    def loss(offsets: np.ndarray) -> float:
        counters["joint_set_calls"] += 1
        counters["equivalent_angle_evaluations"] += k_sources
        value, _, _ = set_loss(points_from_offsets(offsets))
        return value

    radius = float(local_radius_deg)
    bounds: list[tuple[float, float]] = []
    for point in start:
        bounds.extend(
            [
                (-radius, radius),
                (
                    max(-radius, phi_bounds[0] - float(point[1])),
                    min(radius, phi_bounds[1] - float(point[1])),
                ),
            ]
        )
    initial_offsets = np.zeros(2 * k_sources, dtype=float)
    initial_loss = loss(initial_offsets)
    _, initial_rank, initial_noise = set_loss(start)
    local_iteration_cap = max(2, int(np.ceil(np.log2(sensor_count))))
    result = minimize(
        loss,
        initial_offsets,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxfun": int(evaluation_budget_per_coordinate) * k_sources,
            "maxiter": local_iteration_cap,
        },
    )
    proposed = points_from_offsets(result.x)
    proposed_loss, proposed_rank, proposed_noise = set_loss(proposed)
    accepted = bool(
        np.isfinite(proposed_loss)
        and proposed_rank == k_sources
        and proposed_loss <= initial_loss + 64.0 * EPS
    )
    final = proposed if accepted else start
    return final, {
        "method": "bounded_joint_concentrated_stochastic_ML_L-BFGS-B",
        "source_covariance_model": "unknown_full_rank",
        "user_selected_parameters": [],
        "local_radius_rule": "one_coarse_evidence_cell",
        "local_radius_deg": radius,
        "initial_loss": float(initial_loss),
        "final_loss": float(proposed_loss if accepted else initial_loss),
        "accepted": accepted,
        "initial_noise_variance": float(initial_noise),
        "final_noise_variance": float(proposed_noise if accepted else initial_noise),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "optimizer_iteration_cap_rule": "ceil(log2(M))",
        "optimizer_iteration_cap": int(local_iteration_cap),
        "joint_set_objective_calls": int(counters["joint_set_calls"]),
        "equivalent_angle_evaluations": int(counters["equivalent_angle_evaluations"]),
        "initial_manifold_rank": int(initial_rank),
        "final_manifold_rank": int(proposed_rank if accepted else initial_rank),
    }


def irime_v3(
    X: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_range: np.ndarray,
    phi_range: np.ndarray,
    k_sources: int,
    *,
    seed: int | None = None,
    coarse_evidence_step_deg: float = 3.0,
    continuous_search_budget_per_coordinate: int = 112,
    return_diagnostics: bool = False,
):
    """Estimate K two-dimensional DOAs with the clean V3 mechanism.

    The returned evaluation count is an exact count of angle candidates scored
    by a projected subspace or conditional set objective.  It is intentionally
    labelled separately from a classical MUSIC pseudospectrum point count.
    """
    X = np.asarray(X, dtype=complex)
    array_pos = np.asarray(array_pos, dtype=float)
    objective = build_music_objective(X, array_pos, wavelength, k_sources)
    k_sources = int(k_sources)
    if not np.isfinite(continuous_search_budget_per_coordinate) or int(continuous_search_budget_per_coordinate) != continuous_search_budget_per_coordinate:
        raise ValueError("continuous search budget must be a finite integer")
    sample_covariance = np.asarray(objective.covariance, dtype=complex)
    signal_vectors = np.asarray(objective.signal_eigenvectors, dtype=complex)
    sensor_count = int(array_pos.shape[0])
    sample_eigenvalues = np.linalg.eigvalsh(
        0.5 * (sample_covariance + sample_covariance.conj().T)
    )[::-1].real
    noise_values = sample_eigenvalues[k_sources:]
    estimated_noise_variance = float(np.mean(noise_values))
    high_occupancy = bool(2 * k_sources >= sensor_count)
    if high_occupancy:
        covariance, spiked_covariance_diagnostics = _spiked_covariance_projection(
            sample_covariance, k_sources
        )
    else:
        covariance = sample_covariance
        spiked_covariance_diagnostics = {
            "applied": False,
            "rule": "not_applied_when_2K_is_less_than_M",
            "known_signal_rank": k_sources,
            "estimated_noise_variance": estimated_noise_variance,
            "noise_eigenvalue_std_before": float(np.std(noise_values)),
            "noise_eigenvalue_std_after": float(np.std(noise_values)),
            "user_selected_parameters": [],
        }
    covariance_trace = float(np.real(np.trace(covariance)))
    if not np.isfinite(covariance_trace) or covariance_trace <= 0.0:
        raise ValueError("sample covariance must have positive finite trace")

    theta_bounds = _validate_bounds(theta_range, "theta_range")
    phi_bounds = _validate_bounds(phi_range, "phi_range")
    step = float(coarse_evidence_step_deg)
    theta_axis = _grid_axis(*theta_bounds, step, periodic=True)
    phi_axis = _grid_axis(*phi_bounds, step, periodic=False)
    theta_mesh, phi_mesh = np.meshgrid(theta_axis, phi_axis, indexing="ij")
    grid_points = np.column_stack((theta_mesh.ravel(), phi_mesh.ravel()))
    grid_steering = _steering(grid_points, array_pos, wavelength)

    selected: list[np.ndarray] = []
    evidence_candidates: list[np.ndarray] = []
    trap_trace: list[dict] = []
    trap_evaluations = 0
    for q in range(k_sources):
        selected_matrix = (
            _steering(np.asarray(selected), array_pos, wavelength)
            if selected
            else np.empty((sensor_count, 0), dtype=complex)
        )
        complement, selected_rank = _orthogonal_complement(selected_matrix, sensor_count)
        remaining_rank = k_sources - q
        projected_basis, retained_rank = _projected_signal_basis(
            signal_vectors, complement, remaining_rank
        )
        if retained_rank <= 0:
            raise RuntimeError("projected signal subspace lost all remaining dimensions")
        projected_grid = complement @ grid_steering
        denominator = np.sum(np.abs(projected_grid) ** 2, axis=0).real
        numerator = np.sum(
            np.abs(projected_basis.conj().T @ projected_grid) ** 2, axis=0
        ).real
        score = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, -np.inf, dtype=float),
            where=denominator > 128.0 * EPS * max(1, sensor_count),
        )
        best_index = int(np.argmax(score))
        if not np.isfinite(score[best_index]):
            raise RuntimeError("TRAP scan did not produce a finite candidate")
        # Refine each selected direction BEFORE projecting it out. The
        # projected objective and cell bounds use only the observed subspace.
        # Re-score subgrid quadratic predictions at EVERY coarse local
        # maximum before committing a source. No top-N, dB or angular gate.
        surface = score.reshape(len(theta_axis), len(phi_axis))
        local_mask = np.isfinite(surface) & (surface == maximum_filter(surface, size=3, mode=("wrap", "nearest")))
        peak_indices = np.argwhere(local_mask)
        if high_occupancy and len(peak_indices):
            evidence_candidates.extend(
                np.column_stack(
                    (
                        theta_axis[peak_indices[:, 0]],
                        phi_axis[peak_indices[:, 1]],
                    )
                )
            )
        predictions = []
        for ti, pi in peak_indices:
            if pi == 0 or pi == len(phi_axis)-1:
                continue
            tm, tp = (ti-1) % len(theta_axis), (ti+1) % len(theta_axis)
            # Guard the finite-difference stencil before subtraction.  This is
            # semantically equivalent to the former post-arithmetic finite check:
            # any non-finite neighbor would have produced a non-finite gradient or
            # Hessian and the proposal would have been skipped.  Checking first
            # only suppresses NumPy invalid-subtract warnings; it does not change
            # the accepted proposal set or the V7H decision rule.
            stencil = np.asarray([
                surface[ti, pi],
                surface[tm, pi], surface[tp, pi],
                surface[ti, pi - 1], surface[ti, pi + 1],
                surface[tm, pi - 1], surface[tm, pi + 1],
                surface[tp, pi - 1], surface[tp, pi + 1],
            ], dtype=float)
            if not np.isfinite(stencil).all():
                continue
            f = surface[ti,pi]
            gradient = np.array([(surface[tp,pi]-surface[tm,pi])/(2*step),
                                 (surface[ti,pi+1]-surface[ti,pi-1])/(2*step)])
            cross = (surface[tp,pi+1]-surface[tp,pi-1]-surface[tm,pi+1]+surface[tm,pi-1])/(4*step**2)
            hessian = np.array([[(surface[tp,pi]-2*f+surface[tm,pi])/step**2,cross],
                                [cross,(surface[ti,pi+1]-2*f+surface[ti,pi-1])/step**2]])
            if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
                continue
            eigen = np.linalg.eigvalsh(hessian)
            if eigen[-1] >= -EPS * max(1., abs(eigen[0])):
                continue
            offset = np.clip(-np.linalg.solve(hessian, gradient), -step, step)
            point = np.array([theta_axis[ti], phi_axis[pi]]) + offset
            point[0] = (point[0]-theta_bounds[0]) % (theta_bounds[1]-theta_bounds[0]) + theta_bounds[0]
            point[1] = np.clip(point[1], *phi_bounds)
            predictions.append(point)
        if high_occupancy and predictions:
            evidence_candidates.extend(np.asarray(predictions, dtype=float))
        # Preserve the frozen paper fusion boundary.  The replacement candidate
        # changes covariance estimation only; it does not combine multiple
        # mechanism changes or reopen the K=2 pathway that failed the paired
        # development gate.
        if k_sources < 3:
            predictions = []
        subgrid_evaluations = len(predictions)
        center = grid_points[best_index].copy()
        selected_score = float(score[best_index])
        if predictions:
            prediction_points = np.asarray(predictions)
            qa = complement @ _steering(prediction_points, array_pos, wavelength)
            den = np.sum(np.abs(qa)**2, axis=0)
            num = np.sum(np.abs(projected_basis.conj().T @ qa)**2, axis=0)
            actual = np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 128.0*EPS*sensor_count)
            best_prediction = int(np.argmax(actual))
            if actual[best_prediction] > selected_score:
                center = prediction_points[best_prediction].copy()
                selected_score = float(actual[best_prediction])
        # V7f fusion: use subgrid interpolation only as a proposal generator.
        # A proposal is committed only after exact projected-subspace rescoring
        # shows a strict machine-precision improvement over the coarse point.
        # No additional local projected optimizer is run here; this avoids the
        # V7e failure mode where recursive projection can be over-conditioned
        # by an aggressively polished early source.
        accepted_local = bool(selected_score > float(score[best_index]) + 64.0 * EPS)
        selected.append(center.copy())
        trap_evaluations += int(len(grid_points)) + subgrid_evaluations
        trap_trace.append(
            {
                "source_index": int(q),
                "selected_direction": selected[-1].tolist(),
                "coarse_direction": grid_points[best_index].tolist(),
                "subgrid_start": center.tolist(),
                "subgrid_evaluations": subgrid_evaluations,
                "fusion_enabled": bool(k_sources >= 3),
                "continuous_projection_calls": 0,
                "subgrid_proposal_accepted": bool(accepted_local),
                "selected_manifold_rank": int(selected_rank),
                "requested_remaining_signal_rank": int(remaining_rank),
                "retained_projected_signal_rank": int(retained_rank),
                "projected_score": float(score[best_index]),
            }
        )

    estimates = np.asarray(selected, dtype=float)
    initial_gain, initial_rank = _trace_projection_gain(
        estimates, covariance, array_pos, wavelength
    )
    population_size, iterations, coordinate_budget_used = _rime_shape(
        sensor_count, int(continuous_search_budget_per_coordinate)
    )
    rng = np.random.default_rng(seed)
    refinement_trace: list[dict] = []
    conditional_evaluations = 0

    for sweep in range(_AP_SWEEPS):
        for source_index in range(k_sources):
            other = np.delete(estimates, source_index, axis=0)
            other_steering = (
                _steering(other, array_pos, wavelength)
                if len(other)
                else np.empty((sensor_count, 0), dtype=complex)
            )
            complement, other_rank = _orthogonal_complement(other_steering, sensor_count)
            conditional = ConditionalDMLObjective(
                covariance=covariance,
                array_pos=array_pos,
                wavelength=float(wavelength),
                complement=complement,
                covariance_trace=covariance_trace,
            )
            before_point = estimates[source_index].copy()
            # The explicit current point at population[0] makes the update
            # monotonic even though the remaining population is stochastic.
            population = _joint_seed_population(
                before_point,
                population_size,
                theta_bounds,
                phi_bounds,
                rng,
                theta_jitter=max(1.0, 2.0 * step / 3.0),
                phi_jitter=max(0.75, step / 2.0),
            )
            result = _optimize_rime(
                conditional,
                theta_bounds,
                phi_bounds,
                population_size,
                iterations,
                population_init=population,
                rng=rng,
                W=_CANONICAL_RIME_W,
                use_opposition_init=True,
                use_differential_move=False,
                use_adaptive_restart=False,
            )
            if conditional.first_value is None:
                raise RuntimeError("conditional RIME search evaluated no points")
            before_loss = float(conditional.first_value)
            proposed = np.asarray(result["best_pos"], dtype=float)
            proposed_loss = float(result["best_obj"])
            accepted = bool(proposed_loss <= before_loss + 64.0 * EPS)
            if accepted:
                estimates[source_index] = proposed
            after_loss = proposed_loss if accepted else before_loss
            conditional_evaluations += int(conditional.eval_count)
            refinement_trace.append(
                {
                    "sweep": int(sweep + 1),
                    "source_index": int(source_index),
                    "conditioning_manifold_rank": int(other_rank),
                    "before_direction": before_point.tolist(),
                    "after_direction": estimates[source_index].tolist(),
                    "before_conditional_loss": float(before_loss),
                    "after_conditional_loss": float(after_loss),
                    "accepted": accepted,
                    "angle_evaluations": int(conditional.eval_count),
                }
            )

    if high_occupancy:
        evidence_array = (
            np.asarray(evidence_candidates, dtype=float).reshape(-1, 2)
            if evidence_candidates
            else np.empty((0, 2), dtype=float)
        )
        estimates, evidence_repair_diagnostics = _high_occupancy_evidence_swap_repair(
            estimates,
            evidence_array,
            sample_covariance,
            array_pos,
            wavelength,
        )
    else:
        evidence_repair_diagnostics = {
            "method": "below_noise_power_TRAP_evidence_swap_repair",
            "power_gate_rule": "weakest_fitted_source_power <= fitted_noise_variance",
            "triggered": False,
            "initial_zero_power_source_indices": [],
            "initial_weakest_source_index": None,
            "initial_weakest_source_power": None,
            "initial_fitted_noise_variance": None,
            "initial_weakest_power_below_noise": False,
            "candidate_count": 0,
            "initial_loss": None,
            "final_loss": None,
            "accepted_swap_count": 0,
            "swaps": [],
            "joint_set_objective_calls": 0,
            "equivalent_angle_evaluations": 0,
            "initial_manifold_rank": None,
            "user_selected_parameters": [],
            "disabled_reason": "2K_is_less_than_M",
        }

    steering_at_rime = _steering(estimates, array_pos, wavelength)
    manifold_gram = steering_at_rime.conj().T @ steering_at_rime
    normalized_min_manifold_eigenvalue = float(
        np.min(np.linalg.eigvalsh(manifold_gram)).real / sensor_count
    )
    weakest_signal_eigenvalue = float(sample_eigenvalues[k_sources - 1])
    weak_mode_energy_bound = float(
        (sensor_count / k_sources) * estimated_noise_variance
    )
    weak_mode_gate = bool(weakest_signal_eigenvalue <= weak_mode_energy_bound)
    pair_geometry_bound = float(k_sources / sensor_count)
    pair_geometry_gate = bool(
        k_sources == 2
        and normalized_min_manifold_eigenvalue <= pair_geometry_bound
    )
    if high_occupancy:
        joint_polish = _joint_stochastic_ml_polish
        rank_balance_regime = "high_occupancy_stochastic_ml"
    elif weak_mode_gate or pair_geometry_gate:
        joint_polish = _joint_uncorrelated_covariance_polish
        rank_balance_regime = "low_occupancy_structured_covariance_fit"
    else:
        joint_polish = _joint_dml_polish
        rank_balance_regime = "low_occupancy_paper_dml"
    estimates, polish_diagnostics = joint_polish(
        estimates,
        covariance,
        array_pos,
        wavelength,
        theta_bounds,
        phi_bounds,
        step,
        int(continuous_search_budget_per_coordinate),
    )

    final_gain, final_rank = _trace_projection_gain(
        estimates, covariance, array_pos, wavelength
    )
    order = np.argsort(estimates[:, 0])
    estimates = estimates[order]
    theta_est = estimates[:, 0].copy()
    phi_est = estimates[:, 1].copy()
    diagnostics = {
        "mechanism": "rank_balanced_structured_covariance_IRIME",
        "hard_angular_exclusion": False,
        "uses_source_angle_spacing_or_snr_controls": False,
        "universal_exact_rescored_fusion": False,
        "population_seeding": "paper_v1_unchanged",
        "rank_balance_rule": "high occupancy iff 2K >= M",
        "rank_balance_regime": rank_balance_regime,
        "structured_polish_gate": {
            "weak_mode_energy_rule": "lambda_K <= (M/K) * mean_noise_eigenvalue",
            "weakest_signal_eigenvalue": weakest_signal_eigenvalue,
            "estimated_noise_variance": estimated_noise_variance,
            "weak_mode_energy_bound": weak_mode_energy_bound,
            "weak_mode_gate": weak_mode_gate,
            "pair_geometry_rule": "K == 2 and lambda_min(A^H A)/M <= K/M",
            "normalized_min_manifold_eigenvalue": normalized_min_manifold_eigenvalue,
            "pair_geometry_bound": pair_geometry_bound,
            "pair_geometry_gate": pair_geometry_gate,
            "user_selected_parameters": [],
        },
        "spiked_covariance_projection": spiked_covariance_diagnostics,
        "high_occupancy_evidence_repair": evidence_repair_diagnostics,
        "coarse_evidence_step_deg": step,
        "coarse_grid_shape": [int(len(theta_axis)), int(len(phi_axis))],
        "coarse_grid_points": int(len(grid_points)),
        "trap_projected_grid_evaluations": int(trap_evaluations),
        "conditional_dml_rime_evaluations": int(conditional_evaluations),
        "evidence_repair_equivalent_angle_evaluations": int(
            evidence_repair_diagnostics["equivalent_angle_evaluations"]
        ),
        "joint_polish_equivalent_angle_evaluations": int(
            polish_diagnostics["equivalent_angle_evaluations"]
        ),
        "joint_polish_set_objective_calls": int(
            polish_diagnostics["joint_set_objective_calls"]
        ),
        "objective_eval_count": int(
            trap_evaluations
            + conditional_evaluations
            + evidence_repair_diagnostics["equivalent_angle_evaluations"]
            + polish_diagnostics["equivalent_angle_evaluations"]
        ),
        "objective_eval_count_type": "exact_projected_set_objective_angle_evaluations",
        "objective_eval_count_breakdown": {
            "trap_projected_grid_and_continuous_calibration": int(trap_evaluations),
            "conditional_dml_rime": int(conditional_evaluations),
            "high_occupancy_evidence_repair": int(
                evidence_repair_diagnostics["equivalent_angle_evaluations"]
            ),
            "joint_polish_equivalent_angles": int(
                polish_diagnostics["equivalent_angle_evaluations"]
            ),
        },
        "continuous_search_budget_requested_per_coordinate": int(
            continuous_search_budget_per_coordinate
        ),
        "continuous_search_budget_used_per_coordinate": int(coordinate_budget_used),
        "rime_population_size_derived": int(population_size),
        "rime_iterations_derived": int(iterations),
        "rime_W_fixed": _CANONICAL_RIME_W,
        "cyclic_sweeps_fixed": _AP_SWEEPS,
        "initial_dml_projection_gain": float(initial_gain),
        "final_dml_projection_gain": float(final_gain),
        "initial_manifold_rank": int(initial_rank),
        "final_manifold_rank": int(final_rank),
        "trap_trace": trap_trace,
        "refinement_trace": refinement_trace,
        "joint_polish": polish_diagnostics,
    }
    output = (theta_est, phi_est)
    if return_diagnostics:
        return (*output, diagnostics)
    return output


__all__ = ["ConditionalDMLObjective", "irime_v3"]
