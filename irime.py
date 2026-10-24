"""IRIME estimator extracted from the RIME(4).zip experiment implementation.

The estimator follows the Compact IRIME numerical pipeline with the R205
DSC (formerly DSD) angular-seam and elevation-domain correction. The
Some historical paper tables predate this correction; do not claim they were rerun with
this revised implementation. The public wrapper fixes h=2.5 and B=896. Nonnegative
covariance fitting uses monotone active removal (an approximation), and the
high-occupancy terminal criterion is SML-form, not an unconditional likelihood
identity. See README.md for the local DSD chart and reproducibility limits.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
EPS = np.finfo(float).eps
_CANONICAL_RIME_W = 5.0
_AP_SWEEPS = 2

class ArchiveBuffer:
    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def add(self, points: np.ndarray) -> None:
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr[None, :]
        self._chunks.append(arr.copy())

    def stack(self) -> np.ndarray:
        if not self._chunks:
            return np.empty((0, 2), dtype=float)
        return np.vstack(self._chunks)


def clamp_positions(points: np.ndarray, theta_bounds: tuple[float, float], phi_bounds: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(points, dtype=float).copy()
    if arr.ndim == 1:
        arr = arr[None, :]
    theta_min, theta_max = map(float, theta_bounds)
    phi_min, phi_max = map(float, phi_bounds)
    theta_span = theta_max - theta_min
    if theta_span <= 0:
        raise ValueError('theta bounds must satisfy max > min')
    arr[:, 0] = ((arr[:, 0] - theta_min) % theta_span) + theta_min
    arr[:, 1] = np.clip(arr[:, 1], phi_min, phi_max)
    return arr


def make_child_rng(seed: int | None = None) -> np.random.Generator:
    """Create a Generator that remains reproducible when the caller seeded ``np.random``.

    ``np.random.randint`` from the legacy RandomState only supports int32 bounds on some
    NumPy builds, so keep the sampled seed within that range.
    """
    if seed is not None:
        return np.random.default_rng(int(seed))
    legacy_high = int(np.iinfo(np.int32).max)
    child_seed = int(np.random.randint(0, legacy_high))
    return np.random.default_rng(child_seed)


def random_population(rng: np.random.Generator, pop_size: int, theta_bounds: tuple[float, float], phi_bounds: tuple[float, float]) -> np.ndarray:
    population = np.empty((pop_size, 2), dtype=float)
    population[:, 0] = rng.uniform(theta_bounds[0], theta_bounds[1], size=pop_size)
    population[:, 1] = rng.uniform(phi_bounds[0], phi_bounds[1], size=pop_size)
    return population


def make_initial_population(
    population_init: np.ndarray | None,
    pop_size: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if population_init is not None:
        arr = np.asarray(population_init, dtype=float).copy()
        if len(arr) < pop_size:
            extra = random_population(rng, pop_size - len(arr), theta_bounds, phi_bounds)
            arr = np.vstack([arr, extra])
        arr = arr[:pop_size]
    else:
        arr = random_population(rng, pop_size, theta_bounds, phi_bounds)
    return clamp_positions(arr, theta_bounds, phi_bounds)


def _joint_seed_population(
    center: np.ndarray,
    pop_size: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    rng: np.random.Generator,
    *,
    theta_jitter: float = 2.0,
    phi_jitter: float = 1.5,
) -> np.ndarray:
    """Build a local restart around one evidence candidate plus random coverage."""
    population = random_population(rng, int(pop_size), theta_bounds, phi_bounds)
    center = clamp_positions(np.asarray(center, dtype=float).reshape(1, 2), theta_bounds, phi_bounds)[0]
    local_count = min(int(pop_size), max(8, int(round(0.55 * int(pop_size)))))
    if local_count:
        jitter = np.column_stack([
            center[0] + float(theta_jitter) * rng.standard_normal(local_count),
            center[1] + float(phi_jitter) * rng.standard_normal(local_count),
        ])
        population[:local_count] = clamp_positions(jitter, theta_bounds, phi_bounds)
        population[0] = center
    return population


def _select_best_population(points: np.ndarray, fitness: np.ndarray, keep_n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(fitness, dtype=float))[: int(keep_n)]
    return np.asarray(points, dtype=float)[order].copy(), np.asarray(fitness, dtype=float)[order].copy()


def _augment_initial_population(
    population: np.ndarray,
    objective,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pop = np.asarray(population, dtype=float)
    center = 0.5 * (lb + ub)
    opposite = lb + ub - pop
    quasi_a = center + rng.random(pop.shape) * (center - pop)
    quasi_b = center - rng.random(pop.shape) * (center - pop)
    quasi = np.where(rng.random(pop.shape) < 0.5, quasi_a, quasi_b)
    bank = np.vstack([pop, opposite, quasi])
    bank[:, 0] = ((bank[:, 0] - lb[0]) % max(ub[0] - lb[0], 1e-6)) + lb[0]
    bank[:, 1] = np.clip(bank[:, 1], lb[1], ub[1])
    fit = objective.objective_batch(bank)
    return _select_best_population(bank, fit, len(pop))


def _adaptive_restart(
    population: np.ndarray,
    fitness: np.ndarray,
    archive: ArchiveBuffer,
    objective,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    restart_n: int,
    jitter_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pop = np.asarray(population, dtype=float).copy()
    fit = np.asarray(fitness, dtype=float).copy()
    restart_n = int(max(0, min(restart_n, len(pop))))
    if restart_n <= 0:
        return pop, fit

    order = np.argsort(fit)
    worst_idx = order[-restart_n:]
    archive_points = archive.stack()
    if archive_points.size == 0:
        archive_points = pop[order[: max(4, min(len(pop), restart_n))]]
    if archive_points.ndim == 1:
        archive_points = archive_points[None, :]
    elite = archive_points[rng.integers(0, len(archive_points), size=restart_n)]
    opposite_worst = lb + ub - pop[worst_idx]
    guided = 0.55 * elite + 0.45 * opposite_worst
    guided += rng.standard_normal(guided.shape) * jitter_scale
    random_bank = rng.uniform(lb, ub, size=(restart_n, 2))
    bank = np.vstack([guided, opposite_worst, random_bank])
    bank[:, 0] = ((bank[:, 0] - lb[0]) % max(ub[0] - lb[0], 1e-6)) + lb[0]
    bank[:, 1] = np.clip(bank[:, 1], lb[1], ub[1])
    bank_fit = objective.objective_batch(bank)
    sel_points, sel_fit = _select_best_population(bank, bank_fit, restart_n)
    replace_mask = sel_fit < fit[worst_idx]
    if np.any(replace_mask):
        chosen = worst_idx[replace_mask]
        pop[chosen] = sel_points[replace_mask]
        fit[chosen] = sel_fit[replace_mask]
    return pop, fit


def _optimize_rime(
    objective,
    theta_bounds,
    phi_bounds,
    pop_size: int,
    max_iter: int,
    population_init=None,
    rng: np.random.Generator | None = None,
    W: float = 5.0,
    use_opposition_init: bool = True,
    use_differential_move: bool = True,
    use_adaptive_restart: bool = True,
):
    """Run the enhanced finite-budget candidate-generation kernel.

    All objective calls pass through ``MusicObjective`` so exact point-evaluation
    accounting includes vectorized batches as their constituent angle pairs.
    """
    rng = make_child_rng() if rng is None else rng
    pop_size = int(pop_size)
    max_iter = int(max_iter)
    if pop_size <= 0 or max_iter <= 0:
        raise ValueError("pop_size and max_iter must be positive")
    lb = np.array([theta_bounds[0], phi_bounds[0]], dtype=float)
    ub = np.array([theta_bounds[1], phi_bounds[1]], dtype=float)
    span = ub - lb

    population = make_initial_population(population_init, pop_size, theta_bounds, phi_bounds, rng)
    if use_opposition_init:
        population, fitness = _augment_initial_population(population, objective, lb, ub, rng)
    else:
        fitness = objective.objective_batch(population)
    best_idx = int(np.argmin(fitness))
    best_pos = population[best_idx].copy()
    best_obj = float(fitness[best_idx])

    archive = ArchiveBuffer()
    elite_n = max(8, pop_size // 4)
    archive.add(population[np.argsort(fitness)[:elite_n]])
    curve = np.zeros(max_iter, dtype=float)

    stagnation = 0
    prev_best = float(best_obj)
    diff_frac = 0.18
    restart_frac = 0.18

    for it in range(1, max_iter + 1):
        denom_iter = max(1.0, max_iter / 10.0)
        rime_factor = (rng.random() - 0.5) * 2.0 * math.cos(math.pi * it / denom_iter) * (1.0 - round(it * W / max_iter) / W)
        E = math.sqrt(it / max_iter)
        decay = 1.0 / (it + 1.0)
        progress = it / max(1.0, max_iter)
        new_population = population.copy()

        fit_max = float(np.max(fitness))
        fit_min = float(np.min(fitness))
        normalized = (fit_max - fitness) / (fit_max - fit_min + 1e-12)
        normalized = np.clip(normalized, 0.0, 1.0)

        mask_explore = rng.random((pop_size, 2)) < E
        if np.any(mask_explore):
            rand_base = rng.random((pop_size, 2))
            centers = best_pos + rime_factor * (span * rand_base + lb - best_pos)
            perturb = 0.08 * span * rng.standard_normal((pop_size, 2))
            explored = centers + perturb * decay
            new_population = np.where(mask_explore, explored, new_population)

        mask_soft = rng.random((pop_size, 2)) < normalized[:, None]
        if np.any(mask_soft):
            soft_steps = best_pos + 0.018 * span * rng.standard_normal((pop_size, 2)) * decay
            new_population = np.where(mask_soft, soft_steps, new_population)

        # Archive-guided differential move for a small subset: cheap exploitation without full DE overhead.
        if use_differential_move:
            diff_n = max(2, int(round(diff_frac * pop_size)))
            elite_idx = np.argsort(fitness)[:elite_n]
            work_idx = np.argsort(fitness)[-diff_n:]
            bases = population[elite_idx[rng.integers(0, len(elite_idx), size=diff_n)]]
            r1 = population[rng.integers(0, pop_size, size=diff_n)]
            r2 = population[rng.integers(0, pop_size, size=diff_n)]
            F = 0.50 + 0.25 * (1.0 - progress)
            diff_trial = 0.65 * bases + 0.35 * best_pos + F * (r1 - r2)
            diff_trial += (0.04 + 0.05 * (1.0 - progress)) * span * rng.standard_normal((diff_n, 2))
            new_population[work_idx] = diff_trial

        new_population = clamp_positions(new_population, theta_bounds, phi_bounds)
        new_fitness = objective.objective_batch(new_population)
        better = new_fitness < fitness
        population[better] = new_population[better]
        fitness[better] = new_fitness[better]

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < best_obj:
            best_obj = float(fitness[best_idx])
            best_pos = population[best_idx].copy()

        archive.add(population[np.argsort(fitness)[:elite_n]])

        improve_ratio = (prev_best - best_obj) / (abs(prev_best) + 1e-12)
        if improve_ratio > 5e-4:
            stagnation = 0
            prev_best = float(best_obj)
        else:
            stagnation += 1

        diversity = float(np.mean(np.std((population - lb) / np.maximum(span, 1e-9), axis=0)))
        if use_adaptive_restart and ((stagnation >= 2 and progress < 0.92) or (diversity < 0.055 and progress < 0.85)):
            restart_n = max(4, int(round(restart_frac * pop_size)))
            jitter_scale = np.array([max(1.2, 0.08 * span[0]), max(0.8, 0.08 * span[1])], dtype=float) * (1.0 - 0.35 * progress)
            population, fitness = _adaptive_restart(population, fitness, archive, objective, lb, ub, rng, restart_n, jitter_scale)
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_obj:
                best_obj = float(fitness[best_idx])
                best_pos = population[best_idx].copy()
                prev_best = float(best_obj)
            stagnation = 0
            archive.add(population[np.argsort(fitness)[:elite_n]])

        curve[it - 1] = -10.0 * np.log10(best_obj + 1e-12)

    return {
        'best_pos': best_pos.copy(),
        'best_obj': float(best_obj),
        'population': population.copy(),
        'fitness': fitness.copy(),
        'archive_points': archive.stack(),
        'curve': curve,
    }


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
    clear the simultaneously fitted white-noise variance.  This is an operational
    source-support heuristic, not an identifiability boundary.  Every retained swap strictly
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
    """Bounded SML-form local ranking; PSD source-covariance feasibility is not enforced."""
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


def _steering_derivatives(point: np.ndarray, array_pos: np.ndarray, wavelength: float):
    """Compute exact analytic partial derivatives of the UCA steering vector w.r.t. (theta, phi)."""
    theta_rad = np.deg2rad(point[0])
    phi_rad = np.deg2rad(point[1])
    dk_dtheta = np.array([-np.cos(phi_rad) * np.sin(theta_rad), np.cos(phi_rad) * np.cos(theta_rad), 0.0]) * (np.pi / 180.0)
    dk_dphi = np.array([-np.sin(phi_rad) * np.cos(theta_rad), -np.sin(phi_rad) * np.sin(theta_rad), np.cos(phi_rad)]) * (np.pi / 180.0)
    
    a = _steering(point[None, :], array_pos, wavelength)[:, 0]
    k_factor = -2.0 * np.pi / float(wavelength)
    da_dtheta = 1j * (k_factor * (array_pos @ dk_dtheta)) * a
    da_dphi = 1j * (k_factor * (array_pos @ dk_dphi)) * a
    return a, da_dtheta, da_dphi


def _differential_subspace_decouple(
    estimates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Local differential-subspace proposal with a stabilized set-loss safeguard."""
    k_sources = len(estimates)
    if k_sources < 2:
        return estimates, {"applied": False, "reason": "k_less_than_2"}

    sensor_count = len(array_pos)
    cov_tr = float(np.real(np.trace(covariance)))
    theta_span = float(theta_bounds[1] - theta_bounds[0])

    # 1. Manifold Gram condition number detection (Stoica & Nehorai 1989)
    closest_pair = None
    max_cond = 0.0
    for i in range(k_sources):
        for j in range(i + 1, k_sources):
            A2 = _steering(estimates[[i, j]], array_pos, wavelength)
            G = A2.conj().T @ A2
            w = np.linalg.eigvalsh(G)
            cond = float(w[-1] / max(w[0], 1e-12))
            if cond > max_cond:
                max_cond = cond
                closest_pair = (i, j)

    # Condition threshold: kappa(G) > 10.0 (equivalent to correlation rho > 0.82)
    if closest_pair is None or max_cond < 10.0:
        return estimates, {"applied": False, "max_cond": max_cond, "reason": "manifold_well_conditioned"}

    i, j = closest_pair
    other_idx = [idx for idx in range(k_sources) if idx not in (i, j)]
    if other_idx:
        A_other = _steering(estimates[other_idx], array_pos, wavelength)
        Q = np.eye(sensor_count, dtype=complex) - A_other @ np.linalg.pinv(A_other)
    else:
        Q = np.eye(sensor_count, dtype=complex)

    R_proj = Q @ covariance @ Q
    w_cov, v_cov = np.linalg.eigh(R_proj)
    v1 = v_cov[:, -1]  # primary cluster mode (sum beam)
    v2 = v_cov[:, -2]  # secondary difference mode (difference beam)

    # Current DML loss
    A_curr = _steering(estimates, array_pos, wavelength)
    P_curr = A_curr @ np.linalg.pinv(A_curr)
    curr_loss = 1.0 - float(np.real(np.trace(P_curr @ covariance))) / max(cov_tr, EPS)

    # 2. Centroid alignment with primary eigenvector v1 (MLE centroid matched filter)
    # Choose the midpoint on the shorter periodic azimuth arc, not the
    # arithmetic midpoint of angles straddling 0/360 degrees.
    p_mid = 0.5 * (estimates[i] + estimates[j])
    theta_diff = (estimates[j, 0] - estimates[i, 0] + 0.5 * theta_span) % theta_span - 0.5 * theta_span
    p_mid[0] = estimates[i, 0] + 0.5 * theta_diff
    p_mid[0] = (p_mid[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
    p_mid[1] = np.clip(p_mid[1], *phi_bounds)

    def cent_obj(pt: np.ndarray) -> float:
        if not np.all(np.isfinite(pt)) or not (phi_bounds[0] <= pt[1] <= phi_bounds[1]):
            return 1e10
        pt = pt.copy()
        pt[0] = (pt[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        a = _steering(pt[None, :], array_pos, wavelength)[:, 0]
        Qa = Q @ a
        den = float(np.real(np.vdot(Qa, Qa)))
        if den < 128.0 * EPS:
            return 1e10
        return -float(np.abs(np.vdot(v1, Qa)) ** 2) / den

    res_c = opt.minimize(cent_obj, x0=p_mid, method="Nelder-Mead", options={"maxiter": 20})
    cent = np.asarray(res_c.x, dtype=float)
    cent[0] = (cent[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
    cent[1] = np.clip(cent[1], *phi_bounds)

    # 3. Analytic Tangent Bundle Projection for Separation Axis u (Closed-form)
    a_mid, da_dth, da_dph = _steering_derivatives(cent, array_pos, wavelength)
    D = np.column_stack([da_dth, da_dph])

    Qa_mid = Q @ a_mid
    denom = float(np.real(np.vdot(Qa_mid, Qa_mid)))
    if denom < 128.0 * EPS:
        return estimates, {"applied": False, "reason": "singular_centroid"}

    P_mid_perp = np.eye(sensor_count, dtype=complex) - np.outer(Qa_mid, Qa_mid.conj()) / denom
    D_proj = P_mid_perp @ (Q @ D)

    coeff, _, _, _ = np.linalg.lstsq(D_proj, v2, rcond=None)
    # The covariance eigenvector has an arbitrary global complex phase. Align
    # that phase before taking the real coordinate direction so the split axis
    # is invariant (up to sign) to the EVD convention.
    z = np.sum(coeff * coeff)
    phase = 0.5 * np.angle(z) if abs(z) > 1e-15 else np.angle(coeff[np.argmax(np.abs(coeff))])
    u = np.real(coeff * np.exp(-1j * phase))
    cphi = max(abs(np.cos(np.deg2rad(cent[1]))), 1e-3)
    u_norm = float(np.sqrt((cphi * u[0]) ** 2 + u[1] ** 2))
    if u_norm < 1e-12:
        return estimates, {"applied": False, "reason": "zero_tangent_norm"}
    u /= u_norm

    # 4. 1D line search along u bounded by array aperture Rayleigh half-beamwidth
    d_max = float(np.max(np.linalg.norm(array_pos[:, None, :] - array_pos[None, :, :], axis=-1)))
    rayleigh_half_bw_deg = float(np.rad2deg(wavelength / (2.0 * d_max)))
    max_search_s = min(2.5, rayleigh_half_bw_deg / 4.0)

    def candidate_at(s: float) -> np.ndarray | None:
        cand = estimates.copy()
        cand[i] = cent - s * u
        cand[j] = cent + s * u
        pair = cand[[i, j], :]
        if not np.all(np.isfinite(pair)) or np.any(pair[:, 1] < phi_bounds[0]) or np.any(pair[:, 1] > phi_bounds[1]):
            return None
        cand[[i, j], 0] = (cand[[i, j], 0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        return cand

    def dml_1d(s: float) -> float:
        cand = candidate_at(s)
        if cand is None:
            return 1e10
        A_c = _steering(cand, array_pos, wavelength)
        P_c = A_c @ np.linalg.pinv(A_c)
        return 1.0 - float(np.real(np.trace(P_c @ covariance))) / max(cov_tr, EPS)

    res_line = opt.minimize_scalar(dml_1d, bounds=(0.05, max_search_s), method="bounded", options={"xatol": 0.02, "maxiter": 15})
    if res_line.fun < curr_loss - 64.0 * EPS:
        cand = candidate_at(res_line.x)
        if cand is None:
            return estimates, {"applied": False, "reason": "candidate_out_of_bounds"}
        return cand, {
            "applied": True,
            "closest_pair": closest_pair,
            "max_cond": max_cond,
            "s_star_deg": float(res_line.x),
            "loss_drop": float(curr_loss - res_line.fun),
        }

    return estimates, {"applied": False, "max_cond": max_cond, "reason": "no_loss_improvement"}


def _irime_frozen(
    observation: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_range: np.ndarray,
    phi_range: np.ndarray,
    source_num: int,
    *,
    seed: int = 42,
    coarse_evidence_step_deg: float = 2.5,
    continuous_search_budget_per_coordinate: int = 112,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray]:
    """Compact IRIME candidate: original selective fusion plus stabilized search and merged structural refinement."""
    k_sources = int(source_num)
    sensor_count = int(array_pos.shape[0])
    snapshots = int(observation.shape[1])
    sample_cov = (observation @ observation.conj().T) / snapshots
    sample_cov = 0.5 * (sample_cov + sample_cov.conj().T)
    
    # Module 1: rank-stabilized recursive initialization
    covariance, spiked_diag = _spiked_covariance_projection(sample_cov, k_sources)
    covariance_trace = float(np.real(np.trace(covariance)))
    estimated_noise_variance = spiked_diag.get("estimated_noise_variance", float(np.mean(np.linalg.eigvalsh(sample_cov)[:sensor_count - k_sources])))
    sample_eigenvalues = np.linalg.eigvalsh(sample_cov)[::-1]
    
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
    
    for q in range(k_sources):
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
        
    estimates = np.asarray(selected, dtype=float)
    
    # Module 2a: analytic Differential Subspace Decoupling (DSD) before RIME
    estimates, dsd_diag = _differential_subspace_decouple(
        estimates, covariance, array_pos, wavelength, theta_bounds, phi_bounds
    )
    
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
    if high_occupancy and evidence_candidates:
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
    if high_occupancy:
        joint_polish = _joint_stochastic_ml_polish
        rank_regime = "high_occupancy_stochastic_ml"
    elif weak_mode_gate or bool(dsd_diag.get("applied", False)):
        joint_polish = _joint_uncorrelated_covariance_polish
        rank_regime = "weak_or_close_structured_covariance_fit"
    else:
        joint_polish = _joint_dml_polish
        rank_regime = "standard_dml"
        
    estimates, polish_diag = joint_polish(
        estimates,
        covariance,
        array_pos,
        wavelength,
        theta_bounds,
        phi_bounds,
        step,
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
            "method": "IRIME-V7-Compact",
            "rank_regime": rank_regime,
            "dsd_diagnostics": dsd_diag,
            "evidence_repair_diagnostics": evidence_repair_diag,
            "polish_diagnostics": polish_diag,
            "objective_eval_count": total_eval_count,
            "objective_eval_count_type": "incomplete_diagnostic_estimate_not_measured_total",
        }
    return theta_est, phi_est



def irime(X, array_pos, wavelength, theta_grid, phi_grid, k_sources, *,
          h=2.5, B=896, seed=42, return_diagnostics=False):
    """Estimate K known DOAs in degrees using the recorded paper method plus R205 DSD boundary correction.

    ``theta_grid`` and ``phi_grid`` specify domain endpoints, not the internal
    initialization resolution; that is controlled by ``h``. Diagnostics do not
    expose the legacy incomplete total-evaluation estimate as an exact count.
    """
    result = _irime_frozen(
        X, array_pos, wavelength, theta_grid, phi_grid, k_sources,
        seed=seed, coarse_evidence_step_deg=h,
        continuous_search_budget_per_coordinate=B,
        return_diagnostics=return_diagnostics)
    if not return_diagnostics:
        return result
    theta, phi, diagnostics = result
    diagnostics.pop("objective_eval_count", None)
    diagnostics.pop("objective_eval_count_type", None)
    return theta, phi, diagnostics

steering = _steering
__all__ = ["irime", "steering"]
