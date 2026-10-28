"""RIME kernels and the SE-RIME sequential multi-peak MUSIC framework.

The publication contribution is the outer multi-peak task framework (sequential
exclusion, evidence fusion, consensus commitment, and gated refinement), not a
claim of a universally new RIME operator.  ``optimize_original_rime_kernel`` is
kept as the source-faithful 2023 reference kernel; ``_optimize_rime`` is the
finite-budget candidate generator used by the frozen SE-RIME profile.
"""

from __future__ import annotations

import math
import numpy as np

from alg.single_opt_common import (
    ArchiveBuffer,
    build_music_objective,
    clamp_positions,
    make_initial_population,
    make_child_rng,
    run_joint_candidate_pool_search,
    run_sequential_peak_search,
    run_single_pass_peak_search,
)
from alg.model_order import estimate_source_count, validate_model_order_result


RIME_VARIANTS = {
    "raw": {"use_single_pass": True, "use_repulsion": False, "use_guides": False, "use_local_refine": False},
    "seq": {"use_single_pass": False, "use_repulsion": False, "use_guides": False, "use_local_refine": False},
    "repulsion": {"use_single_pass": False, "use_repulsion": True, "use_guides": False, "use_local_refine": False},
    "guide": {"use_single_pass": False, "use_repulsion": True, "use_guides": True, "use_local_refine": False},
    "full": {"use_single_pass": False, "use_repulsion": True, "use_guides": True, "use_local_refine": True},
    "fast_stage0": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
    },
    "fast_consensus": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
    },
    "fast_consensus_polish": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "final_joint_polish": True,
    },
    "fast_stage0_polish": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "final_joint_polish": True,
    },
    "highk_flat": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
    },
    "highk_rescue": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "rescue_low_confidence": True,
        "final_joint_polish": True,
    },
    "evidence_adaptive": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "rescue_low_confidence": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": False,
    },
    # SIViP final unified-controller ablations. These variants reuse the
    # same evidence-adaptive RIME kernel/profile and remove exactly one outer
    # mechanism. They are experiment controls only; the full publication
    # algorithm remains ``evidence_adaptive``.
    "ua_no_exclusion": {
        "use_single_pass": False,
        "use_repulsion": False,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "rescue_low_confidence": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": False,
    },
    "ua_no_evidence": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": False,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": False,
        "rescue_low_confidence": True,
        "use_archive_evidence": False,
        "use_coarse_candidate": False,
    },
    "ua_no_rescue": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "rescue_low_confidence": False,
        "use_archive_evidence": True,
        "use_coarse_candidate": False,
    },
    "full_species": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "refined_guides": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "rescue_low_confidence": True,
    },
    "full_gated": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
    },
    "full_gated_joint": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "final_joint_polish": True,
    },
    "adaptive": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "adaptive_exclusion": True,
        "refined_guides": True,
        "confidence_gated_refine": True,
        "final_joint_polish": False,
    },
    "adaptive_joint": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "adaptive_exclusion": True,
        "refined_guides": True,
        "confidence_gated_refine": True,
        "final_joint_polish": True,
    },
    # Experimental replacement for the fixed-ellipse sequential extractor.
    # RIME remains the local search kernel; the outer mechanism is a shared
    # evidence candidate pool with soft joint source-set selection.
    "joint_pool": {
        "use_single_pass": False,
        "use_joint_pool": True,
        "use_repulsion": False,
        "use_guides": False,
        "use_local_refine": True,
    },
    # Parameter-light replacement for the soft-coherence selector.  It keeps
    # the same RIME candidate generator and commits each additional source by
    # its marginal contribution to the estimated signal subspace.
    "subspace_gain_pool": {
        "use_single_pass": False,
        "use_joint_pool": True,
        "use_repulsion": False,
        "use_guides": False,
        "use_local_refine": True,
        "joint_candidate_selection_mode": "subspace_gain",
    },
    # General set-level selector: exact over the bounded pool for K<=3 and
    # greedy ASG plus loss-decreasing one-swap refinement for larger K.
    "subspace_fit_pool": {
        "use_single_pass": False,
        "use_joint_pool": True,
        "use_repulsion": False,
        "use_guides": False,
        "use_local_refine": True,
        "joint_candidate_selection_mode": "subspace_fit",
    },
    # Analytic stochastic weighted-subspace fitting.  Its source-subspace
    # weights come from the observation eigenvalues and estimated noise level;
    # there is no angle gate or hand-selected selector coefficient.
    "optimal_wsf_pool": {
        "use_single_pass": False,
        "use_joint_pool": True,
        "use_repulsion": False,
        "use_guides": False,
        "use_local_refine": True,
        "joint_candidate_selection_mode": "optimal_wsf",
    },
    "final_v5": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": True,
    },
    # Component-wise ablation variants. All use the final publication budget;
    # only the named component is removed or replaced.
    "ab_full": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
    },
    "ab_no_exclusion": {
        "use_single_pass": False,
        "use_repulsion": False,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
    },
    "ab_isotropic_exclusion": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "isotropic_exclusion": True,
    },
    "ab_no_evidence_fusion": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": False,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": False,
        "use_archive_evidence": False,
        "use_coarse_candidate": False,
    },
    "ab_no_consensus": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": False,
        "use_archive_evidence": True,
    },
    "ab_no_coarse": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": False,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": False,
    },
    "ab_no_guide_injection": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": False,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": True,
    },
    "ab_no_coarse_candidate": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": False,
    },
    "ab_no_archive": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": True,
        "confidence_gated_refine": True,
        "candidate_consensus": True,
        "use_archive_evidence": False,
        "use_coarse_candidate": True,
    },
    "ab_no_local_refine": {
        "use_single_pass": False,
        "use_repulsion": True,
        "use_guides": True,
        "use_local_refine": False,
        "confidence_gated_refine": False,
        "candidate_consensus": True,
        "use_archive_evidence": True,
        "use_coarse_candidate": True,
    },
}


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


def _original_rime_normalized_rates(fitness: np.ndarray) -> np.ndarray:
    """Replicate MATLAB ``normr`` for the positive MUSIC objective vector.

    The official RIME code applies ``normr`` to the 1-by-N fitness row before
    the hard-rime puncture test. MUSIC objective values are non-negative, so
    L2 row normalization gives probabilities in [0, 1].
    """
    values = np.asarray(fitness, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-15:
        return np.zeros_like(values)
    return np.clip(values / norm, 0.0, 1.0)


def optimize_original_rime_kernel(
    objective,
    theta_bounds,
    phi_bounds,
    pop_size: int,
    max_iter: int,
    population_init=None,
    rng: np.random.Generator | None = None,
    W: float = 5.0,
):
    """Original RIME update equations adapted only to the angular bounds.

    The soft-rime update, hard-rime puncture and positive greedy selection
    follow the authors' public MATLAB implementation (Su et al., 2023). The
    common cross-generation elite archive is observational only: it is returned
    to the shared multi-peak outer framework but never feeds back into these
    original RIME updates. Azimuth uses periodic wrapping, while elevation is
    clipped to its physical interval.
    """
    rng = make_child_rng() if rng is None else rng
    lb = np.array([theta_bounds[0], phi_bounds[0]], dtype=float)
    ub = np.array([theta_bounds[1], phi_bounds[1]], dtype=float)
    span = ub - lb
    pop_size = int(pop_size)
    max_iter = int(max_iter)
    if pop_size <= 0 or max_iter <= 0:
        raise ValueError("pop_size and max_iter must be positive")

    population = make_initial_population(population_init, pop_size, theta_bounds, phi_bounds, rng)
    fitness = objective.objective_batch(population)
    best_idx = int(np.argmin(fitness))
    best_pos = population[best_idx].copy()
    best_obj = float(fitness[best_idx])

    archive = ArchiveBuffer()
    elite_n = max(4, pop_size // 4)
    archive.add(population[np.argsort(fitness)[:elite_n]])
    curve = np.empty(max_iter, dtype=float)

    for iteration in range(1, max_iter + 1):
        rime_factor = (
            (rng.random() - 0.5)
            * 2.0
            * math.cos(math.pi * iteration / max(1.0, max_iter / 10.0))
            * (1.0 - round(iteration * W / max_iter) / W)
        )
        attachment = math.sqrt(iteration / max_iter)
        new_population = population.copy()

        soft_mask = rng.random((pop_size, 2)) < attachment
        soft_points = best_pos[None, :] + rime_factor * (span[None, :] * rng.random((pop_size, 2)) + lb[None, :])
        new_population = np.where(soft_mask, soft_points, new_population)

        puncture_prob = _original_rime_normalized_rates(fitness)
        hard_mask = rng.random((pop_size, 2)) < puncture_prob[:, None]
        new_population = np.where(hard_mask, best_pos[None, :], new_population)
        new_population = clamp_positions(new_population, theta_bounds, phi_bounds)
        new_fitness = objective.objective_batch(new_population)

        better = new_fitness < fitness
        if np.any(better):
            population[better] = new_population[better]
            fitness[better] = new_fitness[better]
            best_idx = int(np.argmin(fitness))
            if float(fitness[best_idx]) < best_obj:
                best_obj = float(fitness[best_idx])
                best_pos = population[best_idx].copy()

        archive.add(population[np.argsort(fitness)[:elite_n]])
        curve[iteration - 1] = -10.0 * np.log10(best_obj + 1e-12)

    return {
        "best_pos": best_pos.copy(),
        "best_obj": float(best_obj),
        "population": population.copy(),
        "fitness": fitness.copy(),
        "archive_points": archive.stack(),
        "curve": curve,
        "kernel_family": "original_rime_2023",
    }


def optimize_enhanced_rime_kernel(
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
    """Enhanced finite-budget RIME kernel used by the proposed framework."""
    return _optimize_rime(
        objective=objective,
        theta_bounds=theta_bounds,
        phi_bounds=phi_bounds,
        pop_size=pop_size,
        max_iter=max_iter,
        population_init=population_init,
        rng=rng,
        W=W,
        use_opposition_init=use_opposition_init,
        use_differential_move=use_differential_move,
        use_adaptive_restart=use_adaptive_restart,
    )


def optimize_rime_kernel(*args, **kwargs):
    """Backward-compatible alias for :func:`optimize_enhanced_rime_kernel`."""
    return optimize_enhanced_rime_kernel(*args, **kwargs)

def music_2d_rime_variant(
    X,
    array_pos,
    wavelength,
    theta_range,
    phi_range,
    K,
    pop_size: int = 128,
    max_iter: int = 20,
    population_init=None,
    W: float = 5.0,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 4,
    k_neighbors: int = 8,
    exclusion_theta: float = 14.0,
    exclusion_phi: float = 7.0,
    penalty_strength: float | None = None,
    variant: str = "fast_stage0",
    stage_pop_sizes: list[int] | tuple[int, ...] | None = None,
    stage_max_iters: list[int] | tuple[int, ...] | None = None,
    stage_theta_seps: list[float] | tuple[float, ...] | None = None,
    stage_phi_seps: list[float] | tuple[float, ...] | None = None,
    confidence_improve_ratio: float = 0.0,
    confidence_move_factor: float = 0.0,
    final_polish_rounds: int = 0,
    guides_until_stage: int | None = None,
    local_refine_until_stage: int | None = None,
    consensus_until_stage: int | None = None,
    rescue_until_stage: int | None = None,
    guide_keep: int | None = None,
    max_injected_guides: int | None = None,
    archive_top_pool_scale: float = 2.0,
    archive_diversity_rescue: bool = False,
    kernel_use_opposition_init: bool = True,
    kernel_use_differential_move: bool = True,
    kernel_use_adaptive_restart: bool = True,
    use_archive_evidence_override: bool | None = None,
    use_coarse_candidate_override: bool | None = None,
    local_refine_high_conf_only: bool = False,
    local_refine_mode: str = "fixed",
    local_refine_support_scale: float = 0.52,
    evidence_support_scale: float = 0.52,
    evidence_consensus_weight: float = 0.55,
    evidence_adaptive_controller: bool = False,
    adaptive_coarse_on_low_confidence: bool = True,
    adaptive_differential_rescue: bool = True,
    simplified_rescue_budget: bool = False,
    confidence_gap_threshold: float | None = None,
    confidence_support_threshold: float | None = None,
    confidence_gap_threshold_first: float = 0.012,
    confidence_gap_threshold_later: float = 0.018,
    confidence_support_threshold_first: float = 0.86,
    confidence_support_threshold_later: float = 0.90,
    cache_global_evidence: bool = False,
    adaptive_exclusion_mode: str = "fixed",
    adaptive_exclusion_relative_drop: float = 0.10,
    adaptive_exclusion_min_scale: float = 0.55,
    adaptive_exclusion_max_scale: float = 1.35,
    joint_evidence_theta_step: float = 1.0,
    joint_evidence_phi_step: float = 1.0,
    joint_coherence_weight: float = 1.20,
    joint_evidence_floor_db: float | None = 10.0,
    joint_candidate_selection_mode: str = "coherence",
    joint_include_all_evidence_candidates: bool = False,
    joint_enable_optimizer_restarts: bool = True,
    return_diagnostics: bool = False,
):
    """Execute one explicitly named RIME/framework ablation variant.

    This lower-level entry is used by component ablations.  It never selects a
    profile from true DOAs, source spacing, or power labels; all supplied stage
    schedules are fixed before the observation is processed.
    """
    theta_bounds = (float(theta_range[0]), float(theta_range[-1]))
    phi_bounds = (float(phi_range[0]), float(phi_range[-1]))
    rng = make_child_rng()
    objective = build_music_objective(X, array_pos, wavelength, K)

    if variant not in RIME_VARIANTS:
        raise ValueError(f"Unknown RIME variant: {variant}")
    cfg = dict(RIME_VARIANTS[variant])

    if cfg.get("use_joint_pool", False):
        selection_mode = str(
            cfg.get("joint_candidate_selection_mode", joint_candidate_selection_mode)
        )
        include_all_evidence_candidates = bool(
            cfg.get(
                "joint_include_all_evidence_candidates",
                joint_include_all_evidence_candidates,
            )
        )
        effective_evidence_floor_db = (
            None
            if selection_mode in {"subspace_gain", "subspace_fit", "optimal_wsf"}
            else joint_evidence_floor_db
        )
        output = run_joint_candidate_pool_search(
            base_objective=objective,
            optimizer_fn=_optimize_rime,
            k_targets=K,
            theta_bounds=theta_bounds,
            phi_bounds=phi_bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            population_init=population_init,
            rng=rng,
            optimizer_kwargs={
                "W": W,
                "use_opposition_init": bool(kernel_use_opposition_init),
                "use_differential_move": bool(kernel_use_differential_move),
                "use_adaptive_restart": bool(kernel_use_adaptive_restart),
            },
            stage_pop_sizes=stage_pop_sizes,
            stage_max_iters=stage_max_iters,
            local_refine_mode=local_refine_mode,
            local_refine_support_scale=local_refine_support_scale,
            evidence_theta_step=joint_evidence_theta_step,
            evidence_phi_step=joint_evidence_phi_step,
            coherence_weight=joint_coherence_weight,
            evidence_floor_db=effective_evidence_floor_db,
            candidate_selection_mode=selection_mode,
            include_all_evidence_candidates=include_all_evidence_candidates,
            enable_optimizer_restarts=joint_enable_optimizer_restarts,
        )
        if return_diagnostics:
            diagnostics = objective.diagnostics()
            diagnostics.update(output[-1])
            return (*output[:-1], diagnostics)
        return output[:-1]

    if cfg["use_single_pass"]:
        output = run_single_pass_peak_search(
            base_objective=objective,
            optimizer_fn=_optimize_rime,
            k_targets=K,
            theta_bounds=theta_bounds,
            phi_bounds=phi_bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            population_init=population_init,
            rng=rng,
            optimizer_kwargs={
            'W': W,
            'use_opposition_init': bool(kernel_use_opposition_init),
            'use_differential_move': bool(kernel_use_differential_move),
            'use_adaptive_restart': bool(kernel_use_adaptive_restart),
        },
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            k_neighbors=k_neighbors,
        )
        if return_diagnostics:
            diagnostics = objective.diagnostics()
            diagnostics["stage_decisions"] = []
            return (*output, diagnostics)
        return output

    output = run_sequential_peak_search(
        base_objective=objective,
        optimizer_fn=_optimize_rime,
        k_targets=K,
        theta_bounds=theta_bounds,
        phi_bounds=phi_bounds,
        pop_size=pop_size,
        max_iter=max_iter,
        population_init=population_init,
        rng=rng,
        optimizer_kwargs={
            'W': W,
            'use_opposition_init': bool(kernel_use_opposition_init),
            'use_differential_move': bool(kernel_use_differential_move),
            'use_adaptive_restart': bool(kernel_use_adaptive_restart),
        },
        exclusion_theta=exclusion_theta,
        exclusion_phi=exclusion_phi,
        penalty_strength=penalty_strength,
        use_repulsion=cfg["use_repulsion"],
        use_guides=cfg["use_guides"],
        use_local_refine=cfg["use_local_refine"],
        local_refine_mode=local_refine_mode,
        local_refine_support_scale=local_refine_support_scale,
        evidence_support_scale=evidence_support_scale,
        evidence_consensus_weight=evidence_consensus_weight,
        stage_pop_sizes=stage_pop_sizes,
        stage_max_iters=stage_max_iters,
        stage_theta_seps=stage_theta_seps,
        stage_phi_seps=stage_phi_seps,
        refined_guides=bool(cfg.get("refined_guides", False)),
        adaptive_exclusion=bool(cfg.get("adaptive_exclusion", False)),
        confidence_gated_refine=bool(cfg.get("confidence_gated_refine", False)),
        confidence_improve_ratio=confidence_improve_ratio,
        confidence_move_factor=confidence_move_factor,
        final_joint_polish=bool(cfg.get("final_joint_polish", False)),
        final_polish_rounds=final_polish_rounds,
        candidate_consensus=bool(cfg.get("candidate_consensus", False)),
        rescue_low_confidence=bool(cfg.get("rescue_low_confidence", False)),
        guides_until_stage=guides_until_stage,
        local_refine_until_stage=local_refine_until_stage,
        consensus_until_stage=consensus_until_stage,
        rescue_until_stage=rescue_until_stage,
        guide_keep=guide_keep,
        max_injected_guides=max_injected_guides,
        archive_top_pool_scale=archive_top_pool_scale,
        archive_diversity_rescue=bool(
            archive_diversity_rescue
            and (
                bool(cfg.get("use_archive_evidence", True))
                if use_archive_evidence_override is None
                else bool(use_archive_evidence_override)
            )
        ),
        isotropic_exclusion=bool(cfg.get("isotropic_exclusion", False)),
        use_archive_evidence=(
            bool(cfg.get("use_archive_evidence", True))
            if use_archive_evidence_override is None
            else bool(use_archive_evidence_override)
        ),
        use_coarse_candidate=(
            bool(cfg.get("use_coarse_candidate", True))
            if use_coarse_candidate_override is None
            else bool(use_coarse_candidate_override)
        ),
        local_refine_high_conf_only=bool(local_refine_high_conf_only),
        evidence_adaptive_controller=bool(evidence_adaptive_controller),
        adaptive_coarse_on_low_confidence=bool(adaptive_coarse_on_low_confidence),
        adaptive_differential_rescue=bool(adaptive_differential_rescue),
        simplified_rescue_budget=bool(simplified_rescue_budget),
        confidence_gap_threshold=confidence_gap_threshold,
        confidence_support_threshold=confidence_support_threshold,
        confidence_gap_threshold_first=float(confidence_gap_threshold_first),
        confidence_gap_threshold_later=float(confidence_gap_threshold_later),
        confidence_support_threshold_first=float(confidence_support_threshold_first),
        confidence_support_threshold_later=float(confidence_support_threshold_later),
        cache_global_evidence=bool(cache_global_evidence),
        adaptive_exclusion_mode=adaptive_exclusion_mode,
        adaptive_exclusion_relative_drop=adaptive_exclusion_relative_drop,
        adaptive_exclusion_min_scale=adaptive_exclusion_min_scale,
        adaptive_exclusion_max_scale=adaptive_exclusion_max_scale,
    )
    clustered, scored, theta_est, phi_est, curve, framework_diagnostics = output
    if return_diagnostics:
        diagnostics = objective.diagnostics()
        diagnostics.update(framework_diagnostics)
        return clustered, scored, theta_est, phi_est, curve, diagnostics
    return clustered, scored, theta_est, phi_est, curve


def music_2d_rime(
    X,
    array_pos,
    wavelength,
    theta_range,
    phi_range,
    K,
    pop_size: int = 128,
    max_iter: int = 20,
    population_init=None,
    W: float = 5.0,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 4,
    k_neighbors: int = 8,
    exclusion_theta: float = 14.0,
    exclusion_phi: float = 7.0,
    penalty_strength: float | None = None,
    variant: str = "auto_speed",
    stage_pop_sizes: list[int] | tuple[int, ...] | None = None,
    stage_max_iters: list[int] | tuple[int, ...] | None = None,
    stage_theta_seps: list[float] | tuple[float, ...] | None = None,
    stage_phi_seps: list[float] | tuple[float, ...] | None = None,
    confidence_improve_ratio: float = 0.0,
    confidence_move_factor: float = 0.0,
    final_polish_rounds: int = 0,
    guides_until_stage: int | None = None,
    local_refine_until_stage: int | None = None,
    consensus_until_stage: int | None = None,
    rescue_until_stage: int | None = None,
    guide_keep: int | None = None,
    max_injected_guides: int | None = None,
    archive_top_pool_scale: float = 2.0,
    archive_diversity_rescue: bool = False,
    kernel_use_opposition_init: bool = True,
    kernel_use_differential_move: bool = True,
    kernel_use_adaptive_restart: bool = True,
    use_archive_evidence_override: bool | None = None,
    use_coarse_candidate_override: bool | None = None,
    local_refine_high_conf_only: bool = False,
    local_refine_mode: str = "fixed",
    local_refine_support_scale: float = 0.52,
    evidence_support_scale: float = 0.52,
    evidence_consensus_weight: float = 0.55,
    evidence_adaptive_controller: bool = False,
    adaptive_coarse_on_low_confidence: bool = True,
    adaptive_differential_rescue: bool = True,
    simplified_rescue_budget: bool = False,
    confidence_gap_threshold: float | None = None,
    confidence_support_threshold: float | None = None,
    confidence_gap_threshold_first: float = 0.012,
    confidence_gap_threshold_later: float = 0.018,
    confidence_support_threshold_first: float = 0.86,
    confidence_support_threshold_later: float = 0.90,
    cache_global_evidence: bool = False,
    adaptive_exclusion_mode: str = "fixed",
    adaptive_exclusion_relative_drop: float = 0.10,
    adaptive_exclusion_min_scale: float = 0.55,
    adaptive_exclusion_max_scale: float = 1.35,
    joint_evidence_theta_step: float = 1.0,
    joint_evidence_phi_step: float = 1.0,
    joint_coherence_weight: float = 1.20,
    joint_evidence_floor_db: float | None = 10.0,
    joint_candidate_selection_mode: str = "coherence",
    joint_include_all_evidence_candidates: bool = False,
    joint_enable_optimizer_restarts: bool = True,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Estimate ``K`` distinct 2-D MUSIC peaks with the frozen SE-RIME wrapper.

    ``variant='auto_speed'`` dispatches only on the externally declared source
    count ``K``.  In formal experiments, budgets and stage schedules are already
    resolved by :func:`configs.rime_adaptive.resolve_rime_budget` from ``K`` and
    snapshot count ``L``; this wrapper performs no access to simulation truth.
    """
    chosen_variant = variant
    chosen_stage_pop_sizes = stage_pop_sizes
    chosen_stage_max_iters = stage_max_iters
    chosen_stage_theta_seps = stage_theta_seps
    chosen_stage_phi_seps = stage_phi_seps
    chosen_final_polish_rounds = final_polish_rounds
    chosen_guides_until_stage = guides_until_stage
    chosen_local_refine_until_stage = local_refine_until_stage
    chosen_consensus_until_stage = consensus_until_stage
    chosen_rescue_until_stage = rescue_until_stage
    chosen_guide_keep = guide_keep
    chosen_max_injected_guides = max_injected_guides
    chosen_archive_top_pool_scale = archive_top_pool_scale
    chosen_archive_diversity_rescue = archive_diversity_rescue

    if variant in {"auto", "auto_speed"}:
        if int(K) <= 4:
            chosen_variant = "fast_consensus"
            if chosen_stage_theta_seps is None or list(chosen_stage_theta_seps) == [14.0, 11.0, 8.5]:
                chosen_stage_theta_seps = [18.0, 14.0, 11.0]
            if chosen_stage_phi_seps is None or list(chosen_stage_phi_seps) == [7.0, 5.5, 4.0]:
                chosen_stage_phi_seps = [7.0, 5.5, 4.0]
        elif int(K) == 5:
            chosen_variant = "fast_stage0"
        elif int(K) == 6:
            chosen_variant = "fast_stage0"
            if chosen_stage_theta_seps is None or list(chosen_stage_theta_seps) == [14.0, 11.0, 8.5]:
                chosen_stage_theta_seps = [15.0, 12.0, 9.5]
            if chosen_stage_phi_seps is None or list(chosen_stage_phi_seps) == [7.0, 5.5, 4.0]:
                chosen_stage_phi_seps = [6.5, 5.0, 4.0]
            chosen_consensus_until_stage = 2 if chosen_consensus_until_stage is None else chosen_consensus_until_stage
            chosen_local_refine_until_stage = 4 if chosen_local_refine_until_stage is None else chosen_local_refine_until_stage
            chosen_guides_until_stage = 5 if chosen_guides_until_stage is None else chosen_guides_until_stage
            chosen_archive_top_pool_scale = 1.5 if chosen_archive_top_pool_scale == 2.0 else chosen_archive_top_pool_scale
        else:
            chosen_variant = "highk_flat"
            if chosen_stage_theta_seps is None or list(chosen_stage_theta_seps) == [14.0, 11.0, 8.5]:
                chosen_stage_theta_seps = [16.0, 12.5, 9.5]
            if chosen_stage_phi_seps is None or list(chosen_stage_phi_seps) == [7.0, 5.5, 4.0]:
                chosen_stage_phi_seps = [6.5, 5.0, 4.0]
            chosen_consensus_until_stage = 2 if chosen_consensus_until_stage is None else chosen_consensus_until_stage
            chosen_local_refine_until_stage = 3 if chosen_local_refine_until_stage is None else chosen_local_refine_until_stage
            chosen_guides_until_stage = 4 if chosen_guides_until_stage is None else chosen_guides_until_stage
            chosen_archive_top_pool_scale = 1.25 if chosen_archive_top_pool_scale == 2.0 else chosen_archive_top_pool_scale
            chosen_guide_keep = 4 if chosen_guide_keep is None else chosen_guide_keep
            chosen_max_injected_guides = 8 if chosen_max_injected_guides is None else chosen_max_injected_guides

    return music_2d_rime_variant(
        X,
        array_pos,
        wavelength,
        theta_range,
        phi_range,
        K,
        pop_size=pop_size,
        max_iter=max_iter,
        population_init=population_init,
        W=W,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        k_neighbors=k_neighbors,
        exclusion_theta=exclusion_theta,
        exclusion_phi=exclusion_phi,
        penalty_strength=penalty_strength,
        variant=chosen_variant,
        stage_pop_sizes=chosen_stage_pop_sizes,
        stage_max_iters=chosen_stage_max_iters,
        stage_theta_seps=chosen_stage_theta_seps,
        stage_phi_seps=chosen_stage_phi_seps,
        confidence_improve_ratio=confidence_improve_ratio,
        confidence_move_factor=confidence_move_factor,
        final_polish_rounds=chosen_final_polish_rounds,
        guides_until_stage=chosen_guides_until_stage,
        local_refine_until_stage=chosen_local_refine_until_stage,
        consensus_until_stage=chosen_consensus_until_stage,
        rescue_until_stage=chosen_rescue_until_stage,
        guide_keep=chosen_guide_keep,
        max_injected_guides=chosen_max_injected_guides,
        archive_top_pool_scale=chosen_archive_top_pool_scale,
        archive_diversity_rescue=chosen_archive_diversity_rescue,
        kernel_use_opposition_init=kernel_use_opposition_init,
        kernel_use_differential_move=kernel_use_differential_move,
        kernel_use_adaptive_restart=kernel_use_adaptive_restart,
        use_archive_evidence_override=use_archive_evidence_override,
        use_coarse_candidate_override=use_coarse_candidate_override,
        local_refine_high_conf_only=local_refine_high_conf_only,
        local_refine_mode=local_refine_mode,
        local_refine_support_scale=local_refine_support_scale,
        evidence_support_scale=evidence_support_scale,
        evidence_consensus_weight=evidence_consensus_weight,
        evidence_adaptive_controller=evidence_adaptive_controller,
        adaptive_coarse_on_low_confidence=adaptive_coarse_on_low_confidence,
        adaptive_differential_rescue=adaptive_differential_rescue,
        simplified_rescue_budget=simplified_rescue_budget,
        confidence_gap_threshold=confidence_gap_threshold,
        confidence_support_threshold=confidence_support_threshold,
        confidence_gap_threshold_first=confidence_gap_threshold_first,
        confidence_gap_threshold_later=confidence_gap_threshold_later,
        confidence_support_threshold_first=confidence_support_threshold_first,
        confidence_support_threshold_later=confidence_support_threshold_later,
        cache_global_evidence=cache_global_evidence,
        adaptive_exclusion_mode=adaptive_exclusion_mode,
        adaptive_exclusion_relative_drop=adaptive_exclusion_relative_drop,
        adaptive_exclusion_min_scale=adaptive_exclusion_min_scale,
        adaptive_exclusion_max_scale=adaptive_exclusion_max_scale,
        joint_evidence_theta_step=joint_evidence_theta_step,
        joint_evidence_phi_step=joint_evidence_phi_step,
        joint_coherence_weight=joint_coherence_weight,
        joint_evidence_floor_db=joint_evidence_floor_db,
        joint_candidate_selection_mode=joint_candidate_selection_mode,
        joint_include_all_evidence_candidates=joint_include_all_evidence_candidates,
        joint_enable_optimizer_restarts=joint_enable_optimizer_restarts,
        return_diagnostics=return_diagnostics,
    )


def music_2d_rime_auto_k(
    X,
    array_pos,
    wavelength,
    theta_range,
    phi_range,
    *,
    max_sources: int | None = None,
    model_order_method: str = "mdl",
    random_seed: int | None = None,
    model_order_result: dict | None = None,
    **kwargs,
) -> dict:
    """Run end-to-end SE-RIME-2D-MUSIC without a supplied true source count.

    The count estimator consumes the observation only.  When the selected model
    order is zero, the routine explicitly returns no peaks instead of forcing a
    false detection.  This is intentionally a separate API so the historical
    known-K protocol remains byte-for-byte semantically unchanged.
    """
    observation = np.asarray(X, dtype=complex)
    array_position = np.asarray(array_pos, dtype=float)
    order = (
        estimate_source_count(
            observation,
            method=model_order_method,
            max_sources=max_sources,
        )
        if model_order_result is None
        else validate_model_order_result(model_order_result, observation.shape[0])
    )
    k_hat = int(order["k_hat"])
    if k_hat == 0:
        return {
            "k_hat": 0,
            "model_order": order,
            "theta_est": np.empty((0,), dtype=float),
            "phi_est": np.empty((0,), dtype=float),
            "curve": np.empty((0,), dtype=float),
            "population": np.empty((0, 2), dtype=float),
            "fitness": np.empty((0,), dtype=float),
            "labels": np.empty((0,), dtype=int),
            "diagnostics": {
                "objective_eval_count": 0,
                "objective_scalar_calls": 0,
                "objective_batch_calls": 0,
                "phase_eval_counts": {},
                "stage_decisions": [],
            },
        }
    if random_seed is not None:
        np.random.seed(int(random_seed))
    clustered, scored, theta_est, phi_est, curve, diagnostics = music_2d_rime(
        observation,
        array_position,
        wavelength,
        theta_range,
        phi_range,
        k_hat,
        return_diagnostics=True,
        **kwargs,
    )
    population, labels = clustered
    _, fitness = scored
    return {
        "k_hat": k_hat,
        "model_order": order,
        "theta_est": np.asarray(theta_est, dtype=float),
        "phi_est": np.asarray(phi_est, dtype=float),
        "curve": np.asarray(curve, dtype=float),
        "population": np.asarray(population, dtype=float),
        "fitness": np.asarray(fitness, dtype=float),
        "labels": np.asarray(labels),
        "diagnostics": diagnostics,
    }
