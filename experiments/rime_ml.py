"""Original 2023 RIME operators applied to the complete raw DML DOA vector.

Python port of Su et al.'s official RIME.m, commit
32b02339873ae24e10358f3ccedf9fcaebeae2cb (MIT; see THIRD_PARTY_NOTICES.md).
Only the dimension, DOA box, and objective are problem-specific. Random draws
are batched with NumPy; they are distribution-equivalent, not MATLAB seed
equivalent. No recursive/grid/subspace initialization, sorting during search,
DSD, restart, local polishing, or IRIME terminal controller is used.
"""
from __future__ import annotations
import numpy as np

OFFICIAL_COMMIT = "32b02339873ae24e10358f3ccedf9fcaebeae2cb"


class RawDML:
    """Normalized tr(P_A^perp R), with a rank-aware SVD projector.

    The relative singular-value cutoff 1e-10 is a numerical rank convention,
    not covariance regularization. Rank-deficient candidates remain valid and
    are projected onto their actual column space. Counting includes every
    candidate, including repeated positions and initialization.
    """
    def __init__(self, x, positions, wavelength, k, *, rcond=1e-10):
        x = np.asarray(x, dtype=complex)
        self.positions = np.asarray(positions, dtype=float)
        if x.ndim != 2 or self.positions.shape != (x.shape[0], 3):
            raise ValueError("X must be M x L and positions must be M x 3")
        if not np.isfinite(x).all() or not np.isfinite(self.positions).all():
            raise ValueError("Inputs must be finite")
        if not 0 < int(k) < x.shape[0] or wavelength <= 0 or x.shape[1] < 1:
            raise ValueError("Require 0 < K < M, wavelength > 0 and L >= 1")
        self.k, self.wavelength, self.rcond = int(k), float(wavelength), float(rcond)
        self.covariance = (x @ x.conj().T) / x.shape[1]
        self.trace = float(np.trace(self.covariance).real)
        if self.trace <= 0:
            raise ValueError("Observation energy must be positive")
        self.evaluations = 0

    def steering_batch(self, vectors):
        v = np.asarray(vectors, dtype=float)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2 or v.shape[1] != 2 * self.k:
            raise ValueError("A vector contains K azimuths followed by K elevations")
        theta, phi = np.deg2rad(v[:, :self.k]), np.deg2rad(v[:, self.k:])
        directions = np.stack((np.cos(phi)*np.cos(theta),
                               np.cos(phi)*np.sin(theta), np.sin(phi)), axis=-1)
        phase = np.einsum("mk,njk->nmj", self.positions, directions)
        return np.exp((-2j * np.pi / self.wavelength) * phase)

    def __call__(self, vectors):
        a = self.steering_batch(vectors)
        u, singular, _ = np.linalg.svd(a, full_matrices=False)
        retained = singular > self.rcond * singular[:, :1]
        ru = self.covariance @ u
        captured = np.sum(np.sum(u.conj()*ru, axis=1).real * retained, axis=1)
        self.evaluations += len(a)
        loss = 1.0 - captured / self.trace
        if np.any(loss < -1e-10) or np.any(loss > 1+1e-10):
            raise ArithmeticError("DML projection energy lies outside [0,1]")
        return np.clip(loss, 0.0, 1.0)


def _propose(population, best, fitness, lb, ub, iteration, iterations, rng, W=5.0):
    # MATLAB round is away from zero at ties. Here its argument is positive.
    rounded = np.floor(iteration * W / iterations + 0.5)
    factor = ((rng.random()-0.5)*2 * np.cos(np.pi*iteration/(iterations/10.0))
              * (1.0-rounded/W))
    attachment = np.sqrt(iteration / iterations)
    soft = rng.random(population.shape) < attachment
    positions = best + factor * ((ub-lb)*rng.random(population.shape)+lb)
    candidate = np.where(soft, positions, population)
    norm = np.linalg.norm(fitness)
    rates = fitness/norm if norm > 0 else np.zeros_like(fitness)
    hard = rng.random(population.shape) < rates[:, None]
    candidate = np.where(hard, best, candidate)
    return np.clip(candidate, lb, ub)


def rime_ml(x, positions, wavelength, k, *, seed=42, population=50,
            iterations=2000, return_diagnostics=False):
    """Minimize raw joint ML with original soft/hard/greedy RIME operators.

    Default 50 initial candidates plus 2000 complete generations gives exactly
    100050 joint objective evaluations. This is a fixed solver budget, not an
    equal-runtime or equal-total-evaluation comparison with IRIME.
    """
    population, iterations = int(population), int(iterations)
    if population < 2 or iterations < 1:
        raise ValueError("Require population >= 2 and iterations >= 1")
    objective = RawDML(x, positions, wavelength, k)
    rng = np.random.default_rng(int(seed))
    lb = np.zeros(2*objective.k)
    ub = np.r_[np.full(objective.k, 360.0), np.full(objective.k, 90.0)]
    points = rng.uniform(lb, ub, size=(population, 2*objective.k))
    fitness = objective(points)
    index = int(np.argmin(fitness))
    best, best_loss = points[index].copy(), float(fitness[index])
    initial_loss = best_loss
    for it in range(1, iterations+1):
        proposed = _propose(points, best, fitness, lb, ub, it, iterations, rng)
        new_fitness = objective(proposed)
        accept = new_fitness < fitness
        points[accept], fitness[accept] = proposed[accept], new_fitness[accept]
        index = int(np.argmin(fitness))
        if fitness[index] < best_loss:
            best, best_loss = points[index].copy(), float(fitness[index])
    theta, phi = np.mod(best[:objective.k], 360.0), best[objective.k:].copy()
    order = np.argsort(theta)  # Output presentation only; never feeds search.
    theta, phi = theta[order], phi[order]
    if not return_diagnostics:
        return theta, phi
    return theta, phi, {"objective_evaluations":objective.evaluations,
                        "initial_loss":initial_loss, "final_loss":best_loss,
                        "population":population, "iterations":iterations,
                        "W":5, "rank_rcond":objective.rcond,
                        "official_commit":OFFICIAL_COMMIT}
