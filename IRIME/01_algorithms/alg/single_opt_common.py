from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np
from scipy import linalg, ndimage
from sklearn.cluster import DBSCAN, KMeans

EPS = 1e-12


@dataclass
class MusicObjective:
    array_pos: np.ndarray
    wavelength: float
    noise_projector: np.ndarray
    signal_eigenvectors: np.ndarray | None = None
    signal_eigenvalues: np.ndarray | None = None
    noise_variance: float | None = None
    covariance: np.ndarray | None = None
    eval_count: int = 0
    scalar_calls: int = 0
    batch_calls: int = 0
    active_phase: str = "unclassified"
    phase_counts: dict[str, int] = field(default_factory=dict)

    def reset_counts(self) -> None:
        """Reset exact MUSIC objective point-evaluation counters."""
        self.eval_count = 0
        self.scalar_calls = 0
        self.batch_calls = 0
        self.active_phase = "unclassified"
        self.phase_counts.clear()

    def set_phase(self, phase: str) -> None:
        """Assign subsequent point evaluations to a coarse audit phase."""
        self.active_phase = str(phase or "unclassified")

    def diagnostics(self) -> dict:
        """Return exact point-evaluation totals and coarse phase attribution."""
        return {
            "objective_eval_count": int(self.eval_count),
            "objective_scalar_calls": int(self.scalar_calls),
            "objective_batch_calls": int(self.batch_calls),
            "phase_eval_counts": {k: int(v) for k, v in sorted(self.phase_counts.items())},
        }

    def _record_points(self, count: int) -> None:
        count = int(count)
        self.eval_count += count
        phase = str(self.active_phase or "unclassified")
        self.phase_counts[phase] = int(self.phase_counts.get(phase, 0)) + count

    def objective(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float)
        if point.size != 2 or not np.all(np.isfinite(point)):
            raise ValueError("MUSIC objective point must contain two finite angles")
        point = point.reshape(2)
        self._record_points(1)
        self.scalar_calls += 1
        theta = np.deg2rad(point[0])
        phi = np.deg2rad(point[1])
        u = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
        ], dtype=float)
        phase = -2.0 * np.pi / self.wavelength * (self.array_pos @ u)
        a = np.exp(1j * phase)
        return float(np.abs(np.conj(a) @ self.noise_projector @ a) + EPS)

    def objective_batch(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("MUSIC objective points must have shape (n, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("MUSIC objective points must contain only finite angles")
        self._record_points(len(points))
        self.batch_calls += 1
        theta = np.deg2rad(points[:, 0])
        phi = np.deg2rad(points[:, 1])
        u = np.stack([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
        ], axis=1)
        phase = -2.0 * np.pi / self.wavelength * (u @ self.array_pos.T)
        a = np.exp(1j * phase)
        vals = np.abs(np.einsum('ni,ij,nj->n', np.conj(a), self.noise_projector, a)) + EPS
        return vals.astype(float)

    def score_batch(self, points: np.ndarray) -> np.ndarray:
        return -10.0 * np.log10(self.objective_batch(points))


class PenalizedMusicObjective:
    def __init__(
        self,
        base: MusicObjective,
        centers: np.ndarray | None,
        penalty_strength: float,
        theta_sigma: float,
        phi_sigma: float,
    ):
        self.base = base
        centers_arr = np.asarray(centers if centers is not None else np.empty((0, 2)), dtype=float)
        if centers_arr.size == 0:
            centers_arr = np.empty((0, 2), dtype=float)
        if centers_arr.ndim == 1:
            centers_arr = centers_arr[None, :]
        self.centers = centers_arr
        self.penalty_strength = float(penalty_strength)
        self.theta_sigma = float(max(theta_sigma, 1e-6))
        self.phi_sigma = float(max(phi_sigma, 1e-6))

    def _penalty_batch(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if len(self.centers) == 0:
            return np.zeros(len(pts), dtype=float)
        d_theta = np.abs(pts[:, 0][:, None] - self.centers[:, 0][None, :])
        d_theta = np.minimum(d_theta, 360.0 - d_theta)
        d_phi = np.abs(pts[:, 1][:, None] - self.centers[:, 1][None, :])
        scaled = (d_theta / self.theta_sigma) ** 2 + (d_phi / self.phi_sigma) ** 2
        return self.penalty_strength * np.sum(np.exp(-0.5 * scaled), axis=1)

    def objective(self, point: np.ndarray) -> float:
        return float(self.objective_batch(np.asarray(point, dtype=float))[0])

    def objective_batch(self, points: np.ndarray) -> np.ndarray:
        return self.base.objective_batch(points) + self._penalty_batch(points)


@dataclass
class PeakExtractionResult:
    labels: np.ndarray
    theta_est: np.ndarray
    phi_est: np.ndarray
    points: np.ndarray
    scores: np.ndarray


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



def build_music_objective(X: np.ndarray, array_pos: np.ndarray, wavelength: float, k_sources: int) -> MusicObjective:
    """Construct the continuous 2-D MUSIC objective with explicit audit checks.

    Formal experiment failures should be loud rather than silently producing an
    empty noise subspace or NaN covariance matrix.  These checks do not alter any
    valid paper configuration; they make malformed external reproductions fail at
    the scientific boundary where the assumptions are defined.
    """
    X = np.asarray(X)
    array_pos = np.asarray(array_pos, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional sensor-by-snapshot matrix")
    if X.shape[1] <= 0:
        raise ValueError("X must contain at least one snapshot")
    if array_pos.ndim != 2 or array_pos.shape[1] != 3:
        raise ValueError("array_pos must have shape (number_of_sensors, 3)")
    if X.shape[0] != array_pos.shape[0]:
        raise ValueError("X and array_pos must contain the same number of sensors")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(array_pos)):
        raise ValueError("X and array_pos must contain only finite values")
    wavelength = float(wavelength)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be a positive finite number")
    k_float = float(k_sources)
    k_sources = int(k_sources)
    if k_float != k_sources or not 1 <= k_sources < X.shape[0]:
        raise ValueError("k_sources must satisfy 1 <= k_sources < number of sensors")

    L = X.shape[1]
    R = X @ X.conj().T / L
    eigvals, eigvecs = linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    ordered_values = np.real(eigvals[order])
    ordered_vectors = eigvecs[:, order]
    En = ordered_vectors[:, k_sources:]
    Gn = En @ En.conj().T
    return MusicObjective(
        array_pos=array_pos,
        wavelength=wavelength,
        noise_projector=Gn,
        signal_eigenvectors=ordered_vectors[:, :k_sources].copy(),
        signal_eigenvalues=ordered_values[:k_sources].copy(),
        noise_variance=float(np.mean(ordered_values[k_sources:])),
        covariance=R.copy(),
    )



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



def pairwise_angular_distance(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    theta = pts[:, 0][:, None]
    phi = pts[:, 1][:, None]
    d_theta = np.abs(theta - theta.T)
    d_theta = np.minimum(d_theta, 360.0 - d_theta)
    d_phi = np.abs(phi - phi.T)
    return np.sqrt(d_theta ** 2 + d_phi ** 2)



def exclusion_metric(
    points: np.ndarray,
    centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    ctr = np.asarray(centers if centers is not None else np.empty((0, 2)), dtype=float)
    if ctr.size == 0:
        return np.full(len(pts), np.inf, dtype=float)
    if ctr.ndim == 1:
        ctr = ctr[None, :]
    d_theta = np.abs(pts[:, 0][:, None] - ctr[:, 0][None, :])
    d_theta = np.minimum(d_theta, 360.0 - d_theta)
    d_phi = np.abs(pts[:, 1][:, None] - ctr[:, 1][None, :])
    metric = np.sqrt((d_theta / max(theta_sep, 1e-6)) ** 2 + (d_phi / max(phi_sep, 1e-6)) ** 2)
    return np.min(metric, axis=1)



def far_from_centers_mask(
    points: np.ndarray,
    centers: np.ndarray | list[np.ndarray] | None,
    theta_sep: float,
    phi_sep: float,
    theta_hard_factor: float = 0.72,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    ctr = np.asarray(centers if centers is not None else np.empty((0, 2)), dtype=float)
    if ctr.size == 0:
        return np.ones(len(pts), dtype=bool)
    if ctr.ndim == 1:
        ctr = ctr[None, :]
    d_theta = np.abs(pts[:, 0][:, None] - ctr[:, 0][None, :])
    d_theta = np.minimum(d_theta, 360.0 - d_theta)
    d_phi = np.abs(pts[:, 1][:, None] - ctr[:, 1][None, :])
    metric = np.sqrt((d_theta / max(theta_sep, 1e-6)) ** 2 + (d_phi / max(phi_sep, 1e-6)) ** 2)
    min_metric = np.min(metric, axis=1)
    min_d_theta = np.min(d_theta, axis=1)
    return (min_metric >= 1.0) & (min_d_theta >= float(theta_hard_factor) * max(theta_sep, 1e-6))



def is_far_from_centers(
    point: np.ndarray,
    centers: np.ndarray | list[np.ndarray] | None,
    theta_sep: float,
    phi_sep: float,
    theta_hard_factor: float = 0.72,
) -> bool:
    mask = far_from_centers_mask(np.asarray(point, dtype=float).reshape(1, 2), centers, theta_sep, phi_sep, theta_hard_factor=theta_hard_factor)
    return bool(mask[0])



def resample_avoiding_centers(
    rng: np.random.Generator,
    count: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    if centers is None or len(np.asarray(centers).reshape(-1, 2)) == 0:
        return random_population(rng, count, theta_bounds, phi_bounds)

    accepted: list[np.ndarray] = []
    remaining = count
    for _ in range(64):
        batch_n = max(remaining * 3, 32)
        batch = random_population(rng, batch_n, theta_bounds, phi_bounds)
        mask = far_from_centers_mask(batch, centers, theta_sep, phi_sep)
        if np.any(mask):
            accepted.append(batch[mask][:remaining])
            remaining = count - sum(len(chunk) for chunk in accepted)
            if remaining <= 0:
                break
    if remaining > 0:
        accepted.append(random_population(rng, remaining, theta_bounds, phi_bounds))
    return np.vstack(accepted)[:count]



def avoid_selected_peaks(
    population: np.ndarray,
    rng: np.random.Generator,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    centers: np.ndarray | list[np.ndarray] | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray:
    pop = clamp_positions(population, theta_bounds, phi_bounds)
    ctr = np.asarray(centers if centers is not None else np.empty((0, 2)), dtype=float)
    if ctr.size == 0:
        return pop
    if ctr.ndim == 1:
        ctr = ctr[None, :]
    far_mask = far_from_centers_mask(pop, ctr, theta_sep, phi_sep)
    close_mask = ~far_mask
    if np.any(close_mask):
        pop[close_mask] = resample_avoiding_centers(rng, int(np.sum(close_mask)), theta_bounds, phi_bounds, ctr, theta_sep, phi_sep)
    return pop



def local_maxima_indices(points: np.ndarray, scores: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(points) == 0:
        return np.array([], dtype=int)
    if len(points) == 1:
        return np.array([0], dtype=int)
    k_neighbors = int(max(1, min(k_neighbors, len(points) - 1)))
    dmat = pairwise_angular_distance(points)
    np.fill_diagonal(dmat, np.inf)
    knn = np.argpartition(dmat, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
    idx = []
    for i in range(len(points)):
        nbs = knn[i]
        if nbs.size == 0 or np.all(scores[i] >= scores[nbs]):
            idx.append(i)
    if not idx:
        idx = [int(np.argmax(scores))]
    return np.asarray(idx, dtype=int)



def unique_points_keep_best(points: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    rounded = np.round(np.asarray(points, dtype=float), decimals=6)
    unique: dict[tuple[float, float], tuple[np.ndarray, float]] = {}
    for p, s in zip(rounded, scores):
        key = (float(p[0]), float(p[1]))
        prev = unique.get(key)
        if prev is None or s > prev[1]:
            unique[key] = (p.copy(), float(s))
    out_points = np.array([v[0] for v in unique.values()], dtype=float)
    out_scores = np.array([v[1] for v in unique.values()], dtype=float)
    return out_points, out_scores



def extract_peaks_from_points(
    points: np.ndarray,
    scores: np.ndarray,
    k_targets: int,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 4,
    k_neighbors: int = 8,
) -> PeakExtractionResult:
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    points, scores = unique_points_keep_best(points, scores)
    if len(points) == 0:
        theta_est = np.full(k_targets, np.nan)
        phi_est = np.full(k_targets, np.nan)
        return PeakExtractionResult(labels=np.empty((0,), dtype=int), theta_est=theta_est, phi_est=phi_est, points=points, scores=scores)

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(points)
    peak_points: list[np.ndarray] = []
    peak_scores: list[float] = []
    for label in np.unique(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        best = idx[np.argmax(scores[idx])]
        peak_points.append(points[best])
        peak_scores.append(scores[best])

    def _far_enough(candidate: np.ndarray, selected: list[np.ndarray], min_sep: float = max(10.0, 3.0 * dbscan_eps)) -> bool:
        if not selected:
            return True
        cand = np.asarray(candidate, dtype=float).reshape(1, 2)
        sel = np.asarray(selected, dtype=float)
        d_theta = np.abs(sel[:, 0] - cand[0, 0])
        d_theta = np.minimum(d_theta, 360.0 - d_theta)
        d_phi = np.abs(sel[:, 1] - cand[0, 1])
        dist = np.sqrt(d_theta ** 2 + d_phi ** 2)
        return bool(np.all(dist >= min_sep))

    selected_points: list[np.ndarray] = []
    selected_scores: list[float] = []
    for pt, sc in zip(peak_points, peak_scores):
        if _far_enough(pt, selected_points):
            selected_points.append(np.asarray(pt, dtype=float))
            selected_scores.append(float(sc))

    peak_points = selected_points
    peak_scores = selected_scores

    if len(peak_points) < k_targets:
        local_idx = local_maxima_indices(points, scores, k_neighbors=k_neighbors)
        for idx in local_idx[np.argsort(scores[local_idx])[::-1]]:
            candidate = points[idx]
            if not _far_enough(candidate, peak_points):
                continue
            peak_points.append(candidate)
            peak_scores.append(float(scores[idx]))
            if len(peak_points) >= k_targets:
                break

    if len(peak_points) < k_targets and len(points) >= k_targets:
        km_points = points
        n_clusters = min(k_targets, len(km_points))
        if n_clusters >= 1:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            km_labels = km.fit_predict(km_points)
            for label in np.unique(km_labels):
                idx = np.where(km_labels == label)[0]
                best = idx[np.argmax(scores[idx])]
                candidate = points[best]
                if not _far_enough(candidate, peak_points):
                    continue
                peak_points.append(candidate)
                peak_scores.append(float(scores[best]))
                if len(peak_points) >= k_targets:
                    break

    if len(peak_points) < k_targets:
        order_all = np.argsort(scores)[::-1]
        for idx in order_all:
            candidate = points[idx]
            if not _far_enough(candidate, peak_points):
                continue
            peak_points.append(candidate)
            peak_scores.append(float(scores[idx]))
            if len(peak_points) >= k_targets:
                break

    if len(peak_points) == 0:
        order = np.argsort(scores)[-k_targets:]
        peak_points = [points[i] for i in order]
        peak_scores = [float(scores[i]) for i in order]

    peak_points = np.asarray(peak_points, dtype=float)
    peak_scores = np.asarray(peak_scores, dtype=float)
    order = np.argsort(peak_scores)[::-1][:k_targets]
    peak_points = peak_points[order]

    def _sector_fallback() -> np.ndarray:
        edges = np.linspace(0.0, 360.0, k_targets + 1)
        chosen = []
        for s in range(k_targets):
            if s == k_targets - 1:
                idx = np.where((points[:, 0] >= edges[s]) & (points[:, 0] <= edges[s + 1]))[0]
            else:
                idx = np.where((points[:, 0] >= edges[s]) & (points[:, 0] < edges[s + 1]))[0]
            if idx.size == 0:
                continue
            best = idx[np.argmax(scores[idx])]
            chosen.append(points[best])
        if not chosen:
            return peak_points
        chosen = np.asarray(chosen, dtype=float)
        if len(chosen) < k_targets:
            order_all = np.argsort(scores)[::-1]
            for idx in order_all:
                candidate = points[idx]
                key = (round(float(candidate[0]), 6), round(float(candidate[1]), 6))
                existing = {(round(float(p[0]), 6), round(float(p[1]), 6)) for p in chosen}
                if key in existing:
                    continue
                chosen = np.vstack([chosen, candidate])
                if len(chosen) >= k_targets:
                    break
        return chosen[:k_targets]

    if len(peak_points) >= 2:
        theta_tmp = np.sort(peak_points[:, 0])
        sep = np.diff(np.r_[theta_tmp, theta_tmp[0] + 360.0])
        if np.min(sep) < max(20.0, 360.0 / (2.0 * k_targets)):
            peak_points = _sector_fallback()

    theta_est = peak_points[:, 0]
    phi_est = peak_points[:, 1]
    sort_idx = np.argsort(theta_est)
    theta_est = theta_est[sort_idx]
    phi_est = phi_est[sort_idx]
    if len(theta_est) < k_targets:
        theta_est = np.concatenate([theta_est, np.full(k_targets - len(theta_est), np.nan)])
        phi_est = np.concatenate([phi_est, np.full(k_targets - len(phi_est), np.nan)])

    return PeakExtractionResult(labels=labels, theta_est=theta_est, phi_est=phi_est, points=points, scores=scores)



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



def select_candidate_from_points(
    points: np.ndarray,
    objective: MusicObjective,
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray | None:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return None
    if pts.ndim == 1:
        pts = pts[None, :]
    scores = objective.score_batch(pts)
    pts, scores = unique_points_keep_best(pts, scores)
    order = np.argsort(scores)[::-1]
    for idx in order:
        if is_far_from_centers(pts[idx], excluded_centers, theta_sep, phi_sep):
            return pts[idx].copy()
    return None



def _bounded_grid_axis(
    lower: float,
    upper: float,
    step: float,
    *,
    periodic: bool,
) -> np.ndarray:
    """Build a grid axis without out-of-domain samples or periodic endpoint duplicates.

    For azimuth, ``upper`` represents the periodic boundary and is therefore excluded
    (e.g. 0 deg is retained and 360 deg is omitted).  For elevation, the physical
    upper boundary is included exactly even when the step does not divide the span.
    """
    lower = float(lower)
    upper = float(upper)
    step = float(step)
    if not np.isfinite([lower, upper, step]).all() or step <= 0.0 or upper <= lower:
        raise ValueError("grid bounds must be finite with upper > lower and step > 0")

    tol = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper), abs(step))
    count = int(np.floor((upper - lower) / step + tol))
    values = lower + step * np.arange(count + 1, dtype=float)
    if periodic:
        values = values[values < upper - tol]
    else:
        values = values[values <= upper + tol]
        values = np.minimum(values, upper)
        if values.size == 0 or not np.isclose(values[-1], upper, rtol=0.0, atol=tol):
            values = np.append(values, upper)
    if values.size == 0:
        values = np.array([lower], dtype=float)
    return values


def propose_grid_candidates(
    objective: MusicObjective,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
    theta_step: float = 6.0,
    phi_step: float = 4.0,
    n_keep: int = 6,
) -> np.ndarray:
    theta_vals = _bounded_grid_axis(theta_bounds[0], theta_bounds[1], theta_step, periodic=True)
    phi_vals = _bounded_grid_axis(phi_bounds[0], phi_bounds[1], phi_step, periodic=False)
    grid_theta, grid_phi = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    grid = np.column_stack([grid_theta.ravel(), grid_phi.ravel()])
    mask = far_from_centers_mask(grid, excluded_centers, theta_sep, phi_sep)
    if not np.any(mask):
        mask = np.ones(len(grid), dtype=bool)
    valid = grid[mask]
    scores = objective.score_batch(valid)
    order = np.argsort(scores)[::-1]
    chosen: list[np.ndarray] = []
    for idx in order:
        cand = valid[idx]
        if not is_far_from_centers(cand, chosen, 0.85 * theta_sep, 0.85 * phi_sep, theta_hard_factor=0.60):
            continue
        chosen.append(cand.copy())
        if len(chosen) >= n_keep:
            break
    return np.asarray(chosen, dtype=float) if chosen else np.empty((0, 2), dtype=float)



def build_global_evidence_cache(
    objective: MusicObjective,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    theta_step: float,
    phi_step: float,
) -> dict:
    """Evaluate a static MUSIC grid once and reuse it across sequential stages."""
    theta_vals = _bounded_grid_axis(theta_bounds[0], theta_bounds[1], theta_step, periodic=True)
    phi_vals = _bounded_grid_axis(phi_bounds[0], phi_bounds[1], phi_step, periodic=False)
    grid_theta, grid_phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    points = np.column_stack([grid_theta.ravel(), grid_phi.ravel()])
    objective_values = objective.objective_batch(points)
    return {
        "points": points,
        "objective": objective_values,
        "score": -10.0 * np.log10(objective_values),
        "theta_step": float(theta_step),
        "phi_step": float(phi_step),
    }


def select_cached_grid_candidates(
    cache: dict,
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
    n_keep: int,
) -> np.ndarray:
    points = np.asarray(cache["points"], dtype=float)
    scores = np.asarray(cache["score"], dtype=float)
    mask = far_from_centers_mask(points, excluded_centers, theta_sep, phi_sep)
    if not np.any(mask):
        mask = np.ones(len(points), dtype=bool)
    valid_idx = np.flatnonzero(mask)
    order = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    chosen: list[np.ndarray] = []
    for idx in order:
        cand = points[int(idx)]
        if not is_far_from_centers(cand, chosen, 0.85 * theta_sep, 0.85 * phi_sep, theta_hard_factor=0.60):
            continue
        chosen.append(cand.copy())
        if len(chosen) >= int(n_keep):
            break
    return np.asarray(chosen, dtype=float) if chosen else np.empty((0, 2), dtype=float)


def select_cached_global_candidate(
    cache: dict,
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray | None:
    points = np.asarray(cache["points"], dtype=float)
    values = np.asarray(cache["objective"], dtype=float)
    mask = far_from_centers_mask(points, excluded_centers, theta_sep, phi_sep)
    if not np.any(mask):
        mask = np.ones(len(points), dtype=bool)
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return None
    idx = int(valid_idx[int(np.argmin(values[valid_idx]))])
    return points[idx].copy()


def inject_guided_seeds(
    stage_init: np.ndarray,
    guides: np.ndarray,
    rng: np.random.Generator,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    theta_scale: float,
    phi_scale: float,
    max_inject: int,
) -> np.ndarray:
    if len(guides) == 0 or max_inject <= 0:
        return stage_init
    injected: list[np.ndarray] = []
    per_guide = max(2, int(np.ceil(max_inject / max(1, len(guides)))))
    for g in np.asarray(guides, dtype=float):
        jitter = np.column_stack([
            g[0] + rng.standard_normal(per_guide) * theta_scale,
            g[1] + rng.standard_normal(per_guide) * phi_scale,
        ])
        jitter = clamp_positions(jitter, theta_bounds, phi_bounds)
        injected.append(jitter)
        injected.append(np.asarray(g, dtype=float).reshape(1, 2))
    seeds = np.vstack(injected)[:max_inject]
    out = stage_init.copy()
    out[:len(seeds)] = seeds
    return out



def choose_best_candidate(
    candidates: list[np.ndarray | None],
    objective: MusicObjective,
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
) -> np.ndarray | None:
    valid: list[np.ndarray] = []
    for cand in candidates:
        if cand is None:
            continue
        arr = np.asarray(cand, dtype=float).reshape(1, 2)[0]
        if is_far_from_centers(arr, excluded_centers, theta_sep, phi_sep):
            valid.append(arr.copy())
    if not valid:
        return None
    vals = objective.objective_batch(np.asarray(valid, dtype=float))
    return np.asarray(valid[int(np.argmin(vals))], dtype=float).copy()


def coarse_global_candidate(
    objective: MusicObjective,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
    theta_step: float = 3.0,
    phi_step: float = 2.0,
) -> np.ndarray:
    theta_vals = _bounded_grid_axis(theta_bounds[0], theta_bounds[1], theta_step, periodic=True)
    phi_vals = _bounded_grid_axis(phi_bounds[0], phi_bounds[1], phi_step, periodic=False)
    grid_theta, grid_phi = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    grid = np.column_stack([grid_theta.ravel(), grid_phi.ravel()])
    mask = far_from_centers_mask(grid, excluded_centers, theta_sep, phi_sep)
    if not np.any(mask):
        mask = np.ones(len(grid), dtype=bool)
    valid = grid[mask]
    vals = objective.objective_batch(valid)
    return valid[int(np.argmin(vals))].copy()



def refine_peak_local(
    objective: MusicObjective,
    point: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
) -> np.ndarray:
    best = clamp_positions(point, theta_bounds, phi_bounds)[0]
    best_val = objective.objective(best)
    schedules = [
        (12.0, 8.0, 17, 13),
        (4.0, 3.0, 17, 13),
        (1.5, 1.2, 15, 11),
        (0.6, 0.5, 11, 9),
        (0.25, 0.22, 9, 9),
    ]
    for theta_span, phi_span, n_theta, n_phi in schedules:
        thetas = best[0] + np.linspace(-theta_span, theta_span, n_theta)
        phis = best[1] + np.linspace(-phi_span, phi_span, n_phi)
        grid_theta, grid_phi = np.meshgrid(thetas, phis, indexing='ij')
        grid = np.column_stack([grid_theta.ravel(), grid_phi.ravel()])
        grid = clamp_positions(grid, theta_bounds, phi_bounds)
        vals = objective.objective_batch(grid)
        idx = int(np.argmin(vals))
        if vals[idx] < best_val:
            best = grid[idx].copy()
            best_val = float(vals[idx])
    return best


def refine_peak_adaptive_local(
    objective: MusicObjective,
    point: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    theta_scale: float,
    phi_scale: float,
    support_scale: float = 0.52,
    contraction: float = 0.5,
    relative_stop: float = 1.0 / 64.0,
) -> np.ndarray:
    """Generic evidence-scale local refinement for the SEARIME pilot.

    Unlike the historical fixed five-stage angular schedule, the initial stencil
    is tied to the current stage evidence scale and contracts geometrically.
    The same dimensionless rule is used in every SNR/spacing/K condition.
    """
    best = clamp_positions(point, theta_bounds, phi_bounds)[0]
    best_val = float(objective.objective(best))
    initial = np.asarray([
        max(1e-6, float(support_scale) * float(theta_scale)),
        max(1e-6, float(support_scale) * float(phi_scale)),
    ], dtype=float)
    step = initial.copy()
    contraction = float(np.clip(contraction, 0.1, 0.9))
    relative_stop = float(max(1e-6, relative_stop))
    directions = np.asarray([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
    ], dtype=float)
    while float(np.max(step / initial)) >= relative_stop:
        grid = best[None, :] + directions * step[None, :]
        grid = clamp_positions(grid, theta_bounds, phi_bounds)
        vals = np.asarray(objective.objective_batch(grid), dtype=float)
        idx = int(np.argmin(vals))
        if float(vals[idx]) < best_val:
            best = grid[idx].copy()
            best_val = float(vals[idx])
        step *= contraction
    return best


def refine_peak_by_mode(
    objective: MusicObjective,
    point: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    *,
    mode: str,
    theta_scale: float,
    phi_scale: float,
    support_scale: float = 0.52,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "fixed":
        return refine_peak_local(objective, point, theta_bounds, phi_bounds)
    if mode == "adaptive":
        return refine_peak_adaptive_local(
            objective, point, theta_bounds, phi_bounds,
            theta_scale=theta_scale, phi_scale=phi_scale, support_scale=support_scale,
        )
    if mode == "none":
        return clamp_positions(point, theta_bounds, phi_bounds)[0]
    raise ValueError("local_refine_mode must be 'fixed', 'adaptive', or 'none'")


def refine_peak_de_local(
    objective: MusicObjective,
    point: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    rng: np.random.Generator,
    pop_size: int = 28,
    generations: int = 8,
    F: float = 0.7,
    CR: float = 0.8,
    theta_radius: float = 3.0,
    phi_radius: float = 2.0,
) -> np.ndarray:
    anchor = clamp_positions(point, theta_bounds, phi_bounds)[0]
    pop_size = int(max(6, pop_size))
    generations = int(max(1, generations))
    theta_radius = float(max(1e-6, theta_radius))
    phi_radius = float(max(1e-6, phi_radius))
    F = float(np.clip(F, 0.0, 2.0))
    CR = float(np.clip(CR, 0.0, 1.0))

    jitter = np.column_stack([
        rng.standard_normal(pop_size - 1) * theta_radius,
        rng.standard_normal(pop_size - 1) * phi_radius,
    ])
    population = np.vstack([anchor[None, :], anchor[None, :] + jitter])
    population = clamp_positions(population, theta_bounds, phi_bounds)
    fitness = objective.objective_batch(population)

    for _ in range(generations):
        for i in range(pop_size):
            pool = np.delete(np.arange(pop_size, dtype=int), i)
            if len(pool) < 3:
                continue
            r1, r2, r3 = rng.choice(pool, size=3, replace=False)
            a = population[int(r1)]
            b = population[int(r2)]
            c = population[int(r3)]
            mutant = a + F * (b - c)
            mutant = clamp_positions(mutant, theta_bounds, phi_bounds)[0]
            trial = population[i].copy()
            j_rand = int(rng.integers(0, 2))
            for j in range(2):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = clamp_positions(trial, theta_bounds, phi_bounds)[0]
            val = float(objective.objective(trial))
            if val < fitness[i]:
                population[i] = trial
                fitness[i] = val

    return population[int(np.argmin(fitness))].copy()


def angular_point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    pa = np.asarray(point_a, dtype=float).reshape(2)
    pb = np.asarray(point_b, dtype=float).reshape(2)
    d_theta = abs(pa[0] - pb[0])
    d_theta = min(d_theta, 360.0 - d_theta)
    d_phi = abs(pa[1] - pb[1])
    return float(np.sqrt(d_theta ** 2 + d_phi ** 2))



def _stage_value(schedule, stage: int, default):
    if schedule is None:
        return default
    if isinstance(schedule, np.ndarray):
        if schedule.ndim == 0:
            return schedule.item()
        if len(schedule) == 0:
            return default
        return schedule[min(stage, len(schedule) - 1)].item()
    if isinstance(schedule, (list, tuple)):
        if len(schedule) == 0:
            return default
        return schedule[min(stage, len(schedule) - 1)]
    return schedule



def propose_refined_grid_candidates(
    objective: MusicObjective,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
    n_keep: int = 6,
) -> np.ndarray:
    coarse = propose_grid_candidates(
        objective,
        theta_bounds,
        phi_bounds,
        excluded_centers,
        theta_sep,
        phi_sep,
        theta_step=6.0,
        phi_step=4.0,
        n_keep=max(n_keep * 2, 8),
    )
    if len(coarse) == 0:
        return coarse
    refined_bank = [coarse]
    for cand in coarse[: max(n_keep, 4)]:
        local = refine_peak_local(objective, cand, theta_bounds, phi_bounds)
        refined_bank.append(np.asarray(local, dtype=float).reshape(1, 2))
        thetas = cand[0] + np.linspace(-3.0, 3.0, 9)
        phis = cand[1] + np.linspace(-2.0, 2.0, 7)
        grid_t, grid_p = np.meshgrid(thetas, phis, indexing='ij')
        local_grid = np.column_stack([grid_t.ravel(), grid_p.ravel()])
        local_grid = clamp_positions(local_grid, theta_bounds, phi_bounds)
        refined_bank.append(local_grid)
    pts = np.vstack(refined_bank)
    pts, scores = unique_points_keep_best(pts, objective.score_batch(pts))
    order = np.argsort(scores)[::-1]
    chosen: list[np.ndarray] = []
    for idx in order:
        cand = pts[idx]
        if not is_far_from_centers(cand, chosen, 0.78 * theta_sep, 0.78 * phi_sep, theta_hard_factor=0.55):
            continue
        if not is_far_from_centers(cand, excluded_centers, theta_sep, phi_sep, theta_hard_factor=0.55):
            continue
        chosen.append(cand.copy())
        if len(chosen) >= n_keep:
            break
    return np.asarray(chosen, dtype=float) if chosen else np.empty((0, 2), dtype=float)



def confidence_gate_refine(
    objective: MusicObjective,
    anchor: np.ndarray,
    refined: np.ndarray,
    theta_sep: float,
    phi_sep: float,
    improve_ratio: float,
    move_factor: float,
) -> np.ndarray:
    """Parameter-free refinement acceptance used by the final SEARIME.

    Candidate confidence already controls whether local refinement is activated.
    Once refinement is executed, the refined point is committed only when it
    does not worsen the original MUSIC objective.  The legacy ratio/movement
    arguments remain in the signature for compatibility but are intentionally
    not used by the final rule.
    """
    if anchor is None or refined is None:
        return np.asarray(anchor if anchor is not None else refined, dtype=float)
    anchor = np.asarray(anchor, dtype=float).reshape(2)
    refined = np.asarray(refined, dtype=float).reshape(2)
    anchor_val = float(objective.objective(anchor))
    refined_val = float(objective.objective(refined))
    return refined if refined_val <= anchor_val else anchor



def coordinated_peak_polish(
    base_objective: MusicObjective,
    points: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    theta_sep: float,
    phi_sep: float,
    rounds: int = 2,
    rng: np.random.Generator | None = None,
    de_refine: bool = False,
    de_refine_pop_size: int = 28,
    de_refine_gens: int = 8,
    de_refine_F: float = 0.7,
    de_refine_CR: float = 0.8,
    de_refine_theta_radius: float = 3.0,
    de_refine_phi_radius: float = 2.0,
) -> np.ndarray:
    pts = clamp_positions(np.asarray(points, dtype=float), theta_bounds, phi_bounds)
    if len(pts) <= 1 or rounds <= 0:
        return pts
    rng = make_child_rng() if rng is None else rng
    penalty_strength = float(max(8.0, 1.5 * np.median(base_objective.objective_batch(pts))))
    for _ in range(int(rounds)):
        for i in range(len(pts)):
            others = np.delete(pts, i, axis=0)
            local_obj = PenalizedMusicObjective(
                base_objective,
                others,
                penalty_strength=penalty_strength,
                theta_sigma=max(theta_sep, 1e-3),
                phi_sigma=max(phi_sep, 1e-3),
            )
            if de_refine:
                candidate = refine_peak_de_local(
                    local_obj,
                    pts[i],
                    theta_bounds,
                    phi_bounds,
                    rng=rng,
                    pop_size=de_refine_pop_size,
                    generations=de_refine_gens,
                    F=de_refine_F,
                    CR=de_refine_CR,
                    theta_radius=de_refine_theta_radius,
                    phi_radius=de_refine_phi_radius,
                )
                candidate = refine_peak_de_local(
                    base_objective,
                    candidate,
                    theta_bounds,
                    phi_bounds,
                    rng=rng,
                    pop_size=de_refine_pop_size,
                    generations=max(2, int(round(de_refine_gens * 0.6))),
                    F=de_refine_F,
                    CR=de_refine_CR,
                    theta_radius=max(1e-6, 0.75 * de_refine_theta_radius),
                    phi_radius=max(1e-6, 0.75 * de_refine_phi_radius),
                )
            else:
                candidate = refine_peak_local(local_obj, pts[i], theta_bounds, phi_bounds)
                candidate = refine_peak_local(base_objective, candidate, theta_bounds, phi_bounds)
            if is_far_from_centers(candidate, others, 0.72 * theta_sep, 0.72 * phi_sep, theta_hard_factor=0.50):
                pts[i] = candidate
        order = np.argsort(pts[:, 0])
        pts = pts[order]
    return pts





def collect_diverse_archive_candidates(
    points: np.ndarray,
    objective: MusicObjective,
    excluded_centers: np.ndarray | None,
    theta_sep: float,
    phi_sep: float,
    max_keep: int = 8,
    top_pool: int = 96,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    pts = unique_points_keep_best(pts, objective.score_batch(pts))[0]
    scores = objective.score_batch(pts)
    order = np.argsort(scores)[::-1]
    chosen: list[np.ndarray] = []
    for idx in order[: max(1, int(top_pool))]:
        cand = pts[idx]
        if not is_far_from_centers(cand, excluded_centers, theta_sep, phi_sep, theta_hard_factor=0.50):
            continue
        if not is_far_from_centers(cand, chosen, 0.58 * theta_sep, 0.58 * phi_sep, theta_hard_factor=0.40):
            continue
        chosen.append(cand.copy())
        if len(chosen) >= max_keep:
            break
    return np.asarray(chosen, dtype=float) if chosen else np.empty((0, 2), dtype=float)


def candidate_support_scores(
    candidates: np.ndarray,
    evidence_points: np.ndarray,
    evidence_scores: np.ndarray,
    theta_sep: float,
    phi_sep: float,
    support_scale: float = 0.52,
) -> np.ndarray:
    cands = np.asarray(candidates, dtype=float)
    pts = np.asarray(evidence_points, dtype=float)
    scores = np.asarray(evidence_scores, dtype=float)
    if cands.size == 0:
        return np.empty((0,), dtype=float)
    if cands.ndim == 1:
        cands = cands[None, :]
    if pts.size == 0:
        return np.zeros(len(cands), dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    score_scale = scores - np.min(scores)
    denom = float(np.ptp(score_scale) + 1e-12)
    weights = 0.25 + 0.75 * (score_scale / denom if denom > 0 else np.ones_like(scores))
    d_theta = np.abs(cands[:, 0][:, None] - pts[:, 0][None, :])
    d_theta = np.minimum(d_theta, 360.0 - d_theta)
    d_phi = np.abs(cands[:, 1][:, None] - pts[:, 1][None, :])
    support_scale = float(max(1e-6, support_scale))
    metric = np.exp(-0.5 * ((d_theta / max(support_scale * theta_sep, 1e-6)) ** 2 + (d_phi / max(support_scale * phi_sep, 1e-6)) ** 2))
    return metric @ weights


def stage_low_confidence(
    candidates: np.ndarray,
    objective: MusicObjective,
    evidence_points: np.ndarray,
    evidence_scores: np.ndarray,
    theta_sep: float,
    phi_sep: float,
    min_gap_ratio: float = 0.012,
    min_support_ratio: float = 0.88,
    support_scale: float = 0.52,
) -> tuple[bool, dict]:
    cands = np.asarray(candidates, dtype=float)
    if cands.size == 0:
        return True, {"gap_ratio": 0.0, "support_ratio": 0.0}
    if cands.ndim == 1:
        cands = cands[None, :]
    vals = objective.objective_batch(cands)
    supports = candidate_support_scores(cands, evidence_points, evidence_scores, theta_sep, phi_sep, support_scale=support_scale)
    order = np.argsort(vals)
    best_idx = int(order[0])
    second_val = float(vals[int(order[1])]) if len(order) > 1 else float(vals[best_idx] * 2.0)
    gap_ratio = float((second_val - vals[best_idx]) / (abs(vals[best_idx]) + 1e-12))
    best_support = float(supports[best_idx])
    max_support = float(np.max(supports)) if len(supports) else 0.0
    support_ratio = best_support / (max_support + 1e-12)
    # Equivalent compact confidence form: the original OR rule is exactly
    # Gamma_q < 1 with Gamma_q = min(g/g0, rho/rho0).
    gap_score = gap_ratio / max(float(min_gap_ratio), 1e-12)
    support_score = support_ratio / max(float(min_support_ratio), 1e-12)
    confidence_index = float(min(gap_score, support_score))
    low_conf = bool(confidence_index < 1.0)
    return low_conf, {
        "gap_ratio": gap_ratio,
        "support_ratio": support_ratio,
        "gap_threshold": float(min_gap_ratio),
        "support_threshold": float(min_support_ratio),
        "confidence_index": confidence_index,
        "best_idx": best_idx,
    }


def local_spectrum_exclusion_scales(
    objective: MusicObjective,
    center: np.ndarray,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    base_theta_sep: float,
    base_phi_sep: float,
    *,
    relative_drop: float = 0.10,
    theta_probe_step: float = 1.0,
    phi_probe_step: float = 1.0,
    min_scale: float = 0.55,
    max_scale: float = 1.35,
) -> tuple[float, float, dict]:
    """Derive a stage exclusion ellipse from locally observable MUSIC curvature.

    A sharp, isolated peak should receive a smaller exclusion window than a
    broad or ambiguous one.  The function probes only the local MUSIC objective
    around an already committed estimate, performs no access to simulation
    labels, and falls back safely to the configured base scale.
    """
    center = clamp_positions(np.asarray(center, dtype=float), theta_bounds, phi_bounds)[0]
    base_theta_sep = float(base_theta_sep)
    base_phi_sep = float(base_phi_sep)
    if base_theta_sep <= 0.0 or base_phi_sep <= 0.0:
        raise ValueError("base exclusion scales must be positive")
    relative_drop = float(relative_drop)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if not 0.0 < relative_drop < 1.0:
        raise ValueError("relative_drop must lie between zero and one")
    if not 0.0 < min_scale <= max_scale:
        raise ValueError("adaptive exclusion scale bounds are invalid")

    probe_theta_max = max(base_theta_sep * max_scale, theta_probe_step)
    probe_phi_max = max(base_phi_sep * max_scale, phi_probe_step)
    theta_offsets = np.arange(theta_probe_step, probe_theta_max + 0.5 * theta_probe_step, theta_probe_step)
    phi_offsets = np.arange(phi_probe_step, probe_phi_max + 0.5 * phi_probe_step, phi_probe_step)
    probe_points = [center]
    for offset in theta_offsets:
        probe_points.extend(([center[0] - offset, center[1]], [center[0] + offset, center[1]]))
    for offset in phi_offsets:
        probe_points.extend(([center[0], center[1] - offset], [center[0], center[1] + offset]))
    probe_points = clamp_positions(np.asarray(probe_points, dtype=float), theta_bounds, phi_bounds)
    scores = objective.score_batch(probe_points)
    center_score = float(scores[0])
    threshold = center_score - max(0.05, abs(center_score) * relative_drop)

    def half_width(offsets: np.ndarray, score_pairs: np.ndarray, base: float) -> tuple[float, bool]:
        pair_minimum = np.min(score_pairs, axis=1)
        crossed = np.flatnonzero(pair_minimum <= threshold)
        if crossed.size:
            return float(offsets[int(crossed[0])]), True
        return float(offsets[-1]), False

    theta_scores = scores[1 : 1 + 2 * len(theta_offsets)].reshape(-1, 2)
    phi_scores = scores[1 + 2 * len(theta_offsets) :].reshape(-1, 2)
    theta_width, theta_crossed = half_width(theta_offsets, theta_scores, base_theta_sep)
    phi_width, phi_crossed = half_width(phi_offsets, phi_scores, base_phi_sep)
    theta_scale = float(np.clip(theta_width / base_theta_sep, min_scale, max_scale))
    phi_scale = float(np.clip(phi_width / base_phi_sep, min_scale, max_scale))
    return (
        base_theta_sep * theta_scale,
        base_phi_sep * phi_scale,
        {
            "center_score": center_score,
            "score_drop_threshold": threshold,
            "theta_half_width": theta_width,
            "phi_half_width": phi_width,
            "theta_scale": theta_scale,
            "phi_scale": phi_scale,
            "theta_drop_crossed": bool(theta_crossed),
            "phi_drop_crossed": bool(phi_crossed),
        },
    )


def choose_consensus_candidate(
    candidates: np.ndarray,
    objective: MusicObjective,
    evidence_points: np.ndarray,
    evidence_scores: np.ndarray,
    excluded_centers: np.ndarray | None,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    theta_sep: float,
    phi_sep: float,
    rng: np.random.Generator,
    use_local_refine: bool = True,
    local_refine_mode: str = "fixed",
    local_refine_support_scale: float = 0.52,
    evidence_support_scale: float = 0.52,
    evidence_consensus_weight: float = 0.55,
    confidence_gated_refine: bool = False,
    improve_ratio: float = 0.0,
    move_factor: float = 0.0,
    de_refine: bool = False,
    de_refine_pop_size: int = 28,
    de_refine_gens: int = 8,
    de_refine_F: float = 0.7,
    de_refine_CR: float = 0.8,
    de_refine_theta_radius: float = 3.0,
    de_refine_phi_radius: float = 2.0,
) -> np.ndarray | None:
    cands = np.asarray(candidates, dtype=float)
    if cands.size == 0:
        return None
    if cands.ndim == 1:
        cands = cands[None, :]
    refined_bank: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for cand in cands:
        if not is_far_from_centers(cand, excluded_centers, theta_sep, phi_sep, theta_hard_factor=0.55):
            continue
        anchor = clamp_positions(cand, theta_bounds, phi_bounds)[0]
        final = anchor.copy()
        if use_local_refine:
            if de_refine:
                local = refine_peak_de_local(
                    objective,
                    anchor,
                    theta_bounds,
                    phi_bounds,
                    rng=rng,
                    pop_size=de_refine_pop_size,
                    generations=de_refine_gens,
                    F=de_refine_F,
                    CR=de_refine_CR,
                    theta_radius=de_refine_theta_radius,
                    phi_radius=de_refine_phi_radius,
                )
            else:
                local = refine_peak_by_mode(
                    objective, anchor, theta_bounds, phi_bounds,
                    mode=local_refine_mode, theta_scale=theta_sep, phi_scale=phi_sep,
                    support_scale=local_refine_support_scale,
                )
            if confidence_gated_refine:
                final = confidence_gate_refine(objective, anchor, local, theta_sep, phi_sep, improve_ratio=improve_ratio, move_factor=move_factor)
            else:
                final = local
        if not is_far_from_centers(final, excluded_centers, 0.78 * theta_sep, 0.78 * phi_sep, theta_hard_factor=0.50):
            final = anchor
        refined_bank.append(final)
        anchors.append(anchor)
    if not refined_bank:
        return None
    refined_bank = np.asarray(refined_bank, dtype=float)
    refined_bank, _ = unique_points_keep_best(refined_bank, objective.score_batch(refined_bank))
    vals = objective.objective_batch(refined_bank)
    supports = candidate_support_scores(refined_bank, evidence_points, evidence_scores, theta_sep, phi_sep, support_scale=evidence_support_scale)
    obj_norm = (vals - np.min(vals)) / (np.ptp(vals) + 1e-12)
    sup_norm = (supports - np.min(supports)) / (np.ptp(supports) + 1e-12) if len(supports) > 1 else np.ones_like(supports)
    if evidence_consensus_weight is None:
        # Q = 2 f_norm - C_norm is exactly 2*(f_norm - 0.5 C_norm);
        # therefore it removes the exposed 0.5 weight without changing ranking.
        combined = 2.0 * obj_norm - sup_norm
    else:
        combined = obj_norm - float(evidence_consensus_weight) * sup_norm
    best = int(np.argmin(combined))
    return refined_bank[best].copy()






def run_single_pass_peak_search(
    base_objective: MusicObjective,
    optimizer_fn: Callable[..., dict],
    k_targets: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    pop_size: int,
    max_iter: int,
    population_init: np.ndarray | None,
    rng: np.random.Generator,
    optimizer_kwargs: dict | None = None,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 4,
    k_neighbors: int = 8,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    base_objective.reset_counts()
    base_objective.set_phase("initialization")
    shared_init = make_initial_population(population_init, pop_size, theta_bounds, phi_bounds, rng)
    base_objective.set_phase("search_kernel")
    result = optimizer_fn(
        objective=base_objective,
        theta_bounds=theta_bounds,
        phi_bounds=phi_bounds,
        pop_size=pop_size,
        max_iter=max_iter,
        population_init=shared_init,
        rng=rng,
        **optimizer_kwargs,
    )
    base_objective.set_phase("finalization")
    points = np.asarray(result.get('archive_points', result.get('population')), dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    points = clamp_positions(points, theta_bounds, phi_bounds)
    scores = base_objective.score_batch(points) if len(points) else np.empty((0,), dtype=float)
    peak_res = extract_peaks_from_points(points, scores, k_targets, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples, k_neighbors=k_neighbors)
    curve = np.asarray(result.get('curve', np.empty((0,), dtype=float)), dtype=float)
    return (points, peak_res.labels), (points, scores), peak_res.theta_est, peak_res.phi_est, curve


def _joint_grid_peak_candidates(
    objective: MusicObjective,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    *,
    theta_step: float = 1.0,
    phi_step: float = 1.0,
    max_candidates: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global local maxima without a source-separation exclusion.

    The old sequential path used the same spatial scale both to suppress
    numerical duplicates and to prohibit the next source.  That conflates two
    different tasks and makes close sources impossible to recover.  This
    helper only performs a grid-local maximum test.  It does not compare a
    candidate with an already selected source and therefore does not encode a
    physical minimum angular separation.
    """
    theta_vals = _bounded_grid_axis(theta_bounds[0], theta_bounds[1], theta_step, periodic=True)
    phi_vals = _bounded_grid_axis(phi_bounds[0], phi_bounds[1], phi_step, periodic=False)
    grid_theta, grid_phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    points = np.column_stack([grid_theta.ravel(), grid_phi.ravel()])
    values = np.asarray(objective.objective_batch(points), dtype=float)
    scores = -10.0 * np.log10(values + EPS)
    score_grid = scores.reshape(len(theta_vals), len(phi_vals))
    local = np.isfinite(score_grid) & (
        score_grid >= ndimage.maximum_filter(score_grid, size=3, mode=("wrap", "nearest"))
    )
    local_indices = np.flatnonzero(local.ravel())
    order_local = local_indices[np.argsort(scores[local_indices])[::-1]] if len(local_indices) else np.empty(0, dtype=int)
    order_all = np.argsort(scores)[::-1]
    chosen: list[np.ndarray] = []
    chosen_scores: list[float] = []

    # The only duplicate test here is a numerical identity test.  In
    # particular, it is far below the angular spacings used in a resolution
    # experiment and cannot act as a hidden resolution threshold.
    duplicate_eps = 1e-6

    def append_if_new(index: int) -> None:
        point = points[int(index)]
        if any(angular_point_distance(point, previous) <= duplicate_eps for previous in chosen):
            return
        chosen.append(point.copy())
        chosen_scores.append(float(scores[int(index)]))

    for index in order_local:
        append_if_new(int(index))
        if len(chosen) >= int(max_candidates):
            break
    if len(chosen) < int(max_candidates):
        for index in order_all:
            append_if_new(int(index))
            if len(chosen) >= int(max_candidates):
                break
    return np.asarray(chosen, dtype=float), np.asarray(chosen_scores, dtype=float)


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


def _adaptive_sequential_candidate_selection(
    candidates: np.ndarray,
    objective: MusicObjective,
    k_targets: int,
    *,
    coherence_weight: float = 1.20,
) -> np.ndarray:
    """Commit one candidate at a time using soft steering coherence.

    Individual MUSIC scores alone may select several archive points from the
    same peak.  Each sequential commitment therefore scores the proposed
    ``selected + candidate`` set using a continuous steering-vector coherence
    penalty.  The penalty is zero for orthogonal candidates and approaches one
    only for duplicate steering vectors; it never rejects a candidate because
    its angle is below a configured separation threshold.
    """
    points = np.asarray(candidates, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    k_targets = int(k_targets)
    if k_targets <= 0:
        return np.empty((0, 2), dtype=float)
    if len(points) <= k_targets:
        return points.copy()

    values = np.asarray(objective.objective_batch(points), dtype=float)
    individual_scores = -10.0 * np.log10(values + EPS)
    score_min = float(np.min(individual_scores))
    score_span = float(np.ptp(individual_scores))
    normalized = (individual_scores - score_min) / (score_span + EPS)

    # Keep exhaustive combinations bounded for K>2 while retaining more than
    # K candidates for the joint choice.  The pool is already ranked by the
    # observation-only MUSIC evidence.
    # Keep enough of the global candidate bank to include a weaker second
    # peak.  A small ``4*K`` truncation would recreate the old failure mode
    # whenever several archive points from the strongest peak outrank it.
    pool_size = min(len(points), max(k_targets + 8, min(32, 8 * k_targets)))
    pool = points[np.argsort(individual_scores)[::-1][:pool_size]]
    pool_norm = normalized[np.argsort(individual_scores)[::-1][:pool_size]]
    steering = np.exp(
        -2j * np.pi / objective.wavelength
        * (_angles_to_unit_vectors(pool) @ objective.array_pos.T)
    )
    gram = np.abs(steering @ steering.conj().T / max(1, objective.array_pos.shape[0])) ** 2
    np.fill_diagonal(gram, 0.0)

    def combo_score(indices: tuple[int, ...]) -> float:
        idx = np.asarray(indices, dtype=int)
        if len(idx) <= 1:
            coherence = 0.0
        else:
            upper = gram[np.ix_(idx, idx)]
            coherence = float(np.sum(upper) / max(1, len(idx) * (len(idx) - 1)))
        return float(np.mean(pool_norm[idx]) - float(coherence_weight) * coherence)

    selected = [int(np.argmax(pool_norm))]
    while len(selected) < k_targets and len(selected) < len(pool):
        remaining = [idx for idx in range(len(pool)) if idx not in selected]
        best = max(remaining, key=lambda idx: combo_score(tuple(selected + [int(idx)])))
        selected.append(int(best))
    return pool[np.asarray(selected, dtype=int)].copy()


def _candidate_subspace_fit(
    points: np.ndarray,
    objective: MusicObjective,
    k_targets: int,
    fit_mode: str = "trace_subspace",
) -> tuple[float, float, int]:
    """Measure how well a candidate set spans the estimated signal subspace.

    The steering matrix is orthogonalized with an SVD.  Its numerical rank is
    determined only by the standard machine-precision tolerance; no angular
    separation, coherence weight, or user-selected rank threshold is used.

    ``fit_mode='trace_subspace'`` returns ``loss = K - tr(Q^H P_s Q)``.
    ``fit_mode='optimal_wsf'`` returns the normalized weighted residual
    ``sum_i w_i ||(I-QQ^H)e_i||^2 / sum_i w_i`` with analytic stochastic-WSF
    weights ``w_i=(lambda_i-sigma^2)^2/lambda_i``.  These weights come only
    from the sample eigensystem and add no user-selected balance coefficient.
    """
    mode = str(fit_mode).strip().lower()
    if mode not in {"trace_subspace", "optimal_wsf"}:
        raise ValueError("fit_mode must be 'trace_subspace' or 'optimal_wsf'")
    empty_loss = 1.0 if mode == "optimal_wsf" else float(k_targets)
    candidate_points = np.asarray(points, dtype=float)
    if candidate_points.size == 0:
        return empty_loss, 0.0, 0
    if candidate_points.ndim == 1:
        candidate_points = candidate_points[None, :]
    if candidate_points.ndim != 2 or candidate_points.shape[1] != 2:
        raise ValueError("candidate points must have shape (n, 2)")

    directions = _angles_to_unit_vectors(candidate_points)
    steering = np.exp(
        -2j * np.pi / objective.wavelength
        * (objective.array_pos @ directions.T)
    )
    left_vectors, singular_values, _ = np.linalg.svd(steering, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        return empty_loss, 0.0, 0
    rank_tolerance = (
        np.finfo(float).eps
        * float(max(steering.shape))
        * float(singular_values[0])
    )
    numerical_rank = int(np.sum(singular_values > rank_tolerance))
    if numerical_rank <= 0:
        return empty_loss, 0.0, 0

    basis = left_vectors[:, :numerical_rank]
    if mode == "optimal_wsf":
        if (
            objective.signal_eigenvectors is None
            or objective.signal_eigenvalues is None
            or objective.noise_variance is None
        ):
            raise ValueError(
                "optimal_wsf requires an objective built from the sample covariance"
            )
        signal_vectors = np.asarray(objective.signal_eigenvectors, dtype=complex)
        signal_values = np.asarray(objective.signal_eigenvalues, dtype=float)
        if signal_vectors.ndim != 2 or signal_vectors.shape[1] != len(signal_values):
            raise ValueError("inconsistent signal eigensystem in MusicObjective")
        residual = signal_vectors - basis @ (basis.conj().T @ signal_vectors)
        weights = np.maximum(signal_values - float(objective.noise_variance), 0.0) ** 2
        weights /= np.maximum(signal_values, np.finfo(float).tiny)
        if float(np.sum(weights)) <= np.finfo(float).tiny:
            weights = np.ones_like(signal_values)
        loss = float(
            np.sum(weights * np.sum(np.abs(residual) ** 2, axis=0))
            / np.sum(weights)
        )
        loss = float(np.clip(loss, 0.0, 1.0))
        return loss, float(1.0 - loss), numerical_rank

    sensor_count = int(objective.array_pos.shape[0])
    signal_projector = np.eye(sensor_count, dtype=complex) - np.asarray(
        objective.noise_projector,
        dtype=complex,
    )
    signal_projector = 0.5 * (signal_projector + signal_projector.conj().T)
    captured_energy = float(
        np.real(np.trace(basis.conj().T @ signal_projector @ basis))
    )
    # Roundoff can move an ideal projector trace a few ulps outside its
    # mathematical interval.  Clipping is numerical hygiene, not a tuning
    # parameter.
    captured_energy = float(np.clip(captured_energy, 0.0, float(k_targets)))
    return float(max(0.0, float(k_targets) - captured_energy)), captured_energy, numerical_rank


def _adaptive_subspace_gain_candidate_selection(
    candidates: np.ndarray,
    objective: MusicObjective,
    k_targets: int,
    fit_mode: str = "trace_subspace",
) -> tuple[np.ndarray, list[dict]]:
    """Commit candidates by their marginal signal-subspace contribution.

    The first candidate is the strongest individual MUSIC candidate.  Every
    later commitment minimizes the joint candidate-set subspace loss.  This
    makes duplicate or redundant candidates unattractive without imposing a
    physical exclusion angle or balancing a MUSIC score against a hand-tuned
    coherence penalty.
    """
    points = np.asarray(candidates, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float), []
    if points.ndim == 1:
        points = points[None, :]
    k_targets = int(k_targets)
    if k_targets <= 0:
        return np.empty((0, 2), dtype=float), []

    values = np.asarray(objective.objective_batch(points), dtype=float)
    individual_scores = -10.0 * np.log10(values + EPS)
    ranking = np.argsort(individual_scores)[::-1]
    # For K>3, do not truncate the candidate set by individual MUSIC score.
    # A lower-amplitude but correct source can otherwise be removed before
    # the set-level signal-subspace criterion is evaluated.  The full pool is
    # still cheap here because each candidate fit uses only an M x K steering
    # matrix (M=12 in the paper UCA).  K<=3 keeps the bounded pool used by the
    # exact combinatorial selector.
    pool_size = (
        len(points)
        if k_targets > 3
        else min(len(points), max(k_targets + 8, min(32, 8 * k_targets)))
    )
    pool_indices = ranking[:pool_size]
    pool = points[pool_indices]
    pool_scores = individual_scores[pool_indices]

    selected = [0]
    first_loss, first_energy, first_rank = _candidate_subspace_fit(
        pool[[0]],
        objective,
        k_targets,
        fit_mode=fit_mode,
    )
    empty_loss = (
        1.0 if str(fit_mode).strip().lower() == "optimal_wsf" else float(k_targets)
    )
    trace = [{
        "step": 1,
        "candidate_pool_index": int(pool_indices[0]),
        "theta": float(pool[0, 0]),
        "phi": float(pool[0, 1]),
        "individual_music_score_db": float(pool_scores[0]),
        "subspace_loss": float(first_loss),
        "captured_signal_subspace_energy": float(first_energy),
        "numerical_rank": int(first_rank),
        "marginal_subspace_gain": float(empty_loss - first_loss),
        "set_fit_mode": str(fit_mode),
    }]
    previous_loss = float(first_loss)

    while len(selected) < k_targets and len(selected) < len(pool):
        remaining = [index for index in range(len(pool)) if index not in selected]
        evaluated: list[tuple[float, float, int, int]] = []
        for index in remaining:
            trial_indices = np.asarray(selected + [int(index)], dtype=int)
            loss, energy, rank = _candidate_subspace_fit(
                pool[trial_indices],
                objective,
                k_targets,
                fit_mode=fit_mode,
            )
            evaluated.append((float(loss), -float(pool_scores[index]), int(index), int(rank)))
        best_loss, _, best, best_rank = min(evaluated)
        selected.append(int(best))
        _, best_energy, _ = _candidate_subspace_fit(
            pool[np.asarray(selected, dtype=int)],
            objective,
            k_targets,
            fit_mode=fit_mode,
        )
        trace.append({
            "step": int(len(selected)),
            "candidate_pool_index": int(pool_indices[best]),
            "theta": float(pool[best, 0]),
            "phi": float(pool[best, 1]),
            "individual_music_score_db": float(pool_scores[best]),
            "subspace_loss": float(best_loss),
            "captured_signal_subspace_energy": float(best_energy),
            "numerical_rank": int(best_rank),
            "marginal_subspace_gain": float(previous_loss - best_loss),
            "set_fit_mode": str(fit_mode),
        })
        previous_loss = float(best_loss)

    return pool[np.asarray(selected, dtype=int)].copy(), trace


def _exact_subspace_fit_candidate_selection(
    candidates: np.ndarray,
    objective: MusicObjective,
    k_targets: int,
    fit_mode: str = "trace_subspace",
) -> tuple[np.ndarray, dict]:
    """Return the exact best K-set over the same bounded ASG candidate pool.

    This is intended as a K=2/K=3 development oracle.  It quantifies the cost
    of greedy commitment without changing the publication algorithm.  Larger
    K is rejected because exhaustive subset enumeration would not scale.
    """
    points = np.asarray(candidates, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float), {
            "subspace_loss": (
                1.0 if str(fit_mode).strip().lower() == "optimal_wsf" else float(k_targets)
            ),
            "captured_signal_subspace_energy": 0.0,
            "combinations_evaluated": 0,
            "set_fit_mode": str(fit_mode),
        }
    if points.ndim == 1:
        points = points[None, :]
    k_targets = int(k_targets)
    if k_targets not in {1, 2, 3}:
        raise ValueError("exact subspace-fit oracle is limited to K in {1, 2, 3}")
    if len(points) < k_targets:
        loss, energy, rank = _candidate_subspace_fit(
            points,
            objective,
            k_targets,
            fit_mode=fit_mode,
        )
        return points.copy(), {
            "subspace_loss": loss,
            "captured_signal_subspace_energy": energy,
            "numerical_rank": rank,
            "combinations_evaluated": 1,
            "set_fit_mode": str(fit_mode),
        }

    values = np.asarray(objective.objective_batch(points), dtype=float)
    individual_scores = -10.0 * np.log10(values + EPS)
    ranking = np.argsort(individual_scores)[::-1]
    # For K>3, do not truncate the candidate set by individual MUSIC score.
    # A lower-amplitude but correct source can otherwise be removed before
    # the set-level signal-subspace criterion is evaluated.  The full pool is
    # still cheap here because each candidate fit uses only an M x K steering
    # matrix (M=12 in the paper UCA).  K<=3 keeps the bounded pool used by the
    # exact combinatorial selector.
    # K=2 resolution experiments can afford exact enumeration over the full
    # numerically unique candidate pool.  This avoids discarding a weaker
    # but physically correct second source before the set-level fit is tested.
    # K=3 remains bounded to keep exact enumeration inexpensive.
    pool_size = (
        len(points)
        if k_targets == 2
        else min(len(points), max(k_targets + 8, min(32, 8 * k_targets)))
    )
    pool_indices = ranking[:pool_size]
    pool = points[pool_indices]

    best_indices: tuple[int, ...] | None = None
    best_loss = float("inf")
    best_energy = float("-inf")
    best_rank = 0
    evaluated = 0
    for indices in combinations(range(len(pool)), k_targets):
        evaluated += 1
        loss, energy, rank = _candidate_subspace_fit(
            pool[np.asarray(indices, dtype=int)],
            objective,
            k_targets,
            fit_mode=fit_mode,
        )
        # MUSIC evidence breaks only numerical ties in the set-level loss.
        if best_indices is None or loss < best_loss - 16.0 * np.finfo(float).eps:
            choose = True
        elif abs(loss - best_loss) <= 16.0 * np.finfo(float).eps:
            old_score = float(np.mean(individual_scores[pool_indices[list(best_indices)]]))
            new_score = float(np.mean(individual_scores[pool_indices[list(indices)]]))
            choose = new_score > old_score
        else:
            choose = False
        if choose:
            best_indices = tuple(int(index) for index in indices)
            best_loss = float(loss)
            best_energy = float(energy)
            best_rank = int(rank)

    assert best_indices is not None
    selected = pool[np.asarray(best_indices, dtype=int)].copy()
    return selected, {
        "subspace_loss": best_loss,
        "captured_signal_subspace_energy": best_energy,
        "numerical_rank": best_rank,
        "combinations_evaluated": int(evaluated),
        "set_fit_mode": str(fit_mode),
        "source_candidate_pool_indices": [
            int(pool_indices[index]) for index in best_indices
        ],
    }


def _subspace_gain_exchange_candidate_selection(
    candidates: np.ndarray,
    objective: MusicObjective,
    k_targets: int,
    fit_mode: str = "trace_subspace",
) -> tuple[np.ndarray, dict]:
    """Improve greedy ASG with parameter-free one-candidate exchanges.

    Each accepted exchange must reduce the set-level subspace loss beyond a
    roundoff-scaled tolerance.  Because the candidate set is finite and the
    loss strictly decreases, the procedure terminates without an iteration
    limit or an angle-dependent stopping rule.
    """
    points = np.asarray(candidates, dtype=float)
    normalized_fit_mode = str(fit_mode).strip().lower()

    def evaluate_fit(candidate_points: np.ndarray) -> tuple[float, float, int]:
        # Keep the original three-argument call on the legacy trace path so
        # downstream diagnostic monkeypatches remain source-compatible.
        if normalized_fit_mode == "trace_subspace":
            return _candidate_subspace_fit(candidate_points, objective, k_targets)
        return _candidate_subspace_fit(
            candidate_points,
            objective,
            k_targets,
            fit_mode=normalized_fit_mode,
        )

    if points.size == 0:
        return np.empty((0, 2), dtype=float), {
            "initial_subspace_loss": (
                1.0 if normalized_fit_mode == "optimal_wsf" else float(k_targets)
            ),
            "final_subspace_loss": (
                1.0 if normalized_fit_mode == "optimal_wsf" else float(k_targets)
            ),
            "accepted_exchanges": [],
            "set_fit_mode": normalized_fit_mode,
        }
    if points.ndim == 1:
        points = points[None, :]
    k_targets = int(k_targets)
    if normalized_fit_mode == "trace_subspace":
        greedy_points, greedy_trace = _adaptive_subspace_gain_candidate_selection(
            points,
            objective,
            k_targets,
        )
    else:
        greedy_points, greedy_trace = _adaptive_subspace_gain_candidate_selection(
            points,
            objective,
            k_targets,
            fit_mode=normalized_fit_mode,
        )
    if len(greedy_points) < k_targets:
        final_loss, final_energy, final_rank = evaluate_fit(greedy_points)
        return greedy_points, {
            "initial_subspace_loss": final_loss,
            "final_subspace_loss": final_loss,
            "captured_signal_subspace_energy": final_energy,
            "numerical_rank": final_rank,
            "accepted_exchanges": [],
            "greedy_trace": greedy_trace,
            "set_fit_mode": normalized_fit_mode,
        }

    values = np.asarray(objective.objective_batch(points), dtype=float)
    individual_scores = -10.0 * np.log10(values + EPS)
    ranking = np.argsort(individual_scores)[::-1]
    # For K>3, do not truncate the candidate set by individual MUSIC score.
    # A lower-amplitude but correct source can otherwise be removed before
    # the set-level signal-subspace criterion is evaluated.  The full pool is
    # still cheap here because each candidate fit uses only an M x K steering
    # matrix (M=12 in the paper UCA).  K<=3 keeps the bounded pool used by the
    # exact combinatorial selector.
    pool_size = (
        len(points)
        if k_targets > 3
        else min(len(points), max(k_targets + 8, min(32, 8 * k_targets)))
    )
    pool_indices = ranking[:pool_size]
    pool = points[pool_indices]
    pool_scores = individual_scores[pool_indices]

    selected: list[int] = []
    for greedy_point in greedy_points:
        matches = np.flatnonzero(np.all(pool == greedy_point, axis=1))
        if not len(matches):
            raise RuntimeError("greedy ASG point is missing from its ranked candidate pool")
        selected.append(int(matches[0]))
    current_loss, current_energy, current_rank = evaluate_fit(
        pool[np.asarray(selected, dtype=int)]
    )
    initial_loss = float(current_loss)
    accepted_exchanges: list[dict] = []

    while True:
        selected_set = set(selected)
        remaining = [index for index in range(len(pool)) if index not in selected_set]
        best: tuple[float, float, int, int, float, int] | None = None
        for slot in range(len(selected)):
            for candidate_index in remaining:
                proposal = list(selected)
                proposal[slot] = int(candidate_index)
                loss, energy, rank = evaluate_fit(pool[np.asarray(proposal, dtype=int)])
                tie_score = -float(np.mean(pool_scores[np.asarray(proposal, dtype=int)]))
                record = (
                    float(loss),
                    tie_score,
                    int(slot),
                    int(candidate_index),
                    float(energy),
                    int(rank),
                )
                if best is None or record[:2] < best[:2]:
                    best = record
        if best is None:
            break
        best_loss, _, slot, candidate_index, best_energy, best_rank = best
        improvement_tolerance = (
            64.0 * np.finfo(float).eps * max(1.0, abs(float(current_loss)))
        )
        if not best_loss < float(current_loss) - improvement_tolerance:
            break
        removed = int(selected[slot])
        selected[slot] = int(candidate_index)
        accepted_exchanges.append(
            {
                "slot": int(slot),
                "removed_candidate_pool_index": int(pool_indices[removed]),
                "added_candidate_pool_index": int(pool_indices[candidate_index]),
                "loss_before": float(current_loss),
                "loss_after": float(best_loss),
            }
        )
        current_loss = float(best_loss)
        current_energy = float(best_energy)
        current_rank = int(best_rank)

    return pool[np.asarray(selected, dtype=int)].copy(), {
        "initial_subspace_loss": initial_loss,
        "final_subspace_loss": float(current_loss),
        "captured_signal_subspace_energy": float(current_energy),
        "numerical_rank": int(current_rank),
        "accepted_exchanges": accepted_exchanges,
        "greedy_trace": greedy_trace,
        "set_fit_mode": normalized_fit_mode,
        "source_candidate_pool_indices": [
            int(pool_indices[index]) for index in selected
        ],
    }


def _angles_to_unit_vectors(points: np.ndarray) -> np.ndarray:
    """Convert ``(theta, phi)`` degrees to Cartesian unit vectors."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    theta = np.deg2rad(arr[:, 0])
    phi = np.deg2rad(arr[:, 1])
    return np.column_stack((np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)))


def run_joint_candidate_pool_search(
    base_objective: MusicObjective,
    optimizer_fn: Callable[..., dict],
    k_targets: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    pop_size: int,
    max_iter: int,
    population_init: np.ndarray | None,
    rng: np.random.Generator,
    optimizer_kwargs: dict | None = None,
    stage_pop_sizes: list[int] | tuple[int, ...] | None = None,
    stage_max_iters: list[int] | tuple[int, ...] | None = None,
    local_refine_mode: str = "adaptive",
    local_refine_support_scale: float = 0.5,
    evidence_theta_step: float = 1.0,
    evidence_phi_step: float = 1.0,
    coherence_weight: float = 1.20,
    evidence_floor_db: float | None = 10.0,
    candidate_selection_mode: str = "coherence",
    include_all_evidence_candidates: bool = False,
    enable_optimizer_restarts: bool = True,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Run RIME as independent evidence-guided restarts and select jointly.

    This is the replacement path for the narrow sequential mechanism.  The
    optimizer is still the RIME kernel, but source extraction is no longer
    ``find one -> carve a fixed ellipse -> find the next``.  A shared global
    evidence grid proposes local maxima, each restart searches one proposal
    without an exclusion penalty, and the final K-set is selected by MUSIC
    either the legacy soft steering-coherence score or the parameter-free
    marginal signal-subspace gain.
    """
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    k_targets = int(k_targets)
    if k_targets <= 0:
        raise ValueError("k_targets must be positive")
    selection_mode = str(candidate_selection_mode).strip().lower()
    if selection_mode not in {
        "coherence",
        "subspace_gain",
        "subspace_fit",
        "optimal_wsf",
    }:
        raise ValueError(
            "candidate_selection_mode must be 'coherence', 'subspace_gain', "
            "'subspace_fit', or 'optimal_wsf'"
        )
    base_objective.reset_counts()
    base_objective.set_phase("global_evidence")
    max_candidates = max(8, 4 * k_targets)
    evidence_points, _ = _joint_grid_peak_candidates(
        base_objective,
        theta_bounds,
        phi_bounds,
        theta_step=float(evidence_theta_step),
        phi_step=float(evidence_phi_step),
        max_candidates=max_candidates,
    )
    if len(evidence_points) == 0:
        base_objective.set_phase("fallback")
        fallback = make_initial_population(population_init, max(1, pop_size), theta_bounds, phi_bounds, rng)
        evidence_points = fallback[: max(1, k_targets)].copy()

    restart_count = (
        min(len(evidence_points), max(k_targets, 2 * k_targets))
        if bool(enable_optimizer_restarts)
        else 0
    )
    search_pop = int(max(24, round(float(_stage_value(stage_pop_sizes, 0, pop_size)) * 0.5)))
    search_iter = int(max(6, round(float(_stage_value(stage_max_iters, 0, max_iter)) * 2.0)))
    archive_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    evidence_candidate_seed = (
        evidence_points.copy()
        if include_all_evidence_candidates
        else evidence_points[:restart_count].copy()
    )
    candidate_chunks: list[np.ndarray] = [evidence_candidate_seed]

    for restart, center in enumerate(evidence_points[:restart_count]):
        base_objective.set_phase("search_kernel")
        seeded = _joint_seed_population(
            center,
            search_pop,
            theta_bounds,
            phi_bounds,
            rng,
            theta_jitter=max(1.5, 2.0 * float(evidence_theta_step)),
            phi_jitter=max(1.0, 1.5 * float(evidence_phi_step)),
        )
        result = optimizer_fn(
            objective=base_objective,
            theta_bounds=theta_bounds,
            phi_bounds=phi_bounds,
            pop_size=search_pop,
            max_iter=search_iter,
            population_init=seeded,
            rng=rng,
            **optimizer_kwargs,
        )
        points = np.asarray(result.get("archive_points", result.get("population")), dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        points = clamp_positions(points, theta_bounds, phi_bounds)
        archive_chunks.append(points)
        label_chunks.append(np.full(len(points), restart, dtype=int))
        curve = np.asarray(result.get("curve", np.empty((0,), dtype=float)), dtype=float)
        if curve.size:
            curves.append(curve)
        if len(points):
            point_scores = base_objective.score_batch(points)
            best = np.argsort(point_scores)[::-1][: max(4, k_targets + 2)]
            candidate_chunks.append(points[best].copy())

    base_objective.set_phase("candidate_processing")
    candidate_pool = np.vstack(candidate_chunks) if candidate_chunks else np.empty((0, 2), dtype=float)
    if len(candidate_pool):
        # Independent refinement is allowed to converge to the same peak; the
        # continuous joint selector decides whether it is redundant.  There is
        # no angular exclusion against a committed source.
        pool_scores = base_objective.score_batch(candidate_pool)
        ranked_order = np.argsort(pool_scores)[::-1]
        if include_all_evidence_candidates:
            # The evidence points occupy the first rows of candidate_pool.
            # Preserve and refine each observation-derived local maximum once
            # before filling any remaining refinement slots by MUSIC score.
            refinement_order: list[int] = list(range(len(evidence_candidate_seed)))
            for index in ranked_order:
                if int(index) not in refinement_order:
                    refinement_order.append(int(index))
            refine_count = max(max(8, 4 * k_targets), len(evidence_candidate_seed))
            order = np.asarray(refinement_order[:refine_count], dtype=int)
        else:
            order = ranked_order[: max(8, 4 * k_targets)]
        refined = []
        for index in order:
            point = candidate_pool[int(index)]
            refined.append(
                refine_peak_by_mode(
                    base_objective,
                    point,
                    theta_bounds,
                    phi_bounds,
                    mode=local_refine_mode,
                    theta_scale=max(1.0, 2.0 * float(evidence_theta_step)),
                    phi_scale=max(0.75, 1.5 * float(evidence_phi_step)),
                    support_scale=float(local_refine_support_scale),
                )
            )
        if refined:
            candidate_pool = np.vstack([candidate_pool, np.asarray(refined, dtype=float)])

    candidate_pool = clamp_positions(candidate_pool, theta_bounds, phi_bounds)
    if len(candidate_pool):
        # Collapse only numerical duplicates produced by several restarts
        # converging to the same point.  The tolerance is deliberately tiny;
        # it is not a resolution or source-separation parameter.
        candidate_scores = base_objective.score_batch(candidate_pool)
        candidate_order = np.argsort(candidate_scores)[::-1]
        unique_pool: list[np.ndarray] = []
        for index in candidate_order:
            point = candidate_pool[int(index)]
            if any(angular_point_distance(point, previous) <= 1e-4 for previous in unique_pool):
                continue
            unique_pool.append(point.copy())
        candidate_pool = np.asarray(unique_pool, dtype=float) if unique_pool else np.empty((0, 2), dtype=float)
    if len(candidate_pool) > k_targets and evidence_floor_db is not None:
        # Do not let a very distant noise-grid point win only because it is
        # less coherent with the dominant peak.  This is an evidence floor in
        # MUSIC score, not an angular exclusion and not a source-spacing prior.
        pool_scores = base_objective.score_batch(candidate_pool)
        top_score = float(np.max(pool_scores))
        evidence_mask = pool_scores >= top_score - float(evidence_floor_db)
        if int(np.sum(evidence_mask)) >= k_targets:
            candidate_pool = candidate_pool[evidence_mask]
    selection_trace: list[dict] = []
    selection_details: dict = {}
    set_fit_mode = (
        "optimal_wsf" if selection_mode == "optimal_wsf" else "trace_subspace"
    )
    if selection_mode in {"subspace_fit", "optimal_wsf"} and k_targets <= 3:
        selected_arr, selection_details = _exact_subspace_fit_candidate_selection(
            candidate_pool,
            base_objective,
            k_targets,
            fit_mode=set_fit_mode,
        )
        selection_details["strategy"] = "exact_bounded_candidate_set"
    elif selection_mode in {"subspace_fit", "optimal_wsf"}:
        selected_arr, selection_details = _subspace_gain_exchange_candidate_selection(
            candidate_pool,
            base_objective,
            k_targets,
            fit_mode=set_fit_mode,
        )
        selection_details["strategy"] = "greedy_asg_with_strict_loss_decreasing_exchange"
    elif selection_mode == "subspace_gain":
        selected_arr, selection_trace = _adaptive_subspace_gain_candidate_selection(
            candidate_pool,
            base_objective,
            k_targets,
        )
    else:
        selected_arr = _adaptive_sequential_candidate_selection(
            candidate_pool,
            base_objective,
            k_targets,
            coherence_weight=float(coherence_weight),
        )
    if len(selected_arr) < k_targets:
        scores = base_objective.score_batch(candidate_pool) if len(candidate_pool) else np.empty(0, dtype=float)
        order = np.argsort(scores)[::-1]
        selected_list = [p.copy() for p in selected_arr]
        for index in order:
            candidate = candidate_pool[int(index)]
            if not any(angular_point_distance(candidate, previous) <= 1e-6 for previous in selected_list):
                selected_list.append(candidate.copy())
            if len(selected_list) >= k_targets:
                break
        selected_arr = np.asarray(selected_list, dtype=float)

    base_objective.set_phase("finalization")
    if len(selected_arr):
        order = np.argsort(selected_arr[:, 0])
        selected_arr = selected_arr[order]
    archive_points = np.vstack(archive_chunks) if archive_chunks else np.empty((0, 2), dtype=float)
    archive_scores = base_objective.score_batch(archive_points) if len(archive_points) else np.empty((0,), dtype=float)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty((0,), dtype=int)
    curve_out = np.concatenate(curves) if curves else np.empty((0,), dtype=float)
    preview = []
    candidate_pool_scores = np.empty((0,), dtype=float)
    if len(candidate_pool):
        candidate_pool_scores = base_objective.score_batch(candidate_pool)
        for index in np.argsort(candidate_pool_scores)[::-1][: min(20, len(candidate_pool))]:
            preview.append({
                "theta": float(candidate_pool[int(index), 0]),
                "phi": float(candidate_pool[int(index), 1]),
                "score_db": float(candidate_pool_scores[int(index)]),
            })
    return (
        (archive_points, labels),
        (archive_points, archive_scores),
        selected_arr[:, 0] if len(selected_arr) else np.empty(0, dtype=float),
        selected_arr[:, 1] if len(selected_arr) else np.empty(0, dtype=float),
        curve_out,
        {
            "mechanism": (
                "adaptive_subspace_fit_candidate_pool"
                if selection_mode == "subspace_fit"
                else "adaptive_optimal_wsf_candidate_pool"
                if selection_mode == "optimal_wsf"
                else "adaptive_subspace_gain_candidate_pool"
                if selection_mode == "subspace_gain"
                else "adaptive_sequential_candidate_pool"
            ),
            "stage_decisions": [{
                "mode": (
                    "adaptive_subspace_fit_candidate_pool"
                    if selection_mode == "subspace_fit"
                    else "adaptive_optimal_wsf_candidate_pool"
                    if selection_mode == "optimal_wsf"
                    else "adaptive_subspace_gain_candidate_pool"
                    if selection_mode == "subspace_gain"
                    else "adaptive_sequential_candidate_pool"
                ),
                "evidence_grid_points": int(len(_bounded_grid_axis(theta_bounds[0], theta_bounds[1], evidence_theta_step, periodic=True)) * len(_bounded_grid_axis(phi_bounds[0], phi_bounds[1], evidence_phi_step, periodic=False))),
                "candidate_pool_count": int(len(candidate_pool)),
                "restart_count": int(restart_count),
                "search_population": int(search_pop) if enable_optimizer_restarts else 0,
                "search_iterations": int(search_iter) if enable_optimizer_restarts else 0,
                "optimizer_restarts_enabled": bool(enable_optimizer_restarts),
                "hard_angular_exclusion": False,
                "candidate_selection_mode": selection_mode,
                "include_all_evidence_candidates": bool(include_all_evidence_candidates),
                "coherence_weight": (
                    None
                    if selection_mode in {"subspace_gain", "subspace_fit", "optimal_wsf"}
                    else float(coherence_weight)
                ),
                "evidence_floor_db": (
                    None if evidence_floor_db is None else float(evidence_floor_db)
                ),
                "selector_tunable_parameters": (
                    []
                    if selection_mode in {"subspace_gain", "subspace_fit", "optimal_wsf"}
                    else ["coherence_weight"]
                ),
                "selected_points": selected_arr.tolist(),
                "selection_trace": selection_trace,
                "selection_details": selection_details,
                "candidate_pool_points": candidate_pool.tolist(),
                "candidate_pool_scores_db": candidate_pool_scores.tolist(),
                "candidate_pool_preview": preview,
            }],
        },
    )


def run_sequential_peak_search(
    base_objective: MusicObjective,
    optimizer_fn: Callable[..., dict],
    k_targets: int,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
    pop_size: int,
    max_iter: int,
    population_init: np.ndarray | None,
    rng: np.random.Generator,
    optimizer_kwargs: dict | None = None,
    exclusion_theta: float = 14.0,
    exclusion_phi: float = 7.0,
    penalty_strength: float | None = None,
    use_repulsion: bool = True,
    use_guides: bool = True,
    use_local_refine: bool = True,
    local_refine_mode: str = "fixed",
    local_refine_support_scale: float = 0.52,
    evidence_support_scale: float = 0.52,
    evidence_consensus_weight: float = 0.55,
    stage_pop_sizes: list[int] | tuple[int, ...] | None = None,
    stage_max_iters: list[int] | tuple[int, ...] | None = None,
    stage_theta_seps: list[float] | tuple[float, ...] | None = None,
    stage_phi_seps: list[float] | tuple[float, ...] | None = None,
    refined_guides: bool = False,
    adaptive_exclusion: bool = False,
    confidence_gated_refine: bool = False,
    confidence_improve_ratio: float = 0.0,
    confidence_move_factor: float = 0.0,
    final_joint_polish: bool = False,
    final_polish_rounds: int = 0,
    candidate_consensus: bool = False,
    rescue_low_confidence: bool = False,
    guides_until_stage: int | None = None,
    local_refine_until_stage: int | None = None,
    consensus_until_stage: int | None = None,
    rescue_until_stage: int | None = None,
    guide_keep: int | None = None,
    max_injected_guides: int | None = None,
    archive_top_pool_scale: float = 2.0,
    archive_diversity_rescue: bool = False,
    isotropic_exclusion: bool = False,
    use_archive_evidence: bool = True,
    use_coarse_candidate: bool = True,
    local_refine_high_conf_only: bool = False,
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
    de_refine: bool = False,
    de_refine_pop_size: int = 28,
    de_refine_gens: int = 8,
    de_refine_F: float = 0.7,
    de_refine_CR: float = 0.8,
    de_refine_theta_radius: float = 3.0,
    de_refine_phi_radius: float = 2.0,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    adaptive_exclusion_mode = str(adaptive_exclusion_mode).strip().lower()
    if adaptive_exclusion_mode not in {"fixed", "local_spectrum"}:
        raise ValueError("adaptive_exclusion_mode must be 'fixed' or 'local_spectrum'")
    local_refine_mode = str(local_refine_mode).strip().lower()
    if local_refine_mode not in {"fixed", "adaptive", "none"}:
        raise ValueError("local_refine_mode must be 'fixed', 'adaptive', or 'none'")
    if confidence_gap_threshold is not None:
        confidence_gap_threshold_first = float(confidence_gap_threshold)
        confidence_gap_threshold_later = float(confidence_gap_threshold)
    if confidence_support_threshold is not None:
        confidence_support_threshold_first = float(confidence_support_threshold)
        confidence_support_threshold_later = float(confidence_support_threshold)
    base_objective.reset_counts()
    base_objective.set_phase("initialization")
    shared_init = make_initial_population(population_init, pop_size, theta_bounds, phi_bounds, rng)
    base_scale = float(np.median(base_objective.objective_batch(shared_init)))
    penalty = float(max(8.0, 1.5 * base_scale) if penalty_strength is None else penalty_strength)

    selected: list[np.ndarray] = []
    archive_chunks: list[np.ndarray] = []
    stage_label_chunks: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    stage_decisions: list[dict] = []
    stage_init = shared_init.copy()

    guide_evidence_cache = None
    coarse_evidence_cache = None
    if cache_global_evidence:
        base_objective.set_phase("global_evidence_cache")
        if use_coarse_candidate:
            # One hierarchical evidence grid is evaluated.  The 6x4 guide grid
            # is an exact nested subset of the 3x2 evidence grid.
            coarse_evidence_cache = build_global_evidence_cache(
                base_objective, theta_bounds, phi_bounds, theta_step=3.0, phi_step=2.0
            )
            if use_guides:
                pts = np.asarray(coarse_evidence_cache["points"], dtype=float)
                ti = np.rint((pts[:, 0] - float(theta_bounds[0])) / 3.0).astype(int)
                pi = np.rint((pts[:, 1] - float(phi_bounds[0])) / 2.0).astype(int)
                mask = (ti % 2 == 0) & (pi % 2 == 0)
                guide_evidence_cache = {
                    "points": pts[mask].copy(),
                    "objective": np.asarray(coarse_evidence_cache["objective"])[mask].copy(),
                    "score": np.asarray(coarse_evidence_cache["score"])[mask].copy(),
                    "theta_step": 6.0,
                    "phi_step": 4.0,
                }
        elif use_guides:
            guide_evidence_cache = build_global_evidence_cache(
                base_objective, theta_bounds, phi_bounds, theta_step=6.0, phi_step=4.0
            )

    for stage in range(k_targets):
        stage_pop_size = int(_stage_value(stage_pop_sizes, stage, pop_size))
        stage_max_iter = int(_stage_value(stage_max_iters, stage, max_iter))
        theta_sep_stage = float(_stage_value(stage_theta_seps, stage, exclusion_theta if not adaptive_exclusion else max(8.0, exclusion_theta - 2.5 * stage)))
        phi_sep_stage = float(_stage_value(stage_phi_seps, stage, exclusion_phi if not adaptive_exclusion else max(4.0, exclusion_phi - 1.25 * stage)))
        if isotropic_exclusion:
            # Preserve the original ellipse area while replacing the anisotropic
            # theta/phi scales by a single circular scale in angular coordinates.
            radius = float(np.sqrt(max(theta_sep_stage * phi_sep_stage, 1e-12)))
            theta_sep_stage = radius
            phi_sep_stage = radius
        theta_sep_eff = max(1e-6, theta_sep_stage)
        phi_sep_eff = max(1e-6, phi_sep_stage)
        stage_use_guides = bool(use_guides and (guides_until_stage is None or stage < int(guides_until_stage)))
        stage_use_local_refine = bool(
            use_local_refine and local_refine_mode != "none"
            and (local_refine_until_stage is None or stage < int(local_refine_until_stage))
        )
        stage_use_candidate_consensus = bool(candidate_consensus and (consensus_until_stage is None or stage < int(consensus_until_stage)))
        stage_use_rescue = bool(rescue_low_confidence and (rescue_until_stage is None or stage < int(rescue_until_stage)))
        stage_guide_keep = int(max(2, guide_keep if guide_keep is not None else max(4, k_targets + 1)))
        stage_max_inject = int(max(4, max_injected_guides if max_injected_guides is not None else max(10, stage_pop_size // 5)))
        stage_archive_top_pool = max(24, int(round(float(archive_top_pool_scale) * stage_pop_size)))
        stage_init = make_initial_population(stage_init, stage_pop_size, theta_bounds, phi_bounds, rng)
        if use_repulsion:
            stage_init = avoid_selected_peaks(
                stage_init, rng, theta_bounds, phi_bounds, np.asarray(selected, dtype=float) if selected else None,
                theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
            )
        excluded = np.asarray(selected, dtype=float) if (use_repulsion and selected) else None
        theta_exclusion_eff = theta_sep_eff
        phi_exclusion_eff = phi_sep_eff
        adaptive_meta: dict = {
            "mode": adaptive_exclusion_mode,
            "enabled": False,
            "theta_scale": 1.0,
            "phi_scale": 1.0,
        }
        if use_repulsion and selected and adaptive_exclusion_mode == "local_spectrum":
            base_objective.set_phase("adaptive_exclusion")
            scale_rows = []
            for center in selected:
                theta_local, phi_local, local_meta = local_spectrum_exclusion_scales(
                    base_objective,
                    center,
                    theta_bounds,
                    phi_bounds,
                    theta_sep_eff,
                    phi_sep_eff,
                    relative_drop=adaptive_exclusion_relative_drop,
                    min_scale=adaptive_exclusion_min_scale,
                    max_scale=adaptive_exclusion_max_scale,
                )
                scale_rows.append((theta_local, phi_local, local_meta))
            theta_exclusion_eff = max(value[0] for value in scale_rows)
            phi_exclusion_eff = max(value[1] for value in scale_rows)
            adaptive_meta = {
                "mode": adaptive_exclusion_mode,
                "enabled": True,
                "theta_scale": float(theta_exclusion_eff / max(theta_sep_eff, 1e-9)),
                "phi_scale": float(phi_exclusion_eff / max(phi_sep_eff, 1e-9)),
                "per_center": [value[2] for value in scale_rows],
            }
        base_objective.set_phase("guide_candidates")
        if stage_use_guides and cache_global_evidence and guide_evidence_cache is not None and not refined_guides:
            guides = select_cached_grid_candidates(
                guide_evidence_cache, excluded,
                theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
                n_keep=max(4, k_targets + 1),
            )
        else:
            guides = propose_refined_grid_candidates(
                base_objective, theta_bounds, phi_bounds, excluded,
                theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
                n_keep=stage_guide_keep,
            ) if (stage_use_guides and refined_guides) else (
                propose_grid_candidates(
                    base_objective, theta_bounds, phi_bounds, excluded,
                    theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
                    theta_step=6.0, phi_step=4.0, n_keep=max(4, k_targets + 1),
                ) if stage_use_guides else np.empty((0, 2), dtype=float)
            )
        if stage_use_guides:
            stage_init = inject_guided_seeds(
                stage_init, guides, rng, theta_bounds, phi_bounds,
                theta_scale=max(1.2, 0.18 * theta_sep_eff),
                phi_scale=max(0.8, 0.18 * phi_sep_eff),
                max_inject=stage_max_inject,
            )
        search_objective = PenalizedMusicObjective(
            base_objective,
            excluded if use_repulsion else None,
            penalty_strength=penalty,
            theta_sigma=theta_exclusion_eff,
            phi_sigma=phi_exclusion_eff,
        ) if use_repulsion else base_objective
        base_objective.set_phase("search_kernel")
        result = optimizer_fn(
            objective=search_objective, theta_bounds=theta_bounds, phi_bounds=phi_bounds,
            pop_size=stage_pop_size, max_iter=stage_max_iter, population_init=stage_init, rng=rng, **optimizer_kwargs,
        )
        base_objective.set_phase("candidate_processing")
        final_population = np.asarray(result.get('population'), dtype=float)
        if final_population.ndim == 1:
            final_population = final_population[None, :]
        final_population = clamp_positions(final_population, theta_bounds, phi_bounds)
        optimizer_archive = np.asarray(result.get('archive_points', final_population), dtype=float)
        if optimizer_archive.ndim == 1:
            optimizer_archive = optimizer_archive[None, :]
        optimizer_archive = clamp_positions(optimizer_archive, theta_bounds, phi_bounds)
        # In the no-archive ablation, candidate selection and support are computed
        # from the terminal population only; the optimizer update budget is unchanged.
        stage_points = optimizer_archive if use_archive_evidence else final_population
        excluded = np.asarray(selected, dtype=float) if selected else None
        candidate_archive = select_candidate_from_points(
            stage_points, base_objective, excluded if use_repulsion else None,
            theta_sep=0.80 * theta_sep_eff, phi_sep=0.80 * phi_sep_eff,
        )
        # The coarse candidate is read from the shared 3x2 evidence cache in
        # the publication path; no second coarse scan is required.
        coarse_triggered = False
        if use_coarse_candidate and cache_global_evidence and coarse_evidence_cache is not None:
            candidate_coarse = select_cached_global_candidate(
                coarse_evidence_cache, excluded if use_repulsion else None,
                theta_sep=0.9 * theta_sep_eff, phi_sep=0.9 * phi_sep_eff,
            )
        else:
            initial_coarse_enabled = bool(use_coarse_candidate and not evidence_adaptive_controller)
            candidate_coarse = coarse_global_candidate(
                base_objective, theta_bounds, phi_bounds, excluded if use_repulsion else None,
                theta_sep=0.9 * theta_sep_eff, phi_sep=0.9 * phi_sep_eff,
            ) if initial_coarse_enabled else None
        guide_candidates = list(np.asarray(guides, dtype=float)[: min(len(guides), 4)])
        archive_candidates = collect_diverse_archive_candidates(
            stage_points, base_objective, excluded if use_repulsion else None,
            theta_sep=0.80 * theta_sep_eff, phi_sep=0.80 * phi_sep_eff, max_keep=6, top_pool=stage_archive_top_pool,
        )
        candidate_bank = []
        for cand in [candidate_archive, candidate_coarse, *guide_candidates, *list(np.asarray(archive_candidates, dtype=float))]:
            if cand is None:
                continue
            cand = np.asarray(cand, dtype=float).reshape(2)
            if not is_far_from_centers(cand, excluded if use_repulsion else None, 0.80 * theta_sep_eff, 0.80 * phi_sep_eff, theta_hard_factor=0.50):
                continue
            candidate_bank.append(cand.copy())
        if candidate_bank:
            candidate_bank = unique_points_keep_best(np.asarray(candidate_bank, dtype=float), base_objective.score_batch(np.asarray(candidate_bank, dtype=float)))[0]
        else:
            candidate_bank = np.empty((0, 2), dtype=float)

        evidence_points = stage_points
        evidence_scores = base_objective.score_batch(stage_points) if len(stage_points) else np.empty((0,), dtype=float)
        confidence_candidates = candidate_bank
        if len(confidence_candidates) == 0:
            fallback_conf = candidate_coarse if candidate_coarse is not None else candidate_archive
            if fallback_conf is None and len(stage_points):
                fallback_conf = stage_points[int(np.argmin(base_objective.objective_batch(stage_points)))]
            confidence_candidates = np.asarray(fallback_conf, dtype=float).reshape(1, 2)
        low_conf, conf_meta = stage_low_confidence(
            confidence_candidates,
            base_objective, evidence_points, evidence_scores,
            theta_sep=theta_sep_eff, phi_sep=phi_sep_eff,
            min_gap_ratio=(confidence_gap_threshold_first if stage == 0 else confidence_gap_threshold_later),
            min_support_ratio=(confidence_support_threshold_first if stage == 0 else confidence_support_threshold_later),
            support_scale=evidence_support_scale,
        )
        # Observation-driven coarse rescue: only ambiguous stages pay for a
        # global coarse probe. Recompute confidence after adding that evidence.
        if (
            evidence_adaptive_controller
            and adaptive_coarse_on_low_confidence
            and low_conf
            and candidate_coarse is None
        ):
            base_objective.set_phase("adaptive_coarse_rescue")
            candidate_coarse = coarse_global_candidate(
                base_objective, theta_bounds, phi_bounds, excluded if use_repulsion else None,
                theta_sep=0.9 * theta_sep_eff, phi_sep=0.9 * phi_sep_eff,
            )
            coarse_triggered = candidate_coarse is not None
            if candidate_coarse is not None:
                candidate_bank = np.vstack([candidate_bank, np.asarray(candidate_coarse, dtype=float).reshape(1, 2)]) if len(candidate_bank) else np.asarray(candidate_coarse, dtype=float).reshape(1, 2)
                candidate_bank = unique_points_keep_best(
                    candidate_bank, base_objective.score_batch(candidate_bank)
                )[0]
                confidence_candidates = candidate_bank
                low_conf, conf_meta = stage_low_confidence(
                    confidence_candidates,
                    base_objective, evidence_points, evidence_scores,
                    theta_sep=theta_sep_eff, phi_sep=phi_sep_eff,
                    min_gap_ratio=(confidence_gap_threshold_first if stage == 0 else confidence_gap_threshold_later),
                    min_support_ratio=(confidence_support_threshold_first if stage == 0 else confidence_support_threshold_later),
                    support_scale=evidence_support_scale,
                )

        allow_local_refine = bool(
            stage_use_local_refine
            and not (local_refine_high_conf_only and low_conf)
        )
        differential_rescue_triggered = False
        extra_budget_triggered = False

        if stage_use_rescue and low_conf:
            extra_budget_triggered = True
            if simplified_rescue_budget:
                # Low confidence triggers one independent RIME re-search with
                # the same stage population and iteration budget.
                rescue_pop = int(stage_pop_size)
                rescue_iter = int(stage_max_iter)
            else:
                rescue_pop = max(stage_pop_size + 12, int(round(stage_pop_size * 1.18)))
                rescue_iter = stage_max_iter + (4 if stage == 0 else 3)
            rescue_init = resample_avoiding_centers(
                rng, rescue_pop, theta_bounds, phi_bounds, excluded if use_repulsion else None,
                theta_sep=theta_sep_eff, phi_sep=phi_sep_eff,
            )
            if len(candidate_bank):
                rescue_seeds = candidate_bank
            else:
                rescue_seed = candidate_coarse if candidate_coarse is not None else candidate_archive
                if rescue_seed is None:
                    rescue_seed = stage_points[int(np.argmin(base_objective.objective_batch(stage_points)))]
                rescue_seeds = np.asarray(rescue_seed, dtype=float).reshape(1, 2)
            rescue_init = inject_guided_seeds(
                rescue_init, rescue_seeds, rng, theta_bounds, phi_bounds,
                theta_scale=max(1.2, 0.18 * theta_sep_eff), phi_scale=max(0.8, 0.18 * phi_sep_eff),
                max_inject=stage_max_inject,
            )
            rescue_optimizer_kwargs = dict(optimizer_kwargs)
            if evidence_adaptive_controller and adaptive_differential_rescue:
                rescue_optimizer_kwargs["use_differential_move"] = True
                # Keep the controller interpretable: differential motion is a
                # rescue action, while restart remains disabled in publication use.
                rescue_optimizer_kwargs["use_adaptive_restart"] = False
                differential_rescue_triggered = True
            rescue_result = optimizer_fn(
                objective=search_objective, theta_bounds=theta_bounds, phi_bounds=phi_bounds,
                pop_size=rescue_pop, max_iter=rescue_iter, population_init=rescue_init, rng=rng, **rescue_optimizer_kwargs,
            )
            rescue_points = np.asarray(rescue_result.get('archive_points', rescue_result.get('population')), dtype=float)
            if rescue_points.ndim == 1:
                rescue_points = rescue_points[None, :]
            rescue_points = clamp_positions(rescue_points, theta_bounds, phi_bounds)
            stage_points = np.vstack([stage_points, rescue_points])
            evidence_points = stage_points
            evidence_scores = base_objective.score_batch(stage_points) if len(stage_points) else np.empty((0,), dtype=float)
            rescue_candidates = collect_diverse_archive_candidates(
                rescue_points, base_objective, excluded if use_repulsion else None,
                theta_sep=0.80 * theta_sep_eff, phi_sep=0.80 * phi_sep_eff, max_keep=6, top_pool=max(64, rescue_pop),
            )
            if len(rescue_candidates):
                stacked = np.vstack([candidate_bank, rescue_candidates]) if len(candidate_bank) else rescue_candidates
                candidate_bank = unique_points_keep_best(stacked, base_objective.score_batch(stacked))[0]
            # Re-evaluate ambiguity after rescue; a successfully resolved stage can
            # proceed to local polishing, otherwise the conservative commit path is kept.
            if len(candidate_bank):
                low_conf, conf_meta = stage_low_confidence(
                    candidate_bank,
                    base_objective, evidence_points, evidence_scores,
                    theta_sep=theta_sep_eff, phi_sep=phi_sep_eff,
                    min_gap_ratio=(confidence_gap_threshold_first if stage == 0 else confidence_gap_threshold_later),
                    min_support_ratio=(confidence_support_threshold_first if stage == 0 else confidence_support_threshold_later),
                    support_scale=evidence_support_scale,
                )
                allow_local_refine = bool(
                    stage_use_local_refine
                    and not (local_refine_high_conf_only and low_conf)
                )


        anchor = None
        if anchor is None and stage_use_candidate_consensus and len(candidate_bank) and low_conf:
            anchor = choose_consensus_candidate(
                candidate_bank, base_objective, evidence_points, evidence_scores,
                excluded if use_repulsion else None, theta_bounds, phi_bounds,
                theta_sep=0.80 * theta_sep_eff, phi_sep=0.80 * phi_sep_eff,
                rng=rng,
                use_local_refine=allow_local_refine, local_refine_mode=local_refine_mode,
                local_refine_support_scale=local_refine_support_scale,
                evidence_support_scale=evidence_support_scale,
                evidence_consensus_weight=evidence_consensus_weight,
                confidence_gated_refine=confidence_gated_refine,
                improve_ratio=confidence_improve_ratio + (0.002 if low_conf else 0.0),
                move_factor=max(confidence_move_factor, 0.26),
                de_refine=de_refine,
                de_refine_pop_size=de_refine_pop_size,
                de_refine_gens=de_refine_gens,
                de_refine_F=de_refine_F,
                de_refine_CR=de_refine_CR,
                de_refine_theta_radius=de_refine_theta_radius,
                de_refine_phi_radius=de_refine_phi_radius,
            )
        if anchor is None:
            anchor_candidates = [candidate_archive, candidate_coarse, *guide_candidates]
            anchor = choose_best_candidate(
                anchor_candidates, base_objective, excluded if use_repulsion else None,
                theta_sep=0.80 * theta_sep_eff, phi_sep=0.80 * phi_sep_eff,
            )
        if anchor is None:
            fallback_anchor = candidate_coarse if candidate_coarse is not None else candidate_archive
            if fallback_anchor is None:
                fallback_anchor = stage_points[int(np.argmin(base_objective.objective_batch(stage_points)))]
            anchor = np.asarray(fallback_anchor, dtype=float).copy()

        if allow_local_refine and not (stage_use_candidate_consensus and len(candidate_bank) and low_conf):
            base_objective.set_phase("local_refinement")
            if de_refine:
                refined_anchor = refine_peak_de_local(
                    base_objective,
                    anchor,
                    theta_bounds,
                    phi_bounds,
                    rng=rng,
                    pop_size=de_refine_pop_size,
                    generations=de_refine_gens,
                    F=de_refine_F,
                    CR=de_refine_CR,
                    theta_radius=de_refine_theta_radius,
                    phi_radius=de_refine_phi_radius,
                )
            else:
                refined_anchor = refine_peak_by_mode(
                    base_objective, anchor, theta_bounds, phi_bounds,
                    mode=local_refine_mode, theta_scale=theta_sep_eff, phi_scale=phi_sep_eff,
                    support_scale=local_refine_support_scale,
                )
            if confidence_gated_refine:
                extra_ratio = 0.003 if low_conf else 0.0
                extra_move = 0.10 if low_conf else 0.0
                refined = confidence_gate_refine(
                    base_objective, anchor, refined_anchor, theta_sep_eff, phi_sep_eff,
                    improve_ratio=confidence_improve_ratio + extra_ratio,
                    move_factor=max(0.0, confidence_move_factor - extra_move),
                )
            else:
                refined = refined_anchor
            if use_repulsion and (not is_far_from_centers(refined, selected, 0.80 * theta_sep_eff, 0.80 * phi_sep_eff)):
                fallback = anchor.copy()
                if is_far_from_centers(fallback, selected, 0.80 * theta_sep_eff, 0.80 * phi_sep_eff):
                    refined = fallback
        else:
            refined = anchor.copy()
        base_objective.set_phase("candidate_processing")
        selected.append(np.asarray(refined, dtype=float))
        archive_chunks.append(stage_points)
        stage_label_chunks.append(np.full(len(stage_points), stage, dtype=int))
        stage_decisions.append(
            {
                "stage": int(stage),
                "adaptive_exclusion_mode": adaptive_exclusion_mode,
                "theta_exclusion": float(theta_exclusion_eff),
                "phi_exclusion": float(phi_exclusion_eff),
                "low_confidence": bool(low_conf),
                "candidate_count": int(len(candidate_bank)),
                "evidence_adaptive_controller": bool(evidence_adaptive_controller),
                "coarse_rescue_triggered": bool(coarse_triggered),
                "differential_rescue_triggered": bool(differential_rescue_triggered),
                "extra_budget_triggered": bool(extra_budget_triggered),
                "confidence": {
                    key: float(value)
                    for key, value in conf_meta.items()
                    if isinstance(value, (int, float, np.floating, np.integer))
                },
                "adaptive_exclusion": adaptive_meta,
            }
        )
        curve = np.asarray(result.get('curve', np.empty((0,), dtype=float)), dtype=float)
        if curve.size:
            curves.append(curve)

        if use_repulsion:
            stage_init = avoid_selected_peaks(
                final_population, rng, theta_bounds, phi_bounds, np.asarray(selected, dtype=float),
                theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
            )
            reseed_n = max(4, stage_pop_size // 6)
            stage_init[:reseed_n] = resample_avoiding_centers(
                rng, reseed_n, theta_bounds, phi_bounds, np.asarray(selected, dtype=float),
                theta_sep=0.90 * theta_sep_eff, phi_sep=0.90 * phi_sep_eff,
            )
        else:
            stage_init = clamp_positions(final_population, theta_bounds, phi_bounds)

    base_objective.set_phase("finalization")
    selected_arr = np.asarray(selected, dtype=float)
    archive_points = np.vstack(archive_chunks) if archive_chunks else np.empty((0, 2), dtype=float)
    archive_scores = base_objective.score_batch(archive_points) if len(archive_points) else np.empty((0,), dtype=float)
    if archive_diversity_rescue and len(selected_arr) >= 2 and len(archive_points):
        dmat = pairwise_angular_distance(selected_arr)
        np.fill_diagonal(dmat, np.inf)
        theta_final = float(_stage_value(stage_theta_seps, max(0, k_targets - 1), exclusion_theta))
        phi_final = float(_stage_value(stage_phi_seps, max(0, k_targets - 1), exclusion_phi))
        duplicate_sep = float(np.hypot(theta_final, phi_final))
        if float(np.min(dmat)) < duplicate_sep:
            archive_peaks = extract_peaks_from_points(
                archive_points,
                archive_scores,
                k_targets,
                dbscan_eps=3.0,
                dbscan_min_samples=4,
                k_neighbors=8,
            )
            archive_selected = np.column_stack([archive_peaks.theta_est, archive_peaks.phi_est])
            valid = np.all(np.isfinite(archive_selected), axis=1)
            archive_selected = archive_selected[valid]
            if len(archive_selected) == len(selected_arr):
                archive_dmat = pairwise_angular_distance(archive_selected)
                np.fill_diagonal(archive_dmat, np.inf)
                if float(np.min(archive_dmat)) > float(np.min(dmat)) + 1e-6:
                    selected_arr = archive_selected
    if final_joint_polish and len(selected_arr):
        base_objective.set_phase("local_refinement")
        selected_arr = coordinated_peak_polish(
            base_objective, selected_arr, theta_bounds, phi_bounds,
            theta_sep=float(_stage_value(stage_theta_seps, max(0, k_targets - 1), exclusion_theta)),
            phi_sep=float(_stage_value(stage_phi_seps, max(0, k_targets - 1), exclusion_phi)),
            rounds=max(1, int(final_polish_rounds)),
            rng=rng,
            de_refine=de_refine,
            de_refine_pop_size=de_refine_pop_size,
            de_refine_gens=de_refine_gens,
            de_refine_F=de_refine_F,
            de_refine_CR=de_refine_CR,
            de_refine_theta_radius=de_refine_theta_radius,
            de_refine_phi_radius=de_refine_phi_radius,
        )
    base_objective.set_phase("finalization")
    order = np.argsort(selected_arr[:, 0])
    theta_est = selected_arr[order, 0]
    phi_est = selected_arr[order, 1]

    stage_labels = np.concatenate(stage_label_chunks) if stage_label_chunks else np.empty((0,), dtype=int)
    curve_out = np.concatenate(curves) if curves else np.empty((0,), dtype=float)
    return (
        (archive_points, stage_labels),
        (archive_points, archive_scores),
        theta_est,
        phi_est,
        curve_out,
        {"stage_decisions": stage_decisions},
    )
