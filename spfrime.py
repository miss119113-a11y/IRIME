"""Self-contained reference implementation of SPFRIME.

This file mirrors the method described in the manuscript:
1) rank-truncated recursive projected-subspace initialization;
2) recursion-aware guarded subgrid fusion with exact rescoring;
3) two cyclic conditional-DML RIME sweeps;
4) bounded joint-DML L-BFGS-B refinement.

Angles are in degrees. The source count is assumed known.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

_EPS = np.finfo(float).eps


def _bounds(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("angle grid must contain at least two finite values")
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        raise ValueError("invalid angular bounds")
    return lo, hi


def _axis(lo: float, hi: float, step: float, periodic: bool) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    x = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    if periodic:
        x = x[x < hi - 32 * _EPS]
    else:
        x = x[x <= hi + 32 * _EPS]
        if x.size == 0 or not np.isclose(x[-1], hi):
            x = np.r_[x, hi]
    return x


def _wrap_theta(theta: np.ndarray | float, lo: float, hi: float):
    span = hi - lo
    return lo + np.mod(np.asarray(theta) - lo, span)


def steering(points: np.ndarray, array_pos: np.ndarray, wavelength: float) -> np.ndarray:
    """Return M x N steering matrix for [azimuth,elevation] points in degrees."""
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    th = np.deg2rad(p[:, 0])
    ph = np.deg2rad(p[:, 1])
    u = np.column_stack((np.cos(ph) * np.cos(th),
                         np.cos(ph) * np.sin(th),
                         np.sin(ph)))
    phase = -2.0 * np.pi / float(wavelength) * (np.asarray(array_pos, float) @ u.T)
    return np.exp(1j * phase)


def _orthogonal_complement(A: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    if A.size == 0:
        return np.eye(m, dtype=complex), 0
    if A.ndim == 1:
        A = A[:, None]
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.eye(m, dtype=complex), 0
    tol = _EPS * max(A.shape) * s[0]
    r = int(np.sum(s > tol))
    U = U[:, :r]
    Q = np.eye(m, dtype=complex) - U @ U.conj().T
    return 0.5 * (Q + Q.conj().T), r


def _projected_basis(Us: np.ndarray, Q: np.ndarray, rank: int) -> np.ndarray:
    U, s, _ = np.linalg.svd(Q @ Us, full_matrices=False)
    if s.size == 0:
        return np.empty((Us.shape[0], 0), dtype=complex)
    tol = _EPS * max((Q @ Us).shape) * s[0]
    r = min(int(rank), int(np.sum(s > tol)))
    return U[:, :r]


def _score_points(points: np.ndarray, basis: np.ndarray, Q: np.ndarray,
                  array_pos: np.ndarray, wavelength: float) -> np.ndarray:
    A = steering(points, array_pos, wavelength)
    QA = Q @ A
    den = np.sum(np.abs(QA) ** 2, axis=0).real
    num = np.sum(np.abs(basis.conj().T @ QA) ** 2, axis=0).real
    return np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 128 * _EPS)


def _quadratic_candidate(score_map: np.ndarray, it: int, ip: int,
                         theta_axis: np.ndarray, phi_axis: np.ndarray) -> np.ndarray | None:
    """Fit a local 2-D quadratic to a 3x3 neighbourhood and return its stationary point."""
    nt, np_ = score_map.shape
    if ip <= 0 or ip >= np_ - 1:
        return None
    # azimuth is periodic; elevation is not.
    tids = [(it - 1) % nt, it % nt, (it + 1) % nt]
    center_t = theta_axis[it]
    span_t = (theta_axis[-1] - theta_axis[0]) + (theta_axis[1] - theta_axis[0])
    rows, z = [], []
    for ti in tids:
        dt = float(theta_axis[ti] - center_t)
        if dt > span_t / 2:
            dt -= span_t
        if dt < -span_t / 2:
            dt += span_t
        for pj in (ip - 1, ip, ip + 1):
            dp = float(phi_axis[pj] - phi_axis[ip])
            rows.append([1.0, dt, dp, 0.5 * dt * dt, dt * dp, 0.5 * dp * dp])
            z.append(float(score_map[ti, pj]))
    coef, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(z), rcond=None)
    g = coef[1:3]
    H = np.array([[coef[3], coef[4]], [coef[4], coef[5]]], dtype=float)
    eig = np.linalg.eigvalsh(H)
    if not np.all(eig < -1e-12):
        return None
    try:
        delta = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return None
    dt_lim = float(theta_axis[1] - theta_axis[0])
    dp_lim = float(phi_axis[1] - phi_axis[0])
    if abs(delta[0]) > dt_lim or abs(delta[1]) > dp_lim:
        return None
    return np.array([center_t + delta[0], phi_axis[ip] + delta[1]], dtype=float)


def _joint_gain(D: np.ndarray, R: np.ndarray, array_pos: np.ndarray, wavelength: float) -> float:
    A = steering(D, array_pos, wavelength)
    Q, _ = _orthogonal_complement(A, A.shape[0])
    P = np.eye(A.shape[0], dtype=complex) - Q
    return float(np.real(np.trace(P @ R)))


@dataclass
class ConditionalDMLObjective:
    R: np.ndarray
    Q: np.ndarray
    array_pos: np.ndarray
    wavelength: float
    trace_R: float
    eval_count: int = 0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        A = steering(p, self.array_pos, self.wavelength)
        QA = self.Q @ A
        den = np.sum(np.abs(QA) ** 2, axis=0).real
        num = np.sum(np.conj(QA) * (self.R @ QA), axis=0).real
        gain = np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 128 * _EPS)
        self.eval_count += p.shape[0]
        return 1.0 - gain / max(self.trace_R, _EPS)


def _seed_population(center: np.ndarray, n: int, theta_bounds, phi_bounds, rng) -> np.ndarray:
    t0, t1 = theta_bounds
    p0, p1 = phi_bounds
    n0 = max(1, n // 3)
    base = np.column_stack((rng.uniform(t0, t1, n0), rng.uniform(p0, p1, n0)))
    opp = np.column_stack((t0 + t1 - base[:, 0], p0 + p1 - base[:, 1]))
    mid = np.array([(t0 + t1) / 2, (p0 + p1) / 2])
    qopp = mid + rng.random(base.shape) * (opp - mid)
    pop = np.vstack((center[None, :], base, opp, qopp))[:n]
    if pop.shape[0] < n:
        extra = np.column_stack((rng.uniform(t0, t1, n - len(pop)), rng.uniform(p0, p1, n - len(pop))))
        pop = np.vstack((pop, extra))
    pop[:, 0] = _wrap_theta(pop[:, 0], t0, t1)
    pop[:, 1] = np.clip(pop[:, 1], p0, p1)
    return pop


def _rime_minimize(fun: Callable[[np.ndarray], np.ndarray], center: np.ndarray,
                   theta_bounds, phi_bounds, budget: int, rng) -> tuple[np.ndarray, float]:
    """Finite-budget RIME search used by the paper implementation."""
    TR = max(2, int(math.ceil(math.log2(12))))  # M=12 paper array -> 4 iterations
    n = max(4, int(budget // (TR + 3)))
    pop = _seed_population(np.asarray(center, float), n, theta_bounds, phi_bounds, rng)
    f = np.asarray(fun(pop), float)
    d = np.array([theta_bounds[1] - theta_bounds[0], phi_bounds[1] - phi_bounds[0]], float)
    W = 5.0
    for t in range(1, TR + 1):
        best = pop[int(np.argmin(f))].copy()
        xi = rng.random()
        r = ((2 * xi - 1)
             * math.cos(math.pi * t / max(1.0, TR / 10.0))
             * (1.0 - round(t * W / TR) / W))
        update_prob = math.sqrt(t / TR)
        proposal = pop.copy()
        explore_mask = rng.random(pop.shape) < update_prob
        proposal += explore_mask * (r * 0.08 * d / (t + 1.0)) * rng.normal(size=pop.shape)
        # Fitness-dependent local refinement around the current best.
        fmin, fmax = float(np.min(f)), float(np.max(f))
        quality = (fmax - f) / max(fmax - fmin, 1e-12)
        local_mask = rng.random(pop.shape) < quality[:, None]
        proposal = np.where(local_mask,
                            best + 0.018 * d / (t + 1.0) * rng.normal(size=pop.shape),
                            proposal)
        proposal[:, 0] = _wrap_theta(proposal[:, 0], *theta_bounds)
        proposal[:, 1] = np.clip(proposal[:, 1], *phi_bounds)
        fp = np.asarray(fun(proposal), float)
        accept = fp <= f
        pop[accept], f[accept] = proposal[accept], fp[accept]
        # Keep the current solution explicitly available.
        fc = float(fun(np.asarray(center, float))[0])
        worst = int(np.argmax(f))
        if fc < f[worst]:
            pop[worst], f[worst] = center, fc
    j = int(np.argmin(f))
    return pop[j].copy(), float(f[j])


def trap_dml_rime(X: np.ndarray, array_pos: np.ndarray, wavelength: float,
                  theta_grid: np.ndarray, phi_grid: np.ndarray, k_sources: int,
                  *, evidence_step: float = 2.5, conditional_budget: int = 896,
                  seed: int | None = None, return_diagnostics: bool = False):
    """Estimate K two-dimensional DOAs with SPFRIME.

    Parameters
    ----------
    X : complex ndarray, shape (M,L)
        Array observations.
    array_pos : ndarray, shape (M,3)
        Sensor coordinates in the same length unit as ``wavelength``.
    wavelength : float
        Signal wavelength.
    theta_grid, phi_grid : array-like
        Arrays used only to define the admissible azimuth/elevation ranges.
    k_sources : int
        Known source count.
    evidence_step : float
        Coarse projected-evidence step h in degrees. The final bounded joint
        refinement uses the same h as its local half-width.
    conditional_budget : int
        Conditional-DML RIME evaluation budget per coordinate update.
    """
    X = np.asarray(X, complex)
    pos = np.asarray(array_pos, float)
    M, L = X.shape
    if pos.shape != (M, 3):
        raise ValueError("array_pos must have shape (M,3)")
    K = int(k_sources)
    if K < 1 or K >= M:
        raise ValueError("k_sources must satisfy 1 <= K < M")
    t_bounds, p_bounds = _bounds(theta_grid), _bounds(phi_grid)
    ta = _axis(*t_bounds, evidence_step, periodic=True)
    pa = _axis(*p_bounds, evidence_step, periodic=False)
    T, P = np.meshgrid(ta, pa, indexing="ij")
    grid = np.column_stack((T.ravel(), P.ravel()))

    R = (X @ X.conj().T) / float(L)
    evals, U = np.linalg.eigh(R)
    Us = U[:, np.argsort(evals)[::-1][:K]]
    trace_R = float(np.real(np.trace(R)))
    selected: list[np.ndarray] = []
    init_trace = []
    fusion_active = K >= 3  # automatic consequence of recursion depth

    for q in range(1, K + 1):
        Aprev = steering(np.asarray(selected), pos, wavelength) if selected else np.empty((M, 0), complex)
        Q, _ = _orthogonal_complement(Aprev, M)
        basis = _projected_basis(Us, Q, K - q + 1)
        scores = _score_points(grid, basis, Q, pos, wavelength)
        smap = scores.reshape(len(ta), len(pa))
        # Local maxima are identified first; the globally strongest retained peak is selected.
        loc = smap == maximum_filter(smap, size=3, mode=("wrap", "nearest"))
        masked = np.where(loc, smap, -np.inf)
        it, ip = np.unravel_index(int(np.argmax(masked)), masked.shape)
        coarse = np.array([ta[it], pa[ip]], float)
        chosen = coarse.copy()
        accepted = False
        if fusion_active:
            cand = _quadratic_candidate(smap, it, ip, ta, pa)
            if cand is not None:
                cand[0] = float(_wrap_theta(cand[0], *t_bounds))
                cand[1] = float(np.clip(cand[1], *p_bounds))
                coarse_score = float(_score_points(coarse, basis, Q, pos, wavelength)[0])
                cand_score = float(_score_points(cand, basis, Q, pos, wavelength)[0])
                if cand_score > coarse_score:
                    chosen, accepted = cand, True
        selected.append(chosen)
        init_trace.append({"stage": q, "coarse": coarse.tolist(), "selected": chosen.tolist(),
                           "fusion_considered": fusion_active, "fusion_accepted": accepted})

    D = np.asarray(selected, float)
    rng = np.random.default_rng(seed)
    refine_trace = []
    for sweep in range(2):
        for k in range(K):
            other = np.delete(D, k, axis=0)
            Aother = steering(other, pos, wavelength) if len(other) else np.empty((M, 0), complex)
            Q, _ = _orthogonal_complement(Aother, M)
            obj = ConditionalDMLObjective(R, Q, pos, wavelength, trace_R)
            before = D[k].copy()
            before_f = float(obj(before)[0])
            proposal, proposal_f = _rime_minimize(obj, before, t_bounds, p_bounds,
                                                  int(conditional_budget), rng)
            if proposal_f <= before_f + 64 * _EPS:
                D[k] = proposal
            refine_trace.append({"sweep": sweep + 1, "source": k + 1,
                                 "before": before.tolist(), "after": D[k].tolist(),
                                 "accepted": bool(proposal_f <= before_f + 64 * _EPS),
                                 "evaluations": obj.eval_count})

    # Bounded joint DML refinement: optimize local offsets around the cyclic solution.
    D0 = D.copy()
    h = float(evidence_step)
    def joint_loss(delta):
        C = D0 + np.asarray(delta).reshape(K, 2)
        C[:, 0] = _wrap_theta(C[:, 0], *t_bounds)
        C[:, 1] = np.clip(C[:, 1], *p_bounds)
        return 1.0 - _joint_gain(C, R, pos, wavelength) / max(trace_R, _EPS)

    bounds = []
    for k in range(K):
        bounds.append((-h, h))
        bounds.append((max(-h, p_bounds[0] - D0[k, 1]), min(h, p_bounds[1] - D0[k, 1])))
    # reshape order is [dtheta1,dphi1,dtheta2,dphi2,...]
    res = minimize(joint_loss, np.zeros(2 * K), method="L-BFGS-B", bounds=bounds)
    candidate = D0 + np.asarray(res.x).reshape(K, 2)
    candidate[:, 0] = _wrap_theta(candidate[:, 0], *t_bounds)
    candidate[:, 1] = np.clip(candidate[:, 1], *p_bounds)
    before_joint = float(joint_loss(np.zeros(2 * K)))
    after_joint = float(joint_loss(res.x))
    if np.isfinite(after_joint) and after_joint <= before_joint + 64 * _EPS:
        D = candidate

    order = np.argsort(D[:, 0])
    D = D[order]
    output = (D[:, 0].copy(), D[:, 1].copy())
    if not return_diagnostics:
        return output
    diagnostics = {
        "evidence_step_deg": h,
        "conditional_budget": int(conditional_budget),
        "fusion_activation": "recursion-aware; active for selection depth >= 3",
        "initialization": init_trace,
        "conditional_refinement": refine_trace,
        "joint_refinement": {"accepted": bool(after_joint <= before_joint + 64 * _EPS),
                             "before_loss": before_joint, "after_loss": after_joint},
    }
    return (*output, diagnostics)


__all__ = ["ConditionalDMLObjective", "steering", "trap_dml_rime"]
