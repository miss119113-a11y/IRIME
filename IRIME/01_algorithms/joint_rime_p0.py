"""P0 control in manuscript Sec.4.6: direct 2K-dimensional joint RIME.

Standalone fixed-budget baseline: 128 candidates and four RIME generations.
Uses normalized concentrated DML rather than a conditional source-wise objective.
This is deliberately not used by the proposed IRIME estimator.
"""
from __future__ import annotations
import numpy as np
from irime_v3_clean import _steering


def direct_joint_rime_p0(X, array_pos, wavelength, K, *, seed=42,
                         population_size=128, generations=4):
    K = int(K)
    if K < 1 or K >= len(array_pos):
        raise ValueError('Required 1 <= K < M')
    rng = np.random.default_rng(int(seed))
    cov = (X @ X.conj().T) / X.shape[1]
    trace_cov = max(float(np.real(np.trace(cov))), 1e-15)
    lo = np.tile([0.0, 0.0], K)
    span = np.tile([360.0, 90.0], K)

    def repair(pop):
        p = np.asarray(pop, dtype=float).copy()
        p[:, 0::2] %= 360.0
        p[:, 1::2] = np.clip(p[:, 1::2], 0.0, 90.0)
        return p

    def fitness(pop):
        out = np.empty(len(pop), dtype=float)
        for i, x in enumerate(pop):
            A = _steering(x.reshape(K,2), array_pos, wavelength)
            if not np.isfinite(A).all():
                out[i] = 1e9
                continue
            # Projector from pseudoinverse remains defined at close directions.
            P = A @ np.linalg.pinv(A, rcond=1e-10)
            out[i] = max(0.0, float(np.real(np.trace((np.eye(len(array_pos))-P)@cov)))/trace_cov)
        return out

    n = int(population_size)
    if n < 4 or int(generations) < 1: raise ValueError('Invalid P0 budget')
    pop = repair(rng.random((n,2*K))*span+lo)
    f = fitness(pop)
    best_idx = int(np.argmin(f))
    best = pop[best_idx].copy()
    f_best = f[best_idx]
    # Original-RIME-style population soft exploration, best-guided hard update, greedy retention.
    for t in range(1, int(generations)+1):
        fraction = t / float(generations)
        rime_factor = 2.0*(rng.random()-0.5)*np.cos(np.pi*t/max(1.0,generations/10.0)) \
                      * (1.0-round(t*5.0/generations)/5.0)
        trial = pop.copy()
        explore_mask = rng.random(pop.shape) < np.sqrt(fraction)
        exploration = best + rime_factor*span*(rng.random(pop.shape)-0.5)
        trial = np.where(explore_mask, exploration, trial)
        fit_scale = (np.max(f)-f)/(np.max(f)-np.min(f)+1e-12)
        hard_mask = rng.random(pop.shape) < fit_scale[:,None]
        trial = np.where(hard_mask, best[None,:], trial)
        trial = repair(trial)
        f_new = fitness(trial)
        improve = f_new < f
        pop[improve] = trial[improve]
        f[improve] = f_new[improve]
        winner = int(np.argmin(f))
        if f[winner] < f_best:
            f_best = f[winner]
            best = pop[winner].copy()
    sol = best.reshape(K,2)
    return sol[:,0], sol[:,1]
