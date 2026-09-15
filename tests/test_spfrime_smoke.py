import numpy as np

from spfrime import steering, trap_dml_rime
from experiments.paper_protocol import angle_grids, uca_positions


def test_spfrime_returns_finite_known_source_count():
    rng = np.random.default_rng(7)
    pos = uca_positions(1.0)
    truth = np.array([[40.0, 35.0], [150.0, 50.0]])
    A = steering(truth, pos, 1.0)
    n = np.arange(80)
    S = np.vstack((np.exp(1j * 2 * np.pi * 0.13 * n),
                   np.exp(1j * 2 * np.pi * 0.27 * n)))
    clean = A @ S
    noise = 0.03 * (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape))
    tg, pg = angle_grids()
    th, ph = trap_dml_rime(clean + noise, pos, 1.0, tg, pg, 2,
                           evidence_step=5.0, conditional_budget=80, seed=11)
    assert th.shape == (2,)
    assert ph.shape == (2,)
    assert np.all(np.isfinite(th))
    assert np.all(np.isfinite(ph))
    assert np.all((ph >= 0.0) & (ph <= 90.0))
