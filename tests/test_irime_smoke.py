import numpy as np

from irime import irime, steering
from experiments.paper_protocol import uca_positions


def test_irime_smoke_three_sources():
    rng = np.random.default_rng(7)
    pos = uca_positions(1.0)
    truth = np.array([[30.42, 60.39], [120.27, 29.42], [240.51, 45.55]])
    A = steering(truth, pos, 1.0)
    S = (rng.normal(size=(3, 100)) + 1j * rng.normal(size=(3, 100))) / np.sqrt(2.0)
    N = 0.12 * (rng.normal(size=(12, 100)) + 1j * rng.normal(size=(12, 100))) / np.sqrt(2.0)
    X = A @ S + N
    az, el = irime(X, pos, 1.0, np.array([0.0, 360.0]), np.array([0.0, 90.0]), 3, seed=4)
    assert az.shape == (3,)
    assert el.shape == (3,)
    assert np.all(np.isfinite(az))
    assert np.all(np.isfinite(el))
    assert np.all((az >= 0.0) & (az < 360.0))
    assert np.all((el >= 0.0) & (el <= 90.0))
