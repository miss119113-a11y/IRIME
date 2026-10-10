"""R205 DSD angular seam and elevation-domain numerical regression tests."""
import unittest
import numpy as np
from irime import _differential_subspace_decouple, _steering

POS = np.column_stack([
    0.5 * np.cos(2 * np.pi * np.arange(12) / 12),
    0.5 * np.sin(2 * np.pi * np.arange(12) / 12),
    np.zeros(12),
])
CASES = {
    "interior": np.array([[86.0, 35.0], [87.3, 35.0]]),
    "azimuth_seam": np.array([[359.4, 35.0], [0.6, 35.0]]),
    "upper_elevation": np.array([[76.0, 89.6], [77.2, 89.6]]),
    "lower_elevation": np.array([[76.0, 0.4], [77.2, 0.4]]),
}

def sample_covariance(directions, seed):
    rng = np.random.default_rng(147 * seed + 5)
    signal = rng.normal(size=(2, 100)) + 1j * rng.normal(size=(2, 100))
    noise = rng.normal(size=(12, 100)) + 1j * rng.normal(size=(12, 100))
    x = _steering(directions, POS, 1.0) @ signal + 0.08 * noise
    return x @ x.conj().T / x.shape[1]

def stabilized_loss(angles, covariance):
    manifold = _steering(angles, POS, 1.0)
    proj = manifold @ np.linalg.pinv(manifold)
    return 1.0 - np.trace(proj @ covariance).real / np.trace(covariance).real

def correct(directions, covariance):
    return _differential_subspace_decouple(
        np.array(directions, copy=True), covariance, POS, 1.0,
        (0.0, 360.0), (0.0, 90.0),
    )

class DsdBoundaryRegression(unittest.TestCase):
    def test_four_geometries_in_domain_and_nonworsening(self):
        for label, points in CASES.items():
            for seed in range(15):
                with self.subTest(geometry=label, seed=seed):
                    covariance = sample_covariance(points, seed)
                    out, diagnostics = correct(points, covariance)
                    self.assertTrue(np.all(np.isfinite(out)))
                    self.assertTrue(np.all((out[:, 0] >= 0) & (out[:, 0] < 360)))
                    self.assertTrue(np.all((out[:, 1] >= 0) & (out[:, 1] <= 90)))
                    self.assertLessEqual(
                        stabilized_loss(out, covariance),
                        stabilized_loss(points, covariance) + 1e-10,
                    )
                    self.assertIn("applied", diagnostics)

    def test_periodic_representation_of_seam_pair(self):
        points = CASES["azimuth_seam"]
        wrapped_alias = points.copy()
        wrapped_alias[1, 0] += 360.0
        for seed in range(10):
            with self.subTest(seed=seed):
                covariance = sample_covariance(points, seed)
                a, diag_a = correct(points, covariance)
                b, diag_b = correct(wrapped_alias, covariance)
                self.assertEqual(diag_a["applied"], diag_b["applied"])
                if diag_a["applied"]:
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)

    def test_seam_correction_is_reachable(self):
        points = CASES["azimuth_seam"]
        accepted = sum(
            correct(points, sample_covariance(points, seed))[1]["applied"]
            for seed in range(15)
        )
        self.assertGreater(accepted, 0)

if __name__ == "__main__":
    unittest.main()
