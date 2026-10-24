"""Guard the exact paper estimator and verify saved point-estimate fixtures.

Run from the repository root:
    python -m unittest tests.test_paper_frozen -v

The test uses frozen observation protocols; it does not rerun all paper comparisons.
"""
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np

from paper_frozen.irime import irime as paper_irime
from experiments import paper_protocol as P


class PaperFrozenRegression(unittest.TestCase):
    def test_frozen_implementation_git_blob(self):
        code = Path(__file__).resolve().parents[1] / "paper_frozen" / "irime.py"
        # Exact source bytes are immutable for this historical manuscript version.
        self.assertEqual(hashlib.sha1(b"blob " + str(len(code.read_bytes())).encode() + bytes([0]) + code.read_bytes()).hexdigest(), "47653c5ec71a3c94afdb1fa0240e9565997aefee")

    def test_recorded_estimator_fixtures(self):
        fixtures = json.loads((Path(__file__).parent / "frozen_fixtures.json").read_text())
        for case in fixtures:
            with self.subTest(family=case["family"], trial=case["local_trial"]):
                family, value, local_trial = case["family"], case["value"], case["local_trial"]
                if family in ("snr", "snapshots"):
                    data = P.main_observation(family, value, local_trial)
                elif family == "sources":
                    data = P.source_observation(value, local_trial)
                else:
                    data = P.resolution_observation(value[0], value[1], local_trial)
                cfg, pos, _, x, seed, _, _ = data
                theta, phi = paper_irime(
                    x, pos, 1.0, cfg["theta_range"], cfg["phi_range"],
                    cfg["source_num"], h=P.H, B=P.B, seed=seed,
                )
                np.testing.assert_allclose(theta, case["theta"], rtol=0, atol=2e-5)
                np.testing.assert_allclose(phi, case["phi"], rtol=0, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
