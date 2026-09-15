import numpy as np

from experiments.paper_protocol import M, uca_positions


def test_uca_adjacent_chord_is_half_wavelength():
    p = uca_positions(1.0)
    assert p.shape == (M, 3)
    chord = np.linalg.norm(p[1] - p[0])
    assert np.isclose(chord, 0.5, rtol=0.0, atol=1e-12)


def test_uca_is_planar_and_centered():
    p = uca_positions(1.0)
    assert np.allclose(p[:, 2], 0.0)
    assert np.allclose(p[:, :2].mean(axis=0), 0.0, atol=1e-12)
