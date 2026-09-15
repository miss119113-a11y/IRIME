"""Paper-level experimental settings used by the public SPFRIME code."""
from __future__ import annotations

import numpy as np

M = 12
RADIUS_OVER_WAVELENGTH = 0.5
EVIDENCE_STEP_DEG = 2.5
CONDITIONAL_BUDGET = 896
MONTE_CARLO_TRIALS = 500
THETA_RANGE = (0.0, 360.0)
PHI_RANGE = (0.0, 90.0)

# Nominal three-source geometry used in the main SNR/snapshot/robustness tests.
NOMINAL_THETA = np.array([30.42, 120.27, 240.51], dtype=float)
NOMINAL_PHI = np.array([60.39, 29.42, 45.55], dtype=float)

# Exact spherical two-source resolution protocol.
RESOLUTION_ANCHOR = (83.17, 35.29)
RESOLUTION_SPACINGS_DEG = np.array(
    [0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
     3.25, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0], dtype=float)
RESOLUTION_SNRS_DB = (10.0, 20.0)
RESOLUTION_SNAPSHOTS = 1024

# Generalization protocol reported in the manuscript.
GENERALIZATION_SEPARATED_K = (3, 5, 7)
GENERALIZATION_SEPARATED_SNR_DB = 10.0
GENERALIZATION_SEPARATED_SNAPSHOTS = 300
CLOSE_PAIR_SPACING_DEG = 4.0
CLOSE_PAIR_K5_SNRS_DB = (10.0, 20.0)
CLOSE_PAIR_SNAPSHOTS = 1024
UNEQUAL_POWER_K2_SNR_DB = 20.0
UNEQUAL_POWER_GAPS_DB = (0.0, 6.0, 12.0)


def uca_positions(wavelength: float = 1.0, m: int = M) -> np.ndarray:
    """M-element horizontal UCA used in the paper."""
    radius = RADIUS_OVER_WAVELENGTH * float(wavelength)
    alpha = 2.0 * np.pi * np.arange(m) / m
    return np.column_stack((radius * np.cos(alpha), radius * np.sin(alpha), np.zeros(m)))


def angle_grids():
    """Bounds arrays accepted by the reference implementation."""
    return np.array(THETA_RANGE), np.array(PHI_RANGE)
