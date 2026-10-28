"""Observable-only source-count estimation for end-to-end MUSIC experiments.

The estimator deliberately operates on the sample covariance eigenspectrum and
does not inspect any simulation label.  It is kept separate from the MUSIC
search code so the known-K oracle protocol remains unchanged and the auto-K
extension can be audited independently.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _validate_observation(X: np.ndarray) -> np.ndarray:
    observation = np.asarray(X, dtype=complex)
    if observation.ndim != 2:
        raise ValueError("X must be a two-dimensional sensor-by-snapshot matrix")
    sensors, snapshots = observation.shape
    if sensors < 2:
        raise ValueError("X must contain at least two sensors")
    if snapshots < 2:
        raise ValueError("X must contain at least two snapshots")
    if not np.all(np.isfinite(observation)):
        raise ValueError("X must contain only finite values")
    return observation


def estimate_source_count(
    X: np.ndarray,
    *,
    method: str = "mdl",
    max_sources: int | None = None,
    eigenvalue_floor: float = 1e-12,
) -> dict:
    """Estimate the number of narrowband sources with MDL or AIC.

    The implementation follows the Wax--Kailath likelihood-ratio criterion.  A
    positive covariance-eigenvalue floor keeps finite-snapshot or near-singular
    observations numerically defined without changing valid positive spectra.
    """
    observation = _validate_observation(X)
    method = str(method).strip().lower()
    if method not in {"mdl", "aic"}:
        raise ValueError("method must be 'mdl' or 'aic'")
    eigenvalue_floor = float(eigenvalue_floor)
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be a positive finite value")

    sensors, snapshots = observation.shape
    if max_sources is None:
        max_sources = sensors - 1
    max_sources_float = float(max_sources)
    max_sources = int(max_sources)
    if max_sources_float != max_sources or not 0 <= max_sources < sensors:
        raise ValueError("max_sources must satisfy 0 <= max_sources < number of sensors")

    covariance = observation @ observation.conj().T / float(snapshots)
    eigenvalues = np.sort(np.real(linalg.eigvalsh(covariance)))[::-1]
    scale = max(float(np.max(eigenvalues)), 1.0)
    floored = np.maximum(eigenvalues, eigenvalue_floor * scale)
    candidate_orders = np.arange(max_sources + 1, dtype=int)
    criteria = np.empty(len(candidate_orders), dtype=float)

    for index, k in enumerate(candidate_orders):
        noise_eigs = floored[int(k):]
        noise_dim = int(noise_eigs.size)
        log_geometric_mean = float(np.mean(np.log(noise_eigs)))
        log_arithmetic_mean = float(np.log(np.mean(noise_eigs)))
        likelihood = -float(snapshots * noise_dim) * (
            log_geometric_mean - log_arithmetic_mean
        )
        parameters = int(k * (2 * sensors - k))
        if method == "mdl":
            penalty = 0.5 * parameters * np.log(float(snapshots))
        else:
            penalty = float(parameters)
        criteria[index] = likelihood + penalty

    selected_index = int(np.argmin(criteria))
    return {
        "method": method,
        "k_hat": int(candidate_orders[selected_index]),
        "candidate_orders": candidate_orders,
        "criterion_values": criteria,
        "covariance_eigenvalues": eigenvalues,
        "eigenvalue_floor": eigenvalue_floor,
        "sensors": int(sensors),
        "snapshots": int(snapshots),
    }


def validate_model_order_result(result: dict, sensors: int) -> dict:
    """Validate a cached observable-only model-order result before reuse."""
    if not isinstance(result, dict) or "k_hat" not in result:
        raise ValueError("model_order_result must be a result dictionary with k_hat")
    k_hat_float = float(result["k_hat"])
    k_hat = int(k_hat_float)
    if k_hat_float != k_hat or not 0 <= k_hat < int(sensors):
        raise ValueError("model_order_result k_hat must satisfy 0 <= k_hat < number of sensors")
    return dict(result)
