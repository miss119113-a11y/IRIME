# IRIME: structure-guided RIME for 2D ML-DOA estimation

> **Private pre-release repository.** This repository is being prepared for the IRIME manuscript and will remain private during peer review. A cleaned public release is planned after publication.

This repository contains the reference implementation and paper-level protocol for **IRIME**, a structure-guided RIME framework for multi-source two-dimensional maximum-likelihood direction-of-arrival (2D ML-DOA) estimation.

IRIME combines:

1. rank-stabilized recursive projected-subspace initialization;
2. differential-subspace decoupling (DSD) for locally ill-conditioned close source pairs;
3. finite-budget cyclic source-wise RIME search on a rank-stabilized DML-form objective; and
4. observation-driven bounded terminal refinement for ordinary, weak/close, and high-occupancy states.

The source count is assumed known, consistent with the manuscript.

## Repository status

The previous SPFRIME repository state has been preserved on branch:

`archive/spfrime-20260915`

The `main` branch is now reserved for the IRIME paper code. During peer review the repository remains private. Before public release we will add the final frozen Monte Carlo runners, result manifests, and a release tag tied to the accepted manuscript.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reference implementation

The core estimator is in `irime.py`.

```python
from irime import irime

azimuth, elevation = irime(
    X,
    array_pos,
    wavelength,
    theta_grid,
    phi_grid,
    k_sources,
    h=2.5,
    B=896,
    seed=0,
)
```

The implementation follows the manuscript pipeline and exposes diagnostics when `return_diagnostics=True`.

## Frozen paper-level configuration

`experiments/paper_protocol.py` collects the principal settings used by the manuscript:

- 12-element horizontal UCA with adjacent chord length `lambda/2`;
- azimuth domain `0-360 deg` and elevation domain `0-90 deg`;
- shared angular scale `h = 2.5 deg`;
- conditional RIME budget `B = 896`;
- two cyclic conditional sweeps;
- 500 paired Monte Carlo trials per formal cell unless explicitly stated otherwise;
- exact great-circle two-source resolution protocol with the frozen noninteger anchor used in the paper;
- source-count geometry generated from a trial-specific common azimuth rotation and a golden-ratio elevation sequence.

## Tests

```bash
python -m pytest -q
```

The tests check the UCA geometry, protocol constants, and a basic IRIME smoke case.

## Reproducibility policy

The manuscript distinguishes two error families:

- joint-angular coordinate RMSE for the main SNR, snapshot, source-count, and runtime comparisons;
- spherical great-circle metrics for strict resolution, high-source-count failure tails, cross-geometry transfer, and robustness analyses.

Failed/incomplete estimates are retained through fixed failure penalties rather than silently discarded.

Objective-evaluation counts from heterogeneous estimators are not treated as equal-cost operations; same-machine wall-clock timing is used only as an implementation-level comparison.

## Public release plan

The repository is intentionally private during review. After publication we plan to:

1. freeze an accepted-paper release tag;
2. add the full paper experiment runners and result manifests;
3. verify a clean environment reproduction;
4. switch the repository to public access.
