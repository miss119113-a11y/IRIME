# SPFRIME: 2D DOA estimation code

This repository contains the public reference implementation and paper-level experiment code for SPFRIME, a multi-source two-dimensional direction-of-arrival (DOA) estimator for a uniform circular array.

SPFRIME combines rank-truncated recursive projected-subspace initialization, recursion-aware selective subgrid fusion with exact projected-subspace rescoring, cyclic conditional-DML refinement driven by RIME, and bounded joint-DML refinement.

## Repository scope

This is a clean publication repository. It contains the method implementation, the principal experiment settings, a compact reproduction runner, and a smoke test. Development prototypes, virtual environments, IDE files, large Monte Carlo result folders, and obsolete branches are intentionally excluded.

The manuscript contains the scientific definitions needed to understand the method and experiments. Code-level generation details live here rather than being expanded in the paper.

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

## Method implementation

The self-contained reference implementation is:

```text
spfrime.py
```

Basic use:

```python
from spfrime import trap_dml_rime

azimuth, elevation = trap_dml_rime(
    X,
    array_pos,
    wavelength,
    theta_grid,
    phi_grid,
    k_sources,
    evidence_step=2.5,
    conditional_budget=896,
    seed=0,
)
```

The source count is assumed known, consistent with the manuscript.

## Paper-level settings

The principal settings used by the public runner are collected in:

```text
experiments/paper_protocol.py
```

The final paper configuration uses:

- 12-element UCA;
- projected-evidence step `h = 2.5 deg`;
- conditional-DML RIME budget `B = 896` per coordinate update;
- two cyclic conditional-DML sweeps;
- the same `h` as the half-width of the final bounded joint-DML refinement;
- 500 Monte Carlo trials for the reported statistical experiments.

## Reproduce the principal SPFRIME experiments

Run from the repository root.

### Main SNR experiment

```bash
python -m experiments.run_spfrime_paper --experiment main-snr --trials 500
```

### Exact spherical two-source resolution experiment

```bash
python -m experiments.run_spfrime_paper --experiment resolution --trials 500
```

### Both

```bash
python -m experiments.run_spfrime_paper --experiment all --trials 500
```

Generated CSV files are written to:

```text
results/public_reproduction/
```

## Test

```bash
python -m pytest -q tests/test_spfrime_smoke.py
```

## Reproducibility notes

The public code keeps the manuscript-level experimental definitions explicit while avoiding unnecessary internal development machinery. The repository can be extended with the remaining comparison-method runners if the full baseline reproduction package is released later.

Objective-evaluation counts from heterogeneous estimators should not be interpreted as equal-cost operations; the manuscript uses measured wall-clock time for the implementation-level efficiency comparison.
