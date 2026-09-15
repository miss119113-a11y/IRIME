# SPFRIME: 2D DOA estimation code

This repository contains the algorithm and experiment code used for the SPFRIME manuscript on multi-source two-dimensional direction-of-arrival (DOA) estimation with a uniform circular array.

SPFRIME combines rank-truncated recursive projected-subspace initialization, recursion-aware selective subgrid fusion with exact projected-subspace rescoring, cyclic conditional-DML refinement driven by RIME, and bounded joint-DML refinement.

## Repository scope

The repository intentionally contains code needed for the algorithm and reported experiments. Large raw Monte Carlo outputs, development prototypes, local virtual environments, IDE metadata, and retired experiment branches are excluded. The included runners create their own result directories under `results/`.

Some internal module names retain historical `v7h` identifiers because those files are the tested implementation used for the final numerical results. The paper method name is **SPFRIME**.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Use SPFRIME as an estimator

```python
from spfrime import trap_dml_rime
```

The paper configuration uses a projected-evidence step of 2.5 degrees and a conditional-search budget of 896 evaluations per source coordinate. The source count is assumed known, as stated in the manuscript.

## Reproduce the paper experiments

Run from the repository root.

### Full paper suite

```bash
python run_spfrime_experiments.py --trials 500 --workers 8
```

Equivalent direct command:

```bash
python experiments/protocol/run_spfrime_final_kge3ms_b896_h25_suite.py --trials 500 --workers 8
```

The suite covers the main SNR/snapshot/source-count experiments, exact two-source spherical-resolution experiments, robustness/generalization tests, mechanism ablation, evidence-step sensitivity, and the same-machine single-thread efficiency benchmark.

### Individual experiments

Main performance and robustness:

```bash
python experiments/protocol/run_v7h_legacy_paper_formal.py --trials 500 --workers 8 --methods rime_v7h_fusion
```

Exact spherical two-source resolution:

```bash
python experiments/protocol/run_v7h_azimuth_resolution_formal.py --phase paper_main --trials 500 --workers 8 --methods rime_v7h_fusion
```

Generalization:

```bash
python experiments/protocol/run_v7h_generalization_formal.py --trials 500 --workers 8 --methods rime_v7h_fusion
```

Ablation:

```bash
python experiments/protocol/run_v7h_ablation_formal.py --trials 500 --workers 8
```

Projected-evidence-step sensitivity at fixed `B=896`:

```bash
python experiments/protocol/run_spfrime_h_sensitivity_fixedB896_h25.py --trials 500 --workers 8
```

Single-thread efficiency benchmark:

```bash
python experiments/protocol/run_spfrime_final_serial_efficiency.py --trials 500 --warmup 5
```

## Regression tests

```bash
python -m pytest -q experiments/final_core_execution/tests
```

## Reproducibility notes

All methods in a paired Monte Carlo trial receive the same generated observation for a given condition/seed pair. The paper-facing runners fix the final SPFRIME settings and write manifests/audit files alongside generated results.

Objective-evaluation counts from heterogeneous estimators are not treated as equal-cost operations; wall-clock runtime is the implementation-level efficiency measure used in the manuscript.
