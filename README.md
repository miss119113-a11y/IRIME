# IRIME for two-dimensional ML-DOA estimation

> **2026-10-24 CPU5 evidence update.** The latest verified results are in [experiments/results/CPU5_PROTOCOL_20261024.md](experiments/results/CPU5_PROTOCOL_20261024.md), with separate [main serial CPU](experiments/results/MAIN_SERIAL_CPU_20261024.csv) and [single-scene CPU5](experiments/results/SINGLE_SCENE_CPU5_20261024.csv) tables. The latter uses 100 observations for RMSE and five matched recorded CPU executions per configuration. Both cohorts are CPU-only/serial but use different grid-geometry policies; their nominal 1-degree times are not interchangeable. The experiment bundle records zero new estimator calls and preserves the original trial evidence externally. The current manuscript calls the pair-candidate correction **DSC**; historical code and provenance retain the old **DSD** name. `irime.py` preserves the R205 numerical seam/boundary correction; this documentation change does not modify its estimator operators.

This private review repository contains the proposed estimator and the minimal principal experiment runners, originally extracted for manuscript **R123**, with the **R205 DSD boundary correction** subsequently applied. It also includes the specifically requested original RIME-ML ablation reference. The earlier SPFRIME state remains in Git history and on `archive/spfrime-20260915`.

The estimator is extracted from the implementation in **RIME(4).zip** used by the paper experiments. It retains the numerical path for:

1. rank-stabilized recursive projected-subspace initialization;
2. differential-subspace decoupling (DSD);
3. two cyclic conditional RIME searches; and
4. observation-driven bounded joint terminal refinement.

The earlier short `irime.py` on `main` approximated this pipeline with different population updates, covariance fitting, and terminal settings. It has been replaced by the experiment implementation's dependency closure. Unrelated estimators, historical variants, environments, large results, and manuscript files are excluded from this minimal repository.

## Reproduction versions (manuscript and R205)

The historical manuscript implementation is saved without numerical edits as
[`paper_frozen/irime.py`](paper_frozen/irime.py) (Git blob `47653c5ec71a3c94afdb1fa0240e9565997aefee`,
pre-R205 commit `93b2049831c0`). The repository-root `irime.py` is the
subsequent R205 azimuth/elevation boundary-corrected version. The paper uses the
name **DSC** (earlier code called this DSD); the naming change is not an algorithmic
modification. [Frozen-version instructions](paper_frozen/README.md) and
[`test_paper_frozen.py`](tests/test_paper_frozen.py) protect the historical source
and point-estimate fixtures. Neither version is silently substituted for the other.

```bash
python -m unittest discover -s tests -p 'test_paper_frozen.py' -v
python -m experiments.run_main --estimator-version paper-frozen --family sources --condition K9 --trials 1 --out results/paper_k9_smoke
python -m experiments.run_main --estimator-version revised --family sources --condition K9 --trials 1 --out results/r205_k9_smoke
```

Both outputs carry `estimator_version` and code hashes in `MANIFEST.json`.
The reproducibility tests check representative fixtures; complete statistical
reproduction of all paper comparison algorithms remains outside this minimal repository.

## Install and test

Python 3.11 is the recorded paper environment. `requirements.txt` pins the recorded NumPy and SciPy versions. A standard-library test runner is sufficient:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`tests/frozen_fixtures.json` stores six point-estimate fixtures from the extracted experiment implementation. The tests also check the conditional projector identity, analytic degree derivatives, eigenvector-phase alignment, exact spherical spacing, and penalties for missing directions. Additional DSD regression tests cover azimuth-seam alias invariance, both elevation boundaries, output domains, and non-worsening stabilized loss. Cross-platform floating-point tolerances are used for the fixtures.

During extraction, 28 representative SNR, snapshot, source-count, resolution, and timing cells were checked against the full experiment package: observations and point estimates matched bitwise in the validation environment. This check verifies the extraction, not an independent rerun of every 500-trial paper comparison. `PROVENANCE.json` identifies each source module and its SHA-256.

## Estimate DOAs

```python
import numpy as np
from irime import irime

azimuth, elevation = irime(
    X, array_pos, wavelength,
    np.arange(361), np.arange(91), known_source_count,
    h=2.5, B=896, seed=0,
)
```

`X` has shape `(M, L)` and `array_pos` has shape `(M, 3)`. Angles are in degrees. The grid arguments supply domain bounds; `h` controls the internal evidence grid and terminal neighborhood. The paper assumes known `K < M` and the upper hemisphere. Diagnostics are available through `return_diagnostics=True`.

## Reproduce the proposed method's principal experiments

Run from the repository root:

```bash
# Quick execution check for every principal cell (not formal evidence).
python -m experiments.run_main --family all --trials 1 --out results/smoke

# Frozen principal experiments: append --estimator-version paper-frozen to each command.
python -m experiments.run_main --family snr --trials 500 --out results/snr
python -m experiments.run_main --family snapshots --trials 500 --out results/snapshots
python -m experiments.run_main --family sources --trials 500 --out results/sources
python -m experiments.run_main --family resolution --trials 500 --out results/resolution
python -m experiments.run_main --family efficiency --trials 500 --out results/efficiency

# One source-count cell, with the same frozen seeds.
python -m experiments.run_main --family sources --condition K9 --trials 500 --out results/k9
```

Each output directory contains resumable `trials.csv`, aggregated `summary.csv`, and an environment/configuration `MANIFEST.json`. Observation generation is excluded from estimator timing. Numerical-library threads are set to one before importing NumPy. The formal trial offsets are preserved: 70000 for the SNR/snapshot families, 80000 for resolution, zero for source count, and 180001 for the timing seed sequence.

The SNR/snapshot/timing cells use deterministic complex tones and total clean array-output SNR. The source-count family uses independent Gaussian waveforms normalized to unit finite-sample power, regular azimuth geometry, and fixed per-source SNR. Resolution uses unnormalized independent Gaussian waveforms and per-source expected-power SNR. These conventions intentionally differ and are recorded in the protocol.

This minimal runner reproduces **IRIME's own principal cells**. It does not regenerate the paper's six comparative baselines, sensitivity, mismatch, URA, or supplemental 2000-trial studies. Those full-study records remain part of the authors' experiment archive.

## Original RIME-ML ablation reference

For the R123 local numerical environment, use Python 3.12.14 in a separate
virtual environment and install `requirements-ablation.txt`. The original
`requirements.txt` continues to record the paper's main experiment environment.

```bash
python -m pip install -r requirements-ablation.txt
# Five frozen conditions; each observation is shared by both estimators.
python -m experiments.run_rime_ablation --trials 500 --workers 1 --out results/rime_ablation
python -m experiments.summarize_rime_ablation results/rime_ablation
```

The reference searches all `2*K` angles directly with the original 2023
soft-rime, hard-rime and positive-greedy operators. It uses uniform random
initialization, box absorption, `W=5`, population 50, and 2000 generations:
exactly 100,050 raw joint-DML objective evaluations including initialization.
No recursive initializer, DSD, local polish, sorting during optimization, or
IRIME terminal controller is included. A rank-aware SVD evaluates the raw
projection residual, including duplicate-source candidates. The numerical
rank cutoff is `1e-10` relative to the largest singular value.

The port follows the official MATLAB equations, including MATLAB's rounding
rule and L2 row normalization of the fitness vector. NumPy batches independent
uniform draws, so MATLAB and NumPy random-seed paths are not identical. See
`THIRD_PARTY_NOTICES.md` for the source credit and MIT notice. Four added tests
check the raw projection against a pseudoinverse, permutation/azimuth
invariance, the official scalar update equations, and reproducible evaluation
counts with greedy acceptance. Together with the six original contracts, the
repository has ten tests.

The five cells are -5 dB with K=3, 10 dB with spherical spacing 1.5 degrees,
20 dB with spacing 1 degree, K=8 and K=9. Both methods retain the frozen main
observation generators and matching rules. All per-trial estimates, seeds,
observation hashes, failures and exact reference evaluation counts are saved.
The summarizer checks every observation pair and resamples complete pairs for
95% confidence intervals. Intervals are per condition without multiplicity
correction. This is a whole-framework reference; it does not isolate the value
of IRIME's conditional RIME kernel or equalize total estimator cost.

The R123 local paired run uses Linux, Python 3.12.14, NumPy 2.3.5 and SciPy
1.17.0. It is reported separately from the historical module-ablation table.
Set `--workers` above one only to accelerate accuracy experiments. Those
contended wall times cannot be used as a serial runtime comparison or mixed
with the paper's recorded Windows timing results.

## Criteria and implementation limits

The raw deterministic likelihood, stabilized DML-form search, nonnegative covariance-fit residual, and SML-form terminal score have distinct meanings. The covariance fitter uses monotone active removal and is an approximate nonnegative least-squares rule. The SML-form score does not enforce the positive-semidefinite source-covariance feasibility inequality. The terminal gates are operational heuristics. Non-worsening acceptance concerns each selected criterion, rather than one global raw likelihood.

**R205 DSD boundary correction (post-paper-results numerical version):** The centroid starts from the midpoint of the shortest periodic azimuth arc. The centroid optimizer wraps azimuth but rejects elevation-out-of-range proposals. The symmetric-pair line search wraps azimuth and rejects invalid elevations before evaluating the stabilized DML-form loss. A candidate is accepted only if the tested loss decreases. This addresses the coordinate-seam and elevation-range defects in the local DSD proposal; it does not establish a globally valid spherical parameterization or guarantee improved true-angle error.

The R205 published-in-draft tables/figures were generated with the preceding experimental implementation, **not** the patched DSD. Existing six recorded estimate fixtures and earlier 28-cell extraction checks document the preceding implementation and are not a full statistical validation of the patched version. For scientific reproducibility, re-evaluate all materially affected cells, regenerate corresponding tables/plots as needed, and record the new implementation SHA before attributing those results to the patched code. The repository remains private during review.

`B=896` controls one conditional call, not total estimator work. The historical aggregate evaluation diagnostic omitted several kinds of calls; the public wrapper therefore omits that total rather than reporting it as an exact count. Wall-clock measurements are specific to the stated implementation and machine.

The repository remains private during review. Public access is planned upon publication.
