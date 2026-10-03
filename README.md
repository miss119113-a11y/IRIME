# IRIME for two-dimensional ML-DOA estimation

This private review repository contains the proposed estimator and the minimal principal experiment runners for manuscript revision **R122**. The earlier SPFRIME state remains in Git history and on `archive/spfrime-20260915`.

The estimator is extracted from the implementation in **RIME(4).zip** used by the paper experiments. It retains the numerical path for:

1. rank-stabilized recursive projected-subspace initialization;
2. differential-subspace decoupling (DSD);
3. two cyclic conditional RIME searches; and
4. observation-driven bounded joint terminal refinement.

The earlier short `irime.py` on `main` approximated this pipeline with different population updates, covariance fitting, and terminal settings. It has been replaced by the experiment implementation's dependency closure. Unrelated estimators, historical variants, environments, large results, and manuscript files are excluded from this minimal repository.

## Install and test

Python 3.11 is the recorded paper environment. `requirements.txt` pins the recorded NumPy and SciPy versions. A standard-library test runner is sufficient:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`tests/frozen_fixtures.json` stores six point-estimate fixtures from the extracted experiment implementation. The tests also check the conditional projector identity, analytic degree derivatives, eigenvector-phase alignment, exact spherical spacing, and penalties for missing directions. Cross-platform floating-point tolerances are used for the fixtures.

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

# Frozen 500-trial principal experiments for the proposed estimator.
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

## Criteria and implementation limits

The raw deterministic likelihood, stabilized DML-form search, nonnegative covariance-fit residual, and SML-form terminal score have distinct meanings. The covariance fitter uses monotone active removal and is an approximate nonnegative least-squares rule. The SML-form score does not enforce the positive-semidefinite source-covariance feasibility inequality. The terminal gates are operational heuristics. Non-worsening acceptance concerns each selected criterion, rather than one global raw likelihood.

The frozen DSD implementation uses a local arithmetic angular midpoint and evaluates symmetric proposals before later boundary repair. It does not provide a seam-invariant or globally valid spherical coordinate chart. This convention is retained to preserve the reported estimator. A circular-midpoint or boundary-handling change would define a new numerical version and requires revalidation of affected cells.

`B=896` controls one conditional call, not total estimator work. The historical aggregate evaluation diagnostic omitted several kinds of calls; the public wrapper therefore omits that total rather than reporting it as an exact count. Wall-clock measurements are specific to the stated implementation and machine.

The repository remains private during review. Public access is planned upon publication.
