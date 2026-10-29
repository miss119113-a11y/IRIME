# IRIME — Two-Dimensional ML-DOA Estimation

Research implementation of **Improved Rime Optimization Algorithm (IRIME)** for two-dimensional maximum-likelihood direction-of-arrival estimation. This directory contains **our proposed IRIME algorithm and IRIME-only reproduction scripts**. Implementations of third-party comparison methods are intentionally **not** included.

## Paper and method

The algorithm consists of observation-guided recursive initialization, differential-subspace correction, two cyclic source-wise conditional RIME searches, and an observation-adaptive terminal refinement stage. Core routine: `01_algorithms/irime_v7_compact.py::irime_v7_compact`. The original source's internal optimization helpers are retained to preserve the implementation.

## Install and run

```bash
cd IRIME
python -m pip install -r requirements.txt
python experiments/run_irime_paper.py --scenario snr --trials 1 --smoke
python experiments/run_irime_paper.py --scenario snapshots --trials 500
python experiments/run_irime_paper.py --scenario sources --trials 500
python experiments/run_irime_paper.py --scenario resolution --trials 500
python experiments/run_irime_paper.py --scenario ablation --trials 500
python experiments/run_irime_paper.py --scenario sensitivity --trials 500
python experiments/summarize_irime_paper.py results/resolution_irime_only.csv
python experiments/summarize_irime_paper.py results/ablation_irime_only.csv
```

By default, CSV outputs go into `results/`. Seeds and simulation conditions are recorded. The trial CSV includes joint-angular error, complete-set SR3, and (for close-source cases) strict resolution using both matched spherical errors below `0.49 * separation` and estimated pair separation of at least `0.5 * separation`. The summarizer computes pooled RMSE as `sqrt(mean(trial_rmse ** 2))`, SR3 percentage, and strict-resolution percentage for each condition/variant. For the Table 2 ablation conditions, the IRIME-only runner includes `K=9` at 10 dB and close-source separations of `1.5° at 10 dB`, `0.5° at 20 dB`, and `0.75° at 20 dB`. These are separate conditions, not interchangeable metrics. The single-algorithm scripts provide a reproducible implementation of the stated experimental conditions; published comparison curves were generated separately and those external algorithm implementations are not part of this release. Runtime measurements will depend on local CPU and library versions.

### Principal protocol

- 12-element uniform circular array, adjacent spacing half wavelength, known source count.
- Main SNR experiment: K=3, L=300, complex tones (0.1, 0.2, 0.3 cycles/snapshot), azimuths [30.42,47.00,150.27] degrees and elevations [60.39,46.27,35.55] degrees, total output SNR sweep from -5 to 15 dB.
- Snapshot experiment: 30 to 500 snapshots, total output SNR 0 dB.
- Source-count experiment: K=2,...,9, L=300, independent trial geometry and 10 dB per-source SNR.
- Resolution experiment: independent Gaussian source waveforms, K=2, L=1024, 10 dB per-source SNR and spherical separation grid as in the paper.
- Nominal search: h=2.5 degrees, B=896, two source-wise sweeps.

**Repository scope:** This release contains current IRIME implementation and IRIME-only simulation entrypoints, not third-party reimplementations or the frozen 500-trial data archives.
