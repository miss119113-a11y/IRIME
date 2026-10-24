# 2026-10-24: frozen main-study and CPU5 single-scene results

These two tables are different measurement cohorts, **not** interchangeable 1-degree numbers.

- `MAIN_SERIAL_CPU_20261024.csv`: 500-trial serial CPU principal benchmark, cached grid geometry; RMSE and mean runtime from the Codex verified Table 5.
- `SINGLE_SCENE_CPU5_20261024.csv`: 12-element UCA, K=3, L=100, total array-output SNR=0 dB. Accuracy from 100 observations/configuration; serial CPU elapsed times are arithmetic means of five original matched attempts/configuration, including grid generation (without precomputed geometry). The original archive has 80 individual records and no newly executed estimator calls.
- Five baselines run full-domain grids of 1, 0.1, 0.01 degree. IRIME is fixed at h=2.5 degrees, B=896. No DE-DML fine-grid scan.
- Do not describe these CPU5 results as a newly rerun 100-trial CPU experiment; the accuracy and timing denominators differ.
- Original raw evidence and SHA mappings: the authors' `IRIME_SINGLE_SCENE_CPU5_FINAL_PART1_20261024.zip` and `IRIME_SINGLE_SCENE_CPU5_FINAL_PART2_20261024.zip`; raw CPU5 records live in `CPU5_SINGLE_SCENE_ALL_80_RAW.jsonl.gz` in part 2. Large raw archives are deliberately not duplicated here.
- Repository `irime.py` includes the R205 azimuth-seam and elevation-boundary correction; the current manuscript calls the candidate-pair correction **DSC**, previously DSD in code history. Earlier experiment fixtures are not evidence of a complete rerun of all principal figures after the R205 correction.
- CPU5 input set: trial IDs 70000–70004, 16 configurations, 80 recorded CPU times, zero output mismatches on the recorded comparisons. External-load-monitoring coverage is incomplete; avoid describing measurements as exclusive-machine or load-free.
