# Frozen paper estimator

This directory preserves the **exact pre-R205 paper-estimator implementation** from the repository commit
[`93b2049831c0`](https://github.com/miss119113-a11y/2D-MUSIC/commit/93b2049831c0).
The code in `irime.py` here has Git blob SHA:

`47653c5ec71a3c94afdb1fa0240e9565997aefee`

This file is immutable as part of the manuscript's provenance. It predates the R205
azimuth seam/elevation boundary correction (commit `7e76da1746d8`).
The manuscript uses **DSC**; original implementation history used the label **DSD**.
Do not rename or reformulate the historical numerical operators.

## Reproduce paper-version principal cells

From the repository root (Python 3.11; pinned requirements):

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_paper_frozen.py' -v
python -m experiments.run_main --estimator-version paper-frozen --family snr --condition=-5dB --trials 1 --out results/paper_frozen_smoke
# The full proposed-estimator paper cells can be rerun with the same option:
python -m experiments.run_main --estimator-version paper-frozen --family sources --condition K9 --trials 500 --out results/paper_frozen_sources_k9
```

The run output includes observation identifiers, raw per-trial estimates,
summary metrics and a `MANIFEST.json` with implementation SHA, estimator version,
library versions, seed protocol and thread settings. Use a separate output folder
for every estimator version.

The repository-root `irime.py` is the **post-R205 boundary-corrected implementation**.
To test the boundary-corrected version instead, use `--estimator-version revised`.

The frozen fixture test verifies six recorded point estimates. It is a representative
regression check, **not** evidence that all comparison-algorithm rows or all main
Monte Carlo results have been independently rerun. The frozen runner covers the
proposed estimator's principal cells only; the full experimental archive is separate.
