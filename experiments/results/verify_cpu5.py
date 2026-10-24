"""Validate saved CPU5 summaries against the original authors' Part 2 ZIP.

Usage: python experiments/results/verify_cpu5.py IRIME_SINGLE_SCENE_CPU5_FINAL_PART2_20261024.zip

Standard-library only; does not rerun the estimator.
"""
from __future__ import annotations
import csv
import gzip
import io
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def verify(zip_path: Path) -> None:
    summary_path = Path(__file__).with_name("SINGLE_SCENE_CPU5_20261024.csv")
    with summary_path.open(encoding="utf-8", newline="") as f:
        summary = list(csv.DictReader(f))
    with zipfile.ZipFile(zip_path) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read("CPU5_SINGLE_SCENE_ALL_80_RAW.jsonl.gz"))) as f:
            records = [json.loads(line) for line in f]
    groups = defaultdict(list)
    for rec in records:
        assert rec["status"] == "ok"
        assert rec["setting"] == "snapshots_total0_L100"
        assert rec["CPU_ONLY"] is True and rec["BLAS_threads"] == 1
        assert rec["geometry_precomputed"] is False
        assert rec["reference_equivalent"] is True
        key = (rec["label"], str(rec["grid_step_deg"] or ""))
        groups[key].append(float(rec["runtime_sec"]) * 1000.0)
    assert len(groups) == 16 and len(records) == 80
    for row in summary:
        key = (row["method"], str(row["grid_step_deg"] or ""))
        timings = groups.pop(key)
        assert len(timings) == int(row["cpu_n"]) == 5
        assert abs(statistics.mean(timings) - float(row["cpu_mean_ms"])) < 1e-5, key
    assert not groups
    print("Verified: 80 original CPU records, 16 matched groups, zero mean mismatches")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_cpu5.py <part2.zip>")
    verify(Path(sys.argv[1]))
