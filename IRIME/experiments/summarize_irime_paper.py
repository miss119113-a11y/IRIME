#!/usr/bin/env python3
"""Summarize per-trial IRIME outputs using manuscript-compatible metrics.

The pooled joint-angular RMSE is sqrt(mean(trial RMSE squared)).
SR3 and strict resolution are percentages of successful trials.
Only actual observed values are summarized; no historical results are inserted.
"""
from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

GROUP_KEYS = ("k", "L", "snr_db", "h_deg", "variant", "separation_deg")

def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in GROUP_KEYS)].append(row)
    output = []
    for key, group in grouped.items():
        n = len(group)
        trial_rmse = [float(r["rmse_deg"]) for r in group]
        sr3 = [float(r["complete_sr3"]) for r in group]
        strict = [float(r["strict_resolution"])
                  for r in group if r.get("strict_resolution", "") != ""]
        times = [float(r["runtime_s"]) for r in group if r.get("runtime_s", "") != ""]
        result = dict(zip(GROUP_KEYS, key))
        result.update(
            trials=n,
            pooled_rmse_deg=math.sqrt(sum(x*x for x in trial_rmse)/n),
            complete_sr3_pct=100.0*sum(sr3)/n,
            strict_resolution_pct=(100.0*sum(strict)/len(strict) if strict else ""),
            mean_runtime_s=(sum(times)/len(times) if times else ""),
        )
        output.append(result)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV output of run_irime_paper.py")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("input CSV has no trial records")
    if "rmse_deg" not in rows[0] or "complete_sr3" not in rows[0]:
        parser.error("input CSV lacks required per-trial metrics")
    if "strict_resolution" not in rows[0] and any(r.get("separation_deg") for r in rows):
        parser.error("close-source input lacks strict_resolution; regenerate using the current runner")
    output = args.output or args.input.with_name(args.input.stem + "_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*GROUP_KEYS, "trials",
            "pooled_rmse_deg", "complete_sr3_pct", "strict_resolution_pct", "mean_runtime_s"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {len(summary)} condition summaries to {output}")

if __name__ == "__main__":
    main()
