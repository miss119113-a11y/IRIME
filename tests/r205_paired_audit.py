"""Paired post-R205 DSD-impact audit. NOT a replacement for 500-trial paper data."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from irime import irime as new_irime
from irime import _steering, _differential_subspace_decouple
from experiments import paper_protocol as P

OLD_SHA = "78b79ba2331ccff96d2d7c3c0c4779ad1db5d5a9"
CASES = [
    ("snr", -5, "-5dB"),
    ("snr", 0, "0dB"),
    ("snr", 5, "5dB"),
    ("snapshots", 100, "L100"),
    ("snapshots", 500, "L500"),
    ("sources", 2, "K2"),
    ("sources", 3, "K3"),
    ("sources", 7, "K7"),
    ("sources", 9, "K9"),
    ("resolution", (0.5, 20), "20dB_Delta0.5"),
    ("resolution", (1.5, 10), "10dB_Delta1.5"),
    ("resolution", (2.5, 20), "20dB_Delta2.5"),
    ("efficiency", 0, "K3_L100_0dB"),
]
FIELDS = ["family","condition","trial","seed","old_status","new_status","theta_max_delta_deg","phi_max_delta_deg","estimate_changed","old_joint_rmse_deg","new_joint_rmse_deg","old_spherical_rmse_deg","new_spherical_rmse_deg","old_strict_resolution","new_strict_resolution","old_error","new_error"]

def load_old():
    text = subprocess.check_output(["git", "show", f"{OLD_SHA}:irime.py"], text=True)
    path = Path("/tmp/irime_pre_r205_dsd.py")
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("irime_pre_r205_dsd", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.irime

def observation(family, value, trial):
    if family in ("snr", "snapshots"):
        return P.main_observation(family, value, trial)
    if family == "sources":
        return P.source_observation(value, trial)
    if family == "resolution":
        return P.resolution_observation(value[0], value[1], trial)
    return P.efficiency_observation(trial)

def estimate(method, data):
    cfg, pos, _, x, seed, _, _ = data
    try:
        theta, phi = method(x, pos, 1.0, cfg["theta_range"], cfg["phi_range"],
                            cfg["source_num"], h=P.H, B=P.B, seed=seed)
        points = np.column_stack([theta, phi])
        if points.shape != (cfg["source_num"], 2):
            raise ValueError(f"invalid estimated shape {points.shape}")
        errors = P.pair_doa_errors(cfg["theta_true"], cfg["phi_true"], theta, phi)
        spherical = P.spherical_match(cfg["theta_true"], cfg["phi_true"], theta, phi)
        strict = (P.resolution_metrics(cfg, theta, phi)["strict_resolution_success"]
                  if cfg["source_num"] == 2 and "resolution" in str(cfg) else "")
        return dict(ok=True, points=points, joint=float(errors["joint_rmse"]),
                    spherical=float(spherical["trial_rmse_deg"]), error="")
    except Exception as exc:
        return dict(ok=False, points=None, joint=float("nan"),
                    spherical=float("nan"), error=f"{type(exc).__name__}: {exc}")

def run_family_audit(n, out):
    old_method = load_old()
    rows = []
    for family, value, label in CASES:
        for trial in range(n):
            data = observation(family, value, trial)
            cfg, pos, _, x, seed, _, _ = data
            old = estimate(old_method, data)
            new = estimate(new_irime, data)
            max_theta = max_phi = float("nan")
            changed = "unknown"
            if old["ok"] and new["ok"]:
                a, b = old["points"], new["points"]
                max_theta = float(np.max(np.abs((a[:,0]-b[:,0]+180)%360-180)))
                max_phi = float(np.max(np.abs(a[:,1]-b[:,1])))
                changed = int(max(max_theta, max_phi) > 2e-5)
                if not (np.all(np.isfinite(b)) and np.all((b[:,0]>=0)&(b[:,0]<360)) and
                        np.all((b[:,1]>=0)&(b[:,1]<=90))):
                    raise AssertionError(f"Patched estimate out of bounds: {family} {label} {trial} {b}")
            row=dict(family=family,condition=label,trial=trial,seed=seed,
                old_status="ok" if old["ok"] else "error",new_status="ok" if new["ok"] else "error",
                theta_max_delta_deg=max_theta,phi_max_delta_deg=max_phi,
                estimate_changed=changed,old_joint_rmse_deg=old["joint"],
                new_joint_rmse_deg=new["joint"],
                old_spherical_rmse_deg=old["spherical"],
                new_spherical_rmse_deg=new["spherical"],
                old_strict_resolution="",new_strict_resolution="",
                old_error=old["error"],new_error=new["error"])
            rows.append(row)
        selected=[r for r in rows if r["family"]==family and r["condition"]==label]
        print(f"{family:11s} {label:17s}: changed {sum(x['estimate_changed']==1 for x in selected)}/{n}  failures {sum(x['new_status']!='ok' for x in selected)}", flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"paired_main.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    stats={}
    for family,value,label in CASES:
        rs=[r for r in rows if r["family"]==family and r["condition"]==label]
        stats[family+"/"+label]=dict(trials=len(rs),changed=sum(r["estimate_changed"]==1 for r in rs),
            failed_old=sum(r["old_status"]!="ok" for r in rs),
            failed_new=sum(r["new_status"]!="ok" for r in rs),
            old_pooled_joint_rmse=float(np.sqrt(np.mean([r["old_joint_rmse_deg"]**2 for r in rs]))),
            new_pooled_joint_rmse=float(np.sqrt(np.mean([r["new_joint_rmse_deg"]**2 for r in rs]))))
    (out/"paired_main_summary.json").write_text(json.dumps(stats,indent=2))

def run_directed_boundary(out, n=15):
    positions=P.circular_array_positions()
    rng=np.random.default_rng(123)
    truth_cases={
        "interior":np.array([[86,35],[87.3,35.]]),
        "azimuth_seam":np.array([[359.4,35.],[0.6,35.]]),
        "upper_elevation":np.array([[76,89.6],[77.2,89.6]]),
        "lower_elevation":np.array([[76,0.4],[77.2,0.4]])}
    old_method=load_old()
    module=sys.modules["irime_pre_r205_dsd"]
    results=[]
    for name,points in truth_cases.items():
        for t in range(n):
            rg=np.random.default_rng(147*t+5)
            signal=rg.normal(size=(2,100))+1j*rg.normal(size=(2,100))
            noise=rg.normal(size=(12,100))+1j*rg.normal(size=(12,100))
            X=_steering(points,positions,1)@signal + 0.08*noise
            cov=X@X.conj().T/100
            old,d0=module._differential_subspace_decouple(points.copy(),cov,positions,1,(0,360),(0,90))
            new,d1=_differential_subspace_decouple(points.copy(),cov,positions,1,(0,360),(0,90))
            assert np.all(np.isfinite(new)) and np.all((new[:,0]>=0)&(new[:,0]<360)) and np.all((new[:,1]>=0)&(new[:,1]<=90))
            def f(arr):
                A=_steering(arr,positions,1)
                return float(1-np.trace((A@np.linalg.pinv(A))@cov).real/np.trace(cov).real)
            assert f(new) <= f(points)+1e-10
            results.append(dict(geometry=name,seed=t,old_applied=bool(d0.get("applied")),
                                new_applied=bool(d1.get("applied")),changed=bool(not np.allclose(old,new,atol=1e-8)),
                                old_loss=f(old),new_loss=f(new)))
    with (out/"directed_dsd.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    for name in truth_cases:
        x=[r for r in results if r["geometry"]==name]
        print(f"DSD {name:17s}: changed {sum(z['changed'] for z in x)}/{len(x)} applied_new {sum(z['new_applied'] for z in x)}",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trials",type=int,default=5)
    p.add_argument("--out",type=Path,default=Path("audit/r205_dsd"))
    args=p.parse_args()
    run_family_audit(args.trials,args.out)
    run_directed_boundary(args.out)
