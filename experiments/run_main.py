"""Run the proposed estimator's frozen principal cells, with resumable CSV output."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse, csv, hashlib, json, platform, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from irime import irime
from experiments import paper_protocol as P

FIELDS=("family","condition","local_trial","trial","method_seed","observation_digest","K","L","snr_db","status","failure_message","theta_true_json","phi_true_json","theta_est_json","phi_est_json","joint_rmse_deg","spherical_rmse_deg","sr3","strict_resolution","runtime_ms","crb_joint_deg")

def digest(*arrays):
    h=hashlib.sha256()
    for array in arrays:
        a=np.ascontiguousarray(array)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest().upper()

def cells(family):
    if family=="snr":return [(x,f"{x:g}dB") for x in P.SNR_VALUES]
    if family=="snapshots":return [(x,f"L{x}") for x in P.SNAPSHOT_VALUES]
    if family=="sources":return [(x,f"K{x}") for x in P.SOURCE_COUNTS]
    if family=="resolution":return [((d,s),f"{s}dB_Delta{d:g}") for s in (10,20) for d in P.MAIN_SPACINGS]
    return [(0,"K3_L100_0dB")]

def one(family,value,condition,local_trial):
    if family in ("snr","snapshots"):data=P.main_observation(family,value,local_trial)
    elif family=="sources":data=P.source_observation(int(value),local_trial)
    elif family=="resolution":data=P.resolution_observation(value[0],value[1],local_trial)
    else:data=P.efficiency_observation(local_trial)
    cfg,pos,s,x,seed,sigma2,trial=data
    status,failure="ok",""
    theta,phi=np.array([]),np.array([])
    started=time.perf_counter()
    try:
        theta,phi=irime(x,pos,1.0,cfg["theta_range"],cfg["phi_range"],cfg["source_num"],h=P.H,B=P.B,seed=seed)
    except Exception as e:
        status,failure="failed",f"{type(e).__name__}: {e}"
    elapsed=(time.perf_counter()-started)*1000
    ang=P.pair_doa_errors(cfg["theta_true"],cfg["phi_true"],theta,phi)
    sph=P.spherical_match(cfg["theta_true"],cfg["phi_true"],theta,phi)
    strict=""
    if family=="resolution":strict=P.resolution_metrics(cfg,theta,phi)["strict_resolution_success"]
    crb=P.conditional_crb(cfg["theta_true"],cfg["phi_true"],s,pos,1.0,sigma2)["crb_joint_angular_deg"]
    return dict(zip(FIELDS,(family,condition,local_trial,trial,seed,digest(pos,x),cfg["source_num"],cfg["snapshot"],cfg["snr"],status,failure,json.dumps(cfg["theta_true"].tolist()),json.dumps(cfg["phi_true"].tolist()),json.dumps(theta.tolist()),json.dumps(phi.tolist()),ang["joint_rmse"],sph["trial_rmse_deg"],sph["complete_3deg"],strict,elapsed,crb)))

def write_rows(path,rows,fields):
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['family'],r['condition']].append(r)
    out=[]
    for (family,condition),rs in sorted(groups.items()):
        ang=np.array([float(r['joint_rmse_deg']) for r in rs]);sph=np.array([float(r['spherical_rmse_deg']) for r in rs])
        strict=[float(r['strict_resolution']) for r in rs if r['strict_resolution']!='']
        crb=np.array([float(r['crb_joint_deg']) for r in rs])
        out.append({'family':family,'condition':condition,'trials':len(rs),'failed_trials':sum(r['status']!='ok' for r in rs),'pooled_joint_rmse_deg':float(np.sqrt(np.mean(ang**2))),'pooled_spherical_rmse_deg':float(np.sqrt(np.mean(sph**2))),'median_spherical_rmse_deg':float(np.median(sph)),'p95_spherical_rmse_deg':float(np.percentile(sph,95)),'sr3':float(np.mean([float(r['sr3']) for r in rs])),'p_rmse_gt5':float(np.mean(sph>5)),'p_rmse_gt10':float(np.mean(sph>10)),'strict_resolution':float(np.mean(strict)) if strict else '', 'mean_runtime_ms':float(np.mean([float(r['runtime_ms']) for r in rs])),'crb_joint_deg':float(np.sqrt(np.mean(crb**2))) if np.isfinite(crb).all() else float('nan')})
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('snr','snapshots','sources','resolution','efficiency','all'),default='all')
    parser.add_argument('--trials',type=int,default=500)
    parser.add_argument('--trial-start',type=int,default=0,help='Local offset within the frozen trial sequence.')
    parser.add_argument('--out',type=Path,default=Path('results/main'))
    parser.add_argument('--condition',help='Run one exact printed condition label, e.g. K9 or 10dB_Delta1.5.')
    args=parser.parse_args()
    if args.trials<1 or args.trial_start<0:parser.error('Require trials >= 1 and trial-start >= 0.')
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=args.out/'trials.csv'
    rows=[]
    if checkpoint.exists():
        with checkpoint.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    keys={(r['family'],r['condition'],int(r['local_trial'])) for r in rows}
    families=('snr','snapshots','sources','resolution','efficiency') if args.family=='all' else (args.family,)
    matched=0
    # Warm up the estimator outside the measured interval for the timing family.
    if 'efficiency' in families:
        for i in range(5):one('efficiency',0,'warmup',-1000-i)
    for family in families:
        for value,condition in cells(family):
            if args.condition and args.condition!=condition:continue
            matched+=1
            for t in range(args.trial_start,args.trial_start+args.trials):
                if (family,condition,t) in keys:continue
                row=one(family,value,condition,t);rows.append(row);keys.add((family,condition,t))
                if len(rows)%10==0:write_rows(checkpoint,rows,FIELDS)
            write_rows(checkpoint,rows,FIELDS)
            print(f'{family}/{condition}: {args.trials} requested trials complete',flush=True)
    if not matched:parser.error('No cell matches --condition.')
    summary=summarize(rows);write_rows(args.out/'summary.csv',summary,list(summary[0]))
    manifest={'method':'IRIME','implementation_sha256':hashlib.sha256(Path('irime.py').read_bytes()).hexdigest(),'h':P.H,'B':P.B,'families':families,'requested_trials_per_cell':args.trials,'local_trial_start':args.trial_start,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'timing_excludes_observation_generation':True,'scope':'Proposed estimator only. This minimal runner does not regenerate the six comparative baselines.'}
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
