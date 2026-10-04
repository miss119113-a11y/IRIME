"""Paired original RIME-ML / IRIME test on five frozen ablation conditions.

Run from the repository root: python -m experiments.run_rime_ablation
The default protocol is 500 trials per condition. Parallel execution is for
accuracy only: recorded wall times are not a serial performance benchmark.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse, csv, hashlib, json, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from irime import irime
from experiments.rime_ml import rime_ml, RawDML, OFFICIAL_COMMIT
from experiments import paper_protocol as P

CELLS = (("snr",-5,"-5dB_K3"),
         ("resolution",(1.5,10),"10dB_Delta1.5"),
         ("resolution",(1.0,20),"20dB_Delta1"),
         ("sources",8,"K8"), ("sources",9,"K9"))
FIELDS = ("condition","family","local_trial","trial","method","seed",
          "observation_digest","K","L","snr_db","status","failure_message",
          "theta_true_json","phi_true_json","theta_est_json","phi_est_json",
          "spherical_errors_json","spherical_rmse_deg","joint_rmse_deg","sr3",
          "strict_resolution","raw_dml_loss","runtime_ms","diagnostics_json")


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def one(task):
    family,value,label,t,population,iterations=task
    if family=="snr":data=P.main_observation(family,value,t)
    elif family=="sources":data=P.source_observation(value,t)
    else:data=P.resolution_observation(*value,t)
    cfg,pos,s,x,irime_seed,sigma,trial=data
    observation_digest=digest(pos,x)
    baseline_seed=P.stable_seed("RIME-ML-CANONICAL-20261004",family,label,trial,"rime_ml")
    rows=[]
    for method,seed in (("RIME-ML",baseline_seed),("IRIME",irime_seed)):
        theta,phi=np.array([]),np.array([])
        status,failure,diagnostics="ok","",{}
        start=time.perf_counter()
        try:
            if method=="RIME-ML":
                theta,phi,diagnostics=rime_ml(x,pos,1,cfg['source_num'],seed=seed,
                    population=population,iterations=iterations,return_diagnostics=True)
            else:
                theta,phi=irime(x,pos,1,cfg['theta_range'],cfg['phi_range'],cfg['source_num'],
                               seed=seed,h=P.H,B=P.B)
        except Exception as e:
            status,failure="failed",f"{type(e).__name__}: {e}"
        elapsed=(time.perf_counter()-start)*1000
        sph=P.spherical_match(cfg['theta_true'],cfg['phi_true'],theta,phi)
        ang=P.pair_doa_errors(cfg['theta_true'],cfg['phi_true'],theta,phi)
        strict=P.resolution_metrics(cfg,theta,phi)['strict_resolution_success'] if family=='resolution' else ''
        rawloss=RawDML(x,pos,1,cfg['source_num'])(np.r_[theta,phi])[0] if status=='ok' else ''
        rows.append(dict(zip(FIELDS,(label,family,t,trial,method,seed,observation_digest,
            cfg['source_num'],cfg['snapshot'],cfg['snr'],status,failure,
            json.dumps(cfg['theta_true'].tolist()),json.dumps(cfg['phi_true'].tolist()),
            json.dumps(theta.tolist()),json.dumps(phi.tolist()),json.dumps(sph['errors_deg'].tolist()),
            sph['trial_rmse_deg'],ang['joint_rmse'],sph['complete_3deg'],strict,
            rawloss,elapsed,json.dumps(diagnostics,sort_keys=True)))))
    return rows


def write_rows(path,rows):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        w.writerows(sorted(rows,key=lambda r:(r['condition'],int(r['local_trial']),r['method'])))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=500)
    parser.add_argument('--population',type=int,default=50)
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--out',type=Path,default=Path('results/rime_ablation'))
    args=parser.parse_args()
    if min(args.trials,args.iterations,args.workers)<1 or args.population<2:
        parser.error('Require positive trials/iterations/workers and population >= 2')
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'trials.csv'; manifest_path=args.out/'MANIFEST.json'
    script=Path(__file__).resolve(); root=script.parent.parent
    config={'protocol':'RIME-ML-CANONICAL-20261004','trials_per_cell':args.trials,
            'population':args.population,'iterations':args.iterations,'W':5,
            'joint_objective_evaluations':args.population*(args.iterations+1),
            'cells':[{'family':f,'value':v,'condition':c} for f,v,c in CELLS],
            'irime_h':P.H,'irime_B_per_conditional_call':P.B,
            'official_rime_commit':OFFICIAL_COMMIT,
            'code_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                ('irime.py','experiments/rime_ml.py','experiments/paper_protocol.py','experiments/run_rime_ablation.py')},
            'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'platform':platform.platform(),'workers':args.workers,
            'runtime_scope':'Contended local wall times; not a serial performance benchmark',
            'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'seed_policy':'Frozen observation and IRIME seeds; independent deterministic baseline role',
            'metric_policy':'No trial discarded; frozen Hungarian matching and failure penalties',
            'comparison_scope':'Whole-framework reference, not isolated conditional-kernel ablation',
            'fixed_configuration_before_formal_run':True}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for k in config:
            if previous[k]!=config[k]:raise ValueError(f'Resume configuration changed: {k}')
        config=previous
    else:
        config['started_utc']=datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(config,indent=2)+'\n')
    rows=list(csv.DictReader(path.open(newline='',encoding='utf-8'))) if path.exists() else []
    complete={(r['condition'],int(r['local_trial'])) for r in rows if r['method']=='IRIME'}
    tasks=[(f,v,c,t,args.population,args.iterations) for f,v,c in CELLS
           for t in range(args.trials) if (c,t) not in complete]
    print(f'Frozen config: {args.trials} x {len(CELLS)} paired trials; '
          f'{args.population*(args.iterations+1)} raw-DML evaluations per RIME-ML run',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            rows.extend(future.result())
            if i%20==0:write_rows(path,rows)
            if i%50==0:print(f'{len(rows)//2}/{args.trials*len(CELLS)} observation pairs complete',flush=True)
    write_rows(path,rows)
    config['completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(config,indent=2)+'\n')
    print('All requested observation pairs complete',flush=True)

if __name__=='__main__':main()
