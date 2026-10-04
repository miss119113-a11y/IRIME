"""Audit and summarize paired raw records; bootstrap always resamples pairs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments.run_rime_ablation import CELLS


def wilson(successes, n):
    z=1.959963984540054; p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,float(center-radius)),min(1.,float(center+radius))]


def summarize(folder, *, bootstrap=10000):
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    rows=list(csv.DictReader((folder/'trials.csv').open(newline='',encoding='utf-8')))
    expected=int(manifest['trials_per_cell'])
    if len(rows)!=2*len(CELLS)*expected:raise ValueError('Incomplete or duplicate trial count')
    result={'paired_trials_per_condition':expected,'bootstrap_replicates':bootstrap,
            'confidence_intervals':'95% per-condition intervals, without multiplicity correction',
            'resampling_unit':'Observation pair; all K source errors remain together',
            'conditions':[]}
    for i,(family,value,condition) in enumerate(CELLS):
        groups={m:sorted([r for r in rows if r['condition']==condition and r['method']==m],
                        key=lambda r:int(r['local_trial'])) for m in ('RIME-ML','IRIME')}
        for m,rs in groups.items():
            if [int(r['local_trial']) for r in rs]!=list(range(expected)):
                raise ValueError(f'Incomplete/duplicate pair sequence in {condition}/{m}')
        for a,b in zip(groups['RIME-ML'],groups['IRIME']):
            for key in ('observation_digest','theta_true_json','phi_true_json','K','L','snr_db','trial'):
                if a[key]!=b[key]:raise ValueError(f'Pair mismatch: {condition}/{key}')
            if a['status']=='ok':
                d=json.loads(a['diagnostics_json'])
                if d['objective_evaluations']!=manifest['joint_objective_evaluations']:
                    raise ValueError('Objective evaluation count mismatch')
        rng=np.random.default_rng(20261004+i)
        indices=rng.integers(expected,size=(bootstrap,expected))
        bymethod={}
        squared={};success={}
        for m,rs in groups.items():
            e=np.array([float(r['spherical_rmse_deg']) for r in rs]);squared[m]=e**2
            key='strict_resolution' if family=='resolution' else 'sr3'
            success[m]=np.array([float(r[key]) for r in rs])
            boot_rmse=np.sqrt(np.mean(squared[m][indices],axis=1))
            successes=int(np.sum(success[m]))
            bymethod[m]={'spherical_rmse_deg':float(np.sqrt(np.mean(e**2))),
                         'spherical_rmse_ci95_deg':np.quantile(boot_rmse,[.025,.975]).tolist(),
                         'trial_median_deg':float(np.median(e)),
                         'trial_p95_deg':float(np.percentile(e,95)),
                         'trial_max_deg':float(np.max(e)),
                         'p_trial_rmse_gt5':float(np.mean(e>5)),
                         'success_metric':key,'success_count':successes,
                         'success_rate':float(successes/expected),
                         'success_wilson_ci95':wilson(successes,expected),
                         'execution_failures':sum(r['status']!='ok' for r in rs)}
        difference=np.sqrt(np.mean(squared['RIME-ML'][indices],axis=1))-np.sqrt(np.mean(squared['IRIME'][indices],axis=1))
        boot_success=np.mean((success['IRIME']-success['RIME-ML'])[indices],axis=1)
        n10=int(np.sum((success['IRIME']==1)&(success['RIME-ML']==0)))
        n01=int(np.sum((success['IRIME']==0)&(success['RIME-ML']==1)))
        result['conditions'].append({'condition':condition,'methods':bymethod,
            'paired_rmse_difference_RIME_minus_IRIME_deg':bymethod['RIME-ML']['spherical_rmse_deg']-bymethod['IRIME']['spherical_rmse_deg'],
            'paired_rmse_difference_ci95_deg':np.quantile(difference,[.025,.975]).tolist(),
            'paired_success_difference_IRIME_minus_RIME':float(np.mean(success['IRIME']-success['RIME-ML'])),
            'paired_success_difference_ci95':np.quantile(boot_success,[.025,.975]).tolist(),
            'discordant_IRIME_only':n10,'discordant_RIME_only':n01,
            'mcnemar_exact_two_sided_uncorrected_p':float(binomtest(n10,n10+n01,.5).pvalue) if n10+n01 else 1.0})
    result['all_observations_paired_exactly']=True
    (folder/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=('condition','method','spherical_rmse_deg','trial_median_deg','trial_p95_deg',
            'trial_max_deg','p_trial_rmse_gt5','success_metric','success_count','success_rate','execution_failures')
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in result['conditions']:
            for m,d in c['methods'].items():w.writerow({'condition':c['condition'],'method':m,**{k:d[k] for k in fields[2:]}})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--bootstrap',type=int,default=10000);args=p.parse_args()
    result=summarize(args.folder,bootstrap=args.bootstrap)
    for c in result['conditions']:
        print(c['condition'],json.dumps(c['methods']),flush=True)
