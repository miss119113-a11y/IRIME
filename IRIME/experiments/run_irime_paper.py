#!/usr/bin/env python3
"""IRIME-only Monte Carlo reproduction; no competing algorithms are imported.

These scripts regenerate simulations using the method parameters stated in the
final manuscript. Recorded raw outcomes depend on seeds and numerical versions.
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'01_algorithms'))
from irime_v7_compact import irime_v7_compact
from irime_v7_compact_switchable import irime_v7_switchable
from joint_rime_p0 import direct_joint_rime_p0
M=12; WAVELENGTH=1.0
R=WAVELENGTH/(4*np.sin(np.pi/M))
POS=np.column_stack((R*np.cos(2*np.pi*np.arange(M)/M),R*np.sin(2*np.pi*np.arange(M)/M),np.zeros(M)))
THETA_MAIN=np.array([30.42,47.00,150.27])
PHI_MAIN=np.array([60.39,46.27,35.55])
SNRS=[-5,-3,-1,0,1,3,5,7,10,12,15]
SNAPS=[30,50,75,100,200,300,500]
SPACINGS=[0.25,0.5,1,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4,5,6,8,10]

def dirs_random(k,rng):
    alpha,beta=rng.uniform(0,360),rng.uniform(0,1)
    i=np.arange(k)
    return (alpha+360*i/k)%360,20+50*((beta+i*(np.sqrt(5)-1)/2)%1)

def dirs_close(sep):
    th,ph=83.17,35.29
    cphi=np.cos(np.deg2rad(ph))**2
    dth=np.rad2deg(np.arccos(np.clip((np.cos(np.deg2rad(sep))-(1-cphi))/cphi,-1,1)))
    return np.array([th,(th+dth)%360]),np.array([ph,ph])

def simulate(theta,phi,L,snr_db,per_source=False,gaussian=False,seed=1):
    rng=np.random.default_rng(seed)
    k=len(theta)
    t=np.arange(L)
    manifold=np.column_stack([np.exp(-2j*np.pi*(POS@np.array([np.cos(np.deg2rad(p))*np.cos(np.deg2rad(a)),np.cos(np.deg2rad(p))*np.sin(np.deg2rad(a)),np.sin(np.deg2rad(p))]))/WAVELENGTH) for a,p in zip(theta,phi)])
    if gaussian:
        S=(rng.standard_normal((k,L))+1j*rng.standard_normal((k,L)))/np.sqrt(2)
        S/=np.sqrt(np.mean(np.abs(S)**2,axis=1,keepdims=True))
    else:
        frequencies=np.array([0.1,0.2,0.3]) if k==3 else (np.arange(k)+1)/(2*k)
        S=np.exp(2j*np.pi*frequencies[:,None]*t[None,:])
    clean=manifold@S
    p=1 if per_source else np.mean(np.abs(clean)**2)
    noise_var=p/10**(snr_db/10)
    noise=np.sqrt(noise_var/2)*(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))
    return clean+noise

def angular_errors(theta,phi,est_th,est_ph):
    theta=np.asarray(theta);phi=np.asarray(phi);est_th=np.asarray(est_th);est_ph=np.asarray(est_ph)
    if len(theta)!=len(est_th) or not np.isfinite(est_th).all() or not np.isfinite(est_ph).all():
        return np.full(len(theta),180.),np.full(len(theta),90.),np.full(len(theta),180.)
    dt=((est_th[None,:]-theta[:,None]+180)%360)-180
    dp=est_ph[None,:]-phi[:,None]
    rows,cols=linear_sum_assignment(dt**2+dp**2)
    dt=dt[rows,cols];dp=dp[rows,cols]
    a=np.deg2rad(theta[rows]);b=np.deg2rad(phi[rows]);x=np.deg2rad(est_th[cols]);y=np.deg2rad(est_ph[cols])
    cos_arc=np.sin(b)*np.sin(y)+np.cos(b)*np.cos(y)*np.cos(x-a)
    sphere=np.rad2deg(np.arccos(np.clip(cos_arc,-1,1)))
    return dt,dp,sphere

def estimated_pair_separation_deg(est_th, est_ph):
    """Great-circle angular separation of two estimated source directions."""
    if len(est_th) != 2 or len(est_ph) != 2:
        return float('nan')
    if not np.isfinite(est_th).all() or not np.isfinite(est_ph).all():
        return float('nan')
    a, b = np.deg2rad(np.asarray(est_th, dtype=float))
    x, y = np.deg2rad(np.asarray(est_ph, dtype=float))
    cos_sep = np.sin(x)*np.sin(y) + np.cos(x)*np.cos(y)*np.cos(a-b)
    return float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))

def evaluate_once(k,L,snr,trial,kind='deterministic',h=2.5,variant='full',sep=None):
    seed=20261028+trial
    geo=np.random.default_rng(seed)
    if sep is not None:theta,phi=dirs_close(sep)
    elif kind=='random':theta,phi=dirs_random(k,geo)
    else:theta,phi=THETA_MAIN.copy(),PHI_MAIN.copy()
    X=simulate(theta,phi,L,snr,per_source=(kind=='random' or sep is not None),gaussian=(kind=='random' or sep is not None),seed=seed)
    kwargs=dict(seed=seed,coarse_evidence_step_deg=h,continuous_search_budget_per_coordinate=896)
    t0=time.perf_counter()
    if variant=='full':e_t,e_p=irime_v7_compact(X,POS,WAVELENGTH,[0,360],[0,90],k,**kwargs)
    elif variant=='p0':e_t,e_p=direct_joint_rime_p0(X,POS,WAVELENGTH,k,seed=seed)
    else:
        features={'I':'I' in variant,'D':'D' in variant,'J':'J' in variant}
        e_t,e_p=irime_v7_switchable(X,POS,WAVELENGTH,[0,360],[0,90],k,enable_rank_stabilization=features['I'],enable_dsd=features['D'],enable_terminal_controller=features['J'],**kwargs)
    runtime=time.perf_counter()-t0
    da,de,sph=angular_errors(theta,phi,e_t,e_p)
    rmse=float(np.sqrt(np.mean(da**2+de**2)))
    sr3=float(np.all(sph<3.0))
    # Strict close-source resolution criterion in the manuscript:
    # both assignment-matched spherical errors < 0.49 Delta AND
    # estimated pair separation >= 0.5 Delta.
    estimated_sep = estimated_pair_separation_deg(e_t,e_p) if sep is not None else ''
    strict = (int(bool(np.all(np.asarray(sph)<0.49*float(sep))
                       and np.isfinite(estimated_sep)
                       and estimated_sep>=0.5*float(sep)))
              if sep is not None else '')
    return dict(trial=trial,seed=seed,k=k,L=L,snr_db=snr,h_deg=h,
                variant=variant,separation_deg='' if sep is None else sep,
                rmse_deg=rmse,complete_sr3=sr3,strict_resolution=strict,
                estimated_separation_deg=estimated_sep,runtime_s=runtime)

def tasks(scenario):
    if scenario=='snr':return [(3,300,v,'deterministic',2.5,'full',None) for v in SNRS]
    if scenario=='snapshots':return [(3,v,0.,'deterministic',2.5,'full',None) for v in SNAPS]
    if scenario=='sources':return [(k,300,10.,'random',2.5,'full',None) for k in range(2,10)]
    if scenario=='resolution':return [(2,1024,10.,'deterministic',2.5,'full',sep) for sep in SPACINGS]
    if scenario=='sensitivity':return [(k,L,snr,'random' if k>2 else 'deterministic',h,'full',sep) for k,L,snr,sep in [(2,1024,10.,1.5),(2,1024,20.,1.),(7,300,10.,None)] for h in (2.,2.5,3.,3.5,4.)]
    if scenario=='timing':return [(3,300,0.,'deterministic',2.5,'full',None)]
    if scenario=='ablation':return [(k,L,snr,'random' if k>2 else 'deterministic',2.5,v,sep) for k,L,snr,sep in [(9,300,10.,None),(2,1024,10.,1.5),(2,1024,20.,0.5),(2,1024,20.,0.75)] for v in ['p0','I','D','J','ID','IJ','DJ','full']]
    raise ValueError(scenario)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--scenario',choices=['snr','snapshots','sources','resolution','sensitivity','ablation','timing'],required=True)
    a.add_argument('--trials',type=int,default=500)
    a.add_argument('--smoke',action='store_true',help='one condition, one trial, full algorithm')
    a.add_argument('--output',type=Path)
    args=a.parse_args()
    if args.trials<1: a.error('--trials must be positive')
    entries=tasks(args.scenario)
    if args.smoke:entries=entries[:1]
    n=1 if args.smoke else args.trials
    target=args.output or ROOT/'results'/f'{args.scenario}_irime_only.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='',encoding='utf-8') as f:
        w=None
        for case in entries:
            for t in range(n):
                row=evaluate_once(*case[:3],t,*case[3:])
                if w is None:w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader()
                w.writerow(row)
                f.flush()
                print(args.scenario,case,'trial',t,'RMSE',round(row['rmse_deg'],5))
    print('Saved:',target)
if __name__=='__main__':main()
