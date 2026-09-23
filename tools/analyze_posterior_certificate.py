#!/usr/bin/env python3
"""A posteriori Gaussian geometry bounds for a fixed plan. No pre-solve S claim."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from analyze_wdro_certificate import vertices, transfer_at

ROOT=Path(__file__).resolve().parents[1]


def projected_probability(center,mu,cov,radius):
    center=np.asarray(center,dtype=float);mu=np.asarray(mu,dtype=float);cov=np.asarray(cov,dtype=float)
    if center.shape!=(2,) or mu.shape!=(2,) or cov.shape!=(2,2):raise ValueError('expected 2D positions and 2x2 covariance')
    if not all(np.all(np.isfinite(a)) for a in [center,mu,cov]) or not math.isfinite(radius) or radius<=0:raise ValueError('nonfinite geometry or invalid radius')
    if not np.array_equal(cov,cov.T) or np.linalg.eigvalsh(cov)[0]<0:raise ValueError('covariance must be symmetric positive semidefinite')
    diff=mu-center;distance=math.hypot(*diff)
    direction=diff/distance if distance>0 else np.array([1.,0.])
    mean=float(radius-direction@diff);variance=float(direction@cov@direction)
    if variance<0:raise ValueError('negative projected variance')
    # Strict event ||X-center|| < R implies R-n.(X-center)>0 for any unit n.
    probability=float(mean>0) if variance==0 else .5*math.erfc(-mean/math.sqrt(2*variance))
    return dict(distance=distance,projected_mean=mean,projected_variance=variance,
                probability_upper=probability,zero_distance=int(distance==0),zero_variance=int(variance==0))


def evaluate(snapshot,backend):
    if snapshot.get('obstacles')!=1:raise ValueError('only one obstacle is supported')
    modes=snapshot['modes'];steps=snapshot['steps'];radius=snapshot['collision_radius']
    names=[m['mode'] for m in modes]
    if not 2<=len(modes)<=6 or len(set(names))!=len(names) or not steps:raise ValueError('invalid modes or empty horizon')
    if any('q' in m for m in modes) and not all('q' in m for m in modes):raise ValueError('incomplete q law')
    for law in ['p_hat'] + (['q'] if all('q' in m for m in modes) else []):
        values=[m[law] for m in modes]
        if any(not math.isfinite(x) or x<0 for x in values) or abs(sum(values)-1)>1e-10:raise ValueError('invalid '+law)
    if any(type(m['count']) is not int or m['count']<0 for m in modes):raise ValueError('invalid observation count')
    epsilon=snapshot.get('epsilon',.05);beta_cp=snapshot.get('beta_cp',.05);beta_cert=snapshot.get('beta_cert',.01)
    if any(not 0<x<1 for x in [epsilon,beta_cp,beta_cert]):raise ValueError('invalid risk/confidence')
    rows=[];bounds=[]
    for mode in modes:
        total=0.
        for step in steps:
            if set(step['gaussians'])!=set(names) or not step['disc_centers']:raise ValueError('incomplete geometry')
            gaussian=step['gaussians'][mode['mode']]
            for disc,center in enumerate(step['disc_centers']):
                row=projected_probability(center,gaussian['mean'],gaussian['covariance'],radius)
                total+=row['probability_upper']
                rows.append(dict(mode=mode['mode'],step=step['k'],disc=disc,**row))
        bounds.append(min(1.,total))
    raw=subprocess.check_output([str(backend),'intervals',str(beta_cp),*[str(m['count']) for m in modes]],text=True)
    intervals=[list(map(float,line.split())) for line in raw.splitlines()]
    lower,upper=map(list,zip(*intervals));vv=vertices(lower,upper)
    q=[m.get('q') for m in modes]
    direct=max(sum(p*b for p,b in zip(v,bounds)) for v in vv)
    eta=None;transferred=None
    if snapshot.get('scenario_theorem_eligible') is True:
        if any(x is None for x in q):raise ValueError('actual sampling q is required for transfer')
        if not snapshot.get('scenario_theorem_justification'):raise ValueError('scenario-theorem eligibility needs justification')
        s=snapshot['sample_count'];support=snapshot['total_support_bound']
        if type(s) is not int or type(support) is not int or s<=0 or support<0:raise ValueError('invalid sample/support count')
        eta=float(subprocess.check_output([str(backend),'risk',str(s),str(support),str(beta_cert)],text=True))
        transferred=max(transfer_at(v,q,bounds,eta) for v in vv)
    mode_rows=[dict(m,b=bounds[i],L=lower[i],U=upper[i]) for i,m in enumerate(modes)]
    result=dict(status='conditional_offline_posterior_diagnostic_not_issued_certificate',
        epsilon=epsilon,collision_radius=radius,event='strict disc overlap at supplied discrete future steps',
        mode_bounds=mode_rows,direct_cp_bound=direct,eta_SH=eta,transferred_bound=transferred,
        direct_bound_meets_target=direct<=epsilon,
        transferred_bound_meets_target=transferred<=epsilon if transferred is not None else None,
        scenario_theorem_eligible=snapshot.get('scenario_theorem_eligible',False),
        sample_count=snapshot.get('sample_count'),total_support_bound=snapshot.get('total_support_bound'),
        beta_cp=beta_cp,beta_cert=beta_cert,direct_failure_budget=beta_cp,
        transfer_failure_budget=min(1.,beta_cp+beta_cert) if transferred is not None else None,
        presolve_sample_reduction_claim=False,
        assumptions='Fixed returned trajectory; correct held-mode Gaussian marginals conditional on pre-sampling information; same mode-conditioned laws under P* and Q; stationary iid history for CP. No independence across discs/time is required for union bounds. Scenario theorem eligibility must be established separately. Floating point is not a verified numerical enclosure.')
    return result,rows


def write_outputs(snapshot,backend,output):
    if output.exists():raise ValueError('use a new output directory')
    result,rows=evaluate(snapshot,backend)
    output.mkdir(parents=True)
    for name,data in [('snapshot',snapshot),('posterior_certificate',result)]:
        (output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    for name,data in [('mode_bounds',result['mode_bounds']),('disc_step_bounds',rows)]:
        with (output/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader();writer.writerows(data)
    provenance=dict(backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transfer_script_sha256=hashlib.sha256((ROOT/'tools/analyze_wdro_certificate.py').read_bytes()).hexdigest())
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--backend',type=Path,default=ROOT/'build-certificate/certificate_numeric_backend')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=write_outputs(json.loads(args.snapshot.read_text()),args.backend.resolve(),args.output)
    print(json.dumps({k:r[k] for k in ['direct_cp_bound','eta_SH','transferred_bound','presolve_sample_reduction_claim']}))

if __name__=='__main__':main()
