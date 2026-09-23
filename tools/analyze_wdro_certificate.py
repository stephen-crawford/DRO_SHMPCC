#!/usr/bin/env python3
"""Offline, single-obstacle CP risk-transfer diagnostic; never changes controller settings."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def vertices(lower,upper):
    result=set();m=len(lower)
    for free in range(m):
        other=[i for i in range(m) if i!=free]
        for ends in itertools.product([0,1],repeat=m-1):
            p=[0.]*m
            for i,end in zip(other,ends):p[i]=upper[i] if end else lower[i]
            p[free]=1.-sum(p)
            if lower[free]-1e-12<=p[free]<=upper[free]+1e-12:
                # Numerical diagnostic tolerance, not a proof of exact containment.
                result.add(tuple(p))
    if not result:raise ValueError('empty CP box/simplex intersection')
    return sorted(result)


def transfer_at(p,q,b,eta):
    # Fractional knapsack solves this one-budget LP, including Q-null modes.
    value=sum(pi*bi for pi,qi,bi in zip(p,q,b) if qi==0)
    remaining=eta
    for i in sorted((i for i in range(len(q)) if q[i]>0),key=lambda i:p[i]/q[i],reverse=True):
        v=min(b[i],max(0.,remaining)/q[i]);value+=p[i]*v;remaining-=q[i]*v
    return value


def invert(vv,q,b,epsilon):
    psi=lambda eta:max(transfer_at(p,q,b,eta) for p in vv)
    if psi(0)>epsilon:return None,psi
    if psi(1)<=epsilon:return 1.,psi
    lo,hi=0.,1.
    for _ in range(64):
        mid=(lo+hi)/2
        if psi(mid)<=epsilon:lo=mid
        else:hi=mid
    return lo,psi


def analyze(snapshot,backend):
    if snapshot.get('obstacles')!=1:raise ValueError('initial diagnostic supports one obstacle; joint-product confidence regions are not implemented')
    modes=snapshot['modes'];m=len(modes)
    if not 2<=m<=6:raise ValueError('expected two to six modes')
    p=[x['p_hat'] for x in modes];q=[x['q'] for x in modes]
    counts=[x['count'] for x in modes];b=[x.get('b',1.) for x in modes]
    for name,law in [('p_hat',p),('q',q)]:
        if any(not math.isfinite(x) or x<0 for x in law) or abs(sum(law)-1)>1e-10:raise ValueError('invalid '+name)
    if any(type(n) is not int or n<0 for n in counts):raise ValueError('invalid categorical counts')
    if any(not math.isfinite(x) or not 0<=x<=1 for x in b):raise ValueError('invalid mode upper bounds')
    if any(x<1 for x in b) and not snapshot.get('b_justification'):raise ValueError('b<1 requires a stated independent justification; risk scores are not certified bounds')
    epsilon=snapshot.get('epsilon',.05);beta_cp=snapshot.get('beta_cp',.05);beta_cert=snapshot.get('beta_cert',.01)
    if not 0<epsilon<1 or not 0<beta_cp<1 or not 0<beta_cert<1:raise ValueError('invalid risk/confidence')
    cap=snapshot.get('nonremoved_support_cap',6);removal=snapshot.get('removal_budget',2)
    if type(cap) is not int or type(removal) is not int or min(cap,removal)<0:raise ValueError('invalid support accounting')
    def call(*args):return subprocess.check_output([str(backend),*map(str,args)],text=True)
    intervals=[list(map(float,line.split())) for line in call('intervals',beta_cp,*counts).splitlines()]
    lower,upper=map(list,zip(*intervals));vv=vertices(lower,upper)
    eta,psi=invert(vv,q,b,epsilon)
    def size(risk):
        if risk is None or risk<=0:return None
        s,actual,previous=map(float,call('size',risk,beta_cert,cap,removal).split())
        return dict(S=int(s),risk=actual,previous_S_risk=previous)
    baseline=size(epsilon);wdro=size(eta)
    for i,row in enumerate(modes):row.update(L=lower[i],U=upper[i],b=b[i])
    ratio=wdro['S']/baseline['S'] if wdro else None
    return dict(status='offline_numerical_diagnostic_not_an_issued_certificate',modes=modes,
        rho=snapshot.get('rho'),observed_support=snapshot.get('observed_support'),
        observation_count=sum(counts),cp_vertices=len(vv),epsilon=epsilon,epsilon_Q_max=eta,
        psi_at_epsilon=psi(epsilon),psi_at_zero=psi(0),psi_at_epsilon_Q_max=psi(eta) if eta is not None else None,
        S_SH=baseline,S_WDRO=wdro,scenario_count_ratio=ratio,go_30_percent=ratio is not None and ratio<=.7,
        nonremoved_support_cap=cap,removal_budget=removal,total_support_cap=cap+removal,
        beta_cp=beta_cp,beta_cert=beta_cert,combined_failure_union_bound=min(1.,beta_cp+beta_cert),
        b_justification=snapshot.get('b_justification','v_m <= 1 only'),
        assumptions='Conditional on frozen pre-sampling information: fixed q, iid scenario draws, identical within-mode conditional laws under P* and Q; iid stationary categorical history for CP. Scenario theorem/support hypotheses must separately hold. No joint-obstacle independence assumption is silently imposed.',
        limitation='LP/vertex enumeration and inversion use floating point, not verified interval arithmetic. CP event adds beta_cp to beta_cert; S_SH uses beta_cert alone. Surrogate r is not b. Counts from switching/shared/adaptive histories do not automatically meet iid CP assumptions. No observed support value is inferred if absent.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot',type=Path,help='JSON with one obstacle, mode counts/p_hat/q and optional justified b')
    source.add_argument('--frozen-weights',type=Path,help='weights.csv from the saved production rare-mode fixture; uses conservative b=1')
    parser.add_argument('--backend',type=Path,default=ROOT/'build-certificate/certificate_numeric_backend')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('use a new output directory')
    path=args.snapshot or args.frozen_weights
    if args.snapshot:snapshot=json.loads(path.read_text())
    else:
        with path.open() as f:rows=list(csv.DictReader(f))
        snapshot=dict(obstacles=1,modes=[dict(mode=r['mode'],count=int(r['count']),p_hat=float(r['nominal_probability']),q=float(r['wdro_probability'])) for r in rows],rho=float(rows[0]['rho']),
                      source='production frozen-reference rare-mode diagnostic; not a closed-loop certified control')
    result=analyze(snapshot,args.backend.resolve())
    args.output.mkdir(parents=True)
    (args.output/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    (args.output/'certificate_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'mode_bounds.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in result['modes'] for k in r)));w.writeheader();w.writerows(result['modes'])
    summary={k:result[k] for k in ['status','epsilon','epsilon_Q_max','scenario_count_ratio','go_30_percent',
                                   'rho','observed_support','observation_count','total_support_cap',
                                   'beta_cp','beta_cert','combined_failure_union_bound']}
    summary.update(S_SH=result['S_SH']['S'],S_WDRO=result['S_WDRO']['S'] if result['S_WDRO'] else None)
    with (args.output/'certificate_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary));w.writeheader();w.writerow(summary)
    provenance=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        backend_sha256=hashlib.sha256(args.backend.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['epsilon_Q_max','S_SH','S_WDRO','scenario_count_ratio','go_30_percent']}))

if __name__=='__main__':main()
