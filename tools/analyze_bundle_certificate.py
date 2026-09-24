#!/usr/bin/env python3
"""Conditional bundle-amplification mathematics; no claim of fewer raw draws.

Snapshot: modes [{mode,count,q,score,K(optional),b(optional)}], epsilon,
beta_cp, beta_cert, c0, extras, nonremoved_support_cap, removal_budget.
Nontrivial b requires b_justification. Outputs are floating-point diagnostics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from analyze_wdro_certificate import vertices


def bundle_violation(r,k):
    if any(ki>0 and ri==1 for ri,ki in zip(r,k)): return 1.
    return -math.expm1(sum(ki*math.log1p(-ri) for ri,ki in zip(r,k) if ki>0))


def vertex_threshold(v,k,b,epsilon):
    """KKT bisection with explicit zero-cost coordinates and boundary cases.

    Dual objective supplies the diagnostic lower threshold; the primal value
    and gap are returned so accuracy is inspectable rather than presumed.
    """
    capacity=sum(vi*bi for vi,bi in zip(v,b))
    if capacity<epsilon: return None
    if capacity==epsilon:
        eta=bundle_violation(b,k)
        return dict(eta=eta,primal_eta=eta,dual_gap=0.,r=b)
    free=sum(vi*bi for vi,ki,bi in zip(v,k,b) if ki==0)
    if free>=epsilon:
        r=[bi*epsilon/free if ki==0 else 0 for ki,bi in zip(k,b)]
        return dict(eta=0.,primal_eta=0.,dual_gap=0.,r=r)
    def risk(lam):
        return [bi if ki==0 and vi>0 else (min(bi,max(0.,1-ki/(lam*vi))) if vi>0 else 0.)
                for vi,ki,bi in zip(v,k,b)]
    lo=0.;hi=1.
    while sum(vi*ri for vi,ri in zip(v,risk(hi)))<epsilon:
        hi*=2
        if not math.isfinite(hi): raise ValueError('KKT bracket overflow')
    for _ in range(120):
        mid=(lo+hi)/2
        if sum(vi*ri for vi,ri in zip(v,risk(mid)))<epsilon:lo=mid
        else:hi=mid
    r=risk(hi)
    if any(ri==1 and ki>0 for ri,ki in zip(r,k)):
        raise ValueError('KKT solution rounded onto logarithmic singularity')
    objective=sum(-ki*math.log1p(-ri) for ki,ri in zip(k,r) if ki>0)
    dual=objective+hi*(epsilon-sum(vi*ri for vi,ri in zip(v,r)))
    return dict(eta=-math.expm1(-max(0.,dual)),primal_eta=-math.expm1(-objective),
                dual_gap=objective-dual,r=r)


def analyze(snapshot,backend):
    modes=snapshot['modes']; n=len(modes)
    if not 2<=n<=6: raise ValueError('expected 2..6 modes')
    epsilon=snapshot.get('epsilon',.05); bc=snapshot.get('beta_cp',.05); bs=snapshot.get('beta_cert',.01)
    if any(not 0<x<1 for x in [epsilon,bc,bs]): raise ValueError('risk/confidence outside (0,1)')
    counts=[m['count'] for m in modes]
    if any(type(c)!=int or c<0 for c in counts) or sum(counts)==0: raise ValueError('invalid counts')
    def call(*args):return subprocess.check_output([str(backend),*map(str,args)],text=True)
    intervals=[list(map(float,line.split())) for line in call('intervals',bc,*counts).splitlines()]
    lower,upper=map(list,zip(*intervals));vv=vertices(lower,upper)
    U=[max(v[i] for v in vv) for i in range(n)]
    b=[m.get('b',1.) for m in modes]
    if any(not math.isfinite(x) or not 0<=x<=1 for x in b):raise ValueError('invalid b')
    if any(x<1 for x in b) and not snapshot.get('b_justification'):raise ValueError('b<1 requires uniform pre-solve justification')
    if any('K' in m for m in modes):
        if not all('K' in m for m in modes): raise ValueError('incomplete K')
        k=[m['K'] for m in modes]
    else:
        c0=snapshot.get('c0',2.);extras=snapshot.get('extras',0)
        if not math.isfinite(c0) or c0<=1 or type(extras)!=int or extras<0:raise ValueError('invalid allocation')
        k=[math.ceil(c0*u) for u in U]
        q=[m['q'] for m in modes];scores=[m.get('score',0.) for m in modes]
        if any(not math.isfinite(x) or x<0 for x in q) or abs(sum(q)-1)>1e-10 or any(not math.isfinite(s) for s in scores):raise ValueError('invalid q/score')
        weights=[qi*max(0.,si) for qi,si in zip(q,scores)];total=sum(weights)
        shares=[extras*w/total if total else extras/n for w in weights]
        whole=[math.floor(x) for x in shares]
        for i in sorted(range(n),key=lambda i:(-(shares[i]-whole[i]),i))[:extras-sum(whole)]:whole[i]+=1
        k=[ki+extra for ki,extra in zip(k,whole)]
    if any(type(x)!=int or x<0 for x in k):raise ValueError('invalid integer multiplicities')
    c=min(ki/u for ki,u in zip(k,U) if u>0)
    simple=-math.expm1(c*math.log1p(-epsilon))
    results=[dict(vertex=list(v),**r) for v in vv if (r:=vertex_threshold(v,k,b,epsilon)) is not None]
    exact=min((r['eta'] for r in results),default=1.)
    cap=snapshot.get('nonremoved_support_cap',6);removal=snapshot.get('removal_budget',0)
    if any(type(x)!=int or x<0 for x in [cap,removal]):raise ValueError('invalid support')
    def size(eta):
        if eta<=0:return None
        S,risk,previous=map(float,call('size',eta,bs,cap,removal).split())
        return dict(S=int(S),risk=risk,previous_S_risk=previous)
    baseline=size(epsilon); simple_size=size(simple); exact_size=size(exact)
    omitted=[i for i in range(n) if k[i]==0]
    residual=max(sum(v[i]*b[i] for i in omitted) for v in vv)
    ca=min((k[i]/U[i] for i in range(n) if k[i]>0 and U[i]>0),default=0.)
    pruning=-math.expm1(ca*math.log1p(-epsilon+residual)) if residual<epsilon else 0.
    return dict(status='conditional_numerical_bundle_diagnostic',modes=[dict(m,L=lower[i],U=U[i],K=k[i],b=b[i]) for i,m in enumerate(modes)],
        c_K=c,eta_simple=simple,eta_refined=exact,vertices=results,direct_bound_sufficient=not results,
        S_baseline=baseline,S_simple=simple_size,S_refined=exact_size,
        explicit_count_upper=cap+removal+math.ceil((baseline['S']-cap-removal)/c) if c>=1 else None,
        raw_per_bundle=sum(k),raw_draws_simple=sum(k)*simple_size['S'] if simple_size else None,
        omitted_mode_risk=residual,eta_mode_pruning=pruning,
        beta_cp=bc,beta_cert=bs,combined_failure_budget=min(1.,bc+bs),
        limitation='IID conditional draws and stationary IID CP history are assumptions. Support counts bundles. Exact pruning must preserve the full conjunction. Finite precision is not a verified enclosure. Order-statistic sampling requires nested feasible sets and is not implemented.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--backend',type=Path,default=Path('build-bundles/certificate_numeric_backend'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('use a new output directory')
    result=analyze(json.loads(a.snapshot.read_text()),a.backend.resolve())
    a.output.mkdir(parents=True)
    (a.output/'bundle_certificate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'snapshot.json').write_bytes(a.snapshot.read_bytes())
    (a.output/'provenance.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__),a.backend,a.snapshot]},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['c_K','eta_simple','eta_refined','S_baseline','S_simple','S_refined','raw_draws_simple']},indent=2))
