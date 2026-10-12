#!/usr/bin/env python3
"""Independent tests for experiment-only fixed-plan mode validation."""
import argparse,itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binomtest
from oracle_mode_validation import OTBridge,upper,pareto,optimum,collision_pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);args=ap.parse_args();bridge=OTBridge(args.runner)
    for n in (1,10,100,5000):
        for k in sorted({0,1,n//2,n}):
            expected=binomtest(k,n,alternative='less').proportion_ci(confidence_level=.99,method='exact').high
            assert abs(upper(k,n,.01)-expected)<1e-10,(k,n)
    assert abs(upper(0,100,.01)-(1-.01**.01))<1e-12
    rng=np.random.default_rng(190081)
    for M in (1,2,3,4):
        for _ in range(15):
            p=rng.dirichlet(np.ones(M));positions=rng.normal(size=(M,2));D=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1);rho=float(rng.uniform(0,1));v=rng.uniform(0,1,M)
            obs=dict(modes=[dict(p=x) for x in p],D=D,rho=rho);actual,q=bridge.risk(obs,v)
            lp=linprog(-np.tile(v,M),A_ub=D.reshape(1,-1),b_ub=[rho],A_eq=np.kron(np.eye(M),np.ones((1,M))),b_eq=p,bounds=(0,None),method='highs')
            assert lp.success and abs(actual+lp.fun)<1e-8,(M,actual,-lp.fun)
    obs=dict(modes=[dict(p=.8),dict(p=.2)],D=[[0,1],[1,0]],rho=.1)
    assert abs(bridge.risk(obs,[0,1])[0]-.3)<1e-12
    obs['rho']=0;assert abs(bridge.risk(obs,[0,1])[0]-.2)<1e-12
    obs['D']=[[0,0],[0,0]];assert bridge.risk(obs,[0,1])[0]==1
    bridge.close()
    # Compare pruning/combination against full Cartesian enumeration, including
    # options whose risk does not monotonically decrease with sample count.
    for _ in range(50):
        groups=[[(n,float(rng.uniform(0,.1)),(n,)) for n in (10,20,50,100)] for _ in range(3)]
        all_options=[(sum(x[0] for x in xs),sum(x[1] for x in xs),sum((x[2] for x in xs),())) for xs in itertools.product(*groups)]
        feasible=[x for x in all_options if x[1]<=.1];expected=min(feasible,key=lambda r:(r[0],r[1],r[2])) if feasible else None
        actual=optimum([pareto(g) for g in groups],.1)
        assert (actual is None)==(expected is None)
        if actual:assert actual[0]==expected[0] and abs(actual[1]-expected[1])<1e-14
    context=dict(length=0,H=2,radius=1);ego=np.zeros((2,2));mode=dict(A=np.eye(4),G=np.zeros((4,2)),b=np.zeros(4))
    assert collision_pool(context,dict(state=[1,0,0,0]),mode,ego,20,1).all() # contact
    assert not collision_pool(context,dict(state=[1.01,0,0,0]),mode,ego,20,1).any()
    mode['b']=np.array([-1,0,0,0]);assert collision_pool(context,dict(state=[3,0,0,0]),mode,ego,20,1).all() # second stage
    print('PASS: exact binomial endpoints/independent SciPy inversion; 60 primal-OT/HiGHS comparisons; analytical OT aliases; 50 brute-force allocation comparisons; collision geometry')
if __name__=='__main__':main()
