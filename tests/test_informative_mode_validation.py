#!/usr/bin/env python3
"""Small exhaustive fixtures for the fine-grid oracle and VOI replay."""
import argparse,itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from oracle_mode_validation import OTBridge,upper
from run_informative_mode_validation import study

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);args=ap.parse_args();bridge=OTBridge(args.runner);grid=(25,50,100,200,500)
    rng=np.random.default_rng(81117)
    for obstacles in (1,2):
        for repeat in range(5):
            obs=dict(modes=[dict(p=.9),dict(p=.1)],rho=0.,D=[[0,1],[1,0]])
            context=dict(obstacles=[obs]*obstacles);pools=[]
            for _ in range(obstacles):pools.append([rng.random(500)<.001,rng.random(500)<.01])
            result=study(context,pools,bridge,grid)
            M=obstacles*2;alpha=.01/(M*len(grid));options=[]
            for ns in itertools.product(grid,repeat=M):
                risk=sum([.9,.1][m%2]*upper(int(pools[m//2][m%2][:n].sum()),n,alpha) for m,n in enumerate(ns))
                if risk<=.05:options.append((sum(ns),risk,ns))
            expected=min(options,key=lambda r:(r[0],r[1],r[2])) if options else None
            actual=result['oracle'];assert (actual is None)==(expected is None)
            if actual:assert actual[0]==expected[0] and abs(actual[1]-expected[1])<1e-12
            for name in ('uniform','nominal','voi'):
                option=result[name]
                if option:
                    total,risk,ns=option;assert total==sum(ns) and risk<=.05
                    assert total>=actual[0]
            for earlier,later in zip(result['voi_trace'],result['voi_trace'][1:]):
                assert sum(a!=b for a,b in zip(earlier['counts'],later['counts']))==1
                assert later['N']>earlier['N']
    obs=dict(modes=[dict(p=.7),dict(p=.3)],rho=.1,D=[[0,1],[1,0]])
    for _ in range(30):
        v=rng.uniform(0,.8,2);change=rng.uniform(-.01,.01,2);risk,q=bridge.risk(obs,v);next_risk,_=bridge.risk(obs,v+change)
        assert next_risk>=risk+q@change-1e-10 # q is a subgradient, not necessarily a derivative.
    high=[np.ones(500,dtype=bool),np.ones(500,dtype=bool)]
    failed=study(dict(obstacles=[obs]),[high],bridge,grid)
    assert all(failed[name] is None for name in ('uniform','nominal','voi','oracle'))
    bridge.close();print('PASS: 10 independent exhaustive-grid oracle comparisons; VOI grid/one-mode update checks; 30 OT subgradient checks; uncertifiable finite-grid exhaustion')
if __name__=='__main__':main()
