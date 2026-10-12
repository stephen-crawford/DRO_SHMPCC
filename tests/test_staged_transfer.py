#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from run_staged_transfer import allocation,inverse
from run_oracle_transfer import Bridge
import numpy as np

bridge=Bridge(sys.argv[1])
def one():return dict(rho=0.,modes=[dict(p=1)],D=[[0.]])
for bs,expected in [([np.array([1.]),np.array([.001])],.049),([np.array([1.]),np.array([.001]),np.array([.002])],.047)]:
 obs=[one() for _ in bs];qs=[np.ones(1) for _ in bs]
 equal,_,_=allocation(bridge,obs,qs,bs,True);optimized,budgets,targets=allocation(bridge,obs,qs,bs,False)
 assert abs(equal-.05/len(bs))<1e-9
 assert abs(optimized-expected)<1e-9
 assert optimized>equal and sum(budgets)<=.05
 assert optimized<=min(targets)+1e-12
obs=[dict(rho=0.,modes=[dict(p=.8),dict(p=.2)],D=[[0,1],[1,0]]),one()];qs=[np.array([.5,.5]),np.ones(1)];bs=[np.ones(2),np.ones(1)]
actual,_,_=allocation(bridge,obs,qs,bs,False)
assert abs(actual-.05/2.6)<1e-9
# Independent physical-risk allocation grid checks the same optimum from inverses.
values=[]
for epsilon in np.linspace(0,.05,101):values.append(min(inverse(bridge,obs[0],qs[0],bs[0],epsilon),inverse(bridge,obs[1],qs[1],bs[1],.05-epsilon)))
assert max(values)<=actual+1e-9 and actual-max(values)<.0005
zero,_,_=allocation(bridge,[one(),one()],[np.ones(1),np.ones(1)],[np.zeros(1),np.zeros(1)],False)
assert zero==1
bridge.close()
print('PASS: equal/optimized two- and three-obstacle analytical allocations; unequal slopes; independent allocation grid; zero-risk endpoint')
