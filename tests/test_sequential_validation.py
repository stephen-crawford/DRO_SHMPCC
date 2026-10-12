#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from make_sequential_confirmation import conditional_draw
from run_sequential_validation import replay,POLICIES

class NominalOT:
    calls=0
    def risk(self,obs,v):
        self.calls+=1;p=np.array([m['p'] for m in obs['modes']]);return float(p@v),p

def main():
    context=dict(H=3,length=0,radius=.625,obstacles=[dict(id=0,state=[1,0,.1,0],modes=[dict(p=1.,A=np.eye(4).tolist(),G=np.array([[.1,0],[0,.1],[0,0],[0,0]]).tolist(),b=[0,0,0,0])])]);o=context['obstacles'][0];ego=np.zeros((3,2));seed=[77,15,2,1]
    full=conditional_draw(context,o,o['modes'][0],ego,100,seed)[0]
    for batch in (1,5,10):
        chunk=np.concatenate([conditional_draw(context,o,o['modes'][0],ego,min(batch,100-start),seed,start)[0] for start in range(0,100,batch)])
        assert np.array_equal(full,chunk)
    for policy in POLICIES:
        for batch in (1,5,10):
            ot=NominalOT();safe=replay(context,[[np.zeros(1000,int)]],ot,policy,batch)
            assert safe['status']=='certify' and safe['R_U']<=.05 and safe['R_L']<=safe['R_U']
            bad=replay(context,[[np.ones(1000,int)]],ot,policy,batch)
            assert bad['status']=='reject' and bad['R_L']>.05
            cap=replay(context,[[np.zeros(2,int)]],ot,policy,batch)
            assert cap['status']=='unresolved' and cap['reason']=='stream_exhaustion' and cap['N']==2
    print('PASS: counter streams identical at batch 1/5/10; all five policies certify/reject/abstain on analytical streams; no exhausted-stream fabrication')
if __name__=='__main__':main()
