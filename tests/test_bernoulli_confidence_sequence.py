#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
from bernoulli_confidence_sequence import endpoints,log_mixture,ConfidenceSequence

def main():
    alpha=.01;threshold=-np.log(alpha);assert endpoints(0,0,alpha)==(0.,1.)
    for n in (1,2,10,1000,1000000):
        for k in sorted({0,1,n//2,n-1,n}):
            if k<0 or k>n:continue
            lo,hi=endpoints(n,k,alpha);assert 0<=lo<=k/n<=hi<=1
            for p in (lo,hi):
                if 0<p<1:
                    conditioning=4*np.spacing(p)*(k/p+(n-k)/(1-p))+1e-8
                    assert abs(log_mixture(n,k,p)-threshold)<conditioning,(n,k,p)
            if 0<k<n:
                f=lambda p:log_mixture(n,k,p)-threshold
                reflo=brentq(f,np.finfo(float).tiny,k/n,xtol=1e-15)
                refhi=brentq(f,k/n,np.nextafter(1.,0.),xtol=1e-15)
                assert abs(lo-reflo)<1e-12 and abs(hi-refhi)<1e-12
    for p in (0.,.001,.03,.5,.99,1.):
        rng=np.random.default_rng(91021);cs=ConfidenceSequence(alpha);previous=(0.,1.)
        for y in rng.random(1000)<p:
            current=cs.observe(int(y));assert current[0]>=previous[0] and current[1]<=previous[1];previous=current
    # Finite-horizon exact crossing probability by enumeration of ALL paths.
    # State retains only paths which have never crossed the LR threshold.
    for p in (.01,.1,.5,.9,.99):
        states={0:1.};cross=0.
        for n in range(1,41):
            nxt={}
            for k,prob in states.items():
                for y,weight in ((0,1-p),(1,p)):
                    j=k+y;mass=prob*weight
                    if log_mixture(n,j,p)>=threshold:cross+=mass
                    else:nxt[j]=nxt.get(j,0.)+mass
            states=nxt
        assert cross<=alpha+1e-12,(p,cross)
    # Diagnostic Monte Carlo, NOT a proof: pointwise per-mode coverage over 5000 looks.
    rng=np.random.default_rng(88331);simulations=2000;horizon=5000;diagnostic=[]
    from scipy.special import betaln
    for p in (0.,.001,.03,.5,.99,1.):
        k=np.zeros(simulations,int);cross=np.zeros(simulations,bool)
        for n in range(1,horizon+1):
            k+=rng.random(simulations)<p
            logM=betaln(k+.5,n-k+.5)-betaln(.5,.5)
            if p>0:logM-=k*np.log(p)
            if p<1:logM-=(n-k)*np.log1p(-p)
            cross|=logM>=threshold
        count=int(cross.sum());diagnostic.append(dict(p=p,simulations=simulations,horizon=horizon,ever_crossed=count,fraction=count/simulations))
    print('PASS: boundary/large-count endpoint inversion; independent root checks; monotone intersections; exact finite-path Ville crossing checks')
    print(json.dumps(diagnostic,indent=2))
if __name__=='__main__':main()
