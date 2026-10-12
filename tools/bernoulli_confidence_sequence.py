"""Experiment-only Jeffreys beta-mixture Bernoulli confidence sequence."""
from functools import lru_cache
import math
from scipy.special import betaln

A0=B0=.5

@lru_cache(maxsize=262144)
def endpoints(n,k,alpha):
    if not isinstance(n,int) or not isinstance(k,int) or not 0<=k<=n or not 0<alpha<1:
        raise ValueError('invalid Bernoulli counts or alpha')
    if n==0:return 0.,1.
    constant=float(betaln(k+A0,n-k+B0)-betaln(A0,B0));threshold=-math.log(alpha)
    if k==0:return 0.,min(1.,math.nextafter(-math.expm1((constant-threshold)/n),1.))
    if k==n:return max(0.,math.nextafter(math.exp((constant-threshold)/n),0.)),1.
    centre=k/n
    def inside(p):return constant-k*math.log(p)-(n-k)*math.log1p(-p)<threshold
    lo=0.;hi=centre
    for _ in range(52):
        mid=(lo+hi)/2
        if inside(mid):hi=mid
        else:lo=mid
    lower=lo # outside bracket: conservative enclosure of the open set
    lo=centre;hi=1.
    for _ in range(52):
        mid=(lo+hi)/2
        if mid==1 or not inside(mid):hi=mid
        else:lo=mid
    return max(0.,math.nextafter(lower,0.)),min(1.,math.nextafter(hi,1.))

def log_mixture(n,k,p):
    if not 0<=p<=1 or not 0<=k<=n:raise ValueError('invalid mixture argument')
    constant=float(betaln(k+A0,n-k+B0)-betaln(A0,B0))
    if (p==0 and k>0) or (p==1 and k<n):return math.inf
    return constant-(k*math.log(p) if k else 0)-((n-k)*math.log1p(-p) if k<n else 0)

class ConfidenceSequence:
    def __init__(self,alpha):
        endpoints(0,0,alpha);self.alpha=alpha;self.n=0;self.k=0;self.lower=0.;self.upper=1.
    def observe(self,y):
        if y not in (0,1):raise ValueError('Bernoulli observation required')
        self.n+=1;self.k+=int(y);lo,hi=endpoints(self.n,self.k,self.alpha)
        self.lower=max(self.lower,lo);self.upper=min(self.upper,hi)
        return self.lower,self.upper
    def projected(self,y):
        lo,hi=endpoints(self.n+1,self.k+int(y),self.alpha)
        return max(self.lower,lo),min(self.upper,hi)
