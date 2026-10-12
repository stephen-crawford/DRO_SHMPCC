import sys
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from particle_coreset import robust_probability,collision_distances,medoids,factors,lift
import numpy as np
from scipy.optimize import linprog
assert robust_probability([0,0],[.3,.7],0)==1
assert robust_probability([2],[1],0)==0
assert abs(robust_probability([2],[1],.5)-.25)<1e-12
assert robust_probability([2],[1],2)==1
assert abs(robust_probability([1,3],[.5,.5],.5)-.5)<1e-12
assert abs(robust_probability([1,3],[.5,.5],1.25)-.75)<1e-12
rng=np.random.default_rng(44)
for _ in range(100):
 d=rng.uniform(0,5,10);d[0]=0;w=rng.dirichlet(np.ones(10));R=float(rng.uniform(0,3))
 result=linprog(-np.ones(10),A_ub=d.reshape(1,-1),b_ub=[R],bounds=list(zip(np.zeros(10),w)),method='highs')
 assert result.success and abs(robust_probability(d,w,R)+result.fun)<1e-9
# Move one group along its radial direction to reach the closed collision set.
X=np.array([[[[3.,0.],[5.,0.]]],[[[.5,0.],[5.,0.]]]])
ego=np.zeros((2,1,2));assert np.allclose(collision_distances(X,ego,1),[2,0])
Xmulti=np.repeat(X[:1],2,axis=1);Xmulti[0,1,1]=[1.5,0];assert collision_distances(Xmulti,ego,1)[0]==.5
Xtest=np.array([[0.],[1.],[9.],[10.]])
indices,w,delta=medoids(Xtest,2);dist=np.abs(Xtest-Xtest[indices].T)
# Independent exact OT LP with induced cluster masses.
Aeq=np.vstack([np.kron(np.eye(4),np.ones((1,2))),np.kron(np.ones((1,4)),np.eye(2))]);lp=linprog(dist.ravel(),A_eq=Aeq,b_eq=np.r_[np.ones(4)/4,w],bounds=(0,None),method='highs')
assert lp.success and abs(lp.fun-delta)<1e-12
# Identical time marginals do not imply identical joint trajectory distributions.
P=np.array([[1.,1.],[-1.,-1.]]);Q=np.array([[1.,-1.],[-1.,1.]])
assert np.all(np.sort(P,axis=0)==np.sort(Q,axis=0))
assert np.all(np.linalg.norm(P[:,None]-Q[None,:],axis=2)==2)
print('PASS: six analytical chance fixtures, 100 independent fractional-transport LP comparisons, collision geometry, exact assignment OT, temporal-marginal counterexample')

# A genuinely Gaussian temporal counterexample: x_{k+1}=+/-x_k+noise.
# Both laws have stage variances 1 and 2, but different temporal covariance.
modes=[]
for sign in [1,-1]:
 A=np.eye(4);A[0,0]=sign;G=np.zeros((4,1));G[0,0]=1
 modes.append(dict(A=A.tolist(),G=G.tolist(),b=[0,0,0,0],p=.5))
context=dict(H=2,obstacles=[dict(id=0,state=[0,0,0,0],D=[[0,0],[0,0]],rho=0,modes=modes)])
radius,records=lift(context)
assert abs(radius-2)<1e-12 and abs(records[0]['gap'])<1e-12
assert (np.sqrt(5)-1)/np.sqrt(np.pi)>0.69
assert robust_probability([1,3],[.5,.5],0)==0
print('PASS: zero-cost identical Gaussian stage marginals need positive trajectory lifting radius; reconstructed lifting dual matches analytical LP')

assert abs(robust_probability([1,3],[.5,.5],.7)-robust_probability([10,30],[.5,.5],7))<1e-12
print("PASS: common metric/radius scaling leaves the robust chance value unchanged")
