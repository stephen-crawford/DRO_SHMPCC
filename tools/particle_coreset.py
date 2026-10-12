"""Experiment-only Euclidean trajectory W1 geometry and finite chance dual."""
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

def robust_probability(distances,weights,radius):
 d=np.asarray(distances,float);w=np.asarray(weights,float)
 if d.shape!=w.shape or not len(d) or not np.isfinite(d).all() or np.any(d<0) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-12 or not math.isfinite(radius) or radius<0:
  raise ValueError('invalid robust chance inputs')
 if radius==0:return float(w[d==0].sum())
 order=np.argsort(d,kind='stable');d=d[order];w=w[order]
 mass=np.r_[0.,np.cumsum(w)[:-1]];cost=np.r_[0.,np.cumsum(w*d)[:-1]];positive=d>0
 if not positive.any():return 1.
 values=mass[positive]+(radius-cost[positive])/d[positive]
 return float(np.clip(min(1.,values.min()),0.,1.))

def collision_distances(particles,ego,radius):
 # particles: N x obstacles x stages x 2; ego: stages x discs x 2.
 delta=particles[:,:,:,None,:]-ego[None,None,:,:,:]
 clearance=np.linalg.norm(delta,axis=-1)-radius
 return np.maximum(0.,clearance.min(axis=(1,2,3)))

def factors(model,state,H):
 A=np.asarray(model['A']);G=np.asarray(model['G']);bias=np.asarray(model['b']);mean=np.asarray(state,float);n=G.shape[1]
 L=np.zeros((4,H*n));means=[];noise=[]
 for k in range(H):
  mean=A@mean+bias;L=A@L;L[:,k*n:(k+1)*n]+=G;means.append(mean[:2].copy());noise.append(L[:2].copy())
 return np.array(means).reshape(-1),np.concatenate(noise,axis=0)

def lift(context):
 H=context['H'];radius=0.;records=[]
 for obs in context['obstacles']:
  factors_list=[factors(m,obs['state'],H) for m in obs['modes']];D=np.array(obs['D']);C=np.zeros_like(D)
  for i,(mi,Li) in enumerate(factors_list):
   for j,(mj,Lj) in enumerate(factors_list):C[i,j]=math.sqrt(np.sum((mi-mj)**2)+np.sum((Li-Lj)**2))
  p=np.array([m['p'] for m in obs['modes']]);p=p/p.sum();M=len(p)
  lp=linprog(-C.ravel(),A_ub=D.reshape(1,-1),b_ub=[obs['rho']],A_eq=np.kron(np.eye(M),np.ones((1,M))),b_eq=p,bounds=(0,None),method='highs')
  if not lp.success:raise ValueError('trajectory lifting LP failed')
  multiplier=max(0.,-float(lp.ineqlin.marginals[0]))
  alpha=np.max(C-multiplier*D,axis=1)
  # Reconstruct a feasible OT dual upper bound, rather than taking a potentially
  # understated primal objective as the trajectory radius.
  lifted=float(alpha@p+multiplier*obs['rho']);primal=-float(lp.fun)
  if lifted<primal-1e-9 or lifted-primal>1e-7:raise ValueError('lifting primal/dual gap')
  radius+=lifted
  records.append(dict(obstacle=obs['id'],D=D.tolist(),synchronous_RMS=C.tolist(),rho_mode=obs['rho'],rho_trajectory=lifted,primal=primal,dual_multiplier=multiplier,dual_alpha=alpha.tolist(),gap=lifted-primal))
 return radius,records

def sample(context,N,seed):
 rng=np.random.default_rng(seed);outputs=[]
 for obs in context['obstacles']:
  modes=obs['modes'];chosen=rng.choice(len(modes),N,p=[m['p'] for m in modes]);X=np.zeros((N,context['H'],2))
  for index,mode in enumerate(modes):
   ids=np.where(chosen==index)[0];states=np.tile(obs['state'],(len(ids),1));A=np.array(mode['A']);G=np.array(mode['G']);bias=np.array(mode['b'])
   for k in range(context['H']):
    states=states@A.T+bias+rng.standard_normal((len(ids),G.shape[1]))@G.T;X[ids,k]=states[:,:2]
  outputs.append(X)
 return np.stack(outputs,axis=1)

def medoids(X,K,iterations=5):
 # Two deterministic starts, exact within-cluster medoid updates; local optimum only.
 best=None
 for start in range(2):
  selected=[int(np.argmin(np.linalg.norm(X-X.mean(axis=0),axis=1))) if start==0 else 0]
  distances=np.linalg.norm(X-X[selected[0]],axis=1)
  while len(selected)<min(K,len(X)) and distances.max()>0:
   index=int(np.argmax(distances));selected.append(index);distances=np.minimum(distances,np.linalg.norm(X-X[index],axis=1))
  for _ in range(iterations):
   D=cdist(X,X[selected]);labels=D.argmin(axis=1);updated=[]
   for j,index in enumerate(selected):
    ids=np.where(labels==j)[0]
    updated.append(int(ids[np.argmin(cdist(X[ids],X[ids]).sum(axis=1))]) if len(ids) else index)
   if updated==selected:break
   selected=updated
  D=cdist(X,X[selected]);labels=D.argmin(axis=1);cost=float(D[np.arange(len(X)),labels].mean());weights=np.bincount(labels,minlength=len(selected))/len(X)
  if best is None or cost<best[2]:best=(np.array(selected),weights,cost)
 return best
