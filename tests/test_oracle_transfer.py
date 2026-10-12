#!/usr/bin/env python3
import argparse,itertools,subprocess
import numpy as np
from scipy.optimize import linprog
class Bridge:
 def __init__(self,path):self.proc=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 def query(self,command,p,q,b,D,rho,value=.05,eta=0):
  line=[command,len(p),rho,value,eta,*p,*q,*b,*np.asarray(D).ravel()]
  self.proc.stdin.write(' '.join(map(str,line))+'\n');self.proc.stdin.flush();response=self.proc.stdout.readline()
  if not response:raise RuntimeError('transfer process terminated')
  return float(response)
 def close(self):self.proc.stdin.close();assert self.proc.wait()==0

def independent(p,q,b,D,rho,eta):
 # Enumerate intersections of M active inequalities, independently of box masks.
 M=len(p);A=np.vstack([-np.eye(M),np.eye(M),q]);rhs=np.r_[np.zeros(M),b,eta];best=0.
 for active in itertools.combinations(range(2*M+1),M):
  try:v=np.linalg.solve(A[list(active)],rhs[list(active)])
  except np.linalg.LinAlgError:continue
  if np.any(A@v>rhs+1e-10):continue
  objective=-np.tile(v,M);marginals=np.kron(np.eye(M),np.ones((1,M)))
  lp=linprog(objective,A_ub=np.asarray(D).reshape(1,-1),b_ub=[rho],A_eq=marginals,b_eq=p,bounds=(0,None),method='highs')
  assert lp.success;best=max(best,-lp.fun)
 return best

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);args=ap.parse_args();bridge=Bridge(args.runner)
 p=[.8,.2];q=[.2,.8];b=[0,1];D=[[0,1],[1,0]]
 assert abs(bridge.query('eta',p,q,b,D,0)-.2)<1e-8
 assert abs(bridge.query('eta',p,q,b,D,.1)-(.05/.3*.8))<1e-8
 assert abs(bridge.query('eta',p,p,[1,1],D,0)-.05)<1e-8
 assert bridge.query('eta',p,q,[.01,.02],D,.1)==1
 assert abs(bridge.query('psi',p,q,[1,1],[[0,0],[0,0]],0,eta=.1)-.5)<1e-8
 rng=np.random.default_rng(19003);cases=0
 for M in [2,3,4]:
  for _ in range(12):
   p=rng.dirichlet(np.ones(M));q=rng.dirichlet(np.ones(M));b=rng.uniform(0,1,M);positions=rng.normal(size=(M,2));D=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1);rho=float(rng.uniform(0,.5));eta=float(rng.uniform(0,1))
   actual=bridge.query('psi',p,q,b,D,rho,eta=eta);expected=independent(p,q,b,D,rho,eta)
   assert abs(actual-expected)<1e-7,(M,actual,expected)
   cases+=1
 bridge.close()
 bad=subprocess.run([args.runner],input='eta 2 0 .05 0 .8 .2 1 0 0 1 0 1 1 0\n',text=True,capture_output=True)
 assert bad.returncode!=0 and 'missing proposal support' in bad.stderr
 print('PASS: 5 analytical fixtures; 36 independent active-set/HiGHS comparisons; missing dangerous support rejected')
if __name__=='__main__':main()
