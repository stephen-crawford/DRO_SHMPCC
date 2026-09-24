#!/usr/bin/env python3
import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_bundle_certificate import analyze,bundle_violation,vertex_threshold

class BundleMathTests(unittest.TestCase):
    def test_multiple_obstacle_union_amplification(self):
        rng=np.random.default_rng(81)
        for obstacles in [1,2,3,4]:
            for _ in range(250):
                safe_probability=1.;risk_sum=0.;c=float('inf')
                for o in range(obstacles):
                    p=rng.dirichlet(np.ones(3));U=np.minimum(1,p+.1)
                    K=np.ceil(2*U).astype(int);r=rng.uniform(0,.15,3)
                    c=min(c,min(K/U));risk_sum+=p@r
                    safe_probability*=math.prod((1-ri)**ki for ri,ki in zip(r,K))
                lower=-math.expm1(obstacles*c*math.log1p(-risk_sum/obstacles))
                self.assertGreaterEqual(1-safe_probability+2e-15,lower)
    def test_amplification_and_detection(self):
        rng=np.random.default_rng(77)
        for _ in range(1000):
            p=rng.dirichlet(np.ones(4));U=np.minimum(1,p+.1)
            K=np.ceil(3*U).astype(int);c=min(K/U);r=rng.random(4)
            actual=bundle_violation(r,K)
            lower=-math.expm1(c*math.log1p(-float(p@r)))
            self.assertGreaterEqual(actual+2e-15,lower)
        r=np.array([.1,.2]);K=np.array([2,3]);n=100000
        nohit=np.ones(n,dtype=bool)
        for ri,ki in zip(r,K):nohit &= np.all(rng.random((n,ki))>=ri,axis=1)
        predicted=bundle_violation(r,K)
        self.assertLess(abs((1-nohit.mean())-predicted),6*math.sqrt(predicted*(1-predicted)/n))

    def test_kkt_against_independent_optimizer(self):
        rng=np.random.default_rng(79)
        for _ in range(40):
            v=rng.dirichlet(np.ones(3));k=rng.integers(1,5,3).tolist();b=rng.uniform(.2,.9,3);eps=.05
            result=vertex_threshold(v,k,b,eps)
            objective=lambda r:sum(-ki*math.log1p(-ri) for ki,ri in zip(k,r))
            oracle=minimize(objective,np.full(3,eps),jac=lambda r:np.asarray(k)/(1-r),bounds=list(zip(np.zeros(3),b)),
                constraints=[dict(type='ineq',fun=lambda r:float(v@r)-eps,jac=lambda r:v)],
                method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
            self.assertTrue(oracle.success,oracle.message)
            expected=-math.expm1(-oracle.fun)
            self.assertLess(abs(result['eta']-expected),1e-8)
            self.assertGreaterEqual(sum(vi*ri for vi,ri in zip(v,result['r'])),eps-1e-15)
            self.assertLess(abs(result['dual_gap']),1e-10)

    def test_zero_multiplicities_and_boundaries(self):
        self.assertEqual(bundle_violation([1,0],[0,2]),0)
        self.assertEqual(vertex_threshold([.5,.5],[0,2],[1,1],.1)['eta'],0)
        self.assertIsNone(vertex_threshold([.5,.5],[1,1],[.01,.01],.1))
        self.assertEqual(vertex_threshold([.5,.5],[1,1],[1,1],1)['eta'],1)
        result=vertex_threshold([0,1],[1,2],[1,1],.1)
        self.assertAlmostEqual(result['eta'],.19)

    def test_counts_and_pruning(self):
        snapshot=dict(modes=[dict(mode='a',count=95,q=.7,score=0),dict(mode='b',count=5,q=.3,score=1)],c0=2,extras=2)
        r=analyze(snapshot,ROOT/'build-bundles/certificate_numeric_backend')
        self.assertGreaterEqual(r['c_K'],2)
        self.assertGreaterEqual(r['eta_refined']+1e-14,r['eta_simple'])
        self.assertLessEqual(r['S_simple']['S'],r['explicit_count_upper'])
        self.assertLess(r['S_simple']['S'],r['S_baseline']['S'])
        self.assertLessEqual(r['S_simple']['risk'],r['eta_simple'])
        self.assertGreater(r['S_simple']['previous_S_risk'],r['eta_simple'])
        snapshot['modes'][0].update(K=0,b=0)
        snapshot['modes'][1].update(K=3,b=1)
        snapshot['b_justification']='Synthetic uniform zero-risk mode for numerical test only'
        r=analyze(snapshot,ROOT/'build-bundles/certificate_numeric_backend')
        self.assertEqual(r['eta_simple'],0)
        self.assertGreater(r['eta_mode_pruning'],0)
        self.assertGreater(r['eta_refined'],0)

if __name__=='__main__':unittest.main()
