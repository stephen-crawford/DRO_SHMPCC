#!/usr/bin/env python3
import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_posterior_certificate import projected_probability,evaluate

class PosteriorTests(unittest.TestCase):
    def test_deterministic_strict_boundary_and_coincident_mean(self):
        zero=[[0,0],[0,0]]
        self.assertEqual(projected_probability([0,0],[1,0],zero,1)['probability_upper'],0)
        self.assertEqual(projected_probability([0,0],[.5,0],zero,1)['probability_upper'],1)
        self.assertEqual(projected_probability([0,0],[2,0],zero,1)['probability_upper'],0)
        r=projected_probability([0,0],[0,0],[[1,0],[0,1]],1)
        self.assertEqual(r['zero_distance'],1)
        self.assertAlmostEqual(r['probability_upper'],.8413447460685429)
        with self.assertRaises(ValueError):projected_probability([0,0],[1,0],[[-1,0],[0,1]],1)

    def test_gaussian_projected_event_contains_collision(self):
        rng=np.random.default_rng(71)
        mean=np.array([1.4,.4]);cov=np.array([[.3,.08],[.08,.2]])
        x=rng.multivariate_normal(mean,cov,size=100000);n=mean/np.linalg.norm(mean)
        collision=np.linalg.norm(x,axis=1)<1
        projected=1-x@n>0
        self.assertTrue(np.all(~collision | projected))
        bound=projected_probability([0,0],mean,cov,1)['probability_upper']
        self.assertLessEqual(abs(projected.mean()-bound),6*math.sqrt(bound*(1-bound)/len(x)))

    def test_posterior_transfer_gating_and_no_sample_reduction_claim(self):
        snapshot=dict(obstacles=1,collision_radius=1,modes=[dict(mode='a',count=90,p_hat=.9,q=.5),dict(mode='b',count=10,p_hat=.1,q=.5)],
            steps=[dict(k=1,disc_centers=[[0,0]],gaussians={'a':dict(mean=[3,0],covariance=[[0,0],[0,0]]),'b':dict(mean=[0,0],covariance=[[0,0],[0,0]])})])
        backend=ROOT/'build-certificate/certificate_numeric_backend'
        r,_=evaluate(snapshot,backend)
        self.assertEqual([m['b'] for m in r['mode_bounds']],[0,1])
        self.assertIsNone(r['transferred_bound']);self.assertFalse(r['presolve_sample_reduction_claim'])
        snapshot.update(scenario_theorem_eligible=True,scenario_theorem_justification='Synthetic test assumption only',sample_count=2000,total_support_bound=8)
        r,_=evaluate(snapshot,backend)
        self.assertIsNotNone(r['eta_SH'])
        self.assertLessEqual(r['transferred_bound'],r['direct_cp_bound'])
        self.assertFalse(r['presolve_sample_reduction_claim'])
        bad=copy.deepcopy(snapshot);bad.pop('scenario_theorem_justification')
        with self.assertRaises(ValueError):evaluate(bad,backend)

if __name__=='__main__':unittest.main()
