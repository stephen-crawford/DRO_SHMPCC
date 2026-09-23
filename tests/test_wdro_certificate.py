#!/usr/bin/env python3
import copy
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_wdro_certificate as diagnostic

class TransferTests(unittest.TestCase):
    def test_inner_lp_against_independent_solver(self):
        rng=np.random.default_rng(91)
        for m in [2,3,6]:
            for _ in range(10):
                p=rng.dirichlet(np.ones(m));q=rng.dirichlet(np.ones(m));b=rng.random(m);eta=float(rng.random())
                q[0]=0;q=q/q.sum()
                expected=linprog(-p,A_ub=[q],b_ub=[eta],bounds=list(zip(np.zeros(m),b)),method='highs')
                self.assertTrue(expected.success)
                self.assertAlmostEqual(diagnostic.transfer_at(p,q,b,eta),-expected.fun,places=10)

    def test_conservative_and_null_mode(self):
        vv=diagnostic.vertices([.7,.1],[.9,.3])
        eta,psi=diagnostic.invert(vv,[.5,.5],[1,1],.05)
        self.assertLessEqual(eta,.05+1e-14)
        self.assertLessEqual(psi(eta),.05)
        eta,_=diagnostic.invert(vv,[1,0],[1,1],.05)
        self.assertIsNone(eta)

    def test_production_bounds_and_sizing(self):
        backend=ROOT/'build-certificate/certificate_numeric_backend'
        s=dict(obstacles=1,modes=[dict(mode='safe',count=900,p_hat=.9,q=.5),dict(mode='danger',count=100,p_hat=.1,q=.5)])
        result=diagnostic.analyze(copy.deepcopy(s),backend)
        self.assertLessEqual(result['epsilon_Q_max'],.05+1e-14)
        self.assertGreaterEqual(result['scenario_count_ratio'],1)
        for key in ['S_SH','S_WDRO']:
            bound=result[key];target=.05 if key=='S_SH' else result['epsilon_Q_max']
            self.assertLessEqual(bound['risk'],target)
            self.assertGreater(bound['previous_S_risk'],target)
        s['modes'][0]['b']=0
        with self.assertRaisesRegex(ValueError,'justification'):diagnostic.analyze(copy.deepcopy(s),backend)
        s['b_justification']='SYNTHETIC test assumption: safe mode has exactly zero violation, not inferred from surrogate risk'
        positive=diagnostic.analyze(s,backend)
        self.assertGreater(positive['epsilon_Q_max'],.05)
        self.assertLess(positive['scenario_count_ratio'],.7)
        print('Synthetic justified-b fixture:',positive['epsilon_Q_max'],positive['scenario_count_ratio'])

if __name__=='__main__':unittest.main()
