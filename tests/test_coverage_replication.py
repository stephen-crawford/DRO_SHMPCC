import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
from report_coverage_replication import window_metrics,mcnemar,ordering_check
from run_coverage_design import candidates,select
class CoverageDesignTests(unittest.TestCase):
    def test_window_product_and_missingness(self):
        rows=[dict(step=t,attempt='0',N_d=n,p_d=.1,q_d=.2,scenario_count=10) for t,n in zip([5,6,7],[0,1,2])]
        r=window_metrics(rows,[5,6,7],1)
        self.assertEqual(r['min_N_d'],0);self.assertAlmostEqual(r['fraction_N_ge_k'],2/3)
        self.assertAlmostEqual(r['plugin_C_sampling'],(1-.8**10)**3)
        self.assertEqual(window_metrics(rows[:2],[5,6,7],1)['plugin_C_sampling'],'')
        with self.assertRaises(ValueError):window_metrics(rows+rows[:1],[5,6,7],1)
    def test_targets_selected_without_outcomes(self):
        rows=[dict(danger_count=i,S=10,delta_C=g,collision=1) for i,g in enumerate([.02,.05,.10,.20])]
        chosen=select(rows,[.2,.1,.05,.02]);self.assertEqual([r['delta_C'] for r in chosen],[.2,.1,.05,.02])
        for r in rows:r['collision']=0
        self.assertEqual([r['danger_count'] for r in select(rows,[.2,.1,.05,.02])],[3,2,1,0])
    def test_old_cells_excluded_and_window_required(self):
        weights=[dict(danger_count=20,mode='across',step=t,p=.02,q=.05) for t in [5,6]]
        protocol=dict(old_cells=[[20,10]],candidate_danger_counts=[20],candidate_budgets=[10,20],critical_window=[5,6],kappa=1)
        self.assertEqual([r['S'] for r in candidates(weights,protocol)],[20])
        with self.assertRaises(ValueError):candidates(weights[:1],protocol)
    def test_exact_pair_test(self):
        self.assertEqual(mcnemar(0,0),1)
        self.assertAlmostEqual(mcnemar(12,0),2/2**12)
    def test_ordering_retains_shared_seed_clusters(self):
        selected=[dict(cell='high',delta_C=.2),dict(cell='low',delta_C=.02)]
        pairs=[dict(cell=c,seed=str(s),nominal_collision=int(c=='low'),wdro_collision=0) for c in ['high','low'] for s in range(5)]
        result,contrasts=ordering_check(selected,pairs,100)
        self.assertFalse(result['observed_nondecreasing_benefit_with_gain'])
        self.assertEqual(result['bootstrap_fraction_ordered'],0)
        self.assertEqual(contrasts[0]['bootstrap95_low'],-1)
if __name__=='__main__':unittest.main()
