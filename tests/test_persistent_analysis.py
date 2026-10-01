import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_persistent_coverage import measurements
class PersistentAnalysisTests(unittest.TestCase):
    def test_preswitch_across_is_retained_even_when_not_true_mode(self):
        row=dict(mode='across',obstacle_id='0',true_mode='continue',nominal_probability='.02',sampling_probability='.1',scenario_count='10',sampled_count='1',risk_score='.5',step='3',switch_step='10')
        result=measurements([row])[0]
        self.assertEqual(result['before_switch'],1)
        self.assertEqual(result['N_d'],1)
        self.assertAlmostEqual(result['delta_U_k1'],.98**10-.9**10)
        self.assertEqual(result['r_d'],'.5')
    def test_nominal_missing_risk_not_invented(self):
        row=dict(mode='across',obstacle_id='0',nominal_probability='.05',sampling_probability='.05',scenario_count='20',sampled_count='0',step='10',switch_step='10')
        result=measurements([row])[0]
        self.assertEqual(result['r_d'],'')
        self.assertEqual(result['delta_U_k2'],0)
        self.assertEqual(result['before_switch'],0)
if __name__=='__main__':unittest.main()
