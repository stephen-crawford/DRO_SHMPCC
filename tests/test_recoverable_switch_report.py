import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_recoverable_switch import outcome,paired_metric

class RecoverabilityReportTests(unittest.TestCase):
    def test_refusal_and_limit_are_not_completion(self):
        for termination in ['no_admissible_control','step_limit']:
            self.assertEqual(outcome(dict(status='OK',collision='0',path_completed='0',termination=termination)),termination)
    def test_errors_remain_errors_and_collision_precedes_completion(self):
        self.assertEqual(outcome(dict(status='ERROR')),'execution_error')
        self.assertEqual(outcome(dict(status='OK',collision='1',path_completed='1')),'collision')
        self.assertEqual(outcome(dict(status='OK',collision='0',path_completed='1')),'completed')
    def test_paired_tests_keep_explicit_error_denominator(self):
        rows=[dict(nominal_outcome='execution_error',wdro_outcome='completed'),
              dict(nominal_outcome='collision',wdro_outcome='completed',nominal_path_completed='0',wdro_path_completed='1')]
        r=paired_metric(rows,'path_completed')
        self.assertEqual((r['paired_eligible'],r['excluded_pairs'],r['nominal_only'],r['wdro_only']),(1,1,0,1))

if __name__=='__main__':unittest.main()
