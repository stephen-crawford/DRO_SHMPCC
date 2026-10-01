"""Regression: empty sampled free-space must be recorded without executing a control."""
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as file:return list(csv.DictReader(file))


class LongStressPilotTests(unittest.TestCase):
    def test_uncertified_empty_polygon_is_a_recorded_rejection(self):
        runner=ROOT/'build-base/risk_stress_experiment'
        if not runner.exists():self.skipTest('build risk_stress_experiment first')
        with tempfile.TemporaryDirectory() as temporary:
            for arm in ['sh_mpcc','sh_mpcc_dro','sh_mpcc_resample','sh_mpcc_dro_fallback','sh_mpcc_extra']:
                with self.subTest(arm=arm):
                    output=Path(temporary)/arm
                    result=subprocess.run([str(runner),str(output),'rare_turn','50','20','77','5','40','0',arm],
                                          cwd=ROOT,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr[-1200:])
                    summary=rows(output/'summary.csv')[0]
                    self.assertEqual(summary['status'],'OK')
                    decisions=rows(output/'decisions.csv')
                    self.assertEqual(int(summary['decisions']),len(decisions))
                    self.assertGreaterEqual(len(decisions),17)
                    self.assertTrue(all(r['certified']=='0' and r['certificate_requested']=='0' for r in decisions))
                    # Rejection is not successful control, and must never manufacture an input.
                    for decision in decisions:
                        if decision['success']=='0':
                            self.assertEqual(decision['acceleration'],'')
                            self.assertEqual(decision['omega'],'')
                    if arm in ('sh_mpcc','sh_mpcc_dro'):
                        self.assertEqual(summary['termination'],'no_admissible_control')
                        self.assertEqual(decisions[-1]['step'],'16')
                        self.assertEqual(decisions[-1]['success'],'0')
                    elif arm in ('sh_mpcc_resample','sh_mpcc_dro_fallback'):
                        self.assertEqual(next(r for r in decisions if r['step']=='16')['nominal_fallback_attempted'],'1')


if __name__=='__main__':unittest.main()
