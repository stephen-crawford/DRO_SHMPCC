"""Actual five-step intervention and paired trajectory-bank regression."""
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
class PersistentTests(unittest.TestCase):
    def test_exact_window_and_paired_trajectory_slots(self):
        binary=ROOT/'build-base/persistent_coverage_experiment'
        with tempfile.TemporaryDirectory() as temp:
            by_count={}
            for n in [0,1,3]:
                out=Path(temp)/f'n{n}'
                result=subprocess.run([str(binary),str(out),'late_switch','20','10','7001','0','8','5','sh_mpcc','7','5',str(n)],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr[-1500:])
                modes=[r for r in read(out/'mode_mechanism.csv') if r['obstacle_id']=='0' and r['mode']=='across']
                self.assertEqual([int(r['sampled_count']) for r in modes[:5]],[n]*5)
                self.assertGreaterEqual(len(modes),6)
                self.assertTrue(all(r['certificate_requested']=='0' and r['certified']=='0' for r in read(out/'decisions.csv')))
                by_count[n]={(r['step'],r['slot'],r['obstacle_id'],r['k']):r for r in read(out/'paired_slots.csv')}
            for n in [1,3]:
                for key,row in by_count[0].items():
                    step,slot,obs,k=map(int,key)
                    if step>=5 or obs!=0 or slot<10-n:
                        self.assertEqual(row,by_count[n][key],f'paired realization changed: {n}, {key}')

if __name__=='__main__':unittest.main()
