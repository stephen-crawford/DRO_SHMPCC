"""Regressions for the experiment's selection and inference, not controller math."""
import csv
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from causal_link_analysis import undercoverage, wilson, select_separation, qualify_braking, frozen_report, write, read
spec=importlib.util.spec_from_file_location('causal_runner',ROOT/'tests/run_causal_link.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class CausalAnalysisTests(unittest.TestCase):
    def test_binomial_exact_small_enumeration(self):
        import itertools
        for p in [0,.05,.5,1]:
            for k in [1,2,3]:
                exact=sum(math.prod(p if bit else 1-p for bit in seq)
                          for seq in itertools.product([0,1],repeat=4) if sum(seq)<k)
                self.assertAlmostEqual(undercoverage(p,k,4),exact)
        self.assertAlmostEqual(undercoverage(.1,1,20),.9**20)
        self.assertLess(undercoverage(.181,2,20),.11)

    def test_more_mass_reduces_undercoverage(self):
        for k in [1,2,3]:
            self.assertLess(undercoverage(.2,k,20),undercoverage(.05,k,20))

    def test_intervals_and_missing_data(self):
        self.assertEqual(wilson(0,0),('',''))
        lo,hi=wilson(0,100)
        self.assertAlmostEqual(lo,0)
        self.assertGreater(hi,0)
        self.assertLess(hi,.05)

    def test_outcome_blind_selection(self):
        rows=[dict(candidate=str(i),r=1,delta_u2=i/10,collision=1-i/10) for i in range(10)]
        chosen=select_separation(rows)
        self.assertEqual([r['candidate'] for r in chosen],['9','5','0'])
        for row in rows:row['collision']=100
        self.assertEqual([r['candidate'] for r in select_separation(rows)],['9','5','0'])

    def test_braking_gate_rejects_attrition_and_errors(self):
        rows=[dict(arm=arm,status='OK',switch_reached=1,collision=int(i<6),termination='step_limit')
              for arm in ['sh_mpcc','sh_mpcc_dro'] for i in range(20)]
        self.assertTrue(qualify_braking(rows,20))
        rows[0]['switch_reached']=0;rows[1]['switch_reached']=0
        self.assertFalse(qualify_braking(rows,20))
        rows[0]['switch_reached']=1;rows[1]['switch_reached']=1;rows[0]['status']='ERROR'
        self.assertFalse(qualify_braking(rows,20))

    def test_gamma_missing_strata_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'weights.csv',[dict(mode='across',p=.1,q=.2,S=20)])
            write(p/'draws.csv',[dict(law=law,S=20,n_d=0) for law in ['nominal','wdro']])
            write(p/'conditional.csv',[dict(law=law,n_d=0,collision=0,refusal=1,horizon_completed=0) for law in ['nominal','wdro']])
            frozen_report(p)
            good=next(r for r in read(p/'gamma_estimates.csv') if r['law']=='nominal' and r['k']=='1' and r['group']=='good')
            self.assertAlmostEqual(float(good['partial_identification_high']),1)
            self.assertGreater(float(good['missing_mass']),0)

    def test_decimal_geometry_names_have_distinct_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'binary';binary.write_text('fixture')
            with patch.object(runner.subprocess,'run') as execute:
                execute.return_value.returncode=0
                for g in [100,300]:runner.invoke(binary,root/f'x2.0_y1.2_dy-0.04_g{g}',[g])
                self.assertEqual(execute.call_count,2)
            self.assertEqual(len(list(root.glob('*.done.json'))),2)

if __name__=='__main__':unittest.main()
