#!/usr/bin/env python3
"""Regression tests for reviewer experiment controls and report denominators."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_reviewer_matrix as suite
import run_analysis_matrix as analysis

sys.dont_write_bytecode = True
spec=importlib.util.spec_from_file_location('reviewer_report',suite.ROOT/'tools/report_reviewer_experiments.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.settings=suite.load_settings(suite.ROOT/'configs/reviewer_tests/settings.json')

    def test_ablations_preserve_budget_and_only_change_intended_keys(self):
        cases=[c for c in suite.configurations(self.settings) if c['pair']=='standard_straight_o2_c2_m3']
        configs={c['solver_style']:{k:v.strip() for line in suite.config_text(c,self.settings).splitlines()
                 if line and not line.startswith('#') for k,v in [line.split(':',1)]} for c in cases}
        reference=configs['sh_mpcc_dro_fallback']
        allowed={
            'sh_mpcc':{'mpc_type','dro_enabled'},'sh_mpcc_dro':{'mpc_type'},
            'sh_mpcc_resample':{'nominal_resampling_baseline','dro_enabled'},
            'sh_mpcc_dro_fallback':set(), 'hybrid_zero_one':{'ground_cost'},
            'hybrid_fixed_radius':{'fixed_rho','use_calibrated_radius'}}
        for method,cfg in configs.items():
            changed={k for k in cfg if cfg[k]!=reference[k]}-{'method_name','scenario_tag'}
            self.assertEqual(changed,allowed[method])
            self.assertEqual(cfg['automatically_compute_sample_size'],'true')
        self.assertEqual(len(list(suite.configurations(self.settings))),2880)

    def test_stress_profiles_do_not_change_support_or_controller_budget(self):
        cases=[c for c in suite.configurations(self.settings) if c['solver_style']=='sh_mpcc_resample'
               and c['environment']=='straight' and c['obstacles']==2 and c['classes']==2 and c['modes_per_class']==3]
        configs=[{k:v for line in suite.config_text(c,self.settings).splitlines() if line and not line.startswith('#')
                  for k,v in [line.split(':',1)]} for c in cases]
        self.assertEqual({k for k in configs[0] if configs[0][k]!=configs[1][k]},
                         {'shift_psi','shift_boost','scenario_tag'})

    def test_complete_case_pairing_and_complementarity(self):
        trials=[]
        for seed,a,b in [(1,0,0),(2,1,0),(3,0,1),(4,1,1)]:
            for method,failure in [('a',a),('b',b)]:
                trials.append(dict(case=method+str(seed),seed=seed,environment='straight',obstacles=1,
                    classes=1,modes_per_class=2,solver_style=method,status='OK',repeats=[{'metrics':dict(
                    termination_reason='no_admissible_control' if failure else 'path_complete',
                    collision=0,completed_path=1-failure)}]))
        failed=copy.deepcopy(trials[0]);failed.update(seed=5,status='ERROR');trials.append(failed)
        rates,overlap,matched,excluded=report.paired_outcomes(trials,['a','b'],100)
        self.assertEqual((len(matched),excluded),(4,1))
        for field in ['neither_refuses','only_a_refuses','only_b_refuses','both_refuse']:
            self.assertEqual(overlap[0][field],1)
        self.assertEqual(rates[0]['rate'],.5)
        self.assertEqual(overlap[0]['rate_difference_b_minus_a'],0)
        with self.assertRaises(ValueError):report.paired_outcomes(trials+[trials[0]],['a','b'],100)

    def test_seed_block_bootstrap_preserves_within_seed_dependence(self):
        self.assertEqual(report.seed_block_interval([(77,0),(77,1)],100),(None,None))
        data=[(77,0),(77,0),(78,1),(78,1)]
        self.assertEqual(report.seed_block_interval(data,100),report.seed_block_interval(data,100))
        self.assertEqual(report.seed_block_interval(data,100),(0.,1.))

    def test_cycle_latency_and_strict_clearance_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            analysis.write_csv(p/'decisions.csv',[
                dict(solve_ms=t,nominal_fallback_attempted=f,scenario_count=12,success=s)
                for t,f,s in [(10,0,1),(110,1,1),(210,1,0)]])
            analysis.write_csv(p/'conservatism.csv',[{'margin_m':m} for m in [.3,0.,-.05]])
            analysis.write_csv(p/'geometry.csv',[dict(actor='ego',safety_margin=.1)])
            analysis.write_csv(p/'trace.csv',[dict(step=i,actor='obstacle',obstacle_id=0,mode=m)
                                             for i,m in enumerate(['left','left','right'])])
            m,*_=report.bundle_metrics(p,100,.2)
            self.assertEqual(m['cycle_median_ms'],110)
            self.assertEqual(m['fallback_cycle_median_ms'],160)
            self.assertEqual(m['deadline_misses'],2)
            self.assertEqual(m['maximum_safety_penetration_m'],.05)
            self.assertEqual(m['maximum_physical_penetration_m'],0)
            self.assertAlmostEqual(m['sampled_state_safety_violation_fraction'],1/3)
            self.assertEqual(m['observed_mode_repeat_fraction'],.5)
            self.assertEqual(m['scenarios_min'],12)
            # Old artifacts without flags must report unknown fallback counts.
            d=analysis.rows(p/'decisions.csv')
            for row in d:row.pop('nominal_fallback_attempted')
            analysis.write_csv(p/'decisions.csv',d)
            self.assertIsNone(report.bundle_metrics(p,100,.2)[0]['fallback_fraction'])


def integration(runner):
    with tempfile.TemporaryDirectory(prefix='reviewer-tests-') as tmp:
        p=Path(tmp)
        settings=json.loads((suite.ROOT/'configs/reviewer_tests/settings.json').read_text())
        settings.update(obstacle_counts=[1],class_counts=[1],environments=['straight'],mode_counts=[3],
                        seeds=[77],repeats=2)
        settings['overrides'].update(horizon=4,num_scenarios=8,automatically_compute_sample_size=False,
                                     rollout_steps=3)
        path=p/'settings.json';path.write_text(json.dumps(settings))
        command=['python3',str(Path(suite.__file__)), '--settings',str(path),'--runner',str(runner),
                 '--output',str(p/'matrix'),'--timeout','60']
        subprocess.run(command,check=True)
        results=json.loads((p/'matrix/results.json').read_text())
        assert len(results)==12 and all(r['status']=='OK' for r in results)
        assert all(r['repeatable'] for r in results)
        for r in results:
            bundle=p/'matrix'/r['case']/'seed_77/repeat_0'
            config=(bundle/'resolved_config.yaml').read_text()
            nominal=r['solver_style']=='sh_mpcc_resample'
            assert f'nominal_resampling_baseline: {str(nominal).lower()}\n' in config
            if nominal: assert 'dro_enabled: false\n' in config
            assert {int(x['scenario_count']) for x in analysis.rows(bundle/'decisions.csv')}=={8}
        assert all(r['status']=='OK' for r in analysis.rows(p/'matrix/pairing.csv'))
        nominal=next(r for r in results if r['solver_style']=='sh_mpcc_resample')
        source=p/'matrix'/nominal['case']/'seed_77/repeat_0'
        subprocess.run([str(runner),'--config',str(source/'resolved_config.yaml'),
                        '--seed','77','--output',str(p/'replay'),'--label','resolved'],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        replay=p/'replay/resolved'
        replay_metrics=analysis.analyze(replay,nominal)
        assert analysis.repeat_signature(replay,replay_metrics)==nominal['repeats'][0]['signature']
        before={f:str(f.stat().st_mtime_ns) for f in (p/'matrix').glob('*/seed_*/repeat_0/trace.csv')}
        subprocess.run(command+['--resume'],check=True)
        assert before=={f:str(f.stat().st_mtime_ns) for f in before}
        report.main(['--matrix',str(p/'matrix'),'--output',str(p/'report'),'--methods',*settings['variants'],
                     '--bootstrap-draws','100'])
        assert len(report.rows(p/'report/rollout_diagnostics.csv'))==12
        # Frozen manifest refuses changes instead of mixing old and new experiments.
        settings['fixed_radius']=.08;path.write_text(json.dumps(settings))
        assert subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        print('PASS: 12 variant/profile trials × 2 repeats, pairing, config replay, equal budgets, resume, reporting, stale-manifest rejection')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runner',type=Path)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewerTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.runner:integration(args.runner.resolve())
