#!/usr/bin/env python3
"""Tests for fail-loud mismatch generation and complete mechanism evidence."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_comparison_matrix as matrix
import comparison_evidence as evidence


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.settings=matrix.load_settings(matrix.ROOT/'configs/comparison_matrix/sample_efficiency.json')

    def test_missing_mismatch_key_is_an_error(self):
        case=next(matrix.configurations(self.settings))
        original=matrix.analysis.config_text
        def missing(*args,**kwargs):
            return '\n'.join(line for line in original(*args,**kwargs).splitlines() if not line.startswith('shift_boost:'))
        with patch.object(matrix.analysis,'config_text',missing):
            with self.assertRaisesRegex(ValueError,'shift_boost'):matrix.config_text(case,self.settings)

    def test_uncertified_fixed_budget_configuration(self):
        settings=matrix.load_settings(matrix.ROOT/'configs/comparison_matrix/fixed_budget_uncertified.json')
        cases=list(matrix.configurations(settings))
        self.assertEqual(len(cases),48)
        self.assertEqual(settings['repeats'],2)
        self.assertEqual({c['solver_style'] for c in cases},{'sh_mpcc','sh_mpcc_dro'})
        for case in cases:
            text=matrix.config_text(case,settings)
            self.assertIn('safe_horizon_enabled: false\n',text)
            self.assertIn('automatically_compute_sample_size: false\n',text)
            self.assertIn('num_scenarios: 40\n',text)

    def test_counts_budgets_and_graded_profiles(self):
        cases=list(matrix.configurations(self.settings))
        self.assertEqual(len(cases),360)
        self.assertEqual(len({c['pair'] for c in cases}),72)
        self.assertEqual(self.settings['repeats'],2)
        for c in cases:
            text=matrix.config_text(c,self.settings)
            expected=c['scenario_budget']*(2 if c['solver_style']=='sh_mpcc_extra' else 1)
            self.assertIn(f'num_scenarios: {expected}\n',text)
            self.assertIn('automatically_compute_sample_size: false\n',text)
            self.assertIn('artifact_capture_attempt_diagnostics: true\n',text)
        self.assertEqual([p['shift_boost'] for p in self.settings['mismatch_profiles']],[0,.05,.15,.30])

    def test_exact_discordance_and_no_collision_not_completion(self):
        self.assertEqual(evidence.exact_discordance_p(0,0),1.)
        self.assertEqual(evidence.exact_discordance_p(0,5),.0625)
        self.assertEqual(evidence.exact_discordance_p(5,0),.0625)
        self.assertEqual(evidence.exact_discordance_p(2,2),1.)

    def test_vertex_reachability_and_zero_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            modes=[dict(step=0,attempt=0,obstacle_id=0,mode=m,nominal_probability=p,
                        sampling_probability=q,risk_score=r,rho=.5)
                   for m,p,q,r in [('a',.75,.5,0),('b',.25,.5,1)]]
            edges=[dict(step=0,attempt=0,obstacle_id=0,source_mode=i,target_mode=j,
                        cost=0 if i==j else 2,radius_observation_count=100)
                   for i in ['a','b'] for j in ['a','b']]
            matrix.analysis.write_csv(root/'mode_mechanism.csv',modes)
            matrix.analysis.write_csv(root/'transport_costs.csv',edges)
            vertices,solves=evidence.concentration_rows(root)
            self.assertEqual([r['vertex_distance'] for r in vertices],[.5,1.5])
            self.assertEqual([r['vertex_reachable'] for r in vertices],[1,0])
            self.assertEqual(solves[0]['rho_over_min_vertex_distance'],1)
            self.assertEqual(solves[0]['q_near_one'],0)
            for r in edges:r['cost']=0
            modes[0]['sampling_probability']=1;modes[1]['sampling_probability']=0
            matrix.analysis.write_csv(root/'mode_mechanism.csv',modes)
            matrix.analysis.write_csv(root/'transport_costs.csv',edges)
            _,solves=evidence.concentration_rows(root)
            self.assertIsNone(solves[0]['rho_over_min_vertex_distance'])
            self.assertEqual(solves[0]['zero_min_vertex_distance'],1)
            self.assertEqual(solves[0]['q_near_one'],1)
            self.assertEqual(solves[0]['any_vertex_reachable'],1)


def integration(runner):
    with tempfile.TemporaryDirectory(prefix='sample-efficiency-') as tmp:
        root=Path(tmp)
        s=json.loads((matrix.ROOT/'configs/comparison_matrix/sample_efficiency.json').read_text())
        s.update(obstacle_counts=[1],class_counts=[1],environments=['straight'],seeds=[77],repeats=2,
                 scenario_budgets=[4,8])
        s['overrides'].update(rollout_steps=3,horizon=4)
        config=root/'settings.json';config.write_text(json.dumps(s))
        cmd=[sys.executable,str(Path(matrix.__file__)), '--settings',str(config),'--runner',str(runner),'--output',str(root/'matrix')]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        pairs=json.loads((root/'matrix/all_comparisons.json').read_text())
        assert len(pairs)==80 and all(r['status']=='OK' for r in pairs)
        assert all(r['paired_nominal_prefix_equal'] and r['paired_obstacle_prefix_equal'] for r in pairs)
        results=json.loads((root/'matrix/results.json').read_text())
        assert len(results)==40 and all(r['repeatable'] for r in results)
        summary=matrix.analysis.rows(root/'matrix/mechanism_summary.csv')
        assert len(summary)==40, 'setup/budget summaries must not be pooled'
        assert {r['scenario_budget'] for r in summary}=={'4','8'}
        assert {r['arm'] for r in summary}==set('ABCDE')
        concentration=matrix.analysis.rows(root/'matrix/concentration_per_solve.csv')
        assert concentration and {r['controller'] for r in concentration}=={'sh_mpcc_dro','sh_mpcc_dro_fallback'}
        assert all(float(r['rho'])>=0 and 0<=float(r['max_q'])<=1 for r in concentration)
        per_seed={r['case']:r for r in matrix.analysis.rows(root/'matrix/mechanism_per_seed.csv')}
        for r in results:
            bundle=matrix.analysis.trial_root(root/'matrix',r,77)/'repeat_0'
            mm=matrix.analysis.rows(bundle/'mode_mechanism.csv')
            aa=matrix.analysis.rows(bundle/'attempts.csv')
            assert mm and aa
            decisions=matrix.analysis.rows(bundle/'decisions.csv')
            assert int(per_seed[r['case']]['final_inadmissible'])==sum(d['success']=='0' for d in decisions)
            sr=next(d for d in summary if d['pair']==r['pair'] and d['controller']==r['solver_style'])
            assert float(sr['final_admissibility'])==1-int(per_seed[r['case']]['final_inadmissible'])/len(decisions)
            if r['solver_style']=='sh_mpcc_extra':assert all(int(a['scenario_count'])==2*r['scenario_budget'] for a in aa)
            # Preserve actual baseline-vs-DRO logs, including risk/rho missing for nominal.
            if r['solver_style'] in ['sh_mpcc','sh_mpcc_extra','sh_mpcc_resample']:
                assert all(m['risk_score']=='' and m['rho']=='' for m in mm)
            if r['solver_style']=='sh_mpcc_dro':assert all(m['risk_score']!='' and m['rho']!='' for m in mm)
        original=(root/'matrix/primary_summary.csv').read_bytes()
        subprocess.run(cmd+['--resume'],check=True,stdout=subprocess.DEVNULL)
        assert original==(root/'matrix/primary_summary.csv').read_bytes()
        # Timing may vary, but modified numeric attempt evidence must fail on resume.
        attempt_file=matrix.analysis.trial_root(root/'matrix',results[0],77)/'repeat_1/attempts.csv'
        recorded=matrix.analysis.rows(attempt_file)
        recorded[0]['solve_ms']=str(float(recorded[0]['solve_ms'])+1)
        matrix.analysis.write_csv(attempt_file,recorded)
        subprocess.run(cmd+['--resume'],check=True,stdout=subprocess.DEVNULL)
        recorded[0]['qp_calls']=str(int(recorded[0]['qp_calls'])+1)
        matrix.analysis.write_csv(attempt_file,recorded)
        rejected=subprocess.run(cmd+['--resume'],stdout=subprocess.DEVNULL)
        assert rejected.returncode!=0,'resume accepted altered numerical evidence'
        errors=json.loads((root/'matrix/results.json').read_text())
        assert any('repeat signature' in r.get('error','') for r in errors)
        subprocess.run(cmd+['--resume'],check=True,stdout=subprocess.DEVNULL)
        assert original==(root/'matrix/primary_summary.csv').read_bytes()
        dro_trial=next(r for r in results if r['solver_style']=='sh_mpcc_dro')
        geometry_file=matrix.analysis.trial_root(root/'matrix',dro_trial,77)/'repeat_1/transport_costs.csv'
        geometry=matrix.analysis.rows(geometry_file)
        geometry[0]['cost']=str(float(geometry[0]['cost'])+1)
        matrix.analysis.write_csv(geometry_file,geometry)
        assert subprocess.run(cmd+['--resume'],stdout=subprocess.DEVNULL).returncode!=0
        subprocess.run(cmd+['--resume'],check=True,stdout=subprocess.DEVNULL)
        print('PASS: 40 trials × 2 repeats, all five policies, four severities, two budgets, 80 pair checks, nominal/plant evidence, resume')
        print('PASS: stratified reports, final admissibility, timing excluded, tampered resume rejected and rerun')
        print('PASS: concentration evidence present for WDRO only; altered transport geometry rejected on resume')


def check_rare_artifacts(root):
    read=lambda name:list(csv.DictReader((root/name).open()))
    weights={r['mode']:r for r in read('weights.csv')}
    assert float(weights['cut_in']['reference_safety_margin'])<0
    assert float(weights['cut_in']['wdro_probability'])>float(weights['cut_in']['nominal_probability'])
    trials=read('coverage_trials.csv');groups={}
    for r in trials:groups.setdefault((r['budget'],r['scheme']),[]).append(r)
    for group in groups.values():
        n=len(group);p=float(group[0]['expected_inclusion'])
        observed=sum(int(r['represented']) for r in group)/n
        assert abs(observed-p)<=6*math.sqrt(p*(1-p)/n)+1/n
        assert all(r['draws']==r['budget'] for r in group)
    keyed={(r['budget'],r['seed'],r['scheme']):r for r in trials}
    for budget,seed,scheme in keyed:
        if scheme=='nominal_single':
            assert keyed[budget,seed,scheme]['cut_in_count']==keyed[budget,seed,'nominal_split']['cut_in_count']
    cycles=read('controller_trials.csv');attempts=read('attempts.csv')
    for r in cycles:
        assert None not in r,'malformed CSV row'
        if r['status']!='OK':continue
        aa=[a for a in attempts if all(a[k]==r[k] for k in ['budget_design','base_S','seed','method'])]
        assert len(aa)==int(r['outer_attempts'])==1+int(r['fallback'])
        assert sum(int(a['draws']) for a in aa)==int(r['total_draws'])
        assert int(r['total_draws'])<=2*int(r['base_S'])
        assert aa[0]['success']==r['first_success'] and aa[-1]['success']==r['final_success']
    print(f'PASS: analytic inclusion checks, exact split nominal budget, and {len(cycles)} controller records')


def check_uncertified_artifacts(root):
    results=json.loads((root/'results.json').read_text())
    pairs=json.loads((root/'all_comparisons.json').read_text())
    assert results and all(r['status']=='OK' and r['repeatable'] for r in results)
    assert pairs and all(r['status']=='OK' for r in pairs)
    decisions=attempts=0
    for result in results:
        for repeat in range(len(result['repeats'])):
            bundle=matrix.analysis.trial_root(root,result,result['seed'])/f'repeat_{repeat}'
            config=evidence.flat_yaml(bundle/'resolved_config.yaml')
            assert config['safe_horizon_enabled'] is False
            assert config['automatically_compute_sample_size'] is False
            dd=matrix.analysis.rows(bundle/'decisions.csv')
            aa=matrix.analysis.rows(bundle/'attempts.csv')
            assert len(dd)==len(aa),'unexpected outer retry'
            assert all(d['certificate_requested']=='0' and d['certified']=='0' for d in dd)
            assert all(int(a['scenario_count'])==config['num_scenarios']==40 for a in aa)
            decisions+=len(dd);attempts+=len(aa)
    summary=matrix.analysis.rows(root/'mechanism_summary.csv')
    assert summary and all(r['certification_status']=='not_requested' for r in summary)
    print(f'PASS: {len(results)} repeatable trials, {len(pairs)} paired comparisons, '
          f'{decisions} decisions/{attempts} attempts at exactly 40 scenarios, zero certification requests/certificates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',type=Path);p.add_argument('--rare-artifacts',type=Path)
    p.add_argument('--uncertified-artifacts',type=Path)
    args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.runner:integration(args.runner.resolve())
    if args.rare_artifacts:check_rare_artifacts(args.rare_artifacts)
    if args.uncertified_artifacts:check_uncertified_artifacts(args.uncertified_artifacts)
