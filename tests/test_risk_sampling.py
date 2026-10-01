"""Numerical fixtures for both feedback reports; no favorable effect is assumed."""
import copy
import itertools
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_risk_sampling as risk
import run_comparison_matrix as matrix


class RiskSamplingTests(unittest.TestCase):
    def rows(self):
        return [dict(mode=m,nominal_probability=p,sampling_probability=q,risk_score=r,
                     sampled_count=n,scenario_count=20,true_mode='a')
                for m,p,q,r,n in [('a',.5,0.,1.,0),('b',.2,0.,2.,0),('c',.3,1.,3.,20)]]

    def test_vertex_and_risk_metrics(self):
        result=risk.allocation(self.rows())
        self.assertEqual(result['support_q'],1)
        self.assertEqual(result['zero_mass_modes'],2)
        self.assertEqual(result['risk_coverage'],.5)
        self.assertEqual(result['empirical_risk'],3)
        self.assertEqual(result['risk_error'],0)
        self.assertEqual(result['top_count'],20)
        self.assertEqual(result['top_delta_q'],.7)
        self.assertEqual(result['missed_true_zero_mass'],1)
        self.assertEqual(result['true_p'],.5)

    def test_ties_zero_risk_and_exact_support(self):
        rows=self.rows()
        for r in rows:r['risk_score']=0
        result=risk.allocation(rows)
        self.assertEqual(result['risk_coverage'],'')
        self.assertEqual(result['top_mode'],'')
        rows[0]['risk_score']=rows[2]['risk_score']=3
        self.assertEqual(risk.allocation(rows)['top_mode'],'')
        rows[0]['sampling_probability']=1e-14
        rows[2]['sampling_probability']=1-1e-14
        self.assertEqual(risk.allocation(rows)['support_q'],2)

    def test_invalid_evidence(self):
        for field,value in [('risk_score','nan'),('sampled_count',1.5),('scenario_count',21),
                            ('sampling_probability',-.1),('true_mode','missing')]:
            rows=self.rows();rows[0][field]=value
            with self.assertRaises(ValueError):risk.allocation(rows)
        with self.assertRaises(ValueError):risk.allocation(self.rows()+self.rows()[:1])
        with self.assertRaises(ValueError):risk.allocation(self.rows()[:2])

    def test_presets_and_existing_arm_configs(self):
        root=Path(__file__).resolve().parents[1]
        settings=matrix.load_settings(root/'configs/causal_matrix/hard_mismatch.json')
        cases=list(matrix.configurations(settings))
        self.assertEqual(len(cases),120)
        self.assertEqual({c['obstacles'] for c in cases},{2})
        self.assertEqual({c['scenario_budget'] for c in cases},{20,40,80})
        strat=matrix.load_settings(root/'configs/causal_matrix/stratified_ablation.json')
        scases=list(matrix.configurations(strat))
        self.assertEqual(len(scases),144)
        before={c['case']:matrix.config_text(c,settings) for c in cases}
        for c in scases:
            text=matrix.config_text(c,strat)
            if c['solver_style']=='sh_mpcc_dro_stratified':
                self.assertIn('wdro_stratified_sampling: true\n',text)
                self.assertIn('safe_horizon_enabled: false\n',text)
                self.assertIn('dro_enabled: true\n',text)
            else:self.assertEqual(text,before[c['case']])

    def test_frozen_rare_history_sweep(self):
        root=Path(__file__).resolve().parents[1]
        runner=root/'build-base/rare_mode_experiment'
        if not runner.exists():self.skipTest('build rare_mode_experiment first')
        with tempfile.TemporaryDirectory() as tmp:
            for count in [100,50,20,10]:
                output=Path(tmp)/str(count)
                subprocess.run([str(runner),str(output),'16','1',str(count)],cwd=root,
                               check=True,capture_output=True)
                weights=risk.read(output/'weights.csv')
                danger=next(r for r in weights if r['mode']=='cut_in')
                self.assertEqual(int(danger['count']),count)
                self.assertEqual(sum(int(r['count']) for r in weights),1000)
                p=float(danger['nominal_probability']);q=float(danger['wdro_probability'])
                for row in risk.read(output/'coverage_trials.csv'):
                    budget=int(row['budget'])
                    self.assertEqual(int(row['draws']),budget)
                    if row['scheme']=='wdro_stratified':
                        self.assertEqual(row['represented'],'1')
                        self.assertGreaterEqual(int(row['cut_in_count']),1)
                        self.assertLessEqual(int(row['cut_in_count']),budget-2)
                        self.assertEqual(float(row['expected_inclusion']),1)
                    elif row['scheme'] in ('nominal_single','wdro_single'):
                        probability=p if row['scheme']=='nominal_single' else q
                        self.assertAlmostEqual(float(row['expected_inclusion']),1-(1-probability)**budget)

    def test_matched_events_recovery_repeats_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'out'
            runs=[];modes=[];decisions=[];checks=[]
            for seed in range(4):
                for repeat in ['0','1']:
                    identity=dict(test_root='fixture',matrix_identity='id',pair_case='hard',seed=str(seed),repeat=repeat)
                    for style,variant in risk.ARMS.items():
                        runs.append(dict(identity,variant=variant,trial_status='OK',log_complete='1',
                            path_completed=int(style=='sh_mpcc_dro_fallback'),collision=0,min_actual_clearance=.1))
                        for step in ['0','1']:
                            for obstacle in ['0','1']:
                                for row in self.rows():
                                    row=copy.copy(row)
                                    row['true_mode']='a' if step=='0' else 'c'
                                    row['reference_risk']=1 if row['mode']=='a' else 0
                                    if style!='sh_mpcc_dro':
                                        row['sampled_count']={'a':10,'b':4,'c':6}[row['mode']]
                                        row['sampling_probability']=row['nominal_probability']
                                        row['risk_score']=''
                                    modes.append(dict(identity,solver_style=style,step=step,attempt='0',obstacle_id=obstacle,**row))
                            decisions.append(dict(identity,solver_style=style,step=step,success=1))
                for a,b in itertools.combinations(risk.ARMS,2):
                    checks.append(dict(test_root='fixture',matrix_identity='id',pair='hard',seed=str(seed),controller_a=a,controller_b=b,status='OK'))
            for name,rows in [('run_summary',runs),('artifact_mode_mechanism',modes),('artifact_decisions',decisions),('report_all_comparisons',checks)]:
                risk.write(source/(name+'.csv'),rows,[])
            risk.analyze(source,out)
            self.assertEqual(len(risk.read(out/'risk_per_decision.csv')),16)
            events=risk.read(out/'matched_events.csv')
            switched=[r for r in events if r['event']=='dangerous_switch']
            self.assertEqual(len(switched),16)
            self.assertTrue(all(r['step']=='1' for r in switched))
            self.assertEqual({r['repeat'] for r in events},{'0'})
            outcomes=[r for r in risk.read(out/'event_rollout_outcomes.csv') if r['event']=='dangerous_switch']
            self.assertEqual(len(outcomes),8)  # two obstacles must not duplicate rollout outcomes
            step_summaries=[r for r in risk.read(out/'event_decisions_per_seed.csv') if r['event']=='dangerous_switch']
            self.assertTrue(all(r['observations']=='1' for r in step_summaries))
            recovery=next(r for r in risk.read(out/'ablation_pairs.csv') if r['controller_a']=='sh_mpcc_dro' and r['controller_b']=='sh_mpcc_dro_fallback' and r['metric']=='path_completed')
            self.assertEqual(recovery['paired_seeds'],'4')
            self.assertEqual(recovery['n01'],'4')
            self.assertEqual(float(recovery['exact_mcnemar_p']),.125)
            paired=risk.read(out/'dangerous_event_pairs.csv')
            self.assertEqual(len(paired),16)
            self.assertEqual(len({tuple(r[f] for f in list(risk.KEYS)+['step','obstacle_id']) for r in paired}),16)
            for row in paired:
                self.assertEqual(row['map_mode'],'a')
                self.assertEqual(row['switch_event'],'1' if row['step']=='1' else '')
                if row['step']=='1':
                    self.assertEqual(float(row['delta_count']),14)
                    self.assertEqual(float(row['delta_success']),0)
            risk.analyze(source,out,event_risk_reference='fixed')
            fixed=risk.read(out/'dangerous_event_pairs.csv')
            self.assertTrue(all(r['dangerous_event']==('1' if r['step']=='0' else '0') for r in fixed))
            # A shared seed/plant must not silently admit unequal S into this table.
            for row in modes:
                if row['solver_style']=='sh_mpcc':
                    row['sampled_count']*=2;row['scenario_count']=40
            risk.write(source/'artifact_mode_mechanism.csv',modes,[])
            risk.analyze(source,out)
            self.assertEqual(risk.read(out/'dangerous_event_pairs.csv'),[])
            (source/'report_all_comparisons.csv').unlink()
            risk.analyze(source,out)
            self.assertEqual(risk.read(out/'matched_events.csv'),[])
            self.assertEqual(risk.read(out/'ablation_pairs.csv'),[])
            self.assertEqual(len(risk.read(out/'risk_per_decision.csv')),16)


if __name__=='__main__':unittest.main()
