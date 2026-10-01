"""Paired measurement semantics, stress schedules, and forced-plant integration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_risk_sampling as risk
import dangerous_event_report as danger
import run_risk_stress as stress


class DangerousEventTests(unittest.TestCase):
    def test_map_ties_switches_and_gaps(self):
        rows=[dict(mode=m,nominal_probability=p,true_mode='b') for m,p in [('a',.5),('b',.5)]]
        self.assertEqual(danger.labels(rows,[])['switch_event'],'')
        result=danger.labels(rows,[dict(true_mode='a')])
        self.assertEqual(result['map_mode'],'')
        self.assertEqual(json.loads(result['map_modes']),['a','b'])
        self.assertEqual(result['map_tie'],1)
        self.assertEqual(result['switch_event'],1)
        rows[0]['nominal_probability']=.6;rows[1]['nominal_probability']=.4
        self.assertEqual(danger.labels(rows,rows)['map_mode'],'a')
        self.assertEqual(danger.labels(rows,rows)['switch_event'],0)

    def test_paired_outcomes_and_figure(self):
        common=dict(test_root='fixture',matrix_identity='id',pair_case='case',seed='77',repeat='0',step='1',obstacle_id='0')
        runkey=risk.key(common)
        ref=dict(true_mode='danger',true_risk=.8,true_p=.05,true_q=.4,true_count=8,scenario_count=20)
        source={'danger':dict(source_artifact='fixture.csv')}
        labels=dict(map_mode='safe',map_modes='["safe"]',map_tie=0,switch_event=1)
        decisions={runkey+('sh_mpcc','1'):dict(success=0),runkey+('sh_mpcc_dro','1'):dict(success=1)}
        row=danger.pair_row(common,ref,{'danger':0},source,decisions,runkey,labels,True)
        self.assertAlmostEqual(row['delta_q'],.35)
        self.assertEqual(row['delta_count'],8)
        self.assertEqual(row['delta_success'],1)
        self.assertEqual(row['nominal_outcome'],'inadmissible')
        decisions[runkey+('sh_mpcc','1')]['success']=1
        decisions[runkey+('sh_mpcc_dro','1')]['success']=0
        worse=danger.pair_row(common,ref,{'danger':10},source,decisions,runkey,labels,True)
        self.assertEqual(worse['delta_count'],-2)
        self.assertEqual(worse['delta_success'],-1)
        missing=danger.pair_row(common,ref,{'danger':0},source,{},runkey,labels,False)
        self.assertEqual(missing['delta_success'],'')
        self.assertEqual(missing['wdro_outcome'],'missing')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);danger.export(out,[row,worse,missing])
            self.assertEqual([json.loads(l) for l in (out/'dangerous_event_pairs.log').read_text().splitlines()],[row,worse,missing])
            danger.plot(out/'dangerous_event_pairs.csv',out/'figures')
            self.assertGreater((out/'figures/dangerous_chain_000.pdf').stat().st_size,1000)
            self.assertEqual(json.loads((out/'figures/figures.json').read_text())[0]['dangerous_rows'],2)

    def test_suite_design(self):
        settings=json.loads((ROOT/'configs/risk_stress/settings.json').read_text())
        cases=list(stress.cases(settings))
        self.assertEqual(len(cases),396)
        self.assertEqual(len(cases)*len(settings['seeds'])*len(settings['arms']),19800)
        self.assertEqual(len({c['name'] for c in cases}),396)
        self.assertEqual({c['danger_count'] for c in cases if c['family']=='rare_turn'},{20,50,80,100,150,200})
        self.assertEqual({c['switch'] for c in cases},{5,10,15,20})
        self.assertNotIn('sh_mpcc_dro_stratified',settings['arms'])
        self.assertTrue(all(c['lag']==5 for c in cases if c['family']=='late_switch'))
        self.assertAlmostEqual((1-.05)**20,.3584859224)

    def test_forced_plant_and_negative_controls(self):
        runner=ROOT/'build-base/risk_stress_experiment'
        if not runner.exists():self.skipTest('build risk_stress_experiment first')
        with tempfile.TemporaryDirectory() as tmp:
            for family in ['rare_turn','rare_braking','late_switch','two_sided_trap','rare_benign','common_dangerous','equal_probability','equal_geometry']:
                trajectories=[]
                for arm in ['sh_mpcc','sh_mpcc_dro']:
                    out=Path(tmp)/(family+'_'+arm)
                    count=300 if family=='common_dangerous' else 50
                    run=subprocess.run([str(runner),str(out),family,str(count),'20','77','1','3','2',arm],cwd=ROOT,capture_output=True,text=True)
                    self.assertEqual(run.returncode,0,run.stderr[-1500:])
                    rows=risk.read(out/'mode_mechanism.csv')
                    first=[r for r in rows if r['step']=='0' and r['obstacle_id']=='0']
                    self.assertEqual(sum(int(r['sampled_count']) for r in first),20)
                    if family=='equal_probability':self.assertTrue(all(abs(float(r['nominal_probability'])-1/3)<1e-12 for r in first))
                    if family=='equal_geometry':self.assertEqual(len({r['reference_risk'] for r in first}),1)
                    if family=='rare_benign':self.assertEqual(float(next(r for r in first if r['mode']=='across')['reference_risk']),0)
                    initial={r['mode']:r['nominal_probability'] for r in first}
                    self.assertTrue(all(r['nominal_probability']==initial[r['mode']] for r in rows if r['obstacle_id']=='0'))
                    plant=risk.read(out/'plant.csv');trajectories.append(plant)
                    self.assertTrue(all(r['true_mode']==('continue' if r['step']=='0' else 'across') for r in plant if r['obstacle_id']=='0'))
                    self.assertTrue(all(r['certified']=='0' and r['certificate_requested']=='0' for r in risk.read(out/'decisions.csv')))
                self.assertEqual(trajectories[0],trajectories[1])

if __name__=='__main__':unittest.main()
