#!/usr/bin/env python3
"""Check saved-pair selection, exact replay guards, and isolated probes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import investigate_matrix as investigation
import run_analysis_matrix as analysis


class InvestigationTests(unittest.TestCase):
    def test_replay_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            for root in (a, b):
                root.mkdir()
                analysis.write_csv(root/'trace.csv', [dict(step=0, actor='ego', obstacle_id=-1,
                    mode='', x=0, y=0, theta=0, v=1, vx='', vy='', path_progress=0, collision=0)])
                analysis.write_csv(root/'decisions.csv', [dict(step=0, success=0, solve_ms=1)])
            self.assertTrue(investigation.compare_replay(a, b, 0)[0])
            analysis.write_csv(b/'decisions.csv', [dict(step=0, success=0, solve_ms=100)])
            self.assertTrue(investigation.compare_replay(a, b, 0)[0])
            analysis.write_csv(b/'decisions.csv', [dict(step=0, success=1, solve_ms=100)])
            self.assertFalse(investigation.compare_replay(a, b, 0)[0])
            self.assertFalse(investigation.compare_replay(a, b, 1)[0])

    def test_both_matrix_namings_and_incomplete_baseline(self):
        for suffix in ('roundabout_o2_c2_m4', 'shift_and_boost_roundabout_o2_c2_m4'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for style in analysis.STYLES:
                    case = style+'_'+suffix
                    bundle = root/case/'seed_79'/'repeat_0'
                    bundle.mkdir(parents=True)
                    analysis.write_csv(bundle/'trace.csv', [dict(step=0, actor='obstacle',
                        obstacle_id=0, mode='stop', x=0, y=0, vx=0, vy=0)])
                    metrics = dict(collision=0, completed_path=int(style == 'sh_mpcc'),
                        termination_reason='path_complete' if style == 'sh_mpcc' else 'no_admissible_control',
                        executed_steps=185, plant_seed=1, controller_seed=2, initial_placement=[],
                        backend='test', solver_identity='test')
                    trial = dict(case=case, solver_style=style, seed=79, obstacles=2, classes=2,
                        environment='roundabout', modes_per_class=4, status='OK', repeats=[dict(metrics=metrics)])
                    if suffix.startswith('shift'):
                        trial.update(pair=suffix, profile='shift_and_boost')
                    analysis.dump_json(bundle.parent/'result.json', trial)
                row = investigation.inventory(root, suffix, [79])[0]
                self.assertTrue(row['dro_refusal_nominal_safe_completion'])
                self.assertEqual(row['decision_step_zero_based'], 185)
                self.assertEqual(row['failed_decision_one_based'], 186)
                self.assertEqual(investigation.inventory(root, suffix, [85]), [])
                path = root/('sh_mpcc_'+suffix)/'seed_79/result.json'
                trial = json.loads(path.read_text())
                trial['repeats'][0]['metrics']['completed_path'] = 0
                analysis.dump_json(path, trial)
                self.assertFalse(investigation.inventory(root)[0]['dro_refusal_nominal_safe_completion'])


def integration(probe, runner):
    settings = analysis.load_settings(analysis.ROOT/'configs/analysis_matrix/settings.json')
    settings['overrides'].update(rollout_steps=3, horizon=2, num_scenarios=4,
        automatically_compute_sample_size=False, artifact_write_visualization_gif=False,
        artifact_write_visualization_svg=False, artifact_show_support_scenarios=False,
        artifact_show_linearized_constraints=False)
    case = dict(case='probe_fixture', solver_style='sh_mpcc_dro', environment='straight',
                obstacles=1, classes=1, modes_per_class=2)
    with tempfile.TemporaryDirectory(prefix='counterfactual-regression-') as tmp:
        root = Path(tmp)
        config = root/'config.yaml'
        config.write_text(analysis.config_text(case, settings))
        with (root/'source.log').open('w') as log:
            subprocess.run([str(runner), '--config', str(config), '--seed', '77',
                '--output', str(root/'ordinary'), '--label', 'source'], check=True, stdout=log, stderr=log)
        for repeat in range(2):
            target = root/f'probe_{repeat}'
            with (root/f'probe_{repeat}.log').open('w') as log:
                subprocess.run([str(probe), '--config', str(config), '--seed', '77', '--step', '1',
                    '--samples', '30', '--mc-seed', '456', '--output', str(target),
                    '--radius-scales', '0,1'], check=True, stdout=log, stderr=log)
            matches, reason = investigation.compare_replay(root/'ordinary/source', target/'replay/source', 1)
            assert matches, reason
            snapshot = analysis.rows(target/'snapshot.csv')[0]
            assert snapshot['nominal_success'] == '1', snapshot
            summaries = analysis.rows(target/'counterfactual_summary.csv')
            assert {r['law'] for r in summaries} >= {'p', 'qstar'}, summaries
            assert all(int(r['samples']) == 30 and int(r['collisions']) <= 30 for r in summaries)
            for row in analysis.rows(target/'radius_sweep_weights.csv'):
                assert float(row['rho_requested']) == float(row['rho_used']), row
            # Alpha=1 should reproduce the unscaled Q* categorical sampling plan.
            assert (target/'alpha_1.000000_controls.csv').read_bytes() == (target/'dro_controls.csv').read_bytes()
        for filename in ('snapshot.csv', 'nominal_plan.csv', 'dro_plan.csv', 'weights.csv',
                         'counterfactual_trials.csv', 'counterfactual_summary.csv', 'radius_sweep.csv'):
            assert (root/'probe_0'/filename).read_bytes() == (root/'probe_1'/filename).read_bytes(), filename
        print('PASS: pre-decision cloning preserves live replay; independent MC repeatability; alpha=1 reproduces DRO controls')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--integration':
        integration(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    else:
        unittest.main()
