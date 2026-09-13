#!/usr/bin/env python3
"""Regression contracts for matrix enumeration, denominators, geometry and resume."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('matrix', Path(__file__).with_name('run_analysis_matrix.py'))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.settings = matrix.load_settings(matrix.ROOT/'configs/analysis_matrix/settings.json')

    def test_all_480_combinations_and_exact_axes(self):
        cases = list(matrix.configurations(self.settings))
        self.assertEqual(len(cases), 480)
        self.assertEqual(len({c['case'] for c in cases}), 480)
        self.assertTrue(all(c['classes'] <= c['obstacles'] for c in cases))
        for case in cases:
            text = matrix.config_text(case, self.settings)
            self.assertIn(f"num_classes: {case['classes']}\n", text)
            self.assertIn(f"num_obstacles: {case['obstacles']}\n", text)
            self.assertIn('obstacle_place_on_path: true\n', text)
            self.assertIn('automatically_compute_sample_size: true\n', text)
            self.assertIn('artifact_write_analysis_csv: true\n', text)
            modes = next(line.split(': ', 1)[1] for line in text.splitlines() if line.startswith('obs_modes:'))
            self.assertEqual(len(json.loads(modes)), case['modes_per_class'])

    def test_pairs_only_change_solver_label_and_dro(self):
        cases = list(matrix.configurations(self.settings))
        for a, b in zip(cases[::2], cases[1::2]):
            def common(case):
                return [line for line in matrix.config_text(case, self.settings).splitlines()
                        if not line.startswith(('dro_enabled:', 'method_name:', 'scenario_tag:'))]
            self.assertEqual(common(a), common(b))

    def test_conflicting_overrides_rejected(self):
        self.settings['overrides']['obs_modes'] = ['stop']
        with self.assertRaises(ValueError):
            matrix.config_text(next(matrix.configurations(self.settings)), self.settings)

    def synthetic_bundle(self, root):
        case = next(matrix.configurations(self.settings))
        matrix.write_csv(root/'rollout.csv', [{
            'effective_available_mode_counts': '1', 'total_steps': 1, 'collision': 1,
            'completed_path': 0, 'termination_reason': 'step_limit', 'certified_decisions': 1,
            'safe_horizon_decisions': 2, 'plant_seed': 1, 'controller_seed': 2,
            'qp_backend': 'test', 'qp_solver_identity': 'test'}])
        matrix.write_csv(root/'decisions.csv', [dict(step=i, solve_ms=t, success=1,
            certificate_requested=1, certified=i, applied_control_effort=2, scenario_count=10)
            for i, t in enumerate([2, 6])])
        matrix.write_csv(root/'mode_coverage.csv', [dict(step=i, obstacle_id=0, class_id=0,
            true_mode='stop', sampled_modes=modes, represented=int(i==0), scenario_count=10)
            for i, modes in enumerate(['constant_velocity;stop', 'constant_velocity'])])
        matrix.write_csv(root/'geometry.csv', [dict(actor='ego', radius=.5, length=2,
            num_discs=3, safety_margin=.1), dict(actor='obstacle', radius=.4, length=0,
            num_discs=1, safety_margin=.1)])
        trace = []
        for step, obs_x in enumerate([3, 1.5]):
            for actor, x in [('ego', 0), ('obstacle', obs_x)]:
                trace.append(dict(step=step, time_s=step, actor=actor,
                    obstacle_id=-1 if actor=='ego' else 0, x=x, y=0, theta=0, v=0, vx=0, vy=0,
                    solve_time_ms=0))
        matrix.write_csv(root/'trace.csv', trace)
        return case

    def test_signed_disc_margins_and_rollout_miss_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.synthetic_bundle(root)
            result = matrix.analyze(root, case)
            # Front disc at x=1; R=1; margins at x_obs=3 and 1.5 are 1 and -0.5.
            self.assertAlmostEqual(result['max_conservatism_m'], -.5)
            self.assertAlmostEqual(result['average_conservatism_m'], .25)
            self.assertEqual(result['missed_mode_rollout'], 1)
            self.assertEqual(result['missed_obstacle_decisions'], 1)
            self.assertEqual(result['mode_checks'], 2)
            self.assertEqual(result['control_effort'], 4)
            self.assertEqual(result['solve_mean_ms'], 4)
            self.assertEqual(result['solve_max_ms'], 6)
            self.assertEqual(len(matrix.rows(root/'conservatism.csv')), 2)
            good = {**case, 'status': 'OK', 'repeats': [{'metrics': result}]*2}
            other = copy.deepcopy(good)
            other['repeats'][0]['metrics']['sh_decisions'] = 4
            other['repeats'][0]['metrics']['certified_decisions'] = 0
            other['repeats'][0]['metrics']['collision'] = 0
            other['repeats'][0]['metrics']['missed_mode_rollout'] = 0
            failed = {**case, 'status': 'ERROR', 'repeats': []}
            summary = matrix.aggregate([good, other, failed], [case], 4)[0]
            self.assertEqual(summary['measured_rollouts'], 2)  # repeats do not inflate N
            self.assertEqual(summary['errors'], 1)
            self.assertEqual(summary['pending_rollouts'], 1)
            self.assertEqual(summary['collision_rate'], .5)
            self.assertEqual(summary['missed_mode_rate'], .5)
            self.assertAlmostEqual(summary['sh_certification_rate'], 1/6)

    def test_missing_data_is_not_zero_risk(self):
        case = next(matrix.configurations(self.settings))
        summary = matrix.aggregate([], [case], 10)[0]
        self.assertIsNone(summary['collision_rate'])
        self.assertIsNone(summary['sh_certification_rate'])
        self.assertIsNone(summary['max_conservatism_m'])
        self.assertEqual(summary['pending_rollouts'], 10)


def integration(runner):
    # Every requested combination is parsed and run twice. Small horizon/sample
    # count checks instrumentation only; these are not certification experiments.
    settings = matrix.load_settings(matrix.ROOT/'configs/analysis_matrix/settings.json')
    settings.update(seeds=[77], repeats=2)
    settings['overrides'].update(rollout_steps=1, horizon=2, num_scenarios=4,
        automatically_compute_sample_size=False, artifact_show_linearized_constraints=False,
        artifact_write_visualization_svg=False)
    with tempfile.TemporaryDirectory(prefix='analysis-matrix-regression-') as tmp:
        root = Path(tmp)
        config = root/'settings.json'
        config.write_text(json.dumps(settings))
        output = root/'output'
        command = [sys.executable, str(Path(__file__).with_name('run_analysis_matrix.py')),
                   '--runner', str(runner), '--settings', str(config), '--output', str(output)]
        with (root/'run.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            print((root/'run.log').read_text())
            for log in output.glob('*/seed_*/repeat_*.log'):
                if 'experiment_runner:' in log.read_text():
                    print(log, log.read_text()[-1500:])
        assert completed.returncode == 0, 'matrix smoke run failed'
        results = json.loads((output/'results.json').read_text())
        assert len(results) == 480 and all(r['repeatable'] for r in results)
        assert all(r['repeats'][0]['metrics']['mode_checks'] == r['obstacles'] for r in results)
        assert all(r['repeats'][0]['metrics']['sh_decisions'] == 1 for r in results)
        assert any(len(row['sampled_modes'].split(';')) > 1
                   for path in output.glob('*/seed_*/repeat_0/mode_coverage.csv') for row in matrix.rows(path))
        # Resume must reuse results and refuse changed standardized settings.
        times = {p: p.stat().st_mtime_ns for p in output.glob('*/seed_*/result.json')}
        subprocess.run(command + ['--resume'], check=True, stdout=subprocess.DEVNULL)
        assert all(p.stat().st_mtime_ns == t for p, t in times.items())
        # A filtered rerun must execute only the exact case, even with unrelated
        # cached cases in the same output directory.
        selected = 'sh_mpcc_dro_roundabout_o2_c2_m4'
        siblings = ['sh_mpcc_dro_roundabout_o3_c1_m2', 'sh_mpcc_roundabout_o3_c1_m2']
        untouched = {p: (p.stat().st_mtime_ns, p.read_bytes())
                     for name in siblings
                     for p in list((output/name).rglob('*')) + [output/'configs'/(name+'.yaml')]
                     if p.is_file()}
        filtered = subprocess.run(command+['--case', selected], capture_output=True, text=True)
        assert filtered.returncode == 0, filtered.stdout + filtered.stderr
        assert '1 selected configurations; 1 seed trials; 2 executions' in filtered.stdout
        assert all(name not in filtered.stdout for name in siblings)
        assert all(p.stat().st_mtime_ns == stamp and p.read_bytes() == content
                   for p, (stamp, content) in untouched.items())
        assert {r['case'] for r in json.loads((output/'results.json').read_text())} == {selected}
        assert {r['case'] for r in matrix.rows(output/'summary.csv')} == {selected}
        assert {r['case'] for r in matrix.rows(output/'rollouts.csv')} == {selected}
        rejected_output = root/'unknown'
        invalid = subprocess.run(command+['--output', str(rejected_output), '--generate-only',
                                         '--case', 'sh_mpcc_dro_roundabout_03_c1_m2'],
                                 capture_output=True, text=True)
        assert invalid.returncode != 0 and not rejected_output.exists()
        print('PASS: exact --case rerun leaves both reported sibling cases untouched; reports are scoped')
        settings['overrides']['rollout_steps'] = 2
        config.write_text(json.dumps(settings))
        stale = subprocess.run(command+['--resume'], capture_output=True, text=True)
        assert stale.returncode != 0 and 'different manifest' in stale.stderr
        print('PASS: 480 combinations, 960 executions, identical repeats/paired starts, resume and stale-settings rejection')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--runner':
        integration(Path(sys.argv[2]).resolve())
    else:
        unittest.main()
