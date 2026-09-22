#!/usr/bin/env python3
"""Contracts for comparison selection, paired evidence, profiles, and resume."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_comparison_matrix as matrix


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.settings = matrix.load_settings(matrix.ROOT/'configs/comparison_matrix/settings.json')

    def test_enumeration_and_only_dro_differs(self):
        cases = list(matrix.configurations(self.settings))
        self.assertEqual(len(cases), 2400)
        self.assertEqual(len({c['pair'] for c in cases}), 1200)
        for a, b in zip(cases[::2], cases[1::2]):
            def common(case):
                return [line for line in matrix.config_text(case, self.settings).splitlines()
                        if not line.startswith(('dro_enabled:', 'method_name:', 'scenario_tag:'))]
            self.assertEqual(a['pair'], b['pair'])
            self.assertEqual(common(a), common(b))
        self.assertEqual(cases[0]['pair'], 'straight_o1_c1_m2_standard')
        for percent in (5, 10, 25, 50, 75, 100):
            case = next(c for c in cases if c['pair'] == f'straight_o1_c1_m2_boost_{percent}')
            text = matrix.config_text(case, self.settings)
            self.assertIn(f'shift_boost: {percent/100}\n', text)
            self.assertIn('artifact_write_visualization_gif: true\n', text)
            self.assertIn('artifact_write_visualization_svg: true\n', text)
        self.assertTrue(any(c['pair'] == 'straight_o1_c1_m2_shift' for c in cases))
        boosted = next(c for c in cases if c['profile'] == 'shift_and_boost')
        text = matrix.config_text(boosted, self.settings)
        self.assertIn('shift_psi: 0.25\n', text)
        self.assertIn('shift_boost: 0.15\n', text)
        self.assertIn('boosted_mode: -1\n', text)

    def test_comparison_requires_two_ordered_controllers(self):
        for styles in (['sh_mpcc', 'sh_mpcc_dro', 'sh_mpcc_dro_fallback'],
                       ['sh_mpcc_dro', 'sh_mpcc'], ['sh_mpcc']):
            with self.subTest(styles=styles), tempfile.TemporaryDirectory() as tmp:
                settings = copy.deepcopy(self.settings)
                settings['solver_styles'] = styles
                path = Path(tmp)/'settings.json'
                path.write_text(json.dumps(settings))
                with self.assertRaisesRegex(ValueError, 'sh_mpcc and sh_mpcc_dro'):
                    matrix.load_settings(path)

    def test_invalid_profiles(self):
        for update in ({'shift_boost': 2}, {'shift_psi': float('nan')},
                       {'boosted_mode': 2}, {'rare_mode': 'stop'},
                       {'rare_mode_probability': .1}, {'typo': 1}):
            settings = copy.deepcopy(self.settings)
            settings['mismatch_profiles'] = [dict(name='bad', **update)]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'settings.json'
                path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    matrix.load_settings(path)

    def make_pair(self, root):
        cases = list(matrix.configurations(self.settings))[:2]
        results = {}
        for case in cases:
            bundle = matrix.analysis.trial_root(root, case, 77)/'repeat_0'
            bundle.mkdir(parents=True)
            matrix.analysis.write_csv(bundle/'trace.csv', [dict(step=0, obstacle_id=0,
                actor='obstacle', mode='constant_velocity', x=1, y=0, vx=1, vy=0)])
            metrics = dict(collision=0, completed_path=1, plant_seed=1, controller_seed=2,
                           initial_placement=[], backend='test', solver_identity='test')
            results[case['case'], 77] = dict(status='OK', repeats=[dict(metrics=metrics)])
        return cases, results

    def test_target_reverse_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, results = self.make_pair(root)
            a = results[cases[0]['case'], 77]['repeats'][0]['metrics']
            b = results[cases[1]['case'], 77]['repeats'][0]['metrics']
            a.update(collision=1, completed_path=0)
            pair = matrix.paired_result(cases, 77, results, root)
            self.assertTrue(pair['target_match'])
            self.assertFalse(pair['reverse_match'])
            summary = matrix.summarize([pair])[0]
            self.assertEqual(summary['target_seeds'], '77')
            self.assertEqual(summary['target_match_count'], 1)
            b.update(completed_path=0, termination_reason='no_admissible_control')
            pair = matrix.paired_result(cases, 77, results, root)
            self.assertFalse(pair['target_match'])
            self.assertTrue(pair['nondro_collision_dro_collision_free_incomplete'])
            a.update(collision=0, completed_path=1)
            b.update(collision=1)
            pair = matrix.paired_result(cases, 77, results, root)
            self.assertTrue(pair['reverse_match'])
            self.assertFalse(pair['target_match'])

    def test_errors_pending_and_pairing_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, results = self.make_pair(root)
            missing = matrix.paired_result(cases, 77, {}, root)
            self.assertEqual(missing['status'], 'PENDING')
            self.assertIsNone(matrix.summarize([missing])[0]['target_match_rate'])
            results[cases[0]['case'], 77]['status'] = 'ERROR'
            error = matrix.paired_result(cases, 77, results, root)
            self.assertEqual(error['status'], 'ERROR')
            self.assertFalse(error['target_match'])
            results[cases[0]['case'], 77]['status'] = 'OK'
            path = matrix.analysis.trial_root(root, cases[0], 77)/'repeat_0'/'trace.csv'
            path.write_text(path.read_text().replace('constant_velocity', 'stop'))
            mismatch = matrix.paired_result(cases, 77, results, root)
            self.assertEqual(mismatch['status'], 'ERROR')
            self.assertIn('obstacle trajectories differ', mismatch['error'])


def integration(runner):
    settings = matrix.load_settings(matrix.ROOT/'configs/comparison_matrix/settings.json')
    settings.update(obstacle_counts=[1], class_counts=[1], environments=['straight'],
                    mode_counts=[2], seeds=[77], repeats=2)
    # Small instrumentation fixture; not a safety certification experiment.
    settings['overrides'].update(rollout_steps=3, horizon=2, num_scenarios=4,
                                automatically_compute_sample_size=False)
    with tempfile.TemporaryDirectory(prefix='comparison-matrix-') as tmp:
        root = Path(tmp)
        config = root/'settings.json'
        config.write_text(json.dumps(settings))
        command = [sys.executable, str(Path(matrix.__file__)), '--runner', str(runner),
                   '--settings', str(config), '--output', str(root/'output')]
        subprocess.run(command + ['--generate-only'], check=True)
        subprocess.run(command, check=True)
        pairs = json.loads((root/'output/pairs.json').read_text())
        assert len(pairs) == 10 and all(p['status'] == 'OK' for p in pairs), pairs
        assert all(p['paired_obstacle_prefix_equal'] for p in pairs), pairs
        assert all(not p['target_match'] for p in pairs), pairs
        summaries = json.loads((root/'output/summary.json').read_text())
        assert all(s['measured_pairs'] == 1 for s in summaries), summaries
        for pair in pairs:
            for arm, style in (('nondro', 'sh_mpcc'), ('dro', 'sh_mpcc_dro')):
                bundle = Path(pair[arm+'_artifact'])
                assert bundle == root/'output'/pair['pair']/'seed_77'/style/'repeat_0', bundle
                assert (bundle/'rollout.gif').read_bytes()[:6] in (b'GIF87a', b'GIF89a')
                assert '<svg' in (bundle/'rollout.svg').read_text()
        import investigate_matrix
        inventory = investigate_matrix.inventory(root/'output')
        assert len(inventory) == 10 and all(r['status'] == 'OK' for r in inventory), inventory
        paths = list((root/'output').glob('*/seed_*/sh_mpcc*/result.json'))
        times = {p: p.stat().st_mtime_ns for p in paths}
        subprocess.run(command + ['--resume'], check=True)
        assert all(p.stat().st_mtime_ns == t for p, t in times.items())
        subprocess.run(command + ['--resume', '--case', 'straight_o1_c1_m2_standard'], check=True)
        assert len(json.loads((root/'output/pairs.json').read_text())) == 1
        assert all(p.stat().st_mtime_ns == t for p, t in times.items())
        settings['mismatch_profiles'][1]['shift_psi'] = .5
        config.write_text(json.dumps(settings))
        stale = subprocess.run(command + ['--resume'], capture_output=True, text=True)
        assert stale.returncode != 0 and 'different manifest' in stale.stderr, stale.stderr
        print('PASS: 10 profiles, 40 executions, seed/obstacle pairing, repeatability, resume, filtering, stale rejection')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--runner':
        integration(Path(sys.argv[2]).resolve())
    else:
        unittest.main()
