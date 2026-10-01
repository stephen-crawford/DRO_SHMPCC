#!/usr/bin/env python3
"""Exercise the current matrix and incomplete/duplicate paired analysis."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import run_comparison_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('causal', ROOT/'tools/analyze_comparison_results.py')
causal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal)


class CausalMatrixTests(unittest.TestCase):
    def test_five_arm_matrix(self):
        settings = matrix.load_settings(ROOT/'configs/causal_matrix/settings.json')
        cases = list(matrix.configurations(settings))
        self.assertEqual(len(cases), 360)
        self.assertEqual(len(cases)*len(settings['seeds'])*settings['repeats'], 7200)
        groups = {}
        for case in cases:
            groups.setdefault(case['pair'], []).append(case)
            text = matrix.config_text(case, settings)
            for value in ['artifact_write_analysis_csv: true', 'artifact_capture_attempt_diagnostics: true',
                          'artifact_write_visualization_gif: true', 'artifact_write_visualization_svg: true',
                          'safe_horizon_enabled: false', 'automatically_compute_sample_size: false']:
                self.assertIn(value, text)
            budget = case['scenario_budget']*(2 if case['solver_style']=='sh_mpcc_extra' else 1)
            self.assertIn(f'num_scenarios: {budget}\n', text)
        self.assertTrue(all({c['solver_style'] for c in group} == set(causal.ARMS) for group in groups.values()))

    def test_missing_duplicate_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'csv', Path(tmp)/'report'
            source.mkdir()
            runs, expected = [], []
            for seed in [77,78,79,80]:
                for style, arm in causal.ARMS.items():
                    identity = dict(test_root='fixture', matrix_identity='id', pair_case='straight_boost_5_s20', seed=str(seed), repeat='0')
                    expected.append(dict(identity, solver_style=style))
                    if seed == 78 and arm == 'dro' or seed == 80:
                        continue
                    row = dict(identity, variant=arm, trial_status='OK', log_complete='1',
                               collision='1' if arm=='non_dro' else '0', path_completed='0',
                               min_actual_clearance='-0.1' if arm=='non_dro' else '0.2')
                    runs.append(row)
                    if seed == 79 and arm == 'dro':
                        runs.append(row)
            causal.write(source/'expected_runs.csv', expected, [])
            causal.write(source/'run_summary.csv', runs, [])
            causal.write(source/'report_all_comparisons.csv', [dict(test_root='fixture', matrix_identity='id',
                pair='straight_boost_5_s20', seed='77', controller_a='sh_mpcc', controller_b=style,
                status='OK') for style in causal.ARMS if style != 'sh_mpcc'], [])
            mode_rows = []
            for style in causal.ARMS:
                for mode, probability, count in [('turn_right', .2, 4), ('constant_velocity', .8, 16)]:
                    q = (.6 if mode == 'turn_right' else .4) if style == 'sh_mpcc_dro' else probability
                    mode_rows.append(dict(test_root='fixture', matrix_identity='id',
                        pair_case='straight_boost_5_s20', seed='77', repeat='0',
                        solver_style=style, step='0', attempt='0', obstacle_id='0', mode=mode,
                        nominal_probability=str(probability), sampling_probability=str(q),
                        sampled_count=str(count), scenario_count='20'))
            causal.write(source/'artifact_mode_mechanism.csv', mode_rows, [])
            causal.analyze(source, output)
            mechanism = causal.read(output/'mode_representation.csv')
            self.assertEqual(len(mechanism), 10)
            self.assertEqual({r['shift_group'] for r in mechanism}, {'increased','decreased'})
            self.assertTrue(all(float(r['sampled_fraction']) == .2 for r in mechanism if r['mode']=='turn_right'))
            complete=causal.read(output/'completeness.csv')
            self.assertEqual([r['paired_complete'] for r in complete], ['1','0','0','0'])
            self.assertEqual(complete[2]['dro_status'], 'DUPLICATE')
            self.assertEqual(len(causal.read(output/'paired_runs.csv')), 5)
            for row in causal.read(output/'paired_differences.csv'):
                self.assertAlmostEqual(float(row['delta_min_actual_clearance']), .3)
            for row in causal.read(output/'collision_pairs.csv'):
                self.assertEqual(row['n10'], '1')
                self.assertEqual(row['n01'], '0')
            (source/'report_all_comparisons.csv').unlink()
            causal.analyze(source, output)
            self.assertEqual(causal.read(output/'paired_runs.csv'), [])

    def test_scrub_manifest_without_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw', Path(tmp)/'csv'
            root.mkdir()
            settings = matrix.load_settings(ROOT/'configs/causal_matrix/settings.json')
            settings.update(seeds=[77], repeats=2)
            cases = list(matrix.configurations(settings))[:1]
            (root/'matrix.json').write_text(json.dumps(dict(suite='comparison_matrix', identity='fixture', settings=settings, cases=cases)))
            subprocess.run([sys.executable, str(ROOT/'tools/scrub_artifacts.py'),str(root),'--out',str(output)],check=True,capture_output=True)
            rows = causal.read(output/'expected_runs.csv')
            self.assertEqual(len(rows),2)
            self.assertEqual({r['log_exists'] for r in rows},{'0'})
            self.assertTrue((output/'artifact_rollout.csv').exists())


if __name__ == '__main__':
    unittest.main()
