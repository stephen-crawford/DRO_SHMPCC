#!/usr/bin/env python3
"""Behavioral tests for the saved-artifact verifier; requires the numeric backend build."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from analyze_posterior_certificate import (
    projected_probability,
    projected_probability_with_normal,
)
from report_tube_decomposition import report, read, write


class DecompositionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()

        self.backend = ROOT / 'build-tube/certificate_numeric_backend'

        manifest = dict(
            seeds=[77],
            repeats=2,
            radii=[.3],
            cycles=1,
            horizon=1,
            scenario_budget=1123,
            history={'safe': 95, 'cut': 5},
        )

        (self.source / 'manifest.json').write_text(
            json.dumps(manifest)
        )

        predictions = []

        for mode, x in [('safe', 4), ('cut', .8)]:
            predictions.append(
                dict(
                    mode=mode,
                    stage=1,
                    mean_x=x,
                    mean_y=0,
                    cov_xx=.04,
                    cov_xy=0,
                    cov_yx=0,
                    cov_yy=.04,
                    collision_radius=1,
                )
            )

        write(
            self.source / 'predictions.csv',
            predictions,
        )

        cycles = []
        geometry = []
        attempts = []

        for rep in range(2):
            for arm in ['nominal_resampling', 'wdro']:

                identity = dict(
                    seed=77,
                    repeat=rep,
                    arm=arm,
                    radius=format(.3, '.17g'),
                    cycle=0,
                )

                for disc in range(3):
                    geometry.append(
                        dict(
                            identity,
                            stage=1,
                            disc=disc,
                            x=0,
                            y=0,
                            reference_x=0,
                            reference_y=0,
                            distance=0,
                        )
                    )

                for mode, x in [('safe', 4), ('cut', .8)]:

                    b = min(
                        1,
                        3 * projected_probability(
                            [0, 0],
                            [x, 0],
                            [[.04, 0], [0, .04]],
                            1.3,
                        )['probability_upper'],
                    )

                    # The conservative cut union is one; use a distant
                    # mean for a nontrivial check below.
                    if mode == 'cut':
                        x = 1.6
                        b = min(
                            1,
                            3 * projected_probability(
                                [0, 0],
                                [x, 0],
                                [[.04, 0], [0, .04]],
                                1.3,
                            )['probability_upper'],
                        )

                    p = .95 if mode == 'safe' else .05

                    q = (
                        p
                        if arm == 'nominal_resampling'
                        else (.8 if mode == 'safe' else .2)
                    )

                    cycles.append(
                        dict(
                            identity,
                            success=1,
                            active=1,
                            rejected=0,
                            mode=mode,
                            b=b,
                            p=p,
                            q=q,
                            sampled_scenarios=1123,
                        )
                    )

                    attempts.append(
                        dict(
                            identity,
                            attempt=0,
                            dro_enabled=int(arm == 'wdro'),
                            success=1,
                            mode=mode,
                            p=p,
                            q=q,
                            rho=.1 if arm == 'wdro' else '',
                            risk_score=1,
                            n_bar=6,
                            removal_budget=0,
                            total_support_cap=6,
                            observed_final_support=0,
                        )
                    )

        predictions[1]['mean_x'] = 1.6

        for name, rows in [
            ('predictions', predictions),
            ('cycles', cycles),
            ('disc_geometry', geometry),
            ('attempts', attempts),
        ]:
            write(
                self.source / (name + '.csv'),
                rows,
            )

    def run_report(self):
        return report(
            self.source,
            self.root / 'report',
            self.backend,
        )

    def test_same_geometry_transfer_and_support_settings(self):
        result = self.run_report()

        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['accepted_mode_inequalities'], 8)
        self.assertEqual(result['fixed_normal_exceedances'], 0)

        rows = read(
            self.root / 'report' / 'decomposition.csv'
        )

        self.assertTrue(
            all(
                r['S_baseline'] == '895'
                and r['total_support_cap'] == '6'
                for r in rows
            )
        )

        wdro = [
            r
            for r in rows
            if r['arm'] == 'wdro'
            and r['repeat'] == '0'
        ]

        self.assertEqual(
            wdro[0]['p_dot_b'],
            wdro[1]['p_dot_b'],
        )

        self.assertEqual(
            wdro[0]['q_dot_b'],
            wdro[1]['q_dot_b'],
        )

        self.assertGreater(
            float(wdro[0]['redistribution_delta']),
            0,
        )

    def test_rejects_geometry_escape(self):
        rows = read(
            self.source / 'disc_geometry.csv'
        )

        rows[0]['x'] = '1.6'

        write(
            self.source / 'disc_geometry.csv',
            rows,
        )

        result = self.run_report()

        kinds = {
            v['kind']
            for v in result['violations']
        }

        self.assertEqual(
            result['status'],
            'FAIL',
        )

        self.assertIn(
            'accepted_outside_tube',
            kinds,
        )

        self.assertIn(
            'repeat_mismatch',
            kinds,
        )

    def test_anisotropic_adaptive_bound_need_not_be_below_uniform(self):
        ref = [0.0, 0.0]

        # Exactly on the 0.3 m tube boundary.
        center = [0.3, 0.0]

        mean = [1.4, -0.5]

        # Strongly anisotropic covariance deliberately makes changing
        # the projection direction alter the projected variance.
        covariance = [
            [0.0025, 0.0],
            [0.0, 1.0],
        ]

        collision_radius = 1.0
        tube_radius = 0.3

        dx = mean[0] - ref[0]
        dy = mean[1] - ref[1]
        norm = math.hypot(dx, dy)

        n_ref = [
            dx / norm,
            dy / norm,
        ]

        adaptive = projected_probability(
            center,
            mean,
            covariance,
            collision_radius,
        )['probability_upper']

        fixed = projected_probability_with_normal(
            center,
            mean,
            covariance,
            collision_radius,
            n_ref,
        )['probability_upper']

        uniform = projected_probability_with_normal(
            ref,
            mean,
            covariance,
            collision_radius + tube_radius,
            n_ref,
        )['probability_upper']

        # This is the theorem-relevant ordering.
        self.assertLessEqual(
            fixed,
            uniform + 1e-12,
        )

        # This demonstrates why the adaptive posterior is only
        # diagnostic: changing the normal changes projected variance.
        self.assertGreater(
            adaptive,
            uniform,
        )

    def test_rejects_missing_mode_and_altered_budget(self):
        rows = read(
            self.source / 'cycles.csv'
        )

        rows.pop(0)
        rows[0]['sampled_scenarios'] = '40'

        write(
            self.source / 'cycles.csv',
            rows,
        )

        # The independently checked CSV cardinality/budget failures
        # must not be hidden. Remove attempts for that incomplete group
        # so transfer inversion is not attempted.
        attempts = read(
            self.source / 'attempts.csv'
        )

        write(
            self.source / 'attempts.csv',
            [
                r
                for r in attempts
                if not (
                    r['arm'] == 'nominal_resampling'
                    and r['repeat'] == '0'
                )
            ],
        )

        result = self.run_report()

        kinds = {
            v['kind']
            for v in result['violations']
        }

        self.assertEqual(
            result['status'],
            'FAIL',
        )

        self.assertIn(
            'missing_mode',
            kinds,
        )

        self.assertIn(
            'scenario_budget_changed',
            kinds,
        )


if __name__ == '__main__':
    unittest.main()