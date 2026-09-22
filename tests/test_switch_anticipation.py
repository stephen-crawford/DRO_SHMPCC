#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import analyze_switch_anticipation as switch


class SwitchAnticipationTests(unittest.TestCase):

    def test_parse_real_log_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.log'

            path.write_text(
                '[DRO SUMMARY] step=11 obstacle=0 '
                'S=1123 rho=0.1 transport_cost=0.05 '
                'budget_usage=0.5 nominal_risk=0.02 '
                'qstar_risk=0.08 risk_lift=0.06 '
                'tv=0.2 tv_bound=0.3 radius_m=10\n'
                '[DRO MODE] step=11 obstacle=0 '
                'mode=turn_right p=0.01 risk=0.5 '
                'q=0.21 delta_q=0.20 q_times_S=235.83\n'
            )

            modes, summaries = switch.parse_dro_log(path)

            row = modes[(10, 0, 'turn_right')]

            self.assertAlmostEqual(row['p'], 0.01)
            self.assertAlmostEqual(row['q'], 0.21)
            self.assertAlmostEqual(row['delta_q'], 0.20)
            self.assertAlmostEqual(row['risk'], 0.5)

            summary = summaries[(10, 0)]

            self.assertAlmostEqual(summary['rho'], 0.1)
            self.assertAlmostEqual(summary['risk_lift'], 0.06)

    def test_dangerous_modes_have_positive_selectivity(self):
        observations = [
            {
                'pair': 'case',
                'profile': 'boost_50',
                'seed': 1,
                'event_index': 0,
                'lead_steps': 5,
                'dangerous_target': 1,
                'target_upweighted': 1,
                'delta_q_target': 0.20,
                'target_mode_risk': 0.8,
                'risk_lift': 0.10,
            },
            {
                'pair': 'case',
                'profile': 'boost_50',
                'seed': 2,
                'event_index': 0,
                'lead_steps': 5,
                'dangerous_target': 1,
                'target_upweighted': 1,
                'delta_q_target': 0.10,
                'target_mode_risk': 0.5,
                'risk_lift': 0.05,
            },
            {
                'pair': 'case',
                'profile': 'boost_50',
                'seed': 3,
                'event_index': 0,
                'lead_steps': 5,
                'dangerous_target': 0,
                'target_upweighted': 0,
                'delta_q_target': 0.0,
                'target_mode_risk': 0.0,
                'risk_lift': 0.0,
            },
        ]

        result = switch.summarize_group(
            observations,
            {
                'scope': 'test',
                'profile': 'boost_50',
                'lead_steps': 5,
            },
        )

        self.assertAlmostEqual(
            result['dangerous_target_upweighted_rate'],
            1.0,
        )

        self.assertAlmostEqual(
            result['dangerous_mean_delta_q'],
            0.15,
        )

        self.assertAlmostEqual(
            result['benign_mean_delta_q'],
            0.0,
        )

        self.assertGreater(
            result['risk_selectivity_delta'],
            0.0,
        )


if __name__ == '__main__':
    unittest.main()