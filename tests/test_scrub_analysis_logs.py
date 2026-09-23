#!/usr/bin/env python3
"""Exercise fallback reporting through the scrubber's CSV command line."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/scrub_analysis_logs.py"


class ScrubAnalysisLogsTest(unittest.TestCase):
    def test_mixed_suites_structured_evidence_and_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'tests';output=Path(tmp)/'csv'
            for name,suite,overrides in [('fixed','comparison_matrix',{'safe_horizon_enabled':False}),
                                         ('five_arm','comparison_matrix',{}),('reviewer','reviewer_matrix',{})]:
                test=root/name;test.mkdir(parents=True)
                (test/'matrix.json').write_text(json.dumps(dict(suite=suite,identity=name,
                    settings=dict(scenario_budgets=[40],overrides=overrides))))
                for method in ['sh_mpcc_extra','sh_mpcc_resample','sh_mpcc_dro']:
                    trial=test/'straight_s40'/'seed_77'/method;trial.mkdir(parents=True)
                    (trial/'result.json').write_text(json.dumps(dict(case=method+'_straight_s40',pair='straight_s40',
                        solver_style=method,status='OK',repeatable=True)))
                    for repeat in range(2):
                        (trial/f'repeat_{repeat}.log').write_text('[HARNESS RESULT] step=0 success=1\n')
                        bundle=trial/f'repeat_{repeat}';bundle.mkdir()
                        (bundle/'resolved_config.yaml').write_text('safe_horizon_enabled: false\nautomatically_compute_sample_size: false\nnum_scenarios: 40\n')
                        (bundle/'decisions.csv').write_text('step,certificate_requested,certified\n0,0,0\n')
                        (bundle/'attempts.csv').write_text('step,attempt,scenario_count\n0,0,40\n')
                        (bundle/'transport_costs.csv').write_text('step,attempt,cost\n0,0,0.25\n')
                (test/'concentration_summary.csv').write_text('pair,controller,q_near_one_fraction\nstraight_s40,sh_mpcc_dro,0.5\n')
            failed=root/'fixed'/'straight_s40'/'seed_78'/'sh_mpcc'
            failed.mkdir(parents=True)
            (failed/'repeat_0.log').write_text('experiment_runner: empty polygon\n')
            (failed/'result.json').write_text(json.dumps(dict(status='ERROR',error='empty polygon',solver_style='sh_mpcc',case='sh_mpcc_straight_s40')))
            rare=root/'frozen'/'artifacts';rare.mkdir(parents=True)
            (rare.parent/'manifest.json').write_text('{}')
            (rare/'coverage_trials.csv').write_text('seed,scheme,represented\n10001,nominal_single,1\n')
            subprocess.run([sys.executable,str(SCRIPT),str(root),'--out',str(output)],check=True,capture_output=True)
            def rows(name):
                with (output/name).open() as f:return list(csv.DictReader(f))
            runs=rows('run_summary.csv')
            self.assertEqual(len(runs),19)
            self.assertEqual({r['test_suite'] for r in runs},{'fixed_budget_uncertified','sample_efficiency','reviewer_matrix'})
            self.assertEqual({r['variant'] for r in runs},{'non_dro','extra_nominal','nominal_resample','dro'})
            self.assertEqual({r['repeat'] for r in runs},{'0','1'})
            self.assertEqual(sum(r['trial_status']=='ERROR' for r in runs),1)
            self.assertTrue(all(r['collision']=='' for r in runs),'missing outcomes must stay unknown')
            self.assertEqual(len(rows('artifact_decisions.csv')),18)
            self.assertTrue(all(r['certified']=='0' and r['certification_status']=='not_requested' for r in rows('artifact_decisions.csv')))
            self.assertEqual(len(rows('report_concentration_summary.csv')),3)
            self.assertEqual(rows('rare_coverage_trials.csv')[0]['test_name'],'frozen')
            self.assertEqual(rows('rare_coverage_trials.csv')[0]['test_suite'],'rare_mode')
            for path in output.glob('*.csv'):
                with path.open() as f:self.assertIn('test_suite',next(csv.reader(f)))
            snapshot={p.name:p.read_bytes() for p in output.glob('*.csv')}
            subprocess.run([sys.executable,str(SCRIPT),str(root),'--out',str(output)],check=True,capture_output=True)
            self.assertEqual(snapshot,{p.name:p.read_bytes() for p in output.glob('*.csv')})
            help_text=subprocess.check_output([sys.executable,str(SCRIPT),'--help'],text=True)
            for name in ['run_analysis_matrix.py','run_reviewer_matrix.py','run_comparison_matrix.py','run_rare_mode_experiment.py','fixed_budget_uncertified']:
                self.assertIn(name,help_text)

    def test_fallback_and_legacy_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            output = Path(tmp) / "csv"
            cases = {
                "sh_mpcc_roundabout": "[MPC STEP METRICS] step=0 fallback=1\n",
                "sh_mpcc_dro_roundabout": "[DRO SUMMARY] step=1 rho=0.2\n",
                "sh_mpcc_dro_fallback_roundabout": (
                    "[DRO SUMMARY] step=1 rho=0.2\n"
                    "[NOMINAL FALLBACK] step=1 success=1\n"
                    "[HARNESS RESULT] step=0 success=1\n"
                    "[MPC STEP METRICS] step=0 fallback=0\n"
                    "[MPC STEP METRICS] step=1 fallback=0\n"
                    "[NOMINAL FALLBACK] step=3 success=0\n"
                    "[HARNESS RESULT] step=2 success=0\n"
                    "[MPC STEP METRICS] step=2 fallback=1\n"
                ),
                # The option must remain identifiable even without DRO output
                # or an actual nominal retry.
                "sh_mpcc_dro_fallback_straight": "",
            }
            for case, log in cases.items():
                path = root / "nested" / case / "seed_77" / "run.log"
                path.parent.mkdir(parents=True)
                path.write_text(log)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--out", str(output)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("Parsed 4 logs", result.stdout)
            with (output / "run_summary.csv").open() as stream:
                runs = {Path(row["case"]).name: row for row in csv.DictReader(stream)}
            for case, variant in (
                ("sh_mpcc_roundabout", "non_dro"),
                ("sh_mpcc_dro_roundabout", "dro"),
                ("sh_mpcc_dro_fallback_roundabout", "dro_fallback"),
            ):
                self.assertEqual(runs[case]["variant"], variant)
                self.assertEqual(runs[case]["pair_case"], "nested/roundabout")
            fallback = runs["sh_mpcc_dro_fallback_roundabout"]
            self.assertEqual(fallback["nominal_fallback_attempt_count"], "2")
            self.assertEqual(fallback["nominal_fallback_success_count"], "1")
            self.assertEqual(fallback["fallback_step_count"], "1")
            self.assertEqual(runs["sh_mpcc_dro_fallback_straight"]["variant"], "dro_fallback")
            self.assertEqual(runs["sh_mpcc_roundabout"]["nominal_fallback_attempt_count"], "0")
            with (output / "control_steps.csv").open() as stream:
                controls = [row for row in csv.DictReader(stream)
                            if row["case"] == fallback["case"]]
            self.assertEqual(
                [(row["step"], row["nominal_fallback_attempted"], row["used_nominal_fallback"])
                 for row in controls],
                [("0", "1", "1"), ("1", "0", "0"), ("2", "1", "0")],
            )
            print("Observed: 4 runs; shared pairing key; nominal attempts=2, successes=1; "
                  "braking fallback=1; control flags=(1,1),(0,0),(1,0)")


if __name__ == "__main__":
    unittest.main()
