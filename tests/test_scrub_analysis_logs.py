#!/usr/bin/env python3
"""Exercise fallback reporting through the scrubber's CSV command line."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/scrub_analysis_logs.py"


class ScrubAnalysisLogsTest(unittest.TestCase):
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
