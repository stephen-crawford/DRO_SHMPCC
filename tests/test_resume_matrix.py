#!/usr/bin/env python3
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
RESUME=ROOT/'tests/resume_matrix.py'

class ResumeTest(unittest.TestCase):
    def test_saved_resume_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'matrix'
            settings=json.loads((ROOT/'configs/comparison_matrix/sample_efficiency_pilot.json').read_text())
            settings.update(seeds=[77],additional_controllers=[],mismatch_profiles=[settings['mismatch_profiles'][0]])
            settings['overrides'].update(horizon=4,rollout_steps=3)
            config=root/'settings.json';config.write_text(json.dumps(settings))
            runner=ROOT/'build-concentration/experiment_runner'
            subprocess.run([sys.executable,str(ROOT/'tests/run_comparison_matrix.py'),'--settings',str(config),
                            '--runner',str(runner),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
            paths=sorted(out.glob('*/seed_*/*/result.json'))
            kept=paths[0];failed=paths[1]
            kept_bytes=kept.read_bytes();manifest_bytes=(out/'matrix.json').read_bytes()
            r=json.loads(failed.read_text());r.update(status='ERROR',error='injected interrupted-trial fixture')
            failed.write_text(json.dumps(r))
            subprocess.run([sys.executable,str(RESUME),str(out),'--limit','1'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(kept.read_bytes(),kept_bytes)
            self.assertEqual((out/'matrix.json').read_bytes(),manifest_bytes)
            self.assertEqual(json.loads(failed.read_text())['status'],'OK')
            archived=list((out/'resume_history').glob('*/**/result.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(json.loads(archived[0].read_text())['status'],'ERROR')
            # A missing result also resumes, archiving its partial evidence first.
            failed.unlink()
            subprocess.run([sys.executable,str(RESUME),str(out),'--limit','1'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(json.loads(failed.read_text())['status'],'OK')
            yaml=next((out/'configs').glob('*.yaml'));original=yaml.read_text();yaml.write_text(original+'# changed\n')
            blocked=subprocess.run([sys.executable,str(RESUME),str(out),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(blocked.returncode,2);self.assertIn('changed frozen configuration',blocked.stdout)
            yaml.write_text(original)
            bad=root/'wrong-runner';bad.write_text('not the saved binary')
            blocked=subprocess.run([sys.executable,str(RESUME),str(out),'--runner',str(bad),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(blocked.returncode,2)
            new=root/'new-cohort'
            subprocess.run([sys.executable,str(RESUME),str(out),'--restart-to',str(new),'--runner',str(runner)],check=True,stdout=subprocess.DEVNULL)
            self.assertFalse(json.loads((new/'restart_origin.json').read_text())['old_results_reused'])
            self.assertEqual(len(json.loads((new/'results.json').read_text())),2)
            self.assertEqual(kept.read_bytes(),kept_bytes)
            print('PASS: preserved success and manifest; retried failure and partial trial with archives; refused changed config/binary; separate restart ran all trials')

if __name__=='__main__':unittest.main()
