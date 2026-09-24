#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import unittest
import run_bundle_analysis_matrix as bundle
import run_analysis_matrix as analysis

class BundleMatrixTests(unittest.TestCase):
    def test_full_matrix_axes_and_pairing(self):
        settings=bundle.load_settings(bundle.ROOT/'configs/bundle_analysis_matrix/full.json')
        cases=list(bundle.configurations(settings))
        self.assertEqual(len(cases),2160)
        self.assertEqual(len({c['baseline_case'] for c in cases}),720)
        for base in analysis.configurations(settings):
            group=[c for c in cases if c['baseline_case']==base['case']]
            configs=[]
            for case in group:
                text=bundle.config_text(case,settings)
                values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in text.splitlines() if line and not line.startswith('#')}
                self.assertEqual(json.loads(values['num_obstacles']),base['obstacles'])
                self.assertEqual(json.loads(values['num_modes']),base['modes_per_class'])
                self.assertEqual(values['artifact_capture_attempt_diagnostics'],'true')
                for k in ['bundle_amplification','bundle_extra_draws','scenario_tag']:values.pop(k)
                configs.append(values)
            self.assertTrue(all(c==configs[0] for c in configs))

    def test_rejects_switching_or_overridden_variants(self):
        import tempfile
        settings=bundle.load_settings(bundle.ROOT/'configs/bundle_analysis_matrix/full.json')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'settings.json'
            for bad in [{'markov_jump_system':True},{'bundle_amplification':3}]:
                settings['overrides']=bad;path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):bundle.load_settings(path)

if __name__=='__main__':unittest.main()
