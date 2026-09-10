#!/usr/bin/env python3
"""Preview toggles must not perturb control and must replace stale forecast data."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[1] / 'configs/base_tests/mpcc_1_obstacles.yaml'
with tempfile.TemporaryDirectory(prefix='scenario-preview-') as directory:
    root = Path(directory)
    def run(enabled, count):
        config = root / 'config.yaml'
        config.write_text(source.read_text() + '\nrollout_steps: 5\n' +
                          f'artifact_show_sampled_scenarios: {str(enabled).lower()}\n' +
                          f'artifact_scenario_preview_count: {count}\n')
        subprocess.run([str(runner), '--config', str(config), '--seed', '77',
                        '--output', str(root), '--label', 'same_bundle'],
                       check=True, stdout=subprocess.DEVNULL)
        bundle = root / 'same_bundle'
        with (bundle / 'trace.csv').open() as stream:
            trace = list(csv.DictReader(stream))
        for row in trace:
            row.pop('solve_time_ms')
        with (bundle / 'sampled_scenarios.csv').open() as stream:
            preview = list(csv.DictReader(stream))
        return trace, preview
    on, samples = run(True, 3)
    groups = {}
    for row in samples:
        groups.setdefault(row['step'], set()).add(row['scenario_id'])
    assert len(groups) == 5 and all(len(ids) == 3 for ids in groups.values())
    off, empty = run(False, 3)
    assert on == off, 'preview option changed numerical trace'
    assert not empty, 'disabled preview retained stale forecast data'
print('PASS: custom count, unchanged rollout, and disabling removes stale previews')
