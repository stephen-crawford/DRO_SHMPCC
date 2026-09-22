#!/usr/bin/env python3
"""Support forecasts use exact solver IDs and preserve numerical decisions."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[1]/'configs/traffic_tests/support_preview.yaml'

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

with tempfile.TemporaryDirectory(prefix='support-preview-') as directory:
    root = Path(directory)
    config = root/'config.yaml'
    config.write_text(source.read_text() + '\nartifact_show_sampled_scenarios: false\nartifact_scenario_preview_count: 1\n')
    def run(label, option):
        with (root/(label+'.log')).open('w') as log:
            subprocess.run([str(runner), '--config', str(config), '--seed', '77',
                            '--output', str(root), '--label', label, option],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return root/label
    on = run('support', '--support-scenarios')
    off = run('boundaries', '--no-support-scenarios')
    expected = {r['step']: set(filter(None, r['scenario_ids'].split(';')))
                for r in rows(on/'support_scenarios.csv')}
    actual = {step: set() for step in expected}
    for row in rows(on/'sampled_scenarios.csv'):
        actual[row['step']].add(row['scenario_id'])
    assert actual == expected and any(expected.values())
    assert any(not ids for ids in expected.values()), 'exercise empty-support frames'
    for bundle in (on, off):
        trace = rows(bundle/'trace.csv')
        for row in trace:
            row.pop('solve_time_ms')
        if bundle == on:
            reference = trace
        else:
            assert reference == trace, 'visualization changed plant/ego outcome'
    assert rows(on/'support_scenarios.csv') == rows(off/'support_scenarios.csv')
    assert not rows(off/'sampled_scenarios.csv'), 'preview is disabled in boundary mode'
    svg = (on/'rollout.svg').read_text()
    assert 'id="support-scenarios"' in svg and 'id="linearized-constraints"' not in svg
    assert 'stroke="#00e5ff" stroke-width="1"' in svg, 'support forecasts need thin, distinct styling'
    assert 'id="linearized-constraints"' in (off/'rollout.svg').read_text()
    assert (on/'rollout.gif').read_bytes() != (off/'rollout.gif').read_bytes()
    assert 'artifact_show_support_scenarios: true' in (on/'resolved_config.yaml').read_text()
    # A near obstacle creates more support IDs than preview_count; all must appear
    # even if the optimizer cannot provide an admissible input for that decision.
    config.write_text(config.read_text().replace('[6.0,0.0,0.2,0.0]', '[3.0,0.0,0.2,0.0]'))
    near = run('near', '--support-scenarios')
    ids = set(rows(near/'support_scenarios.csv')[0]['scenario_ids'].split(';'))
    drawn = {r['scenario_id'] for r in rows(near/'sampled_scenarios.csv')}
    assert len(ids) > 1 and ids == drawn, 'support IDs were capped by preview_count'
    assert (near/'rollout.svg').read_text().count('<polyline points=') == 1, 'near-duplicate paths should share one visual preview'
    # Render limits must not truncate the exported solver support evidence.
    config.write_text(config.read_text() + '\nartifact_support_preview_count: 1\nobstacle_prediction_noise: true\n')
    capped = run('capped', '--support-scenarios')
    complete = set(rows(capped/'support_scenarios.csv')[0]['scenario_ids'].split(';'))
    assert len(complete) > 1
    assert {r['scenario_id'] for r in rows(capped/'sampled_scenarios.csv')} == complete
    assert (capped/'rollout.svg').read_text().count('<polyline points=') == 1
    assert '1 displayed /' in (capped/'rollout.svg').read_text()
print('PASS: exact support IDs, empty support, uncapped support, SVG/GIF override, and unchanged numerical trace')
