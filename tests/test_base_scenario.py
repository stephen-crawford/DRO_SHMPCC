#!/usr/bin/env python3
"""Run one base scenario twice and retain both canonical artifact bundles."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def gif_frames(path):
    data = path.read_bytes()
    assert data[:6] in (b'GIF87a', b'GIF89a'), 'invalid GIF header'
    offset = 13 + (3 * 2 ** ((data[10] & 7) + 1) if data[10] & 128 else 0)
    count = 0
    while offset < len(data):
        block = data[offset]
        offset += 1
        if block == 0x3B:
            return count
        if block == 0x2C:
            packed = data[offset + 8]
            offset += 9
            if packed & 128:
                offset += 3 * 2 ** ((packed & 7) + 1)
            offset += 1  # LZW minimum code size
            count += 1
        else:
            assert block == 0x21, 'invalid GIF block'
            offset += 1  # extension label
        while data[offset]:
            offset += data[offset] + 1
        offset += 1
    raise AssertionError('missing GIF trailer')


def trace(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.pop('solve_time_ms')  # Only elapsed wall-clock time is nondeterministic.
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    name = args.config.stem
    args.output = args.output / name
    bundles = []
    for suffix in ('', '_repeat'):
        command = [str(args.runner.resolve()), '--config', str(args.config.resolve()),
                   '--seed', '77', '--output', str(args.output.resolve()),
                   '--label', name + suffix]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)
        bundles.append(args.output / (name + suffix))
    first, second = bundles
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    rows = trace(first / 'trace.csv')
    check(rows == trace(second / 'trace.csv'), 'same-seed traces differ')
    with (first / 'rollout.csv').open() as stream:
        result = next(csv.DictReader(stream))
    manifest = (first / 'reproducibility.yaml').read_text()
    check('completed_path: true' in manifest, 'path completion was not reached')
    check(float(result['total_progress']) >= 1.0, 'progress is below 100%')
    check(result['collision'] == '0', 'collision recorded')
    steps = int(result['total_steps'])
    expected_obstacles = int(name.split('_')[-2])
    check(len(rows) == (steps + 1) * (expected_obstacles + 1), 'trace actor/frame count differs')
    for step in range(steps + 1):
        frame = [row for row in rows if int(row['step']) == step]
        check(sum(row['actor'] == 'ego' for row in frame) == 1, 'missing/duplicate ego state')
        check(sum(row['actor'] == 'obstacle' for row in frame) == expected_obstacles,
              f'incorrect obstacle count at step {step}')
    for row in rows:
        if row['actor'] == 'obstacle' and 'dynamic' not in name:
            check(row['mode'] == 'stop' and float(row['x']) == 12.5 and
                  float(row['y']) == 0.0 and float(row['vx']) == 0.0 and
                  float(row['vy']) == 0.0, 'stationary obstacle moved or changed mode')
    if 'dynamic' in name:
        for row in rows:
            if row['actor'] == 'obstacle':
                check(row['mode'] == 'constant_velocity' and
                      abs(float(row['x']) - (12.5 + 0.5 * float(row['time_s']))) < 1e-9 and
                      float(row['y']) == 0 and float(row['vx']) == 0.5 and float(row['vy']) == 0,
                      'dynamic obstacle differs from constant-velocity fixture')
    for bundle in bundles:
        check(gif_frames(bundle / 'rollout.gif') == steps + 1,
              'GIF does not contain every initial/post-step frame')
        for filename in ('scene.csv', 'geometry.csv', 'rollout.rviz', 'resolved_config.yaml'):
            check((bundle / filename).stat().st_size > 0, f'missing/empty {filename}')
    previews = []
    for bundle in bundles:
        with (bundle / 'sampled_scenarios.csv').open() as stream:
            preview = list(csv.DictReader(stream))
        previews.append(preview)
        check(bool(preview) == bool(expected_obstacles), 'preview missing or present without obstacles')
        groups = {}
        states = {(int(row['step']), int(row['obstacle_id'])): row
                  for row in rows if row['actor'] == 'obstacle'}
        for row in preview:
            key = (int(row['step']), int(row['obstacle_id']))
            groups.setdefault(key, set()).add(row['scenario_id'])
            check(key in states, 'preview references absent obstacle/frame')
            if row['horizon_step'] == '0' and key in states:
                check(abs(float(row['x']) - float(states[key]['x'])) < 1e-12 and
                      abs(float(row['y']) - float(states[key]['y'])) < 1e-12,
                      'forecast not anchored at decision-time obstacle')
        check(all(len(ids) == 8 for ids in groups.values()), 'preview count differs from default eight')
        check(len(groups) >= steps * expected_obstacles, 'missing decision previews')
    check(previews[0] == previews[1], 'same-seed scenario previews differ')
    if 'no_noise' in name:
        for bundle in bundles:
            with (bundle / 'sampled_scenario_summary.csv').open() as stream:
                summary = list(csv.DictReader(stream))
            check(len(summary) >= steps, 'missing full-set sample diagnostics')
            check(all(int(row['scenario_count']) == int(result['S']) and
                      float(row['max_sample_deviation']) == 0.0 for row in summary),
                  'noise-free full scenario set differs within a decision')
        for row in previews[0]:
            state = states[(int(row['step']), int(row['obstacle_id']))]
            check(abs(float(row['x']) - (float(state['x']) + 0.05 * int(row['horizon_step']))) < 1e-10 and
                  float(row['y']) == 0.0, 'noise-free sample differs from deterministic prediction')
    report = {key: result[key] for key in ('total_steps', 'total_progress', 'collision',
              'min_clearance', 'S', 'safe_horizon_decisions', 'certified_decisions', 'certificate_rate')}
    report.update(scenario=name, seed=77, reproducible=rows == trace(second / 'trace.csv'),
                  errors=errors, status='FAIL' if errors else 'PASS')
    (first / 'base_test_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
