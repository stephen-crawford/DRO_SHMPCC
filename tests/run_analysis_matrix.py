#!/usr/bin/env python3
"""Generate/run the paired obstacle × class × environment × mode × solver matrix."""
import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {'straight': 'two_lane_highway', 's_curve': 's_curve',
                'four_way_intersection': 'four_way_intersection', 'roundabout': 'two_lane_roundabout'}
STYLES = ('sh_mpcc', 'sh_mpcc_dro', 'sh_mpcc_dro_fallback')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        path.write_text('')
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader()
        writer.writerows(records)


def load_settings(path):
    settings = json.loads(path.read_text())
    for key, allowed in [('obstacle_counts', range(1, 5)), ('class_counts', range(1, 5)),
                         ('mode_counts', range(1, 7)), ('environments', ENVIRONMENTS),
                         ('solver_styles', STYLES)]:
        values = settings[key]
        if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
            raise ValueError(f'invalid or duplicate {key}')
    seeds = settings['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in seeds):
        raise ValueError('seeds must be unique unsigned 32-bit integers')
    if type(settings['repeats']) is not int or settings['repeats'] < 1:
        raise ValueError('repeats must be positive')
    catalog = settings['mode_catalog']
    if len(catalog) < max(settings['mode_counts']) or len(catalog) != len(set(catalog)):
        raise ValueError('mode_catalog must have enough distinct modes')
    valid_modes = {'constant_velocity', 'accelerating', 'decelerating', 'stop', 'turn_left',
                   'turn_right', 'turn_left_sharp', 'turn_right_sharp', 'lane_change_left',
                   'lane_change_right', 'lane_change_left_fast', 'lane_change_right_fast'}
    if any(mode not in valid_modes for mode in catalog):
        raise ValueError('mode_catalog contains an unknown mode')
    return settings


def configurations(settings):
    for n, c, env, m, style in itertools.product(
            settings['obstacle_counts'], settings['class_counts'], settings['environments'],
            settings['mode_counts'], settings['solver_styles']):
        if c <= n:
            yield {'case': f'{style}_{env}_o{n}_c{c}_m{m}', 'obstacles': n, 'classes': c,
                   'environment': env, 'modes_per_class': m, 'solver_style': style}


def config_text(case, settings):
    # Preserve every canonical flat-YAML value in the generated file, freezing defaults.
    entries = {}
    for line in (ROOT / 'configs/default.yaml').read_text().splitlines():
        if line and not line.lstrip().startswith('#'):
            key, value = line.split(':', 1)
            entries[key.strip()] = value.strip()
    common = {
        'rare_mode': '', 'rare_mode_probability': 0.0, 'randomize_available_modes': False,
        'randomize_modes_per_obstacle': False, 'obstacle_behavior': 'mode_switching',
        'obstacles_per_class': 0, 'obstacle_history': 'shared_history_classes',
        'obstacle_place_on_path': True, 'obstacle_initial_states': [],
        'obs_arc_fractions': [0.20, 0.35, 0.50, 0.65],
        'artifact_show_sampled_scenarios': False,
        'artifact_write_rviz_replay': False,
    }
    common.update(settings.get('overrides', {}))
    # Prevent settings from silently invalidating the matrix axes or paired plant policy.
    protected = {'mpc_type', 'dro_enabled', 'num_obstacles', 'num_classes', 'num_modes',
                 'obs_modes', 'randomize_available_modes', 'randomize_modes_per_obstacle',
                 'rare_mode', 'rare_mode_probability', 'obstacle_behavior', 'environment',
                 'obstacle_place_on_path', 'obstacle_initial_states', 'obstacles_per_class'}
    if protected.intersection(settings.get('overrides', {})):
        raise ValueError('overrides may not replace matrix axes, mode supports, or placement policy')
    common.update({
        'mpc_type': ('sh_mpcc_dro_fallback' if case['solver_style'] == 'sh_mpcc_dro_fallback'
                     else 'sh_mpcc'), 'dro_enabled': case['solver_style'] != 'sh_mpcc',
        'num_obstacles': case['obstacles'], 'num_classes': case['classes'],
        'num_modes': case['modes_per_class'],
        'obs_modes': settings['mode_catalog'][:case['modes_per_class']],
        'environment': ENVIRONMENTS[case['environment']],
        'scenario_tag': case['case'], 'method_name': case['solver_style'],
        'artifact_write_analysis_csv': True, 'artifact_write_trace_csv': True,
        'artifact_write_manifest': True,
    })
    for key, value in common.items():
        entries[key] = json.dumps(value, allow_nan=False)
    return '# Generated by run_analysis_matrix.py; frozen canonical defaults + settings.\n' + ''.join(
        f'{key}: {value}\n' for key, value in entries.items())


def numeric_trace(bundle):
    result = rows(bundle / 'trace.csv')
    for row in result:
        row.pop('solve_time_ms')
    return result


FAILURE_FIELDS = (
    'backup_available', 'backup_removal_budget_exceeded', 'backup_dro_failed',
    'braking_collision_feasible', 'any_homotopy_geometrically_feasible',
    'last_qp_converged', 'sqp_sampled_collision_feasible',
    'fallback_sampled_collision_feasible',
)
FAILURE_CLASSES = ('solver_nonconvergence_and_sampled_collision',
                   'sampled_collision', 'solver_nonconvergence',
                   'other_rejection', 'unknown')


def failure_metrics(record, decisions):
    result = {field: -1 for field in FAILURE_FIELDS}
    result['failure_class'] = 'not_applicable'
    if record['termination_reason'] != 'no_admissible_control':
        return result
    if not decisions or decisions[-1]['success'] != '0':
        raise ValueError('no_admissible_control lacks a failed final decision')
    for field in FAILURE_FIELDS:
        result[field] = int(decisions[-1].get(field, -1))
        if result[field] not in (-1, 0, 1):
            raise ValueError(f'invalid failure diagnostic: {field}')
    collision = any(result[f] == 0 for f in
                    ('sqp_sampled_collision_feasible', 'fallback_sampled_collision_feasible'))
    nonconvergence = result['last_qp_converged'] == 0
    if collision and nonconvergence:
        result['failure_class'] = 'solver_nonconvergence_and_sampled_collision'
    elif collision:
        result['failure_class'] = 'sampled_collision'
    elif nonconvergence:
        result['failure_class'] = 'solver_nonconvergence'
    elif all(result[f] != -1 for f in ('last_qp_converged',
              'sqp_sampled_collision_feasible', 'fallback_sampled_collision_feasible')):
        result['failure_class'] = 'other_rejection'
    else:
        result['failure_class'] = 'unknown'
    return result


def analyze(bundle, case):
    record = rows(bundle / 'rollout.csv')[0]
    decisions = rows(bundle / 'decisions.csv')
    coverage = rows(bundle / 'mode_coverage.csv')
    trace = rows(bundle / 'trace.csv')
    geometry = {row['actor']: row for row in rows(bundle / 'geometry.csv')}
    n = case['obstacles']
    if [int(v) for v in record['effective_available_mode_counts'].split(';')] != [case['modes_per_class']] * n:
        raise ValueError('effective mode support differs from requested matrix axis')
    if not decisions or len(coverage) != n * len(decisions):
        raise ValueError('incomplete decision or all-scenario mode coverage data')
    if len({(r['step'], r['obstacle_id']) for r in coverage}) != len(coverage):
        raise ValueError('duplicate coverage records')
    for row in coverage:
        if int(row['class_id']) != int(row['obstacle_id']) % case['classes']:
            raise ValueError('class assignment mismatch')
        if (row['true_mode'] in row['sampled_modes'].split(';')) != (row['represented'] == '1'):
            raise ValueError('inconsistent mode coverage evidence')
    egos = {row['step']: row for row in trace if row['actor'] == 'ego'}
    obstacles = [row for row in trace if row['actor'] == 'obstacle']
    if len(egos) != int(record['total_steps']) + 1 or len(obstacles) != n * len(egos):
        raise ValueError('incomplete realized trace')
    ego_geometry = geometry['ego']
    count = int(ego_geometry['num_discs'])
    length = float(ego_geometry['length'])
    offsets = [0.] if count == 1 else [-length / 2 + i * length / (count - 1) for i in range(count)]
    radius = float(ego_geometry['radius']) + float(geometry['obstacle']['radius']) + float(ego_geometry['safety_margin'])
    margins = []
    for obstacle in obstacles:
        ego = egos[obstacle['step']]
        x, y, theta = (float(ego[k]) for k in ('x', 'y', 'theta'))
        distance = min(math.hypot(x + d * math.cos(theta) - float(obstacle['x']),
                                 y + d * math.sin(theta) - float(obstacle['y'])) for d in offsets)
        margins.append({'step': int(obstacle['step']), 'time_s': float(obstacle['time_s']),
                        'obstacle_id': int(obstacle['obstacle_id']), 'distance_m': distance,
                        'combined_radius_m': radius, 'margin_m': distance - radius})
    write_csv(bundle / 'conservatism.csv', margins)
    times = [float(d['solve_ms']) for d in decisions]
    certified = sum(int(d['certified']) for d in decisions)
    requested = sum(int(d['certificate_requested']) for d in decisions)
    if certified != int(record['certified_decisions']) or requested != int(record['safe_horizon_decisions']):
        raise ValueError('certificate counters disagree with decision evidence')
    return {
        **failure_metrics(record, decisions),
        'collision': int(record['collision']), 'completed_path': int(record['completed_path']),
        'termination_reason': record['termination_reason'], 'executed_steps': int(record['total_steps']),
        'certified_decisions': certified, 'sh_decisions': requested,
        'missed_mode_rollout': int(any(r['represented'] == '0' for r in coverage)),
        'missed_obstacle_decisions': sum(r['represented'] == '0' for r in coverage),
        'mode_checks': len(coverage), 'solve_count': len(times),
        'solve_total_ms': sum(times), 'solve_mean_ms': statistics.mean(times),
        'solve_max_ms': max(times),
        'control_effort': sum(float(d['applied_control_effort']) for d in decisions),
        'max_conservatism_m': min(r['margin_m'] for r in margins),
        'average_conservatism_m': statistics.mean(r['margin_m'] for r in margins),
        'margin_sum_m': sum(r['margin_m'] for r in margins), 'margin_count': len(margins),
        'plant_seed': int(record['plant_seed']), 'controller_seed': int(record['controller_seed']),
        'initial_placement': [{k: row[k] for k in ('actor', 'obstacle_id', 'x', 'y', 'theta', 'v', 'vx', 'vy')}
                              for row in trace if row['step'] == '0'],
        'backend': record['qp_backend'], 'solver_identity': record['qp_solver_identity'],
    }


def repeat_signature(bundle, metrics):
    stable_metrics = {k: v for k, v in metrics.items() if not k.startswith('solve_')}
    decisions = rows(bundle / 'decisions.csv')
    for row in decisions:
        row.pop('solve_ms')
    evidence = [numeric_trace(bundle), decisions, rows(bundle / 'mode_coverage.csv'), stable_metrics]
    if (bundle/'attempts.csv').exists():
        attempts = rows(bundle/'attempts.csv')
        for row in attempts:
            row.pop('solve_ms')
        evidence.extend([attempts, rows(bundle/'mode_mechanism.csv')])
        if (bundle/'transport_costs.csv').exists():
            evidence.append(rows(bundle/'transport_costs.csv'))
    return digest(json.dumps(evidence, sort_keys=True).encode())


def trial_root(output, case, seed):
    if 'pair_directory' in case:
        return output / case['pair_directory'] / f'seed_{seed}' / case['solver_style']
    return output / case['case'] / f'seed_{seed}'


def run_trial(case, seed, args, settings, identity):
    root = trial_root(args.output, case, seed)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / 'result.json'
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['identity'] != identity:
            raise ValueError(f'stale result: {result_path}')
        if previous['status'] == 'OK':
            return previous
    result = {**case, 'seed': seed, 'identity': identity, 'status': 'ERROR', 'repeats': []}
    try:
        signatures = []
        for repeat in range(settings['repeats']):
            label = f'repeat_{repeat}'

            config_path = (
                args.output /
                'configs' /
                (case['case'] + '.yaml')
            ).resolve()

            if not config_path.exists():
                raise FileNotFoundError(
                    f'config missing immediately before launch: '
                    f'{config_path}'
                )

            if not config_path.is_file():
                raise RuntimeError(
                    f'config is not a regular file immediately before launch: '
                    f'{config_path}'
                )

            # Force an actual open/read from the same Python process immediately
            # before launching the C++ runner.
            try:
                config_bytes = config_path.read_bytes()
            except OSError as error:
                raise OSError(
                    f'config exists but Python cannot read it immediately '
                    f'before launch: {config_path}: {error}'
                ) from error

            if not config_bytes:
                raise RuntimeError(
                    f'config is empty immediately before launch: '
                    f'{config_path}'
                )

            command = [
                str(args.runner),
                '--config',
                str(config_path),
                '--seed',
                str(seed),
                '--output',
                str(root),
                '--label',
                label,
            ]
            started = time.monotonic()
            with (root / (label + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
            metrics = analyze(root / label, case)
            signatures.append(repeat_signature(root / label, metrics))
            result['repeats'].append({'wall_seconds': time.monotonic() - started,
                                      'metrics': metrics, 'signature': signatures[-1]})
            if len(set(signatures)) != 1:
                raise ValueError('same-seed numerical outcomes or mode coverage differ across repeats')
        result['status'] = 'OK'
        result['repeatable'] = True if settings['repeats'] > 1 else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as error:
        result['error'] = str(error)
    dump_json(result_path, result)
    return result


def aggregate(results, cases=None, seed_count=None):
    grouped = {}
    for result in results:
        grouped.setdefault(result['case'], []).append(result)
    templates = {c['case']: c for c in (cases or results)}
    for name in templates:
        grouped.setdefault(name, [])
    summaries = []
    for name, group in sorted(grouped.items()):
        good = [r['repeats'][0]['metrics'] for r in group if r['status'] == 'OK']
        summary = {k: templates[name][k] for k in ('case', 'obstacles', 'classes', 'environment', 'modes_per_class', 'solver_style')}
        summary.update(expected_rollouts=seed_count if seed_count is not None else len(group),
                       attempted_rollouts=len(group), measured_rollouts=len(good), errors=len(group)-len(good),
                       pending_rollouts=max(0, (seed_count or len(group))-len(group)))
        sh = sum(m['sh_decisions'] for m in good)
        solves = sum(m['solve_count'] for m in good)
        margins = sum(m['margin_count'] for m in good)
        summary.update(
            collision_rate=sum(m['collision'] for m in good)/len(good) if good else None,
            sh_certification_rate=sum(m['certified_decisions'] for m in good)/sh if sh else None,
            sh_decisions=sh,
            missed_mode_rollouts=sum(m['missed_mode_rollout'] for m in good),
            missed_mode_rate=sum(m['missed_mode_rollout'] for m in good)/len(good) if good else None,
            compute_mean_ms=sum(m['solve_total_ms'] for m in good)/solves if solves else None,
            compute_max_ms=max((m['solve_max_ms'] for m in good), default=None),
            mean_control_effort=statistics.mean(m['control_effort'] for m in good) if good else None,
            max_conservatism_m=min((m['max_conservatism_m'] for m in good), default=None),
            average_conservatism_m=sum(m['margin_sum_m'] for m in good)/margins if margins else None,
            completed_rollouts=sum(m['completed_path'] for m in good),
            no_admissible_control_rollouts=sum(m['termination_reason']=='no_admissible_control' for m in good))
        failures = [m for m in good if m['termination_reason'] == 'no_admissible_control']
        for field in FAILURE_FIELDS:
            for value, label in ((1, 'true'), (0, 'false'), (-1, 'unknown')):
                summary[f'failure_{field}_{label}_rollouts'] = sum(
                    m.get(field, -1) == value for m in failures)
        for classification in FAILURE_CLASSES:
            summary[f'failure_{classification}_rollouts'] = sum(
                m.get('failure_class', 'unknown') == classification for m in failures)
        summaries.append(summary)
    return summaries


def per_seed_summary(results):
    summaries = []
    for result in sorted(results, key=lambda r: (r['case'], r['seed'])):
        summary = aggregate([result], [result], 1)[0]
        summaries.append({'case': result['case'], 'seed': result['seed'],
                          'status': result['status'], 'error': result.get('error', ''),
                          **summary})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=ROOT/'configs/analysis_matrix/settings.json')
    parser.add_argument('--runner', type=Path, default=ROOT/'build-base/experiment_runner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--case', help='Exact case name; manifests always contain the whole configured matrix')
    parser.add_argument('--jobs', type=int, default=1, help='Use 1 for comparable timing measurements')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    args.output = args.output.resolve()
    args.runner = args.runner.resolve()
    settings = load_settings(args.settings)
    cases = list(configurations(settings))
    if not cases:
        parser.error('no valid combinations (classes must be <= obstacles)')
    selected = [c for c in cases if args.case is None or c['case'] == args.case]
    if not selected:
        parser.error('unknown case')
    selected_names = {c['case'] for c in selected}
    texts = {c['case']: config_text(c, settings) for c in cases}
    manifest = {'schema': 1, 'settings': settings, 'cases': cases,
                'configs': {name: digest(text.encode()) for name, text in texts.items()},
                'runner_sha256': digest(args.runner.read_bytes()),
                'analysis_script_sha256': digest(Path(__file__).read_bytes()),
                'runner': str(args.runner), 'jobs': args.jobs, 'timeout_seconds': args.timeout}
    identity = digest(json.dumps(manifest, sort_keys=True).encode())
    manifest['identity'] = identity
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'matrix.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('output contains a different manifest; use a new output directory')
    dump_json(manifest_path, manifest)
    (args.output/'configs').mkdir(exist_ok=True)
    for name, text in texts.items():
        if name not in selected_names:
            continue
        (args.output/'configs'/(name+'.yaml')).write_text(text)
    print(f'{len(selected)} selected configurations; {len(selected)*len(settings["seeds"])} seed trials; '
          f'{len(selected)*len(settings["seeds"])*settings["repeats"]} executions', flush=True)
    if args.generate_only:
        return 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_trial, c, seed, args, settings, identity)
                   for c in selected for seed in settings['seeds']]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f'{result["case"]} seed={result["seed"]}: {result["status"]} {result.get("error", "")}', flush=True)
    # Reuse only the selected cases; unrelated case directories remain untouched.
    saved = [json.loads(p.read_text()) for name in sorted(selected_names)
             for p in (args.output/name).glob('seed_*/result.json')]
    saved = [r for r in saved if r['identity'] == identity and r['case'] in selected_names]
    dump_json(args.output/'results.json', saved)
    flat = []
    for result in saved:
        base = {k: result[k] for k in ('case', 'seed', 'status', 'obstacles', 'classes',
                                       'environment', 'modes_per_class', 'solver_style')}
        base['error'] = result.get('error', '')
        if not result['repeats']:
            flat.append(base)
        for repeat, observation in enumerate(result['repeats']):
            flat.append({**base, 'repeat': repeat, 'wall_seconds': observation['wall_seconds'],
                         **{k: v for k, v in observation['metrics'].items() if k != 'initial_placement'}})
    write_csv(args.output/'rollouts.csv', flat)
    summary = aggregate(saved, selected, len(settings['seeds']))
    write_csv(args.output/'summary.csv', summary)
    dump_json(args.output/'summary.json', summary)
    write_csv(args.output/'summary_per_seed.csv', per_seed_summary(saved))
    # Paired placements must match across classes/modes/styles; counts use a common prefix.
    placements = {}
    for result in saved:
        if result['status'] != 'OK':
            continue
        for actor in result['repeats'][0]['metrics']['initial_placement']:
            key = (result['environment'], result['seed'], actor['actor'], actor['obstacle_id'])
            if key in placements and placements[key] != actor:
                raise ValueError(f'paired initial placement mismatch: {key}')
            placements[key] = actor
    errors = sum(r['status'] != 'OK' for r in results)
    print(f'Matrix complete: {len(results)}/{len(selected)*len(settings["seeds"])} '
          f'seed trials covered; {errors} errors. Reports written to {args.output}.', flush=True)
    return int(errors != 0)


if __name__ == '__main__':
    sys.exit(main())
