#!/usr/bin/env python3
"""Inventory paired outcomes and probe nominal plans at saved DRO decision states."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import run_analysis_matrix as analysis
import run_comparison_matrix as comparison


def inventory(root, pair_filter=None, seeds=None):
    results, grouped = {}, {}
    paths = list(root.glob('*/seed_*/result.json'))
    paths += list(root.glob('*/seed_*/sh_mpcc*/result.json'))
    for path in sorted(paths):
        result = json.loads(path.read_text())
        style = result['solver_style']
        if style not in comparison.PAIR_STYLES:
            continue
        pair = result.get('pair', result['case'].removeprefix(style+'_'))
        if pair_filter is not None and pair != pair_filter:
            continue
        if seeds is not None and result['seed'] not in seeds:
            continue
        key = pair, result['seed']
        grouped.setdefault(key, {})[style] = {**result, 'pair': pair,
                                               'profile': result.get('profile', 'analysis_matrix')}
        results[result['case'], result['seed']] = result
    rows = []
    for (pair, seed), styles in sorted(grouped.items()):
        exemplar = next(iter(styles.values()))
        cases = [styles.get(style, {**exemplar, 'case': style+'_'+pair, 'solver_style': style})
                 for style in comparison.PAIR_STYLES]
        row = comparison.paired_result(cases, seed, results, root)
        row['dro_refusal_nominal_safe_completion'] = bool(row['status'] == 'OK'
            and row['nondro_safe_completion'] and row['dro_termination_reason'] == 'no_admissible_control')
        row['investigation_label'] = ('dro_refusal_nominal_safe_completion'
            if row['dro_refusal_nominal_safe_completion'] else 'other_observed_outcome')
        if row['dro_refusal_nominal_safe_completion']:
            row['decision_step_zero_based'] = row['dro_executed_steps']
            row['failed_decision_one_based'] = row['dro_executed_steps']+1
        rows.append(row)
    return rows


def compare_replay(source, replay, step):
    """Require identical serialized state and decision evidence over the saved prefix."""
    def trace(bundle):
        result = []
        for row in analysis.rows(bundle/'trace.csv'):
            if int(row['step']) > step:
                continue
            result.append({k: row[k] for k in ('step', 'actor', 'obstacle_id', 'mode',
                            'x', 'y', 'theta', 'v', 'vx', 'vy', 'path_progress', 'collision')})
        return result
    left, right = trace(source), trace(replay)
    if not left or left != right:
        return False, 'saved state/obstacle trace prefix differs from replay'
    a = [r for r in analysis.rows(source/'decisions.csv') if int(r['step']) <= step]
    b = [r for r in analysis.rows(replay/'decisions.csv') if int(r['step']) <= step]
    if len(a) != step+1 or len(b) != len(a):
        return False, 'decision prefix is incomplete'
    for old, new in zip(a, b):
        for key, value in old.items():
            if key != 'solve_ms' and new.get(key) != value:
                return False, f'decision {old["step"]} field {key} differs'
    return True, 'saved trace and decision prefix match exactly (timing excluded)'


def probe(row, args):
    source = Path(row['dro_artifact'])
    step = args.step if args.step is not None else row['decision_step_zero_based']
    target = args.output/row['pair']/f'seed_{row["seed"]}'/f'step_{step}'
    config = source/'resolved_config.yaml'
    command = [str(args.runner), '--config', str(config), '--seed', str(row['seed']),
               '--step', str(step), '--output', str(target), '--samples', str(args.samples),
               '--mc-seed', str(args.mc_seed)]
    if args.radius_scales:
        command += ['--radius-scales', args.radius_scales]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f'probe output exists; use a new output directory: {target}')
    result = dict(pair=row['pair'], seed=row['seed'], step_zero_based=step,
                  source_bundle=str(source), output=str(target), command=command, status='ERROR')
    source_files = [config, source/'trace.csv', source/'decisions.csv', source/'reproducibility.yaml',
                    args.matrix/'matrix.json', source.parent/'result.json']
    result['source_sha256'] = {str(p): analysis.digest(p.read_bytes()) for p in source_files}
    result['probe_sha256'] = analysis.digest(args.runner.read_bytes())
    result['investigation_script_sha256'] = analysis.digest(Path(__file__).read_bytes())
    log = target.parent/f'step_{step}.log'
    try:
        with log.open('w') as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
        if run.returncode:
            raise ValueError(f'probe exited {run.returncode}; see {log}')
        matches, reason = compare_replay(source, target/'replay/source', step)
        result.update(numeric_replay_matches=matches, replay_comparison=reason)
        result['source_unchanged'] = all(analysis.digest(Path(p).read_bytes()) == sha
                                       for p, sha in result['source_sha256'].items())
        if not result['source_unchanged']:
            raise ValueError('source artifacts changed during probe')
        result['status'] = 'OK' if matches else 'REPLAY_MISMATCH'
        result['counterfactuals'] = analysis.rows(target/'counterfactual_summary.csv')
        result['snapshot'] = analysis.rows(target/'snapshot.csv')[0]
        result['radius_sweep'] = analysis.rows(target/'radius_sweep.csv')
        # Compare current executable against the old matrix manifest as well as
        # the actual numerical replay; never claim binary identity from state equality.
        old_manifest = json.loads((args.matrix/'matrix.json').read_text())
        result['source_runner_sha256'] = old_manifest.get('runner_sha256')
        result['historical_binary_identity_established'] = False
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        result['error'] = str(error)
    analysis.dump_json(target.parent/f'step_{step}_report.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True, help='Existing analysis or comparison matrix directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair', help='Pair suffix, e.g. roundabout_o2_c2_m4 (letter o, not zero)')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--probe', action='store_true', help='Probe selected DRO refusals with nominal safe completion')
    parser.add_argument('--step', type=int, help='Override with a zero-based decision near the refusal')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--mc-seed', type=int, default=12345)
    parser.add_argument('--radius-scales', help='Optional frozen-state experimental radii: 0,0.25,0.5,0.75,1')
    parser.add_argument('--runner', type=Path, default=analysis.ROOT/'build-base/counterfactual_probe')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    if (args.samples < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not 0 <= args.mc_seed <= 2**32-1
            or (args.step is not None and not 0 <= args.step < 2**31-1)):
        parser.error('positive samples/timeout and nonnegative step required')
    args.matrix, args.output, args.runner = args.matrix.resolve(), args.output.resolve(), args.runner.resolve()
    if args.output == args.matrix or args.matrix in args.output.parents:
        parser.error('investigation output must be outside the source matrix')
    rows = inventory(args.matrix, args.pair, args.seeds)
    if not rows:
        parser.error('no matching saved pairs; use the exact suffix, e.g. roundabout_o2_c2_m4')
    args.output.mkdir(parents=True, exist_ok=True)
    analysis.write_csv(args.output/'paired_outcomes.csv', rows)
    analysis.dump_json(args.output/'paired_outcomes.json', rows)
    refusals = [r for r in rows if r['dro_refusal_nominal_safe_completion']]
    analysis.write_csv(args.output/'refusals.csv', refusals)
    analysis.dump_json(args.output/'refusals.json', refusals)
    counts = dict(pairs=len(rows), measured_pairs=sum(r['status'] == 'OK' for r in rows),
                  error_pairs=sum(r['status'] == 'ERROR' for r in rows),
                  pending_pairs=sum(r['status'] == 'PENDING' for r in rows),
                  dro_refusal_nominal_safe_completion=len(refusals))
    analysis.dump_json(args.output/'summary.json', counts)
    print(json.dumps(counts), flush=True)
    reports = []
    if args.probe:
        for row in refusals:
            result = probe(row, args)
            reports.append(result)
            analysis.dump_json(args.output/'probes.json', reports)
            for filename, field in (('counterfactuals.csv', 'counterfactuals'),
                                    ('radius_sweep.csv', 'radius_sweep')):
                analysis.write_csv(args.output/filename, [
                    dict(pair=r['pair'], seed=r['seed'], step_zero_based=r['step_zero_based'],
                         probe_status=r['status'], **measurement)
                    for r in reports for measurement in r.get(field, [])])
            print(f'{row["pair"]} seed={row["seed"]}: {result["status"]}', flush=True)
    return int(any(r['status'] == 'ERROR' for r in rows) or any(r['status'] != 'OK' for r in reports))


if __name__ == '__main__':
    sys.exit(main())
