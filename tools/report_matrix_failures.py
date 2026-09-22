#!/usr/bin/env python3
"""Report unsuccessful logging in a saved analysis-matrix snapshot.

Run from the repository root:
    python3 tools/report_matrix_failures.py
    python3 tools/report_matrix_failures.py --seed 77

Uses matrix.json for planned trials and results.json for runner status. OK means
successful logging, not path completion or collision-free motion. Reads existing
artifacts only; writes reports to a separate directory and never reruns trials.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


CONDITION = ('environment', 'obstacles', 'classes', 'modes_per_class', 'seed')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(root, output, seeds=None):
    manifest = json.loads((root / 'matrix.json').read_text())
    results = json.loads((root / 'results.json').read_text())
    planned_seeds = manifest['settings']['seeds']
    if seeds is not None:
        unknown = set(seeds) - set(planned_seeds)
        if unknown:
            raise ValueError(f'Seeds absent from manifest: {sorted(unknown)}')
        planned_seeds = [seed for seed in planned_seeds if seed in seeds]
    indexed = {}
    for result in results:
        key = result['case'], result['seed']
        if key in indexed:
            raise ValueError(f'Duplicate result: {key}')
        if result.get('identity') != manifest.get('identity'):
            raise ValueError(f'Result identity differs from manifest: {key}')
        indexed[key] = result

    styles = sorted({case['solver_style'] for case in manifest['cases']})
    counts = {style: Counter() for style in styles}
    groups = defaultdict(dict)
    failures = []
    for case in manifest['cases']:
        style = case['solver_style']
        for seed in planned_seeds:
            condition = {field: case[field] for field in CONDITION if field != 'seed'}
            condition['seed'] = seed
            key = tuple(condition[field] for field in CONDITION)
            if style in groups[key]:
                raise ValueError(f'Duplicate controller in paired condition: {key}, {style}')
            result = indexed.get((case['case'], seed))
            status = result['status'] if result is not None else 'MISSING'
            groups[key][style] = status
            counts[style][status] += 1
            if status == 'OK':
                continue
            trial = root / case.get('pair_directory', case['case']) / f'seed_{seed}'
            if 'pair_directory' in case:
                trial /= style
            logs = sorted(trial.glob('repeat_*.log'))
            excerpts = []
            for log in logs:
                with log.open(errors='replace') as stream:
                    tail = ''.join(deque(stream, maxlen=8)).strip()
                excerpts.append(f'{log.name}:\n{tail}')
            failures.append({
                **condition, 'solver_style': style, 'case': case['case'],
                'status': status,
                'error': result.get('error', '') if result is not None else 'No saved result',
                'result_path': str(trial / 'result.json'),
                'config_path': str(root / 'configs' / (case['case'] + '.yaml')),
                'log_paths': '\n'.join(str(log) for log in logs),
                'log_tail': '\n\n'.join(excerpts),
            })

    pairs = []
    for key, statuses in sorted(groups.items()):
        absent = [style for style in styles if statuses.get(style) != 'OK']
        pairs.append({
            **dict(zip(CONDITION, key)),
            **{style: statuses.get(style, 'NOT_PLANNED') for style in styles},
            'all_methods_ok': int(not absent),
            'unsuccessful_methods': ';'.join(absent),
        })
    failures.sort(key=lambda row: (row['seed'], row['case']))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'failed_rollouts.csv',
              [*CONDITION, 'solver_style', 'case', 'status', 'error',
               'result_path', 'config_path', 'log_paths', 'log_tail'], failures)
    pair_fields = [*CONDITION, *styles, 'all_methods_ok', 'unsuccessful_methods']
    write_csv(output / 'paired_conditions.csv', pair_fields, pairs)
    excluded = [row for row in pairs if not row['all_methods_ok']]
    write_csv(output / 'excluded_conditions.csv', pair_fields, excluded)
    seed_rows = []
    for seed in sorted(planned_seeds):
        selected = [row for row in failures if row['seed'] == seed]
        seed_rows.append({
            'seed': seed,
            **{style: sum(row['solver_style'] == style for row in selected) for style in styles},
            'unsuccessful_rollouts': len(selected),
            'excluded_conditions': sum(row['seed'] == seed for row in excluded),
        })
    write_csv(output / 'failures_per_seed.csv',
              ['seed', *styles, 'unsuccessful_rollouts', 'excluded_conditions'], seed_rows)
    planned = sum(sum(counter.values()) for counter in counts.values())
    ok = sum(counter['OK'] for counter in counts.values())
    print(f'Planned: {planned}; logging OK: {ok}; unsuccessful/missing: {planned - ok}')
    for style, counter in counts.items():
        print(f'  {style}: ' + ', '.join(f'{status}={count}' for status, count in sorted(counter.items())))
    print(f'Paired conditions: {len(pairs)}; all methods OK: {len(pairs) - len(excluded)}; '
          f'excluded: {len(excluded)}')
    print('Unsuccessful rollouts by seed: ' + ', '.join(
        f"{row['seed']}={row['unsuccessful_rollouts']}" for row in seed_rows))
    print(f'Reports: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--matrix', type=Path, default=Path('build-base/analysis-matrix'),
                        help='Directory containing matrix.json and results.json')
    parser.add_argument('--output', type=Path,
                        help='Report directory (default: MATRIX/failure-report)')
    parser.add_argument('--seed', type=int, action='append',
                        help='Only report this seed; may be repeated')
    args = parser.parse_args()
    root = args.matrix.resolve()
    output = args.output.resolve() if args.output else root / 'failure-report'
    try:
        report(root, output, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
