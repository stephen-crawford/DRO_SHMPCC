#!/usr/bin/env python3
"""Descriptive, complete-five-arm comparisons of normalized scrubber outputs."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

ARMS = {'sh_mpcc': 'non_dro', 'sh_mpcc_dro': 'dro',
        'sh_mpcc_extra': 'extra_nominal', 'sh_mpcc_resample': 'nominal_resample',
        'sh_mpcc_dro_fallback': 'dro_fallback'}
KEYS = ('test_root', 'matrix_identity', 'pair_case', 'seed', 'repeat')
METRICS = ('collision', 'path_completed', 'final_path_fraction', 'min_actual_clearance',
           'mean_controller_solve_ms', 'fallback_step_count')


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(fields + [k for r in rows for k in r])))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row.get(k, '') for k in KEYS)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    planned = defaultdict(set)
    for row in read(source/'expected_runs.csv'):
        planned[key(row)].add(ARMS.get(row['solver_style'], row['solver_style']))
    for row in read(source/'run_summary.csv'):
        groups[key(row)][row['variant']].append(row)
    pairing = {}
    for row in read(source/'report_all_comparisons.csv'):
        a, b = row.get('controller_a'), row.get('controller_b')
        if a == 'sh_mpcc' and b in ARMS:
            pairing[(row.get('test_root', ''), row.get('matrix_identity', ''),
                     row.get('pair', ''), row.get('seed', ''), ARMS[b])] = row.get('status')
    availability, paired, deltas = [], [], []
    for cell in sorted(set(planned) | set(groups)):
        group = groups[cell]
        identity = dict(zip(KEYS, cell))
        row = dict(identity)
        accepted = {}
        for arm in ARMS.values():
            runs = group.get(arm, [])
            good = [r for r in runs if r.get('log_complete') == '1' and
                    r.get('trial_status') == 'OK' and
                    number(r.get('collision')) in (0, 1) and
                    number(r.get('path_completed')) in (0, 1)]
            row[arm] = len(good)
            row[arm+'_status'] = ('DUPLICATE' if len(runs) > 1 else
                'OK' if len(good) == 1 else 'MISSING' if not runs else
                ('INCOMPLETE' if runs[0].get('trial_status') == 'OK' else runs[0].get('trial_status', 'INCOMPLETE')))
            if len(runs) == len(good) == 1:
                accepted[arm] = good[0]
        paired_plant = all(pairing.get(cell[:4]+(arm,)) == 'OK'
                           for arm in ARMS.values() if arm != 'non_dro')
        row['plant_pairing_status'] = 'OK' if paired_plant else 'MISSING_OR_ERROR'
        complete = len(accepted) == len(ARMS) and paired_plant
        row['paired_complete'] = int(complete)
        availability.append(row)
        if not complete:
            continue
        for arm, run in accepted.items():
            paired.append(dict(run, **identity))
            if arm == 'non_dro':
                continue
            delta = dict(identity, variant=arm)
            for metric in METRICS:
                a, b = number(run.get(metric)), number(accepted['non_dro'].get(metric))
                delta['delta_'+metric] = a-b if a is not None and b is not None else ''
            delta['nominal_collision'] = accepted['non_dro']['collision']
            delta['arm_collision'] = run['collision']
            deltas.append(delta)
    write(output/'completeness.csv', availability, list(KEYS)+['paired_complete'])
    write(output/'paired_runs.csv', paired, list(KEYS)+['variant'])
    write(output/'paired_differences.csv', deltas, list(KEYS)+['variant'])
    # Keep repeat strata separate: repeated runs never increase the seed denominator.
    strata = defaultdict(list)
    for run in paired:
        strata[tuple(run.get(k, '') for k in KEYS if k != 'seed')+(run['variant'],)].append(run)
    summaries = []
    fields = [k for k in KEYS if k != 'seed']+['variant']
    for cell, runs in sorted(strata.items()):
        row = dict(zip(fields, cell), paired_seeds=len({r['seed'] for r in runs}))
        for metric in METRICS:
            values = [v for r in runs if (v := number(r.get(metric))) is not None]
            row[metric+'_n'] = len(values)
            row[metric+'_mean'] = statistics.mean(values) if values else ''
            row[metric+'_median'] = statistics.median(values) if values else ''
        summaries.append(row)
    write(output/'outcomes.csv', summaries, fields+['paired_seeds'])
    discordance = defaultdict(lambda: dict(n00=0, n01=0, n10=0, n11=0))
    for row in deltas:
        cell = tuple(row[k] for k in fields)
        discordance[cell]['n'+str(int(float(row['nominal_collision'])))+str(int(float(row['arm_collision'])))] += 1
    write(output/'collision_pairs.csv', [dict(zip(fields, cell), **counts)
          for cell, counts in sorted(discordance.items())], fields+['n00','n01','n10','n11'])
    # Use structured attempt zero evidence. Retry allocations stay in raw artifacts.
    complete_keys = {key(r) for r in paired}
    modes = [r for r in read(source/'artifact_mode_mechanism.csv')
             if key(r) in complete_keys and r.get('attempt') == '0']
    def mode_key(r):
        return key(r)+(r['step'], r['obstacle_id'], r['mode'])
    reference = {mode_key(r): r for r in modes if r.get('solver_style') == 'sh_mpcc_dro'}
    budgets = defaultdict(float)
    for r in modes:
        budgets[key(r)+(r.get('solver_style'), r['step'], r['obstacle_id'])] += float(r['sampled_count'])
    mechanism = []
    for r in modes:
        ref = reference.get(mode_key(r))
        if ref is None:
            continue
        p, q = number(ref.get('nominal_probability')), number(ref.get('sampling_probability'))
        n, budget = number(r.get('sampled_count')), number(r.get('scenario_count'))
        if p is None or q is None or n is None:
            continue
        # scenario_count is carried by attempts.csv, not all mechanism schema versions.
        if budget is None:
            budget = budgets[key(r)+(r.get('solver_style'), r['step'], r['obstacle_id'])]
        mechanism.append(dict(zip(KEYS, key(r)), variant=ARMS[r['solver_style']],
            step=r['step'], obstacle_id=r['obstacle_id'], mode=r['mode'],
            wdro_p=p, wdro_q=q, delta_p=q-p,
            shift_group='increased' if q > p else 'decreased' if q < p else 'unchanged',
            nominal_probability=r['nominal_probability'], sampling_probability=r['sampling_probability'],
            sampled_count=n, scenario_count=budget, sampled_fraction=n/budget if budget else '',
            represented=int(n > 0), represented_ge_2=int(n >= 2)))
    write(output/'mode_representation.csv', mechanism, list(KEYS)+['variant','step','obstacle_id','mode'])
    mgroups = defaultdict(list)
    mfields = list(KEYS)+['variant','shift_group']
    for row in mechanism:
        mgroups[tuple(row[k] for k in mfields)].append(row)
    write(output/'mechanism_per_seed.csv', [dict(zip(mfields, cell), mode_checks=len(rr),
          represented_fraction=statistics.mean(r['represented'] for r in rr),
          represented_ge_2_fraction=statistics.mean(r['represented_ge_2'] for r in rr),
          mean_sampled_count=statistics.mean(r['sampled_count'] for r in rr))
          for cell, rr in sorted(mgroups.items())], mfields)
    (output/'NOTES.txt').write_text(
        'Only unique, complete, trial-OK five-arm cells with runner-checked plant pairing enter outcomes. Missing arms remain in completeness.csv.\n'
        'Repeat strata remain separate; decisions are descriptive mechanism observations, not independent seeds.\n'
        'Mechanism uses attempt zero and WDRO probability-shift classification at the same step/obstacle/mode.\n'
        'Rows without a shared WDRO step are excluded from mechanism comparisons; compare mode_checks.\n'
        'Controller no_admissible_control is an outcome; trial ERROR is an execution/evidence error.\n'
        'Certificates must come from artifact_decisions.csv; legacy support counts are not certificates.\n'
        'No causal identification or probability guarantee follows from these descriptive summaries.\n')
    print(f'{len(availability)} planned/observed cells; {sum(r["paired_complete"] for r in availability)} complete five-arm cells')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (args.source/'expected_runs.csv').exists():
        parser.error('expected_runs.csv is required; rerun tools/scrub_artifacts.py')
    if args.source.resolve() == args.out.resolve():
        parser.error('use a separate analysis output directory')
    analyze(args.source, args.out)


if __name__ == '__main__':
    main()
