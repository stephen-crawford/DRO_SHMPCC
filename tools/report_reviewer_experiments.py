#!/usr/bin/env python3
"""Read-only paired outcomes, cycle latency, clearance, and mode-history audit."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from collections import defaultdict


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    with path.open('w') as stream:
        if data:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(index-lo)*(values[hi]-values[lo])


def seed_block_interval(records, draws=2000, seed=1729):
    """Percentile interval resampling entire master-seed blocks across setups.

    Records are (master_seed, value). This preserves within-seed pairing and
    cross-configuration dependence. With <2 seeds uncertainty is unidentified.
    It is a descriptive bootstrap, not a finite-sample coverage guarantee.
    """
    blocks = defaultdict(list)
    for key, value in records:
        blocks[key].append(value)
    if len(blocks) < 2:
        return None, None
    totals = [(sum(v), len(v)) for _, v in sorted(blocks.items())]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = rng.choices(totals, k=len(totals))
        estimates.append(sum(s for s, n in sample)/sum(n for s, n in sample))
    return quantile(estimates, .025), quantile(estimates, .975)


def paired_outcomes(results, methods, draws=2000):
    groups = defaultdict(dict)
    for r in results:
        if r['solver_style'] not in methods:
            continue
        key = tuple(r.get(k, 'standard') for k in
                    ('profile', 'environment', 'obstacles', 'classes', 'modes_per_class', 'seed'))
        if r['solver_style'] in groups[key]:
            raise ValueError(f'duplicate configuration/seed/controller: {key}')
        groups[key][r['solver_style']] = r
    good = []
    for key, group in groups.items():
        if all(m in group and group[m]['status']=='OK' for m in methods):
            good.append((key, group))
    rates, overlap = [], []
    scopes = sorted({key[0] for key in groups})
    for scope in scopes:
        selected = [(k, g) for k, g in good if k[0]==scope]
        fail = lambda r: int(r['repeats'][0]['metrics']['termination_reason']=='no_admissible_control')
        for method in methods:
            records = [(key[-1], fail(group[method])) for key, group in selected]
            lo, hi = seed_block_interval(records, draws)
            metrics = [g[method]['repeats'][0]['metrics'] for _, g in selected]
            rates.append(dict(profile=scope, controller=method, matched_rollouts=len(records),
                no_admissible_control=sum(x for _, x in records),
                rate=statistics.mean(x for _, x in records) if records else None,
                seed_block_ci95_low=lo, seed_block_ci95_high=hi,
                master_seed_blocks=len({s for s, _ in records}),
                collisions=sum(m['collision'] for m in metrics),
                completed_rollouts=sum(m['completed_path'] for m in metrics)))
        for a, b in itertools.combinations(methods, 2):
            counts = defaultdict(int)
            delta = []
            for key, group in selected:
                fa, fb = fail(group[a]), fail(group[b])
                counts[fa, fb] += 1
                delta.append((key[-1], fb-fa))
            lo, hi = seed_block_interval(delta, draws)
            overlap.append(dict(profile=scope, controller_a=a, controller_b=b,
                matched_rollouts=len(delta), neither_refuses=counts[0,0],
                only_b_refuses=counts[0,1], only_a_refuses=counts[1,0], both_refuse=counts[1,1],
                rate_difference_b_minus_a=statistics.mean(v for _, v in delta) if delta else None,
                paired_seed_block_ci95_low=lo, paired_seed_block_ci95_high=hi))
    return rates, overlap, good, len(groups)-len(good)


def bundle_metrics(bundle, deadline_ms, near_margin):
    decisions = rows(bundle/'decisions.csv')
    if not decisions:
        raise ValueError(f'empty decisions: {bundle}')
    times = [float(r['solve_ms']) for r in decisions]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError(f'invalid latency: {bundle}')
    flags_available = all('nominal_fallback_attempted' in r and
                          r['nominal_fallback_attempted'] in ('0', '1') for r in decisions)
    fallback = [float(r['solve_ms']) for r in decisions if r.get('nominal_fallback_attempted')=='1']
    ordinary = [float(r['solve_ms']) for r in decisions if r.get('nominal_fallback_attempted')=='0']
    margins = [float(r['margin_m']) for r in rows(bundle/'conservatism.csv')]
    if not margins or any(not math.isfinite(m) for m in margins):
        raise ValueError(f'missing/nonfinite realized margins: {bundle}')
    geometry = {r['actor']: r for r in rows(bundle/'geometry.csv')}
    safety = float(geometry['ego']['safety_margin'])
    trace = rows(bundle/'trace.csv')
    modes = defaultdict(list)
    for r in trace:
        if r['actor']=='obstacle':
            modes[r['obstacle_id']].append((int(r['step']), r['mode']))
    transitions = [(a[1], b[1]) for seq in modes.values()
                   for a, b in zip(sorted(seq), sorted(seq)[1:]) if b[0]==a[0]+1]
    result = dict(decisions=len(times), fallback_decisions=len(fallback) if flags_available else None,
        fallback_fraction=len(fallback)/len(times) if flags_available else None,
        cycle_median_ms=quantile(times,.5), cycle_p95_ms=quantile(times,.95),
        cycle_max_ms=max(times), deadline_ms=deadline_ms,
        deadline_misses=sum(t > deadline_ms for t in times),
        deadline_miss_rate=sum(t > deadline_ms for t in times)/len(times),
        fallback_cycle_median_ms=quantile(fallback,.5), fallback_cycle_p95_ms=quantile(fallback,.95),
        no_fallback_cycle_median_ms=quantile(ordinary,.5), no_fallback_cycle_p95_ms=quantile(ordinary,.95),
        scenarios_min=min(int(r['scenario_count']) for r in decisions),
        scenarios_max=max(int(r['scenario_count']) for r in decisions),
        minimum_safety_margin_m=min(margins), margin_p01_m=quantile(margins,.01),
        maximum_safety_penetration_m=max(0., -min(margins)),
        maximum_physical_penetration_m=max(0., -min(margins)-safety),
        near_collision_threshold_m=near_margin, near_collision_rollout=int(min(margins)<near_margin),
        sampled_state_safety_violation=int(min(margins)<0),
        sampled_state_obstacle_exposures=len(margins),
        sampled_state_safety_violation_fraction=sum(m<0 for m in margins)/len(margins),
        executed_steps=max(int(r['step']) for r in trace),
        observed_transitions=len(transitions),
        observed_mode_repeat_fraction=sum(a==b for a,b in transitions)/len(transitions) if transitions else None)
    return result, times, fallback, ordinary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['sh_mpcc','sh_mpcc_dro','sh_mpcc_dro_fallback'])
    p.add_argument('--case-contains', default='')
    p.add_argument('--outcomes-only', action='store_true')
    p.add_argument('--deadline-ms', type=float, default=100.)
    p.add_argument('--near-margin', type=float, default=.2)
    p.add_argument('--bootstrap-draws', type=int, default=2000)
    args = p.parse_args(argv)
    args.matrix, args.output = args.matrix.resolve(), args.output.resolve()
    if args.output==args.matrix or args.matrix in args.output.parents:
        p.error('report output must be outside the source matrix')
    if (not math.isfinite(args.deadline_ms) or args.deadline_ms<=0 or
        not math.isfinite(args.near_margin) or args.near_margin<0 or args.bootstrap_draws<100 or
        len(set(args.methods))!=len(args.methods) or len(args.methods)<2):
        p.error('invalid deadline, margin, bootstrap count, or methods')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('report output must be empty; source reports are never overwritten')
    source = args.matrix/'results.json'
    results = [r for r in json.loads(source.read_text()) if args.case_contains in r['case']]
    if not results:
        p.error('no selected results')
    if set(args.methods)-{r['solver_style'] for r in results}:
        p.error('a requested controller is absent; run its experiment before comparing')
    rates, overlap, matched, excluded = paired_outcomes(results, args.methods, args.bootstrap_draws)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'rates.csv', rates)
    write_csv(args.output/'paired_overlap.csv', overlap)
    details, cycles = [], defaultdict(lambda: ([], [], []))
    if not args.outcomes_only:
        for key, group in matched:
            for method in args.methods:
                r = group[method]
                bundle = args.matrix/r['case']/f'seed_{r["seed"]}'/'repeat_0'
                metrics, times, fallback, ordinary = bundle_metrics(bundle, args.deadline_ms, args.near_margin)
                details.append(dict(case=r['case'], seed=r['seed'], profile=key[0], controller=method, **metrics))
                for target, data in zip(cycles[key[0],method], (times, fallback, ordinary)):
                    target.extend(data)
    write_csv(args.output/'rollout_diagnostics.csv', details)
    latency = []
    for (profile,method), (times,fallback,ordinary) in sorted(cycles.items()):
        selected=[r for r in details if r['profile']==profile and r['controller']==method]
        known=all(r['fallback_decisions'] is not None for r in selected)
        latency.append(dict(profile=profile, controller=method, decisions=len(times),
            cycle_median_ms=quantile(times,.5), cycle_p95_ms=quantile(times,.95),
            fallback_cycle_median_ms=quantile(fallback,.5), fallback_cycle_p95_ms=quantile(fallback,.95),
            no_fallback_cycle_median_ms=quantile(ordinary,.5), no_fallback_cycle_p95_ms=quantile(ordinary,.95),
            fallback_fraction=len(fallback)/len(times) if known else None,
            deadline_ms=args.deadline_ms, deadline_miss_rate=sum(t>args.deadline_ms for t in times)/len(times),
            near_collision_rollouts=sum(r['near_collision_rollout'] for r in selected),
            measured_rollouts=len(selected),
            rollout_min_margin_p05_m=quantile([r['minimum_safety_margin_m'] for r in selected],.05)))
    write_csv(args.output/'cycle_latency.csv', latency)
    summary = dict(matched_groups=len(matched), excluded_groups=excluded,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods=args.methods, deadline_ms=args.deadline_ms, near_margin_m=args.near_margin,
        bootstrap_draws=args.bootstrap_draws, bootstrap_seed=1729,
        uncertainty='95% percentile bootstrap over master-seed blocks; descriptive, not a coverage guarantee',
        selection='all requested controllers OK, repeat zero; same configuration and master seed',
        caveats=['A non-refusal is not necessarily completion or collision-free navigation.',
                 'Standalone overlap does not reproduce a hybrid at the same ego state.',
                 'Latency pools complete control cycles, including failed final decisions; not rollout means.',
                 'Clearance is evaluated at recorded states, not continuously between integration steps.',
                 'Observed mode persistence does not validate independence or stationarity.',
                 'Few master seeds limit bootstrap precision; exclusion can bias inference.',
                 'Legacy outcome reports do not independently recheck shared plant traces.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(matched_groups=len(matched),excluded_groups=excluded,
                          diagnostic_rollouts=len(details),rates=rates),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
