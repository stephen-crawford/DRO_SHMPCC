#!/usr/bin/env python3
"""Descriptive risk allocation and matched-event reports from scrubbed CSVs."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import dangerous_event_report as danger

from analyze_comparison_results import ARMS, KEYS, key, number, read, write

ARMS = dict(ARMS, sh_mpcc_dro_stratified='dro_stratified')

DECISION = list(KEYS) + ['solver_style', 'step', 'obstacle_id']


def stream(path):
    if path.exists():
        with path.open(newline='') as file:
            yield from csv.DictReader(file)


def unique(rows, fields):
    result = {}
    for row in rows:
        identity = tuple(row.get(f, '') for f in fields)
        if identity in result:
            raise ValueError(f'duplicate evidence: {identity}')
        result[identity] = row
    return result


def allocation(rows):
    """Use exact logged zeros; undefined risk ratios and tied maxima stay blank."""
    modes = unique(rows, ['mode'])
    p, q, r, n = [], [], [], []
    for row in rows:
        values = [number(row.get(f)) for f in
                  ('nominal_probability', 'sampling_probability', 'risk_score', 'sampled_count')]
        if any(v is None or v < 0 for v in values):
            raise ValueError('missing, negative, or nonfinite mode evidence')
        a, b, c, d = values
        if a > 1 or b > 1 or not d.is_integer():
            raise ValueError('invalid probability or sample count')
        p.append(a); q.append(b); r.append(c); n.append(d)
    if not math.isclose(sum(p), 1, abs_tol=1e-9) or not math.isclose(sum(q), 1, abs_tol=1e-9):
        raise ValueError('incomplete or unnormalized mode law')
    budget = sum(n)
    if budget <= 0 or any(number(row.get('scenario_count')) != budget for row in rows):
        raise ValueError('mode counts differ from actual scenario count')
    truth = {row.get('true_mode') for row in rows}
    if len(truth) != 1 or (next(iter(truth)),) not in modes:
        raise ValueError('inconsistent or absent realized mode')
    true = next(iter(truth))
    ti = next(i for i, row in enumerate(rows) if row['mode'] == true)
    top = [i for i, risk in enumerate(r) if risk == max(r)]
    best = top[0] if len(top) == 1 and max(r) > 0 else None
    empirical = sum(count * risk for count, risk in zip(n, r)) / budget
    target = sum(prob * risk for prob, risk in zip(q, r))
    return dict(scenario_count=budget, mode_count=len(rows), support_q=sum(v > 0 for v in q),
        zero_mass_modes=sum(v == 0 for v in q), any_zero_mass=int(any(v == 0 for v in q)),
        vertex=int(sum(v > 0 for v in q) == 1), max_q=max(q),
        risk_coverage=sum(risk for count, risk in zip(n, r) if count > 0)/sum(r) if sum(r) else '',
        empirical_risk=empirical, target_risk=target, risk_error=abs(empirical-target),
        nominal_risk=sum(prob*risk for prob, risk in zip(p, r)),
        top_mode=rows[best]['mode'] if best is not None else '',
        top_count=n[best] if best is not None else '',
        top_fraction=n[best]/budget if best is not None else '',
        top_delta_q=q[best]-p[best] if best is not None else '',
        true_mode=true, true_p=p[ti], true_q=q[ti], true_risk=r[ti], true_count=n[ti],
        true_represented=int(n[ti] > 0), true_zero_mass=int(q[ti] == 0),
        missed_true_zero_mass=int(n[ti] == 0 and q[ti] == 0),
        true_not_map=int(p[ti] < max(p)))


def summarize(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    result = []
    for identity, group in sorted(groups.items()):
        item = dict(zip(fields, identity), observations=len(group))
        for metric in metrics:
            values = [v for row in group if (v := number(row.get(metric))) is not None]
            item[metric+'_n'] = len(values)
            item[metric+'_mean'] = statistics.mean(values) if values else ''
        result.append(item)
    return result


def analyze(source, output, repeat='0', risk_threshold=None, event_risk_reference='wdro'):
    output.mkdir(parents=True, exist_ok=True)
    runs = unique([r for r in read(source/'run_summary.csv') if r.get('repeat') == repeat],
                  list(KEYS)+['variant'])
    good = {k: r for k, r in runs.items() if r.get('trial_status') == 'OK' and r.get('log_complete') == '1'}
    keep = DECISION+['mode','nominal_probability','sampling_probability','risk_score','sampled_count',
                     'scenario_count','true_mode','source_artifact','reference_clearance','reference_risk']
    raw = [{f:r.get(f,'') for f in keep} for r in stream(source/'artifact_mode_mechanism.csv') if r.get('repeat') == repeat
           and r.get('attempt') == '0' and key(r)+(ARMS.get(r.get('solver_style')), ) in good]
    unique(raw, DECISION+['mode'])
    groups = defaultdict(list)
    for row in raw:
        groups[tuple(row[f] for f in DECISION)].append(row)
    enriched=[]
    annotations={}
    for identity,rows in groups.items():
        previous=groups.get(identity[:-2]+(str(int(identity[-2])-1),identity[-1]),[])
        annotations[identity]=danger.labels(rows,previous)
        enriched.extend(dict(row,**annotations[identity]) for row in rows)
    write(output/'mode_mechanism_enriched.csv',enriched,keep+['map_mode','map_modes','map_tie','switch_event'])
    diagnostics, excluded, scatter = [], [], []
    valid = {}
    for identity, rows in sorted(groups.items()):
        if identity[len(KEYS)] != 'sh_mpcc_dro':
            continue
        common = dict(zip(DECISION, identity))
        try:
            metrics = allocation(rows)
        except ValueError as error:
            excluded.append(dict(common, reason=str(error)))
            continue
        valid[identity] = metrics
        diagnostics.append(dict(common, **metrics))
        for row in rows:
            scatter.append(dict(common, mode=row['mode'], risk_score=float(row['risk_score']),
                delta_q=float(row['sampling_probability'])-float(row['nominal_probability']),
                source_artifact=row.get('source_artifact', '')))
    write(output/'risk_per_decision.csv', diagnostics, DECISION)
    write(output/'risk_mass_scatter.csv', scatter, DECISION+['mode','risk_score','delta_q'])
    write(output/'excluded_evidence.csv', excluded, DECISION+['reason'])
    metrics = ['risk_coverage','empirical_risk','target_risk','risk_error','top_count','top_fraction',
               'top_delta_q','support_q','max_q','any_zero_mass','vertex','true_represented',
               'missed_true_zero_mass','true_p']
    write(output/'risk_per_seed.csv', summarize(diagnostics, list(KEYS), metrics), list(KEYS))
    histogram = defaultdict(int)
    for row in diagnostics:
        histogram[key(row)+(row['support_q'],)] += 1
    write(output/'support_distribution.csv', [dict(zip(KEYS, k[:-1]), support_q=k[-1], decisions=v)
          for k, v in sorted(histogram.items())], list(KEYS)+['support_q','decisions'])
    # Pairing is admitted only with the runner's plant/belief check, never just a shared seed.
    checked = set()
    for row in read(source/'report_all_comparisons.csv'):
        if row.get('status') == 'OK':
            checked.add((row.get('test_root',''), row.get('matrix_identity',''), row['pair'], row['seed'],
                         frozenset((row['controller_a'], row['controller_b']))))
    def paired(identity, a, b):
        return identity[:4]+(frozenset((a,b)),) in checked
    decisions = unique([r for r in stream(source/'artifact_decisions.csv') if r.get('repeat') == repeat],
                       list(KEYS)+['solver_style','step'])
    matched, matched_risk, events, danger_pairs = [], [], [], []
    for identity, ref in valid.items():
        runkey, _, step, obstacle = identity[:len(KEYS)], *identity[len(KEYS):]
        rr = {r['mode']: r for r in groups[identity]}
        for style in ARMS:
            target = runkey+(style,step,obstacle)
            if target not in groups or not paired(runkey, 'sh_mpcc_dro', style):
                continue
            other = {r['mode']: r for r in groups[target]}
            if set(other) != set(rr) or any(other[m]['true_mode'] != rr[m]['true_mode'] or
                    number(other[m]['nominal_probability']) != number(rr[m]['nominal_probability']) for m in rr):
                continue
            counts = {m: number(row.get('sampled_count')) for m, row in other.items()}
            if any(n is None or n < 0 or not n.is_integer() for n in counts.values()):
                continue
            budget = sum(counts.values())
            if budget <= 0 or any(number(r.get('scenario_count')) != budget for r in other.values()):
                continue
            common = dict(zip(KEYS, runkey), solver_style=style, step=step, obstacle_id=obstacle)
            reference_risks = {m:float(r['risk_score']) for m,r in rr.items()}
            total_risk = sum(reference_risks.values())
            empirical = sum(counts[m]*reference_risks[m] for m in rr)/budget
            matched_risk.append(dict(common, scenario_count=budget,
                risk_reference='wdro_pre_reweight_plan',
                risk_coverage=sum(reference_risks[m] for m in rr if counts[m]>0)/total_risk if total_risk else '',
                empirical_risk=empirical, wdro_target_risk=ref['target_risk'],
                error_to_wdro_target=abs(empirical-ref['target_risk']),
                top_count=counts[ref['top_mode']] if ref['top_mode'] else '',
                top_fraction=counts[ref['top_mode']]/budget if ref['top_mode'] else ''))
            for mode, row in rr.items():
                matched.append(dict(common, mode=mode, risk_score=float(row['risk_score']),
                    delta_q=float(row['sampling_probability'])-float(row['nominal_probability']),
                    wdro_count=float(row['sampled_count']), comparison_count=counts[mode],
                    delta_count=float(row['sampled_count'])-counts[mode],
                    wdro_budget=ref['scenario_count'], comparison_budget=budget,
                    unique_positive_top=int(mode == ref['top_mode'])))
            if style != 'sh_mpcc':
                continue
            dangerous = (ref['true_risk'] >= risk_threshold if risk_threshold is not None
                         else ref['true_mode'] == ref['top_mode'])
            if event_risk_reference == 'fixed':
                fixed=number(rr[ref['true_mode']].get('reference_risk'))
                nominal_fixed=number(other[ref['true_mode']].get('reference_risk'))
                if fixed is None or fixed != nominal_fixed:
                    continue
                dangerous = fixed >= risk_threshold if risk_threshold is not None else fixed > 0
            if budget == ref['scenario_count']:
                danger_pairs.append(danger.pair_row(common,ref,counts,rr,decisions,runkey,annotations[identity],dangerous))
                danger_pairs[-1]['event_risk_reference']=event_risk_reference
            previous = groups.get(runkey+('sh_mpcc_dro',str(int(step)-1),obstacle), [])
            switched = int(previous[0]['true_mode'] != ref['true_mode']) if previous else ''
            for event, active in [('all_matched', True), ('non_map', ref['true_not_map']),
                    ('mode_switch', switched == 1), ('dangerous_realized', dangerous),
                    ('dangerous_non_map', dangerous and ref['true_not_map']),
                    ('dangerous_switch', dangerous and switched == 1)]:
                if not active:
                    continue
                for arm, count in [('sh_mpcc', counts[ref['true_mode']]), ('sh_mpcc_dro', ref['true_count'])]:
                    decision = decisions.get(runkey+(arm,step), {})
                    success = number(decision.get('success'))
                    events.append(dict(common, solver_style=arm, event=event, true_count=count,
                        true_represented=int(count > 0), inadmissible=1-success if success in (0,1) else '',
                        risk_reference='wdro_pre_reweight_plan', true_risk=ref['true_risk']))
    danger.export(output,danger_pairs)
    write(output/'matched_mode_allocation.csv', matched, DECISION+['mode'])
    for row in matched:
        row['shift_group'] = 'increased' if row['delta_q']>0 else 'decreased' if row['delta_q']<0 else 'unchanged'
        row['more_samples'] = int(row['delta_count']>0)
        row['fewer_samples'] = int(row['delta_count']<0)
        row['tied_samples'] = int(row['delta_count']==0)
    effect_fields=list(KEYS)+['solver_style','shift_group','unique_positive_top']
    write(output/'allocation_effects_per_seed.csv',summarize(matched,effect_fields,
        ['delta_q','delta_count','more_samples','fewer_samples','tied_samples']),effect_fields)
    write(output/'matched_risk_coverage.csv', matched_risk, DECISION)
    write(output/'matched_events.csv', events, DECISION+['event'])
    event_fields = list(KEYS)+['solver_style','event']
    write(output/'events_per_seed.csv', summarize(events,event_fields,['true_count','true_represented']),event_fields)
    decision_fields=event_fields+['step']
    event_decisions={tuple(row[f] for f in decision_fields): row for row in events}
    write(output/'event_decisions_per_seed.csv',summarize(list(event_decisions.values()),event_fields,
        ['inadmissible']),event_fields)
    # An event selects a rollout once. Completion/clearance are rollout outcomes, not step outcomes.
    selected = {tuple(r[k] for k in event_fields) for r in events}
    outcomes = []
    for identity in sorted(selected):
        run = good[identity[:len(KEYS)]+(ARMS[identity[-2]],)]
        outcomes.append(dict(zip(event_fields,identity), **{f: run.get(f,'') for f in
            ('collision','path_completed','min_actual_clearance','mean_controller_solve_ms')}))
    write(output/'event_rollout_outcomes.csv', outcomes, event_fields)
    # Keep recovery comparisons explicit and use each seed once, never pool different cells.
    comparisons = []
    for a,b in [('sh_mpcc','sh_mpcc_dro'),('sh_mpcc_dro','sh_mpcc_dro_fallback'),
                ('sh_mpcc','sh_mpcc_resample'),('sh_mpcc_extra','sh_mpcc_dro'),
                ('sh_mpcc_resample','sh_mpcc_dro_fallback'),('sh_mpcc_dro','sh_mpcc_dro_stratified')]:
        for identity in sorted({k[:-1] for k in good}):
            left, right = good.get(identity+(ARMS[a],)), good.get(identity+(ARMS[b],))
            if not left or not right or not paired(identity,a,b):
                continue
            for metric in ('path_completed','collision'):
                x,y = number(left.get(metric)),number(right.get(metric))
                if x not in (0,1) or y not in (0,1):
                    continue
                comparisons.append(dict(zip(KEYS,identity), controller_a=a,controller_b=b,metric=metric,
                                        n00=int(x==0 and y==0),n01=int(x==0 and y==1),
                                        n10=int(x==1 and y==0),n11=int(x==1 and y==1)))
    strata = [k for k in KEYS if k != 'seed']+['controller_a','controller_b','metric']
    cg = defaultdict(list)
    for row in comparisons:
        cg[tuple(row[k] for k in strata)].append(row)
    summary = []
    for identity, rows in sorted(cg.items()):
        counts = {k:sum(r[k] for r in rows) for k in ('n00','n01','n10','n11')}
        n = counts['n01']+counts['n10']
        p = min(1.,2*sum(math.comb(n,k) for k in range(min(counts['n01'],counts['n10'])+1))/2**n) if n else 1.
        summary.append(dict(zip(strata,identity),paired_seeds=len(rows),**counts,exact_mcnemar_p=p))
    write(output/'ablation_pairs.csv',summary,strata)
    (output/'notes.json').write_text(json.dumps(dict(repeat=repeat,risk_threshold=risk_threshold,event_risk_reference=event_risk_reference,
        population='Attempt zero, unique complete OK runs. Per-arm diagnostics may have different populations. Matched outputs require runner pairing checks and identical nominal beliefs/realized modes.',
        risk='Risk scores precede reweighting but depend on the WDRO ego plan. Shared labels are exploratory, NOT controller-independent physical danger labels. Default danger is unique positive maximum; tied/all-zero maxima are unlabelled. With event_risk_reference=fixed, paired event labels use identical logged fixed-reference physical risk (>0 unless a threshold is supplied), independent of controller plans.',
        events='true_mode is the logged realized mode, not an inferred next mode. Switch requires consecutive recorded steps. Inadmissibility is decision success=0, not a safety certificate. Representation counts obstacle-decisions; inadmissibility deduplicates decisions; rollout outcomes deduplicate seeds/events.',
        inference='Per-cell seed McNemar tests are exploratory, unadjusted. No pooling across repeated seeds/cells. Steps are not independent trials. No certified sample-reduction claim.',
        counts=dict(risk_decisions=len(diagnostics),excluded=len(excluded),matched_modes=len(matched),events=len(events))),indent=2)+'\n')
    print(f'{len(diagnostics)} risk decisions; {len(excluded)} excluded; {len(matched)} matched mode rows; {len(events)} event rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeat',default='0')
    parser.add_argument('--plots',action='store_true',help='write per-condition PNG/PDF dangerous-event figures')
    parser.add_argument('--risk-threshold',type=float)
    parser.add_argument('--event-risk-reference',choices=['wdro','fixed'],default='wdro')
    args = parser.parse_args()
    if args.risk_threshold is not None and (not math.isfinite(args.risk_threshold) or args.risk_threshold < 0):
        parser.error('risk threshold must be finite and nonnegative')
    if not (args.source/'run_summary.csv').exists() or args.source.resolve() == args.out.resolve():
        parser.error('provide scrubbed run_summary.csv and a separate output directory')
    analyze(args.source,args.out,args.repeat,args.risk_threshold,args.event_risk_reference)
    if args.plots:danger.plot(args.out/'dangerous_event_pairs.csv',args.out/'figures')


if __name__ == '__main__':
    main()
