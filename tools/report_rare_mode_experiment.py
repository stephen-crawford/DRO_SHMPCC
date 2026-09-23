#!/usr/bin/env python3
"""Report fixed-budget sampling and separately conditional controller fallback."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import statistics


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(path,data):
    with path.open('w') as f:
        if not data:return
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('use a new report directory')
    coverage=rows(args.input/'coverage_trials.csv')
    groups=defaultdict(list)
    for r in coverage:groups[int(r['budget']),r['scheme']].append(r)
    samples=[]
    for (budget,scheme),data in sorted(groups.items()):
        samples.append(dict(budget=budget,scheme=scheme,replicates=len(data),
            inclusion_rate=statistics.mean(int(r['represented']) for r in data),
            expected_inclusion=float(data[0]['expected_inclusion']),
            mean_dangerous_draws=statistics.mean(int(r['cut_in_count']) for r in data)))
    trials=rows(args.input/'controller_trials.csv')
    grouped=defaultdict(dict)
    for r in trials:
        key=r['budget_design'],r['base_S'],r['seed']
        if r['method'] in grouped[key]:raise ValueError('duplicate controller trial')
        grouped[key][r['method']]=r
    good=[g for g in grouped.values() if len(g)==4 and all(r['status']=='OK' for r in g.values())]
    policy=[]
    for design,s,method in sorted({(r['budget_design'],int(r['base_S']),r['method']) for r in trials}):
        data=[g[method] for g in good if g[method]['budget_design']==design and int(g[method]['base_S'])==s]
        if not data:continue
        mean=lambda name:statistics.mean(float(r[name]) for r in data)
        policy.append(dict(budget_design=design,base_S=s,method=method,matched_trials=len(data),
            first_attempt_inadmissibility=1-mean('first_success'),final_inadmissibility=1-mean('final_success'),
            fallback_rate=mean('fallback'),first_dangerous_inclusion=statistics.mean(int(r['first_cut_in_count'])>0 for r in data),
            any_attempt_dangerous_inclusion=mean('any_cut_in'),mean_total_draws=mean('total_draws'),
            mean_outer_attempts=mean('outer_attempts'),mean_failed_attempts=mean('failed_attempts'),
            mean_qp_calls=mean('qp_calls'),mean_cycle_ms=mean('cycle_ms')))
    args.output.mkdir(parents=True)
    save(args.output/'coverage_summary.csv',samples)
    save(args.output/'policy_summary.csv',policy)
    notes=dict(matched_groups=len(good),excluded_groups=len(grouped)-len(good),
        sampling='Fixed nominal history, fixed geometric reference; split batches are unconditional and spend exactly the same total budget.',
        controller='Real conditional fallback; counts and probabilities refer to actual draws. No iid union formula is asserted for conditional retries.',
        holdout='plan_risk.csv evaluates accepted open-loop plans exactly over three deterministic modes under stated synthetic laws. Not closed-loop rollout risk.',
        safety='Manual low scenario budgets do not inherit the automatic sample-count guarantee; sample_count_sufficient is exported.',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.input.glob('*.csv'))})
    (args.output/'notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    print(json.dumps(dict(coverage_rows=len(samples),policy_rows=len(policy),matched_groups=len(good),excluded_groups=notes['excluded_groups'])))


if __name__=='__main__':main()
