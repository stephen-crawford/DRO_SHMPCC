#!/usr/bin/env python3
"""Evaluate saved accepted plans of tools/rare_mode_experiment.cpp (deterministic modes)."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from analyze_posterior_certificate import ROOT, write_outputs


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',type=Path,default=ROOT/'build-certificate/certificate_numeric_backend')
    args=p.parse_args()
    if args.output.exists():p.error('use a new output directory')
    weights=read(args.fixture/'weights.csv');scene=read(args.fixture/'scene.csv')
    if {r['mode'] for r in weights}!={'away','straight','cut_in'}:p.error('expected the three-mode rare_mode_experiment fixture')
    means={(r['actor'],int(r['k'])):[float(r['x']),float(r['y'])] for r in scene if r['actor']!='ego_reference'}
    key=lambda r:tuple(r[k] for k in ['budget_design','base_S','seed','method'])
    plans=defaultdict(list)
    for r in read(args.fixture/'plans.csv'):plans[key(r)].append(r)
    cycles=read(args.fixture/'controller_trials.csv');args.output.mkdir(parents=True)
    summaries=[]
    for cycle in cycles:
        identity={k:cycle[k] for k in ['budget_design','base_S','seed','method']}
        if cycle['status']!='OK' or cycle['final_success']!='1':
            summaries.append(dict(identity,evaluation_status='no_accepted_plan',direct_cp_bound=None));continue
        plan=sorted(plans[key(cycle)],key=lambda r:int(r['k']))
        if [int(r['k']) for r in plan]!=list(range(21)):raise ValueError('incomplete saved horizon-20 plan')
        steps=[]
        for row in plan[1:]:
            x,y,theta=map(float,[row['x'],row['y'],row['theta']]);k=int(row['k'])
            steps.append(dict(k=k,disc_centers=[[x+d*math.cos(theta),y+d*math.sin(theta)] for d in [-1.,0.,1.]],
                gaussians={m:dict(mean=means[m,k],covariance=[[0.,0.],[0.,0.]]) for m in ['away','straight','cut_in']}))
        snapshot=dict(obstacles=1,collision_radius=.95,steps=steps,
            modes=[dict(mode=w['mode'],count=int(w['count']),p_hat=float(w['nominal_probability'])) for w in weights],
            scenario_theorem_eligible=False,
            source='Saved accepted plan from deterministic rare-mode fixture: three discs, length 2; radius .5+.35+.1. q is not fully logged per attempt; no Q-transfer claim.',
            history_assumption='Synthetic fixed counts, not evidence of iid observations. CP-based confidence remains conditional on that model.')
        folder=args.output/'_'.join(key(cycle))
        result=write_outputs(snapshot,args.backend.resolve(),folder)
        summaries.append(dict(identity,evaluation_status='accepted_plan_diagnostic',direct_cp_bound=result['direct_cp_bound'],
            target_met=result['direct_bound_meets_target'],**{'b_'+m['mode']:m['b'] for m in result['mode_bounds']}))
    with (args.output/'posterior_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in summaries for k in r)));w.writeheader();w.writerows(summaries)
    (args.output/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixture.glob('*.csv')},indent=2)+'\n')
    print(json.dumps(dict(trials=len(summaries),accepted=sum(r['evaluation_status']=='accepted_plan_diagnostic' for r in summaries),
        conditional_direct_target_met=sum(bool(r.get('target_met')) for r in summaries))))

if __name__=='__main__':main()
