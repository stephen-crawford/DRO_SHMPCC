#!/usr/bin/env python3
"""Predeclared recoverability pilots followed by independent persistent-count confirmation."""
import argparse
import json
from pathlib import Path
import sys
from run_causal_link import ROOT,invoke,digest
sys.path.insert(0,str(ROOT/'tools'))
from causal_link_analysis import read,write,wilson


def run(root,case,seed,nd):
    folder=root/case['name']/f'seed_{seed}'/f'nd_{nd}'
    invoke(ROOT/'build-base/persistent_coverage_experiment',folder,
        ['late_switch',20,10,seed,case['switch'],120,5,'sh_mpcc',case['x'],5,nd])
    summary=read(folder/'summary.csv')[0]
    decisions=read(folder/'decisions.csv')
    start=max(0,case['switch']-2)
    observed=[r for r in decisions if start<=int(r['step'])<start+5]
    modes=[r for r in read(folder/'mode_mechanism.csv') if r['obstacle_id']=='0' and r['mode']=='across' and r['attempt']=='0']
    assert all(int(r['sampled_count'])==nd for r in modes if start<=int(r['step'])<start+5)
    return dict(summary,case=case['name'],seed=seed,target_nd=nd,window_start=start,window_decisions=len(observed),
        window_controls_executed=sum(int(r['success']) for r in observed),x=case['x'],switch=case['switch'],source=str(folder))


def summarize(rows):
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in rows:grouped[r['case'],r['target_nd']].append(r)
    result=[]
    for (case,nd),rs in grouped.items():
        for metric in ['collision','refusal','path_completed','window_completed']:
            def event(r):
                if metric=='refusal':return r['termination']=='no_admissible_control'
                if metric=='window_completed':return r['window_controls_executed']==5
                return int(r[metric])
            hits=sum(event(r) for r in rs);lo,hi=wilson(hits,len(rs))
            result.append(dict(case=case,target_nd=nd,metric=metric,seeds=len(rs),events=hits,rate=hits/len(rs),ci95_low=lo,ci95_high=hi))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/persistent-coverage/intervention-updated-belief');args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    cases=[dict(name=f'x{x}_switch{sw}',x=x,switch=sw) for x in [4,5,6] for sw in [0,2]]
    record=dict(cases=cases,pilot_seeds=list(range(2001,2021)),pilot_counts=[0,3],confirmation_seeds=list(range(2101,2201)),confirmation_counts=[0,1,2,3,5,10],
        critical_window='max(0,switch-2)..start+4; paired nominal after window',
        selection='first predeclared case: covered Nd3 safe path completion >=50%, uncovered Nd0 collision/refusal >=20%, all statuses OK',
        post_window_belief='same lagged observation updates as the production controller; no frozen-belief carryover',
        noise='zero-diffusion original stress modes; common/danger banks share seed and each non-replaced slot is identical',
        guarantees='not_requested; finite-window intervention is non-IID',
        sources={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'tools/persistent_coverage_experiment.cpp',ROOT/'src/mpc_controller.cpp',ROOT/'include/mpc_controller.hpp']})
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=record:raise RuntimeError('changed manifest; use new directory')
    manifest.write_text(json.dumps(record,indent=2))
    pilots=[]
    for case in cases:
        for seed in record['pilot_seeds']:
            for nd in record['pilot_counts']:pilots.append(run(root,case,seed,nd))
    write(root/'pilot_outcomes.csv',pilots);write(root/'pilot_summary.csv',summarize(pilots))
    selected=None
    for case in cases:
        low=[r for r in pilots if r['case']==case['name'] and r['target_nd']==0]
        covered=[r for r in pilots if r['case']==case['name'] and r['target_nd']==3]
        if (all(r['status']=='OK' for r in low+covered) and
            sum(int(r['path_completed']) for r in covered)>=10 and
            sum(int(r['collision']) or r['termination']=='no_admissible_control' for r in low)>=4):
            selected=case;break
    # Always measure the count-response curve at a fixed pre-switch state, even if
    # no geometry qualifies as recoverable. Label that failure rather than claiming success.
    confirmation=selected or next(c for c in cases if c['name']=='x5_switch2')
    selection=dict(recoverability_gate_passed=selected is not None,selected=confirmation,
                   selection_basis='predeclared first qualifier' if selected else 'predeclared unqualified reference')
    path=root/'selection_before_confirmation.json'
    if path.exists() and json.loads(path.read_text())!=selection:raise RuntimeError('changed selection')
    path.write_text(json.dumps(selection,indent=2))
    rows=[run(root,confirmation,seed,nd) for seed in record['confirmation_seeds'] for nd in record['confirmation_counts']]
    write(root/'confirmation_outcomes.csv',rows);write(root/'confirmation_summary.csv',summarize(rows))
    pairs=[]
    for seed in record['confirmation_seeds']:
        rr={r['target_nd']:r for r in rows if r['seed']==seed}
        for nd in record['confirmation_counts'][1:]:
            a,b=rr[0],rr[nd]
            pairs.append(dict(seed=seed,target_nd=nd,collision_0=a['collision'],collision_n=b['collision'],
                refusal_0=int(a['termination']=='no_admissible_control'),refusal_n=int(b['termination']=='no_admissible_control'),
                completed_0=a['path_completed'],completed_n=b['path_completed'],window_controls_0=a['window_controls_executed'],window_controls_n=b['window_controls_executed']))
    write(root/'paired_outcomes.csv',pairs)

if __name__=='__main__':main()
