#!/usr/bin/env python3
"""Targeted causal-link follow-up; never expands or resumes the broad stress grid."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from causal_link_analysis import read, write, undercoverage, select_separation, qualify_braking, frozen_report, paired_summary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(binary, folder, args):
    folder=Path(folder)
    command=[str(binary.resolve()),str(folder.resolve()),*map(str,args)]
    signature=dict(command=command,binary_sha256=digest(binary),config_sha256=digest(ROOT/'configs/default.yaml'))
    checkpoint=Path(str(folder)+'.done.json')
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if old['signature']!=signature or old['exit_code']!=0:
            raise RuntimeError(f'Cannot resume changed or failed run: {folder}')
        return
    if folder.exists():
        raise RuntimeError(f'Incomplete run retained for inspection: {folder}')
    folder.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(folder)+'.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    checkpoint.write_text(json.dumps(dict(signature=signature,exit_code=result.returncode),indent=2))
    if result.returncode:
        raise RuntimeError(f'Experiment failed; inspect {Path(str(folder)+".log")}')
    print('completed',folder,flush=True)


def rollout(root,name,seed,arm,switch=5,family='late_switch',x=5,y=2,drift=1,speed=2,g=1000,steps=80):
    folder=root/name/f'seed_{seed}'/arm
    invoke(ROOT/'build-base/causal_link_rollout',folder,
           [family,50,20,seed,switch,steps,5 if family=='late_switch' else 0,arm,x,y,drift,speed,g,2])
    rows=read(folder/'summary.csv')
    if len(rows)!=1:raise RuntimeError(f'missing summary: {folder}')
    return dict(rows[0],case=name,seed=seed,arm=arm,switch=switch,source=str(folder))


def prescreen(root):
    candidates=[]
    for x,y,dy,g in itertools.product([2.,3.,4.],[1.2,2.],[-.04,-.12,-.18],[100,300,1000]):
        name=f'x{x}_y{y}_dy{dy}_g{g}'
        folder=root/'prescreen'/name
        invoke(ROOT/'build-base/causal_link_probe',folder,[x,y,dy,g,20,1,0,2,'probe'])
        row=next(r for r in read(folder/'weights.csv') if r['mode']=='across')
        transport=next(r for r in read(folder/'transport.csv') if r['from_mode']=='continue' and r['to_mode']=='across')
        candidates.append(dict(row,candidate=name,transport_from_common=transport['cost'],
            **{f'delta_u{k}':undercoverage(float(row['p']),k,20)-undercoverage(float(row['q']),k,20) for k in [1,2,3]}))
    # Benign control participates in the near-zero selection on the same criterion.
    folder=root/'prescreen/benign'
    invoke(ROOT/'build-base/causal_link_probe',folder,[3,2,.18,1000,20,1,0,2,'probe'])
    row=next(r for r in read(folder/'weights.csv') if r['mode']=='across')
    candidates.append(dict(row,candidate='benign',transport_from_common='',
        **{f'delta_u{k}':undercoverage(float(row['p']),k,20)-undercoverage(float(row['q']),k,20) for k in [1,2,3]}))
    write(root/'prescreen.csv',candidates)
    selection=select_separation(candidates)
    path=root/'selection_before_outcomes.json'
    record=dict(criterion='rank delta_U2 among positive-risk candidates; high=max, medium=median, near_zero=min absolute over all candidates',
                raw_count_proportions=dict(across=.05,away=.10,continue_mode=.85),
                caveat='Dirichlet-smoothed p varies slightly with g; actual p is logged. No radius override.',
                prescreen_sha256=digest(root/'prescreen.csv'),selected=selection)
    if path.exists() and json.loads(path.read_text())!=record:raise RuntimeError('frozen selection changed')
    path.write_text(json.dumps(record,indent=2))
    return selection


def frozen(root,reps,quota):
    selection=json.loads((root/'selection_before_outcomes.json').read_text())['selected']
    all_checks=[]
    for row in selection:
        folder=root/'frozen'/row['stratum']
        invoke(ROOT/'build-base/causal_link_probe',folder,[row['x'],row['y'],row['across_dy'],row['g'],20,reps,quota,2,'conditional'])
        all_checks.extend(dict(r,stratum=row['stratum']) for r in frozen_report(folder))
    write(root/'frozen_coverage_checks.csv',all_checks)
    if not all(int(r['coverage_check_pass']) for r in all_checks):raise RuntimeError('Frozen coverage validation failed; inspect CSV')


def commitment(root,seeds):
    # A no-switch NOMINAL reference defines each seed's tc before either adverse arm runs.
    schedule=[]
    for seed in seeds:
        r=rollout(root,'commitment_reference',seed,'sh_mpcc',switch=79)
        tc=int(r['commitment_step'])
        schedule.append(dict(seed=seed,reference_status=r['status'],reference_tc=tc,
            eligible=int(r['status']=='OK' and tc>=5),exclusion='' if r['status']=='OK' and tc>=5 else 'reference_did_not_reach_commitment'))
    write(root/'commitment_schedule_before_outcomes.csv',schedule)
    rows=[]
    for r in schedule:
        if not r['eligible']:continue
        for offset in [-5,-3,-1,0]:
            for arm in ['sh_mpcc','sh_mpcc_dro']:
                rows.append(dict(rollout(root,f'commitment_offset_{offset}',r['seed'],arm,
                    switch=r['reference_tc']+offset),offset=offset,reference_tc=r['reference_tc']))
    write(root/'commitment_outcomes.csv',rows)
    gates=[]
    eligible=sum(r['eligible'] for r in schedule)
    for offset in [-5,-3,-1,0]:
        for arm in ['sh_mpcc','sh_mpcc_dro']:
            rr=[r for r in rows if r['offset']==offset and r['arm']==arm]
            reached=sum(int(r['switch_reached']) for r in rr if r['status']=='OK')
            gates.append(dict(offset=offset,arm=arm,planned_seeds=len(seeds),eligible_seeds=eligible,
                observed_seeds=len(rr),reached=reached,reach_rate_eligible=reached/eligible if eligible else '',
                reach_rate_all_planned=reached/len(seeds),passes95_all_planned=int(reached/len(seeds)>=.95)))
    write(root/'commitment_attrition.csv',gates)


def braking(root,pilot_seeds,confirmation_seeds):
    rows=[];designs=[]
    for x,speed,drift in itertools.product([6,8],[1,2],[.5,1.]):
        name=f'braking_x{x}_v{speed}_drift{drift}'
        rr=[rollout(root,name,seed,arm,switch=3,family='rare_braking',x=x,y=1.8,drift=drift,speed=speed)
            for seed in pilot_seeds for arm in ['sh_mpcc','sh_mpcc_dro']]
        rows.extend(rr)
        designs.append(dict(case=name,x=x,speed=speed,drift=drift,qualifies=qualify_braking(rr,len(pilot_seeds))))
    write(root/'braking_pilot_outcomes.csv',rows)
    write(root/'braking_designs.csv',designs)
    selected=next((r for r in designs if r['qualifies']),None)
    path=root/'braking_selection_before_confirmation.json'
    record=dict(criterion='first predeclared design with both arms reaching switch >=95% and nominal collision-or-refusal 20–50%',selected=selected)
    if path.exists() and json.loads(path.read_text())!=record:raise RuntimeError('braking selection changed')
    path.write_text(json.dumps(record,indent=2))
    confirmations=[]
    if selected:
        for seed in confirmation_seeds:
            for arm in ['sh_mpcc','sh_mpcc_dro']:
                confirmations.append(rollout(root,selected['case']+'_confirmation',seed,arm,switch=3,
                    family='rare_braking',x=selected['x'],y=1.8,drift=selected['drift'],speed=selected['speed']))
    write(root/'braking_confirmation.csv',confirmations)


def scaling(root,seeds):
    rows=[]
    for obstacles in [2,3,4]:
        folder=root/'scaling'/f'obstacles_{obstacles}'
        invoke(ROOT/'build-base/causal_link_probe',folder,[3,2,-.12,1000,20,seeds,0,obstacles,'scaling'])
        qp={};current=None
        with Path(str(folder)+'.log').open() as log:
            for line in log:
                m=re.search(r'\[CAUSAL SOLVE\] law=(\w+) seed=(\d+) step=(\d+)',line)
                if m:current=(m[1],m[2]);qp.setdefault(current,[])
                m=re.search(r'\[SQP\] before_solve.* constraints=(\d+)',line)
                if m and current:qp[current].append(int(m[1]))
        rows.extend(dict(r,qp_constraints_max=max(qp.get((r['law'],r['seed']),[]),default=''),
            qp_calls_logged=len(qp.get((r['law'],r['seed']),[]))) for r in read(folder/'scaling.csv'))
    write(root/'scaling_measurements.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/causal-link-followup/extension-v2')
    p.add_argument('--stage',choices=['all','prescreen','frozen','commitment','braking','scaling'],default='all')
    p.add_argument('--seeds',type=int,default=50)
    p.add_argument('--replicates',type=int,default=5000)
    p.add_argument('--quota',type=int,default=30)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.seeds<1 or args.replicates<1 or args.quota<1:p.error('positive trial counts required')
    manifest=dict(seeds=args.seeds,replicates=args.replicates,quota=args.quota,
        sources={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'tools/causal_link_analysis.py',ROOT/'tools/causal_link_probe.cpp',ROOT/'tools/causal_link_rollout.cpp']},
        guarantees='not_requested; conditional selection is an experimental intervention, not IID certification',
        outcome_definition='collision at discrete plant steps; refusal ends a rollout without an executable control',
        conditional_definition='Nd controlled by outcome-blind seed rejection; later draws use the same frozen law, not a maintained Nd quota',
        timing_definition='trajectory generation and raw constraint construction are separate fixed-reference microbenchmarks; other timings are actual controller diagnostics')
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Manifest changed: use a new output directory')
    path.write_text(json.dumps(manifest,indent=2))
    stages=[args.stage] if args.stage!='all' else ['prescreen','frozen','commitment','braking','scaling']
    for stage in stages:
        if stage=='prescreen':prescreen(root)
        elif stage=='frozen':frozen(root,args.replicates,args.quota)
        elif stage=='commitment':commitment(root,list(range(187,187+args.seeds)))
        elif stage=='braking':braking(root,list(range(287,307)),list(range(387,387+args.seeds)))
        elif stage=='scaling':scaling(root,args.seeds)
    paired_summary(root.parent)

if __name__=='__main__':main()
