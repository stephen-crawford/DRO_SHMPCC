#!/usr/bin/env python3
"""Complete count strata and live timing; refine only the near-qualifying braking pilot."""
import argparse
import itertools
import json
from pathlib import Path
import re
from run_causal_link import ROOT, digest, invoke, rollout, frozen, scaling
from causal_link_analysis import read, write, qualify_braking


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,default=ROOT/'results/causal-link-followup/extension-v2')
    p.add_argument('--output',type=Path,default=ROOT/'results/causal-link-followup/refinement')
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    record=dict(prior=str(args.prior.resolve()),selection_sha256=digest(args.prior/'selection_before_outcomes.json'),
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'tools/causal_link_probe.cpp',ROOT/'tools/causal_link_analysis.py',ROOT/'tests/run_causal_link.py',ROOT/'src/mpc_controller.cpp',ROOT/'include/types.hpp']},
        count_selection='first 30 seeds per Nd stratum, all Nd=0..20; 5000 independent sets per law; no selection by outcomes',
        braking_refinement='8 initial designs failed. x6,v1,drift1 had 15% refusal; prospectively test x=5.5/6,v=1.1/1.2,drift1 on NEW pilot seeds 307..326.',
        confirmation='first qualifying refined pilot; 50 NEW seeds 387..436; no confirmation if no qualifying pilot',
        scaling='50 seeds per arm and obstacle count; live generation timing excludes DRO computation; no overlapping simulations')
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=record:raise RuntimeError('Refinement provenance changed; use new output directory')
    manifest.write_text(json.dumps(record,indent=2))
    selected=root/'selection_before_outcomes.json'
    if selected.exists() and digest(selected)!=record['selection_sha256']:raise RuntimeError('selection mismatch')
    selected.write_bytes((args.prior/'selection_before_outcomes.json').read_bytes())
    frozen(root,5000,30)
    rows=[];designs=[]
    for x,speed in itertools.product([5.5,6.],[1.1,1.2]):
        name=f'braking_refined_x{x}_v{speed}'
        rr=[rollout(root,name,seed,arm,switch=3,family='rare_braking',x=x,y=1.8,drift=1,speed=speed)
            for seed in range(307,327) for arm in ['sh_mpcc','sh_mpcc_dro']]
        rows.extend(rr);designs.append(dict(case=name,x=x,speed=speed,qualifies=qualify_braking(rr,20)))
    write(root/'braking_pilot_outcomes.csv',rows);write(root/'braking_designs.csv',designs)
    chosen=next((r for r in designs if r['qualifies']),None)
    choice=dict(criterion='first predeclared design: both reach >=95%, nominal collision-or-refusal 20–50%',selected=chosen)
    path=root/'braking_selection_before_confirmation.json'
    if path.exists() and json.loads(path.read_text())!=choice:raise RuntimeError('confirmation selection changed')
    path.write_text(json.dumps(choice,indent=2))
    confirmations=[]
    if chosen:
        for seed in range(387,437):
            for arm in ['sh_mpcc','sh_mpcc_dro']:
                confirmations.append(rollout(root,chosen['case']+'_confirmation',seed,arm,switch=3,family='rare_braking',x=chosen['x'],y=1.8,drift=1,speed=chosen['speed']))
    write(root/'braking_confirmation.csv',confirmations)
    scaling(root,50)
    groups=[]
    for obstacles in [2,3,4]:
        log=root/'scaling'/f'obstacles_{obstacles}.log';current=None
        with log.open() as stream:
            for line in stream:
                m=re.search(r'\[CAUSAL SOLVE\] law=(\w+) seed=(\d+) step=(\d+)',line)
                if m:current=dict(law=m[1],seed=m[2],step=m[3],obstacles=obstacles)
                m=re.search(r'\[FREE POLY\] k=(\d+) disc=(\d+) raw=(\d+) facets=(\d+)',line)
                if m and current:groups.append(dict(current,k=m[1],disc=m[2],raw_constraints=m[3],retained_facets=m[4]))
    write(root/'scaling_constraint_groups.csv',groups)

if __name__=='__main__':main()
