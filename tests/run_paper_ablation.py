#!/usr/bin/env python3
"""Three-arm paper ablation using the canonical matrix runner; no controller changes."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import run_analysis_matrix as analysis
import run_reviewer_matrix as reviewer

ROOT=analysis.ROOT
ARMS={'baseline':'sh_mpcc','reweight_only':'sh_mpcc_dro_reweight_only','full_transfer':'sh_mpcc_dro'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_value(text,key,value):
    lines=text.splitlines(); found=False
    for i,line in enumerate(lines):
        if line.split(':',1)[0]==key:
            lines[i]=f'{key}: {json.dumps(value)}';found=True
    if not found: raise ValueError(f'missing config key {key}')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runner',type=Path,default=ROOT/'build-base/experiment_runner')
    parser.add_argument('--practicality',type=Path,default=ROOT/'results/acc-paper-practicality-final-20261002')
    parser.add_argument('--scaling',type=Path,default=ROOT/'results/acc-paper-scaling-final-20261002')
    parser.add_argument('--timeout',type=float,default=600.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--generate-only',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.runner=args.runner.resolve()
    args.practicality=args.practicality.resolve();args.scaling=args.scaling.resolve()
    inputs=[args.practicality,args.scaling]
    manifests={str(p):json.loads((p/'matrix.json').read_text()) for p in inputs}
    runner_hash=digest(args.runner)
    for manifest in manifests.values():
        if manifest['runner_sha256']!=runner_hash:
            raise ValueError('reuse requires the exact same executable hash')
    # Freeze before seeing new outcomes. Five paired seeds on both S-curve cases,
    # retaining the original straight/intersection practicality seed sets.
    groups=[dict(environment=e,obstacles=1,classes=1,modes_per_class=2,
                 seeds=list(range(77,82)) if e=='s_curve' else [77,78,79])
            for e in ['straight','s_curve','four_way_intersection']]
    groups.append(dict(environment='s_curve',obstacles=2,classes=2,modes_per_class=3,seeds=list(range(77,82))))
    cases=[];texts={};sources={};s0s={}
    for group in groups:
        source=args.practicality if group['obstacles']==1 else args.scaling
        settings=manifests[str(source)]['settings']
        pair=f"standard_{group['environment']}_o{group['obstacles']}_c{group['classes']}_m{group['modes_per_class']}"
        baseline_name='sh_mpcc_'+pair
        candidates=list((source/baseline_name).glob('seed_*/repeat_0/decisions.csv'))
        observed={int(r['scenario_count']) for p in candidates for r in csv.DictReader(p.open())}
        if len(observed)!=1: raise ValueError('baseline must have one observed sample budget')
        s0=observed.pop();s0s[pair]=s0
        for arm,style in ARMS.items():
            case={k:v for k,v in group.items() if k!='seeds'}
            case.update(case=style+'_'+pair,pair=pair,profile='standard',solver_style=style,arm=arm,seeds=group['seeds'])
            base=dict(case,solver_style='sh_mpcc' if arm=='baseline' else 'sh_mpcc_dro')
            text=reviewer.config_text(base,settings)
            if arm=='reweight_only':
                text=replace_value(text,'method_name',style)
                text=replace_value(text,'automatically_compute_sample_size',False)
                text=replace_value(text,'num_scenarios',s0)
            texts[case['case']]=text;cases.append(case);sources[case['case']]=str(source)
    manifest=dict(schema=1,suite='paper_three_arm_ablation',groups=groups,cases=cases,
        nominal_sample_counts=s0s,runner=str(args.runner),runner_sha256=runner_hash,
        configs={k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},
        sources={str(p):digest(p/'matrix.json') for p in inputs},
        generator_sha256=digest(Path(__file__)),timeout_seconds=args.timeout,jobs=1,
        interpretation='Reweight-only is an intentional fixed-budget ablation without a true-law transfer certificate.')
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();manifest['identity']=identity
    args.output.mkdir(parents=True,exist_ok=True);manifest_path=args.output/'matrix.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('manifest changed; choose a new output directory')
    analysis.dump_json(manifest_path,manifest);(args.output/'configs').mkdir(exist_ok=True)
    for name,text in texts.items():(args.output/'configs'/(name+'.yaml')).write_text(text)
    print(f"Frozen design: {sum(len(c['seeds']) for c in cases)} runs, three arms, S0={s0s}",flush=True)
    if args.generate_only:return 0
    results=[]
    for case in cases:
        for seed in case['seeds']:
            source=Path(sources[case['case']]);original=source/case['case']/f'seed_{seed}'
            target=analysis.trial_root(args.output,case,seed)
            if args.resume and (target/'result.json').exists():
                result=analysis.run_trial(case,seed,args,dict(repeats=1),identity)
            elif case['arm']!='reweight_only' and (original/'result.json').exists():
                old=json.loads((original/'result.json').read_text())
                if old['status']!='OK':raise ValueError('cannot silently reuse unsuccessful trial')
                if (source/'configs'/(case['case']+'.yaml')).read_text()!=texts[case['case']]:
                    raise ValueError('reuse config mismatch')
                shutil.copytree(original,target)
                result=dict(old,**{k:v for k,v in case.items() if k not in old})
                result.update(identity=identity,reused_from=str(original),source_identity=old['identity'])
                analysis.dump_json(target/'result.json',result)
            else:
                result=analysis.run_trial(case,seed,args,dict(repeats=1),identity)
            results.append(result)
            analysis.dump_json(args.output/'results.json',results)
            print(f"{case['arm']} {case['pair']} seed={seed}: {result['status']} "
                  f"{'reused' if result.get('reused_from') else 'new'} {result.get('error','')}",flush=True)
    pairs=[]
    for group in groups:
        pair=f"standard_{group['environment']}_o{group['obstacles']}_c{group['classes']}_m{group['modes_per_class']}"
        for seed in group['seeds']:
            matching=[r for r in results if r['pair']==pair and r['seed']==seed]
            status,error=reviewer.check_pairing(matching,args.output)
            pairs.append(dict(pair=pair,seed=seed,status=status,error=error))
    analysis.write_csv(args.output/'pairing.csv',pairs)
    print(f"Done: {len(results)} runs; {sum(bool(r.get('reused_from')) for r in results)} reused; "
          f"{sum(r['status']!='OK' for r in results)} errors",flush=True)
    return int(any(r['status']!='OK' for r in results) or any(r['status']!='OK' for r in pairs))

if __name__=='__main__':raise SystemExit(main())
