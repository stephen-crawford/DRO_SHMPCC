#!/usr/bin/env python3
"""Matched reviewer experiments using the unchanged analysis-matrix rollout path."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import run_analysis_matrix as analysis
from run_comparison_matrix import obstacle_prefix_matches

ROOT = analysis.ROOT
VARIANTS = {
    'sh_mpcc': ('sh_mpcc', {}),
    'sh_mpcc_dro': ('sh_mpcc_dro', {}),
    'sh_mpcc_resample': ('sh_mpcc_dro_fallback', {'nominal_resampling_baseline': True, 'dro_enabled': False}),
    'sh_mpcc_dro_fallback': ('sh_mpcc_dro_fallback', {}),
    'hybrid_zero_one': ('sh_mpcc_dro_fallback', {'ground_cost': 'zero_one'}),
    'hybrid_fixed_radius': ('sh_mpcc_dro_fallback', {'use_calibrated_radius': False}),
}


def load_settings(path):
    settings = json.loads(path.read_text())
    variants = settings['variants']
    if not variants or len(set(variants)) != len(variants) or set(variants)-set(VARIANTS):
        raise ValueError('invalid or duplicate variants')
    base = dict(settings, solver_styles=['sh_mpcc'])
    # Reuse the existing validation without writing a temporary settings file.
    for key, allowed in [('obstacle_counts', range(1,5)),('class_counts',range(1,5)),
                         ('mode_counts',range(1,7)),('environments',analysis.ENVIRONMENTS)]:
        values=base[key]
        if not values or len(set(values))!=len(values) or any(v not in allowed for v in values):
            raise ValueError(f'invalid {key}')
    if (not base['seeds'] or len(set(base['seeds']))!=len(base['seeds']) or
        any(type(s) is not int or not 0<=s<2**32 for s in base['seeds']) or
        type(base['repeats']) is not int or base['repeats']<1):
        raise ValueError('invalid seeds/repeats')
    catalog=base['mode_catalog']
    valid={'constant_velocity','turn_left','turn_right','accelerating','decelerating','stop'}
    if len(catalog)<max(base['mode_counts']) or len(set(catalog))!=len(catalog) or set(catalog)-valid:
        raise ValueError('invalid mode catalog')
    radius=settings['fixed_radius']
    if not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<=0:
        raise ValueError('fixed_radius must be finite and positive')
    reserved={'nominal_resampling_baseline','ground_cost','fixed_rho','use_calibrated_radius',
              'shift_psi','shift_boost','boosted_mode'}
    if reserved.intersection(settings.get('overrides',{})):
        raise ValueError('overrides conflict with ablation/profile axes')
    profiles=settings['profiles']
    if not profiles or any(not name.replace('_','').isalnum() for name in profiles):
        raise ValueError('invalid profile names')
    for profile in profiles.values():
        if set(profile)-{'shift_psi','shift_boost','boosted_mode'}:
            raise ValueError('unsupported profile key')
        for k in ['shift_psi','shift_boost']:
            v=profile.get(k,0.)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError('shift probabilities must lie in [0,1]')
        mode=profile.get('boosted_mode',-1)
        if type(mode) is not int or mode < -1 or mode >= min(settings['mode_counts']):
            raise ValueError('boosted mode must belong to every support')
    return base


def configurations(settings):
    for base in analysis.configurations(settings):
        for profile, variant in itertools.product(settings['profiles'], settings['variants']):
            pair=f"{profile}_{base['environment']}_o{base['obstacles']}_c{base['classes']}_m{base['modes_per_class']}"
            yield dict(base, case=f'{variant}_{pair}', pair=pair, profile=profile, solver_style=variant)


def config_text(case, settings):
    base_style, overrides = VARIANTS[case['solver_style']]
    text=analysis.config_text(dict(case,solver_style=base_style), settings)
    values=dict(overrides, method_name=case['solver_style'], scenario_tag=case['case'])
    values.update(settings['profiles'][case['profile']])
    if case['solver_style']=='hybrid_fixed_radius':
        values['fixed_rho']=settings['fixed_radius']
    lines=[]
    for line in text.splitlines():
        key=line.split(':',1)[0]
        lines.append(f'{key}: {json.dumps(values.pop(key))}' if key in values else line)
    if values:
        raise ValueError(f'unknown config keys: {values}')
    return '\n'.join(lines)+'\n'


def check_pairing(group, output):
    if any(r['status']!='OK' for r in group):
        return 'INCOMPLETE', 'at least one controller execution is not OK'
    try:
        base=group[0]
        for other in group[1:]:
            for repeat, (a,b) in enumerate(zip(base['repeats'],other['repeats'])):
                for field in ['plant_seed','controller_seed','initial_placement','backend','solver_identity']:
                    if a['metrics'][field]!=b['metrics'][field]:
                        raise ValueError(f'paired {field} mismatch')
                bundles=[analysis.trial_root(output,r,r['seed'])/f'repeat_{repeat}' for r in [base,other]]
                if not obstacle_prefix_matches(*bundles):
                    raise ValueError('plant trajectories differ over shared trace prefix')
        return 'OK',''
    except (OSError,KeyError,ValueError) as error:
        return 'ERROR',str(error)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'configs/reviewer_tests/settings.json')
    p.add_argument('--runner',type=Path,default=ROOT/'build-base/experiment_runner')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair',help='Exact profile/environment/count combination; always selects every variant')
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--timeout',type=float,default=1800.)
    args=p.parse_args(argv)
    args.runner,args.output=args.runner.resolve(),args.output.resolve()
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('timeout must be finite and positive')
    settings=load_settings(args.settings)
    cases=list(configurations(settings))
    selected=[c for c in cases if args.pair is None or c['pair']==args.pair]
    if not selected:
        p.error('no selected configurations')
    texts={c['case']:config_text(c,settings) for c in cases}
    manifest=dict(schema=1,suite='reviewer_matrix',settings=settings,cases=cases,
        configs={k:analysis.digest(v.encode()) for k,v in texts.items()},
        runner=str(args.runner),runner_sha256=analysis.digest(args.runner.read_bytes()),
        sources={str(f.relative_to(ROOT)):analysis.digest(f.read_bytes()) for f in
                 [Path(__file__).resolve(),Path(analysis.__file__).resolve(),
                  ROOT/'tests/run_comparison_matrix.py']},jobs=1,timeout_seconds=args.timeout)
    identity=analysis.digest(json.dumps(manifest,sort_keys=True).encode())
    manifest['identity']=identity
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'matrix.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        p.error('different manifest; use a new output directory')
    analysis.dump_json(path,manifest)
    (args.output/'configs').mkdir(exist_ok=True)
    for c in selected:
        (args.output/'configs'/(c['case']+'.yaml')).write_text(texts[c['case']])
    print(f'{len(selected)} configurations × {len(settings["seeds"])} seeds; serial execution',flush=True)
    if args.generate_only:
        return 0
    results=[]
    for case in selected:
        for seed in settings['seeds']:
            r=analysis.run_trial(case,seed,args,settings,identity)
            results.append(r)
            print(f'{case["case"]} seed={seed}: {r["status"]} {r.get("error", "")}',flush=True)
    pairs=[]
    for pair,seed in itertools.product(sorted({c['pair'] for c in selected}),settings['seeds']):
        group=[r for r in results if r['pair']==pair and r['seed']==seed]
        status,error=check_pairing(group,args.output)
        pairs.append(dict(pair=pair,seed=seed,status=status,error=error))
        if status=='ERROR':
            # Keep broken pairing out of all matched analyses, even if runs completed.
            for r in group:
                r['status']='ERROR';r['error']='pairing: '+error
    analysis.dump_json(args.output/'results.json',results)
    analysis.write_csv(args.output/'pairing.csv',pairs)
    analysis.write_csv(args.output/'summary.csv',analysis.aggregate(results,selected,len(settings['seeds'])))
    analysis.write_csv(args.output/'summary_per_seed.csv',analysis.per_seed_summary(results))
    print(f'Reviewer matrix complete: {len(results)} trials; {sum(r["status"]!="OK" for r in results)} errors',flush=True)
    return int(any(r['status']!='OK' for r in results))


if __name__=='__main__':
    sys.exit(main())
