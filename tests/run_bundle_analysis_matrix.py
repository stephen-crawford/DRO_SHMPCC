#!/usr/bin/env python3
"""Full analysis-matrix axes, paired baseline/bundle control and compute costs."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import sys
import run_analysis_matrix as analysis

ROOT=analysis.ROOT


def configurations(settings):
    for base in analysis.configurations(settings):
        for variant in settings['bundle_variants']:
            yield dict(base,case=base['case']+'__'+variant['name'],baseline_case=base['case'],
                variant=variant['name'],bundle_amplification=variant['amplification'],bundle_extra_draws=variant['extras'])


def load_settings(path):
    s=analysis.load_settings(path)
    variants=s['bundle_variants']
    if not variants or variants[0]!={'name':'baseline','amplification':0,'extras':0}:
        raise ValueError('first bundle variant must be baseline with amplification/extras zero')
    if len({v['name'] for v in variants})!=len(variants):raise ValueError('duplicate variant name')
    for v in variants:
        if not v['name'].replace('_','').isalnum() or type(v['extras'])!=int or v['extras']<0:
            raise ValueError('invalid variant')
        if v['name']!='baseline' and not 1<v['amplification']<=1000:raise ValueError('amplification must exceed one')
    overrides=s.get('overrides',{})
    if any(k in overrides for k in ['bundle_amplification','bundle_extra_draws']):raise ValueError('bundle knobs belong in bundle_variants')
    if overrides.get('markov_jump_system',False) or overrides.get('safe_horizon_enabled',True) is False:
        raise ValueError('comparison requires held-mode Safe Horizon prediction')
    return s


def config_text(case,settings):
    s=dict(settings,overrides=dict(settings.get('overrides',{}),
        bundle_amplification=case['bundle_amplification'],bundle_extra_draws=case['bundle_extra_draws'],
        artifact_capture_attempt_diagnostics=True))
    return analysis.config_text(case,s)


def metrics(bundle):
    decisions=analysis.rows(bundle/'decisions.csv'); costs=analysis.rows(bundle/'bundle_costs.csv')
    record=analysis.rows(bundle/'rollout.csv')[0]
    if [r['step'] for r in decisions]!=[r['step'] for r in costs]:raise ValueError('incomplete cost records')
    timings=[float(r['solve_ms']) for r in decisions]
    attempts=analysis.rows(bundle/'attempts.csv')
    return dict(progress=float(record['total_progress']),contouring_error=float(record['mean_contouring_err']),
        velocity_error=float(record['mean_velocity_err']),collision=int(record['collision']),
        completed=int(record['completed_path']),termination=record['termination_reason'],executed_steps=int(record['total_steps']),
        control_effort=sum(float(r['applied_control_effort']) for r in decisions),
        solve_mean_ms=statistics.mean(timings),solve_p95_ms=sorted(timings)[max(0,__import__('math').ceil(.95*len(timings))-1)],
        solve_total_ms=sum(timings),solve_qp_total_ms=sum(float(r['qp_ms']) for r in costs),
        solve_constraints_total_ms=sum(float(r['constraint_ms']) for r in costs),
        sample_groups_total=sum(int(r['scenario_count']) for r in attempts),
        raw_trajectories_total=sum(int(r['raw_trajectories']) for r in attempts),
        qp_calls=sum(int(r['qp_calls']) for r in attempts),
        mean_final_facets=statistics.mean(int(r['retained_facets']) for r in costs),
        failure_budget_max=max(float(r['combined_failure_budget']) for r in costs))


def run(case,seed,args,settings,identity):
    result=analysis.run_trial(case,seed,args,settings,identity)
    if result['status']=='OK':
        try:
            for i,observation in enumerate(result['repeats']):
                bundle=analysis.trial_root(args.output,case,seed)/f'repeat_{i}'
                if analysis.repeat_signature(bundle,observation['metrics'])!=observation['signature']:
                    raise ValueError('saved artifact signature changed')
                observation['bundle_metrics']=metrics(bundle)
        except (OSError,KeyError,ValueError) as error:
            result.update(status='ERROR',error=str(error))
        analysis.dump_json(analysis.trial_root(args.output,case,seed)/'result.json',result)
    return result


def reports(output,results,cases,seeds):
    flat=[]
    for r in results:
        for repeat,obs in enumerate(r['repeats']):
            if 'bundle_metrics' in obs:
                flat.append(dict(case=r['case'],baseline_case=r['baseline_case'],variant=r['variant'],seed=r['seed'],
                    repeat=repeat,status=r['status'],**obs['bundle_metrics']))
    analysis.write_csv(output/'bundle_rollouts.csv',flat)
    indexed={(r['baseline_case'],r['variant'],r['seed']):r for r in results}
    pairs=[]
    for case in cases:
        if case['variant']=='baseline':continue
        for seed in seeds:
            a=indexed.get((case['baseline_case'],'baseline',seed));b=indexed.get((case['baseline_case'],case['variant'],seed))
            row=dict(baseline_case=case['baseline_case'],variant=case['variant'],seed=seed,obstacles=case['obstacles'],
                classes=case['classes'],environment=case['environment'],modes_per_class=case['modes_per_class'],solver_style=case['solver_style'],
                baseline_status=a['status'] if a else 'PENDING',bundle_status=b['status'] if b else 'PENDING',paired=False)
            if a and b and a['status']==b['status']=='OK':
                ar=a['repeats'][0];br=b['repeats'][0]
                if ar['metrics']['initial_placement']!=br['metrics']['initial_placement'] or ar['metrics']['plant_seed']!=br['metrics']['plant_seed']:
                    raise ValueError('paired plant identity differs')
                aa=analysis.rows(analysis.trial_root(output,a,seed)/'repeat_0/decisions.csv')
                bb=analysis.rows(analysis.trial_root(output,b,seed)/'repeat_0/decisions.csv')
                at=analysis.rows(analysis.trial_root(output,a,seed)/'repeat_0/trace.csv')
                bt=analysis.rows(analysis.trial_root(output,b,seed)/'repeat_0/trace.csv')
                ao={(r['step'],r['obstacle_id']):r for r in at if r['actor']=='obstacle'}
                bo={(r['step'],r['obstacle_id']):r for r in bt if r['actor']=='obstacle'}
                for k in ao.keys()&bo.keys():
                    if any(ao[k][f]!=bo[k][f] for f in ['x','y','vx','vy']):raise ValueError('paired obstacle path differs')
                row['paired']=True
                for field in ar['bundle_metrics']:
                    av=ar['bundle_metrics'][field];bv=br['bundle_metrics'][field]
                    row['baseline_'+field]=av;row['bundle_'+field]=bv
                    if isinstance(av,(int,float)):row['delta_'+field]=bv-av
                horizon=min(len(aa),len(bb));row['common_decisions']=horizon
                row['delta_common_control_effort']=sum(float(bb[i]['applied_control_effort'])-float(aa[i]['applied_control_effort']) for i in range(horizon))
                row['compute_mean_ratio']=br['bundle_metrics']['solve_mean_ms']/ar['bundle_metrics']['solve_mean_ms']
                row['equal_confidence']=ar['bundle_metrics']['failure_budget_max']==br['bundle_metrics']['failure_budget_max']
            pairs.append(row)
    analysis.write_csv(output/'bundle_pairs.csv',pairs)
    summary=[]
    for base,variant in sorted({(r['baseline_case'],r['variant']) for r in pairs}):
        rr=[r for r in pairs if (r['baseline_case'],r['variant'])==(base,variant)]
        good=[r for r in rr if r['paired']]
        summary.append(dict(baseline_case=base,variant=variant,expected_pairs=len(rr),measured_pairs=len(good),
            missing_or_failed_pairs=len(rr)-len(good),**{f:statistics.mean(r[f] for r in good) if good else '' for f in
            ['compute_mean_ratio','delta_progress','delta_control_effort','delta_common_control_effort','delta_collision','delta_completed','delta_solve_p95_ms','delta_raw_trajectories_total','delta_mean_final_facets']}))
    analysis.write_csv(output/'bundle_summary.csv',summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'configs/bundle_analysis_matrix/full.json')
    p.add_argument('--runner',type=Path,default=ROOT/'build-bundles/experiment_runner')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=1,help='use 1 for comparable solver timing')
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--base-case',help='exact original analysis-matrix case; runs baseline and all bundle variants')
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='reuse matching completed trials after checking saved numerical signatures')
    p.add_argument('--report-only',action='store_true',help='rebuild paired reports from saved results without running simulations')
    a=p.parse_args(argv);a.output=a.output.resolve();a.runner=a.runner.resolve()
    if a.jobs<1 or a.timeout<=0:p.error('jobs/timeout must be positive')
    settings=load_settings(a.settings);all_cases=list(configurations(settings))
    cases=[c for c in all_cases if a.base_case is None or c['baseline_case']==a.base_case]
    if not cases:p.error('unknown base case')
    texts={c['case']:config_text(c,settings) for c in all_cases}
    manifest=dict(schema=1,suite='bundle_analysis_matrix',settings=settings,cases=all_cases,runner=str(a.runner),
        runner_sha256=analysis.digest(a.runner.read_bytes()),configs={n:analysis.digest(t.encode()) for n,t in texts.items()},
        script_sha256=analysis.digest(Path(__file__).read_bytes()),analysis_script_sha256=analysis.digest(Path(analysis.__file__).read_bytes()),
        jobs=a.jobs,timeout_seconds=a.timeout,timing_comparable=a.jobs==1)
    identity=analysis.digest(json.dumps(manifest,sort_keys=True).encode());manifest['identity']=identity
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'matrix.json'
    if path.exists() and json.loads(path.read_text())!=manifest:p.error('manifest changed; use a new output directory')
    analysis.dump_json(path,manifest);(a.output/'configs').mkdir(exist_ok=True)
    for name,text in texts.items():(a.output/'configs'/(name+'.yaml')).write_text(text)
    print(f'{len(all_cases)} total configurations; {len(cases)} selected; {len(cases)*len(settings["seeds"])*settings["repeats"]} selected executions',flush=True)
    if a.generate_only:return 0
    if not a.report_only:
        # Serial is the default. Reverse variant order on alternating seeds to
        # reduce systematic order bias while keeping controller/plant RNG fixed.
        tasks=[(c,seed) for si,seed in enumerate(settings['seeds']) for c in (cases if si%2==0 else list(reversed(cases)))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(run,c,seed,a,settings,identity) for c,seed in tasks]
            for future in concurrent.futures.as_completed(futures):
                r=future.result();print(f'{r["case"]} seed={r["seed"]}: {r["status"]} {r.get("error","")}',flush=True)
    saved=[]
    for case in all_cases:
        for seed in settings['seeds']:
            path=analysis.trial_root(a.output,case,seed)/'result.json'
            if path.exists():
                r=json.loads(path.read_text())
                if r['identity']!=identity:raise ValueError('stale trial identity')
                saved.append(r)
    analysis.dump_json(a.output/'results.json',saved)
    analysis.write_csv(a.output/'summary.csv',analysis.aggregate(saved,all_cases,len(settings['seeds'])))
    reports(a.output,saved,all_cases,settings['seeds'])
    return int(any(r['status']!='OK' for r in saved))

if __name__=='__main__':sys.exit(main())
