#!/usr/bin/env python3
"""Prescribed adverse-switch suite, retaining every result including failures."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_comparison_results import ARMS, read, write
import analyze_risk_sampling as analysis


def cases(settings):
    for family in settings['families']:
        for count,budget,switch in itertools.product(family['danger_counts'],settings['scenario_budgets'],settings['switch_steps']):
            yield dict(family=family['name'],danger_count=count,budget=budget,switch=switch,
                       lag=family['history_lag'],name=f"{family['name']}_p{count}_s{budget}_switch{switch}")


def normalize(root,settings,selected,identity):
    runs=[];modes=[];decisions=[];comparisons=[];schedule=[];trajectories=[]
    for case in selected:
        for seed in settings['seeds']:
            evidence={}
            for arm in settings['arms']:
                folder=root/case['name']/f'seed_{seed}'/arm
                common=dict(test_suite='adverse_switch_stress',test_root=str(root),matrix_identity=identity,
                    pair_case=case['name'],seed=str(seed),repeat='0',solver_style=arm,
                    stress_family=case['family'],switch_step=case['switch'],history_lag=case['lag'],
                    scenario_budget=case['budget'],certification_status='not_requested')
                schedule.append(dict(common,log_exists=int((folder/'run.log').exists())))
                status=read(folder/'summary.csv')
                if len(status)!=1:continue
                row=status[0]
                runs.append(dict(common,variant=ARMS[arm],trial_status=row['status'],log_complete=int(row['status']=='OK'),**{k:v for k,v in row.items() if k!='status'}))
                evidence[arm]=(row,read(folder/'plant.csv'),read(folder/'mode_mechanism.csv'))
                for name,target in [('mode_mechanism',modes),('decisions',decisions)]:
                    target.extend(dict(common,**r,source_artifact=str(folder/(name+'.csv'))) for r in read(folder/(name+'.csv')))
                trajectories.extend(dict(common,**r) for r in read(folder/'plant.csv'))
            for a,b in itertools.combinations(settings['arms'],2):
                status='PENDING'
                if a in evidence and b in evidence:
                    aa,bb=evidence[a],evidence[b]
                    status='ERROR'
                    if aa[0]['status']==bb[0]['status']=='OK':
                        n=min(len(aa[1]),len(bb[1]))
                        ma={(r['step'],r['obstacle_id'],r['mode']):r['nominal_probability'] for r in aa[2] if r['attempt']=='0'}
                        mb={(r['step'],r['obstacle_id'],r['mode']):r['nominal_probability'] for r in bb[2] if r['attempt']=='0'}
                        shared=ma.keys()&mb.keys()
                        if n and aa[1][:n]==bb[1][:n] and shared and all(ma[k]==mb[k] for k in shared):status='OK'
                comparisons.append(dict(test_root=str(root),matrix_identity=identity,pair=case['name'],seed=str(seed),controller_a=a,controller_b=b,status=status))
    out=root/'csv';out.mkdir(exist_ok=True)
    for name,rows in [('run_summary',runs),('artifact_mode_mechanism',modes),('artifact_decisions',decisions),
                      ('report_all_comparisons',comparisons),('expected_runs',schedule),('plant',trajectories)]:
        write(out/(name+'.csv'),rows,[])
    analysis.analyze(out,root/'report',event_risk_reference='fixed')
    # Candidate dangerous mode is explicit, including BEFORE the prescribed switch.
    by_step={(r['pair_case'],r['seed'],r['solver_style'],r['step']):r for r in decisions}
    timeseries=[]
    for row in modes:
        if row['attempt']!='0' or row['obstacle_id']!='0' or row['mode']!='across':continue
        decision=by_step.get((row['pair_case'],row['seed'],row['solver_style'],row['step']),{})
        timeseries.append(dict(row,time_seconds=.1*int(row['step']),switch_time_seconds=.1*int(row['switch_step']),
            **{f:decision.get(f,'') for f in ['ego_speed','acceleration','omega','actual_clearance','success','collision']}))
    write(root/'report/dangerous_mode_timeseries.csv',timeseries,[])
    write(root/'report/stress_outcomes.csv',runs,[])
    from collections import defaultdict
    from statistics import mean
    grouped=defaultdict(list)
    for row in runs:grouped[row['pair_case'],row['solver_style']].append(row)
    summaries=[]
    for (case,arm),rows in sorted(grouped.items()):
        good=[r for r in rows if r['trial_status']=='OK']
        record=dict(pair_case=case,solver_style=arm,expected_seeds=len(settings['seeds']),
                    observed_seeds=len(rows),ok_seeds=len(good),error_seeds=len(rows)-len(good))
        for metric in ['collision','path_completed','switch_reached','min_actual_clearance','mean_controller_solve_ms']:
            values=[float(r[metric]) for r in good if r.get(metric,'')!='']
            record[metric+'_mean']=mean(values) if values else ''
        summaries.append(record)
    write(root/'report/stress_summary.csv',summaries,[])
    return sum(r['trial_status']!='OK' for r in runs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'configs/risk_stress/settings.json')
    p.add_argument('--runner',type=Path,default=ROOT/'build-base/risk_stress_experiment')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',help='exact condition name, e.g. rare_turn_p50_s20_switch15')
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--plots',action='store_true')
    p.add_argument('--timeout',type=float,default=300)
    args=p.parse_args();settings=json.loads(args.settings.read_text())
    selected=[c for c in cases(settings) if not args.case or c['name']==args.case]
    if not selected:p.error('no matching stress condition')
    if args.timeout<=0 or settings['steps']<=max(settings['switch_steps']):p.error('invalid timing settings')
    args.output=args.output.resolve();args.runner=args.runner.resolve()
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [args.runner,Path(__file__),ROOT/'tools/risk_stress_experiment.cpp',ROOT/'configs/default.yaml']}
    manifest=dict(settings=settings,hashes=hashes,timeout=args.timeout,policy='prescribed_adverse_switch; not a population collision-rate estimate')
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:p.error('different manifest: use a new output directory')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    write(args.output/'schedule.csv',[dict(c,seed=seed,arm=arm) for c in cases(settings)
          for seed in settings['seeds'] for arm in settings['arms']],[])
    print(f'{len(selected)} conditions; {len(selected)*len(settings["seeds"])*len(settings["arms"])} executions',flush=True)
    if args.generate_only:return 0
    for case in selected:
        for seed in settings['seeds']:
            for arm in settings['arms']:
                folder=args.output/case['name']/f'seed_{seed}'/arm
                if folder.exists():
                    if args.resume and read(folder/'summary.csv'):continue
                    p.error(f'existing incomplete run {folder}; use a new output directory')
                folder.parent.mkdir(parents=True,exist_ok=True)
                # Runner requires a new output directory; console log lives beside it until completion.
                log=folder.with_suffix('.log')
                cmd=[str(args.runner),str(folder),case['family'],str(case['danger_count']),str(case['budget']),
                     str(seed),str(case['switch']),str(settings['steps']),str(case['lag']),arm]
                with log.open('w') as file:
                    try:result=subprocess.run(cmd,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        folder.mkdir(exist_ok=True);write(folder/'summary.csv',[dict(status='ERROR',termination='timeout')],[])
                        result=None
                if not (folder/'summary.csv').exists() or not read(folder/'summary.csv'):
                    folder.mkdir(exist_ok=True);write(folder/'summary.csv',[dict(status='ERROR',termination='execution_error')],[])
                log.replace(folder/'run.log')
                print(case['name'],seed,arm,'exit',result.returncode if result else 'timeout',flush=True)
    errors=normalize(args.output,settings,list(cases(settings)),identity)
    if args.plots:
        analysis.danger.plot(args.output/'report/dangerous_event_pairs.csv',args.output/'report/figures')
        analysis.danger.plot_stress(args.output/'report/dangerous_mode_timeseries.csv',args.output/'report/timelines')
    return int(errors>0)

if __name__=='__main__':raise SystemExit(main())
