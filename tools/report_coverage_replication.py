#!/usr/bin/env python3
"""Seed-level replication and prospective coverage-gain validation; selected-file bundle."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import zipfile
from collections import defaultdict
from statistics import mean
from causal_link_analysis import read,write,undercoverage,wilson
from analyze_persistent_coverage import measurements


def window_metrics(rows,window,k):
    rr=[r for r in rows if int(r['step']) in window and str(r['attempt'])=='0']
    steps=[int(r['step']) for r in rr]
    if len(steps)!=len(set(steps)):raise ValueError('duplicate window decision')
    complete=set(steps)==set(window)
    return dict(observed_decisions=len(rr),expected_decisions=len(window),window_complete=int(complete),
        min_N_d=min(int(r['N_d']) for r in rr) if complete else '',
        fraction_N_ge_k=sum(int(r['N_d'])>=k for r in rr)/len(window) if complete else '',
        all_steps_N_ge_k=int(all(int(r['N_d'])>=k for r in rr)) if complete else '',
        plugin_C_p=math.prod(1-undercoverage(float(r['p_d']),k,int(r['scenario_count'])) for r in rr) if complete else '',
        plugin_C_sampling=math.prod(1-undercoverage(float(r['q_d']),k,int(r['scenario_count'])) for r in rr) if complete else '')


def mcnemar(a,b):
    discordant=a+b
    return min(1.,2*sum(math.comb(discordant,i) for i in range(min(a,b)+1))/2**discordant) if discordant else 1.


def summarize(cell,paired,initial):
    record=dict(cell=cell,paired_seeds=len(paired),initial_nominal_U1=mean(float(r['U_p_k1']) for r in initial))
    for metric in ['collision','refusal','completed']:
        a_only=sum(r[f'nominal_{metric}'] and not r[f'wdro_{metric}'] for r in paired)
        b_only=sum(r[f'wdro_{metric}'] and not r[f'nominal_{metric}'] for r in paired)
        record.update({f'nominal_only_{metric}':a_only,f'wdro_only_{metric}':b_only,f'{metric}_exact_mcnemar_p':mcnemar(a_only,b_only)})
        for arm in ['nominal','wdro']:
            events=sum(r[f'{arm}_{metric}'] for r in paired);lo,hi=wilson(events,len(paired))
            record.update({f'{arm}_{metric}_count':events,f'{arm}_{metric}_ci95_low':lo,f'{arm}_{metric}_ci95_high':hi})
    record['collision_benefit']=(record['nominal_collision_count']-record['wdro_collision_count'])/len(paired)
    return record


def collect(root,cell,window):
    out=root/cell/'report';runs=read(out/'stress_outcomes.csv')
    modes=measurements(read(out/'dangerous_mode_timeseries.csv'))
    if len(runs)!=200 or any(r['trial_status']!='OK' for r in runs):raise ValueError(f'incomplete run set: {cell}')
    statuses=read(root/cell/'csv/report_all_comparisons.csv')
    if len(statuses)!=100 or any(r['status']!='OK' for r in statuses):raise ValueError('plant/belief pairing failed')
    by_run={(r['seed'],r['solver_style']):r for r in runs}
    by_modes=defaultdict(list)
    for r in modes:by_modes[r['seed'],r['solver_style']].append(r)
    paired=[];decisions=[];windows=[]
    for seed in sorted({r['seed'] for r in runs},key=int):
        a,b=(by_run[seed,arm] for arm in ['sh_mpcc','sh_mpcc_dro'])
        pair=dict(cell=cell,seed=seed,nominal_collision=int(a['collision']),wdro_collision=int(b['collision']),
            nominal_refusal=int(a['termination']=='no_admissible_control'),wdro_refusal=int(b['termination']=='no_admissible_control'),
            nominal_completed=int(a['path_completed']),wdro_completed=int(b['path_completed']),
            nominal_switch_reached=int(a['switch_reached']),wdro_switch_reached=int(b['switch_reached']))
        for label,arm in [('nominal','sh_mpcc'),('wdro','sh_mpcc_dro')]:
            rr=by_modes[seed,arm]
            for k in [1,2,3]:
                w=window_metrics(rr,window,k)
                windows.append(dict(cell=cell,seed=seed,arm=arm,kappa=k,**w))
                pair.update({f'{label}_k{k}_{key}':value for key,value in w.items()})
        paired.append(pair)
        nom={r['step']:r for r in by_modes[seed,'sh_mpcc'] if r['attempt']=='0'}
        dro={r['step']:r for r in by_modes[seed,'sh_mpcc_dro'] if r['attempt']=='0'}
        for step in sorted(nom.keys()&dro.keys(),key=int):
            n,d=nom[step],dro[step]
            decisions.append(dict(cell=cell,seed=seed,step=step,true_mode=n['true_mode'],p_d_nominal=n['p_d'],p_d_wdro=d['p_d'],q_d_wdro=d['q_d'],r_d_wdro=d['r_d'],
                N_d_nominal=n['N_d'],N_d_wdro=d['N_d'],delta_q=d['q_d']-d['p_d'],delta_N=d['N_d']-n['N_d'],
                collision_improvement=pair['nominal_collision']-pair['wdro_collision'],**{f'delta_U_k{k}':d[f'delta_U_k{k}'] for k in [1,2,3]}))
    initial=[r for r in modes if r['solver_style']=='sh_mpcc' and r['step']=='0' and r['attempt']=='0']
    return summarize(cell,paired,initial),paired,[dict(r,cell=cell) for r in modes],decisions,windows,runs


def ordering_check(selected,paired,replicates=5000):
    ordered=sorted(selected,key=lambda r:float(r['delta_C']),reverse=True)
    cells=[r['cell'] for r in ordered]
    by_cell={cell:{r['seed']:r['nominal_collision']-r['wdro_collision'] for r in paired if r['cell']==cell} for cell in cells}
    seeds=sorted(by_cell[cells[0]],key=int)
    if any(set(by_cell[c])!=set(seeds) for c in cells):raise ValueError('validation seed sets differ')
    benefit=[mean(by_cell[c][s] for s in seeds) for c in cells]
    rng=random.Random(91371);boot=[]
    for _ in range(replicates):
        sample=rng.choices(seeds,k=len(seeds))
        boot.append([mean(by_cell[c][s] for s in sample) for c in cells])
    comparisons=[]
    for i in range(len(cells)-1):
        differences=sorted(row[i]-row[i+1] for row in boot)
        comparisons.append(dict(higher_predicted_gain_cell=cells[i],lower_predicted_gain_cell=cells[i+1],
            observed_benefit_difference=benefit[i]-benefit[i+1],bootstrap95_low=differences[int(.025*replicates)],bootstrap95_high=differences[int(.975*replicates)]))
    return dict(cells_descending_predicted_gain=cells,observed_collision_benefits=benefit,
        observed_nondecreasing_benefit_with_gain=all(benefit[i]>=benefit[i+1] for i in range(len(cells)-1)),
        clustered_bootstrap_replicates=replicates,cluster='shared seed across all validation cells and both arms',
        bootstrap_fraction_ordered=sum(all(row[i]>=row[i+1] for i in range(len(cells)-1)) for row in boot)/replicates,
        caveat='Bootstrap fraction is a descriptive stability measure, not a posterior probability or hypothesis-test p-value'),comparisons


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/coverage-replication'));a=p.parse_args();root=a.root
    protocol=json.loads((root/'design_protocol_before_outcomes.json').read_text());selected=json.loads((root/'selection_before_validation_outcomes.json').read_text())['selected']
    all_summary=[];all_paired=[];bundle_files=[]
    for group,cells in [('replication',['replication']),('designed_validation',[r['cell'] for r in selected])]:
        summary=[];paired=[];modes=[];decisions=[];windows=[];runs=[]
        for cell in cells:
            s,pp,mm,dd,ww,rr=collect(root,cell,protocol['critical_window'])
            if group=='designed_validation':s.update({key:value for key,value in next(r for r in selected if r['cell']==cell).items() if key!='cell'})
            s['study']=group;summary.append(s);paired.extend(pp);modes.extend(mm);decisions.extend(dd);windows.extend(ww);runs.extend(dict(r,cell=cell) for r in rr)
        # Holm adjustment across four validation collision tests; independent replication remains separate.
        if group=='designed_validation':
            ordered=sorted(summary,key=lambda r:r['collision_exact_mcnemar_p']);previous=0.
            for rank,r in enumerate(ordered):
                previous=max(previous,min(1.,(len(ordered)-rank)*r['collision_exact_mcnemar_p']));r['collision_holm_p']=previous
        target=root/'analysis'/group
        for name,rows in [('transition_summary',summary),('paired_seed_outcomes',paired),('across_every_solve',modes),('across_paired_decisions',decisions),('critical_window_coverage',windows),('run_outcomes',runs)]:
            path=target/(name+'.csv');write(path,rows);bundle_files.append((path,f'{group}/{name}.csv'))
        all_summary.extend(summary);all_paired.extend(paired)
    calibration=[]
    for design in selected:
        rr=[r for r in all_paired if r['cell']==design['cell']]
        record=dict(cell=design['cell'],predicted_delta_C=design['delta_C'],predicted_C_p=design['C_p'],predicted_C_q=design['C_q'])
        for arm in ['nominal','wdro']:
            complete=[r for r in rr if r[f'{arm}_k1_window_complete']]
            record[arm+'_complete_windows']=len(complete)
            record[arm+'_planned_seeds']=len(rr)
            record[arm+'_empirical_all_steps_covered']=mean(r[f'{arm}_k1_all_steps_N_ge_k'] for r in complete) if complete else ''
            record[arm+'_mean_realized_path_plugin_C']=mean(r[f'{arm}_k1_plugin_C_sampling'] for r in complete) if complete else ''
        record['empirical_coverage_gain_complete_windows']=record['wdro_empirical_all_steps_covered']-record['nominal_empirical_all_steps_covered'] if all(record[a+'_complete_windows'] for a in ['nominal','wdro']) else ''
        calibration.append(record)
    write(root/'analysis/design_calibration.csv',calibration)
    bundle_files.append((root/'analysis/design_calibration.csv','designed_validation/design_calibration.csv'))
    ordering,contrasts=ordering_check(selected,all_paired)
    (root/'analysis/ordering.json').write_text(json.dumps(ordering,indent=2)+'\n')
    write(root/'analysis/ordering_contrasts.csv',contrasts)
    bundle_files.extend([(root/'analysis/ordering.json','designed_validation/ordering.json'),(root/'analysis/ordering_contrasts.csv','designed_validation/ordering_contrasts.csv')])
    for filename in ['design_selected.csv','design_candidates.csv','design_window_weights.csv','design_protocol_before_outcomes.json','selection_before_validation_outcomes.json']:
        bundle_files.append((root/filename,'design/'+filename))
    prior=Path('results/persistent-coverage')
    for filename in ['confirmation_summary.csv','confirmation_outcomes.csv','paired_outcomes.csv']:
        bundle_files.append((prior/'intervention-updated-belief'/filename,'intervention-updated-belief/'+filename))
    recovery=prior/'recoverability-updated-belief'
    filenames=['confirmation_summary.csv','confirmation_outcomes.csv'] if all((recovery/f).exists() for f in ['confirmation_summary.csv','confirmation_outcomes.csv']) else ['pilot_selection_scores.csv']
    for filename in filenames:bundle_files.append((recovery/filename,'recoverability-updated-belief/'+filename))
    lines=['# Independent replication and coverage-gain design','',
        'Replication uses fresh paired seeds 3001–3100. Validation uses paired seeds 4001–4100, shared across the four cells. Both use the original late-switch geometry, switch step 10, history lag 5, and a 120-step limit. No controller code or mathematical formulation was modified for this study.','',
        '| Study/cell | Predicted ΔC | Nominal collisions | WDRO collisions | Exact paired p |','|---|---:|---:|---:|---:|']
    for r in all_summary:lines.append(f"| {r['cell']} | {r.get('delta_C','—')} | {r['nominal_collision_count']}/100 | {r['wdro_collision_count']}/100 | {r['collision_exact_mcnemar_p']:.5g} |")
    lines += ['', '## Interpretation', '',
        f"Observed collision-benefit ordering follows predicted gain: **{ordering['observed_nondecreasing_benefit_with_gain']}**. See ordering_contrasts.csv for shared-seed bootstrap intervals. The bootstrap ordering fraction is descriptive, not a hypothesis-test p-value.",
        'Predicted C_W is the product of single-decision coverage probabilities for kappa=1 and steps 5–9 on a deterministic fixed reference. It is an outcome-free design proxy, not a theorem about adaptive closed-loop trajectories. No validation outcome was used to select cells. Candidate targets were 0.20, 0.10, 0.05 and 0.02.',
        'Critical-window CSVs report actual minimum count, coverage fraction, all-step coverage, and plug-in probability products for kappa=1,2,3 per seed/arm. Incomplete windows retain their observed denominator and leave full-window metrics blank. Products evaluated along realized adaptive trajectories are descriptive quantities, not marginal probability guarantees.',
        'Nominal and WDRO states diverge after applying different controls. Paired decision tables are descriptive; paired seed outcomes are the independent units. Collision rescue is not successful traversal; inspect refusal, path completion, and switch reach as separate outcomes.',
        'The existing corrected intervention results are bundled unchanged and kept separate. Their five-step count intervention did not identify a useful kappa. No recoverability confirmation exists, so its pilot selection scores are included. Older frozen-belief pilots are excluded.', '',
        'Candidate analysis implementation tested; behavior and scientific interpretation require user verification.']
    report=root/'REPORT.md';report.write_text('\n'.join(lines)+'\n');bundle_files.append((report,'README.md'))
    manifest=[dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,name in bundle_files]
    manifest_path=root/'bundle_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');bundle_files.append((manifest_path,'bundle_manifest.json'))
    archive=root/'replication-and-coverage-validation.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path,name in bundle_files:z.write(path,arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(bundle_files)
    print(report);print(archive,archive.stat().st_size,'bytes',len(bundle_files),'files')

if __name__=='__main__':main()
