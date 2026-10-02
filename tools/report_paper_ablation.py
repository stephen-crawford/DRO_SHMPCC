#!/usr/bin/env python3
"""Report the predeclared three-arm experiment and the feedback's two figures."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

ARMS=('baseline','reweight_only','full_transfer')
LABELS={'baseline':'SH-MPCC','reweight_only':'Reweight only','full_transfer':'Full transfer'}


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write(path,data):
    if not data:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;lo=math.floor(x);hi=math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def optional_median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    matrix=args.matrix;manifest=json.loads((matrix/'matrix.json').read_text())
    results=json.loads((matrix/'results.json').read_text())
    expected=sum(len(c['seeds']) for c in manifest['cases'])
    if len(results)!=expected:raise ValueError('experiment still incomplete')
    rollouts=[];decision_rows=[];audit_rows=[];traces={};decision_data={};risk_data={}
    for trial in results:
        pair,arm,seed=trial['pair'],trial['arm'],trial['seed'];s0=manifest['nominal_sample_counts'][pair]
        base=dict(pair=pair,arm=arm,seed=seed,status=trial['status'],reused=bool(trial.get('reused_from')),
                  environment=trial['environment'],obstacles=trial['obstacles'],modes=trial['modes_per_class'])
        item=dict(base,completion=None,completion_steps=None,collision=None,minimum_clearance_m=None,
            final_progress_fraction=None,termination=None,decisions=None,median_ms=None,p95_ms=None,max_ms=None,
            max_S=None,median_S_over_S0=None,max_zeta=None,mean_obstacle_risk_lift=None,max_obstacle_risk_lift=None,
            final_support_count=None,final_support_evaluated=None,max_support_count=None,
            transfer_conditions_met=None,deadline_miss_fraction=None,error=trial.get('error',''))
        if trial['status']!='OK':rollouts.append(item);continue
        bundle=matrix/trial['case']/f'seed_{seed}'/'repeat_0'
        record=rows(bundle/'rollout.csv')[0];decisions=rows(bundle/'decisions.csv')
        support={r['step']:r for r in rows(bundle/'support_scenarios.csv')}
        mechanism=rows(bundle/'mode_mechanism.csv')
        nominal={(r['step'],r['obstacle_id'],r['mode']):float(r['nominal_probability'])
                 for r in mechanism if r['attempt']=='0'}
        transfer=defaultdict(list)
        for r in rows(bundle/'distribution_transfer.csv'):transfer[(r['step'],r['obstacle_id'])].append(r)
        lifts=[];step_lifts=defaultdict(float);rhos=defaultdict(float)
        for (step,obstacle),modes in transfer.items():
            q=[float(m['q']) for m in modes];p=[nominal[(step,obstacle,m['mode'])] for m in modes]
            risk=[float(m['risk']) for m in modes];u=[float(m['envelope']) for m in modes]
            z=float(modes[0]['domination']);zh=float(modes[0]['nominal_domination'])
            lift=sum((a-b)*r for a,b,r in zip(q,p,risk));lifts.append(lift);step_lifts[step]+=lift
            rho=float(modes[0]['rho']);rhos[step]=max(rhos[step],rho)
            dangerous=[i for i,r in enumerate(risk) if r>=0.01]
            dangerous_error=max([0.]+[p[i]-q[i] for i in dangerous])
            budget_error=max(0.,float(modes[0]['transport_cost'])-rho)
            domination_error=max([0.,z-zh]+[a-z*b for a,b in zip(u,q)])
            floor_error=max([0.]+[a/zh-b for a,b in zip(u,q)])
            audit_rows.append(dict(base,step=int(step),obstacle=int(obstacle),rho=rho,zeta=z,
                nominal_zeta=zh,risk_lift=lift,dangerous_modes=len(dangerous),
                dangerous_mass_error=dangerous_error,transport_budget_error=budget_error,
                domination_error=domination_error,domination_floor_error=floor_error,
                simplex_error=abs(sum(q)-1),risk_lift_error=max(0.,-lift)))
        times=[float(d['solve_ms']) for d in decisions];samples=[int(d['scenario_count']) for d in decisions]
        if arm in ('baseline','reweight_only') and set(samples)!={s0}:
            raise ValueError('fixed-budget arm changed the sample count')
        if arm=='reweight_only' and any(int(d['transfer_bound_satisfied']) for d in decisions):
            raise ValueError('reweight-only was labeled transfer certified')
        evaluated=[int(r['support_count']) for r in support.values() if r['support_evaluated']=='1']
        final_support=support.get(decisions[-1]['step'])
        item.update(completion=int(record['completed_path']),
            completion_steps=int(record['total_steps']) if record['completed_path']=='1' else None,
            collision=int(record['collision']),minimum_clearance_m=float(record['min_clearance']),
            final_progress_fraction=float(record['total_progress']),termination=record['termination_reason'],
            decisions=len(decisions),median_ms=statistics.median(times),p95_ms=percentile(times,.95),max_ms=max(times),
            max_S=max(samples),median_S_over_S0=statistics.median(samples)/s0,
            max_zeta=max(float(d['domination_factor']) for d in decisions) if arm!='baseline' else None,
            mean_obstacle_risk_lift=statistics.mean(lifts) if lifts else (0. if arm=='baseline' else None),
            max_obstacle_risk_lift=max(lifts) if lifts else (0. if arm=='baseline' else None),
            final_support_count=int(final_support['support_count']) if final_support and final_support['support_evaluated']=='1' else None,
            final_support_evaluated=int(final_support['support_evaluated']) if final_support else None,
            max_support_count=max(evaluated) if evaluated else None,
            transfer_conditions_met=sum(int(d['transfer_bound_satisfied']) for d in decisions),
            deadline_miss_fraction=sum(t>100 for t in times)/len(times))
        rollouts.append(item);key=(pair,arm,seed)
        decision_data[key]=decisions;risk_data[key]=(step_lifts,rhos)
        log=(bundle.parent/'repeat_0.log').read_text()
        lengths=[float(x) for x in re.findall(r'path_length=([0-9.eE+-]+)',log)]
        if not lengths or max(lengths)!=min(lengths):raise ValueError('missing or changing route length')
        ego=[r for r in rows(bundle/'trace.csv') if r['actor']=='ego']
        progress=[dict(step=int(r['step']),normalized_progress=(float(r['path_progress'])-float(ego[0]['path_progress']))/lengths[0]) for r in ego]
        traces[key]=progress
        for d in decisions:
            step=d['step'];supp=support.get(step)
            decision_rows.append(dict(base,step=int(step),solve_ms=float(d['solve_ms']),S=int(d['scenario_count']),
                S0=s0,S_over_S0=int(d['scenario_count'])/s0,
                zeta=float(d['domination_factor']) if arm!='baseline' else None,
                risk_lift_sum=0. if arm=='baseline' else step_lifts.get(step),
                rho_max=rhos.get(step) if arm!='baseline' else None,
                support_count=int(supp['support_count']) if supp and supp['support_evaluated']=='1' else None,
                support_evaluated=int(supp['support_evaluated']) if supp else None,
                success=int(d['success']),transfer_conditions_met=int(d['transfer_bound_satisfied'])))
    write(args.output/'ablation_rollouts.csv',rollouts);write(args.output/'ablation_decisions.csv',decision_rows)
    write(args.output/'mechanism_audit.csv',audit_rows)
    summary=[]
    for pair,arm in sorted({(r['pair'],r['arm']) for r in rollouts}):
        trials=[r for r in rollouts if r['pair']==pair and r['arm']==arm];valid=[r for r in trials if r['status']=='OK']
        ds=[r for r in decision_rows if r['pair']==pair and r['arm']==arm]
        times=[r['solve_ms'] for r in ds]
        summary.append(dict(pair=pair,arm=arm,attempted=len(trials),usable=len(valid),errors=len(trials)-len(valid),
            completed=sum(r['completion'] for r in valid),collisions=sum(r['collision'] for r in valid),
            completion_steps_median=optional_median([r['completion_steps'] for r in valid]),
            minimum_clearance_m=min((r['minimum_clearance_m'] for r in valid),default=None),
            median_ms=statistics.median(times) if times else None,p95_ms=percentile(times,.95) if times else None,
            max_S=max((r['S'] for r in ds),default=None),median_S_over_S0=optional_median([r['S_over_S0'] for r in ds]),
            max_zeta=max((r['max_zeta'] for r in valid if r['max_zeta'] is not None),default=None),
            mean_risk_lift_per_rollout=statistics.mean(r['mean_obstacle_risk_lift'] for r in valid) if valid else None,
            final_support_count_median=optional_median([r['final_support_count'] for r in valid]),
            final_support_unavailable=sum(r['final_support_count'] is None for r in valid),
            deadline_miss_fraction=sum(t>100 for t in times)/len(times) if times else None))
    write(args.output/'ablation_summary.csv',summary)
    paired=[]
    for pair,seed in sorted({(r['pair'],r['seed']) for r in rollouts}):
        entry=dict(pair=pair,seed=seed)
        for arm in ARMS:
            match=next(r for r in rollouts if r['pair']==pair and r['seed']==seed and r['arm']==arm)
            for field in ['status','completion','completion_steps','collision','final_progress_fraction','median_ms','p95_ms','max_S','median_S_over_S0','max_zeta','final_support_count']:
                entry[arm+'_'+field]=match[field]
        paired.append(entry)
    write(args.output/'paired_outcomes.csv',paired)
    error_fields=['dangerous_mass_error','transport_budget_error','domination_error','domination_floor_error','simplex_error','risk_lift_error']
    audit=dict(tolerance=1e-7,obstacle_decisions=len(audit_rows),
        maxima={field:max((r[field] for r in audit_rows),default=0.) for field in error_fields},
        violations=sum(any(r[field]>1e-7 for field in error_fields) for r in audit_rows),
        fixed_budget_and_noncertification_checks='passed',
        paired_groups=rows(matrix/'pairing.csv'))
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'baseline':'#2864a4','full_transfer':'#d46822','reweight_only':'#3b875c'}
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    figure1=[];latencies=[];medians=[]
    for arm in ('baseline','full_transfer'):
        selected=[k for k in traces if '_o1_c1_m2' in k[0] and k[2] in [77,78,79] and k[1]==arm]
        if len(selected)!=9:raise ValueError('Figure 1 requires all original nine practicality runs per arm')
        curves=[];values=[]
        for key in selected:
            data=traces[key];xs=[r['step'] for r in data];ys=[r['normalized_progress'] for r in data]
            axes[0].plot(xs,ys,color=colors[arm],alpha=.18,lw=.8)
            curves.append(np.interp(np.arange(201),xs,ys))
            figure1.extend(dict(pair=key[0],arm=arm,seed=key[2],**r) for r in data)
            for d in decision_data[key]:
                values.append(float(d['solve_ms']));latencies.append(dict(pair=key[0],arm=arm,seed=key[2],step=int(d['step']),solve_ms=float(d['solve_ms'])))
        median=np.median(curves,axis=0)
        axes[0].plot(range(201),median,color=colors[arm],lw=2.2,label=LABELS[arm])
        medians.extend(dict(arm=arm,step=i,median_normalized_progress=float(value),
                            observed_runs=sum(i<=traces[k][-1]['step'] for k in selected),
                            carried_completed_runs=sum(i>traces[k][-1]['step'] for k in selected)) for i,value in enumerate(median))
        values.sort();axes[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=colors[arm],lw=1.8,label=LABELS[arm])
    axes[0].axhline(.95,color='black',ls=':',lw=1);axes[0].set(xlabel='Control step',ylabel='Normalized path progress',xlim=(0,200),ylim=(0,1.02),title='(a) Route progress')
    axes[1].axvline(100,color='black',ls=':',lw=1,label='100 ms period');axes[1].set(xlabel='Control-cycle wall time [ms]',ylabel='Empirical CDF',xscale='log',ylim=(0,1.01),title='(b) Computational cost')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(args.output/f'figure1_practicality.{ext}',dpi=200)
    plt.close(fig)
    write(args.output/'figure1_progress.csv',figure1);write(args.output/'figure1_progress_medians.csv',medians);write(args.output/'figure1_latency.csv',latencies)
    key=('standard_s_curve_o1_c1_m2','full_transfer',77)
    representative=[r for r in decision_rows if (r['pair'],r['arm'],r['seed'])==key]
    write(args.output/'figure2_mechanism.csv',representative)
    fig=plt.figure(figsize=(9,4.8));grid=fig.add_gridspec(2,3)
    for i,(field,label) in enumerate([('rho_max',r'$\rho_t$'),('zeta',r'$\zeta_t$'),('S_over_S0',r'$S_t/S_0$')]):
        ax=fig.add_subplot(grid[0,i]);ax.plot([r['step'] for r in representative],[r[field] for r in representative],color=colors['full_transfer']);ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlabel('Control step')
    ax=fig.add_subplot(grid[1,:]);ax.plot([r['step'] for r in representative],[r['risk_lift_sum'] for r in representative],color=colors['full_transfer']);ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Control step',ylabel=r'$\langle q_t^\star-\widehat p_t,r_t\rangle$');ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(args.output/f'figure2_mechanism.{ext}',dpi=200)
    plt.close(fig)
    print(json.dumps(dict(runs=len(rollouts),new=sum(not r['reused'] for r in rollouts),
        errors=sum(r['status']!='OK' for r in rollouts),audit_violations=audit['violations']),indent=2))
    return int(audit['violations']>0)

if __name__=='__main__':raise SystemExit(main())
