#!/usr/bin/env python3
"""Paper tables and inspectable numerical checks from canonical matrix artifacts."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def percentile(values, p):
    values=sorted(values); position=(len(values)-1)*p
    lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def write(path, records):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list);audits=[];all_rollouts=[]
    for matrix in args.matrix:
        manifest=json.loads((matrix/'matrix.json').read_text())
        results=json.loads((matrix/'results.json').read_text())
        if any(r['status']!='OK' for r in results):
            raise ValueError('matrix contains errors; do not silently exclude failed trials')
        if any(r['status']!='OK' for r in rows(matrix/'pairing.csv')):
            raise ValueError('matrix contains incomplete or broken pairs')
        threshold=manifest['settings']['overrides']['dangerous_risk_threshold']
        for trial in results:
            bundle=matrix/trial['case']/f"seed_{trial['seed']}"/'repeat_0'
            record=rows(bundle/'rollout.csv')[0];decisions=rows(bundle/'decisions.csv')
            key=(trial['environment'],trial['obstacles'],trial['modes_per_class'],trial['solver_style'])
            groups[key].append((record,decisions))
            all_rollouts.append(dict(matrix=str(matrix),case=trial['case'],seed=trial['seed'],
                completed=int(record['completed_path']),collision=int(record['collision']),
                termination=record['termination_reason'],steps=int(record['total_steps']),
                mean_ms=statistics.mean(float(d['solve_ms']) for d in decisions),
                p95_ms=percentile([float(d['solve_ms']) for d in decisions],.95),
                max_ms=max(float(d['solve_ms']) for d in decisions)))
            nominal={(r['step'],r['obstacle_id'],r['mode']):float(r['nominal_probability'])
                     for r in rows(bundle/'mode_mechanism.csv') if r['attempt']=='0'}
            distributions=defaultdict(list)
            for row in rows(bundle/'distribution_transfer.csv'):
                distributions[(row['step'],row['obstacle_id'])].append(row)
            for (step,obstacle),modes in distributions.items():
                z=float(modes[0]['domination']);zh=float(modes[0]['nominal_domination'])
                q=[float(m['q']) for m in modes];r=[float(m['risk']) for m in modes]
                p=[nominal[(step,obstacle,m['mode'])] for m in modes]
                dangerous=[i for i,x in enumerate(r) if x>=threshold]
                floor=min((q[i]-p[i] for i in dangerous),default=0.)
                risk_lift=sum((a-b)*x for a,b,x in zip(q,p,r))
                cost=float(modes[0]['transport_cost']);rho=float(modes[0]['rho'])
                violation=max(0.,z-zh,-floor,-risk_lift,cost-rho,abs(sum(q)-1),
                              max(float(m['envelope'])/zh-a for m,a in zip(modes,q)))
                audits.append(dict(case=trial['case'],seed=trial['seed'],step=step,obstacle=obstacle,
                    domination=z,nominal_domination=zh,risk_lift=risk_lift,
                    dangerous_modes=len(dangerous),minimum_dangerous_mass_change=floor,
                    constraint_error=violation))
    table=[]
    for (environment,obstacles,modes,method),runs in sorted(groups.items()):
        decisions=[d for _,ds in runs for d in ds]
        times=[float(d['solve_ms']) for d in decisions]
        samples=[int(d['scenario_count']) for d in decisions]
        table.append(dict(environment=environment,obstacles=obstacles,modes=modes,method=method,
            rollouts=len(runs),completed=sum(int(r['completed_path']) for r,_ in runs),
            collisions=sum(int(r['collision']) for r,_ in runs),decisions=len(times),
            mean_ms=statistics.mean(times),median_ms=statistics.median(times),p95_ms=percentile(times,.95),
            max_ms=max(times),deadline_miss_fraction=sum(t>100 for t in times)/len(times),
            scenarios_min=min(samples),scenarios_max=max(samples),
            domination_max=max(float(d['domination_factor']) for d in decisions),
            sampling_certified=sum(int(d['certified']) for d in decisions),
            transfer_conditions_met=sum(int(d['transfer_bound_satisfied']) for d in decisions)))
    write(args.output/'practicality.csv',table);write(args.output/'rollouts.csv',all_rollouts)
    if audits: write(args.output/'reweighting_audit.csv',audits)
    audit=dict(obstacle_decisions=len(audits),tolerance=1e-7,
        max_constraint_error=max((a['constraint_error'] for a in audits),default=0.),
        violations=sum(a['constraint_error']>1e-7 for a in audits),
        note='Numerical feasibility/monotonicity checks; not a proof of plant-distribution assumptions.')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    latex=[r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Setup & Method & Complete & Collide & Median [ms] & P95 [ms] & $S_{\max}$ \\',r'\midrule']
    for r in table:
        scene=r['environment'].replace('_',r'\_')
        method='WDRO' if r['method']=='sh_mpcc_dro' else 'SH-MPCC'
        latex.append(f"{scene}, {r['obstacles']} obs. & {method} & {r['completed']}/{r['rollouts']} & "
                     f"{r['collisions']}/{r['rollouts']} & {r['median_ms']:.1f} & {r['p95_ms']:.1f} & {r['scenarios_max']} " + r'\\')
    latex.extend([r'\bottomrule',r'\end{tabular}'])
    (args.output/'practicality.tex').write_text('\n'.join(latex)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for count,axis in zip((1,2),axes):
        for method in ('sh_mpcc','sh_mpcc_dro'):
            times=sorted(float(d['solve_ms']) for key,runs in groups.items()
                         if key[1]==count and key[3]==method for _,ds in runs for d in ds)
            if times:
                axis.step(times,[(i+1)/len(times) for i in range(len(times))],where='post',
                          label='WDRO' if method=='sh_mpcc_dro' else 'SH-MPCC')
        axis.axvline(100,color='black',ls=':',lw=1,label='100 ms')
        axis.set_xscale('log');axis.set_ylim(0,1.01);axis.set_title(f'{count} obstacle(s)')
        axis.set_xlabel('Control-cycle wall time [ms]');axis.grid(alpha=.2)
    axes[0].set_ylabel('Empirical CDF');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'latency.pdf');fig.savefig(args.output/'latency.svg')
    print(json.dumps(audit,indent=2))
    return int(audit['violations']>0)

if __name__=='__main__':
    raise SystemExit(main())
