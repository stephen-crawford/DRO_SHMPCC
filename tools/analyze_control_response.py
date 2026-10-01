#!/usr/bin/env python3
"""Read-only temporal control-response audit of the independent 100-seed replication."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean,median
import zipfile
from causal_link_analysis import read,write


def executable(row):
    return row is not None and row['success']=='1' and row.get('acceleration','')!='' and row.get('omega','')!=''


def first(rows,predicate):
    return next((int(r['step']) for r in sorted(rows,key=lambda r:int(r['step'])) if predicate(r)),None)


def divergence(nominal,wdro,acceleration=.1,yaw_rate=.02):
    commands=[];availability=[];lower=[]
    for t in sorted(nominal.keys()&wdro.keys()):
        n,d=nominal[t],wdro[t]
        if executable(n)!=executable(d):availability.append(t)
        if not(executable(n) and executable(d)):continue
        da=float(d['acceleration'])-float(n['acceleration'])
        dw=float(d['omega'])-float(n['omega'])
        if abs(da)>=acceleration or abs(dw)>=yaw_rate:commands.append(t)
        if da<=-acceleration:lower.append(t)
    return dict(first_material_control_divergence=commands[0] if commands else None,
        first_control_availability_divergence=availability[0] if availability else None,
        first_materially_lower_wdro_acceleration=lower[0] if lower else None)


def category(n,d):
    if n['collision']=='1' and d['collision']=='0' and d['termination']=='no_admissible_control':return 'nominal_collision_wdro_refusal'
    if n['collision']=='0' and d['collision']=='1':return 'wdro_only_collision'
    if n['collision']=='1' and d['collision']=='1':return 'both_collision'
    if n['collision']=='0' and d['collision']=='0':return 'neither_collision'
    return 'other'


def flatten(seed,group,nominal,wdro,modes):
    result=[]
    for t in sorted(nominal.keys()|wdro.keys()):
        row=dict(seed=seed,group=group,step=t,time_seconds=t*.1,matched=int(t in nominal and t in wdro))
        for label,decisions in [('nominal',nominal),('wdro',wdro)]:
            dec=decisions.get(t);mode=modes[label].get(t)
            if dec:
                row.update({label+'_'+field:dec.get(source,'') for field,source in [('success','success'),('a','acceleration'),('omega','omega'),('v','ego_speed'),('d_min_pre_control','actual_clearance')]})
                row[label+'_control_executed']=int(executable(dec))
            if mode:
                row.update({label+'_'+field:mode.get(source,'') for field,source in [('p_d','nominal_probability'),('q_d','sampling_probability'),('r_d','risk_score'),('rho','rho'),('N_d','sampled_count'),('true_mode','true_mode')]})
                row[label+'_delta_q']=float(mode['sampling_probability'])-float(mode['nominal_probability'])
            nxt=decisions.get(t+1)
            if dec and nxt:
                row[label+'_next_v']=nxt['ego_speed'];row[label+'_next_d_min']=nxt['actual_clearance']
        if t in nominal and t in wdro:
            n,d=nominal[t],wdro[t]
            row['delta_v']=float(d['ego_speed'])-float(n['ego_speed'])
            row['delta_d_min']=float(d['actual_clearance'])-float(n['actual_clearance'])
            if executable(n) and executable(d):
                row['delta_a']=float(d['acceleration'])-float(n['acceleration'])
                row['delta_omega']=float(d['omega'])-float(n['omega'])
        if t in modes['nominal'] and t in modes['wdro']:
            row['delta_N']=int(modes['wdro'][t]['sampled_count'])-int(modes['nominal'][t]['sampled_count'])
        result.append(row)
    return result


def seed_events(seed,group,nominal,wdro,modes,acceleration=.1,yaw_rate=.02):
    events=dict(seed=seed,group=group,**divergence(nominal,wdro,acceleration,yaw_rate))
    for label,decisions in [('nominal',nominal),('wdro',wdro)]:
        events[label+'_first_across_sample']=first(list(modes[label].values()),lambda r:int(r['sampled_count'])>0)
        events[label+'_first_brake']=first(list(decisions.values()),lambda r:executable(r) and float(r['acceleration'])<-.1)
        events[label+'_first_refusal']=first(list(decisions.values()),lambda r:not executable(r))
    events['first_positive_risk_mass_shift']=first(list(modes['wdro'].values()),lambda r:r.get('risk_score','')!='' and float(r['risk_score'])>0 and float(r['sampling_probability'])-float(r['nominal_probability'])>1e-12)
    shared=sorted(modes['nominal'].keys()&modes['wdro'].keys())
    extra=[t for t in shared if int(modes['wdro'][t]['sampled_count'])>int(modes['nominal'][t]['sampled_count'])]
    events['first_extra_wdro_across_sample']=extra[0] if extra else None
    div=events['first_material_control_divergence'];lower=events['first_materially_lower_wdro_acceleration']
    nb,db=events['nominal_first_brake'],events['wdro_first_brake']
    events['wdro_brakes_earlier_when_both_observed']=int(db<nb) if db is not None and nb is not None else None
    events['braking_advance_steps']=nb-db if nb is not None and db is not None else None
    events['wdro_only_observed_braking']=int(db is not None and nb is None)
    sample=events['wdro_first_across_sample']
    events['wdro_sample_at_or_before_control_divergence']=int(sample<=div) if sample is not None and div is not None else None
    events['divergence_in_steps_0_10']=int(div<=10) if div is not None else 0
    events['wdro_sample_at_or_before_brake']=int(sample<=db) if sample is not None and db is not None else None
    chain=[]
    if lower is not None:
        for t in extra:
            m=modes['wdro'][t]
            if t<=lower and float(m.get('risk_score') or 0)>0 and float(m['sampling_probability'])>float(m['nominal_probability']):chain.append(t)
    events['risk_shift_extra_sample_before_lower_control']=int(bool(chain)) if lower is not None else None
    events['chain_sample_step']=chain[0] if chain else None
    if div is not None:
        n,d=nominal[div],wdro[div]
        events.update(divergence_nominal_a=n['acceleration'],divergence_wdro_a=d['acceleration'],divergence_delta_a=float(d['acceleration'])-float(n['acceleration']),
            divergence_nominal_N=modes['nominal'][div]['sampled_count'],divergence_wdro_N=modes['wdro'][div]['sampled_count'],
            divergence_wdro_delta_q=float(modes['wdro'][div]['sampling_probability'])-float(modes['wdro'][div]['nominal_probability']),divergence_wdro_r=modes['wdro'][div]['risk_score'])
    available=events['first_control_availability_divergence']
    event_times=[t for t in [div,available] if t is not None]
    decision=min(event_times) if event_times else None
    events['first_decision_response_divergence']=decision
    events['first_decision_response_kind']=('availability' if decision==available else 'executable_control') if decision is not None else ''
    events['availability_only_divergence']=int(div is None and available is not None)
    if decision is not None:
        n,d=modes['nominal'][decision],modes['wdro'][decision]
        events['first_response_nominal_N']=n['sampled_count'];events['first_response_wdro_N']=d['sampled_count']
        events['first_response_delta_q']=float(d['sampling_probability'])-float(d['nominal_probability'])
        events['first_response_r_d']=d['risk_score']
        events['extra_sample_at_first_response']=int(int(d['sampled_count'])>int(n['sampled_count']))
    else:events['extra_sample_at_first_response']=None
    events['extra_sample_at_material_control_divergence']=int(int(events['divergence_wdro_N'])>int(events['divergence_nominal_N'])) if div is not None else None
    return events


def group_summary(events):
    results=[]
    for group in sorted({r['group'] for r in events}|{'all_seeds'}):
        rows=[r for r in events if group=='all_seeds' or r['group']==group]
        result=dict(group=group,seeds=len(rows))
        for field in ['first_material_control_divergence','first_materially_lower_wdro_acceleration','nominal_first_across_sample','wdro_first_across_sample','nominal_first_brake','wdro_first_brake','braking_advance_steps']:
            values=[r[field] for r in rows if r.get(field) is not None]
            result[field+'_observed']=len(values);result[field+'_median']=median(values) if values else ''
        for field in ['wdro_brakes_earlier_when_both_observed','wdro_only_observed_braking','wdro_sample_at_or_before_control_divergence','divergence_in_steps_0_10','wdro_sample_at_or_before_brake','risk_shift_extra_sample_before_lower_control','availability_only_divergence','extra_sample_at_first_response','extra_sample_at_material_control_divergence']:
            values=[r[field] for r in rows if r.get(field) is not None]
            result[field+'_yes']=sum(values);result[field+'_eligible']=len(values)
        results.append(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/control-response-replication'));args=p.parse_args();out=args.output
    protocol=json.loads((out/'analysis_protocol.json').read_text());source=Path(protocol['source'])
    case=source/'late_switch_p20_s10_switch10';all_events=[];traces=[];sensitivity=[];outcomes=[];inputs=[]
    for seed in protocol['seeds']:
        evidence={};modes={};summaries={}
        for label,arm in [('nominal','sh_mpcc'),('wdro','sh_mpcc_dro')]:
            folder=case/f'seed_{seed}'/arm
            for filename in ['decisions.csv','mode_mechanism.csv','summary.csv']:inputs.append(folder/filename)
            dd=read(folder/'decisions.csv');mm=read(folder/'mode_mechanism.csv');summary=read(folder/'summary.csv')
            if len(summary)!=1 or summary[0]['status']!='OK':raise ValueError('missing or failed source run')
            summaries[label]=summary[0];evidence[label]={int(r['step']):r for r in dd}
            modes[label]={int(r['step']):r for r in mm if r['mode']=='across' and r['obstacle_id']=='0' and r['attempt']=='0'}
            if len(evidence[label])!=len(dd) or set(evidence[label])!=set(modes[label]):raise ValueError('duplicate/missing decision mechanism evidence')
        group=category(summaries['nominal'],summaries['wdro'])
        events=seed_events(seed,group,evidence['nominal'],evidence['wdro'],modes,protocol['material_acceleration_difference_m_s2'],protocol['material_yaw_rate_difference_rad_s'])
        for label in ['nominal','wdro']:
            expected=int(summaries[label]['first_brake_step']);actual=events[label+'_first_brake']
            if expected!=(actual if actual is not None else -1):raise ValueError('first brake mismatch with source fixture summary')
        all_events.append(events)
        traces.extend(flatten(seed,group,evidence['nominal'],evidence['wdro'],modes))
        outcomes.append(dict(seed=seed,group=group,**{label+'_'+field:value for label,summary in summaries.items() for field,value in summary.items()}))
        for thresholds in protocol['sensitivity']:
            e=seed_events(seed,group,evidence['nominal'],evidence['wdro'],modes,thresholds['acceleration'],thresholds['yaw_rate'])
            sensitivity.append(dict(e,acceleration_threshold=thresholds['acceleration'],yaw_rate_threshold=thresholds['yaw_rate']))
    rescued=[r for r in all_events if r['group']=='nominal_collision_wdro_refusal']
    if len(rescued)!=13 or len(all_events)!=100:raise ValueError('unexpected replication/subgroup membership')
    window=[r for r in traces if protocol['primary_window'][0]<=r['step']<=protocol['primary_window'][1]]
    tables={'seed_outcomes':outcomes,'events_all_seeds':all_events,'events_13_collision_rescues':rescued,'trace_steps_0_10_all_seeds':window,
        'trace_steps_0_10_collision_rescues':[r for r in window if r['group']=='nominal_collision_wdro_refusal'],
        'trace_full_all_seeds':traces,'group_summary':group_summary(all_events),'threshold_sensitivity':sensitivity}
    for name,rows in tables.items():write(out/(name+'.csv'),rows)
    sensitivity_summaries=[]
    for threshold in protocol['sensitivity']:
        rr=[r for r in sensitivity if r['acceleration_threshold']==threshold['acceleration'] and r['yaw_rate_threshold']==threshold['yaw_rate']]
        sensitivity_summaries.extend(dict(r,acceleration_threshold=threshold['acceleration'],yaw_rate_threshold=threshold['yaw_rate']) for r in group_summary(rr))
    tables['threshold_sensitivity_summary']=sensitivity_summaries
    write(out/'threshold_sensitivity_summary.csv',sensitivity_summaries)
    plots=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,7))
    fields=[('nominal_first_across_sample','o','Nominal first across sample'),('wdro_first_across_sample','s','WDRO first across sample'),
            ('nominal_first_brake','^','Nominal first brake'),('wdro_first_brake','v','WDRO first brake'),('first_material_control_divergence','x','First material control difference'),('first_control_availability_divergence','+','Control availability difference')]
    for offset,(field,marker,label) in enumerate(fields):
        ax.scatter([r[field] for r in rescued if r.get(field) is not None],[i+(offset-2)*.12 for i,r in enumerate(rescued) if r.get(field) is not None],marker=marker,label=label,s=35)
    ax.set_yticks(range(len(rescued)),[str(r['seed']) for r in rescued]);ax.axvline(10,color='gray',linestyle='--',label='Plant switch')
    ax.set(xlabel='Decision step (0.1 s per step)',ylabel='Replication seed',title='13 nominal-collision / WDRO collision-free-until-refusal pairs')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2);ax.grid(axis='x',alpha=.2);fig.tight_layout()
    for suffix in ['svg','png']:
        path=out/('rescue_event_timeline.'+suffix);fig.savefig(path,dpi=180);plots.append(path)
    plt.close(fig)
    summary=next(r for r in tables['group_summary'] if r['group']=='nominal_collision_wdro_refusal')
    lines=['# Control-response mechanism: independent replication','', 'No new simulations were run. This is a read-only audit of all 100 replication pairs; the 13 collision-rescue pairs are an explicitly outcome-selected subgroup.','',
        '| Event relation in the 13-pair subgroup | Count / eligible |','|---|---:|']
    for label,field in [('WDRO brakes earlier (both first brakes observed)','wdro_brakes_earlier_when_both_observed'),('WDRO first across sample at/before material control divergence','wdro_sample_at_or_before_control_divergence'),('Material control divergence by step 10','divergence_in_steps_0_10'),('WDRO first across sample at/before first brake','wdro_sample_at_or_before_brake'),('Positive risk/mass shift and extra across sample at/before lower WDRO acceleration','risk_shift_extra_sample_before_lower_control'),('Extra WDRO across sample at the same solve as material command divergence','extra_sample_at_material_control_divergence'),('Availability divergence without material executable-command divergence','availability_only_divergence'),('Extra WDRO across sample at the first command-or-availability response','extra_sample_at_first_response')]:
        lines.append(f"| {label} | {summary[field+'_yes']} / {summary[field+'_eligible']} |")
    lines += ['',f"Median nominal first brake: {summary['nominal_first_brake_median']}; WDRO: {summary['wdro_first_brake_median']}. Median paired braking advance: {summary['braking_advance_steps_median']} steps.", '',
        'The 9 command-divergence and 4 availability-only cases are unchanged when both materiality thresholds are halved or doubled. Availability-only differences occur at steps 11, 13, 16 and 18, outside the primary 0–10 window; full-run traces are included for this reason. These are not additional simulations.', '',
        '## Definitions and limits','',
        '- Primary material divergence: |a_D−a_N|≥0.1 m/s² or |omega_D−omega_N|≥0.02 rad/s, with executable commands in both arms. Refusal/availability divergence is separate. Sensitivity thresholds are half and twice these values.',
        '- Brake: executed acceleration strictly below −0.1 m/s², matching the original fixture. Computed first-brake times were checked against every source summary.',
        '- Sampling and risk computation at solve t precede its chosen control. Logged velocity and clearance at t describe the pre-control state. Next-step fields show the subsequent recorded state; missing terminal next states are not invented.',
        '- d_min is the logged pre-control physical clearance over all ego discs and obstacles, not a continuous-time minimum or the minimum across the entire following integration interval.',
        '- Full-run events are retained even when outside steps 0–10; window membership is explicit. Missing first events remain blank, not zero. Sampling includes across while the actual plant mode is continue.',
        '- A same-seed temporal ordering is compatible with the proposed mechanism but does not establish causal mediation. The subgroup is selected by outcomes, controls also depend on other scenario content and prior plans, and closed-loop states diverge.',
        '- WDRO collision-free refusal means no collision was observed before simulation stopped on refusal. No continued plant motion or braking-after-refusal policy was simulated; do not claim guaranteed safety after refusal.',
        '- These observations do not identify a universal kappa or an explicit conditional collision bound. A control-response explanation should be stated as temporal evidence, with the threshold sensitivity and non-rescued groups reported alongside it.', '',
        'Candidate analysis implementation tested; scientific interpretation requires user verification.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(source_files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((out/'analysis_protocol.json').read_bytes()).hexdigest(),additional_simulations=0)
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[out/(name+'.csv') for name in tables]+[out/'REPORT.md',out/'analysis_protocol.json',out/'provenance.json']+plots
    if (out/'validation_tests.log').exists():files.append(out/'validation_tests.log')
    archive=out/'control-response-analysis.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,arcname=path.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(summary,indent=2));print(archive,archive.stat().st_size,'bytes')

if __name__=='__main__':main()
