#!/usr/bin/env python3
"""Across-mode coverage at every solve; paired outcomes without selecting favorable seeds."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from causal_link_analysis import read,write,undercoverage,wilson


def measurements(rows):
    result=[]
    for r in rows:
        if r.get('mode')!='across' or r.get('obstacle_id')!='0':continue
        p=float(r['nominal_probability']);q=float(r['sampling_probability']);size=int(r['scenario_count'])
        result.append(dict(r,p_d=p,q_d=q,r_d=r.get('risk_score',''),N_d=int(r['sampled_count']),
            before_switch=int(int(r['step'])<int(r['switch_step'])),
            **{f'U_p_k{k}':undercoverage(p,k,size) for k in [1,2,3]},
            **{f'U_q_k{k}':undercoverage(q,k,size) for k in [1,2,3]},
            **{f'delta_U_k{k}':undercoverage(p,k,size)-undercoverage(q,k,size) for k in [1,2,3]}))
    return result


def analyze(root):
    all_modes=[];all_runs=[]
    for cell in ['p20_s10','p20_s20','p50_s10','p80_s10','p50_s20']:
        modes=measurements(read(root/cell/'report/dangerous_mode_timeseries.csv'))
        runs=read(root/cell/'report/stress_outcomes.csv')
        if len(runs)!=200 or any(r['trial_status']!='OK' for r in runs):raise ValueError(f'incomplete cell: {cell}')
        all_modes.extend(dict(r,cell=cell) for r in modes);all_runs.extend(dict(r,cell=cell) for r in runs)
    write(root/'across_every_solve.csv',all_modes);write(root/'all_run_outcomes.csv',all_runs)
    runs={(r['cell'],r['seed'],r['solver_style']):r for r in all_runs}
    modes={(r['cell'],r['seed'],r['solver_style'],r['step']):r for r in all_modes if r['attempt']=='0'}
    pairs=[];seed_pairs=[];summary=[]
    for cell in ['p20_s10','p20_s20','p50_s10','p80_s10','p50_s20']:
        seeds=sorted({r['seed'] for r in all_runs if r['cell']==cell},key=int)
        for seed in seeds:
            a,b=(runs[cell,seed,arm] for arm in ['sh_mpcc','sh_mpcc_dro'])
            paired=dict(cell=cell,seed=seed,nominal_collision=int(a['collision']),wdro_collision=int(b['collision']),
                nominal_refusal=int(a['termination']=='no_admissible_control'),wdro_refusal=int(b['termination']=='no_admissible_control'),
                nominal_completed=int(a['path_completed']),wdro_completed=int(b['path_completed']),
                collision_improvement=int(a['collision'])-int(b['collision']))
            steps=sorted({r['step'] for r in all_modes if r['cell']==cell and r['seed']==seed and r['attempt']=='0'},key=int)
            shared=[]
            for step in steps:
                n=modes.get((cell,seed,'sh_mpcc',step));d=modes.get((cell,seed,'sh_mpcc_dro',step))
                if n is None or d is None:continue
                if n['true_mode']!=d['true_mode'] or n['scenario_count']!=d['scenario_count']:raise ValueError('invalid pairing')
                row=dict(paired,step=step,true_mode=n['true_mode'],before_switch=n['before_switch'],
                    p_d_nominal=n['p_d'],p_d_wdro=d['p_d'],q_d_wdro=d['q_d'],r_d_wdro=d['r_d'],
                    N_d_nominal=n['N_d'],N_d_wdro=d['N_d'],delta_N=d['N_d']-n['N_d'],
                    **{f'delta_U_k{k}':d[f'delta_U_k{k}'] for k in [1,2,3]})
                pairs.append(row);shared.append(row)
            # Fixed five-step pre-switch window, not a seed-selected interval.
            window=[r for r in shared if 5<=int(r['step'])<10]
            paired.update(matched_preswitch_decisions=len(window),
                mean_preswitch_delta_U1=mean(r['delta_U_k1'] for r in window) if window else '',
                mean_preswitch_delta_N=mean(r['delta_N'] for r in window) if window else '')
            seed_pairs.append(paired)
        rr=[r for r in seed_pairs if r['cell']==cell]
        n_only=sum(r['nominal_collision'] and not r['wdro_collision'] for r in rr)
        d_only=sum(r['wdro_collision'] and not r['nominal_collision'] for r in rr)
        discordant=n_only+d_only
        pvalue=min(1.,2*sum(math.comb(discordant,i) for i in range(min(n_only,d_only)+1))/2**discordant) if discordant else 1.
        initial=[r for r in all_modes if r['cell']==cell and r['solver_style']=='sh_mpcc' and r['step']=='0' and r['attempt']=='0']
        record=dict(cell=cell,seeds=len(rr),initial_nominal_U1=mean(r['U_p_k1'] for r in initial),
            nominal_only_collisions=n_only,wdro_only_collisions=d_only,exact_mcnemar_p=pvalue)
        for arm in ['nominal','wdro']:
            for metric in ['collision','refusal','completed']:
                count=sum(r[f'{arm}_{metric}'] for r in rr);lo,hi=wilson(count,len(rr))
                record[f'{arm}_{metric}_count']=count;record[f'{arm}_{metric}_ci95_low']=lo;record[f'{arm}_{metric}_ci95_high']=hi
        summary.append(record)
    write(root/'across_paired_decisions.csv',pairs)
    write(root/'paired_seed_outcomes.csv',seed_pairs)
    write(root/'transition_summary.csv',summary)
    return summary


def figures(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,marker in [('nominal','o'),('wdro','s')]:
        rr=sorted(summary,key=lambda r:r['initial_nominal_U1'])
        axes[0].plot([r['initial_nominal_U1'] for r in rr],[r[f'{arm}_collision_count']/r['seeds'] for r in rr],marker=marker,label=arm)
    axes[0].set(xlabel='Initial nominal P(Nd < 1)',ylabel='Observed collision fraction',title='100 paired seeds; fixed switch step 10');axes[0].legend()
    for r in summary:axes[0].annotate(r['cell'],(r['initial_nominal_U1'],r['nominal_collision_count']/r['seeds']),fontsize=7)
    for study,style in [('intervention-updated-belief','--'),('recoverability-updated-belief','-')]:
        rr=read(root/study/'confirmation_summary.csv')
        for metric in ['collision','refusal','path_completed']:
            selected=[r for r in rr if r['metric']==metric]
            if selected:axes[1].plot([int(r['target_nd']) for r in selected],[float(r['rate']) for r in selected],style,marker='.',label=f'{study}: {metric}')
    axes[1].set(xlabel='Dangerous samples per controlled decision',ylabel='Outcome fraction',title='Finite-window paired intervention');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'coverage_transition_and_intervention.svg');fig.savefig(root/'coverage_transition_and_intervention.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/persistent-coverage'));args=p.parse_args()
    summary=analyze(args.root);figures(args.root,summary)
    lines=['# Persistent coverage follow-up','',
        'All five transition cells use late-switch geometry, switch step 10, 120-step limit, two raw arms, and fresh paired seeds 1001–1100. Existing p50/S20/switch5 evidence remains in the earlier results; it is not pooled with the new switch10 control.','',
        '| Cell | Initial U(p;1,S) | Nominal collisions | WDRO collisions | Paired exact p |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['cell']} | {r['initial_nominal_U1']:.3f} | {r['nominal_collision_count']}/100 | {r['wdro_collision_count']}/100 | {r['exact_mcnemar_p']:.4g} |")
    lines += ['', 'These are descriptive paired comparisons; p-values are unadjusted for five comparisons. No transition-cell run completed the path. Collision reduction must not be presented as successful traversal.', '',
        '## CSVs', '',
        '- `across_every_solve.csv`: p_d, q_d, r_d, N_d and ΔU for k=1,2,3 at every recorded across-mode solve, including when true_mode is continue. Nominal risk scores are unavailable and remain blank, rather than borrowing the WDRO risk vector.',
        '- `across_paired_decisions.csv`: common decision indices, WDRO risk/mass change, paired counts and seed-level outcomes. Different closed-loop ego states mean these rows are descriptive, not controlled state comparisons.',
        '- `paired_seed_outcomes.csv`: fixed pre-switch steps 5–9 exposure summaries and seed-level outcomes. Matched denominators are explicit; repeated decisions are not independent outcome trials.',
        '- `transition_summary.csv`: counts, Wilson intervals, initial nominal undercoverage and exact paired McNemar tests.',
        '- `intervention-updated-belief/{pilot_outcomes,confirmation_outcomes,confirmation_summary,paired_outcomes}.csv`: original five-step count-controlled study.',
        '- `recoverability-updated-belief/{pilot_outcomes,pilot_selection_scores,confirmation_outcomes,confirmation_summary}.csv`: bounded separation/window refinement; confirmation files exist only if a recoverable design was found.', '',
        '## Intervention semantics', '',
        'Exact counts are imposed only during the predeclared window. Any refusal stops the run; unreached decisions are not imputed. Post-window draws use the same lagged belief updates as the production controller. Earlier frozen-belief pilots are retained under intervention/ and recoverability/ and are not pooled with these corrected-protocol runs. Every retained non-dangerous slot and each stationary-obstacle realization is paired, including after the window. Scenario banks use shared per-step seeds and the original zero-diffusion modes. Original IID sampling advances its RNG normally; the opt-in one-shot batch replacement changes only the requested uncertified experiment.', '',
        'Raw intervention mode_mechanism.csv reference_iid_probability columns retain nominal reference probabilities: they are NOT the law of the forced-count intervention. Do not apply the IID binomial formula to those batches. Use intervention.csv, actual sampled_count, and paired_slots.csv for treatment and pairing evidence. No scenario certificate is requested.', '',
        'The five-step reference study observed refusal in every arm and no collision or completion. It therefore did not identify a useful kappa. The failed recovery screen and all pilots are retained. Later refinements use separate pilot and confirmation seeds. Larger-count successes, if any, do not by themselves prove a threshold or a formal collision guarantee.', '',
        'Candidate implementation tested; controller behavior and scientific interpretation require user verification.']
    (args.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
