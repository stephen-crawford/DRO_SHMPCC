#!/usr/bin/env python3
"""Export the follow-up figures and a transparent completion/result report."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from causal_link_analysis import read, write, paired_summary


def report(root, extension, frozen_root=None):
    frozen_root=frozen_root or extension
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir=extension/'figures';figure_dir.mkdir(exist_ok=True)
    prescreen=read(extension/'prescreen.csv')
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    scatter=axes[0].scatter([float(r['r']) for r in prescreen],[float(r['q'])-float(r['p']) for r in prescreen],c=[float(r['rho']) for r in prescreen])
    axes[0].set(xlabel='Frozen dangerous-mode risk',ylabel='q(d) − p(d)',title='Risk and allocation');fig.colorbar(scatter,ax=axes[0],label='Calibrated radius')
    for label in ['high','medium','near_zero']:
        folder=frozen_root/'frozen'/label
        draws=read(folder/'draws.csv');w=next(r for r in read(folder/'weights.csv') if r['mode']=='across')
        counts={law:mean(int(r['n_d']) for r in draws if r['law']==law) for law in ['nominal','wdro']}
        axes[1].scatter(float(w['q'])-float(w['p']),counts['wdro']-counts['nominal'],label=label)
        checks=read(folder/'coverage_check.csv')
        for law,marker in [('nominal','o'),('wdro','x')]:
            rr=[r for r in checks if r['law']==law]
            axes[2].scatter([float(r['theory']) for r in rr],[float(r['empirical']) for r in rr],marker=marker,label=f'{label}: {law}')
    axes[1].set(xlabel='q(d) − p(d)',ylabel='Observed mean Nd increase',title='Allocation and sample counts');axes[1].legend()
    axes[2].plot([0,1],[0,1],color='gray',linewidth=1);axes[2].set(xlabel='Binomial P(Nd < k)',ylabel='Observed frequency',title='Frozen coverage, k = 1, 2, 3');axes[2].legend(fontsize=7)
    estimates=read(frozen_root/'frozen/high/conditional_estimates.csv')
    for law in ['nominal','wdro']:
        for metric,style in [('collision','-'),('refusal','--')]:
            rr=[r for r in estimates if r['law']==law and r['metric']==metric and int(r['seeds'])]
            axes[3].plot([int(r['n_d']) for r in rr],[float(r['rate']) for r in rr],style,marker='.',label=f'{law}: {metric}')
    axes[3].set(xlabel='Verified initial Nd',ylabel='Conditional outcome rate',title='Outcome link: high-separation case',ylim=(-.04,1.04));axes[3].legend(fontsize=7)
    fig.tight_layout();fig.savefig(figure_dir/'risk_mass_coverage.svg');fig.savefig(figure_dir/'risk_mass_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharey=True)
    for col,label in enumerate(['high','medium','near_zero']):
        estimates=read(frozen_root/'frozen'/label/'conditional_estimates.csv')
        for row,metric in enumerate(['collision','refusal']):
            for law in ['nominal','wdro']:
                rr=[r for r in estimates if r['law']==law and r['metric']==metric and int(r['seeds'])]
                xx=[int(r['n_d']) for r in rr];yy=[float(r['rate']) for r in rr]
                axes[row,col].errorbar(xx,yy,yerr=[[max(0,y-float(r['ci95_low'])) for r,y in zip(rr,yy)],
                    [max(0,float(r['ci95_high'])-y) for r,y in zip(rr,yy)]],marker='o',label=law,capsize=2)
            axes[row,col].set(xlabel='Verified initial Nd',ylabel=f'P({metric} | forced mode, Nd)',title=label,ylim=(-.04,1.04))
            axes[row,col].legend()
    fig.suptitle('Conditional outcomes: 40-step frozen-law rollout; 95% Wilson intervals')
    fig.tight_layout();fig.savefig(figure_dir/'sample_count_outcomes.svg');fig.savefig(figure_dir/'sample_count_outcomes.png',dpi=180);plt.close(fig)
    timings=read(frozen_root/'scaling_measurements.csv');groups=defaultdict(list)
    for r in timings:groups[r['obstacles'],r['law']].append(r)
    summary=[]
    for (obstacles,law),rows in sorted(groups.items()):
        record=dict(obstacles=obstacles,law=law,seeds=len(rows))
        for field in ['trajectory_generation_ms','controller_trajectory_generation_ms','fixed_reference_constraint_ms','fixed_reference_raw_constraints','controller_constraint_ms','retained_facets','qp_constraints_max','qp_ms','solve_ms','success']:
            values=[float(r[field]) for r in rows if r.get(field,'')!='']
            record[field+'_mean']=mean(values) if values else ''
        summary.append(record)
    write(extension/'scaling_summary.csv',summary)
    paired=paired_summary(root)
    lines=['# Targeted causal-link follow-up', '',
        'The complete v3 broad matrix remains frozen (396/396 cells). Original selection seeds 77–86 are excluded from the 100-seed confirmation runs (87–186).', '',
        '## Paired collision results', '', '| Case | Paired seeds | Nominal only | WDRO only | Both | Exact McNemar p |', '|---|---:|---:|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['case']} | {r['paired_seeds']} | {r['nominal_only_collision']} | {r['wdro_only_collision']} | {r['both_collision']} | {r['exact_mcnemar_p']:.5g} |")
    checks=read(frozen_root/'frozen_coverage_checks.csv')
    outcomes=[]
    for label in ['high','medium','near_zero']:
        outcomes.extend(read(frozen_root/'frozen'/label/'conditional.csv'))
    confirmations=read(frozen_root/'braking_confirmation.csv')
    lines += ['', '## Measured results and screening failures', '',
        f'- Frozen coverage: {sum(int(r["coverage_check_pass"]) for r in checks)}/{len(checks)} checks passed across 30,000 sampled sets (5,000 per law and geometry).',
        f'- Conditional runs: {len(outcomes)} count-selected rollouts, {sum(int(r["collision"]) for r in outcomes)} collisions. A collision benefit from increased Nd has NOT been demonstrated; the measured differences concern refusal/survival.',
        '- Commitment-relative screening failed: only 9/50 reference runs reached tc; all-planned reach rates were 14–18%, below the required 95%. These comparisons must not support a late-switch effectiveness claim.',
        '- Eight initial braking designs failed the screen. Four prospectively defined refinements used fresh pilot seeds; the first qualifying design was selected before confirmation.',
        '', '| Braking confirmation arm | Seeds | Reach switch | Collision | Refusal |', '|---|---:|---:|---:|---:|']
    for arm in ['sh_mpcc','sh_mpcc_dro']:
        rr=[r for r in confirmations if r['arm']==arm]
        lines.append(f"| {arm} | {len(rr)} | {sum(int(r['switch_reached']) for r in rr)} | {sum(int(r['collision']) for r in rr)} | {sum(r['termination']=='no_admissible_control' for r in rr)} |")
    lines += ['', 'Scaling uses 50 seeds per arm at each of 2, 3, and 4 obstacles. See scaling_summary.csv for all acceptance rates as well as timings. Extra nominal roughly doubles sampling time, but total solve-time differences are modest; these data do not justify a broad efficiency claim.']
    lines += ['', f'Completed frozen/measurement refinement: `{frozen_root}`.', '', 'These p-values are descriptive, unadjusted for multiple comparisons. Refusal is a separate outcome; absence of collision does not establish successful navigation.', '',
        '## Experimental checks', '',
        '- Geometry/calibration selection was saved before conditional outcomes. High/medium separation ranks use ΔU2; near-zero uses minimum absolute separation.',
        '- Calibration uses unchanged radius estimation and unchanged smoothing. Raw history proportions are constant; the actual posterior-predictive p changes slightly with g and is logged.',
        '- Frozen sampling holds the state, belief and q fixed. Conditional runs select seeds by initial Nd before observing outcomes; actual controller counts must match. Later decisions retain the frozen law, but their Nd is not fixed.',
        '- Per-count outcome intervals use Wilson 95% intervals. Gamma estimates reweight count strata using binomial mass; missing strata remain explicit partial-identification uncertainty. Simultaneous gamma intervals are approximate, not theorem certificates.',
        '- Commitment tc comes from a separate nominal no-switch reference, shared between arms. Attrition is reported against all planned seeds, including references that never commit.',
        '- Braking candidates qualify only with ≥95% switch reach in both arms and 20–50% nominal collision-or-refusal. A null selection means no confirmation was run.',
        '- Scaling generation and raw constraint timings are fixed-reference microbenchmarks using production functions. Controller constraint/solve times and retained facets come from actual solves; QP row counts are parsed from actual solver logs. Live sampler timing is additionally logged as controller_trajectory_generation_ms, excluding DRO evaluation. Raw constraint microbenchmarks include k=0; actual per-group raw/facet counts are in scaling_constraint_groups.csv.',
        '- Plant collision checks occur at discrete simulation steps. These experiments request no scenario certificate.', '',
        '## Artifacts', '',
        '- `prescreen.csv`, `selection_before_outcomes.json`',
        '- `frozen/*/{weights,transport,draws,conditional,conditional_modes,coverage_check,conditional_estimates,gamma_estimates}.csv`',
        '- `commitment_schedule_before_outcomes.csv`, `commitment_outcomes.csv`, `commitment_attrition.csv`',
        '- `braking_pilot_outcomes.csv`, `braking_designs.csv`, `braking_selection_before_confirmation.json`',
        '- `scaling_measurements.csv`, `scaling_summary.csv`',
        '- `figures/risk_mass_coverage.svg`, `figures/sample_count_outcomes.svg`', '',
        'Candidate experiment implementation tested; simulation behavior and interpretation require user verification.']
    (extension/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/causal-link-followup'))
    p.add_argument('--extension',type=Path,default=Path('results/causal-link-followup/extension-v2'))
    p.add_argument('--frozen-root',type=Path)
    args=p.parse_args();report(args.root,args.extension,args.frozen_root)
