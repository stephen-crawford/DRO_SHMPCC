#!/usr/bin/env python3
"""Create the revised two-figure/two-table selection from existing current-formulation data."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(a.input / name)
    names = ['plot_true_law_validation.csv', 'plot_obstacle_scaling.csv', 'certificate_instances.csv',
             'heldout_decisions.csv', 'shift_rollouts.csv', 'table2_shift_summary.csv',
             'experiment_protocol.csv', 'radius_runtime_summary.csv']
    for name in names:
        shutil.copy2(a.input / name, a.output / name)
    risk = read(names[0])
    scaling = read(names[1])
    cert = read(names[2])
    held = read(names[3])
    runs = read(names[4])
    shifts = read(names[5])
    colors = {'nominal': '#2769A0', 'reweight_only': '#539843', 'full_proposed': '#D66D22', 'envelope_minimum': '#539843'}
    labels = {'nominal': 'Nominal', 'reweight_only': 'Reweight only', 'full_proposed': 'Full proposed',
              'envelope_minimum': 'Envelope minimum', 'envelope_optimal': 'Envelope optimum',
              'wasserstein_min_zeta': 'Wasserstein min. ζ'}
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    captions = []

    def save(fig, name, caption):
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(a.output / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        captions.append(dict(artifact=name, caption=caption))

    fig, ax = plt.subplots(figsize=(4.7, 3.25))
    for arm, marker, size in [('nominal', 'o', 6), ('reweight_only', 'x', 4), ('full_proposed', '^', 4)]:
        d = risk[risk.arm == arm].sort_values('delta')
        ax.plot(d.delta, d.mean_violation, color=colors[arm], marker=marker, markersize=size,
                markerfacecolor='none', lw=1.4, label=labels[arm])
        ax.plot(d.delta, d.median_pointwise_CP95_upper, color=colors[arm], ls=':', lw=1.2)
    ax.axhline(.05, color='black', ls='--', lw=.8, label='Target ε = 0.05')
    ax.set(xlabel='True-law parameter δ', ylabel='Held-out horizon violation probability',
           yscale='symlog', xlim=(-.025, .625))
    ax.set_yscale('symlog', linthresh=1e-5)
    ax.set_ylim(1e-5, .075)
    ax.set_xticks([0, .2, .4, .6])
    ax.grid(alpha=.2, lw=.5)
    ax.text(.03, .72, 'Dotted: median pointwise 95% CP upper endpoint\nNominal and reweight-only overlap',
            transform=ax.transAxes, fontsize=6.5)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.40), ncol=2)
    fig.subplots_adjust(bottom=.30, left=.16, right=.98, top=.98)
    save(fig, 'figure1_true_law_validation',
         'Current formulation; straight, one-obstacle, three-mode geometry; 30 paired seeds per true law and arm. '
         'Solid curves show equal-rollout-weighted mean violation estimates; dotted curves show the median of '
         'pointwise exact 95% Clopper–Pearson upper endpoints, not confidence bands on the mean. All dotted '
         'curves coincide. Calibration is redrawn from each true law: mismatch is finite-sample estimation error, '
         'not deliberately stale beliefs. Each returned plan uses 1000 independent held-mode Gaussian validation '
         'trajectories. The event is joint geometric horizon collision; it is not asserted equivalent to every '
         'recovery-dependent solver halfspace. The simulated plant redraws modes each step, unlike held-mode '
         'validation. These finite-sample observations do not prove the theorem or safety.')

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85))
    for ax, quantity, ylabel, title in zip(axes, ['zeta_joint', 'required_S'],
            ['Joint domination factor ζ', 'Required scenarios S'],
            ['(a) Multiplicative certificate factor', '(b) Implied scenario requirement']):
        for arm in ['envelope_minimum', 'full_proposed']:
            d = scaling[(scaling.arm == arm) & (scaling.quantity == quantity)].sort_values('V')
            ax.plot(d.V, d['median'], marker='o', ms=3, color=colors[arm], label=labels[arm])
            ax.fill_between(d.V, d.p05, d.p95, color=colors[arm], alpha=.16, lw=0)
        ax.set(xlabel='Number of obstacles V', ylabel=ylabel, yscale='log', title=title)
        ax.set_xticks(range(1, 7))
        ax.grid(alpha=.2, lw=.5)
    axes[0].legend()
    fig.tight_layout()
    save(fig, 'figure2_transfer_scaling',
         'Current-formulation offline scaling: 100 independent calibration seed sets, M=3 and g=100 per obstacle, '
         'V=1–6 with prefix pairing. Curves are medians; bands are empirical 5th–95th percentiles, not confidence '
         'intervals for the median. Per-obstacle beta=.05/6 is fixed across V. Nominal is omitted because its '
         'factor and required-S curves overlap full proposed. Required S uses epsilon=.05, beta_SH=.01, '
         'support cap 6 and removal budget 2. This is scenario sizing, not a controller-runtime sweep.')

    subset = cert[(cert.M == 5) & (cert.g == 100)]
    law_order = ['envelope_optimal', 'wasserstein_min_zeta', 'nominal', 'full_proposed']
    fields = ['normalized_zeta', 'zeta', 'required_S', 'risk_lift', 'dangerous_mass_change', 'old_new_envelope_ratio']
    t1 = subset.groupby('arm')[fields].median().reindex(law_order).reset_index()
    t1['instances'] = 100
    t1.to_csv(a.output / 'table1_sampling_law_ablation.csv', index=False)
    rows = []
    for arm in ['nominal', 'reweight_only', 'full_proposed']:
        r = runs[runs.arm == arm]
        d = held[(held.arm == arm) & (held.n_validation > 0)]
        q = risk[risk.arm == arm]
        s = shifts[shifts.arm == arm]
        rows.append(dict(arm=arm, attempted=len(r), completed=int(r.completed.sum()), collisions=int(r.collision.sum()),
                         mean_violation=d.groupby(['delta', 'seed']).violation_estimate.mean().mean(),
                         min_mean_across_delta=q.mean_violation.min(), max_mean_across_delta=q.mean_violation.max(),
                         max_across_delta_p95_CP_upper=q.p95_pointwise_CP95_upper.max(),
                         min_S=int(d.S.min()), max_S=int(d.S.max()),
                         min_shift_median_solve_ms=s.median_solve_ms_concurrent.min(),
                         max_shift_median_solve_ms=s.median_solve_ms_concurrent.max(),
                         evaluated_decisions=len(d), transferred_decisions=int(d.conditional_transfer.sum()),
                         transferred_estimate_above_epsilon=int(((d.conditional_transfer == 1) & (d.violation_estimate > .05)).sum())))
    t2 = pd.DataFrame(rows)
    t2.to_csv(a.output / 'table2_closed_loop_validation.csv', index=False)

    def table(name, headers, rows, widths, footnote, caption):
        fig, ax = plt.subplots(figsize=(10, .9 + .35 * len(rows)))
        ax.axis('off')
        tab = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc='center', bbox=[0, .23, 1, .77])
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        for (i, j), cell in tab.get_celld().items():
            cell.set_linewidth(.35)
            if i == 0:
                cell.set_facecolor('#E7EDF2')
                cell.set_text_props(weight='bold')
        ax.text(0, .035, footnote, fontsize=7, va='bottom', transform=ax.transAxes)
        save(fig, name, caption)
        # Native LaTeX tables for use in the paper, not rasterized tables.
        escape = lambda s: s.replace('ζ', r'$\zeta$').replace('Δ', r'$\Delta$').replace('δ', r'$\delta$').replace('−', '-').replace('–', '--').replace('\n', ' ')
        tex = ('% Requires graphicx.\n\\begin{table*}[t]\n\\centering\n\\caption{' + name.replace('_', ' ') + '}\n'
               '\\resizebox{\\textwidth}{!}{\\begin{tabular}{l' + 'r' * (len(headers)-1) + '}\n\\hline\n' +
               ' & '.join(escape(h) for h in headers) + r' \\' + '\n\\hline\n' +
               '\n'.join(' & '.join(escape(str(x)) for x in row) + r' \\' for row in rows) +
               '\n\\hline\n\\end{tabular}}\n\\par\\smallskip\n{\\footnotesize ' + escape(footnote) + '}\n\\end{table*}\n')
        (a.output / f'{name}.tex').write_text(tex)

    table('table1_sampling_law_ablation', ['Sampling law', 'ζ / minimum ζ', 'Median ζ', 'Median S', 'Risk lift', 'Δ dangerous mass'],
          [[labels[r.arm], f'{r.normalized_zeta:.3f}', f'{r.zeta:.3f}', f'{r.required_S:,.1f}',
            f'{r.risk_lift:.4f}', f'{r.dangerous_mass_change:+.4f}'] for r in t1.itertuples()],
          [.25, .16, .13, .14, .14, .18],
          'M = 5, g = 100; 100 calibration instances. Median integer counts can be half-integers.',
          'Current-formulation controlled sampling-law ablation, M=5 and g=100; medians over 100 calibration '
          'instances. The normalized factor is the median of per-instance ratios. Risk lift is the change '
          'in expected surrogate risk relative to nominal, not true collision probability. Median S=1930.5 '
          'is preserved rather than rounded to 1931. Equal medians alone do not establish instance-wise equality.')
    table('table2_closed_loop_validation', ['Controller', 'Completed', 'Collisions', 'Mean V', 'Range of means\nover δ',
                                          'Max p95\nCP upper', 'S', 'Shift median\nsolve [ms]'],
          [[labels[r.arm], f'{r.completed}/{r.attempted}', str(r.collisions), f'{r.mean_violation:.3g}',
            f'{r.min_mean_across_delta:.3g}–{r.max_mean_across_delta:.3g}', f'{r.max_across_delta_p95_CP_upper:.5f}',
            str(r.max_S) if r.min_S == r.max_S else f'{r.min_S}–{r.max_S}',
            f'{r.min_shift_median_solve_ms:.1f}–{r.max_shift_median_solve_ms:.1f}'] for r in t2.itertuples()],
          [.14, .10, .09, .10, .20, .10, .12, .15],
          'Four concurrent processes. Mean V weights rollouts equally; intervals are pointwise, not mission-wide.',
          'Current-formulation straight-geometry study: 120 runs per arm across four true laws, 360 total. '
          'Mean V weights each rollout equally. Range is across the four per-law means. Max p95 CP upper '
          'is the maximum over true laws of the 95th percentile of per-decision CP upper endpoints, not '
          'the maximum endpoint. S shows the observed range. Solve times are the range of per-shift pooled '
          'decision medians under four concurrent processes, excluding validation. No isolated real-time claim.')
    pd.DataFrame(captions).to_csv(a.output / 'captions.csv', index=False)

    # Check claims instance by instance, rather than infer them from equal rounded medians.
    pivot = subset.pivot(index='instance', columns='arm', values='zeta')
    np.testing.assert_allclose(pivot.full_proposed, pivot.nominal, rtol=1e-7, atol=1e-8)
    minimum = subset[subset.arm == 'nominal'].set_index('instance').minimum_zeta.reindex(pivot.index)
    np.testing.assert_allclose(pivot.wasserstein_min_zeta, minimum, rtol=1e-7, atol=1e-8)
    assert len(pivot) == 100 and len(runs) == 360 and runs.completed.sum() == 360 and runs.collision.sum() == 0
    assert (held.violation_estimate <= .05).all()
    assert t2.loc[t2.arm == 'full_proposed', 'transferred_decisions'].iloc[0] == 16193
    reduction = 1 - risk[risk.arm == 'full_proposed'].set_index('delta').mean_violation / risk[risk.arm == 'nominal'].set_index('delta').mean_violation
    pd.DataFrame({'delta': reduction.index, 'relative_reduction': reduction.values}).to_csv(a.output / 'relative_reduction.csv', index=False)
    (a.output / 'README.md').write_text('''# Current-formulation main-paper artifacts

Existing counterparts in ../acc-results-visuals-20261002:
- Figure 1: figure1_true_law_validation.pdf, panel (a). New version uses a single panel and solid mean curves.
- Figure 2: figure3_obstacle_scaling.pdf. New version omits overlapping nominal curves and uses Figure 2 numbering.
- Table I: table1_certificate_construction.pdf was related but lacked median zeta and risk lift. Created requested columns.
- Table II: table2_true_law_results.pdf had 12 shift-specific rows. Created the requested three-controller summary.
- Optional supplement already exists: figure4_calibration_evidence.pdf.

This directory contains exactly two main figures and two tables in PDF, SVG, PNG; native LaTeX tables;
underlying current-formulation CSVs; captions and reproducible reporting code. No historical outer-ball
rollout or archived-state ablation data are included. Existing packages remain unchanged.

Numerical precision and interpretation:
- Envelope-optimum and Wasserstein minimum median S is 1930.5, retained exactly rather than rounded to 1931.
- Full proposed S spans 3145–14828. Timing is measured under four concurrent processes.
- All 100 controlled M5/g100 instances were checked: proposed zeta equals nominal, and Wasserstein-minimum
  zeta equals the envelope minimum within rtol=1e-7, atol=1e-8. This is a numerical result for this set.
- No decision estimate exceeds .05. This does not prove safety, the theorem, or nominal-controller failure.
- 16193/16556 full-proposed evaluated decisions carry the conditional-transfer flag; none exceeds .05.
- Calibration is redrawn from each truth, not stale. Validation holds obstacle modes over a horizon;
  the plant redraws modes each step. Geometric collision validation is not claimed to match every solver halfspace.
- 5th–95th percentile scaling bands are empirical intervals, not confidence intervals for the median.
- The risk lift is a surrogate objective, not collision probability. Lower held-out estimates cannot be
  attributed to reweighting alone: nominal and reweight-only means coincide and full transfer uses more scenarios.

Change: added tools/make_current_results.py for reporting only; no controller code changes.
Mathematical impact: No mathematical guarantee or formulation was modified.
Validation: rendered from existing CSVs, checked 360 completions/zero collisions, per-instance factor
comparisons, recorded conditional-transfer count, and archive CRC. No new simulation or build was needed.
Command: MPLCONFIGDIR=/tmp/paper-mpl-cache python3 tools/make_current_results.py
  --input results/acc-results-visuals-20261002 --output results/acc-current-main-results-20261002
Status: reporting implementation executed; interpretation remains for author review.
''')
    shutil.copy2(Path(__file__), a.output / Path(__file__).name)
    (a.output / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n'
        for f in sorted(a.output.iterdir()) if f.is_file() and f.name != 'SHA256SUMS.txt'))
    archive = a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(f'Created {archive}; 4 artifacts; all numerical assertions passed.')


if __name__ == '__main__':
    main()
