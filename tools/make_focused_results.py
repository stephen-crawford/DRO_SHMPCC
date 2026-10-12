#!/usr/bin/env python3
"""Render the requested two figures and two tables from the historical CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for src in args.input.glob('*.csv'):
        shutil.copy2(src, out / src.name)
    read = lambda name: pd.read_csv(args.input / name)
    decisions = read('ablation_decisions.csv')
    rollouts = read('ablation_rollouts.csv')
    progress = read('figure1_progress_medians.csv')
    latency = read('figure1_latency.csv')
    mechanism = read('figure2_mechanism.csv').sort_values('step')
    arms = ['baseline', 'reweight_only', 'full_transfer']
    labels = {'baseline': 'Baseline', 'reweight_only': 'Reweight only', 'full_transfer': 'Full transfer'}
    colors = {'baseline': '#24689E', 'full_transfer': '#D66B24'}
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                         'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    captions = []

    def save(fig, name, caption):
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        captions.append({'artifact': name, 'caption': caption})

    # Pool logged decisions, rather than averaging per-rollout quantiles.
    rows1, rows2 = [], []
    for v, m in [(1, 2), (2, 3)]:
        for arm in arms:
            r = rollouts[(rollouts.obstacles == v) & (rollouts.modes == m) & (rollouts.arm == arm)]
            d = decisions[(decisions.obstacles == v) & (decisions.modes == m) & (decisions.arm == arm)]
            rows1.append(dict(obstacles=v, modes=m, arm=arm, attempted=len(r),
                              usable=int((r.status == 'OK').sum()), errors=int((r.status != 'OK').sum()),
                              completed=int(r.completion.sum()), completion_known=int(r.completion.notna().sum()),
                              collisions=int(r.collision.sum()), collision_known=int(r.collision.notna().sum()),
                              decisions=len(d), median_solve_ms=d.solve_ms.median(),
                              p95_solve_ms=d.solve_ms.quantile(.95), median_S_over_S0=d.S_over_S0.median()))
            if arm == 'full_transfer':
                assert d.zeta.notna().all() and d.transfer_conditions_met.notna().all()
                rows2.append(dict(obstacles=v, modes=m, decisions=len(d),
                                  median_zeta=d.zeta.median(), p95_zeta=d.zeta.quantile(.95), max_zeta=d.zeta.max(),
                                  median_S_over_S0=d.S_over_S0.median(), p95_S_over_S0=d.S_over_S0.quantile(.95),
                                  transfer_conditions_met_count=int(d.transfer_conditions_met.sum()),
                                  transfer_conditions_met_fraction=d.transfer_conditions_met.mean(),
                                  max_S=int(d.S.max()), S0=int(d.S0.iloc[0])))
    table1, table2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    table1.to_csv(out / 'table1_controller_ablation.csv', index=False)
    table2.to_csv(out / 'table2_certificate_scalability.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ecdf, figstats = [], []
    for arm in ['baseline', 'full_transfer']:
        d = progress[progress.arm == arm].sort_values('step')
        axes[0].plot(d.step, d.median_normalized_progress, color=colors[arm],
                     ls='-' if arm == 'baseline' else '--', lw=1.6, label=labels[arm])
        values = np.sort(latency.loc[latency.arm == arm, 'solve_ms'].to_numpy())
        probability = np.arange(1, len(values) + 1) / len(values)
        axes[1].step(values, probability, where='post', color=colors[arm], lw=1.4, label=labels[arm])
        ecdf.extend(dict(arm=arm, solve_ms=x, ecdf=y) for x, y in zip(values, probability))
        figstats.append(dict(arm=arm, final_carried_median_progress=d.median_normalized_progress.iloc[-1],
                             median_solve_ms=np.median(values), p95_solve_ms=np.quantile(values, .95),
                             timed_decisions=len(values), paired_rollouts=9))
    axes[0].set(xlabel='MPC step', ylabel='Median normalized path progress',
                title='(a) Closed-loop progress', xlim=(0, 200), ylim=(0, 1.01))
    axes[1].set(xlabel='Logged solve time [ms]', ylabel='Empirical CDF',
                title='(b) Computational overhead', xscale='log', ylim=(0, 1.01))
    for ax in axes:
        ax.grid(alpha=.2, lw=.5)
        ax.legend(loc='lower right')
    fig.tight_layout()
    cap1 = ('Closed-loop effect of distribution transfer in the historical one-obstacle, two-mode experiments. '
            '(a) Median normalized path progress for nine paired runs per arm (straight, S-curve and four-way intersection, '
            'seeds 77–79). Completed terminal progress is carried to step 200; carried points are not additional observations. '
            '(b) ECDF of the corresponding pooled logged per-step solve times. Progress is similar and full transfer has '
            'higher observed computational cost. This selected cohort differs from the complete 11-run-per-arm cohort '
            'in Table I. These runs used the earlier outer-ball-envelope formulation, not the revised CP-envelope implementation.')
    save(fig, 'figure1_behavior_overhead', cap1)
    pd.DataFrame(ecdf).to_csv(out / 'figure1_latency_ecdf.csv', index=False)
    pd.DataFrame(figstats).to_csv(out / 'figure1_statistics.csv', index=False)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.35), sharex=True)
    for ax, field, label in zip(axes, ['rho_max', 'zeta', 'S_over_S0'],
                               [r'Transport budget $\rho_t$', r'Domination factor $\zeta_t$', r'Scenario ratio $S_t/S_0$']):
        ax.plot(mechanism.step, mechanism[field], color=colors['full_transfer'], lw=1.25)
        ax.set_ylabel(label)
        ax.grid(alpha=.2, lw=.5)
        ax.set_xlim(0, int(mechanism.step.max()))
    for ax, letter in zip(axes, 'abc'):
        ax.text(.96, .85, f'({letter})', transform=ax.transAxes, ha='right')
    axes[-1].set_xlabel('MPC step')
    fig.tight_layout(h_pad=.45)
    cap2 = ('Evolution of the transfer mechanism in the historical full-transfer S-curve rollout, one obstacle, '
            'two modes, seed 77. The effective calibrated transport budget declines over the rollout, alongside the '
            'domination factor and Safe-Horizon scenario-count ratio. Both calibration information and the '
            'geometry-dependent transport costs can influence the budget; the plotted CSV alone does not isolate their '
            'contributions. These results use the earlier outer-ball-envelope formulation.')
    save(fig, 'figure2_certificate_mechanism', cap2)
    pd.DataFrame([dict(quantity=f, first=mechanism[f].iloc[0], last=mechanism[f].iloc[-1],
                       median=mechanism[f].median()) for f in ['rho_max', 'zeta', 'S_over_S0', 'solve_ms']]).to_csv(
                           out / 'figure2_statistics.csv', index=False)

    def table_image(name, headers, rows, caption, footnote, widths):
        fig, ax = plt.subplots(figsize=(7.16, .85 + .30 * len(rows)))
        ax.axis('off')
        table = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc='center',
                         bbox=[0, .19, 1, .8])
        table.auto_set_font_size(False)
        table.set_fontsize(6.7)
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(.35)
            if i == 0:
                cell.set_facecolor('#E7EDF2')
                cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F6F8FA')
        ax.text(0, .03, footnote, fontsize=6.3, transform=ax.transAxes, va='bottom')
        save(fig, name, caption)

    display1, latex1 = [], []
    for row in rows1:
        completed = f"{row['completed']}/{row['completion_known']}" + ('*' if row['errors'] else '')
        setting = f"{row['obstacles']} obs., {row['modes']} modes"
        cells = [setting, labels[row['arm']], completed, f"{row['collisions']}/{row['collision_known']}",
                 f"{row['median_solve_ms']:.2f}", f"{row['p95_solve_ms']:.2f}", f"{row['median_S_over_S0']:.2f}"]
        display1.append(cells)
        latex1.append(' & '.join(cells) + r' \\')
    cap_table1 = ('Controller ablation for the historical formulation. The one-obstacle block aggregates straight, '
                  'S-curve and four-way intersection; the two-obstacle block is S-curve. Timing quantiles and scenario '
                  'ratios pool logged decisions. Completion and collision denominators count runs with known outcomes. '
                  'The two-obstacle reweight-only arm had five attempts: four usable runs and one execution error with '
                  'unknown completion/collision outcomes. No failed run is treated as collision-free. Similar baseline '
                  'and reweight-only total timings do not independently measure allocation overhead.')
    table_image('table1_controller_ablation', ['Setting', 'Arm', 'Completed', 'Collisions', 'Median\n[ms]', 'p95\n[ms]', 'Median\nS/S₀'],
                display1, cap_table1, '* 4 usable runs from 5 attempts; 1 execution error. Quantiles pool logged decisions.',
                [.18, .18, .13, .12, .13, .13, .13])
    display2, latex2 = [], []
    for row in rows2:
        cells = [f"{row['obstacles']} obs., {row['modes']} modes", f"{row['median_zeta']:.3f}",
                 f"{row['p95_zeta']:.3f}", f"{row['max_zeta']:.2f}", f"{row['median_S_over_S0']:.3f}",
                 f"{row['p95_S_over_S0']:.3f}", f"{100 * row['transfer_conditions_met_fraction']:.1f}%"]
        display2.append(cells)
        latex2.append(' & '.join(cells).replace('%', r'\%') + r' \\')
    cap_table2 = ('Certificate scalability in the historical full-transfer runs. Statistics pool all logged decisions '
                  'in each setting (1734 and 973, respectively). Transfer conditions met is the fraction of logged '
                  'decisions whose transfer_conditions_met flag is one; it is not an empirical safety probability. '
                  'Obstacle count, mode count and geometry differ between blocks, so this is a complexity comparison, '
                  'not an isolated causal obstacle-count sweep. Maximum S in the harder case is 458859, versus S0=1123.')
    table_image('table2_certificate_scalability', ['Setting', 'Median ζ', 'p95(ζ)', 'Max ζ', 'Median\nS/S₀', 'p95\n(S/S₀)', 'Transfer\nconditions met'],
                display2, cap_table2, 'Full-transfer decisions: n = 1,734 / 973. Condition flags are not safety probabilities.',
                [.20, .12, .12, .12, .13, .13, .18])
    for name, title, header, body, note in [
        ('table1_controller_ablation', 'Controller ablation (historical formulation)',
         r'Setting & Arm & Completed & Collisions & Median [ms] & $p_{95}$ [ms] & Median $S/S_0$ \\',
         latex1, r'*Four usable runs from five attempts; one execution error has unknown outcomes. Quantiles pool logged decisions.'),
        ('table2_certificate_scalability', 'Certificate scalability (historical full-transfer runs)',
         r'Setting & Median $\zeta$ & $p_{95}(\zeta)$ & Max $\zeta$ & Median $S/S_0$ & $p_{95}(S/S_0)$ & Conditions met \\',
         latex2, r'Condition flags are not safety probabilities. The two settings contain 1,734 and 973 logged decisions.')]:
        tex = ('% Requires graphicx; uses standard LaTeX table rules.\n'
               '\\begin{table*}[t]\n\\centering\n\\caption{' + title + '}\n\\label{tab:' + name + '}\n'
               '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llrrrrr}\n\\hline\n' + header + '\n\\hline\n' +
               '\n'.join(body) + '\n\\hline\n\\end{tabular}}\n\\par\\smallskip\n{\\footnotesize ' + note + '}\n\\end{table*}\n')
        (out / f'{name}.tex').write_text(tex)
    pd.DataFrame(captions).to_csv(out / 'captions.csv', index=False)

    draft = r'''\section{Results}
\paragraph{Experimental setup.}
The results below describe the historical outer-ball-envelope implementation;
they should not be attributed to the revised CP-envelope implementation.
We compare nominal sampling (baseline), probability reweighting with the nominal
scenario budget (reweight only), and reweighting with the transferred scenario
requirement (full transfer). The one-obstacle, two-mode regime includes straight,
S-curve, and four-way-intersection geometries, with 11 attempted runs per arm.
The two-obstacle, three-mode regime uses an S-curve, with five attempts per arm.
Timing and certificate quantiles pool logged decisions and are descriptive;
successive decisions within a rollout are not independent replicates.

\paragraph{Ablation and closed-loop behavior.}
Table~\ref{tab:table1_controller_ablation} shows nearly equal baseline and
reweight-only median solve times in the one-obstacle regime (13.98 and 13.99 ms).
Full transfer raises the median to 16.68 ms and the 95th percentile from 46.21
to 70.58 ms; all three arms complete 10 of 11 runs. Figure~\ref{fig:behavior}
uses the original nine paired runs per arm: the terminal-carried median normalized
progress is 0.9517 for baseline and 0.9533 for full transfer, while median solve
time rises from 13.97 to 16.43 ms (95th percentile: 46.05 to 70.23 ms).
These observations support similar progress in the tested simple cases and
associate most of the observed extra cost with enabling certificate tightening.
They do not establish zero allocation overhead or statistical equivalence.
The harder reweight-only arm has one execution error; its completion and
collision outcomes are unknown and excluded from the four-usable-run denominator.

\paragraph{Transfer mechanism.}
Figure~\ref{fig:mechanism} shows the representative full-transfer S-curve run
(one obstacle, two modes, seed 77). The effective calibrated transport budget
decreases from 0.0455 to 0.0055, while the domination factor decreases from
3.95 to 1.19 and scenario inflation from 5.02 to 1.23. The median factor is
1.259, median scenario ratio 1.319, and median solve time 15.85 ms.
The formulation links a larger $\zeta_t$ to a smaller transferred target
$\epsilon/\zeta_t$ and a larger scenario requirement. The observed co-evolution
is consistent with that mechanism; the CSV does not isolate calibration growth
from changes in geometry-dependent transport costs.

\paragraph{Scalability limitation.}
Table~\ref{tab:table2_certificate_scalability} shows the median domination
factor increasing from 1.268 to 3.501 between the two regimes, with its 95th
percentile increasing from 2.035 to 23.413. Median scenario inflation rises
from 1.330 to 4.370, and the harder case reaches $S_{\max}=458{,}859$
against $S_0=1123$. Its full-transfer median and 95th-percentile solve times
are 126.13 and 2202.68 ms. These costs expose a substantial practical limitation
of the transferred certificate in the tested configuration, consistent with
the compounding construction $\zeta_t=\prod_v\zeta_t^v$.
Because obstacle count, mode count and geometry change together, these two
regimes do not isolate the effect of obstacle count alone. The logged transfer
conditions hold on 90.2\% and 71.7\% of decisions, respectively; these fractions
are diagnostic acceptance rates, not safety probabilities. Zero observed
collisions in usable runs does not establish improved safety under distribution
shift. The separate new held-out study is outside these four historical artifacts.
'''
    (out / 'results_section.tex').write_text(draft)
    (out / 'figure_includes.tex').write_text(r'''% Requires \usepackage{graphicx}; see captions.csv for complete cohort details.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure1_behavior_overhead.pdf}
\caption{Closed-loop behavior and computational overhead in the historical
one-obstacle, two-mode experiments. Nine paired runs per arm; completed terminal
progress is carried forward. The timing ECDF pools logged decisions.}
\label{fig:behavior}
\end{figure*}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figure2_certificate_mechanism.pdf}
\caption{Effective transport budget, domination factor, and scenario inflation
in the historical full-transfer S-curve rollout (one obstacle, two modes, seed 77).
The plot does not separate calibration and geometry effects.}
\label{fig:mechanism}
\end{figure}
''')
    # Check summaries against the independently supplied requested rounded values.
    np.testing.assert_allclose(table1.median_solve_ms, [13.98, 13.99, 16.68, 27.84, 27.65, 126.13], atol=.005, rtol=0)
    np.testing.assert_allclose(table1.p95_solve_ms, [46.21, 46.15, 70.58, 89.27, 87.49, 2202.68], atol=.005, rtol=0)
    assert table1.attempted.sum() == 48 and table1.errors.sum() == 1
    assert table2.max_S.max() == 458859
    assert len(latency.groupby(['pair', 'seed', 'arm'])) == 18
    assert len(mechanism) == 160
    assert len(captions) == 4
    (out / 'README.md').write_text('''# Focused Results package: two figures and two tables

Figure 1 is a two-panel progress/latency figure; Figure 2 is a three-panel vertical mechanism figure.
Table I is the controller ablation; Table II is certificate scalability. Each is exported as PDF,
editable SVG and 300-dpi PNG. The table CSVs retain full precision and explicit denominators.
LaTeX tables, figure includes and a four-paragraph Results-section draft are included.
Use table* and figure* for full-width artifacts in the two-column paper; Figure 2 is single-column.

All eight supplied CSVs are copied unchanged. Derived CSVs contain the table values, plotted ECDF,
figure statistics and captions. Quantiles use NumPy/pandas linear interpolation on pooled decisions.
Figure 1 uses nine original paired runs per arm; Table I uses all 11 simple-case attempts per arm
and all five harder-case attempts per arm. One harder-case reweight-only execution error remains
explicit; its unknown outcomes are not converted into zero collisions or successful completion.

These are historical outer-ball-envelope results. No new simulation was needed for this request.
They do not describe the revised CP-envelope implementation, and are not mixed with the later
held-out experiment. The prior comprehensive ZIP remains unchanged. See captions.csv for limits
on causal, safety, timing and scalability interpretations. The prose is a draft for author review.

## Change made
File: tools/make_focused_results.py; section: reporting/export pipeline.
Added a standalone renderer and aggregator for exactly the four requested artifacts.
This keeps the existing simulation and reporting outputs unchanged and is independently removable.

## Mathematical impact
No mathematical guarantee or formulation was modified.

## Validation performed
Build: not applicable; no controller code changed.
Run: MPLCONFIGDIR=/tmp/paper-mpl-cache python3 tools/make_focused_results.py
  --input results/acc-paper-results-csvs-20261002 --output results/acc-focused-results-20261002
Checks: 48 attempts, one error; all six timing rows match the requested values to rounding;
18 paired Figure 1 rollout records; 160 Figure 2 decisions; maximum S=458859.
Assertions check these against source data. SHA256SUMS.txt records artifact hashes.
The supplied LaTeX snippets compiled together using pdflatex in a two-column article
wrapper (two passes, three pages, no LaTeX warnings). All four rendered artifacts were
visually inspected. Matplotlib reported unavailable optional Axes3D support; these
artifacts use only 2D plotting and exported successfully.

## Current status
Candidate reporting implementation executed and checked; figures and interpretation require user review.
''')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    provenance = [dict(file=p.name, source=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in sorted(args.input.glob('*.csv'))]
    pd.DataFrame(provenance).to_csv(out / 'input_provenance.csv', index=False)
    (out / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n'
                                             for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt'))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for source in args.input.glob('*.csv'):
            assert z.read(source.name) == source.read_bytes()
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, artifacts=len(captions),
                          attempted_runs=int(table1.attempted.sum()), execution_errors=int(table1.errors.sum())), indent=2))


if __name__ == '__main__':
    main()
