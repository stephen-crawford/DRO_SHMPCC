#!/usr/bin/env python3
"""Plot all complete seed groups, without selecting favorable outcomes."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    manifest = json.loads((args.input/'matrix.json').read_text())
    profiles = manifest['settings']['mismatch_profiles']
    # A one-dimensional boost plot is only meaningful with other shift axes held fixed.
    other = [{k: v for k, v in p.items() if k not in ('name', 'shift_boost')} for p in profiles]
    if any(p != other[0] for p in other) or len({p.get('shift_boost', 0) for p in profiles}) != len(profiles):
        parser.error('plot requires unique shift_boost values and identical other profile parameters')
    boost = {p['name']: p.get('shift_boost', 0) for p in profiles}
    cases = {c['case']: c for c in manifest['cases']}
    methods = manifest['settings']['solver_styles'] + manifest['settings'].get('additional_controllers', [])
    comparisons = defaultdict(list)
    for r in json.loads((args.input/'all_comparisons.json').read_text()):
        comparisons[r['pair'], str(r['seed'])].append(r)
    with (args.input/'mechanism_per_seed.csv').open() as source:
        records = list(csv.DictReader(source))
    groups = defaultdict(dict)
    for r in records:
        groups[r['pair'], r['seed']][r['controller']] = r
    complete = {}
    for key, group in groups.items():
        pairs = comparisons[key]
        if set(group) == set(methods) and len(pairs) == len(methods)*(len(methods)-1)//2 and all(p['status'] == 'OK' for p in pairs):
            complete[key] = group
    if not complete:
        parser.error('no complete, pairing-checked seed groups')
    aggregates = defaultdict(list)
    for group in complete.values():
        for method, r in group.items():
            c = cases[r['case']]
            setup = f"{c['environment']}_o{c['obstacles']}_c{c['classes']}_m{c['modes_per_class']}_s{c.get('scenario_budget', 'auto')}"
            aggregates[setup, r['profile'], method].append(r)
    plotted = []
    for (setup, profile, method), rr in sorted(aggregates.items()):
        total = lambda key: sum(float(r[key]) for r in rr)
        plotted.append(dict(setup=setup, profile=profile, shift_boost=boost[profile], controller=method,
            matched_seeds=len(rr), collision_rate=total('collision')/len(rr),
            safe_completion_rate=total('safe_completion')/len(rr),
            focus_omission_rate=1-total('focus_represented')/total('focus_checks'),
            mean_cycle_ms=total('cycle_total_ms')/total('decisions')))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/dro-comparison-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True)
    with (args.output/'plotted_rates.csv').open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(plotted[0]))
        writer.writeheader(); writer.writerows(plotted)
    labels = {'sh_mpcc': 'Nominal S', 'sh_mpcc_dro': 'WDRO S',
              'sh_mpcc_extra': 'Nominal 2S', 'sh_mpcc_resample': 'Nominal → nominal',
              'sh_mpcc_dro_fallback': 'WDRO → nominal'}
    for setup in sorted({r['setup'] for r in plotted}):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout='constrained')
        fields = [('collision_rate', 'Collision (%)', 100),
                  ('safe_completion_rate', 'Collision-free completion (%)', 100),
                  ('focus_omission_rate', 'Focus-mode omission (%)', 100),
                  ('mean_cycle_ms', 'Mean cycle time (ms)', 1)]
        for ax, (field, title, scale) in zip(axes, fields):
            for method in methods:
                rr = sorted((r for r in plotted if r['setup'] == setup and r['controller'] == method), key=lambda r:r['shift_boost'])
                ax.plot([r['shift_boost'] for r in rr], [scale*r[field] for r in rr], marker='o', label=labels.get(method, method))
            ax.set(xlabel='Configured plant override probability', ylabel=title)
            ax.grid(alpha=.2)
            if scale == 100: ax.set_ylim(-2, 102)
        axes[-1].legend(fontsize=7)
        counts = ', '.join(f"{r['shift_boost']:g}: n={r['matched_seeds']}" for r in plotted if r['setup'] == setup and r['controller'] == methods[0])
        fig.suptitle(setup + '\nPaired seeds per boost: ' + counts, fontsize=10)
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(args.output/f'{setup}.{extension}', dpi=180)
        plt.close(fig)
    (args.output/'README.txt').write_text(
        'Descriptive curves; all controllers use the same complete seed groups at each setup/profile.\n'
        'See plotted_rates.csv for denominators. Missing/error groups are excluded, never counted safe.\n'
        'The x axis is configured plant override probability, not measured belief divergence.\n'
        'Omission is per obstacle/first attempt over observed decisions; early stops alter exposure.\n'
        'Timing includes opt-in diagnostics. These curves are not confidence bounds.\n'
        'Use primary_summary.csv for per-comparison paired inference and missing-pair counts.\n')
    print(f'Wrote {len(plotted)} rate rows to {args.output}')


if __name__ == '__main__':
    main()
