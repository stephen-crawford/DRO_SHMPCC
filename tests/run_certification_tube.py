#!/usr/bin/env python3
"""Paired fixed-tube pilot with frozen provenance, repeat checks and CSV summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from analyze_wdro_certificate import analyze
from report_tube_decomposition import report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, default=ROOT / 'build-tube/certification_tube_experiment')
    p.add_argument('--backend', type=Path, default=ROOT / 'build-tube/certificate_numeric_backend')
    p.add_argument('--seeds', type=int, default=3, help='number of paired seeds starting at 77 (each repeated twice)')
    p.add_argument('--cycles', type=int, default=4, help='maximum decisions per rollout; rejected plans terminate it')
    p.add_argument('--output', type=Path, required=True, help='new directory; existing data is never overwritten')
    p.add_argument('--scenarios', type=int, default=40,
                   help='fixed budget in every arm (default: 40, uncertified smoke test; use 1123 for the conservative-budget study)')
    args = p.parse_args()
    runner, backend = args.runner.resolve(), args.backend.resolve()
    if args.output.exists(): p.error('use a new output directory')
    if min(args.scenarios,args.seeds,args.cycles) < 1: p.error('counts must be positive')
    files = [runner, backend, Path(__file__), ROOT / 'tools/certification_tube_experiment.cpp',
             ROOT / 'include/certification_tube.hpp', ROOT / 'src/mpc_controller.cpp',
             ROOT / 'include/config.hpp', ROOT / 'tools/analyze_wdro_certificate.py',
             ROOT / 'tools/report_tube_decomposition.py', ROOT / 'tools/analyze_posterior_certificate.py']
    manifest = dict(sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    seeds=list(range(77,77+args.seeds)), repeats=2, radii=[0, .1, .2, .3, .4, .5, .75], cycles=args.cycles,
                    scenario_budget=args.scenarios, horizon=8, history={'safe':95, 'cut':5},
                    history_status='synthetic fixture counts, not observed iid evidence',
                    sample_reduction_enabled=False, status='running')
    args.output.mkdir(parents=True)
    def save():
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    with (args.output / 'experiment.log').open('w') as log:
        run = subprocess.run([str(runner), str((args.output / 'cycles.csv').resolve()), str(args.scenarios),str(args.seeds),str(args.cycles)],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    manifest['returncode'] = run.returncode
    if run.returncode:
        manifest['status'] = 'failed'; save(); return run.returncode
    rows = list(csv.DictReader((args.output / 'cycles.csv').open()))
    # Compare every observed field except timing and repeat identity.
    def signature(data, repeat):
        return [{k:v for k,v in row.items() if k not in ('repeat','solve_seconds')}
                for row in data if row['repeat'] == str(repeat)]
    geometry = list(csv.DictReader((args.output / 'disc_geometry.csv').open()))
    attempt_rows = list(csv.DictReader((args.output / 'attempts.csv').open()))
    support_settings = {tuple(r[k] for k in ('seed','repeat','arm','radius','cycle')):
                        (int(r['n_bar']),int(r['removal_budget'])) for r in attempt_rows}
    manifest['repeat_exact_except_timing'] = all(signature(data,0) == signature(data,1) for data in [rows,geometry])
    manifest['status'] = 'simulation_complete' if manifest['repeat_exact_except_timing'] else 'repeat_mismatch'
    save()
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('seed','repeat','arm','radius','cycle'))
        groups.setdefault(key, []).append(row)
    diagnostics = []
    for key, pair in groups.items():
        row = pair[0]
        if row['active'] != '1' or any(not m['q'] or not m['p'] for m in pair): continue
        # Only an offline hypothetical sample count. Actual budget stays fixed.
        snapshot = dict(obstacles=1, modes=[dict(mode=m['mode'], count=manifest['history'][m['mode']],
            p_hat=float(m['p']), q=float(m['q']), b=float(m['b'])) for m in pair],
            nonremoved_support_cap=support_settings[key][0],removal_budget=support_settings[key][1],
            b_justification='Fixed pre-sampling disc tube, held affine Gaussian fixture; uniform union bound. '
                'Synthetic counts and scenario theorem eligibility remain assumptions, not an issued certificate.')
        r = analyze(snapshot, backend)
        diagnostics.append(dict(zip(('seed','repeat','arm','radius','cycle'), key),
            accepted=row['success'], S_baseline=r['S_SH']['S'],
            S_hypothetical=r['S_WDRO']['S'] if r['S_WDRO'] else '',
            ratio=r['scenario_count_ratio'], epsilon_Q_max=r['epsilon_Q_max'],
            actual_S=row['sampled_scenarios'], beta_cp=r['beta_cp'], beta_cert=r['beta_cert'],
            combined_failure_budget=r['combined_failure_union_bound'], certificate_issued=False))
    def write(name, records):
        if not records: return
        with (args.output / name).open('w') as f:
            writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write('conditional_sample_counts.csv', diagnostics)
    summaries=[]
    for arm, radius in sorted(set((r['arm'], r['radius']) for r in rows)):
        cycles=[v[0] for v in groups.values() if v[0]['arm']==arm and v[0]['radius']==radius]
        active=[r for r in cycles if r['active']=='1']
        summaries.append(dict(arm=arm,radius=radius,decisions=len(cycles),
            accepted=sum(r['success']=='1' for r in cycles),tube_active=len(active),
            tube_rejected=sum(r['rejected']=='1' for r in cycles),
            accepted_outside_tube=sum(r['success']=='1' and float(r['max_displacement'])>float(radius) for r in active),
            repeat_exact_except_timing=manifest['repeat_exact_except_timing']))
    write('summary.csv', summaries)
    validation=report(args.output,args.output/'decomposition',backend)
    manifest['decomposition_validation']=validation['status']
    manifest['status'] = 'complete' if manifest['repeat_exact_except_timing'] and validation['status']=='PASS' else 'validation_failed'
    save()
    for row in summaries: print(row)
    return 0 if manifest['status']=='complete' else 1


if __name__ == '__main__': raise SystemExit(main())
