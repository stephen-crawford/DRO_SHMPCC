#!/usr/bin/env python3
"""Read-only, cohort-separated temporal audit; preserves stopped-run missingness."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from analyze_control_response import category, flatten, seed_events, group_summary
from causal_link_analysis import read, write
from report_coverage_replication import mcnemar

PRIMARY_FIELDS = [
    ('nominal_first_brake', '^', 'Nominal first brake'),
    ('wdro_first_brake', 'v', 'WDRO first brake'),
    ('first_extra_wdro_across_sample', 's', 'First extra WDRO across sample'),
    ('first_material_control_divergence', 'x', 'First material command difference'),
    ('first_control_availability_divergence', '+', 'First availability difference')]
FLAGS = ['positive_chain_at_first_response', 'first_extra_equals_first_response',
         'first_extra_equals_first_material', 'first_extra_before_first_response']


def extend_event(event):
    e = dict(event)
    response = e['first_decision_response_divergence']
    extra = e['first_extra_wdro_across_sample']
    e['positive_chain_at_first_response'] = (int(
        e['extra_sample_at_first_response'] == 1 and
        float(e.get('first_response_r_d') or 0) > 0 and
        float(e.get('first_response_delta_q') or 0) > 0)
        if response is not None else None)
    for name, target in [('first_response', response),
                         ('first_material', e['first_material_control_divergence'])]:
        e['first_extra_equals_' + name] = int(extra == target) if target is not None else None
    e['first_extra_before_first_response'] = int(extra is not None and extra < response) if response is not None else None
    return e


def summarize_cohort(events):
    result = group_summary(events)
    for row in result:
        subset = [e for e in events if row['group'] == 'all_seeds' or e['group'] == row['group']]
        for field in FLAGS:
            values = [e[field] for e in subset if e[field] is not None]
            row[field + '_yes'] = sum(values)
            row[field + '_eligible'] = len(values)
    return result


def timeline(rows, path, title, supplementary=False):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = PRIMARY_FIELDS if not supplementary else [
        ('nominal_first_across_sample', 'o', 'Nominal first across sample'),
        ('wdro_first_across_sample', 's', 'WDRO first across sample')] + PRIMARY_FIELDS
    fig, ax = plt.subplots(figsize=(12, max(5, len(rows)*.35+2)))
    for j, (field, marker, label) in enumerate(fields):
        points = [(r[field], i+(j-(len(fields)-1)/2)*.10) for i, r in enumerate(rows) if r.get(field) is not None]
        ax.scatter([p[0] for p in points], [p[1] for p in points], marker=marker, label=label, s=35)
    ax.axvline(10, color='gray', linestyle='--', label='Plant switch')
    ax.set_yticks(range(len(rows)), [str(r['seed']) for r in rows])
    ax.set(xlabel='Decision step (0.1 s per step)', ylabel='Seed', title=title)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.13), ncol=2)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    for suffix in ['png', 'svg']:
        fig.savefig(path.with_suffix('.'+suffix), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/control-response-expansion'))
    args = parser.parse_args()
    root = args.root
    protocol_path = root/'protocol_before_new_outcomes.json'
    protocol = json.loads(protocol_path.read_text())
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    events_all, traces, outcomes, groups, sensitivity, summaries, inputs = [], [], [], [], [], [], []
    for cohort in protocol['cohorts']:
        source = Path(cohort['source'])
        manifest = source/'manifest.json'
        settings = json.loads(manifest.read_text())['settings']
        pairs_path = source/'csv/report_all_comparisons.csv'
        pairs = read(pairs_path)
        assert len(pairs) == len(settings['seeds']) == 100
        assert {int(p['seed']) for p in pairs} == set(settings['seeds'])
        assert all(p['status'] == 'OK' for p in pairs)
        inputs.extend([manifest, pairs_path])
        tag = dict(cohort=cohort['name'], role=cohort['role'])
        local_events, local_outcomes, local_sensitivity = [], [], []
        for pair in pairs:
            seed = int(pair['seed'])
            decisions, modes, run_summary = {}, {}, {}
            for label, arm in [('nominal','sh_mpcc'), ('wdro','sh_mpcc_dro')]:
                folder = source/pair['pair']/f'seed_{seed}'/arm
                inputs.extend(folder/f for f in ['decisions.csv','mode_mechanism.csv','summary.csv'])
                dd = read(folder/'decisions.csv')
                mm = [r for r in read(folder/'mode_mechanism.csv') if r['mode']=='across' and r['obstacle_id']=='0' and r['attempt']=='0']
                ss = read(folder/'summary.csv')
                assert len(ss)==1 and ss[0]['status']=='OK'
                decisions[label] = {int(r['step']):r for r in dd}
                modes[label] = {int(r['step']):r for r in mm}
                assert len(decisions[label])==len(dd) and len(modes[label])==len(mm)
                assert decisions[label].keys()==modes[label].keys()
                run_summary[label] = ss[0]
            group = category(run_summary['nominal'], run_summary['wdro'])
            e = extend_event(seed_events(seed, group, decisions['nominal'], decisions['wdro'], modes,
                                        protocol['material_acceleration'], protocol['material_yaw_rate']))
            for label in run_summary:
                brake = e[label+'_first_brake']
                assert int(run_summary[label]['first_brake_step']) == (-1 if brake is None else brake)
            local_events.append(dict(e, **tag))
            traces.extend(dict(r, **tag) for r in flatten(seed, group, decisions['nominal'], decisions['wdro'], modes))
            local_outcomes.append(dict(seed=seed, group=group, **tag,
                **{label+'_'+k:v for label, s in run_summary.items() for k,v in s.items()}))
            for threshold in protocol['sensitivity']:
                se = extend_event(seed_events(seed, group, decisions['nominal'], decisions['wdro'], modes,
                                              threshold['acceleration'], threshold['yaw_rate']))
                local_sensitivity.append(dict(se, **tag, acceleration_threshold=threshold['acceleration'], yaw_rate_threshold=threshold['yaw_rate']))
        groups.extend(dict(r, **tag) for r in summarize_cohort(local_events))
        events_all.extend(local_events)
        outcomes.extend(local_outcomes)
        sensitivity.extend(local_sensitivity)
        n_only = sum(r['nominal_collision']=='1' and r['wdro_collision']=='0' for r in local_outcomes)
        d_only = sum(r['nominal_collision']=='0' and r['wdro_collision']=='1' for r in local_outcomes)
        summary = dict(**tag, paired_seeds=len(pairs), nominal_only_collision=n_only, wdro_only_collision=d_only,
                       exact_paired_p_unadjusted=mcnemar(n_only,d_only))
        for arm in ['nominal','wdro']:
            for field in ['collision','path_completed']:
                summary[arm+'_'+field] = sum(r[arm+'_'+field]=='1' for r in local_outcomes)
            summary[arm+'_refusal'] = sum(r[arm+'_termination']=='no_admissible_control' for r in local_outcomes)
        summaries.append(summary)
        rescued = [r for r in local_events if r['group']=='nominal_collision_wdro_refusal']
        title = f"Temporal mechanism in {len(rescued)} nominal-collision / WDRO collision-free-to-refusal\n{cohort['name']} pairs"
        timeline(rescued, out/(cohort['name']+'_primary'), title)
        timeline(rescued, out/(cohort['name']+'_supplement'), title, True)
        print(cohort['name'], summary['nominal_collision'], summary['wdro_collision'], flush=True)
    tables = dict(cohort_summary=summaries, group_summary=groups, events_all_seeds=events_all,
                  events_collision_to_refusal=[e for e in events_all if e['group']=='nominal_collision_wdro_refusal'],
                  seed_outcomes=outcomes, trace_full_all_seeds=traces,
                  trace_steps_0_10_all_seeds=[r for r in traces if 0<=r['step']<=10], threshold_sensitivity=sensitivity)
    sensitivity_summary = []
    for cohort in protocol['cohorts']:
        for threshold in protocol['sensitivity']:
            selected = [r for r in sensitivity if r['cohort']==cohort['name'] and r['acceleration_threshold']==threshold['acceleration']]
            sensitivity_summary.extend(dict(r, cohort=cohort['name'], **threshold) for r in summarize_cohort(selected))
    tables['threshold_sensitivity_summary'] = sensitivity_summary
    for name, rows in tables.items():
        write(out/(name+'.csv'), rows)
    lines = ['# Expanded control-response evidence', '',
        '400 new simulations: 100 paired seeds (5001–5100) in each of two conditions. Ten existing cohorts were reanalyzed. All cohorts remain separate; shared seed numbers across conditions are not independent pooled trials.', '',
        '| Cohort | Nominal collisions | WDRO collisions | Nominal-only / WDRO-only | Exact paired p (unadjusted) |', '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['cohort']} | {s['nominal_collision']}/100 | {s['wdro_collision']}/100 | {s['nominal_only_collision']} / {s['wdro_only_collision']} | {s['exact_paired_p_unadjusted']:.5g} |")
    lines += ['', '## Fresh-cohort response evidence', '',
        '| Cohort, collision-to-refusal subgroup | Positive chain at first response | First extra sample equals first response | Earlier WDRO brake / both observed | Median paired brake advance (steps) |',
        '|---|---:|---:|---:|---:|']
    for g in groups:
        if g['role']=='new_confirmation' and g['group']=='nominal_collision_wdro_refusal':
            counts = [str(g[f+'_yes'])+'/'+str(g[f+'_eligible']) for f in
                      ['positive_chain_at_first_response','first_extra_equals_first_response','wdro_brakes_earlier_when_both_observed']]
            lines.append(f"| {g['cohort']} (n={g['seeds']}) | {' | '.join(counts)} | {g['braking_advance_steps_median']} |")
    lines += ['', 'The fresh low-sampling collision reduction replicates the outcome pattern, but the earlier-braking pattern is weaker than in the preceding replication. Positive sampling/control chains also occur in neither-collision and WDRO-only collision groups; they are not sufficient evidence of collision rescue. No run in these cohorts completed the path.', '']
    lines += ['', 'Primary figures use the first **extra** WDRO across sample, not the first WDRO sample. Supplementary figures include both arms’ first samples. A missing figure means no collision-to-refusal pairs in that cohort.', '',
        'group_summary.csv includes every observed outcome group and explicit eligible denominators. events_all_seeds.csv retains nonaligned events and absent events as blanks. The positive-chain flag requires positive risk, positive mass shift, and an extra WDRO sample at the first command-or-availability response. First-extra equality is assessed separately: an earlier extra sample need not cause an immediate control difference.', '',
        'Material command difference: |Δa|≥0.1 m/s² or |Δω|≥0.02 rad/s, both commands executable. Half/double thresholds are included. First brake: executed a<−0.1. Sampling at solve t precedes its control; speed and clearance are pre-control measurements. Clearance is not a continuous-time minimum. Full traces retain events after step 10; no post-refusal motion or missing next state is imputed.', '',
        'The collision-to-refusal subgroup is selected by outcomes. Similar sampling/control timing in non-rescued pairs is evidence about control response, not proof that each extra sample prevents collision. State divergence and other sampled trajectories prevent a causal mediation claim. Collision-free until refusal does not imply safety after refusal. No universal kappa or conditional collision guarantee is inferred.', '',
        'No mathematical guarantee or formulation was modified. Candidate analysis implemented and tested; behavior and scientific interpretation require user verification.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    inputs.extend([protocol_path, Path(__file__), Path('tools/analyze_control_response.py'), Path('tools/report_coverage_replication.py')])
    (out/'provenance.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs], indent=2)+'\n')
    archive = root/'control-response-expanded-evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file(): z.write(p, 'analysis/'+p.name)
        for p in [protocol_path, root/'fresh_low_sampling_settings.json', root/'fresh_negative_control_settings.json']:
            z.write(p, p.name)
        for p in [Path(__file__), Path('tools/analyze_control_response.py'), Path('tests/test_control_response_expansion.py'), Path('tests/test_control_response.py')]:
            z.write(p, 'code/'+str(p))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(archive)

if __name__ == '__main__':
    main()
