#!/usr/bin/env python3
"""Report the one-shot earlier-switch experiment, including negative outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
from causal_link_analysis import read, write
from report_coverage_replication import mcnemar


def outcome(row):
    if row.get('status') != 'OK':
        return 'execution_error'
    if row['collision'] == '1':
        return 'collision'
    if row['path_completed'] == '1':
        return 'completed'
    return row['termination']


def paired_metric(rows, metric):
    eligible = [r for r in rows if r['nominal_outcome'] != 'execution_error' and r['wdro_outcome'] != 'execution_error']
    n_only = sum(r['nominal_'+metric]=='1' and r['wdro_'+metric]=='0' for r in eligible)
    d_only = sum(r['nominal_'+metric]=='0' and r['wdro_'+metric]=='1' for r in eligible)
    return dict(metric=metric, paired_eligible=len(eligible), excluded_pairs=len(rows)-len(eligible),
                nominal_only=n_only, wdro_only=d_only, exact_paired_p_unadjusted=mcnemar(n_only,d_only))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/recoverable-switch-once'))
    root=parser.parse_args().root
    protocol=json.loads((root/'protocol_before_outcomes.json').read_text())
    settings=protocol['settings']
    manifest=json.loads((root/'runs/manifest.json').read_text())
    assert manifest['settings']==settings
    assert manifest['hashes']=={str(Path(p).resolve()):v for p,v in protocol['hashes'].items()}
    for path,digest in protocol['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    case='late_switch_p20_s10_switch2'
    paired=[];inputs=[];counts={arm:Counter() for arm in ['nominal','wdro']}
    for seed in settings['seeds']:
        row=dict(seed=seed)
        for arm,folder in [('nominal','sh_mpcc'),('wdro','sh_mpcc_dro')]:
            base=root/'runs'/case/f'seed_{seed}'/folder
            path=base/'summary.csv';summaries=read(path)
            assert len(summaries)==1, str(path)
            summary=summaries[0]
            result=outcome(summary)
            counts[arm][result]+=1
            row[arm+'_outcome']=result
            row.update({arm+'_'+key:value for key,value in summary.items()})
            inputs.extend([path,base/'fixture.csv',base/'plant.csv'])
            fixture=read(base/'fixture.csv')
            assert len(fixture)==1 and fixture[0]['switch_step']=='2' and fixture[0]['base_S']=='10'
        # Compare actual prescribed plant states on every common recorded step.
        n=read(root/'runs'/case/f'seed_{seed}'/'sh_mpcc/plant.csv')
        d=read(root/'runs'/case/f'seed_{seed}'/'sh_mpcc_dro/plant.csv')
        assert n[:min(len(n),len(d))]==d[:min(len(n),len(d))]
        paired.append(row)
    assert len(paired)==100 and len({r['seed'] for r in paired})==100
    categories=['completed','collision','no_admissible_control','step_limit','execution_error']
    totals=[dict(arm=arm,runs=sum(c.values()),**{k:c[k] for k in categories}) for arm,c in counts.items()]
    metrics=[paired_metric(paired,m) for m in ['path_completed','collision']]
    transitions=Counter((r['nominal_outcome'],r['wdro_outcome']) for r in paired)
    write(root/'paired_seed_outcomes.csv',paired)
    write(root/'outcome_summary.csv',totals)
    write(root/'paired_tests.csv',metrics)
    write(root/'paired_outcome_transitions.csv',[dict(nominal_outcome=n,wdro_outcome=d,seeds=k) for (n,d),k in sorted(transitions.items())])
    mixed=any(c['completed'] for c in counts.values()) and any(c['collision'] for c in counts.values())
    lines=['# One-shot recoverability test','',
        'Predeclared change: switch step 10 → 2, 0.8 seconds earlier. All other fixture settings remain unchanged: p=.02, S=10, history lag 5, 120-step limit, two raw uncertified arms. Fresh paired seeds 6001–6100. No pilot or additional design search.', '',
        '| Arm | Completion | Collision | Refusal | Step limit | Error |','|---|---:|---:|---:|---:|---:|']
    for r in totals:lines.append('| '+r['arm']+' | '+' | '.join(str(r[k]) for k in categories)+' |')
    lines+=['',f'Predeclared mixed completion/collision criterion met: **{mixed}**.',
        'The one-condition experiment is finished. No further geometry tuning or simulation is performed.', '',
        'Completion retains the existing condition ego.x ≥ 15.2 after a collision-free simulated update; the reference path ends at x=16. Refusal means simulation stopped without executing a control, not guaranteed safety afterward. Step-limit runs are not completions. Errors are retained and excluded explicitly from paired tests.', '',
        'The tests are exact paired McNemar tests on completion and collision, reported separately without multiplicity adjustment. This is a prescribed adverse-switch experiment, not a population accident-rate estimate. Different seeds are independent pairs; neither decisions nor obstacles are outcome replicates.', '',
        'Previous failed prospective coverage ordering and count-intervention evidence are retained as boundaries of the result. This experiment does not establish a sample-count threshold or conditional collision guarantee.', '',
        '## Reproduce','', '```bash',
        'python3 tests/run_risk_stress.py --settings results/recoverable-switch-once/settings.json --output results/recoverable-switch-once/runs --resume',
        'python3 tools/report_recoverable_switch.py', '```', '',
        'No mathematical guarantee or formulation was modified. Existing executable reused; no build required. Source hashes, complete 100-pair denominators, fixture switch/budget settings and identical common plant prefixes were checked. Candidate experiment implemented and test executed; behavior requires user verification.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'analysis_provenance.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs+[Path(__file__),root/'protocol_before_outcomes.json']],indent=2)+'\n')
    files=list(root.glob('*.csv'))+list(root.glob('*.json'))+[root/'REPORT.md',root/'run.log']
    files+=list((root/'runs/report').glob('*.csv'))+list((root/'runs/csv').glob('*.csv'))
    if (root/'validation_tests.log').exists():files.append(root/'validation_tests.log')
    with zipfile.ZipFile(root/'recoverable-switch-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,str(path.relative_to(root)))
        z.write(__file__,'code/'+Path(__file__).name)
    with zipfile.ZipFile(root/'recoverable-switch-evidence.zip') as z:assert z.testzip() is None
    print(json.dumps(totals,indent=2));print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
