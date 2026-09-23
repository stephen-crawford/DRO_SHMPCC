#!/usr/bin/env python3
"""Resume a saved matrix using its frozen configurations, or restart a separate cohort."""
import argparse
import fcntl
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import run_analysis_matrix as analysis
import run_comparison_matrix as comparison
import run_reviewer_matrix as reviewer

ROOT = analysis.ROOT


def read_results(root, manifest):
    results = {}
    for case in manifest['cases']:
        for seed in manifest['settings']['seeds']:
            path = analysis.trial_root(root, case, seed)/'result.json'
            if path.exists():
                r = json.loads(path.read_text())
                if r.get('identity') != manifest['identity']:
                    raise ValueError(f'stale result identity: {path}')
                if r.get('case') != case['case'] or r.get('seed') != seed:
                    raise ValueError(f'result case/seed mismatch: {path}')
                results[case['case'], seed] = r
    return results


def preflight(root, manifest, runner):
    issues = []
    frozen = {k:v for k,v in manifest.items() if k != 'identity'}
    if analysis.digest(json.dumps(frozen, sort_keys=True).encode()) != manifest['identity']:
        issues.append('Saved manifest identity does not match its contents.')
    if not runner.is_file() or analysis.digest(runner.read_bytes()) != manifest['runner_sha256']:
        issues.append('Executable differs from the recorded binary (or is missing).')
    default_keys = set(comparison.evidence.flat_yaml(ROOT/'configs/default.yaml'))
    for case in manifest['cases']:
        path = root/'configs'/(case['case']+'.yaml')
        if not path.exists() or analysis.digest(path.read_bytes()) != manifest['configs'][case['case']]:
            issues.append(f'Missing or changed frozen configuration: {path}')
            break
        missing = default_keys - set(comparison.evidence.flat_yaml(path))
        if missing:
            issues.append('Current default YAML contains keys absent from frozen configurations: '+', '.join(sorted(missing)))
            break
    if manifest.get('jobs', 1) != 1:
        issues.append('This resumer supports saved serial matrices only; use a separate restart for parallel designs.')
    return issues


def reports(root, manifest, results):
    cases = manifest['cases']; settings = manifest['settings']
    values = sorted(results.values(), key=lambda r:(r['case'],r['seed']))
    analysis.dump_json(root/'results.json', values)
    suite = manifest.get('suite', 'analysis_matrix')
    if suite == 'comparison_matrix':
        comparisons = []
        for name in sorted({c['pair'] for c in cases}):
            arms = [c for c in cases if c['pair'] == name]
            for left, right in itertools.combinations(arms, 2):
                for seed in settings['seeds']:
                    comparisons.append(comparison.paired_result([left,right], seed, results, root))
        pairs = [r for r in comparisons if (r['controller_a'],r['controller_b']) == comparison.PAIR_STYLES]
        for name, rows in [('pairs',pairs),('all_comparisons',comparisons),
                           ('matches',[r for r in pairs if r['target_match']]),('summary',comparison.summarize(pairs))]:
            analysis.dump_json(root/(name+'.json'), rows)
            analysis.write_csv(root/(name+'.csv'), rows)
        comparison.evidence.write_evidence_reports(root,cases,settings,results,comparisons)
        return any(r['status']=='ERROR' for r in comparisons)
    else:
        summary=analysis.aggregate(values,cases,len(settings['seeds']))
        analysis.write_csv(root/'summary.csv', summary)
        analysis.dump_json(root/'summary.json', summary)
        analysis.write_csv(root/'summary_per_seed.csv', analysis.per_seed_summary(values))
        if suite == 'reviewer_matrix':
            pairs=[]
            for pair,seed in itertools.product(sorted({c['pair'] for c in cases}),settings['seeds']):
                arms=[c for c in cases if c['pair']==pair]
                group=[results[c['case'],seed] for c in arms if (c['case'],seed) in results]
                status,error=reviewer.check_pairing(group,root) if len(group)==len(arms) else ('PENDING','missing trials')
                pairs.append(dict(pair=pair,seed=seed,status=status,error=error))
            analysis.write_csv(root/'pairing.csv',pairs)
            return any(r['status']=='ERROR' for r in pairs)
        flat=[]
        for r in values:
            base={k:r[k] for k in ['case','seed','status','obstacles','classes','environment','modes_per_class','solver_style']}
            base['error']=r.get('error','')
            if not r['repeats']:flat.append(base)
            for i,observation in enumerate(r['repeats']):
                flat.append(dict(base,repeat=i,wall_seconds=observation['wall_seconds'],
                                 **{k:v for k,v in observation['metrics'].items() if k!='initial_placement'}))
        analysis.write_csv(root/'rollouts.csv',flat)
    return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,epilog='Do not run another matrix process against the same output directory. Failed/partial trials are archived before rerunning all their repeats. A restart reruns ALL trials with a new identity; it never imports old successes.')
    p.add_argument('output',type=Path,help='existing directory containing matrix.json')
    p.add_argument('--dry-run',action='store_true',help='audit compatibility and count remaining trials without writes or simulations')
    p.add_argument('--limit',type=int,help='run at most this many incomplete seed trials in this invocation')
    p.add_argument('--runner',type=Path,help='same-hash executable for resume; new executable for --restart-to')
    p.add_argument('--restart-to',type=Path,help='new cohort directory when the original executable/defaults are incompatible')
    args=p.parse_args(argv)
    root=args.output.resolve();manifest=json.loads((root/'matrix.json').read_text())
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    runner=(args.runner or Path(manifest['runner'])).resolve()
    results=read_results(root,manifest)
    expected=len(manifest['cases'])*len(manifest['settings']['seeds'])
    complete=sum(r['status']=='OK' for r in results.values())
    errors=sum(r['status']!='OK' for r in results.values())
    print(f'{root}: {complete}/{expected} OK; {errors} failed; {expected-len(results)} without a result',flush=True)
    if args.restart_to:
        destination=args.restart_to.resolve()
        if destination.exists():p.error('--restart-to requires a new directory; resume an existing new cohort directly')
        if args.limit:p.error('--limit applies to in-place resume, not restart')
        script={'comparison_matrix':'run_comparison_matrix.py','reviewer_matrix':'run_reviewer_matrix.py'}.get(manifest.get('suite'),'run_analysis_matrix.py')
        command=[sys.executable,str(ROOT/'tests'/script),'--settings',str(destination/'source_settings.json'),
                 '--runner',str(runner),'--output',str(destination),'--timeout',str(manifest.get('timeout',manifest.get('timeout_seconds',1800)))]
        print('Separate restart command: '+ ' '.join(command),flush=True)
        if args.dry_run:return 0
        destination.mkdir(parents=True)
        analysis.dump_json(destination/'source_settings.json',manifest['settings'])
        analysis.dump_json(destination/'restart_origin.json',dict(source=str(root),identity=manifest['identity'],old_results_reused=False))
        return subprocess.call(command)
    issues=preflight(root,manifest,runner)
    if issues:
        for issue in issues:print('BLOCKED: '+issue)
        print('Use the original compatible environment, or --restart-to NEW_DIRECTORY --runner NEW_EXECUTABLE.')
        return 2
    print('Frozen configuration hashes and executable match. Script changes will be recorded in resume provenance.',flush=True)
    if args.dry_run:return 0
    with (root/'.resume.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('another saved-matrix resumer holds this directory lock')
        # Validate successful trials before reusing them; preserve old files on failure.
        for case in manifest['cases']:
            for seed in manifest['settings']['seeds']:
                r=results.get((case['case'],seed))
                if not r or r['status']!='OK':continue
                if len(r['repeats'])!=manifest['settings']['repeats']:
                    p.error(f'incomplete repeats marked OK: {case["case"]} seed={seed}')
                if len({o['signature'] for o in r['repeats']})!=1:
                    p.error(f'inconsistent repeats marked OK: {case["case"]} seed={seed}')
                for repeat,record in enumerate(r['repeats']):
                    bundle=analysis.trial_root(root,case,seed)/f'repeat_{repeat}'
                    if analysis.repeat_signature(bundle,analysis.analyze(bundle,case))!=record['signature']:
                        p.error(f'saved evidence changed: {bundle}')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history=root/'resume_history'/stamp;history.mkdir(parents=True)
        analysis.dump_json(history/'session.json',dict(original_identity=manifest['identity'],runner=str(runner),
            sources={str(f.relative_to(ROOT)):analysis.digest(f.read_bytes()) for f in
                [Path(__file__),Path(analysis.__file__),Path(comparison.__file__),Path(reviewer.__file__),Path(comparison.evidence.__file__)]}))
        run_args=SimpleNamespace(output=root,runner=runner,resume=False,timeout=manifest.get('timeout',manifest.get('timeout_seconds',1800)))
        ran=0
        for case,seed in itertools.product(manifest['cases'],manifest['settings']['seeds']):
            r=results.get((case['case'],seed))
            if r and r['status']=='OK':continue
            if args.limit is not None and ran>=args.limit:break
            trial=analysis.trial_root(root,case,seed)
            if trial.exists():
                archive=history/trial.relative_to(root);archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(trial),str(archive))
            fn=comparison.evidence.run_checked_trial if manifest.get('suite')=='comparison_matrix' else analysis.run_trial
            result=fn(case,seed,run_args,manifest['settings'],manifest['identity'])
            results[case['case'],seed]=result;ran+=1
            print(f'{case["case"]} seed={seed}: {result["status"]} {result.get("error", "")}',flush=True)
        pairing_error=reports(root,manifest,results)
        print(f'Ran {ran} trials; {sum(r["status"]=="OK" for r in results.values())}/{expected} OK.',flush=True)
        return int(pairing_error or any(r['status']!='OK' for r in results.values()))


if __name__=='__main__':
    raise SystemExit(main())
