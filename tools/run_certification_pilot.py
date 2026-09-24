#!/usr/bin/env python3
"""Only the three requested pilot cells; canonical settings, no reduced-S controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from run_analysis_matrix import config_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, default=ROOT/'build-certificate/experiment_runner')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args()
    if args.output.exists() or min(args.seeds, args.steps, args.timeout) < 1:
        p.error('use a new output directory and positive counts')
    args.output.mkdir(parents=True)
    settings = json.loads((ROOT/'configs/analysis_matrix/settings.json').read_text())
    settings['overrides'].update(rollout_steps=args.steps, artifact_capture_attempt_diagnostics=True,
        artifact_write_visualization_gif=False, artifact_write_visualization_svg=False,
        artifact_show_linearized_constraints=False, artifact_show_support_scenarios=False)
    manifest = dict(status='running', seeds=list(range(77, 77+args.seeds)), steps=args.steps,
        runner=str(args.runner.resolve()), runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        settings=settings, runs=[], sample_reduction_enabled=False)
    def save():
        (args.output/'pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    for environment, modes in [('straight', 2), ('straight', 3), ('s_curve', 2)]:
        case = dict(case=f'sh_mpcc_dro_{environment}_o1_c1_m{modes}', obstacles=1, classes=1,
                    environment=environment, modes_per_class=modes, solver_style='sh_mpcc_dro')
        config = args.output/(case['case']+'.yaml')
        config.write_text(config_text(case, settings))
        for seed in manifest['seeds']:
            output = args.output/case['case']/f'seed_{seed}'
            output.mkdir(parents=True)
            command = [str(args.runner.resolve()), '--config', str(config.resolve()), '--seed', str(seed),
                       '--output', str(output.resolve()), '--label', 'rollout', '--no-gif', '--no-svg', '--no-rviz']
            run = dict(case=case['case'], seed=seed, command=command)
            with (output/'run.log').open('w') as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    run['returncode'] = completed.returncode
                except subprocess.TimeoutExpired:
                    run['returncode'] = None
                    run['error'] = 'timeout; no completed rollout artifacts assumed'
            manifest['runs'].append(run)
            save()
            print(json.dumps(run), flush=True)
    manifest['status'] = 'completed' if all(r['returncode']==0 for r in manifest['runs']) else 'completed_with_failures'
    save()

if __name__ == '__main__':
    main()
