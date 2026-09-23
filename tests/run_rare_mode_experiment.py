#!/usr/bin/env python3
"""Run a frozen rare-mode experiment and retain executable/configuration identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,default=ROOT/'build-rare-mode/rare_mode_experiment')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-replicates',type=int,default=2000)
    p.add_argument('--solve-seeds',type=int,default=40)
    args=p.parse_args()
    if args.coverage_replicates<1 or args.solve_seeds<1:p.error('counts must be positive')
    if args.output.exists():p.error('use a new output directory')
    args.output.mkdir(parents=True)
    runner=args.runner.resolve()
    manifest=dict(runner=str(runner),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        coverage_replicates=args.coverage_replicates,solve_seeds=args.solve_seeds,
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [ROOT/'tools/rare_mode_experiment.cpp',ROOT/'configs/default.yaml']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (args.output/'experiment.log').open('w') as log:
        run=subprocess.run([str(runner),str((args.output/'artifacts').resolve()),
                            str(args.coverage_replicates),str(args.solve_seeds)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([sys.executable,str(ROOT/'tools/report_rare_mode_experiment.py'),
                    '--input',str(args.output/'artifacts'),'--output',str(args.output/'report')],check=True)
    return run.returncode


if __name__=='__main__':raise SystemExit(main())
