#!/usr/bin/env python3
"""Serial fixed-cap pilot sweep; preserve raw production and clone diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runner', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[77, 78])
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a new output directory')
    args.output.mkdir(parents=True)
    manifest = dict(M=[0,5,10,20,40,80,160], h=10, S=1351,
                    seeds=args.seeds, steps=args.steps,
                    binary_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    records = []
    for m in manifest['M']:
        for config in sorted(args.configs.glob('*.yaml')):
            cfg = args.output/'configs'/f'{config.stem}_M{m}.yaml'
            cfg.parent.mkdir(exist_ok=True)
            cfg.write_text(config.read_text()+f'\nanchor_pilot_samples: {m}\n')
            for seed in args.seeds:
                root = args.output/f'M{m}'/config.stem/f'seed_{seed}'
                root.mkdir(parents=True)
                cmd = [str(args.runner.resolve()),str(cfg.resolve()),str(root.resolve()),
                       str(seed),'10','1',str(args.steps),'--fixed-h10']
                with (root/'run.log').open('w') as log:
                    completed = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                record = dict(M=m,case=config.stem,seed=seed,exit_code=completed.returncode,command=cmd)
                if completed.returncode == 0:
                    with (root/'summary.csv').open() as f:
                        record.update(next(csv.DictReader(f)))
                    with (root/'baseline_decisions.csv').open() as f:
                        rows = list(csv.DictReader(f))
                    record['decisions']=len(rows)
                    record['certified']=sum(int(r['certified']) for r in rows)
                records.append(record)
                (args.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
                print(m,config.stem,seed,record.get('steps'),record.get('termination'),flush=True)
    return int(any(r['exit_code'] for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
