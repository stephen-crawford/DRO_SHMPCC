#!/usr/bin/env python3
"""Serial long-encounter frozen-context screen; no learned/adaptive live policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[77, 78])
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = sorted((args.output / 'configs').glob('*.yaml'))
    if not configs:
        parser.error('prepare frozen YAML configurations in output/configs first')
    manifest = dict(runner=str(args.runner.resolve()), runner_sha256=digest(args.runner),
                    script_sha256=digest(Path(__file__)), source_sha256=digest(Path('tools/week2_budget_response.cpp')),
                    configs={p.name: digest(p) for p in configs}, seeds=args.seeds,
                    steps=args.steps, stride=args.stride, repeats=args.repeats, timeout=args.timeout,
                    jobs=1, caps=[1,2,3,4,5,6,7,10], epsilon=0.05, beta=0.01)
    path = args.output / 'experiment_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        parser.error('different experiment manifest; use a new output directory')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for config in configs:
        for seed in args.seeds:
            root = args.output / config.stem / f'seed_{seed}'
            root.mkdir(parents=True, exist_ok=True)
            command = [str(args.runner.resolve()), str(config.resolve()), str(root.resolve()),
                       str(seed), str(args.stride), str(args.repeats), str(args.steps)]
            record = dict(case=config.stem, seed=seed, command=command)
            start = time.monotonic()
            try:
                with (root / 'run.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=args.timeout)
                record['status'] = 'OK'
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                record.update(status='ERROR', error=str(error))
            record['wall_seconds'] = time.monotonic() - start
            results.append(record)
            (args.output / 'execution_results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(config.stem, seed, record['status'], flush=True)
    return int(any(r['status'] != 'OK' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
