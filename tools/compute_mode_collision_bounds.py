#!/usr/bin/env python3
"""Batch returned-plan diagnostics. Does not change or issue controller certificates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from analyze_posterior_certificate import projected_probability

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def support_audit(snapshot):
    active = set(snapshot['support_active'])
    removed = set(snapshot['removed_ids'])
    union = set(snapshot['support_union'])
    evaluated = snapshot['support_evaluated']
    return dict(support_evaluated=evaluated, support_active=len(active),
                removed_ids=';'.join(map(str, sorted(removed))), removed_count=len(removed),
                support_union=';'.join(map(str, sorted(union))),
                support_count_used_for_certificate=snapshot['support_count_used_for_certificate'],
                unique_ids=len(union) == len(snapshot['support_union']),
                count_matches=snapshot['support_count_used_for_certificate'] == len(union),
                removed_in_union=removed <= union if evaluated else None,
                active_in_union=active <= union if evaluated else None)


def compute(snapshot, backend):
    if len(snapshot['obstacles']) != 1:
        raise ValueError('pilot requires one obstacle; no implicit joint-mode law')
    if snapshot['bundle_sampling'] or snapshot['markov_jump_system']:
        raise ValueError('requires ordinary held-mode sampling')
    n = snapshot['horizon']
    if len(snapshot['trajectory']) != n + 1 or len(snapshot['disc_centers']) != n + 1:
        raise ValueError('incomplete returned horizon')
    obstacle = snapshot['obstacles'][0]
    modes = obstacle['modes']
    if not modes or len({m['mode'] for m in modes}) != len(modes):
        raise ValueError('empty or duplicate mode support')
    for key in ('p_hat', 'q_star'):
        law = [m[key] for m in modes]
        if any(x is None or not math.isfinite(x) or x < 0 for x in law) or abs(sum(law)-1) > 1e-10:
            raise ValueError('invalid or missing ' + key)
    counts = [m['count'] for m in modes]
    if any(type(c) is not int or c < 0 for c in counts):
        raise ValueError('invalid history counts')
    intervals = [list(map(float, line.split())) for line in subprocess.check_output(
        [str(backend), 'intervals', str(snapshot['beta_cp']), *map(str, counts)], text=True).splitlines()]
    if len(intervals) != len(modes):
        raise ValueError('backend interval count mismatch')
    pairs, bounds = [], []
    for mode, (lower, upper) in zip(modes, intervals):
        if not mode['held_affine_gaussian']:
            raise ValueError('unsupported non-affine or switching predictive law')
        if len(mode['predictions']) != n + 1:
            raise ValueError('incomplete predictions')
        probabilities = []
        for k in range(1, n + 1):
            prediction = mode['predictions'][k]
            if not snapshot['disc_centers'][k]:
                raise ValueError('empty disc geometry')
            for disc, center in enumerate(snapshot['disc_centers'][k]):
                result = projected_probability(center, prediction['mean'], prediction['covariance'], snapshot['collision_radius'])
                probabilities.append(result['probability_upper'])
                pairs.append(dict(obstacle=obstacle['obstacle'], mode=mode['mode'], k=k, disc=disc, **result))
        total = math.fsum(probabilities)
        shift = 'increased' if mode['q_star'] > mode['p_hat'] else 'decreased' if mode['q_star'] < mode['p_hat'] else 'unchanged'
        bounds.append(dict(obstacle=obstacle['obstacle'], mode=mode['mode'],
                           p_hat=mode['p_hat'], q_star=mode['q_star'], L=lower, U=upper,
                           count=mode['count'], b_mode=min(1., total), max_pair_prob=max(probabilities),
                           sum_pair_prob=total, pair_count=len(probabilities), mass_shift=shift,
                           rho=obstacle['rho']))
    return bounds, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', type=Path, default=ROOT/'build-certificate/certificate_numeric_backend')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output filename')
    paths = sorted(args.root.glob('**/certification/step_*_attempt_*.json'))
    if not paths:
        parser.error('no certification snapshots found')
    all_bounds, all_pairs, audits, outcomes, hashes = [], [], [], [], {}
    for path in paths:
        snapshot = json.loads(path.read_text())
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = {key: snapshot[key] for key in ('case', 'seed', 'step', 'attempt')}
        identity['snapshot'] = str(path.resolve())
        audit = support_audit(snapshot)
        audits.append(dict(identity, **audit))
        outcome = dict(identity, success=snapshot['success'], certificate_status=snapshot['certificate_status'])
        if not snapshot['success'] or not snapshot['returned_attempt'] or not snapshot['dro_enabled']:
            outcome['status'] = 'excluded_failed_nonreturned_or_nominal'
        else:
            try:
                bounds, pairs = compute(snapshot, args.backend.resolve())
                all_bounds.extend(dict(identity, **row) for row in bounds)
                all_pairs.extend(dict(identity, **row) for row in pairs)
                outcome['status'] = 'evaluated'
            except ValueError as error:
                outcome['status'] = 'unsupported: ' + str(error)
        outcomes.append(outcome)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, all_bounds)
    for suffix, rows in [('pairs', all_pairs), ('support_audit', audits), ('outcomes', outcomes)]:
        write_csv(args.output.with_name(args.output.stem + '_' + suffix + '.csv'), rows)
    provenance = dict(source_sha256=hashes, backend_sha256=hashlib.sha256(args.backend.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      projection_sha256=hashlib.sha256((ROOT/'tools/analyze_posterior_certificate.py').read_bytes()).hexdigest(),
                      event='strict disc overlap at future stages 1..N', joint_obstacles=1,
                      presolve_sample_reduction_claim=False)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(snapshots=len(paths), evaluated=sum(r['status']=='evaluated' for r in outcomes),
                         modes=len(all_bounds), pairs=len(all_pairs), exclusions=len(paths)-sum(r['status']=='evaluated' for r in outcomes))))

if __name__ == '__main__':
    main()
