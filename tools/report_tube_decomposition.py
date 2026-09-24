#!/usr/bin/env python3
"""Independently check saved tube geometry and decompose conditional sample counts.

Reads artifacts only; never runs or changes the controller. All transfer results
use the existing CP-box transfer diagnostic, not a new Wasserstein theorem.
"""
import argparse
import csv
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from analyze_posterior_certificate import (
    projected_probability,
    projected_probability_with_normal,
)
from analyze_wdro_certificate import analyze, vertices, transfer_at

KEY = ('seed', 'repeat', 'arm', 'radius', 'cycle')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row[k] for k in KEY)


def report(source, output, backend):
    if output.exists():
        raise ValueError('use a new report directory')
    output.mkdir(parents=True)

    cycles = read(source/'cycles.csv')
    geometry = read(source/'disc_geometry.csv')
    attempts = read(source/'attempts.csv')
    predictions = {
        (r['mode'], r['stage']): r
        for r in read(source/'predictions.csv')
    }
    manifest = json.loads((source/'manifest.json').read_text())
    (output/'source_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )

    grouped, shapes, tries = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in cycles:
        grouped[key(r)].append(r)
    for r in geometry:
        shapes[key(r)].append(r)
    for r in attempts:
        tries[(key(r), r['attempt'])].append(r)

    violations, bounds = [], []
    tolerance = 1e-12

    for identity, pair in grouped.items():
        shape = shapes[identity]
        expected = {
            (str(k), str(j))
            for k in range(1, manifest['horizon'] + 1)
            for j in range(3)
        }
        if pair[0]['success'] == '1' and (
            len(shape) != len(expected)
            or {(s['stage'], s['disc']) for s in shape} != expected
        ):
            violations.append(dict(kind='incomplete_geometry', identity=identity))

        for row in pair:
            posterior_adaptive = 0.0
            posterior_fixed = 0.0
            uniform = 0.0
            active = row['active'] == '1'
            inside_tube = True

            for disc in shape:
                g = predictions[(row['mode'], disc['stage'])]
                mean = [
                    float(g['mean_x']),
                    float(g['mean_y']),
                ]
                cov = [
                    [float(g['cov_xx']), float(g['cov_xy'])],
                    [float(g['cov_yx']), float(g['cov_yy'])],
                ]
                radius = float(g['collision_radius'])
                center = [
                    float(disc['x']),
                    float(disc['y']),
                ]

                # Diagnostic returned-plan upper bound. This recomputes the
                # projection normal at the returned center. It is useful as an
                # a-posteriori diagnostic, but it is not theorem-ordered below
                # the fixed-normal pre-solve tube bound for anisotropic Sigma.
                posterior_adaptive += projected_probability(
                    center,
                    mean,
                    cov,
                    radius,
                )['probability_upper']

                if active:
                    ref = [
                        float(disc['reference_x']),
                        float(disc['reference_y']),
                    ]
                    tube_radius = float(row['radius'])

                    # Fixed certification normal determined entirely by the
                    # pre-solve reference geometry.
                    dx = mean[0] - ref[0]
                    dy = mean[1] - ref[1]
                    norm = math.hypot(dx, dy)
                    n_ref = [dx / norm, dy / norm] if norm > 0 else [1.0, 0.0]

                    # Returned-plan upper bound evaluated using the same fixed
                    # normal as the pre-solve certificate.
                    posterior_fixed += projected_probability_with_normal(
                        center,
                        mean,
                        cov,
                        radius,
                        n_ref,
                    )['probability_upper']

                    # Uniform pre-solve tube bound. If ||center-ref|| <= r,
                    # strict collision at radius R is contained in the fixed
                    # reference half-space with inflated radius R+r.
                    uniform += projected_probability_with_normal(
                        ref,
                        mean,
                        cov,
                        radius + tube_radius,
                        n_ref,
                    )['probability_upper']

                    distance = math.dist(center, ref)
                    if distance > tube_radius:
                        inside_tube = False
                        if row['success'] == '1':
                            violations.append(dict(
                                kind='accepted_outside_tube',
                                identity=identity,
                                stage=disc['stage'],
                                disc=disc['disc'],
                                distance=distance,
                                tube_radius=tube_radius,
                            ))

            posterior_adaptive = min(1.0, posterior_adaptive)
            posterior_fixed = min(1.0, posterior_fixed)
            uniform = min(1.0, uniform)

            valid = active and row['success'] == '1'
            adaptive_gap = (
                posterior_adaptive - float(row['b'])
                if active else None
            )
            fixed_gap = (
                posterior_fixed - float(row['b'])
                if active else None
            )

            # The stored pre-solve b must agree with independent recomputation.
            if active and abs(uniform - float(row['b'])) > tolerance:
                violations.append(dict(
                    kind='presolve_bound_mismatch',
                    identity=identity,
                    mode=row['mode'],
                    stored=float(row['b']),
                    recomputed=uniform,
                ))

            # The theorem-relevant inequality is the fixed-normal posterior
            # bound <= the uniform fixed-normal tube bound. Only test it when
            # the returned plan actually satisfies the tube premise.
            if valid and inside_tube and fixed_gap > tolerance:
                violations.append(dict(
                    kind='fixed_normal_exceeds_uniform',
                    identity=identity,
                    mode=row['mode'],
                    gap=fixed_gap,
                ))

            # The adaptive posterior can legitimately exceed the fixed-normal
            # uniform bound because it uses a different projection direction.
            # Record it as a diagnostic only; do not invalidate the report.
            adaptive_exceeded = valid and adaptive_gap > tolerance

            if not valid:
                fixed_status = 'not_applicable'
            elif not inside_tube:
                fixed_status = 'outside_tube'
            elif fixed_gap <= tolerance:
                fixed_status = 'pass'
            else:
                fixed_status = 'FAIL'

            bounds.append(dict(
                zip(KEY, identity),
                mode=row['mode'],
                accepted=row['success'],
                active=active,
                inside_tube=inside_tube,
                posterior_adaptive_b=posterior_adaptive,
                posterior_fixed_normal_b=posterior_fixed,
                uniform_b=row['b'],
                recomputed_uniform_b=uniform if active else '',
                adaptive_minus_uniform=adaptive_gap if active else '',
                fixed_minus_uniform=fixed_gap if active else '',
                adaptive_exceeds_uniform=adaptive_exceeded,
                fixed_normal_inequality_status=fixed_status,
            ))

    # Require complete cohorts, allowing termination only after an explicit failure.
    cohorts = defaultdict(list)
    for identity, pair in grouped.items():
        cohorts[identity[:4]].append(pair[0])

    expected_cohorts = {
        (str(s), str(rep), arm, format(float(radius), '.17g'))
        for s in manifest['seeds']
        for rep in range(manifest['repeats'])
        for arm in ['nominal_resampling', 'wdro']
        for radius in manifest['radii']
    }
    if set(cohorts) != expected_cohorts:
        violations.append(dict(kind='missing_or_extra_cohort'))

    for identity, rows in cohorts.items():
        rows.sort(key=lambda r: int(r['cycle']))
        if (
            [int(r['cycle']) for r in rows] != list(range(len(rows)))
            or (len(rows) != manifest['cycles'] and rows[-1]['success'] != '0')
            or any(r['success'] == '0' for r in rows[:-1])
        ):
            violations.append(dict(kind='incomplete_cohort', identity=identity))

    for identity, pair in grouped.items():
        if len(pair) != 2 or {r['mode'] for r in pair} != {'safe', 'cut'}:
            violations.append(dict(kind='missing_mode', identity=identity))
        if any(int(r['sampled_scenarios']) != manifest['scenario_budget'] for r in pair):
            violations.append(dict(kind='scenario_budget_changed', identity=identity))
        if (identity, '0') not in tries:
            violations.append(dict(kind='missing_attempt', identity=identity))

    for data in [cycles, geometry, attempts]:
        signature = lambda rep: [
            {k: v for k, v in r.items() if k not in ('repeat', 'solve_seconds')}
            for r in data if r['repeat'] == str(rep)
        ]
        if signature(0) != signature(1):
            violations.append(dict(kind='repeat_mismatch'))

    write(output/'bound_checks.csv', bounds)

    @lru_cache(None)
    def cached(snapshot):
        return analyze(json.loads(snapshot), backend)

    @lru_cache(None)
    def baseline(nbar, removal, beta):
        return int(float(subprocess.check_output([
            str(backend), 'size', '.05', str(beta), str(nbar), str(removal)
        ], text=True).split()[0]))

    decomposition, curves = [], []
    for (identity, attempt), pair in tries.items():
        original = grouped[identity]
        if original[0]['active'] != '1':
            continue

        bm = {r['mode']: float(r['b']) for r in original}
        if len(pair) != 2 or {r['mode'] for r in pair} != {'safe', 'cut'}:
            raise ValueError('incomplete attempt modes')

        pair.sort(key=lambda r: r['mode'])
        p = [float(r['p']) for r in pair]
        q = [float(r['q']) for r in pair]
        b = [bm[r['mode']] for r in pair]
        nbar = int(pair[0]['n_bar'])
        removal = int(pair[0]['removal_budget'])

        # The identical b vector and p are used for both sampling-law evaluations.
        for law, weights in [('nominal_same_geometry', p), ('actual_attempt', q)]:
            snapshot = dict(
                obstacles=1,
                modes=[
                    dict(
                        mode=r['mode'],
                        count=manifest['history'][r['mode']],
                        p_hat=p[i],
                        q=weights[i],
                        b=b[i],
                    )
                    for i, r in enumerate(pair)
                ],
                nonremoved_support_cap=nbar,
                removal_budget=removal,
                b_justification=(
                    'Fixed pre-sampling tube; synthetic held affine Gaussian '
                    'fixture, conditional diagnostic only.'
                ),
            )
            result = cached(json.dumps(snapshot, sort_keys=True))
            vv = vertices(
                [m['L'] for m in result['modes']],
                [m['U'] for m in result['modes']],
            )
            direct = max(
                sum(pi * bi for pi, bi in zip(v, b))
                for v in vv
            )
            nominal_mass = sum(pi * bi for pi, bi in zip(p, b))
            actual_mass = sum(qi * bi for qi, bi in zip(q, b))
            base = dict(zip(KEY, identity), attempt=attempt, law=law)

            decomposition.append(dict(
                base,
                dro_enabled=pair[0]['dro_enabled'],
                accepted=original[0]['success'],
                rho=pair[0]['rho'],
                n_bar=nbar,
                removal_budget=removal,
                total_support_cap=nbar + removal,
                observed_final_support=pair[0]['observed_final_support'],
                max_b=max(b),
                all_b_lt_one=all(x < 1 for x in b),
                near_one=any(x >= 1 - 1e-6 for x in b),
                p_dot_b=nominal_mass,
                q_dot_b=actual_mass,
                redistribution_delta=actual_mass - nominal_mass,
                direct_cp_bound=direct,
                psi_epsilon=result['psi_at_epsilon'],
                psi_known_p=transfer_at(p, weights, b, .05),
                cp_transfer_excess=(
                    result['psi_at_epsilon'] - transfer_at(p, weights, b, .05)
                ),
                epsilon_Q_max=result['epsilon_Q_max'],
                S_hypothetical=(
                    result['S_WDRO']['S'] if result['S_WDRO'] else ''
                ),
                S_baseline=result['S_SH']['S'],
                ratio=result['scenario_count_ratio'],
                S_baseline_same_total_confidence=baseline(nbar, removal, .06),
                direct_bound_already_sufficient=direct <= .05,
                beta_cp=.05,
                beta_cert=.01,
                combined_failure_budget=.06,
                actual_S=original[0]['sampled_scenarios'],
                certificate_issued=False,
            ))

            eta_values = [0, .01, .025, .05, .075, .1, .2, .5, 1]
            if result['epsilon_Q_max'] is not None:
                eta_values.append(result['epsilon_Q_max'])
            for eta in sorted(set(eta_values)):
                curves.append(dict(
                    base,
                    eta=eta,
                    psi_cp_box=max(
                        transfer_at(v, weights, b, eta)
                        for v in vv
                    ),
                    psi_known_p=transfer_at(p, weights, b, eta),
                ))

    write(output/'decomposition.csv', decomposition)
    write(output/'transfer_curves.csv', curves)

    summary = []
    for arm, radius in sorted(
        {(r['arm'], r['radius']) for r in cycles},
        key=lambda x: (x[0], float(x[1])),
    ):
        decisions = [
            v[0]
            for v in grouped.values()
            if v[0]['arm'] == arm
            and v[0]['radius'] == radius
            and v[0]['repeat'] == '0'
        ]
        active = [r for r in decisions if r['active'] == '1']
        ds = [
            r for r in decomposition
            if r['arm'] == arm
            and r['radius'] == radius
            and r['repeat'] == '0'
            and r['attempt'] == '0'
            and r['law'] == 'actual_attempt'
        ]
        sizes = [
            r['S_hypothetical']
            for r in ds
            if r['S_hypothetical'] != ''
        ]
        first_attempt = next(
            v[0]
            for (identity, _), v in tries.items()
            if identity[2] == arm and identity[3] == radius
        )
        baseline_s = baseline(
            int(first_attempt['n_bar']),
            int(first_attempt['removal_budget']),
            .01,
        )
        summary.append(dict(
            arm=arm,
            radius=float(radius),
            unique_decisions=len(decisions),
            active_decisions=len(active),
            tube_rejection_rate=(
                sum(r['rejected'] == '1' for r in active) / len(active)
                if active else ''
            ),
            all_b_lt_one_fraction=(
                sum(r['all_b_lt_one'] for r in ds) / len(ds)
                if ds else ''
            ),
            S_hypothetical_median=statistics.median(sizes) if sizes else '',
            S_hypothetical_min=min(sizes) if sizes else '',
            S_hypothetical_max=max(sizes) if sizes else '',
            no_finite_S_count=len(ds) - len(sizes),
            S_baseline=baseline_s,
            accepted_informative_saving_fraction=(
                sum(
                    r['accepted'] == '1'
                    and r['all_b_lt_one']
                    and r['ratio'] is not None
                    and r['ratio'] < 1
                    for r in ds
                ) / len(ds)
                if ds else ''
            ),
        ))

    write(output/'radius_summary.csv', summary)

    os.environ.setdefault(
        'MPLCONFIGDIR',
        str((output/'matplotlib-cache').resolve()),
    )
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for arm in ['nominal_resampling', 'wdro']:
        data = [r for r in summary if r['arm'] == arm and r['active_decisions']]
        xs = [r['radius'] for r in data]
        for ax, field in zip(
            axes,
            ['tube_rejection_rate', 'all_b_lt_one_fraction', 'S_hypothetical_median'],
        ):
            ax.plot(xs, [r[field] for r in data], marker='o', label=arm)

    axes[0].set_ylabel('Tube rejection fraction (active solves)')
    axes[1].set_ylabel('Fraction with all mode bounds < 1')
    axes[2].set_ylabel('Hypothetical S (median)')
    axes[2].set_yscale('log')

    for baseline_s in sorted({r['S_baseline'] for r in summary}):
        axes[2].axhline(
            baseline_s,
            color='black',
            linestyle='--',
            label=f'Baseline S = {baseline_s}',
        )
    for ax in axes:
        ax.set_xlabel('Tube radius (m)')
        ax.grid(alpha=.25)
    axes[2].legend(fontsize=7)
    fig.suptitle(
        'Conditional diagnostics; fixed actual budget; sample reduction is not claimed'
    )
    fig.savefig(output/'radius_decomposition.pdf')
    fig.savefig(output/'radius_decomposition.png', dpi=160)
    plt.close(fig)

    fixed_normal_exceedances = sum(
        r['fixed_normal_inequality_status'] == 'FAIL'
        for r in bounds
    )
    adaptive_posterior_exceedances = sum(
        bool(r['adaptive_exceeds_uniform'])
        for r in bounds
    )

    validation = dict(
        status='PASS' if not violations else 'FAIL',
        violations=violations,
        accepted_mode_inequalities=sum(
            r['fixed_normal_inequality_status'] == 'pass'
            for r in bounds
        ),
        fixed_normal_exceedances=fixed_normal_exceedances,
        # Backward-compatible alias: now theorem-relevant fixed-normal failures.
        posterior_exceedances=fixed_normal_exceedances,
        # Diagnostic only. This does not make validation fail.
        adaptive_posterior_exceedances=adaptive_posterior_exceedances,
        comparison_tolerance=tolerance,
        presolve_recomputation_tolerance=tolerance,
        limitation=(
            'Gaussian projection upper bounds, not exact collision probabilities. '
            'The hard posterior-vs-uniform check uses one fixed pre-solve reference '
            'normal; adaptive returned-plan projection bounds are diagnostic only. '
            'CP-box transfer, not a Wasserstein-ball optimization. Synthetic '
            'histories; unequal total confidence budgets; no reduced-budget certificate.'
        ),
        input_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                source/'cycles.csv',
                source/'disc_geometry.csv',
                source/'attempts.csv',
                source/'predictions.csv',
            ]
        },
        source_manifest_sha256=hashlib.sha256(
            (output/'source_manifest.json').read_bytes()
        ).hexdigest(),
        backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
    )

    (output/'validation.json').write_text(
        json.dumps(validation, indent=2) + '\n'
    )
    print(json.dumps({
        k: validation[k]
        for k in [
            'status',
            'accepted_mode_inequalities',
            'fixed_normal_exceedances',
            'adaptive_posterior_exceedances',
        ]
    }))
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new report directory')
    parser.add_argument(
        '--backend',
        type=Path,
        default=Path('build-tube/certificate_numeric_backend'),
    )
    args = parser.parse_args()
    result = report(args.input, args.output, args.backend.resolve())
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
