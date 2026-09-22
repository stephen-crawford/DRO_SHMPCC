#!/usr/bin/env python3
"""
Measure whether DRO anticipates realized obstacle behavior switches.

For every realized plant transition

    A -> B

this tool looks N decisions before the switch and records

    p_t(B)
    q*_t(B)
    q*_t(B) - p_t(B)
    risk_t(B)

The key scientific question is whether modes that are dangerous at the
pre-switch state receive additional q* probability before the plant actually
switches into them.

This is a read-only post-processing tool. It modifies neither controller.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import run_analysis_matrix as analysis


def parse_tokens(line, marker):
    if marker not in line:
        return None

    tail = line.split(marker, 1)[1].strip()
    result = {}

    for token in tail.split():
        if '=' not in token:
            continue

        key, value = token.split('=', 1)
        result[key] = value

    return result


def parse_dro_log(path):
    """
    Returns

        mode_records[(decision, obstacle, mode)]
        summary_records[(decision, obstacle)]

    The controller log labels decisions one-based, while decisions.csv and the
    matrix tools use zero-based decision indices, hence logged step - 1.
    """

    mode_records = {}
    summary_records = {}

    with path.open() as stream:
        for line in stream:
            fields = parse_tokens(line, '[DRO MODE]')

            if fields is not None:
                required = {
                    'step',
                    'obstacle',
                    'mode',
                    'p',
                    'risk',
                    'q',
                }

                if not required.issubset(fields):
                    continue

                decision = int(fields['step']) - 1
                obstacle = int(fields['obstacle'])
                mode = fields['mode']

                p = float(fields['p'])
                q = float(fields['q'])

                mode_records[
                    (decision, obstacle, mode)
                ] = {
                    'p': p,
                    'q': q,
                    'delta_q': float(
                        fields.get(
                            'delta_q',
                            q - p)),
                    'risk': float(fields['risk']),
                    'q_times_S': float(
                        fields.get(
                            'q_times_S',
                            'nan')),
                }

                continue

            fields = parse_tokens(line, '[DRO SUMMARY]')

            if fields is not None:
                required = {
                    'step',
                    'obstacle',
                }

                if not required.issubset(fields):
                    continue

                decision = int(fields['step']) - 1
                obstacle = int(fields['obstacle'])

                def number(name):
                    return float(
                        fields.get(name, 'nan'))

                summary_records[
                    (decision, obstacle)
                ] = {
                    'rho': number('rho'),
                    'rho_raw': number('rho_raw'),
                    'transport_cost':
                        number('transport_cost'),
                    'budget_usage':
                        number('budget_usage'),
                    'tv': number('tv'),
                    'tv_bound': number('tv_bound'),
                    'nominal_risk':
                        number('nominal_risk'),
                    'qstar_risk':
                        number('qstar_risk'),
                    'risk_lift':
                        number('risk_lift'),
                    'radius_m':
                        number('radius_m'),
                }

    return mode_records, summary_records


def realized_switches(bundle):
    """
    Extract actual mode transitions from trace.csv.

    switch_step is the first trace step at which the new mode appears.
    """

    trajectories = {}

    for row in analysis.rows(bundle / 'trace.csv'):
        if row['actor'] != 'obstacle':
            continue

        obstacle = int(row['obstacle_id'])

        trajectories.setdefault(
            obstacle, []
        ).append((
            int(row['step']),
            row['mode'],
        ))

    events = []

    for obstacle, trajectory in sorted(
            trajectories.items()):

        trajectory.sort()

        previous_step = None
        previous_mode = None

        for step, mode in trajectory:
            if (
                previous_mode is not None
                and mode != previous_mode
            ):
                events.append({
                    'obstacle_id': obstacle,
                    'switch_step': step,
                    'from_mode': previous_mode,
                    'to_mode': mode,
                    'previous_step':
                        previous_step,
                })

            previous_step = step
            previous_mode = mode

    return events


def result_paths(root):
    """
    Old analysis matrix:

        sh_mpcc_dro_roundabout.../
            seed_79/
                result.json

    New comparison matrix:

        roundabout_o2..._boost_50/
            seed_79/
                sh_mpcc_dro/
                    result.json
    """

    paths = set(
        root.glob('*/seed_*/result.json')
    )

    paths.update(
        root.glob(
            '*/seed_*/sh_mpcc*/result.json')
    )

    return sorted(paths)


def comparison_outcomes(root):
    path = root / 'pairs.json'

    if not path.exists():
        return {}

    output = {}

    for row in json.loads(path.read_text()):
        output[
            (row['pair'], int(row['seed']))
        ] = row

    return output


def finite_mean(values):
    usable = []

    for value in values:
        if value is None:
            continue

        value = float(value)

        if math.isfinite(value):
            usable.append(value)

    return (
        statistics.mean(usable)
        if usable else None
    )


def collect(
    root,
    pair_filter,
    seeds,
    lead_steps,
    risk_threshold,
    delta_tolerance,
):
    observations = []
    missing = []

    paired_outcomes = comparison_outcomes(root)

    for result_path in result_paths(root):
        result = json.loads(
            result_path.read_text())

        if (
            result.get('solver_style')
            != 'sh_mpcc_dro'
        ):
            continue

        if result.get('status') != 'OK':
            continue

        pair = result.get(
            'pair',
            result['case'].removeprefix(
                'sh_mpcc_dro_'))

        seed = int(result['seed'])

        if (
            pair_filter is not None
            and pair != pair_filter
        ):
            continue

        if (
            seeds is not None
            and seed not in seeds
        ):
            continue

        profile = result.get(
            'profile',
            'analysis_matrix')

        controller_root = result_path.parent

        bundle = controller_root / 'repeat_0'

        log = controller_root / 'repeat_0.log'

        if not bundle.exists() or not log.exists():
            missing.append({
                'pair': pair,
                'seed': seed,
                'reason':
                    'missing repeat_0 or repeat_0.log',
            })
            continue

        mode_records, summary_records = \
            parse_dro_log(log)

        switches = realized_switches(bundle)

        paired = paired_outcomes.get(
            (pair, seed),
            {})

        for event_index, switch in enumerate(
                switches):

            obstacle = switch['obstacle_id']

            target_mode = switch['to_mode']

            switch_step = switch['switch_step']

            for lead in lead_steps:
                decision = switch_step - lead

                if decision < 0:
                    continue

                key = (
                    decision,
                    obstacle,
                    target_mode,
                )

                if key not in mode_records:
                    missing.append({
                        'pair': pair,
                        'seed': seed,
                        'event_index':
                            event_index,
                        'obstacle_id':
                            obstacle,
                        'switch_step':
                            switch_step,
                        'lead_steps':
                            lead,
                        'target_mode':
                            target_mode,
                        'reason':
                            'missing DRO mode evidence',
                    })
                    continue

                mode = mode_records[key]

                summary = summary_records.get(
                        (decision, obstacle),
                        {})

                target_risk = mode['risk']

                delta_q = mode['delta_q']

                row = {
                    'pair': pair,
                    'profile': profile,
                    'seed': seed,
                    'event_index':
                        event_index,
                    'obstacle_id':
                        obstacle,

                    'switch_step':
                        switch_step,
                    'from_mode':
                        switch['from_mode'],
                    'to_mode':
                        target_mode,

                    'lead_steps':
                        lead,
                    'decision_step':
                        decision,

                    'p_target':
                        mode['p'],
                    'qstar_target':
                        mode['q'],
                    'delta_q_target':
                        delta_q,
                    'target_mode_risk':
                        target_risk,

                    'target_upweighted':
                        int(
                            delta_q >
                            delta_tolerance),

                    'target_downweighted':
                        int(
                            delta_q <
                            -delta_tolerance),

                    'dangerous_target':
                        int(
                            target_risk >=
                            risk_threshold),

                    'rho':
                        summary.get('rho'),

                    'transport_cost':
                        summary.get(
                            'transport_cost'),

                    'budget_usage':
                        summary.get(
                            'budget_usage'),

                    'nominal_risk':
                        summary.get(
                            'nominal_risk'),

                    'qstar_risk':
                        summary.get(
                            'qstar_risk'),

                    'risk_lift':
                        summary.get(
                            'risk_lift'),

                    'tv':
                        summary.get('tv'),

                    'tv_bound':
                        summary.get(
                            'tv_bound'),
                }

                # Add paired outcome information when
                # analyzing a comparison matrix.
                for field in (
                    'nondro_collision',
                    'dro_collision',
                    'nondro_safe_completion',
                    'dro_safe_completion',
                    'nondro_termination_reason',
                    'dro_termination_reason',
                ):
                    if field in paired:
                        row[field] = paired[field]

                observations.append(row)

    return observations, missing


def summarize_group(group, label):
    dangerous = [
        row for row in group
        if row['dangerous_target']
    ]

    benign = [
        row for row in group
        if not row['dangerous_target']
    ]

    dangerous_delta = finite_mean(
            row['delta_q_target']
            for row in dangerous)

    benign_delta = finite_mean(
            row['delta_q_target']
            for row in benign)

    return {
        **label,

        'observations':
            len(group),

        'switch_events':
            len({
                (
                    row['pair'],
                    row['seed'],
                    row['event_index'],
                )
                for row in group
            }),

        'target_upweighted_rate':
            (
                sum(
                    row['target_upweighted']
                    for row in group)
                / len(group)
                if group else None
            ),

        'mean_delta_q_target':
            finite_mean(
                row['delta_q_target']
                for row in group),

        'mean_target_mode_risk':
            finite_mean(
                row['target_mode_risk']
                for row in group),

        'mean_risk_lift':
            finite_mean(
                row['risk_lift']
                for row in group),

        'dangerous_observations':
            len(dangerous),

        'dangerous_target_upweighted_rate':
            (
                sum(
                    row['target_upweighted']
                    for row in dangerous)
                / len(dangerous)
                if dangerous else None
            ),

        'dangerous_mean_delta_q':
            dangerous_delta,

        'benign_observations':
            len(benign),

        'benign_target_upweighted_rate':
            (
                sum(
                    row['target_upweighted']
                    for row in benign)
                / len(benign)
                if benign else None
            ),

        'benign_mean_delta_q':
            benign_delta,

        # Desired sign is positive.
        'risk_selectivity_delta':
            (
                dangerous_delta -
                benign_delta
                if (
                    dangerous_delta
                    is not None
                    and benign_delta
                    is not None
                )
                else None
            ),
    }


def build_summaries(observations):
    summaries = []

    leads = sorted({
        row['lead_steps']
        for row in observations
    })

    for lead in leads:
        group = [
            row
            for row in observations
            if row['lead_steps'] == lead
        ]

        summaries.append(
            summarize_group(
                group,
                {
                    'scope': 'all',
                    'profile': 'all',
                    'lead_steps': lead,
                }))

    profiles = sorted({
        row['profile']
        for row in observations
    })

    for profile in profiles:
        for lead in leads:
            group = [
                row
                for row in observations
                if (
                    row['profile'] == profile
                    and
                    row['lead_steps'] == lead
                )
            ]

            if not group:
                continue

            summaries.append(
                summarize_group(
                    group,
                    {
                        'scope': 'profile',
                        'profile': profile,
                        'lead_steps': lead,
                    }))

    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument(
        '--matrix',
        type=Path,
        required=True)

    parser.add_argument(
        '--output',
        type=Path,
        required=True)

    parser.add_argument(
        '--pair')

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+')

    parser.add_argument(
        '--lead-steps',
        default='1,2,5,10')

    parser.add_argument(
        '--risk-threshold',
        type=float,
        default=0.01)

    parser.add_argument(
        '--delta-tolerance',
        type=float,
        default=1e-9)

    args = parser.parse_args(argv)

    args.matrix = args.matrix.resolve()

    args.output = args.output.resolve()

    try:
        lead_steps = sorted({
            int(value)
            for value
            in args.lead_steps.split(',')
        })
    except ValueError:
        parser.error(
            '--lead-steps must be comma-separated integers')

    if (
        not lead_steps
        or any(step < 0 for step in lead_steps)
        or not math.isfinite(
            args.risk_threshold)
        or not math.isfinite(
            args.delta_tolerance)
        or args.delta_tolerance < 0
    ):
        parser.error(
            'invalid lead steps or thresholds')

    observations, missing = collect(
        args.matrix,
        args.pair,
        (
            set(args.seeds)
            if args.seeds is not None
            else None
        ),
        lead_steps,
        args.risk_threshold,
        args.delta_tolerance,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True)

    analysis.write_csv(
        args.output /
        'switch_anticipation.csv',
        observations)

    analysis.dump_json(
        args.output /
        'switch_anticipation.json',
        observations)

    summaries = build_summaries(observations)

    analysis.write_csv(
        args.output /
        'switch_anticipation_summary.csv',
        summaries)

    analysis.dump_json(
        args.output /
        'switch_anticipation_summary.json',
        summaries)

    analysis.write_csv(
        args.output /
        'missing_switch_evidence.csv',
        missing)

    report = {
        'observations':
            len(observations),

        'switch_events':
            len({
                (
                    row['pair'],
                    row['seed'],
                    row['event_index'],
                )
                for row in observations
            }),

        'missing_records':
            len(missing),

        'lead_steps':
            lead_steps,

        'risk_threshold':
            args.risk_threshold,
    }

    analysis.dump_json(
        args.output /
        'switch_validation_report.json',
        report)

    print(
        json.dumps(
            report,
            indent=2))

    return 0 if observations else 1


if __name__ == '__main__':
    sys.exit(main())