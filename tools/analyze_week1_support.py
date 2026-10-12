#!/usr/bin/env python3
"""Week 1: audit SH labels and describe support versus pre-sampling context.

No predictor is trained and no controller/certificate formula is changed.
Sample-derived geometry, current support and post-solve status are outcomes,
never context features. Correlations are descriptive, not IID significance tests.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        path.write_text('')
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader()
        writer.writerows(records)


def context(cfg, actors):
    ego = next((r for r in actors if r['actor'] == 'ego'), None)
    obstacles = [r for r in actors if r['actor'] == 'obstacle']
    if ego is None:
        return {}
    x, y, theta, v = [float(ego[k]) for k in ('x', 'y', 'theta', 'v')]
    horizon = float(cfg['dt']) * int(cfg['horizon'])
    closest, encounter = math.inf, math.nan
    distance = math.inf
    for obstacle in obstacles:
        relative = np.array([float(obstacle['x']) - x, float(obstacle['y']) - y])
        velocity = np.array([float(obstacle['vx']) - v * math.cos(theta),
                             float(obstacle['vy']) - v * math.sin(theta)])
        speed_squared = float(velocity @ velocity)
        time = float(np.clip(-relative @ velocity / speed_squared, 0, horizon)) if speed_squared else 0.0
        approach = float(np.linalg.norm(relative + time * velocity))
        distance = min(distance, float(np.linalg.norm(relative)))
        if approach < closest:
            closest, encounter = approach, time
    return dict(obstacle_count=len(obstacles), mode_count=len(cfg.get('obs_modes', [])),
                environment=cfg.get('environment', 'unknown'), ego_speed=v,
                road_width=cfg.get('road_width'), horizon_seconds=horizon,
                closest_observed_center_distance=distance if obstacles else math.nan,
                cv_closest_approach=closest if obstacles else math.nan,
                cv_encounter_time=encounter, prediction_noise=bool(cfg.get('obstacle_prediction_noise', True)),
                prediction_model=cfg.get('obstacle_prediction_model', 'framework'),
                position_noise_variance=(cfg.get('obstacle_position_noise_variance')
                                         if cfg.get('obstacle_prediction_model') == 'gaussian_position' else None))


def extract(roots):
    inventory, decisions = [], []
    seen = set()
    for root in roots:
        for support_path in sorted(root.rglob('support_scenarios.csv')):
            bundle = support_path.parent.resolve()
            if bundle in seen:
                continue
            seen.add(bundle)
            cfg_path = bundle / 'resolved_config.yaml'
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
            support = read_csv(support_path)
            eligible = (cfg.get('safe_horizon_enabled') is True and
                        cfg.get('mpc_type') == 'sh_mpcc' and cfg.get('dro_enabled') is False)
            reason = ('eligible_base_sh' if eligible else
                      'safe_horizon_disabled' if cfg.get('safe_horizon_enabled') is False else
                      'missing_config_or_other_controller')
            inventory.append(dict(bundle=str(bundle), root=str(root), reason=reason,
                                  decisions=len(support), evaluated=sum(r['support_evaluated'] == '1' for r in support)))
            if not eligible:
                continue
            traces = defaultdict(list)
            for actor in read_csv(bundle / 'trace.csv'):
                traces[int(actor['step'])].append(actor)
            solves = {int(r['step']): r for r in read_csv(bundle / 'decisions.csv')}
            previous_support = None
            total_cap = int(cfg['support_cap_n_bar']) + int(cfg['scenario_removal_budget'])
            run = read_csv(bundle / 'rollout.csv')
            for label in sorted(support, key=lambda r: int(r['step'])):
                step = int(label['step'])
                solve = solves.get(step, {})
                evaluated = label['support_evaluated'] == '1'
                value = int(label['support_count']) if evaluated else None
                certified = solve.get('certified') == '1'
                record = dict(bundle=str(bundle), seed=run[0].get('seed') if run else None,
                              step=step, context_available=any(a['actor'] == 'ego' for a in traces[step]),
                              support_evaluated=evaluated, support=value, total_support_cap=total_cap,
                              previous_support=previous_support, success=solve.get('success') == '1',
                              certified=certified, accepted_certified=certified and solve.get('success') == '1',
                              solve_ms=solve.get('solve_ms'), scenario_count=solve.get('scenario_count'),
                              last_qp_converged=solve.get('last_qp_converged'),
                              termination_reason=run[0].get('termination_reason') if run else None,
                              collision=run[0].get('collision') if run else None)
                record.update(context(cfg, traces[step]))
                decisions.append(record)
                previous_support = value
    return inventory, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis_provenance.json').write_text(json.dumps(dict(
        roots=[str(p.resolve()) for p in args.root],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, pandas_version=pd.__version__,
        matplotlib_version=matplotlib.__version__), indent=2) + '\n')
    inventory, records = extract(args.root)
    write_csv(args.output / 'artifact_inventory.csv', inventory)
    write_csv(args.output / 'decisions.csv', records)
    inv = pd.DataFrame(inventory)
    summary = dict(bundles=len(inventory), exclusion_counts=inv.reason.value_counts().to_dict() if len(inv) else {},
                   extracted_decisions=len(records))
    data = pd.DataFrame(records)
    if len(data):
        summary.update(evaluated_decisions=int(data.support_evaluated.sum()),
                       unevaluated_decisions=int((~data.support_evaluated).sum()),
                       rejected_decisions=int((~data.success).sum()),
                       executable_uncertified_decisions=int((data.success & ~data.certified).sum()))
        data.groupby(['support_evaluated', 'success', 'certified']).size().rename('decisions').reset_index().to_csv(
            args.output / 'decision_outcomes.csv', index=False)
        data.groupby(['environment', 'obstacle_count', 'mode_count']).agg(
            runs=('bundle', 'nunique'), decisions=('support_evaluated', 'size'),
            evaluated=('support_evaluated', 'sum'), certified=('accepted_certified', 'sum'),
            executable=('success', 'sum')).reset_index().to_csv(args.output / 'scene_outcomes.csv', index=False)
    accepted = (data[data.accepted_certified & data.support_evaluated & data.context_available].copy()
                if len(data) else pd.DataFrame())
    summary['accepted_evaluated_decisions'] = len(accepted)
    if len(accepted):
        accepted['support'] = pd.to_numeric(accepted.support)
        accepted['solve_ms'] = pd.to_numeric(accepted.solve_ms)
        summary.update(accepted_runs=accepted.bundle.nunique(), accepted_seeds=accepted.seed.nunique(),
                       support_min=int(accepted.support.min()), support_max=int(accepted.support.max()),
                       support_median=float(accepted.support.median()),
                       support_quantiles={str(q): float(accepted.support.quantile(q)) for q in [0.1, 0.5, 0.9, 0.95]},
                       fraction_support_1_to_4=float(accepted.support.between(1, 4).mean()),
                       fraction_support_8_to_10=float(accepted.support.between(8, 10).mean()),
                       fraction_support_zero=float((accepted.support == 0).mean()),
                       scenario_counts=sorted(int(v) for v in accepted.scenario_count.unique()),
                       solve_median_ms=float(accepted.solve_ms.median()))
        grouped = accepted.groupby(['environment', 'obstacle_count', 'mode_count']).agg(
            decisions=('support', 'size'), runs=('bundle', 'nunique'), median=('support', 'median'),
            mean=('support', 'mean'), maximum=('support', 'max'), solve_median_ms=('solve_ms', 'median')).reset_index()
        grouped.to_csv(args.output / 'context_groups.csv', index=False)
        runs = accepted.groupby('bundle').agg(
            seed=('seed', 'first'), environment=('environment', 'first'), obstacle_count=('obstacle_count', 'first'),
            mode_count=('mode_count', 'first'), support=('support', 'median'),
            closest_observed_center_distance=('closest_observed_center_distance', 'median'),
            cv_closest_approach=('cv_closest_approach', 'median'), ego_speed=('ego_speed', 'median'))
        runs.to_csv(args.output / 'accepted_run_summaries.csv')
        associations = []
        for feature in ['obstacle_count', 'mode_count', 'closest_observed_center_distance', 'cv_closest_approach', 'ego_speed']:
            subset = runs[[feature, 'support']].dropna()
            rho = subset.corr(method='spearman').iloc[0, 1] if len(subset) > 2 else math.nan
            associations.append(dict(feature=feature, runs=len(subset), spearman_rho=None if pd.isna(rho) else float(rho)))
        write_csv(args.output / 'descriptive_associations.csv', associations)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        counts = accepted.support.value_counts().sort_index()
        axes[0].bar(counts.index, counts.values)
        axes[0].set(xlabel='Returned support (certified decisions)', ylabel='Decision count')
        axes[1].scatter(accepted.cv_closest_approach, accepted.support, alpha=0.45)
        axes[1].set(xlabel='Pre-sampling CV closest approach (m)', ylabel='Returned support')
        axes[2].scatter(accepted.previous_support, accepted.support, alpha=0.45)
        axes[2].set(xlabel='Previous evaluated support', ylabel='Current returned support')
        fig.tight_layout()
        fig.savefig(args.output / 'support_context.png', dpi=180)
        fig.savefig(args.output / 'support_context.pdf')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(grouped)), grouped['mean'])
        ax.set_xticks(range(len(grouped)), [f'{r.environment}\nO={r.obstacle_count}, M={r.mode_count}' for r in grouped.itertuples()])
        ax.set(ylabel='Mean returned support', title='Descriptive context groups; repeated decisions are dependent')
        fig.tight_layout()
        fig.savefig(args.output / 'support_by_scene.png', dpi=180)
        fig.savefig(args.output / 'support_by_scene.pdf')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(accepted.support, accepted.solve_ms, alpha=0.45)
        ax.set(xlabel='Returned support (certified decisions)', ylabel='Solve time (ms)',
               title='Solve time is an outcome, not a context feature')
        fig.tight_layout()
        fig.savefig(args.output / 'support_runtime.png', dpi=180)
        fig.savefig(args.output / 'support_runtime.pdf')
        plt.close(fig)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
