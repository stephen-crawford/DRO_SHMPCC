#!/usr/bin/env python3
"""Plot actual frozen-context budget responses, including failures and nonmonotonicity."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'analysis_provenance.json').write_text(json.dumps(dict(
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pandas_version=pd.__version__, matplotlib_version=matplotlib.__version__), indent=2)+'\n')
    responses, contexts, checks, baselines, rollouts = [], [], [], [], []
    for run in sorted(root.glob('*/seed_*')):
        if not (run / 'budget_responses.csv').exists():
            continue
        for filename, target in [('budget_responses.csv',responses),('contexts.csv',contexts),
                                 ('clone_checks.csv',checks),('baseline_decisions.csv',baselines),('summary.csv',rollouts)]:
            if (run / filename).exists():
                data = pd.read_csv(run / filename)
                data['case'] = run.parent.name
                target.append(data)
    data = pd.concat(responses, ignore_index=True)
    context = pd.concat(contexts, ignore_index=True)
    check = pd.concat(checks, ignore_index=True)
    baseline = pd.concat(baselines, ignore_index=True)
    rollout = pd.concat(rollouts, ignore_index=True)
    curve = pd.read_csv(root / 'scenario_budget_curve.csv')
    assert check.rng_unchanged.eq(1).all() and check.baseline_replay_matches.eq(1).all()
    assert not context.duplicated(['case','seed','step']).any()
    assert not data.duplicated(['case','seed','step','replicate','h']).any()
    assert data.groupby(['case','seed','step','replicate']).h.nunique().eq(8).all()
    assert data.groupby(['case','seed','step','replicate']).sampling_seed.nunique().eq(8).all()
    mapping = dict(zip(curve.h, curve.S))
    assert data.S.eq(data.h.map(mapping)).all()
    accepted = data.certified.eq(1) & data.success.eq(1) & data.fallback.eq(0) & data.cap_ok.eq(1)
    assert (data.certified.eq(1) == accepted).all()
    assert data.loc[accepted,'support_evaluated'].eq(1).all()
    assert (data.loc[accepted,'support'] <= data.loc[accepted,'h']).all()
    data['accepted'] = accepted
    data['support_label'] = data.support.where(data.support_evaluated.eq(1))
    data.to_csv(out / 'budget_responses.csv',index=False)
    context.to_csv(out / 'contexts.csv',index=False)
    rollout.to_csv(out / 'baseline_rollouts.csv',index=False)
    baseline.to_csv(out / 'baseline_decisions.csv',index=False)
    check.to_csv(out / 'clone_checks.csv',index=False)
    rates = data.groupby(['h','S']).agg(attempts=('accepted','size'),
        accepted=('accepted','sum'), acceptance_rate=('accepted','mean'),
        support_cap_rate=('cap_ok','mean'), median_ms=('solve_ms','median'),
        p95_ms=('solve_ms',lambda x:x.quantile(.95)), cap_terminated=('cap_terminated','sum'),
        median_predicted_progress=('predicted_progress','median')).reset_index()
    rates.to_csv(out / 'acceptance_curve.csv',index=False)
    data.groupby(['case','h']).agg(attempts=('accepted','size'), acceptance_rate=('accepted','mean'),
        median_ms=('solve_ms','median')).reset_index().to_csv(out / 'acceptance_by_case.csv',index=False)
    oracle = []
    nonmonotone = 0
    for key, group in data.groupby(['case','seed','step','replicate']):
        group = group.sort_values('h')
        good = group[group.accepted]
        found = len(good)>0
        chosen = good.iloc[0] if found else group[group.h.eq(10)].iloc[0]
        nonmonotone += int(any(a>b for a,b in zip(group.accepted.astype(int),group.accepted.astype(int).iloc[1:])))
        oracle.append(dict(zip(['case','seed','step','replicate'],key)) | dict(
            oracle_found=found, oracle_h=int(chosen.h) if found else None,
            S_accounted=int(chosen.S), solve_ms_selected=float(chosen.solve_ms),
            sampled_clearance=float(chosen.sampled_clearance), predicted_progress=float(chosen.predicted_progress),
            cost=float(chosen.cost)))
    oracle = pd.DataFrame(oracle)
    oracle.to_csv(out / 'oracle.csv',index=False)
    savings = oracle.groupby('case').agg(context_draw_blocks=('oracle_found','size'),
        oracle_found_rate=('oracle_found','mean'), mean_S_accounted=('S_accounted','mean')).reset_index()
    savings['sample_reduction'] = 1-savings.mean_S_accounted/mapping[10]
    savings.to_csv(out / 'oracle_savings.csv',index=False)
    summary = dict(runs=len(rollout), contexts=len(context), attempts=len(data),
        context_draw_blocks=len(oracle), clone_checks=len(check), all_clone_checks_pass=True,
        baseline_steps=int(rollout.steps.sum()), baseline_completed=int(rollout.completed.sum()),
        baseline_collisions=int(rollout.collision.sum()), terminations=rollout.termination.value_counts().to_dict(),
        accepted_probe_attempts=int(accepted.sum()),
        oracle_found_rate=float(oracle.oracle_found.mean()),
        oracle_sample_reduction_all_blocks=float(1-oracle.S_accounted.mean()/mapping[10]),
        oracle_sample_reduction_found_only=float(1-oracle.loc[oracle.oracle_found,'S_accounted'].mean()/mapping[10]),
        nonmonotone_blocks=nonmonotone, nonmonotone_fraction=float(nonmonotone/len(oracle)),
        baseline_certified_fraction=float(baseline.certified.mean()),
        accepted_negative_sampled_clearance=int((accepted & data.sampled_clearance.lt(0)).sum()),
        minimum_accepted_sampled_clearance=float(data.loc[accepted,'sampled_clearance'].min()),
        minimum_accepted_retained_clearance=float(data.loc[accepted,'retained_sampled_clearance'].min()),
        accepted_retained_clearance_below_minus_qp_tolerance=int((accepted & data.retained_sampled_clearance.lt(-0.0001)).sum()))
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def save(name):
        plt.tight_layout()
        plt.savefig(out / (name+'.png'),dpi=180)
        plt.savefig(out / (name+'.pdf'))
        plt.close()
    plt.figure(figsize=(6,4));plt.plot(curve.h,curve.S,'o-')
    plt.xlabel('Total support cap h');plt.ylabel('Production required scenarios S(h)');save('01_scenario_budget_curve')
    plt.figure(figsize=(6,4));plt.plot(rates.h,rates.acceptance_rate,'o-',label='Full implementation certificate')
    plt.plot(rates.h,rates.support_cap_rate,'s--',label='Evaluated support <= h')
    plt.xlabel('Total support cap h');plt.ylabel('Observed attempt fraction');plt.ylim(0,1.05);plt.legend();save('02_acceptance_curve')
    plt.figure(figsize=(6,4));counts=oracle.oracle_h.value_counts().sort_index()
    labels=[str(int(h)) for h in counts.index]+['None']
    values=list(counts.values)+[int((~oracle.oracle_found).sum())]
    plt.bar(labels,values)
    plt.xlabel('Smallest observed certifying h (oracle)');plt.ylabel('Context × replicate blocks');save('03_oracle_distribution')
    plt.figure(figsize=(10,4));plt.bar(savings.case,savings.sample_reduction*100)
    plt.xticks(rotation=20,ha='right');plt.ylabel('Oracle sample reduction (%)');save('04_oracle_savings')
    plt.figure(figsize=(6,4));plt.plot(rates.S,rates.median_ms,'o-',label='Median')
    plt.plot(rates.S,rates.p95_ms,'s--',label='95th percentile')
    plt.xlabel('Scenario count');plt.ylabel('Solve time (ms)');plt.legend();save('05_runtime_by_budget')
    (out / 'validation.txt').write_text('PASS: all clone RNG/replay checks; unique complete 8-cap blocks; production sample counts; certificate/support/fallback gates.\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
