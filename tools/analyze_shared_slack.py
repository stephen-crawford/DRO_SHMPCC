#!/usr/bin/env python3
"""Separate QP convergence, nonlinear augmented feasibility and zero-slack checks."""
import argparse,csv,json,statistics
from pathlib import Path


def analyze(root):
    conflicts=json.load(Path('results/facet-conflict-analysis/conflicts.json').open())['events']
    summaries=[];bound_checks=0
    for seed in (77,78):
        directory=root/f'seed_{seed}'
        rows=list(csv.DictReader((directory/'slack_replays.csv').open()))
        previous={r['replicate']:r for r in csv.DictReader((Path('results/facet-conflict-analysis')/f'seed_{seed}'/'replications.csv').open())}
        assert len(rows)==100 and len({r['replicate'] for r in rows})==100
        assert len({r['sampling_seed'] for r in rows})==100
        first={str(e['replicate']):e for e in conflicts if e['first_event'] and e['context']==f'seed_{seed}'}
        traces=[]
        for row in rows:
            old=previous[row['replicate']]
            assert row['sampling_seed']==old['sampling_seed'] and row['candidate']==old['candidate']
            assert row['baseline_certified']==old['certified'] and row['baseline_qp']==old['entered_qp']
            trace=list(csv.DictReader((directory/f"rep{row['replicate']}_slack.csv").open()));traces.extend(trace)
            if row['replicate'] in first:
                # A Farkas dual gives an independent lower bound for *any*
                # feasible position with shared slack. No removal is allowed.
                floor=first[row['replicate']]['raw_witness']['uniform_relaxation']
                for qp in trace:
                    if qp['converged']=='1' and float(qp['minimum_row_residual'])>=-1e-4:
                        assert float(qp['slack'])>=floor-1.0001e-4,(seed,row['replicate'],floor,qp)
                        bound_checks+=1
            if row['augmented_feasible']=='1':
                assert row['all_qps_converged']=='1' and row['deterministic_admissible']=='1'
                assert float(row['final_violation'])<=float(row['slack'])+1e-4
            if row['strict_zero']=='1':
                assert row['augmented_feasible']=='1' and float(row['slack'])<=0 and float(row['final_violation'])==0
        solved=[r for r in rows if r['augmented_feasible']=='1']
        empty=[r for r in rows if r['replicate'] in first]
        summary=dict(context_seed=seed,batches=100,qp_calls=len(traces),
                     converged_qp_calls=sum(r['converged']=='1' for r in traces),
                     all_qps_converged=sum(r['all_qps_converged']=='1' for r in rows),
                     augmented_feasible=len(solved),
                     raw_slack_nonpositive=sum(r['all_qps_converged']=='1' and float(r['slack'])<=0 for r in rows),
                     zero_slack_candidate_check=sum(r['strict_zero']=='1' for r in rows),
                     slack_le_1e4_and_augmented_feasible=sum(r['slack_le_1e4']=='1' for r in rows),
                     median_final_active=statistics.median(int(r['n_active']) for r in solved),
                     median_cumulative_active=statistics.median(int(r['union_active']) for r in solved),
                     max_cumulative_active=max(int(r['union_active']) for r in rows),
                     previous_empty_batches=len(empty),
                     previous_empty_all_qps_converged=sum(r['all_qps_converged']=='1' for r in empty),
                     previous_empty_augmented_feasible=sum(r['augmented_feasible']=='1' for r in empty),
                     previous_empty_near_zero=sum(r['slack_le_1e4']=='1' for r in empty),
                     previous_empty_slack_min=min(float(r['slack']) for r in empty),
                     previous_empty_slack_max=max(float(r['slack']) for r in empty))
        summaries.append(summary)
    result=dict(contexts=summaries,farkas_lower_bound_checks=bound_checks)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
