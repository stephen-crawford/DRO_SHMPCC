#!/usr/bin/env python3
"""Check frozen directions, paired batches and independent raw intersections."""
import argparse,csv,json,statistics
from pathlib import Path
from scipy.optimize import linprog


def analyze(root):
    summaries=[];lp_checks=0
    for seed in (77,78):
        directory=root/f'seed_{seed}'
        rows=list(csv.DictReader((directory/'replays.csv').open()))
        previous={r['replicate']:r for r in csv.DictReader((Path('results/facet-conflict-analysis')/f'seed_{seed}'/'replications.csv').open())}
        groups=list(csv.DictReader((directory/'groups.csv').open()))
        assert len(rows)==200
        reference={};raw_feasible={}
        for replicate in range(100):
            pair=[r for r in rows if int(r['replicate'])==replicate]
            assert len(pair)==2 and {r['variant'] for r in pair}=={'production','fixed'}
            baseline=next(r for r in pair if r['variant']=='production');fixed=next(r for r in pair if r['variant']=='fixed')
            old=previous[str(replicate)]
            for key in ('sampling_seed','candidate','entered_qp','certified','support_evaluated','support'):
                assert baseline[key]==old[key],(seed,replicate,key)
            assert baseline['sampling_seed']==fixed['sampling_seed']
            stage_groups=[r for r in groups if int(r['replicate'])==replicate]
            assert len(stage_groups)==20
            for row in stage_groups:
                key=tuple(row[k] for k in ('k','disc','obstacle','mode'))
                direction=tuple(row[k] for k in ('a_x','a_y','anchor_x','anchor_y'))
                if key not in reference:reference[key]=direction
                assert reference[key]==direction,(seed,replicate,key)
                assert abs(float(row['a_x'])**2+float(row['a_y'])**2-1)<1e-12
                assert int(row['sample_count'])==1351
            feasible=True
            for stage in sorted({(r['k'],r['disc']) for r in stage_groups}):
                g=[r for r in stage_groups if (r['k'],r['disc'])==stage]
                lp=linprog([0,0],A_ub=[[-float(r['a_x']),-float(r['a_y'])] for r in g],b_ub=[-float(r['max_b']) for r in g],bounds=[(None,None)]*2,method='highs')
                assert lp.status in (0,2),lp.message
                feasible=feasible and lp.success;lp_checks+=1
            raw_feasible[str(replicate)]=feasible
        checks=json.load((directory/'pair_checks.json').open())
        assert checks['sampled_trajectories_identical'] and checks['source_rng_unchanged']
        fixed=[r for r in rows if r['variant']=='fixed'];empty=[r for r in fixed if previous[r['replicate']]['failure_stage']=='facet_empty']
        certified=[r for r in fixed if r['certified']=='1'];evaluated=[r for r in fixed if r['support_evaluated']=='1']
        summary=dict(context_seed=seed,batches=len(fixed),raw_intersections_nonempty=sum(raw_feasible.values()),
            **{key:sum(r[key]=='1' for r in fixed) for key in ('entered_qp','any_qp_converged','last_qp_converged','hard_feasible','certified','support_evaluated','strict_physical','facet_empty')},
            median_logged_support=statistics.median(int(r['support']) for r in evaluated),
            max_logged_support=max(int(r['support']) for r in evaluated),
            min_certified_physical_clearance=min(float(r['physical_clearance']) for r in certified),
            original_empty_batches=len(empty),original_empty_qp_entered=sum(r['entered_qp']=='1' for r in empty),
            original_empty_certified=sum(r['certified']=='1' for r in empty),
            degeneracy_calls=sum('rule=previous' in line or 'rule=road' in line for r in fixed for line in (directory/f"rep{r['replicate']}_fixed.log").read_text().splitlines()),
            rejected_replicates=[int(r['replicate']) for r in fixed if r['certified']!='1'])
        summaries.append(summary)
    result=dict(contexts=summaries,raw_LP_checks=lp_checks)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
