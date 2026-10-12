#!/usr/bin/env python3
"""Summarize paired full-controller replays without treating flags as proofs."""
import argparse,csv,json,re
from pathlib import Path


def analyze(root):
    summaries=[]
    for seed in (77,78):
        directory=root/f'seed_{seed}'
        rows=list(csv.DictReader((directory/'replays.csv').open()))
        chosen={r['replicate']:r for r in csv.DictReader((directory/'selections.csv').open())}
        previous={r['replicate']:r for r in csv.DictReader((Path('results/facet-conflict-analysis')/f'seed_{seed}'/'replications.csv').open())}
        assert len(chosen)==19 and len(rows)==38
        for rep,selection in chosen.items():
            pair=[r for r in rows if r['replicate']==rep]
            assert len(pair)==2 and {r['variant'] for r in pair}=={'production','forced'}
            baseline=next(r for r in pair if r['variant']=='production')
            forced=next(r for r in pair if r['variant']=='forced')
            for r in pair:
                assert r['sampling_seed']==selection['sampling_seed'] and r['candidate']==selection['candidate']
            for key in ('sampling_seed','candidate','entered_qp','certified','support','support_evaluated','retained_clearance'):
                assert baseline[key]==previous[rep][key],(seed,rep,key)
            assert baseline['entered_qp']=='0' and baseline['certified']=='0'
            assert forced['requested_deletion']==selection['removed_id']==forced['actual_deletion']
            if forced['certified']=='1':
                assert forced['success']=='1' and forced['fallback']=='0'
                assert forced['support_evaluated']=='1' and 1<=int(forced['support'])<=10
                log=(directory/f"{forced['candidate']}_rep{rep}_forced.log").read_text()
                union=re.search(r'^\[SUPPORT UNION IDS\](.*)$',log,re.MULTILINE)
                assert union and int(selection['removed_id']) in list(map(int,union[1].split()))
        checks=json.load((directory/'pair_checks.json').open())
        assert checks['initial_rows_identical'] and checks['source_rng_unchanged']
        selected=[r for r in rows if r['variant']=='forced']
        metrics={key:sum(r[key]=='1' for r in selected) for key in ('entered_qp','any_qp_converged','last_qp_converged','certified','physical_strict','physical_numerical')}
        certified=[r for r in selected if r['certified']=='1']
        metrics.update(context_seed=seed,replayed_batches=len(selected),baseline_certified=0,
                       min_certified_retained_clearance=min(float(r['retained_clearance']) for r in certified),
                       max_certified_support=max(int(r['support']) for r in certified),
                       counterfactual_certified_total=(66 if seed==77 else 72)+len(certified),
                       rejected_replicates=[int(r['replicate']) for r in selected if r['certified']=='0'])
        summaries.append(metrics)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
