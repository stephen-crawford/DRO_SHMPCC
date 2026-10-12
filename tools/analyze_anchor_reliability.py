#!/usr/bin/env python3
"""Summarize fixed-context candidate replications without pooling contexts."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    summaries=[]
    for directory in sorted(args.root.glob('seed_*')):
        rows=list(csv.DictReader((directory/'replications.csv').open()))
        per_rep=collections.defaultdict(list)
        for r in rows:per_rep[r['replicate']].append(r)
        streams=set()
        for rep,block in per_rep.items():
            if len(block)!=4 or {r['candidate'] for r in block}!={'nominal','brake','left','right'}:
                raise RuntimeError('incomplete candidate block')
            seeds={r['sampling_seed'] for r in block}
            if len(seeds)!=1 or next(iter(seeds)) in streams:
                raise RuntimeError('unpaired or repeated replication stream')
            streams.update(seeds)
        for candidate in ['nominal','brake','left','right']:
            block=[r for r in rows if r['candidate']==candidate]
            certified=[r for r in block if r['certified']=='1']
            evaluated=[int(r['support']) for r in block if r['support_evaluated']=='1']
            stages=dict(collections.Counter(r['failure_stage'] for r in block))
            if any(r['entered_qp']!='1' or r['support_evaluated']!='1' or int(r['support'])>10 for r in certified):
                raise RuntimeError('acceptance accounting inconsistency')
            summaries.append(dict(context=directory.name,candidate=candidate,batches=len(block),
                qp_entered=sum(r['entered_qp']=='1' for r in block),certified=len(certified),
                support_evaluated=len(evaluated),median_evaluated_support=statistics.median(evaluated) if evaluated else None,
                median_certified_support=statistics.median(int(r['support']) for r in certified) if certified else None,
                median_solve_ms=statistics.median(float(r['solve_ms']) for r in block),
                min_certified_retained_clearance=min(float(r['retained_clearance']) for r in certified) if certified else None,
                failure_stages=stages))
    (args.root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (args.root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    for r in summaries:print(r)


if __name__=='__main__':main()
