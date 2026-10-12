#!/usr/bin/env python3
"""Diagnose emitted empty polygons using independent two-dimensional LPs."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def analyze(root):
    log=(root/'run.log').read_text()
    evidence=[]
    for combo in range(4):
        section=log.split(f'[FOUR WAY BEGIN] combination={combo}',1)[1].split('[FOUR WAY END]',1)[0]
        rows=list(csv.DictReader((root/f'rows_{combo}.csv').open()))
        removed=set()
        for line in section.splitlines():
            if line.startswith('[SQP SCENARIO REMOVAL]'):
                removed.add(int(re.search(r'id=(\d+)',line)[1]))
            if not line.startswith('[FREE POLY EMPTY]'):
                continue
            fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
            k,disc,offender=map(int,(fields['k'],fields['disc'],fields['scenario']))
            group=[r for r in rows if int(r['k'])==k and int(r['disc'])==disc and int(r['id']) not in removed]
            stop=next(i for i,r in enumerate(group) if int(r['id'])==offender)
            prefix=group[:stop+1]
            a=np.array([[float(r['a_x']),float(r['a_y'])] for r in prefix])
            b=np.array([float(r['b']) for r in prefix])
            feasibility=linprog([0,0],A_ub=-a,b_ub=-b,bounds=[(None,None)]*2,method='highs')
            # Uniform relaxation t: a_i p >= b_i-t. Dual weights provide a
            # small inconsistent subset when the unbounded halfspace system is empty.
            relaxed=linprog([0,0,1],A_ub=np.column_stack((-a,-np.ones(len(a)))),b_ub=-b,
                            bounds=[(None,None),(None,None),(0,None)],method='highs')
            item=dict(combination=combo,k=k,disc=disc,offending_id=offender,
                      removed_ids=sorted(removed),prefix_rows=len(prefix),
                      raw_rows_in_stage=len(group),normal_norm_min=min(float(r['normal_norm']) for r in group),
                      vertices_before=int(fields['vertices_before']),vertices_after=0,
                      raw_halfspace_LP_status=int(feasibility.status),
                      unbounded_intersection_infeasible=feasibility.status==2)
            if relaxed.success and relaxed.fun>1e-8:
                weights=-relaxed.ineqlin.marginals
                indices=np.flatnonzero(weights>1e-8)
                item.update(minimum_uniform_relaxation=float(relaxed.fun),
                            weighted_normal=(weights@a).tolist(),weighted_offset=float(weights@b),
                            witness=[dict(id=int(prefix[i]['id']),weight=float(weights[i]),
                                          a=a[i].tolist(),b=float(b[i]),
                                          anchor_slack=float(prefix[i]['anchor_slack']),
                                          relative=[float(prefix[i]['relative_x']),float(prefix[i]['relative_y'])]) for i in indices])
            evidence.append(item)
    (root/'geometric_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    for seed in sorted(args.root.glob('seed_*')):
        result=analyze(seed)
        print(seed.name,[(r['combination'],r['k'],r['offending_id'],r['raw_halfspace_LP_status'],r.get('minimum_uniform_relaxation')) for r in result])
