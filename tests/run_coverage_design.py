#!/usr/bin/env python3
"""Select validation cells by outcome-free window coverage, then run fresh paired seeds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from causal_link_analysis import read,write,undercoverage


def candidates(weights,protocol):
    rows=[];old={tuple(pair) for pair in protocol['old_cells']}
    for count in protocol['candidate_danger_counts']:
        window=[r for r in weights if int(r['danger_count'])==count and r['mode']=='across']
        window.sort(key=lambda r:int(r['step']))
        if [int(r['step']) for r in window]!=protocol['critical_window']:raise ValueError('incomplete fixed-reference window')
        for size in protocol['candidate_budgets']:
            if (count,size) in old:continue
            cp=math.prod(1-undercoverage(float(r['p']),protocol['kappa'],size) for r in window)
            cq=math.prod(1-undercoverage(float(r['q']),protocol['kappa'],size) for r in window)
            rows.append(dict(danger_count=count,S=size,C_p=cp,C_q=cq,delta_C=cq-cp))
    return rows


def select(rows,targets):
    from scipy.optimize import linear_sum_assignment
    costs=[[abs(float(r['delta_C'])-target) for r in rows] for target in targets]
    ii,jj=linear_sum_assignment(costs)
    return [dict(rows[j],target_delta_C=targets[i],target_error=costs[i][j],
                 cell=f"gain_{targets[i]:.2f}_p{rows[j]['danger_count']}_s{rows[j]['S']}") for i,j in zip(ii,jj)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/coverage-replication');p.add_argument('--select-only',action='store_true');args=p.parse_args();root=args.root.resolve()
    protocol=json.loads((root/'design_protocol_before_outcomes.json').read_text())
    rows=candidates(read(root/'design_window_weights.csv'),protocol);chosen=select(rows,protocol['targets'])
    write(root/'design_candidates.csv',rows);write(root/'design_selected.csv',chosen)
    record=dict(protocol_sha256=sha(root/'design_protocol_before_outcomes.json'),weights_sha256=sha(root/'design_window_weights.csv'),
        candidate_sha256=sha(root/'design_candidates.csv'),probe_binary_sha256=sha(ROOT/'build-base/coverage_window_probe'),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'tools/coverage_window_probe.cpp']},selected=chosen)
    path=root/'selection_before_validation_outcomes.json'
    if path.exists() and json.loads(path.read_text())!=record:raise RuntimeError('frozen selection/provenance changed')
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(chosen,indent=2),flush=True)
    if args.select_only:return
    if len(read(root/'replication/report/stress_outcomes.csv'))!=200:raise RuntimeError('Complete independent replication first')
    for c in chosen:
        settings=dict(families=[dict(name='late_switch',danger_counts=[c['danger_count']],history_lag=5)],scenario_budgets=[c['S']],switch_steps=[10],seeds=list(range(protocol['validation_seeds'][0],protocol['validation_seeds'][1]+1)),steps=120,arms=['sh_mpcc','sh_mpcc_dro'])
        settings_path=root/(c['cell']+'_settings.json');settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        with (root/(c['cell']+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'tests/run_risk_stress.py'),'--settings',str(settings_path),'--output',str(root/c['cell']),'--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('completed',c['cell'],flush=True)

if __name__=='__main__':main()
