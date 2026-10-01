#!/usr/bin/env python3
"""Production-path checks for frozen counts, mechanism logs, and scaling timers."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile


def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,default=Path('build-base/causal_link_probe'));a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='causal-link-runtime-') as tmp:
        root=Path(tmp)
        for phase,obstacles,reps,quota in [('conditional',2,120,1),('scaling',2,2,0),('scaling',4,2,0)]:
            out=root/f'{phase}_{obstacles}'
            with (root/f'{out.name}.log').open('w') as log:
                subprocess.run([str(a.probe.resolve()),str(out),'3','1.2','-.04','100','20',str(reps),str(quota),str(obstacles),phase],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            if phase=='conditional':
                outcomes=rows(out/'conditional.csv')
                assert outcomes and any(int(r['n_d'])>4 for r in outcomes), 'higher-count conditional strata missing'
                assert all(r['n_d']==r['verified_n_d'] and r['status']=='OK' for r in outcomes), 'sampler/controller mismatch'
                from collections import defaultdict
                grouped=defaultdict(list)
                for r in rows(out/'conditional_modes.csv'):grouped[r['law'],r['seed'],r['step']].append(r)
                for rr in grouped.values():
                    assert sum(int(r['n_m']) for r in rr)==20
                    assert abs(sum(float(r['p']) for r in rr)-1)<1e-12
                    assert abs(sum(float(r['q']) for r in rr)-1)<1e-12
                    assert all(r['true_mode']=='across' and float(r['r'])>=0 and float(r['rho'])>=0 for r in rr)
                plant=rows(out/'plant.csv')
                for before,after in zip(plant,plant[1:]):
                    assert abs(float(after['x'])-float(before['x'])-.12)<1e-12
                    assert abs(float(after['y'])-float(before['y'])+.04)<1e-12
            else:
                timings=rows(out/'scaling.csv')
                assert len(timings)==6
                for r in timings:
                    assert float(r['controller_trajectory_generation_ms'])>0
                    assert float(r['trajectory_generation_ms'])>0
                    assert float(r['fixed_reference_constraint_ms'])>=0
                    # The fixed-reference API includes the initial k=0 state plus 20 predicted states.
                    assert int(r['fixed_reference_raw_constraints'])==int(r['S'])*obstacles*21*3
                    assert int(r['retained_facets'])>=0 and float(r['solve_ms'])>0
        print('PASS: production sampling/count agreement, all count strata, complete mode logs, forced plant, and live timing at 2/4 obstacles')

if __name__=='__main__':main()
