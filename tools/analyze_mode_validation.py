#!/usr/bin/env python3
"""Independent LP/replay checks and plots for the frozen mode validation screen."""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dro-mode-validation-mpl')
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oracle_mode_validation import upper,GRID

def lp_risk(obs,values):
    p=np.array([m['p'] for m in obs['modes']],float);p/=p.sum();M=len(p)
    lp=linprog(-np.tile(values,M),A_ub=np.array(obs['D']).reshape(1,-1),b_ub=[obs['rho']],A_eq=np.kron(np.eye(M),np.ones((1,M))),b_eq=p,bounds=(0,None),method='highs')
    if not lp.success:raise RuntimeError(lp.message)
    return -float(lp.fun)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args();root=Path(args.root)
    records=json.loads(Path('results/coreset-kill-test/contexts.json').read_text());contexts={r['context']:json.loads(Path(r['parameters']).read_text()) for r in records}
    modes=list(csv.DictReader((root/'modes.csv').open()));point=list(csv.DictReader((root/'oracle_contexts.csv').open()));allocations=list(csv.DictReader((root/'allocations.csv').open()));s=json.loads((root/'summary.json').read_text());checks=0
    for row in point:
        c=contexts[row['context']];risks=[]
        for obs in c['obstacles']:
            pairs=[r for r in modes if r['context']==row['context'] and int(r['obstacle'])==obs['id']];values=[]
            for pair in pairs:
                pack=np.load(root/(row['context']+f'_obs{obs["id"]}_mode{pair["mode_index"]}.npz'));hits=np.unpackbits(pack['hits'])[:int(pack['N'])]
                assert int(hits.sum())==int(pair['K']);values.append(float(hits.mean()));checks+=1
            risks.append(lp_risk(obs,values))
        assert abs(min(1.,sum(risks))-float(row['R_DRO_union']))<1e-8
    nonuniform_within_obstacle=0
    for row in allocations:
        c=contexts[row['context']];alpha=float(row['alpha_per_mode_count'])
        for name in ('uniform','oracle','heuristic'):
            if not int(row[name+'_pass']):continue
            ns=json.loads(row[name+'_counts']);assert sum(ns)==int(row[name+'_N']);offset=0;risks=[]
            for obs in c['obstacles']:
                values=[];group_ns=ns[offset:offset+len(obs['modes'])]
                if name=='oracle' and len(set(group_ns))>1:nonuniform_within_obstacle+=1
                for index,mode in enumerate(obs['modes']):
                    pack=np.load(root/(row['context']+f'_obs{obs["id"]}_mode{index}.npz'));n=ns[offset];hits=np.unpackbits(pack['hits'])[:n];offset+=1;values.append(upper(int(hits.sum()),n,alpha))
                risks.append(lp_risk(obs,values));checks+=1
            assert abs(sum(risks)-float(row[name+'_risk']))<1e-8 and sum(risks)<=.05
    extra=dict(independent_checks=checks,nonuniform_oracle_obstacle_groups=nonuniform_within_obstacle,variants={})
    for variant in ('fixed_pointwise','simultaneous_grid'):
        rows=[r for r in allocations if r['variant']==variant];total_u=sum(int(r['uniform_N']) for r in rows);total_o=sum(int(r['oracle_N']) for r in rows)
        extra['variants'][variant]=dict(total_uniform=total_u,total_oracle=total_o,pooled_saving=1-total_o/total_u,mean_context_saving=float(np.mean([float(r['G_alloc']) for r in rows])))
    (root/'analysis.json').write_text(json.dumps(extra,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4));indices=np.arange(len(point));axes[0].bar(indices,[float(r['R_DRO_union']) for r in point]);axes[0].set_ylabel('Point-estimate robust risk');axes[0].set_xlabel('Frozen context index');axes[0].set_yscale('symlog',linthresh=1e-6);axes[0].axhline(.05,color='black',linestyle='--');axes[0].set_ylim(0,.1)
    rows=[r for r in allocations if r['variant']=='simultaneous_grid']
    for name in ('uniform','oracle','heuristic'):axes[1].plot(indices,[int(r[name+'_N']) for r in rows],'.-',label=name)
    axes[1].set_ylabel('Validation trajectories (simultaneous grid)');axes[1].set_xlabel('Frozen context index');axes[1].legend();fig.tight_layout();fig.savefig(root/'mode_validation.png',dpi=170);fig.savefig(root/'mode_validation.svg');plt.close(fig)
    print('PASS:',checks,'independent saved-pool/HiGHS checks; oracle groups with unequal within-obstacle mode counts:',nonuniform_within_obstacle)
if __name__=='__main__':main()
