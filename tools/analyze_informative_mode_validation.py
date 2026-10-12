#!/usr/bin/env python3
"""Independent LP checks, full-pool reference bands, and informative-bank plots."""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dro-informative-mpl')
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oracle_mode_validation import upper,write
from analyze_mode_validation import lp_risk

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args();root=Path(args.root);bank=json.loads((root/'frozen_bank.json').read_text());contexts={s['context']:json.loads(Path(s['parameters']).read_text()) for s in bank['specifications']}
    modes=list(csv.DictReader((root/'modes.csv').open()));points=list(csv.DictReader((root/'points.csv').open()));allocations=list(csv.DictReader((root/'allocations.csv').open()));alpha=.01/(2*len(modes));bands=[];checks=0;nonuniform=0
    for row in points:
        c=contexts[row['context']];lo=[];hi=[];point=[]
        for obs in c['obstacles']:
            L=[];U=[];V=[]
            for index in range(len(obs['modes'])):
                pair=next(r for r in modes if r['context']==row['context'] and int(r['obstacle'])==obs['id'] and int(r['mode_index'])==index);pack=np.load(root/f'{row["context"]}_obs{obs["id"]}_mode{index}.npz');h=np.unpackbits(pack['hits'])[:int(pack['N'])];k=int(h.sum());n=len(h);assert k==int(pair['K']);checks+=1
                L.append(0. if k==0 else float(beta.ppf(alpha,k,n-k+1)));U.append(upper(k,n,alpha));V.append(k/n)
            lo.append(lp_risk(obs,L));hi.append(lp_risk(obs,U));point.append(lp_risk(obs,V));checks+=3
        assert abs(min(1.,sum(point))-float(row['R_DRO_union']))<1e-8
        bands.append(dict(context=row['context'],alpha_per_mode_tail=alpha,union_functional_lower=min(1.,sum(lo)),union_functional_upper=min(1.,sum(hi)),product_model_lower=1-float(np.prod(1-np.array(lo))),product_model_upper=1-float(np.prod(1-np.array(hi)))))
    for row in allocations:
        c=contexts[row['context']];alpha_grid=float(row['alpha'])
        for name in ('uniform','nominal','voi','oracle'):
            if not int(row[name+'_pass']):continue
            counts=json.loads(row[name+'_counts']);assert sum(counts)==int(row[name+'_N']);offset=0;risks=[]
            for obs in c['obstacles']:
                ns=counts[offset:offset+len(obs['modes'])]
                if name=='oracle' and len(set(ns))>1:nonuniform+=1
                values=[]
                for index,n in enumerate(ns):
                    pack=np.load(root/f'{row["context"]}_obs{obs["id"]}_mode{index}.npz');h=np.unpackbits(pack['hits'])[:n];values.append(upper(int(h.sum()),n,alpha_grid));offset+=1
                risks.append(lp_risk(obs,values));checks+=1
            assert abs(sum(risks)-float(row[name+'_risk']))<1e-8 and sum(risks)<=.05
    write(root/'reference_bands.csv',bands)
    paired=[r for r in allocations if r['G_alloc']!=''];totals={name:sum(int(r[name+'_N']) for r in paired) for name in ('uniform','nominal','voi','oracle')};summary=dict(checks=checks,oracle_nonuniform_obstacle_groups=nonuniform,bank_reference_failure_budget=.01,reference_upper_below_threshold=sum(r['union_functional_upper']<=.05 for r in bands),reference_lower_above_threshold=sum(r['union_functional_lower']>.05 for r in bands),product_reference_lower_above_threshold=sum(r['product_model_lower']>.05 for r in bands),paired_totals=totals,pooled_oracle_saving=1-totals['oracle']/totals['uniform'],median_VOI_saving=float(np.median([1-int(r['voi_N'])/int(r['uniform_N']) for r in paired])),median_nominal_saving=float(np.median([1-int(r['nominal_N'])/int(r['uniform_N']) for r in paired])),risk_strata={})
    for label,a,b in [('below_0.01',0,.01),('0.01_to_0.03',.01,.03),('0.03_to_0.05',.03,.05),('above_0.05',.05,1.)]:
        ps=[r for r in points if a<=float(r['R_DRO_union'])<b];rs=[r for r in paired if a<=float(r['R_DRO_point'])<b]
        summary['risk_strata'][label]=dict(contexts=len(ps),paired=len(rs),median_G_alloc=float(np.median([float(r['G_alloc']) for r in rs])) if rs else None)
    summary['median_oracle_saving_vs_VOI']=float(np.median([1-int(r['oracle_N'])/int(r['voi_N']) for r in paired]))
    summary['pooled_oracle_saving_vs_VOI']=1-totals['oracle']/totals['voi']
    (root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4));idx=np.arange(len(points));axes[0].plot(idx,[float(r['R_DRO_union']) for r in points],'.',label='point estimate');axes[0].vlines(idx,[r['union_functional_lower'] for r in bands],[r['union_functional_upper'] for r in bands],color='grey');axes[0].axhline(.05,color='black',linestyle='--');axes[0].set_ylabel('Robust marginal-union risk');axes[0].set_xlabel('Frozen candidate');axes[0].legend()
    for name in ('uniform','nominal','voi','oracle'):axes[1].plot([float(r['R_DRO_point']) for r in paired],[int(r[name+'_N']) for r in paired],'.',label=name)
    axes[1].set_yscale('log');axes[1].set_xlabel('Reference robust risk');axes[1].set_ylabel('Validation trajectories');axes[1].legend()
    axes[2].scatter([float(r['R_DRO_point']) for r in paired],[100*float(r['G_alloc']) for r in paired],label='oracle');axes[2].scatter([float(r['R_DRO_point']) for r in paired],[100*(1-int(r['voi_N'])/int(r['uniform_N'])) for r in paired],marker='x',label='VOI');axes[2].axhline(20,color='grey',linestyle='--');axes[2].set_xlabel('Reference robust risk');axes[2].set_ylabel('Savings vs uniform (%)');axes[2].legend();fig.tight_layout();fig.savefig(root/'informative_validation.png',dpi=170);fig.savefig(root/'informative_validation.svg');plt.close(fig)
    print('PASS:',checks,'saved-pool/independent LP checks;',nonuniform,'oracle obstacle groups use unequal mode counts');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
