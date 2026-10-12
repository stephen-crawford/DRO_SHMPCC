#!/usr/bin/env python3
"""Analyze candidate acceptance, paired curves and context-only oracle cost."""
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUDGETS=[3,4,5,6,7,8,10]
COUNTS={3:550,4:666,5:781,6:895,7:1009,8:1123,10:1351}
def read(path):return list(csv.DictReader(path.open()))
def lower(k,n):return float(beta.ppf(.05,k,n-k+1)) if k else 0.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args();root=Path(args.root)
 manifest=json.loads((root/'manifest.json').read_text());R=manifest['repeats'];contexts=[];prefix_checks=0;nonmonotone=0;direction_checks=0
 for job in manifest['jobs']:
  folder=root/f"context_{job['context']:02}"
  for feature in read(folder/'contexts.csv'):
   step=int(feature['step']);name=f"{job['context']:02}_{step}";data={};C=dict(name=name,encounter=job['context'],source=job['source'],family=job['family'],features=feature)
   master={(int(r['replicate']),int(r['count'])):r['fingerprint'] for r in read(folder/f'step_{step}'/'h_10'/'prefixes.csv')}
   for h in BUDGETS:
    directory=folder/f'step_{step}'/f'h_{h}';rows=read(directory/'screen.csv');assert len(rows)==R
    assert {int(r['replicate']) for r in rows}==set(range(R))
    for row in rows:
     assert int(row['S'])==COUNTS[h]
     accepted=int(row['candidate_accept']);expected=int(row['solver_pass']) and int(row['deterministic_pass']) and float(row['s_cert'])<=.0002 and int(row['reconstruction'])<=h
     assert accepted==bool(expected)
     if accepted: assert float(row['physical_clearance'])>=0
    for r in read(directory/'prefixes.csv'):
     assert r['fingerprint']==master[(int(r['replicate']),int(r['count']))],(name,h,r)
     prefix_checks+=1
    data[h]=rows
   directions={}
   for h in BUDGETS:
    for row in read(folder/f'step_{step}'/f'h_{h}'/'extrema.csv'):
     key=(row['k'],row['disc'],row['obstacle'],row['mode'])
     direction=(float(row['a_x']),float(row['a_y']))
     assert abs(math.hypot(*direction)-1)<=1e-12
     if key in directions:assert directions[key]==direction,(name,key)
     directions[key]=direction;direction_checks+=1
   C['data']=data;contexts.append(C)
   for rep in range(R):
    outcomes=[int(data[h][rep]['candidate_accept']) for h in BUDGETS]
    nonmonotone+=int(any(a>b for a,b in zip(outcomes,outcomes[1:])))
 curves=[]
 for c in contexts:
  for h in BUDGETS:
   rows=c['data'][h];k=sum(int(r['candidate_accept']) for r in rows);op=sum(int(r['operational_accept']) for r in rows)
   curves.append(dict(context=c['name'],source=c['source'],family=c['family'],h=h,S=COUNTS[h],accepted=k,operational=op,repetitions=R,p=k/R,p_op=op/R,lower95=lower(k,R),mean_reconstruction=np.mean([int(r['reconstruction']) for r in rows]),median_solve_ms=np.median([float(r['solve_ms']) for r in rows])))
 with (root/'curves.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 def oracle(outcome='candidate_accept',conservative=False):
  rates={h:np.mean([np.mean([int(r[outcome]) for r in c['data'][h]]) for c in contexts]) for h in BUDGETS}
  fixed=next((h for h in BUDGETS if (np.mean([lower(sum(int(r[outcome]) for r in c['data'][h]),R) for c in contexts]) if conservative else rates[h])>=.95),None)
  choices={c['name']:next((h for h in BUDGETS if (lower(sum(int(r[outcome]) for r in c['data'][h]),R) if conservative else np.mean([int(r[outcome]) for r in c['data'][h]]))>=.95),None) for c in contexts}
  covered=sum(h is not None for h in choices.values());cost=float(np.mean([COUNTS[h] for h in choices.values()])) if covered==len(contexts) else None
  return dict(mean_acceptance_by_h=rates,best_fixed_h=fixed,fixed_cost=COUNTS[fixed] if fixed else None,context_choices=choices,covered_contexts=covered,oracle_cost=cost,oracle_gap=1-cost/COUNTS[fixed] if cost is not None and fixed else None,choice_histogram=dict(Counter(str(h) for h in choices.values())))
 summary=dict(contexts=len(contexts),synthetic=sum(c['source']=='synthetic' for c in contexts),snapshots=sum(c['source']=='encounter' for c in contexts),solves=len(contexts)*R*len(BUDGETS),replications=R,prefix_checks=prefix_checks,direction_checks=direction_checks,paired_nonmonotone_context_replications=nonmonotone,point=oracle(),operational=oracle('operational_accept'),conservative=oracle(conservative=True),max_context_lower95=lower(R,R),engineering_target=.95,confidence_level=.95)
 # Split entire encounters, never adjacent snapshots or replications.
 feature_names=['ego_speed','obstacles','modes','observed_distance','mode_entropy','terminal_within_mode_covariance_trace','terminal_modal_dispersion','min_nominal_reference_distance','path_progress']
 X=[];y=[];groups=[]
 for c in contexts:
  for h in BUDGETS:
   for row in c['data'][h]:
    X.append([float(c['features'][f]) for f in feature_names]+[h]);y.append(int(row['candidate_accept']));groups.append(c['encounter'])
 X=np.array(X);y=np.array(y);groups=np.array(groups);pred=np.zeros(len(y));constant=np.zeros(len(y))
 # Small regression tree with Bernoulli leaf means; squared-error splitting.
 def tree(indices,depth):
  mean=float(np.mean(y[indices]));best=None;best_gain=0
  if depth==0:return mean
  for column in range(X.shape[1]):
   values=np.unique(X[indices,column]);thresholds=(values[:-1]+values[1:])/2
   if len(thresholds)>24:thresholds=thresholds[np.linspace(0,len(thresholds)-1,24).astype(int)]
   for threshold in thresholds:
    left=indices[X[indices,column]<=threshold];right=indices[X[indices,column]>threshold]
    if min(len(left),len(right))<100:continue
    gain=len(left)*(float(np.mean(y[left]))-mean)**2+len(right)*(float(np.mean(y[right]))-mean)**2
    if gain>best_gain:best_gain=gain;best=(column,float(threshold),left,right)
  if best is None:return mean
  column,threshold,left,right=best
  return (column,threshold,tree(left,depth-1),tree(right,depth-1))
 def evaluate(model,row):
  while isinstance(model,tuple):
   col,threshold,left,right=model;model=left if row[col]<=threshold else right
  return model
 group_values=np.unique(groups);rng=np.random.default_rng(20261009);rng.shuffle(group_values)
 for held in np.array_split(group_values,5):
  test=np.where(np.isin(groups,held))[0];train=np.where(~np.isin(groups,held))[0]
  model=tree(train,3);pred[test]=[evaluate(model,row) for row in X[test]]
  for h in BUDGETS:
   selected=test[X[test,-1]==h];constant[selected]=np.mean(y[train[X[train,-1]==h]])
 def balanced(prob):return float((np.mean((prob[y==1]>=.5))+np.mean((prob[y==0]<.5)))/2)
 summary['predictability']=dict(features=feature_names+['h'],folds=5,group='entire encounter/seed',tree_depth=3,min_leaf=100,tree_brier=float(np.mean((pred-y)**2)),constant_by_budget_brier=float(np.mean((constant-y)**2)),tree_balanced_accuracy=balanced(pred),constant_balanced_accuracy=balanced(constant),exploratory=True)
 failures={}
 for h in BUDGETS:
  rows=[r for c in contexts for r in c['data'][h]]
  failures[h]=dict(over_reconstruction_cap=sum(int(r['reconstruction'])>h for r in rows),solver_failed=sum(not int(r['solver_pass']) for r in rows),deterministic_failed=sum(not int(r['deterministic_pass']) for r in rows),slack_over_tau=sum(float(r['s_cert'])>.0002 or not math.isfinite(float(r['s_cert'])) for r in rows))
 summary['gate_failures_nonexclusive']=failures
 covered=[c for c in contexts if summary['point']['context_choices'][c['name']] is not None]
 subset_rates={h:float(np.mean([int(r['candidate_accept']) for c in covered for r in c['data'][h]])) for h in BUDGETS}
 subset_fixed=next((h for h in BUDGETS if subset_rates[h]>=.95),None)
 subset_cost=float(np.mean([COUNTS[summary['point']['context_choices'][c['name']]] for c in covered]))
 summary['posthoc_covered_subset']=dict(contexts=len(covered),not_primary_result=True,mean_acceptance_by_h=subset_rates,best_fixed_h=subset_fixed,oracle_cost=subset_cost,oracle_gap=1-subset_cost/COUNTS[subset_fixed] if subset_fixed else None)
 summary['family_acceptance']={str(f):{h:float(np.mean([int(r['candidate_accept']) for c in contexts if c['family']==f for r in c['data'][h]])) for h in BUDGETS} for f in range(4)}
 summary['feature_ranges']={f:[min(float(c['features'][f]) for c in contexts),max(float(c['features'][f]) for c in contexts)] for f in feature_names}
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
 for c in contexts:
  ax=axes.flat[min(c['family'],3)];ax.plot(BUDGETS,[np.mean([int(r['candidate_accept']) for r in c['data'][h]]) for h in BUDGETS],alpha=.5,marker='.',label=c['name'])
 for i,ax in enumerate(axes.flat):
  ax.axhline(.95,color='black',linestyle='--',linewidth=.8);ax.set_title(['Single mode (+ encounter snapshots)','Two modes, one obstacle','Three modes, two obstacles','Four modes, three obstacles'][i]);ax.set_ylim(-.02,1.03);ax.set_xticks(BUDGETS);ax.set_xlabel('Total reconstruction cap h');ax.set_ylabel('Candidate acceptance fraction')
 fig.tight_layout();fig.savefig(root/'budget_curves.png',dpi=170);fig.savefig(root/'budget_curves.svg');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
