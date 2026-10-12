#!/usr/bin/env python3
"""Primary exact single-obstacle screen; secondary equal/optimized allocation."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from run_oracle_transfer import Bridge,candidates,floor,context_eta

def optimize_single(bridge,obs,b,epsilon=.05,starts=()):
 best=-1.;selected=None
 for q in candidates(obs,b)+list(starts):
  value=bridge.evaluate('eta',obs,q,b,risk=epsilon)
  if value>best:best=value;selected=q
  if best==1:break
 status='not_needed'
 if len(selected)>1 and best<1:
  objective=lambda q:-bridge.evaluate('eta',obs,floor(q),b,risk=epsilon)
  result=minimize(objective,selected,method='SLSQP',bounds=[(.01,1.)]*len(selected),constraints=[dict(type='eq',fun=lambda q:float(np.sum(q)-1))],options=dict(maxiter=40,ftol=1e-10,eps=1e-6))
  status=str(result.message);q=floor(result.x);value=-objective(q)
  if value>best:best=value;selected=q
 return selected,best,status

def inverse(bridge,obs,q,b,risk):
 if risk>0:return bridge.evaluate('eta',obs,q,b,risk=risk)
 if bridge.evaluate('psi',obs,q,b,eta=1)==0:return 1.
 lo=0.;hi=1.
 for _ in range(36):
  mid=(lo+hi)/2
  if bridge.evaluate('psi',obs,q,b,eta=mid)<=risk:lo=mid
  else:hi=mid
 return lo

def allocation(bridge,obstacles,qs,bs,equal):
 n=len(obstacles)
 if equal:
  budgets=[.05/n]*n;targets=[inverse(bridge,o,q,b,risk) for o,q,b,risk in zip(obstacles,qs,bs,budgets)]
  eta=min(targets)
 else:
  eta=context_eta(bridge,obstacles,qs,bs)
  budgets=[bridge.evaluate('psi',o,q,b,eta=eta) for o,q,b in zip(obstacles,qs,bs)]
  assert sum(budgets)<=.05
  budgets[-1]+=.05-sum(budgets)
  targets=[inverse(bridge,o,q,b,risk) for o,q,b,risk in zip(obstacles,qs,bs,budgets)]
  eta=min(eta,*targets)
 assert sum(bridge.evaluate('psi',o,q,b,eta=eta) for o,q,b in zip(obstacles,qs,bs))<=.05+1e-15
 return eta,budgets,targets

def optimize_joint(bridge,obstacles,bs,starts):
 sizes=[len(o['modes']) for o in obstacles];edges=np.cumsum([0]+sizes)
 def unpack(vector):return [floor(vector[a:b]) for a,b in zip(edges[:-1],edges[1:])]
 best=-1;selected=None
 for qs in starts:
  value=context_eta(bridge,obstacles,qs,bs)
  if value>best:best=value;selected=qs
 def objective(vector):return -context_eta(bridge,obstacles,unpack(vector),bs)
 constraints=[dict(type='eq',fun=lambda vector,a=a,b=b:float(np.sum(vector[a:b])-1)) for a,b in zip(edges[:-1],edges[1:])]
 result=minimize(objective,np.concatenate(selected),method='SLSQP',bounds=[(.01,1.)]*edges[-1],constraints=constraints,options=dict(maxiter=40,ftol=1e-9,eps=1e-6))
 proposed=unpack(result.x);value=context_eta(bridge,obstacles,proposed,bs)
 if value>best:selected=proposed
 return selected,str(result.message)

def write(path,rows):
 with path.open('w') as f:
  out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
 source=Path(args.source);entries=json.loads((source/'contexts/bank.json').read_text());details={d['context']:d for d in json.loads((source/'screen/details.json').read_text())};bridge=Bridge(args.runner)
 primary=[];four=[];secondary=[];records=[]
 for entry in entries:
  context=json.loads(Path(entry['file']).read_text());obstacles=context['obstacles'];units=details[entry['context']]['units']
  for variant in ['point','upper']:
   bs=[np.array(u[variant]['b']) for u in units];p=[np.array([m['p'] for m in o['modes']]) for o in obstacles];dro=[floor(np.array(u['q_dro_raw'])) for u in units]
   if len(obstacles)==1:
    best,_,status=optimize_single(bridge,obstacles[0],bs[0]);proposals={'nominal':p[0],'dro':dro[0],'oracle_search':best}
    result=dict(context=entry['context'],source=entry['source'],variant=variant,num_modes=len(p[0]),baseline_S=1351)
    for name,q in proposals.items():
     eta=bridge.evaluate('eta',obstacles[0],q,bs[0]);count=bridge.samples(eta);result.update({f'eta_{name}':eta,f'S_{name}':count,f'saving_{name}':1-count/1351})
    result['oracle_incremental_vs_nominal']=1-result['S_oracle_search']/result['S_nominal'];primary.append(result)
    records.append(dict(context=entry['context'],variant=variant,scope='primary_single',proposals={name:q.tolist() for name,q in proposals.items()},optimizer_status=status))
   else:
    equal_q=[]
    for obs,b in zip(obstacles,bs):q,_,_=optimize_single(bridge,obs,b,.05/len(obstacles));equal_q.append(q)
    uniform=[np.ones(len(q))/len(q) for q in p]
    optimized_q,status=optimize_joint(bridge,obstacles,bs,[p,dro,equal_q,uniform])
    proposals={'nominal':p,'dro':dro,'oracle_equal_search':equal_q,'oracle_allocated_search':optimized_q}
    for name,qs in proposals.items():
     for kind in ['equal','optimized']:
      eta,budgets,targets=allocation(bridge,obstacles,qs,bs,kind=='equal');count=bridge.samples(eta)
      secondary.append(dict(context=entry['context'],variant=variant,num_obstacles=len(obstacles),proposal=name,allocation=kind,eta_joint=eta,S=count,saving=1-count/1351,epsilon_j=json.dumps(budgets),eta_j=json.dumps(targets),q_j=json.dumps([q.tolist() for q in qs])))
    records.append(dict(context=entry['context'],variant=variant,scope='secondary_multi',optimizer_status=status))
    if len(p[0])==4:
     q,eta,status4=optimize_single(bridge,obstacles[0],bs[0]);nom_eta=bridge.evaluate('eta',obstacles[0],p[0],bs[0]);dro_eta=bridge.evaluate('eta',obstacles[0],dro[0],bs[0])
     four.append(dict(context=entry['context']+'_obs0',variant=variant,scope='projected_first_obstacle',eta_nominal=nom_eta,eta_dro=dro_eta,eta_oracle_search=eta,S_nominal=bridge.samples(nom_eta),S_dro=bridge.samples(dro_eta),S_oracle_search=bridge.samples(eta),saving_oracle_search=1-bridge.samples(eta)/1351,q_oracle=json.dumps(q.tolist()),optimizer_status=status4))
  print(entry['context'],'complete',flush=True)
 bridge.close();write(root/'primary_single.csv',primary);write(root/'supplemental_four_mode.csv',four);write(root/'secondary_multi.csv',secondary);(root/'optimization.json').write_text(json.dumps(records,indent=2))
 summary=dict(primary_contexts=30,supplemental_four_mode_marginals=10,secondary_contexts=20,reused_MC=True)
 for variant in ['point','upper']:
  summary[variant]={}
  for name in ['nominal','dro','oracle_search']:
   rows=[r for r in primary if r['variant']==variant];summary[variant][name]=dict(median_eta=float(np.median([r['eta_'+name] for r in rows])),median_S=float(np.median([r['S_'+name] for r in rows])),contexts_above_epsilon=sum(r['eta_'+name]>.050000001 for r in rows),mean_saving=float(np.mean([r['saving_'+name] for r in rows])))
  rows=[r for r in primary if r['variant']==variant];summary[variant]['incremental_vs_nominal']=dict(median=float(np.median([r['oracle_incremental_vs_nominal'] for r in rows])))
  rows=[r for r in four if r['variant']==variant];summary[variant]['four_mode']=dict(median_eta=float(np.median([r['eta_oracle_search'] for r in rows])),median_S=float(np.median([r['S_oracle_search'] for r in rows])),contexts_above_epsilon=sum(r['eta_oracle_search']>.050000001 for r in rows))
  summary[variant]['secondary']={}
  for name in ['nominal','dro','oracle_equal_search','oracle_allocated_search']:
   for allocation_name in ['equal','optimized']:
    rows=[r for r in secondary if r['variant']==variant and r['proposal']==name and r['allocation']==allocation_name]
    summary[variant]['secondary'][name+'_'+allocation_name]=dict(median_eta=float(np.median([r['eta_joint'] for r in rows])),median_S=float(np.median([r['S'] for r in rows])),mean_saving=float(np.mean([r['saving'] for r in rows])))
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 files=[__file__,'tools/run_oracle_transfer.py','tools/oracle_transfer.hpp',args.runner,str(source/'screen/details.json'),str(source/'contexts/bank.json')]+[e['file'] for e in entries]
 (root/'manifest.json').write_text(json.dumps(dict(command=sys.argv,sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
if __name__=='__main__':main()
