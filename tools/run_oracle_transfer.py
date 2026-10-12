#!/usr/bin/env python3
"""Offline conditional danger bounds and experiment-only exact-vertex transfers."""
import argparse,csv,hashlib,json,math,subprocess,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Bridge:
 def __init__(self,path):self.proc=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 def request(self,line):
  self.proc.stdin.write(line+'\n');self.proc.stdin.flush();result=self.proc.stdout.readline()
  if not result:raise RuntimeError('transfer bridge terminated')
  return float(result)
 def evaluate(self,command,obs,q,b,risk=.05,eta=0,rho=None):
  args=[command,len(q),obs['rho'] if rho is None else rho,risk,eta,*[m['p'] for m in obs['modes']],*q,*b,*np.array(obs['D']).ravel()]
  return self.request(' '.join(map(str,args)))
 def samples(self,eta):return int(self.request(f'samples {eta} 10')) if eta>0 else None
 def close(self):self.proc.stdin.close();assert self.proc.wait()==0

def trig_range(lo,hi,sine=False):
 fn=math.sin if sine else math.cos;values=[fn(lo),fn(hi)];offset=math.pi/2 if sine else 0
 for k in range(math.ceil((lo-offset)/math.pi),math.floor((hi-offset)/math.pi)+1):values.append(fn(offset+k*math.pi))
 return min(values),max(values)

def reachable_boxes(context):
 # RK4 derivatives depend only on heading and linearly evolving speed. Interval
 # sums over the four stages enclose each admissible discrete displacement.
 x,y,theta,v=context['ego'];lo=np.array([x,y]);hi=lo.copy();boxes=[];dt=context['dt']
 for k in range(context['H']):
  dl=np.zeros(2);du=np.zeros(2)
  for fraction,weight in [(0,1/6),(.5,1/3),(.5,1/3),(1,1/6)]:
   t=(k+fraction)*dt;vl=max(0.,v+context['amin']*t);vu=min(context['vmax'],v+context['amax']*t)
   assert vl<=vu
   for axis in range(2):
    a,b=trig_range(theta-context['omega']*t,theta+context['omega']*t,sine=axis==1)
    values=[vl*a,vl*b,vu*a,vu*b];dl[axis]+=weight*min(values)*dt;du[axis]+=weight*max(values)*dt
  lo=lo+dl;hi=hi+du
  # An explicit outward engineering padding; formal libm interval error is not proved.
  boxes.append([*(lo-1e-10),*(hi+1e-10)])
 return np.array(boxes)

def danger(context,obs,mode,draws,seed):
 rng=np.random.default_rng(seed);A=np.array(mode['A']);G=np.array(mode['G']);bias=np.array(mode['b']);states=np.tile(obs['state'],(draws,1));hit=np.zeros(draws,dtype=bool)
 boxes=reachable_boxes(context);radius=context['radius']+context['length']/2
 for k in range(context['H']):
  states=states@A.T+bias+rng.standard_normal((draws,G.shape[1]))@G.T
  lo=boxes[k,:2];hi=boxes[k,2:];distance=np.maximum(np.maximum(lo-states[:,:2],states[:,:2]-hi),0.)
  hit|=np.sum(distance**2,axis=1)<=radius**2
 return int(hit.sum()),boxes

def floor(q,minimum=.01):
 q=np.maximum(q,0);q=q/q.sum();small=q.min()
 if small<minimum:
  mixing=(minimum-small)/(1/len(q)-small);q=(1-mixing)*q+mixing/len(q)
 assert q.min()>=minimum-1e-14
 return q

def candidates(obs,b):
 p=np.array([m['p'] for m in obs['modes']]);raw=np.array([m['q_dro'] for m in obs['modes']]);M=len(p)
 proposals=[p,floor(raw),np.ones(M)/M,floor(p*np.maximum(b,1e-12)),floor(p*np.sqrt(np.maximum(b,1e-12)))]
 # Deterministic simplex grid with denominator 5 (up to 56 points in four modes).
 def compositions(total,size):
  if size==1:yield [total]
  else:
   for n in range(total+1):
    for rest in compositions(total-n,size-1):yield [n]+rest
 for counts in compositions(5,M):proposals.append(.01+(1-M*.01)*np.array(counts)/5)
 unique={tuple(q):q for q in proposals};return list(unique.values())

def context_eta(bridge,obstacles,qs,bs):
 if len(obstacles)==1:return bridge.evaluate('eta',obstacles[0],qs[0],bs[0])
 def transfer(eta):return sum(bridge.evaluate('psi',o,q,b,eta=eta) for o,q,b in zip(obstacles,qs,bs))
 if transfer(1)<=.05:return 1.
 lo=0.;hi=1.
 for _ in range(36):
  mid=(lo+hi)/2
  if transfer(mid)<=.05:lo=mid
  else:hi=mid
 return lo

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--contexts',required=True);ap.add_argument('--output',required=True);ap.add_argument('--draws',type=int,default=10000);args=ap.parse_args()
 root=Path(args.output);root.mkdir(parents=True,exist_ok=True);bank=json.loads((Path(args.contexts)/'bank.json').read_text());data=[json.loads(Path(entry['file']).read_text()) for entry in bank]
 total_pairs=sum(len(o['modes']) for c in data for o in c['obstacles']);alpha=.01/total_pairs;bridge=Bridge(args.runner);rows=[];details=[];started=time.time()
 for entry,context in zip(bank,data):
  bs_point=[];bs_upper=[];nominal=[];qdro=[];units=[]
  for obs in context['obstacles']:
   point=[];upper=[];unit=dict(obstacle=obs['id'],p=[m['p'] for m in obs['modes']],rho=obs['rho'],D=obs['D'],q_dro_raw=[m['q_dro'] for m in obs['modes']],modes=[])
   for index,mode in enumerate(obs['modes']):
    seed=np.random.SeedSequence([0x44414e47,int(entry['context'].split('_')[0]),context['step'],obs['id'],index]);count,boxes=danger(context,obs,mode,args.draws,seed)
    point.append(count/args.draws);upper.append(float(beta.ppf(1-alpha,count+1,args.draws-count)) if count<args.draws else 1.)
    unit['modes'].append(dict(name=mode['name'],hits=count,draws=args.draws,b_point=point[-1],b_upper=upper[-1],seed_entropy=seed.entropy))
   unit['reachable_boxes']=boxes.tolist();units.append(unit);bs_point.append(np.array(point));bs_upper.append(np.array(upper));nominal.append(np.array(unit['p']));qdro.append(floor(np.array(unit['q_dro_raw'])))
  for variant,bs in [('point',bs_point),('upper',bs_upper)]:
   qbest=[];marginal=[]
   for obs,b in zip(context['obstacles'],bs):
    p=np.array([m['p'] for m in obs['modes']]);current=None;best=-1;target=.05/len(context['obstacles'])
    for q in candidates(obs,b):
     eta=bridge.evaluate('eta',obs,q,b,risk=target)
     if eta>best:best=eta;current=q
     if best==1:break
    optimizer_status='not_needed';optimizer_success=True
    if len(p)>1 and best<1:
     def objective(q):return -bridge.evaluate('eta',obs,floor(q),b,risk=target)
     solution=minimize(objective,current,method='SLSQP',bounds=[(.01,1.)]*len(p),constraints=[dict(type='eq',fun=lambda q:float(np.sum(q)-1))],options=dict(maxiter=40,ftol=1e-10,eps=1e-6))
     optimizer_status=str(solution.message);optimizer_success=bool(solution.success)
     proposed=floor(solution.x);value=bridge.evaluate('eta',obs,proposed,b,risk=target)
     if value>best:best=value;current=proposed
    qbest.append(current)
    marginal.append(dict(eta_oracle_allocated=best,q_oracle=current.tolist(),optimizer_success=optimizer_success,optimizer_status=optimizer_status))
   proposals={'nominal':nominal,'dro':qdro,'oracle_search':qbest}
   result=dict(context=entry['context'],source=entry['source'],family=entry['family'],variant=variant,num_obstacles=len(context['obstacles']),num_modes=sum(len(o['modes']) for o in context['obstacles']),scope='exact_single_obstacle' if len(context['obstacles'])==1 else 'conservative_union_bound',baseline_S=1351)
   for name,qs in proposals.items():
    eta=context_eta(bridge,context['obstacles'],qs,bs);S=bridge.samples(eta)
    result[f'eta_{name}']=eta;result[f'S_{name}']=S;result[f'saving_{name}']=1-S/1351 if S is not None else None
   result['incremental_saving_vs_nominal']=1-result['S_oracle_search']/result['S_nominal'] if result['S_nominal'] else None
   rows.append(result)
   for unit,b,q,margin in zip(units,bs,qbest,marginal):unit[variant]=dict(b=b.tolist(),**margin)
  details.append(dict(context=entry['context'],units=units));(root/'details.json').write_text(json.dumps(details,indent=2)+'\n')
  print(entry['context'],[(r['variant'],round(r['eta_oracle_search'],5),r['S_oracle_search']) for r in rows[-2:]],flush=True)
 bridge.close()
 with (root/'transfers.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=dict(contexts=len(bank),mode_context_pairs=total_pairs,draws_per_pair=args.draws,total_draws=args.draws*total_pairs,beta_cal=.01,alpha_per_pair=alpha,elapsed_seconds=time.time()-started)
 for variant in ['point','upper']:
  subset=[r for r in rows if r['variant']==variant];summary[variant]={}
  for proposal in ['nominal','dro','oracle_search']:
   etas=[r['eta_'+proposal] for r in subset];gains=[r['saving_'+proposal] for r in subset]
   summary[variant][proposal]=dict(median_eta=float(np.median(etas)),mean_eta=float(np.mean(etas)),contexts_eta_above_epsilon=sum(v>.050000001 for v in etas),contexts_eta_above_010=sum(v>.1 for v in etas),median_saving=float(np.median(gains)),mean_saving=float(np.mean(gains)),min_saving=min(gains),max_saving=max(gains))
  summary[variant]['incremental_vs_nominal']=dict(median=float(np.median([r['incremental_saving_vs_nominal'] for r in subset])),mean=float(np.mean([r['incremental_saving_vs_nominal'] for r in subset])))
 limits=[]
 for context in details:
  result=dict(context=context['context'])
  for variant in ['point','upper']:
   minimum=min(1.,sum(min(unit[variant]['b']) for unit in context['units']))
   result[variant+'_joint_min_bound']=minimum
   result[variant+'_any_proposal_eta_at_most_epsilon']=minimum>.05
  limits.append(result)
 (root/'proposal_independent_limits.json').write_text(json.dumps(limits,indent=2)+'\n')
 summary['proposal_independent_limits']={v:sum(row[v+'_any_proposal_eta_at_most_epsilon'] for row in limits) for v in ['point','upper']}
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 upper=[r for r in rows if r['variant']=='upper']
 for name in ['nominal','dro','oracle_search']:
  axes[0].plot(range(50),[r['eta_'+name] for r in upper],'.',label=name);axes[1].plot(range(50),[r['saving_'+name]*100 for r in upper],'.',label=name)
 axes[0].axhline(.05,color='black',linestyle='--');axes[0].set_ylabel('Transferred Q-risk target eta (upper danger bounds)');axes[1].set_ylabel('Scenario saving vs S=1351 (%)')
 for ax in axes:ax.set_xlabel('Context index');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(root/'transfer_curves.png',dpi=170);fig.savefig(root/'transfer_curves.svg');plt.close(fig)
 files=[args.runner,__file__,'tools/oracle_transfer.hpp','tools/oracle_transfer.cpp','tests/export_transfer_bank.py','tests/test_oracle_transfer.py','tools/week2_budget_response.cpp','results/oracle-transfer/protocol.md',str(Path(args.contexts)/'bank.json')]+[entry['file'] for entry in bank]
 (root/'manifest.json').write_text(json.dumps(dict(command=__import__('sys').argv,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
if __name__=='__main__':main()
