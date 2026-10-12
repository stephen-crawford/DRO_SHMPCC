#!/usr/bin/env python3
"""Re-solve updated CP-envelope laws on archived decision inputs, not new trajectories."""
import argparse,csv,io,json,subprocess,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta
from scipy.optimize import linprog

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 proc=subprocess.Popen(['/tmp/paper_frozen_allocator'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 rows=[];modes_out=[];coverage=[]
 with zipfile.ZipFile(a.archive) as z:
  def read(name):return list(csv.DictReader(io.StringIO(z.read(name).decode())))
  for name in z.namelist():
   if '/raw/ablation/' not in name or not name.endswith('/repeat_0/transport_costs.csv'):continue
   prefix=name[:-len('transport_costs.csv')];case=prefix.split('/raw/ablation/')[1].split('/')[0];seed=int(prefix.split('/seed_')[1].split('/')[0])
   transport=read(name);mechanism=read(prefix+'mode_mechanism.csv');old=read(prefix+'distribution_transfer.csv')
   if not old:
    coverage.append(dict(case=case,seed=seed,status='no_logged_DRO_inputs',obstacle_decisions=0));continue
   groups=defaultdict(list);costs=defaultdict(list);original=defaultdict(list)
   for r in mechanism:
    if r['attempt']=='0':groups[r['step'],r['obstacle_id']].append(r)
   for r in transport:
    if r['attempt']=='0':costs[r['step'],r['obstacle_id']].append(r)
   for r in old:original[r['step'],r['obstacle_id']].append(r)
   count=0
   for key,ms in groups.items():
    if key not in costs or key not in original:continue
    ms=sorted(ms,key=lambda r:r['mode']);ids=[r['mode'] for r in ms];m=len(ms)
    p=np.array([float(r['nominal_probability']) for r in ms]);risk=np.array([float(r['risk_score']) for r in ms]);rho=float(ms[0]['rho']);g=int(costs[key][0]['radius_observation_count'])
    counts=np.rint(p*(g+.5*m)-.5).astype(int)
    assert min(counts)>=0 and sum(counts)==g and max(abs((counts+.5)/(g+.5*m)-p))<1e-10
    V=len({obs for step,obs in groups if step==key[0]});tail=.05/(2*V*m)
    lo=np.array([0 if n==0 else beta.ppf(tail,n,g-n+1) for n in counts]);hi=np.array([1 if n==g else beta.isf(tail,n+1,g-n) for n in counts]);u=np.minimum(hi,1-lo.sum()+lo)
    D=np.zeros((m,m))
    for c in costs[key]:D[ids.index(c['source_mode']),ids.index(c['target_mode'])]=float(c['cost'])
    source=np.zeros((m,m*m+1));destination=np.zeros_like(source)
    for j in range(m):source[j,j*m:(j+1)*m]=1;destination[j,j:m*m:m]=1
    ineq=-destination.copy();ineq[:,-1]=u;c=np.zeros(m*m+1);c[-1]=-1
    lp=linprog(c,A_ub=np.vstack([np.r_[D.ravel(),0],ineq]),b_ub=np.r_[rho,np.zeros(m)],A_eq=source,b_eq=p,bounds=(0,None),method='highs');assert lp.success
    qw=destination@lp.x
    payload=[m,rho,*p,*risk,*u,*D.ravel()];proc.stdin.write(' '.join(map(str,payload))+'\n');proc.stdin.flush();ans=np.fromstring(proc.stdout.readline(),sep=' ');assert ans[0]==1;qstar=ans[2:]
    old_by={r['mode']:r for r in original[key]};qold=np.array([float(old_by[mid]['q']) for mid in ids]);ub=np.array([float(old_by[mid]['envelope']) for mid in ids]);minimum=u.sum();danger=risk>=.01
    ident=f'{case}_s{seed}_t{key[0]}_o{key[1]}'
    for arm,q in [('nominal',p),('envelope_optimal',u/minimum),('wasserstein_min_zeta',qw),('full_proposed',qstar),('archived_q_evaluated_on_CP',qold)]:
     factor=max(u/q);assert factor>=minimum-1e-7
     rows.append(dict(instance=ident,case=case,seed=seed,step=key[0],obstacle=key[1],V=V,M=m,g=g,arm=arm,rho=rho,minimum_zeta=minimum,zeta=factor,normalized_zeta=factor/minimum,dangerous_mass_change=(q-p)[danger].sum(),risk_lift=(q-p)@risk,old_new_envelope_ratio=ub.sum()/minimum,q=json.dumps(q.tolist()),analysis='offline_updated_CP_on_archived_inputs'))
    for j in range(m):modes_out.append(dict(instance=ident,mode=ids[j],count=counts[j],g=g,p=p[j],risk=risk[j],L=lo[j],U=hi[j],u_C=u[j],u_B_logged=ub[j],q_archived=qold[j],q_full_updated=qstar[j],q_W=qw[j],q_env=u[j]/minimum))
    count+=1
   coverage.append(dict(case=case,seed=seed,status='offline_inputs_reconstructed',obstacle_decisions=count))
 proc.stdin.close();assert proc.wait()==0
 for name,data in [('frozen_law_ablation.csv',rows),('frozen_law_modes.csv',modes_out),('frozen_law_coverage.csv',coverage)]:
  with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0],lineterminator='\n');w.writeheader();w.writerows(data)
 print('Reconstructed obstacle decisions',sum(r['obstacle_decisions'] for r in coverage),'law evaluations',len(rows))
if __name__=='__main__':main()
