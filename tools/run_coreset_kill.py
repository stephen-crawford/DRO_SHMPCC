#!/usr/bin/env python3
"""Staged offline empirical-reference ambiguity/compression headroom test."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from particle_coreset import lift,sample,collision_distances,robust_probability,medoids
root=Path('results/coreset-kill-test');records=json.loads((root/'contexts.json').read_text());N=4096;Kvalues=[4,8,16,32,64,128];base=[];compression=[];lifting=[]
for record in records:
 context=json.loads(Path(record['parameters']).read_text());plan=list(csv.DictReader(Path(record['plan']).open()))
 ego=np.array([[float(r['x']),float(r['y'])] for r in plan[1:]])[:,None,:];assert len(ego)==context['H'] and context['length']==0
 rho,lifts=lift(context);lifting.append(dict(context=record['context'],rho_trajectory=rho,obstacles=lifts))
 seed=[0x434f5245,int(record['context'].split('_')[0]),0];particles=sample(context,N,np.random.SeedSequence(seed));X=particles.reshape(N,-1);d=collision_distances(particles,ego,context['radius']);weights=np.ones(N)/N
 empirical=robust_probability(d,weights,0);risk=robust_probability(d,weights,rho)
 screen=record['original_screen'];good=bool(int(screen['solver_pass']) and int(screen['deterministic_pass']) and float(screen['progress'])>=.5 and empirical<=.05)
 base.append(dict(context=record['context'],family=record['family'],obstacles=len(context['obstacles']),modes_per_obstacle=len(context['obstacles'][0]['modes']),dimension=X.shape[1],N=N,rho=rho,empirical_risk=empirical,ambiguity_risk=risk,ambiguity_pass=int(risk<=.05),reference_good=int(good),original_candidate_accept=screen['candidate_accept'],seed=json.dumps(seed)))
 np.savez_compressed(root/(record['context']+'_reference.npz'),particles=particles,ego=ego,distances=d)
 # Latest request: compression proceeds only for individual ambiguity survivors.
 if risk<=.05:
  for K in Kvalues:
   start=time.perf_counter();ids,w,delta=medoids(X,K);construction=time.perf_counter()-start;start=time.perf_counter();comp=robust_probability(d[ids],w,delta);combined=comp if rho==0 else (robust_probability(d[ids],w,rho+delta) if comp<=.05 else float("nan"));certificate=time.perf_counter()-start
   compression.append(dict(context=record['context'],family=record['family'],K=K,effective_K=len(ids),rho=rho,delta=delta,R=rho+delta,compression_risk=comp,compression_pass=int(comp<=.05),combined_risk=combined,combined_evaluated=int(comp<=.05),combined_pass=int(comp<=.05 and combined<=.05),medoid_seconds=construction,certificate_seconds=certificate))
   np.savez_compressed(root/(record['context']+f'_K{K}.npz'),ids=ids,weights=w)
 print(record['context'],'rho',rho,'empirical',empirical,'ambiguity',risk,flush=True)
def write(name,rows):
 if not rows:return
 with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('ambiguity.csv',base);write('compression.csv',compression);(root/'lifting.json').write_text(json.dumps(lifting,indent=2)+'\n')
summary=dict(contexts=len(base),N=N,reference_good=sum(r['reference_good'] for r in base),ambiguity_pass=sum(r['ambiguity_pass'] for r in base),good_ambiguity_pass=sum(r['ambiguity_pass'] and r['reference_good'] for r in base),compressed_contexts=len({r['context'] for r in compression}),families={})
for family in range(4):
 rows=[r for r in base if r['family']==family];summary['families'][family]=dict(contexts=len(rows),reference_good=sum(r['reference_good'] for r in rows),ambiguity_pass=sum(r['ambiguity_pass'] for r in rows))
summary['K']={K:dict(tested=sum(r['K']==K for r in compression),compression_pass=sum(r['K']==K and r['compression_pass'] for r in compression),combined_pass=sum(r['K']==K and r['combined_pass'] for r in compression),median_delta=float(np.median([r['delta'] for r in compression if r['K']==K]))) for K in Kvalues} if compression else {}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13,4))
axes[0].plot(range(len(base)),[r['empirical_risk'] for r in base],'.',label='empirical');axes[0].plot(range(len(base)),[r['ambiguity_risk'] for r in base],'.',label='rho-only');axes[0].set_xlabel('Context index');axes[0].set_ylabel('Collision probability bound');axes[0].legend();axes[0].axhline(.05,color='black',linestyle='--')
for name in sorted({r['context'] for r in compression}):
 rows=[r for r in compression if r['context']==name];axes[1].plot([r['K'] for r in rows],[r['delta'] for r in rows],'.-',label=name);axes[2].plot([r['K'] for r in rows],[r['combined_risk'] for r in rows],'.-',label=name)
axes[1].set_ylabel('Exact assignment W1 compression error (m)');axes[2].set_ylabel('Combined robust risk');axes[2].axhline(.05,color='black',linestyle='--')
for ax in axes[1:]:ax.set_xlabel('Requested K');ax.set_xscale('log',base=2);ax.legend()
fig.tight_layout();fig.savefig(root/'kill_test.png',dpi=170);fig.savefig(root/'kill_test.svg');plt.close(fig)
files=['tools/particle_coreset.py',__file__,'tests/test_particle_coreset.py','tests/recover_coreset_plans.py','tools/week2_budget_response.cpp','/tmp/dro-shmpcc-week1-build/week2_budget_response']+[r['plan'] for r in records]+[r['parameters'] for r in records]
(root/'manifest.json').write_text(json.dumps(dict(N=N,K=Kvalues,sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
