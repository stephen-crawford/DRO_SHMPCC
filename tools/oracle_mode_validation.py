#!/usr/bin/env python3
"""Offline fixed-plan mode-risk and finite-grid validation-allocation screen."""
import argparse,csv,hashlib,itertools,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import beta

GRID=(10,20,50,100,200,500,1000,2000,5000)

def upper(k,n,alpha):
    if n<1 or not 0<=k<=n or not 0<alpha<1:raise ValueError('invalid binomial bound input')
    return 1. if k==n else float(beta.isf(alpha,k+1,n-k))

class OTBridge:
    """Use the existing production primal OT through its offline executable."""
    def __init__(self,runner):
        self.proc=subprocess.Popen([runner],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.calls=0
    def risk(self,obs,values):
        p=np.array([m['p'] for m in obs['modes']],float);p/=p.sum();v=np.asarray(values,float)
        line=['dro',len(p),obs['rho'],.05,0,*p,*p,*v,*np.asarray(obs['D']).ravel()]
        self.proc.stdin.write(' '.join(map(str,line))+'\n');self.proc.stdin.flush()
        response=self.proc.stdout.readline()
        if not response:raise RuntimeError('primal OT executable terminated')
        q=np.fromstring(response,sep=' ')
        if len(q)!=len(p) or not np.isfinite(q).all() or np.min(q)<-1e-10 or abs(q.sum()-1)>1e-8:raise RuntimeError('invalid OT destination weights')
        self.calls+=1
        return float(q@v),q
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait()!=0:raise RuntimeError('primal OT exit failure')

def collision_pool(context,obs,mode,ego,draws,seed):
    rng=np.random.default_rng(seed);states=np.tile(obs['state'],(draws,1));A=np.array(mode['A']);G=np.array(mode['G']);b=np.array(mode['b']);hits=np.zeros(draws,dtype=bool)
    if context['length']!=0:raise ValueError('this bank uses one zero-length ego disc')
    for k in range(context['H']):
        states=states@A.T+b+rng.standard_normal((draws,G.shape[1]))@G.T
        hits|=np.sum((states[:,:2]-ego[k])**2,axis=1)<=context['radius']**2
    return hits

def pareto(options):
    # (cost, risk, count vector); exact dominance for the stored finite values.
    result=[];best=math.inf
    for option in sorted(options,key=lambda r:(r[0],r[1],r[2])):
        if option[1]<best:result.append(option);best=option[1]
    return result

def combine(fronts):
    combined=[(0,0.,())]
    for front in fronts:
        combined=pareto((a[0]+b[0],a[1]+b[1],a[2]+b[2]) for a in combined for b in front)
    return combined

def optimum(fronts,epsilon=.05):
    feasible=[r for r in combine(fronts) if r[1]<=epsilon]
    return min(feasible,key=lambda r:(r[0],r[1],r[2])) if feasible else None

def write(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def allocation_study(context,pools,bridge,grid,beta_val,variant):
    total_modes=sum(len(o['modes']) for o in context['obstacles'])
    alpha=beta_val/total_modes/(len(grid) if variant=='simultaneous_grid' else 1)
    costs=[];fronts=[];tables=[];allbounds=[]
    for obs,hits in zip(context['obstacles'],pools):
        bounds=np.array([[upper(int(h[:n].sum()),n,alpha) for n in grid] for h in hits]);allbounds.append(bounds)
        options=[];table={}
        for indices in itertools.product(range(len(grid)),repeat=len(hits)):
            counts=tuple(grid[i] for i in indices);values=bounds[np.arange(len(hits)),indices]
            risk,_=bridge.risk(obs,values);option=(sum(counts),risk,counts);options.append(option);table[counts]=risk
        fronts.append(pareto(options));tables.append(table)
    best=optimum(fronts);uniform=None
    for n in grid:
        risk=sum(t[(n,)*len(o['modes'])] for o,t in zip(context['obstacles'],tables))
        if risk<=.05:uniform=(total_modes*n,risk,(n,)*total_modes);break
    # Pre-sampling nominal-mass heuristic; no current validation hits are features.
    masses=np.concatenate([np.array([m['p'] for m in o['modes']]) for o in context['obstacles']]);weights=np.sqrt(masses);weights/=weights.mean()
    heuristic=None
    for scale in grid:
        target=np.maximum(grid[0],scale*weights);counts=tuple(grid[min(np.searchsorted(grid,x),len(grid)-1)] for x in target)
        offset=0;risk=0.
        for o,t in zip(context['obstacles'],tables):
            size=len(o['modes']);risk+=t[counts[offset:offset+size]];offset+=size
        if risk<=.05:
            candidate=(sum(counts),risk,counts)
            if heuristic is None or candidate[:2]<heuristic[:2]:heuristic=candidate
    return dict(alpha=alpha,uniform=uniform,oracle=best,heuristic=heuristic,front_sizes=[len(f) for f in fronts],bounds=[b.tolist() for b in allbounds])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--contexts',default='results/coreset-kill-test/contexts.json');ap.add_argument('--output',required=True);ap.add_argument('--pool',type=int,default=100000);ap.add_argument('--beta-val',type=float,default=.01);args=ap.parse_args()
    if args.pool<max(GRID):ap.error('pool must cover the largest validation grid count')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);records=json.loads(Path(args.contexts).read_text());bridge=OTBridge(args.runner);mode_rows=[];context_rows=[];data=[];started=time.perf_counter()
    for record in records:
        c=json.loads(Path(record['parameters']).read_text());plan=list(csv.DictReader(Path(record['plan']).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in plan[1:]])
        if len(ego)!=c['H']:raise ValueError('plan horizon mismatch')
        pools=[];nominal=[];robust=[];units=[]
        for obs in c['obstacles']:
            hits=[];values=[]
            for index,mode in enumerate(obs['modes']):
                seed=[0x4d56414c,int(record['context'].split('_')[0]),obs['id'],index]
                h=collision_pool(c,obs,mode,ego,args.pool,np.random.SeedSequence(seed));hits.append(h);values.append(float(h.mean()))
                mode_rows.append(dict(context=record['context'],family=record['family'],obstacle=obs['id'],mode=mode['name'],mode_index=index,p=mode['p'],rho_mode=obs['rho'],N=args.pool,K=int(h.sum()),v_point=float(h.mean()),seed=json.dumps(seed)))
                np.savez_compressed(root/(record['context']+f'_obs{obs["id"]}_mode{index}.npz'),hits=np.packbits(h),N=args.pool)
            p=np.array([m['p'] for m in obs['modes']],float);p/=p.sum();risk,q=bridge.risk(obs,values);nominal.append(float(p@values));robust.append(risk);pools.append(hits)
            units.append(dict(obstacle=obs['id'],v_point=values,q_worst=q.tolist(),R_nominal=nominal[-1],R_DRO=risk))
        nominal_product=1-float(np.prod(1-np.array(nominal)));robust_product=1-float(np.prod(1-np.array(robust)));nominal_union=min(1.,sum(nominal));robust_union=min(1.,sum(robust))
        context_rows.append(dict(context=record['context'],family=record['family'],obstacles=len(c['obstacles']),modes=sum(len(o['modes']) for o in c['obstacles']),R_nominal_product=nominal_product,R_DRO_product=robust_product,R_nominal_union=nominal_union,R_DRO_union=robust_union,nominal_pass=int(nominal_union<=.05),DRO_pass=int(robust_union<=.05),reference_good=1))
        data.append((record,c,pools,units));print(record['context'],'point nominal',nominal_union,'point DRO',robust_union,flush=True)
    write(root/'modes.csv',mode_rows);write(root/'oracle_contexts.csv',context_rows)
    pass_count=sum(r['DRO_pass'] for r in context_rows);majority=pass_count>=math.ceil(.75*len(records))
    allocation_rows=[];details=[]
    # Predeclared engineering trigger: >=75% on the conservative union diagnostic.
    if majority:
        for (record,c,pools,units),row in zip(data,context_rows):
            if not row['DRO_pass']:continue
            for variant in ('fixed_pointwise','simultaneous_grid'):
                study=allocation_study(c,pools,bridge,GRID,args.beta_val,variant)
                result=dict(context=record['context'],family=record['family'],variant=variant,alpha_per_mode_count=study['alpha'])
                for name in ('uniform','oracle','heuristic'):
                    option=study[name];result.update({name+'_N':option[0] if option else '',name+'_risk':option[1] if option else '',name+'_counts':json.dumps(option[2]) if option else '',name+'_pass':int(option is not None)})
                result['G_alloc']=1-study['oracle'][0]/study['uniform'][0] if study['uniform'] and study['oracle'] else ''
                allocation_rows.append(result);details.append(dict(context=record['context'],variant=variant,obstacles=units,**study));print(record['context'],variant,'uniform',result['uniform_N'],'oracle',result['oracle_N'],'heuristic',result['heuristic_N'],flush=True)
    bridge.close();write(root/'allocations.csv',allocation_rows);(root/'details.json').write_text(json.dumps(details,indent=2)+'\n')
    summary=dict(contexts=len(records),mode_pairs=len(mode_rows),pool_per_pair=args.pool,total_trajectories=len(mode_rows)*args.pool,point_DRO_pass=pass_count,point_nominal_pass=sum(r['nominal_pass'] for r in context_rows),allocation_trigger=majority,trigger_fraction=.75,beta_val=args.beta_val,grid=GRID,OT_calls=bridge.calls,elapsed_seconds=time.perf_counter()-started,families={})
    for family in range(4):
        rows=[r for r in context_rows if r['family']==family];summary['families'][family]=dict(contexts=len(rows),point_DRO_pass=sum(r['DRO_pass'] for r in rows),max_point_DRO=max(r['R_DRO_union'] for r in rows))
    for variant in ('fixed_pointwise','simultaneous_grid'):
        rows=[r for r in allocation_rows if r['variant']==variant];paired=[r for r in rows if r['G_alloc']!='']
        summary[variant]=dict(contexts=len(rows),uniform_pass=sum(r['uniform_pass'] for r in rows),oracle_pass=sum(r['oracle_pass'] for r in rows),heuristic_pass=sum(r['heuristic_pass'] for r in rows),median_uniform_N=float(np.median([r['uniform_N'] for r in rows if r['uniform_pass']])) if rows else None,median_oracle_N=float(np.median([r['oracle_N'] for r in rows if r['oracle_pass']])) if rows else None,median_G_alloc=float(np.median([r['G_alloc'] for r in paired])) if paired else None)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files=[__file__,'tools/particle_coreset.py',args.runner,args.contexts]+[r['parameters'] for r in records]+[r['plan'] for r in records]
    (root/'manifest.json').write_text(json.dumps(dict(command=sys.argv,numpy=np.__version__,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
