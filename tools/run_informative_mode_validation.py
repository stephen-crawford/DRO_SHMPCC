#!/usr/bin/env python3
"""Pre-specified frozen-context fine-grid allocation and retrospective VOI screen."""
import argparse,copy,csv,hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import ncx2
from particle_coreset import factors
from oracle_mode_validation import OTBridge,collision_pool,upper,pareto,optimum,write

GRID=(25,50,75,100,150,200,300,400,500,650,800,1000,1250,1500,2000,2500,3000,4000,5000,6500,8000,10000,15000,20000,30000,50000,75000,100000)

def freeze_bank(root,design_bridge):
    source=Path('results/oracle-transfer/contexts/context_23/step_0.json');template=json.loads(source.read_text());obs=template['obstacles'][0]
    plan='results/coreset-kill-test/replays/context_00/step_0/h_10/plans.csv'
    # Common initial-position translation leaves all mode-pair costs unchanged:
    # position dynamics have identity position block, and velocity does not use position.
    for mode in obs['modes']:
        A=np.array(mode['A']);assert np.array_equal(A[:2,:2],np.eye(2)) and np.all(A[2:,:2]==0)
    # Deterministic Gaussian marginal geometry supplies a design proxy only.
    # It is neither validation data nor an exact horizon-union probability.
    rows=list(csv.DictReader(Path(plan).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in rows[1:]])
    reference=copy.deepcopy(obs);reference['state'][:2]=[1.4,0.]
    gaussian=[factors(m,reference['state'],template['H']) for m in reference['modes']]
    def proxy(obstacles,clearance,sign):
        total=0.
        for j in range(obstacles):
            values=[]
            for mu,L in gaussian:
                means=mu.reshape(template['H'],2)+[.3*j,sign*(clearance+.5*j)];stage=[]
                for k in range(template['H']):
                    B=L[2*k:2*k+2];cov=B@B.T;variance=np.trace(cov)/2
                    assert np.allclose(cov,np.eye(2)*variance,rtol=1e-10,atol=1e-12)
                    stage.append(float(ncx2.cdf(template['radius']**2/variance,2,np.sum((means[k]-ego[k])**2)/variance)))
                values.append(math.sqrt(max(stage)*min(1.,sum(stage))))
            risk,_=design_bridge.risk(obs,values);total+=risk
        return total
    specifications=[]
    for obstacles in (1,2,3):
        for target in (.005,.015,.03,.05,.08,.14):
            for sign in (-1,1):
                lo=.6;hi=8.
                assert proxy(obstacles,lo,sign)>target and proxy(obstacles,hi,sign)<target
                for _ in range(30):
                    mid=(lo+hi)/2
                    if proxy(obstacles,mid,sign)>target:lo=mid
                    else:hi=mid
                clearance=hi
                c=copy.deepcopy(template);c['ego']=[0,0,0,1];c['obstacles']=[]
                for j in range(obstacles):
                    o=copy.deepcopy(obs);o['id']=j;o['state'][0]=1.4+.3*j;o['state'][1]=sign*(clearance+.5*j);c['obstacles'].append(o)
                name=f'c{len(specifications):02d}';path=root/'contexts'/f'{name}.json';path.write_text(json.dumps(c,indent=2)+'\n')
                specifications.append(dict(context=name,obstacles=obstacles,clearance=clearance,sign=sign,design_proxy_target=target,design_proxy=proxy(obstacles,clearance,sign),parameters=str(path),plan=plan))
    bank=dict(source=str(source),source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),plan=plan,plan_hash=hashlib.sha256(Path(plan).read_bytes()).hexdigest(),grid=GRID,beta_val=.01,alpha_rule='.01/(number_of_modes * number_of_grid_points)',pool_per_mode=100000,seed_domain=0x52495332,selection='all 36 combinations; no validation-based filtering or tuning',specifications=specifications)
    (root/'frozen_bank.json').write_text(json.dumps(bank,indent=2)+'\n')
    return bank

def study(context,pools,bridge,grid=GRID,beta_val=.01):
    M=sum(len(o['modes']) for o in context['obstacles']);alpha=beta_val/(M*len(grid));bounds=[];prefix=[]
    for obs,hits in zip(context['obstacles'],pools):
        counts=np.array([[int(h[:n].sum()) for n in grid] for h in hits]);prefix.append(counts);bounds.append(np.array([[upper(k,n,alpha) for k,n in zip(ks,grid)] for ks in counts]))
    def evaluate(indices):
        offset=0;total=0.;qs=[]
        for obs,b in zip(context['obstacles'],bounds):
            size=len(obs['modes']);local=indices[offset:offset+size];risk,q=bridge.risk(obs,b[np.arange(size),local]);total+=risk;qs.extend(q);offset+=size
        return total,np.array(qs)
    # The componentwise minimum gives a lower bound on every grid allocation's
    # U-based robust risk, even when realized bounds are nonmonotone in N.
    best_possible=sum(bridge.risk(o,b.min(axis=1))[0] for o,b in zip(context['obstacles'],bounds))
    uniform=None
    for index,n in enumerate(grid):
        risk,_=evaluate([index]*M)
        if risk<=.05:uniform=(M*n,risk,(n,)*M);break
    masses=np.concatenate([np.array([m['p'] for m in o['modes']]) for o in context['obstacles']]);weights=np.sqrt(masses);weights/=weights.mean();nominal=None
    for scale in grid:
        indices=np.minimum(np.searchsorted(grid,np.maximum(grid[0],scale*weights)),len(grid)-1).tolist();risk,_=evaluate(indices);counts=tuple(grid[i] for i in indices)
        if risk<=.05:
            choice=(sum(counts),risk,counts)
            if nominal is None or choice[:2]<nominal[:2]:nominal=choice
    fronts=[];oracle=None;voi=None;trace=[]
    if best_possible<=.05:
        for obs,b in zip(context['obstacles'],bounds):
            options=[]
            # Mode-wise dominance is safe: risk is monotone in each U coordinate.
            available=[]
            for values in b:
                best=math.inf;indices=[]
                for i,value in enumerate(values):
                    if value<best:indices.append(i);best=value
                available.append(indices)
            for indices in itertools.product(*available):
                ns=tuple(grid[i] for i in indices);risk,_=bridge.risk(obs,b[np.arange(len(b)),indices]);options.append((sum(ns),risk,ns))
            fronts.append(pareto(options))
        oracle=optimum(fronts)
        # Retrospective batched replay. One-sample expected change ranks modes;
        # actual updates advance only to predeclared grid points.
        indices=[0]*M;flat_bounds=np.concatenate(bounds);flat_prefix=np.concatenate(prefix)
        for step in range(M*(len(grid)-1)+1):
            risk,q=evaluate(indices);ns=tuple(grid[i] for i in indices);trace.append(dict(step=step,N=sum(ns),risk=risk,counts=ns,q=q.tolist()))
            if risk<=.05:voi=(sum(ns),risk,ns);break
            scores=np.full(M,-np.inf)
            for m,index in enumerate(indices):
                if index==len(grid)-1:continue
                n=grid[index];k=int(flat_prefix[m,index]);predictive=(k+.5)/(n+1)
                expected=(1-predictive)*upper(k,n+1,alpha)+predictive*upper(k+1,n+1,alpha)
                scores[m]=q[m]*max(0.,flat_bounds[m,index]-expected)
            if not np.isfinite(scores).any():break
            indices[int(np.argmax(scores))]+=1
    return dict(alpha=alpha,best_componentwise_grid_risk=best_possible,uniform=uniform,nominal=nominal,oracle=oracle,voi=voi,front_sizes=[len(f) for f in fronts],bounds=[b.tolist() for b in bounds],prefix_counts=[p.tolist() for p in prefix],voi_trace=trace)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False);(root/'contexts').mkdir();bridge=OTBridge(args.runner);bank=freeze_bank(root,bridge);started=time.perf_counter();modes=[];points=[];allocations=[];details=[]
    plan=list(csv.DictReader(Path(bank['plan']).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in plan[1:]])
    for spec in bank['specifications']:
        c=json.loads(Path(spec['parameters']).read_text());pools=[];risks=[];nominal=[];units=[]
        for obs in c['obstacles']:
            hits=[];values=[]
            for m,mode in enumerate(obs['modes']):
                seed=[bank['seed_domain'],int(spec['context'][1:]),obs['id'],m];h=collision_pool(c,obs,mode,ego,bank['pool_per_mode'],np.random.SeedSequence(seed));hits.append(h);values.append(float(h.mean()));np.savez_compressed(root/f'{spec["context"]}_obs{obs["id"]}_mode{m}.npz',hits=np.packbits(h),N=len(h))
                modes.append(dict(context=spec['context'],obstacle=obs['id'],mode=mode['name'],mode_index=m,p=mode['p'],N=len(h),K=int(h.sum()),v=float(h.mean()),seed=json.dumps(seed)))
            risk,q=bridge.risk(obs,values);p=np.array([m['p'] for m in obs['modes']],float);p/=p.sum();risks.append(risk);nominal.append(float(p@values));pools.append(hits)
            units.append(dict(obstacle=obs['id'],v=values,q=q.tolist(),risk=risk,nominal=nominal[-1],max_risk_mode=int(np.argmax(values)),max_nominal_modes=np.flatnonzero(p==p.max()).tolist()))
        point=dict(context=spec['context'],obstacles=spec['obstacles'],clearance=spec['clearance'],sign=spec['sign'],R_nominal_union=min(1.,sum(nominal)),R_DRO_union=min(1.,sum(risks)),R_DRO_product=1-float(np.prod(1-np.array(risks))),point_pass=int(sum(risks)<=.05),heterogeneous_obstacles=sum(max(u['v'])-min(u['v'])>=.001 for u in units),rare_dangerous_obstacles=sum(u['max_risk_mode'] not in u['max_nominal_modes'] for u in units));points.append(point)
        result=study(c,pools,bridge);row=dict(context=spec['context'],obstacles=spec['obstacles'],R_DRO_point=point['R_DRO_union'],alpha=result['alpha'],best_componentwise_grid_risk=result['best_componentwise_grid_risk'])
        for name in ('uniform','nominal','voi','oracle'):
            option=result[name];row.update({name+'_N':option[0] if option else '',name+'_risk':option[1] if option else '',name+'_counts':json.dumps(option[2]) if option else '',name+'_pass':int(option is not None)})
        row['G_alloc']=1-result['oracle'][0]/result['uniform'][0] if result['oracle'] and result['uniform'] else '';allocations.append(row);details.append(dict(context=spec['context'],units=units,**result))
        write(root/'modes.csv',modes);write(root/'points.csv',points);write(root/'allocations.csv',allocations);(root/'details.json').write_text(json.dumps(details,indent=2)+'\n')
        print(spec['context'],'point',point['R_DRO_union'],'uniform',row['uniform_N'],'oracle',row['oracle_N'],'VOI',row['voi_N'],flush=True)
    bridge.close();paired=[r for r in allocations if r['G_alloc']!=''];summary=dict(contexts=len(points),mode_pairs=len(modes),total_trajectories=sum(r['N'] for r in modes),point_pass=sum(r['point_pass'] for r in points),heterogeneous_contexts=sum(r['heterogeneous_obstacles']>0 for r in points),rare_dangerous_contexts=sum(r['rare_dangerous_obstacles']>0 for r in points),paired_contexts=len(paired),median_G_alloc=float(np.median([r['G_alloc'] for r in paired])) if paired else None,contexts_gain_ge_20=sum(r['G_alloc']>=.2 for r in paired),contexts_gain_ge_30=sum(r['G_alloc']>=.3 for r in paired),OT_calls=bridge.calls,elapsed_seconds=time.perf_counter()-started)
    for name in ('uniform','nominal','voi','oracle'):summary[name+'_pass']=sum(r[name+'_pass'] for r in allocations)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');files=[__file__,'tools/oracle_mode_validation.py',args.runner,bank['source'],bank['plan'],str(root/'frozen_bank.json')]+[s['parameters'] for s in bank['specifications']]
    (root/'manifest.json').write_text(json.dumps(dict(command=sys.argv,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}),indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
