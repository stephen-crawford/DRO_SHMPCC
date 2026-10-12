#!/usr/bin/env python3
"""Freeze an independent multi-model bank, then draw counter-indexed streams."""
import argparse,copy,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import ncx2
from particle_coreset import factors
from oracle_mode_validation import OTBridge

SOURCES=(11,12,21,22,23,24,31,32)

def conditional_draw(context,obs,mode,ego,count,seed,start=0):
    H=context['H'];G=np.array(mode['G']);d=G.shape[1];stride=4*math.ceil(H*d/4)
    tick=time.perf_counter();rng=np.random.Generator(np.random.Philox(np.random.SeedSequence(seed)));rng.bit_generator.advance(start*(stride//4))
    u=rng.random((count,stride))[:,:H*d];noise=ndtri(np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.))).reshape(count,H,d)
    states=np.tile(obs['state'],(count,1));A=np.array(mode['A']);b=np.array(mode['b']);generation=time.perf_counter()-tick;collision=0.;hits=np.zeros(count,bool)
    for k in range(H):
        tick=time.perf_counter();states=states@A.T+b+noise[:,k]@G.T;generation+=time.perf_counter()-tick
        tick=time.perf_counter();hits|=np.sum((states[:,:2]-ego[k])**2,axis=1)<=context['radius']**2;collision+=time.perf_counter()-tick
    return hits,generation,collision

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False);(root/'contexts').mkdir();bridge=OTBridge(args.runner)
    originals=[json.loads(Path(f'results/oracle-transfer/contexts/context_{i:02d}/step_0.json').read_text()) for i in SOURCES];specs=[]
    for J in (1,2,3):
        for i in range(8):
            for high in (False,True):
                plan=f'results/coreset-kill-test/replays/context_{i%5:02d}/step_0/h_10/plans.csv';rows=list(csv.DictReader(Path(plan).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in rows[1:]])
                context=copy.deepcopy(originals[i]);context['obstacles']=[];gaussian=[];source_ids=[]
                for j in range(J):
                    index=(i+3*j)%8;o=copy.deepcopy(originals[index]['obstacles'][0]);o['id']=j
                    for m in o['modes']:
                        A=np.array(m['A']);assert np.array_equal(A[:2,:2],np.eye(2)) and np.all(A[2:,:2]==0)
                    o['state'][:2]=[0.,0.];mu,_=factors(o['modes'][0],o['state'],context['H']);offset=ego[context['H']//2]-mu.reshape(-1,2)[context['H']//2];o['state'][:2]=offset.tolist();context['obstacles'].append(o);gaussian.append([factors(m,o['state'],context['H']) for m in o['modes']]);source_ids.append(SOURCES[index])
                target=(.08,.10,.14)[i%3] if high else (.015,.03,.05)[i%3];sign=-1 if i%2 else 1
                def proxy(clearance):
                    total=0.
                    for j,(o,models) in enumerate(zip(context['obstacles'],gaussian)):
                        values=[]
                        for mu,L in models:
                            means=mu.reshape(-1,2)+[0,sign*(clearance+.25*j)];stage=[]
                            for k in range(context['H']):
                                cov=L[2*k:2*k+2]@L[2*k:2*k+2].T;var=np.trace(cov)/2;assert np.allclose(cov,np.eye(2)*var,rtol=1e-10,atol=1e-12)
                                stage.append(float(ncx2.cdf(context['radius']**2/var,2,np.sum((means[k]-ego[k])**2)/var)))
                            values.append(math.sqrt(max(stage)*min(1.,sum(stage))))
                        total+=bridge.risk(o,values)[0]
                    return total
                lo=0.;hi=10.;assert proxy(lo)>target and proxy(hi)<target
                for _ in range(30):
                    mid=(lo+hi)/2
                    if proxy(mid)>target:lo=mid
                    else:hi=mid
                for j,o in enumerate(context['obstacles']):o['state'][1]+=sign*(hi+.25*j)
                name=f'q{len(specs):02d}';path=root/'contexts'/f'{name}.json';path.write_text(json.dumps(context,indent=2)+'\n');specs.append(dict(context=name,parameters=str(path),plan=plan,obstacles=J,source_ids=source_ids,proxy_target=target,sign=sign))
    bank=dict(specifications=specs,seed_domain=0x434f4e46,pool_per_mode=100000,design='Gaussian marginal proxy; all 48 cases frozen before any confirmation stream',generator='counter-indexed Philox uniforms and normal inverse CDF',sources=SOURCES)
    (root/'frozen_bank.json').write_text(json.dumps(bank,indent=2)+'\n');bridge.close();profiles=[]
    for spec in specs:
        c=json.loads(Path(spec['parameters']).read_text());rows=list(csv.DictReader(Path(spec['plan']).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in rows[1:]])
        for o in c['obstacles']:
            for m,mode in enumerate(o['modes']):
                seed=[bank['seed_domain'],int(spec['context'][1:]),o['id'],m];h,g,t=conditional_draw(c,o,mode,ego,100000,seed);np.savez_compressed(root/f'{spec["context"]}_obs{o["id"]}_mode{m}.npz',hits=np.packbits(h),N=len(h));profiles.append(dict(context=spec['context'],obstacle=o['id'],mode=m,N=len(h),K=int(h.sum()),generation_seconds=g,collision_seconds=t))
        print(spec['context'],'pools frozen',flush=True)
    (root/'pool_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n');paths=[__file__,'tools/particle_coreset.py',args.runner]+[str(root/'frozen_bank.json')]+[s['parameters'] for s in specs]+list({s['plan'] for s in specs})+[f'results/oracle-transfer/contexts/context_{i:02d}/step_0.json' for i in SOURCES];(root/'manifest.json').write_text(json.dumps(dict(command=__import__('sys').argv,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
if __name__=='__main__':main()
