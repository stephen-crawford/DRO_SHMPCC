#!/usr/bin/env python3
"""Anytime-CS replay on paired, fixed-candidate saved conditional streams."""
import argparse,csv,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from bernoulli_confidence_sequence import ConfidenceSequence
from oracle_mode_validation import OTBridge,write

POLICIES=('uniform','nominal','upper_voi','boundary_voi','width_voi')

def replay(context,streams,bridge,policy,batch=1,warm=10):
    flat=[h for group in streams for h in group];M=len(flat);alpha=.01/M;cs=[ConfidenceSequence(alpha) for _ in flat];edges=np.cumsum([0]+[len(o['modes']) for o in context['obstacles']]);groups=[range(a,b) for a,b in zip(edges[:-1],edges[1:])];mode_group=[j for j,g in enumerate(groups) for _ in g]
    masses=np.concatenate([np.array([m['p'] for m in o['modes']],float) for o in context['obstacles']]);masses/=masses.sum();cost=dict(cs_seconds=0.,ot_seconds=0.,policy_seconds=0.,replay_read_seconds=0.);OTstart=bridge.calls;started=time.perf_counter();raw_steps=0;turn=0;risksL=np.zeros(len(groups));risksU=np.ones(len(groups));qL=np.zeros(M);qU=np.zeros(M);dirty=set(range(len(groups)));projection=[None]*M;status='unresolved';reason='stream_exhaustion'
    def consume(m,count):
        begin=time.perf_counter();ys=flat[m][cs[m].n:min(cs[m].n+count,len(flat[m]))];cost['replay_read_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter()
        for y in ys:cs[m].observe(int(y))
        cost['cs_seconds']+=time.perf_counter()-begin;dirty.add(mode_group[m]);projection[m]=None
    for m in range(M):consume(m,warm)
    while True:
        if any(c.lower>c.upper for c in cs):reason='empty_confidence_intersection';break
        begin=time.perf_counter()
        for j in dirty:
            g=list(groups[j]);risksL[j],q=bridge.risk(context['obstacles'][j],[cs[m].lower for m in g]);qL[g]=q
            risksU[j],q=bridge.risk(context['obstacles'][j],[cs[m].upper for m in g]);qU[g]=q
        dirty.clear();cost['ot_seconds']+=time.perf_counter()-begin;RL=min(1.,float(risksL.sum()));RU=min(1.,float(risksU.sum()))
        if RU<=.05:status='certify';reason='upper_boundary';break
        if RL>.05:status='reject';reason='lower_boundary';break
        available=[m for m in range(M) if cs[m].n<len(flat[m])]
        if not available:break
        begin=time.perf_counter()
        if policy=='uniform':
            while turn%M not in available:turn+=1
            selected=turn%M;turn+=1
        elif policy=='nominal':selected=min(available,key=lambda m:(cs[m].n/masses[m],m))
        else:
            dU=np.zeros(M);dL=np.zeros(M)
            for m in available:
                c=cs[m]
                if projection[m] is None:
                    l0,u0=c.projected(0);l1,u1=c.projected(1);p=(c.k+.5)/(c.n+1)
                    projection[m]=(max(0.,c.upper-((1-p)*u0+p*u1)),max(0.,(1-p)*l0+p*l1-c.lower))
                dU[m],dL[m]=projection[m]
            if policy=='upper_voi':scores=qU*dU
            elif policy=='boundary_voi':scores=qU*dU if RU-.05<.05-RL else qL*dL
            elif policy=='width_voi':scores=qU*dU+qL*dL
            else:raise ValueError('unknown policy')
            selected=max(available,key=lambda m:(scores[m],-cs[m].n,-m))
        cost['policy_seconds']+=time.perf_counter()-begin;consume(selected,batch);raw_steps+=1
    return dict(policy=policy,batch=batch,status=status,reason=reason,N=sum(c.n for c in cs),counts=json.dumps([c.n for c in cs]),hits=json.dumps([c.k for c in cs]),R_L=RL,R_U=RU,OT_calls=bridge.calls-OTstart,decisions=raw_steps,seconds=time.perf_counter()-started,**cost)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',required=True);ap.add_argument('--runner',required=True);ap.add_argument('--output',required=True);ap.add_argument('--batches',type=int,nargs='+',default=[1]);args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False);source=Path(args.bank);bank=json.loads((source/'frozen_bank.json').read_text());bridge=OTBridge(args.runner);rows=[];timings=[]
    for spec in bank['specifications']:
        c=json.loads(Path(spec['parameters']).read_text());streams=[]
        for obs in c['obstacles']:
            group=[]
            for m in range(len(obs['modes'])):
                packed=np.load(source/f'{spec["context"]}_obs{obs["id"]}_mode{m}.npz');group.append(np.unpackbits(packed['hits'])[:int(packed['N'])])
            streams.append(group)
        for batch in args.batches:
            for policy in POLICIES:
                result=replay(c,streams,bridge,policy,batch);result=dict(context=spec['context'],obstacles=len(c['obstacles']),**result);rows.append(result);write(root/'decisions.csv',rows)
                print(spec['context'],batch,policy,result['status'],result['N'],round(result['seconds'],3),flush=True)
    bridge.close();files=[__file__,'tools/bernoulli_confidence_sequence.py','tools/oracle_mode_validation.py',args.runner,str(source/'frozen_bank.json')];(root/'manifest.json').write_text(json.dumps(dict(command=sys.argv,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},warm_start=10,alpha='.01/M_total',a0=.5,b0=.5,source_bank=str(source)),indent=2)+'\n')
if __name__=='__main__':main()
