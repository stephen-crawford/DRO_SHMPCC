#!/usr/bin/env python3
"""Prospective generation/check timing on counter-paired confirmation streams."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from make_sequential_confirmation import conditional_draw
from run_sequential_validation import replay,POLICIES
from oracle_mode_validation import OTBridge,write

class ConditionalStream:
    def __init__(self,context,obs,mode,ego,seed,expected):
        self.context=context;self.obs=obs;self.mode=mode;self.ego=ego;self.seed=seed;self.expected=expected;self.generation=0.;self.collision=0.;self.checked=0
    def __len__(self):return len(self.expected)
    def __getitem__(self,selection):
        begin=selection.start;end=selection.stop
        hits,g,c=conditional_draw(self.context,self.obs,self.mode,self.ego,end-begin,self.seed,begin)
        if not np.array_equal(hits,self.expected[begin:end]):raise RuntimeError('counter-paired stream mismatch')
        self.generation+=g;self.collision+=c;self.checked+=len(hits);return hits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',required=True);ap.add_argument('--runner',required=True);ap.add_argument('--output',required=True);ap.add_argument('--batches',type=int,nargs='+',default=[1,5,10]);args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False);source=Path(args.bank);bank=json.loads((source/'frozen_bank.json').read_text());bridge=OTBridge(args.runner);rows=[]
    for spec in bank['specifications']:
        context=json.loads(Path(spec['parameters']).read_text());plan=list(csv.DictReader(Path(spec['plan']).open()));ego=np.array([[float(r['x']),float(r['y'])] for r in plan[1:]])
        for batch in args.batches:
            for policy in POLICIES:
                streams=[]
                for obs in context['obstacles']:
                    group=[]
                    for m,mode in enumerate(obs['modes']):
                        p=np.load(source/f'{spec["context"]}_obs{obs["id"]}_mode{m}.npz');expected=np.unpackbits(p['hits'])[:int(p['N'])];seed=[bank['seed_domain'],int(spec['context'][1:]),obs['id'],m];group.append(ConditionalStream(context,obs,mode,ego,seed,expected))
                    streams.append(group)
                result=replay(context,streams,bridge,policy,batch);result=dict(context=spec['context'],obstacles=len(context['obstacles']),**result);result['generation_seconds']=sum(s.generation for g in streams for s in g);result['collision_seconds']=sum(s.collision for g in streams for s in g);result['paired_samples_checked']=sum(s.checked for g in streams for s in g);rows.append(result);write(root/'decisions.csv',rows)
                print(spec['context'],batch,policy,result['status'],result['N'],round(result['seconds'],3),flush=True)
    bridge.close();files=[__file__,'tools/run_sequential_validation.py','tools/bernoulli_confidence_sequence.py','tools/make_sequential_confirmation.py',args.runner,str(source/'frozen_bank.json')];(root/'manifest.json').write_text(json.dumps(dict(command=sys.argv,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},profiling='on-demand complete conditional trajectories; geometry rechecked; exact matches to saved paired stream',warm_start=10),indent=2)+'\n')
if __name__=='__main__':main()
