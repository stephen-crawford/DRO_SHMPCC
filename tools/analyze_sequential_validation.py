#!/usr/bin/env python3
"""Aggregate paired anytime decisions and independently check terminal OT risks."""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dro-sequential-mpl')
import numpy as np
from scipy.stats import beta
from oracle_mode_validation import upper,write
from bernoulli_confidence_sequence import ConfidenceSequence
from analyze_mode_validation import lp_risk

def main():
    ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('--bank',required=True);args=ap.parse_args();root=Path(args.results);source=Path(args.bank);bank=json.loads((source/'frozen_bank.json').read_text());specs={s['context']:s for s in bank['specifications']};rows=list(csv.DictReader((root/'decisions.csv').open()));bands=[];reference={};checks=0
    pairs=sum(len(o['modes']) for s in specs.values() for o in json.loads(Path(s['parameters']).read_text())['obstacles']);alpha=.01/(2*pairs)
    for name,spec in specs.items():
        c=json.loads(Path(spec['parameters']).read_text());L=[];U=[];V=[]
        for o in c['obstacles']:
            ls=[];us=[];vs=[]
            for m in range(len(o['modes'])):
                p=np.load(source/f'{name}_obs{o["id"]}_mode{m}.npz');h=np.unpackbits(p['hits'])[:int(p['N'])];n=len(h);k=int(h.sum());ls.append(0. if k==0 else float(beta.ppf(alpha,k,n-k+1)));us.append(upper(k,n,alpha));vs.append(k/n)
            L.append(lp_risk(o,ls));U.append(lp_risk(o,us));V.append(lp_risk(o,vs))
        low=min(1.,sum(L));high=min(1.,sum(U));label='safe' if high<=.05 else ('reject' if low>.05 else 'boundary_uncertain');reference[name]=label;bands.append(dict(context=name,label=label,R_point=min(1.,sum(V)),R_lower=low,R_upper=high))
    wrong=0;empty=0;terminal_cache={}
    # Each recorded terminal prefix is replayed from n=0; stopping flags are
    # independently compared against SciPy/HiGHS objective evaluations.
    for row in rows:
        name=row['context'];c=json.loads(Path(specs[name]['parameters']).read_text());ns=json.loads(row['counts']);ks=json.loads(row['hits']);M=len(ns);offset=0;risksL=[];risksU=[]
        assert sum(ns)==int(row['N'])
        for o in c['obstacles']:
            ls=[];us=[]
            for m in range(len(o['modes'])):
                key=(name,o['id'],m)
                if key not in terminal_cache:
                    requested={json.loads(r['counts'])[offset] for r in rows if r['context']==name};p=np.load(source/f'{name}_obs{o["id"]}_mode{m}.npz');h=np.unpackbits(p['hits'])[:max(requested)];cs=ConfidenceSequence(.01/M);stored={}
                    for n,y in enumerate(h,1):
                        cs.observe(int(y))
                        if n in requested:stored[n]=(cs.lower,cs.upper,cs.k)
                    terminal_cache[key]=stored
                lo,hi,k=terminal_cache[key][ns[offset]];assert k==ks[offset];ls.append(lo);us.append(hi);offset+=1;checks+=1
            risksL.append(lp_risk(o,ls));risksU.append(lp_risk(o,us));checks+=2
        low=min(1.,sum(risksL));high=min(1.,sum(risksU));assert abs(low-float(row['R_L']))<1e-8 and abs(high-float(row['R_U']))<1e-8
        if row['status']=='certify':assert high<=.05
        if row['status']=='reject':assert low>.05
        empty+=row['reason']=='empty_confidence_intersection';label=reference[name]
        wrong+=int((label=='safe' and row['status']=='reject') or (label=='reject' and row['status']=='certify'))
    summary=dict(rows=len(rows),contexts=len(specs),terminal_checks=checks,reference={label:sum(v==label for v in reference.values()) for label in ('safe','reject','boundary_uncertain')},reference_disagreements=wrong,empty_intersections=empty,variants={})
    for batch in sorted({int(r['batch']) for r in rows}):
        for policy in sorted({r['policy'] for r in rows}):
            subset=[r for r in rows if int(r['batch'])==batch and r['policy']==policy];baseline={r['context']:r for r in rows if int(r['batch'])==batch and r['policy']=='uniform'};entry=dict(contexts=len(subset),certify=sum(r['status']=='certify' for r in subset),reject=sum(r['status']=='reject' for r in subset),unresolved=sum(r['status']=='unresolved' for r in subset),total_N=sum(int(r['N']) for r in subset),median_N=float(np.median([int(r['N']) for r in subset])),p90_N=float(np.percentile([int(r['N']) for r in subset],90)),median_seconds=float(np.median([float(r['seconds']) for r in subset])),median_sample_saving=float(np.median([1-int(r['N'])/int(baseline[r['context']]['N']) for r in subset])),median_time_saving=float(np.median([1-float(r['seconds'])/float(baseline[r['context']]['seconds']) for r in subset])))
            for label in ('safe','reject','boundary_uncertain'):
                selected=[r for r in subset if reference[r['context']]==label];entry[label]=dict(contexts=len(selected),resolved=sum(r['status']!='unresolved' for r in selected),median_saving=float(np.median([1-int(r['N'])/int(baseline[r['context']]['N']) for r in selected])) if selected else None)
            summary['variants'][f'b{batch}_{policy}']=entry
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');write(root/'reference.csv',bands)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
