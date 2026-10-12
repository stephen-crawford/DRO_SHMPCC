#!/usr/bin/env python3
"""Offline LP witnesses and scenario deletion bounds; no controller changes."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def conflict(rows,deleted=frozenset()):
    kept=[r for r in rows if r['id'] not in deleted]
    a=np.array([[r['a_x'],r['a_y']] for r in kept]); b=np.array([r['b'] for r in kept])
    lp=linprog([0,0],A_ub=-a,b_ub=-b,bounds=[(None,None)]*2,method='highs')
    if lp.success:return None
    if lp.status!=2:raise RuntimeError(f'Unexpected feasibility LP: {lp.message}')
    relaxed=linprog([0,0,1],A_ub=np.column_stack((-a,-np.ones(len(a)))),b_ub=-b,bounds=[(None,None)]*2+[(0,None)],method='highs')
    if not relaxed.success:raise RuntimeError(relaxed.message)
    weights=-relaxed.ineqlin.marginals
    idx=np.flatnonzero(weights>1e-9)
    if len(idx)>3 or relaxed.fun<=1e-8:raise RuntimeError('Unreliable LP witness')
    # A two-facet conflict requires opposing parallel normals. Check all pairs
    # independently so a returned triad is not mislabelled as a smallest subset.
    for i in range(len(a)):
        cross=a[i,0]*a[i+1:,1]-a[i,1]*a[i+1:,0]
        norm=np.linalg.norm(a[i]); norms=np.linalg.norm(a[i+1:],axis=1)
        dot=a[i+1:]@a[i]
        near=np.flatnonzero((np.abs(cross)<1e-12*norm*norms)&(dot<0))
        for off in near:
            j=i+1+off
            if b[i]/norm+b[j]/norms[off]>1e-8:
                idx=np.array([i,j]);weights=np.zeros(len(a));weights[i]=1/norm;weights[j]=1/norms[off]
                weights/=weights.sum();break
        if len(idx)==2:break
    residual=weights@a; offset=float(weights@b)
    if np.linalg.norm(residual)>1e-8 or offset<=1e-8:raise RuntimeError('Invalid numerical Farkas witness')
    witness=[dict(kept[i],weight=float(weights[i])) for i in idx]
    cosines=[float(a[i]@a[j]/np.linalg.norm(a[i])/np.linalg.norm(a[j])) for pos,i in enumerate(idx) for j in idx[pos+1:]]
    return dict(size=len(idx),facets=witness,min_cosine=min(cosines) if cosines else None,weighted_normal=residual.tolist(),weighted_offset=offset,uniform_relaxation=float(relaxed.fun))


def removals(rows):
    # Branch on a witness: every feasible scenario deletion must hit it.
    # Exhaustive for <=2 deletions, with undeletable domain facets id=-1.
    def search(deleted,remaining):
        w=conflict(rows,deleted)
        if w is None:return sorted(deleted)
        if not remaining:return None
        for sid in sorted({r['id'] for r in w['facets'] if r['id']>=0}):
            result=search(deleted|{sid},remaining-1)
            if result is not None:return result
        return None
    for budget in range(3):
        result=search(frozenset(),budget)
        if result is not None:return dict(minimum=budget,deletion_ids=result)
    return dict(minimum='>=3',deletion_ids=None)


def horizon_removals(rows):
    groups=[[r for r in rows if (r['k'],r['disc'])==key]
            for key in sorted({(r['k'],r['disc']) for r in rows if r['k']>0})]
    cache={}
    def search(deleted,remaining):
        for index,group in enumerate(groups):
            key=(index,tuple(sorted(deleted)))
            if key not in cache:cache[key]=conflict(group,deleted)
            witness=cache[key]
            if witness is None:continue
            if not remaining:return None
            for sid in sorted({r['id'] for r in witness['facets']}):
                result=search(deleted|{sid},remaining-1)
                if result is not None:return result
            return None
        return sorted(deleted)
    for budget in range(3):
        result=search(frozenset(),budget)
        if result is not None:return dict(minimum=budget,deletion_ids=result)
    return dict(minimum='>=3',deletion_ids=None)


def read_rows(path):
    rows=list(csv.DictReader(path.open()))
    for r in rows:
        for key in ('k','disc','id','obstacle','removed_final'):r[key]=int(r[key])
        for key in ('a_x','a_y','b','anchor_x','anchor_y','anchor_slack','obs_x','obs_y'):r[key]=float(r[key])
    return rows


def export_first_events(root,events):
    fields=['context','replicate','sampling_seed','candidate','k','disc','witness_size',
            'minimum_cosine','same_obstacle','same_mode','stage_r_min',
            'bounded_stage_r_min','raw_horizon_r_min','anchor_slack_min',
            'anchor_slack_max','removed_after_first_event']
    with (root/'first_event_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for event in events:
            if not event['first_event']:continue
            witness=event['raw_witness']
            following=next((e for e in events if e['context']==event['context'] and
                            e['replicate']==event['replicate'] and not e['first_event']),None)
            item={key:event[key] for key in fields if key in event}
            item.update(witness_size=witness['size'] if witness else '',
                        minimum_cosine=witness['min_cosine'] if witness else '',
                        stage_r_min=event['initial_stage_removals']['minimum'],
                        bounded_stage_r_min=event['bounded_removals']['minimum'],
                        raw_horizon_r_min=event['raw_horizon_removals']['minimum'],
                        anchor_slack_min=min(r['anchor_slack'] for r in witness['facets']) if witness else '',
                        anchor_slack_max=max(r['anchor_slack'] for r in witness['facets']) if witness else '',
                        removed_after_first_event=following['removed_before'] if following else [])
            writer.writerow(item)


def main(root):
    events=[]; accepted=[]
    for seed in sorted(root.glob('seed_*')):
        for trial in csv.DictReader((seed/'replications.csv').open()):
            stem=f"{trial['candidate']}_rep{trial['replicate']}"
            path=seed/(stem+'_rows.csv')
            if not path.exists():continue
            rows=read_rows(path)
            if trial['failure_stage']!='facet_empty':
                if trial['certified']=='1':
                    minimum=1.
                    for key in sorted({(r['k'],r['disc']) for r in rows}):
                        group=[r for r in rows if (r['k'],r['disc'])==key]
                        a=np.array([[r['a_x'],r['a_y']] for r in group]);a/=np.linalg.norm(a,axis=1)[:,None]
                        minimum=min(minimum,float((a@a.T).min()))
                    accepted.append(dict(context=seed.name,replicate=int(trial['replicate']),candidate=trial['candidate'],min_cosine=minimum,snapshot='initial pre-reduction rows'))
                continue
            removed=set();first=True
            for line in (seed/(stem+'.log')).read_text().splitlines():
                if line.startswith('[SQP SCENARIO REMOVAL]'):removed.add(int(re.search(r'id=(\d+)',line)[1]))
                if not line.startswith('[FREE POLY EMPTY]'):continue
                fields=dict(re.findall(r'(\w+)=([^ ]+)',line)); k=int(fields['k']);disc=int(fields['disc'])
                raw=[r for r in rows if r['k']==k and r['disc']==disc]
                group=[r for r in raw if r['id'] not in removed]
                center=[float(fields['center_x']),float(fields['center_y'])];rho=float(fields['rho'])
                box=[]
                for axis in range(2):
                    for sign in (-1,1):
                        a=[0.,0.];a[axis]=sign
                        box.append(dict(id=-1,obstacle=-1,mode='domain',k=k,disc=disc,a_x=a[0],a_y=a[1],b=sign*center[axis]-rho))
                witness=conflict(group);bounded=conflict(group+box)
                if bounded is None:raise RuntimeError('Production empty polygon is feasible in independent bounded LP')
                item=dict(context=seed.name,replicate=int(trial['replicate']),sampling_seed=int(trial['sampling_seed']),candidate=trial['candidate'],first_event=first,k=k,disc=disc,removed_before=sorted(removed),rows=len(group),raw_witness=witness,bounded_witness=bounded,raw_removals=removals(group),bounded_removals=removals(group+box),initial_stage_removals=removals(raw),box_center=center,box_rho=rho)
                if first:
                    item['raw_horizon_removals']=horizon_removals(rows)
                if witness:
                    fs=witness['facets'];item.update(same_obstacle=len({r['obstacle'] for r in fs})==1,same_mode=len({r['mode'] for r in fs})==1)
                events.append(item);first=False
                # Complete pre-reduction stage including already removed rows.
                (seed/(stem+f'_stage_{len([e for e in events if e["context"]==seed.name and e["replicate"]==item["replicate"]])}.json')).write_text(json.dumps(dict(rows=raw,removed_before=sorted(removed),box=box),indent=2)+'\n')
    first=[e for e in events if e['first_event']]
    summary=dict(empty_batches=len(first),empty_events=len(events),raw_infeasible=sum(e['raw_witness'] is not None for e in first),witness_sizes={str(n):sum(e['raw_witness'] is not None and e['raw_witness']['size']==n for e in first) for n in (2,3)},cosine_below_minus_09=sum(e['raw_witness'] is not None and e['raw_witness']['min_cosine']<-.9 for e in first),initial_stage_removals={str(n):sum(str(e['initial_stage_removals']['minimum'])==str(n) for e in first) for n in (0,1,2,'>=3')},accepted_comparisons=accepted)
    print('raw horizon removal counts:',{str(n):sum(str(e['raw_horizon_removals']['minimum'])==str(n) for e in first) for n in (0,1,2,'>=3')})
    export_first_events(root,events)
    (root/'conflicts.json').write_text(json.dumps(dict(summary=summary,events=events),indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);main(ap.parse_args().root)
