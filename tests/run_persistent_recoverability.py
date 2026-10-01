#!/usr/bin/env python3
"""Bounded refinement after five-step refusal: larger separation and 5/15-step windows."""
import json
from pathlib import Path
from run_causal_link import ROOT,invoke,digest
from run_persistent_intervention import summarize
from causal_link_analysis import read,write


def run(root,case,seed,nd):
    folder=root/case['name']/f'seed_{seed}'/f'nd_{nd}'
    invoke(ROOT/'build-base/persistent_coverage_experiment',folder,
        ['late_switch',20,10,seed,0,120,5,'sh_mpcc',case['x'],case['window'],nd])
    r=read(folder/'summary.csv')[0];dd=read(folder/'decisions.csv')
    return dict(r,case=case['name'],seed=seed,target_nd=nd,window=case['window'],x=case['x'],
        window_controls_executed=sum(int(r['success']) for r in dd if int(r['step'])<case['window']),source=str(folder))


def main():
    root=ROOT/'results/persistent-coverage/recoverability-updated-belief';root.mkdir(parents=True,exist_ok=True)
    cases=[dict(name=f'x{x}_window{window}',x=x,window=window) for x in [6,7,8] for window in [5,15]]
    record=dict(cases=cases,pilot_seeds=list(range(2301,2321)),pilot_counts=[0,3],confirmation_seeds=list(range(2401,2501)),counts=[0,1,2,3,5,10],
        rationale='Five-step pilot was dominated by refusal. Test larger separation and a longer critical window; retain all failed pilots.',
        selection='first covered completion>=50%, uncovered collision/refusal>=20%, completion advantage>=20 percentage points; otherwise first recoverable covered>=50% case, explicitly no demonstrated coverage benefit',
        sources={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'tools/persistent_coverage_experiment.cpp']})
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=record:raise RuntimeError('changed refinement manifest')
    path.write_text(json.dumps(record,indent=2))
    rows=[run(root,c,s,n) for c in cases for s in record['pilot_seeds'] for n in [0,3]]
    write(root/'pilot_outcomes.csv',rows)
    scores=[]
    for c in cases:
        a=[r for r in rows if r['case']==c['name'] and r['target_nd']==0]
        b=[r for r in rows if r['case']==c['name'] and r['target_nd']==3]
        ca=sum(int(r['path_completed']) for r in a)/20;cb=sum(int(r['path_completed']) for r in b)/20
        fa=sum(int(r['collision']) or r['termination']=='no_admissible_control' for r in a)/20
        scores.append(dict(c,completion_0=ca,completion_3=cb,failure_0=fa,qualifies=cb>=.5 and fa>=.2 and cb-ca>=.2))
    write(root/'pilot_selection_scores.csv',scores)
    selected=next((c for c in scores if c['qualifies']),None)
    if selected is None:selected=next((c for c in scores if c['completion_3']>=.5),None)
    selection=dict(selected=selected,coverage_benefit_screen_passed=bool(selected and selected['qualifies']))
    path=root/'selection_before_confirmation.json'
    if path.exists() and json.loads(path.read_text())!=selection:raise RuntimeError('selection changed')
    path.write_text(json.dumps(selection,indent=2))
    if selected:
        confirmations=[run(root,selected,s,n) for s in record['confirmation_seeds'] for n in record['counts']]
        write(root/'confirmation_outcomes.csv',confirmations)
        # Reuse outcome statistics; calculate actual window completion separately (5 or 15).
        summary=[r for r in summarize(confirmations) if r['metric']!='window_completed']
        from causal_link_analysis import wilson
        for n in record['counts']:
            rr=[r for r in confirmations if r['target_nd']==n];hits=sum(r['window_controls_executed']==selected['window'] for r in rr);lo,hi=wilson(hits,len(rr))
            summary.append(dict(case=selected['name'],target_nd=n,metric='window_completed',seeds=len(rr),events=hits,rate=hits/len(rr),ci95_low=lo,ci95_high=hi))
        write(root/'confirmation_summary.csv',summary)

if __name__=='__main__':main()
