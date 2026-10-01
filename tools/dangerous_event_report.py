"""Paired decision evidence and descriptive figures; no controller changes."""
from collections import defaultdict
import json
from analyze_comparison_results import KEYS, number, write


def labels(rows, previous):
    probabilities = [number(r.get('nominal_probability')) for r in rows]
    winners = sorted(r['mode'] for r,p in zip(rows, probabilities)
                     if p is not None and p == max(v for v in probabilities if v is not None)) if any(p is not None for p in probabilities) else []
    truth = {r.get('true_mode','') for r in rows}
    prior = {r.get('true_mode','') for r in previous}
    switched = int(truth != prior) if len(truth)==len(prior)==1 and '' not in truth|prior else ''
    return dict(map_mode=winners[0] if len(winners)==1 else '',map_modes=json.dumps(winners),
                map_tie=int(len(winners)>1),switch_event=switched)


def pair_row(common, ref, nominal, wdro_rows, decisions, runkey, annotation, dangerous):
    row = {k:common[k] for k in list(KEYS)+['step','obstacle_id']}
    row.update(annotation, true_mode=ref['true_mode'],risk_score=ref['true_risk'],
        nominal_probability=ref['true_p'],wdro_probability=ref['true_q'],
        nominal_count=nominal[ref['true_mode']],wdro_count=ref['true_count'],
        delta_q=ref['true_q']-ref['true_p'],delta_count=ref['true_count']-nominal[ref['true_mode']],
        scenario_count=ref['scenario_count'],dangerous_event=int(dangerous),
        risk_reference='wdro_pre_reweight_plan',
        wdro_source_artifact=wdro_rows[ref['true_mode']].get('source_artifact',''))
    source=wdro_rows[ref['true_mode']]
    row['reference_clearance']=source.get('reference_clearance','')
    row['reference_risk']=source.get('reference_risk','')
    for label,style in [('nominal','sh_mpcc'),('wdro','sh_mpcc_dro')]:
        outcome=decisions.get(runkey+(style,common['step']),{})
        success=number(outcome.get('success'))
        row[label+'_success']=success if success in (0,1) else ''
        row[label+'_outcome']='admissible' if success==1 else 'inadmissible' if success==0 else 'missing'
        row[label+'_decision_source']=outcome.get('source_artifact','')
        for field in ('ego_speed','acceleration','omega','actual_clearance','collision'):
            row[label+'_'+field]=outcome.get(field,'')
    a,b=number(row['nominal_success']),number(row['wdro_success'])
    row['delta_success']=b-a if a is not None and b is not None else ''
    return row


def export(output, rows):
    fields=list(KEYS)+['step','obstacle_id','true_mode','risk_score','nominal_probability',
        'wdro_probability','nominal_count','wdro_count','nominal_outcome','wdro_outcome',
        'map_mode','map_modes','map_tie','switch_event','dangerous_event','delta_q','delta_count','delta_success']
    write(output/'dangerous_event_pairs.csv',rows,fields)
    with (output/'dangerous_event_pairs.log').open('w') as file:
        for row in rows:file.write(json.dumps(row,allow_nan=False)+'\n')
    print(f'[DANGEROUS EVENTS] matched={len(rows)} dangerous={sum(r["dangerous_event"] for r in rows)}; measurements: {output/"dangerous_event_pairs.log"}')


def plot(table, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_comparison_results import read
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in read(table):
        groups[tuple(row[k] for k in KEYS if k!='seed')].append(row)
    manifest=[]
    for index,(identity,all_rows) in enumerate(sorted(groups.items())):
        rows=[r for r in all_rows if r['dangerous_event']=='1']
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,x,y,title in zip(axes,['risk_score','delta_q','delta_count'],
                ['delta_q','delta_count','delta_success'],
                ['Risk → mass shift','Mass shift → sample increase','Sample increase → action outcome']):
            points=[(number(r.get(x)),number(r.get(y))) for r in rows]
            points=[p for p in points if None not in p]
            if points:ax.scatter(*zip(*points),s=18,alpha=.35)
            else:ax.text(.5,.5,'No eligible observations',ha='center',transform=ax.transAxes)
            ax.axhline(0,color='grey',linewidth=.7);ax.set(xlabel=x,ylabel=y,title=title)
        axes[2].set_yticks([-1,0,1],['WDRO worse','tie','WDRO better'])
        fig.suptitle(f'{identity[2]} | dangerous obstacle-decisions={len(rows)} | seeds={len({r["seed"] for r in rows})}',fontsize=11)
        fig.text(.5,.01,'Danger labels: '+','.join(sorted({r.get('event_risk_reference','wdro') for r in all_rows}))+'; associations, not causal proof. Outcome = decision admissibility.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.91))
        stem=f'dangerous_chain_{index:03d}'
        for extension in ['png','pdf']:fig.savefig(output/(stem+'.'+extension),dpi=180)
        plt.close(fig)
        manifest.append(dict(zip([k for k in KEYS if k!='seed'],identity),figure=stem,dangerous_rows=len(rows)))
    (output/'figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'[FIGURES] {len(manifest)} condition-specific figures in {output}')


def plot_stress(table, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_comparison_results import read
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in read(table):groups[row['pair_case'],row['seed']].append(row)
    manifest=[]
    for index,((case,seed),rows) in enumerate(sorted(groups.items())):
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,field in zip(axes.flat,['nominal_probability','sampling_probability','sampled_count','risk_score','ego_speed','actual_clearance']):
            for style in sorted({r['solver_style'] for r in rows}):
                points=sorted((float(r['time_seconds']),number(r.get(field))) for r in rows if r['solver_style']==style and number(r.get(field)) is not None)
                if points:ax.plot(*zip(*points),label=style,linewidth=1)
            ax.axvline(float(rows[0]['switch_time_seconds']),color='black',linestyle='--',linewidth=.8)
            ax.set(xlabel='time (s)',ylabel=field)
        axes[0,0].legend(fontsize=6)
        fig.suptitle(f'{case}, seed {seed}: prescribed switch (dashed); stopped arms end at termination')
        fig.tight_layout(rect=(0,0,1,.95));stem=f'stress_timeline_{index:03d}'
        for ext in ['png','pdf']:fig.savefig(output/(stem+'.'+ext),dpi=180)
        plt.close(fig);manifest.append(dict(pair_case=case,seed=seed,figure=stem))
    (output/'figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
