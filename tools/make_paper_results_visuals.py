#!/usr/bin/env python3
"""Publication figures with explicit updated/offline/historical provenance."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'nominal':'#2769A0','envelope_optimal':'#539843','wasserstein_min_zeta':'#9263A5','full_proposed':'#D66D22','reweight_only':'#539843','baseline':'#2769A0','full_transfer':'#D66D22','envelope_minimum':'#539843'}
LABELS={'nominal':'Nominal','envelope_optimal':'Envelope optimum','wasserstein_min_zeta':'Wasserstein min. ζ','full_proposed':'Full proposed','reweight_only':'Reweight only','baseline':'Nominal SH-MPC','full_transfer':'Full transfer','envelope_minimum':'Envelope minimum'}
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def main():
 p=argparse.ArgumentParser();p.add_argument('--new',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 for src in a.new.glob('*.csv'):
  if src.name!='csv_manifest.csv':shutil.copy2(src,out/src.name)
 for src in a.legacy.glob('*.csv'):shutil.copy2(src,out/('historical_'+src.name))
 data=lambda name:pd.read_csv(out/name)
 captions=[]
 def save(fig,name,caption):
  for ext in ('pdf','svg','png'):fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
  plt.close(fig);captions.append(dict(figure=name,caption=caption))
 def grid(ax):ax.grid(alpha=.2,linewidth=.5)
 def bands(df,x,y,groups):
  rows=[]
  for group in groups:
   for value,d in df[df.arm==group].groupby(x):
    v=d[y].to_numpy();rows.append(dict(arm=group,**{x:value},median=float(np.median(v)),p05=float(np.percentile(v,5)),p95=float(np.percentile(v,95)),n=len(v),quantity=y))
  return pd.DataFrame(rows)
 def drawbands(ax,summary,x):
  for arm,d in summary.groupby('arm',sort=False):
   d=d.sort_values(x);ax.plot(d[x],d['median'],marker='x' if arm=='full_proposed' else 'o',markersize=4,lw=1.4,ls='--' if arm=='full_proposed' and summary.arm.nunique()>1 else '-',color=COLORS[arm],label=LABELS[arm]);ax.fill_between(d[x],d.p05,d.p95,color=COLORS[arm],alpha=.12,linewidth=0)
  grid(ax)

 # Updated held-out study: distinguish estimates, pointwise uncertainty, and measured mismatch.
 held=data('heldout_decisions.csv');evaluated=held[held.n_validation>0];table2=data('table2_shift_summary.csv')
 risk_plot=[]
 for (delta,arm),d in evaluated.groupby(['delta','arm']):
  per_seed=d.groupby('seed').violation_estimate.mean()
  risk_plot.append(dict(delta=delta,arm=arm,mean_violation=per_seed.mean(),median_pointwise_CP95_upper=d.cp95_upper.median(),p95_pointwise_CP95_upper=d.cp95_upper.quantile(.95),fraction_estimate_above_epsilon=(d.violation_estimate>.05).mean(),mean_calibration_tv=d.groupby('seed').calibration_tv.first().mean(),seeds=d.seed.nunique(),decisions=len(d)))
 risk_plot=pd.DataFrame(risk_plot);risk_plot.to_csv(out/'plot_true_law_validation.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(9.6,3.1))
 for arm in ['nominal','reweight_only','full_proposed']:
  d=risk_plot[risk_plot.arm==arm].sort_values('delta');color=COLORS[arm]
  style=dict(marker='o' if arm=='nominal' else 'x',markersize=5 if arm=='nominal' else 3,markerfacecolor='none',ls='--' if arm=='reweight_only' else '-')
  axes[0].plot(d.delta,d.mean_violation,color=color,label=LABELS[arm],**style)
  axes[0].plot(d.delta,d.median_pointwise_CP95_upper,ls=':',lw=1,color=color)
  axes[1].plot(d.delta,d.fraction_estimate_above_epsilon,color=color,label=LABELS[arm],**style)
  axes[2].plot(d.delta,d.mean_calibration_tv,color=color,label=LABELS[arm],**style)
 axes[0].axhline(.05,color='black',ls='--',lw=.8,label='Target ε = 0.05');axes[0].set(title='(a) Frozen-plan true-law violation',ylabel='Violation probability',ylim=(-.001,.055))
 axes[0].set_yscale('symlog',linthresh=1e-5);axes[0].set_ylim(0,.07)
 axes[0].text(.02,.77,'Dotted: median pointwise\n95% CP upper endpoint',transform=axes[0].transAxes,fontsize=6.5)
 axes[1].set(title='(b) Decisions above the target',ylabel='Fraction with V̂ > 0.05',ylim=(-.015,max(.1,float(risk_plot.fraction_estimate_above_epsilon.max())*1.2)))
 axes[2].set(title='(c) Realized calibration mismatch',ylabel='Mean TV(p̂, p*)')
 for ax in axes:ax.set_xlabel('True-law shift δ');ax.set_xticks([0,.2,.4,.6]);grid(ax)
 handles,labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=4)
 fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.88,wspace=.42)
 save(fig,'figure1_true_law_validation','Updated formulation; straight interaction; 30 paired seeds per shift and arm; 1000 independent held-mode Gaussian validation trajectories per executable returned plan. Solid curves show equal-seed-weighted mean violation estimates on a symmetric-log probability axis (linear below1e-5). Dotted curves show median pointwise 95% Clopper–Pearson upper endpoints, not confidence bands for the curve. Calibration is redrawn from each true law; TV is measured rather than assumed to increase with δ. Decisions within a rollout are dependent. The physical simulation redraws modes each step, whereas held-out horizon validation holds each sampled mode fixed.')

 # Updated offline re-solves on historical frozen inputs.
 frozen=data('frozen_law_ablation.csv');frozen=frozen[frozen.arm.isin(['nominal','envelope_optimal','wasserstein_min_zeta','full_proposed'])].copy()
 frozen['setup']=frozen.V.astype(str)+' obstacle(s), '+frozen.M.astype(str)+' modes'
 law_plot=[]
 for (setup,arm),d in frozen.groupby(['setup','arm']):
  r=dict(setup=setup,arm=arm,obstacle_decisions=len(d))
  for field in ['normalized_zeta','dangerous_mass_change']:
   for suffix,pct in [('median',50),('p05',5),('p95',95)]:r[field+'_'+suffix]=np.percentile(d[field],pct)
  law_plot.append(r)
 law_plot=pd.DataFrame(law_plot);law_plot.to_csv(out/'plot_frozen_law_ablation.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(7.16,2.8));arms=['envelope_optimal','nominal','wasserstein_min_zeta','full_proposed'];setups=sorted(law_plot.setup.unique())
 for sidx,setup in enumerate(setups):
  for idx,arm in enumerate(arms):
   r=law_plot[(law_plot.setup==setup)&(law_plot.arm==arm)].iloc[0];x=idx+(sidx-(len(setups)-1)/2)*.19
   for ax,field in zip(axes,['normalized_zeta','dangerous_mass_change']):
    ax.errorbar(x,r[field+'_median'],yerr=[[max(0,r[field+'_median']-r[field+'_p05'])],[max(0,r[field+'_p95']-r[field+'_median'])]],fmt='o' if sidx==0 else 's',markersize=4,capsize=3,color=COLORS[arm],label=setup if idx==0 else None)
 for ax in axes:ax.set_xticks(range(4),['Envelope\noptimum','Nominal','Wasserstein\nmin. ζ','Full risk\naware']);grid(ax)
 axes[0].axhline(1,color='black',ls=':',lw=.8);axes[0].set(title='(a) Certificate-tightness cost',ylabel='ζ(q) / Σu');axes[1].axhline(0,color='black',ls=':',lw=.8);axes[1].set(title='(b) Dangerous-mode mass change',ylabel='Σ dangerous (q − p̂)');axes[0].legend(fontsize=6);fig.tight_layout()
 save(fig,'figure2_sampling_law_ablation','Offline re-solves of 6870 archived obstacle-decision inputs under the updated CP envelope. Points are medians; whiskers span the empirical 5th–95th percentiles, not independent-sample confidence intervals. Circle/square identify one-obstacle/two-obstacle archived settings. The frozen state, nominal law, risk scores, geometry and radius are held fixed across laws. Re-solved laws were not executed on these archived trajectories. Rows with no logged DRO inputs cannot be reconstructed; coverage is explicit in frozen_law_coverage.csv.')

 # Controlled product scaling with 90% empirical calibration intervals.
 scaling=data('scaling_instances.csv');groups=['envelope_minimum','nominal','full_proposed'];fig,axes=plt.subplots(1,2,figsize=(7.16,2.8));sc=[]
 for ax,field,label in zip(axes,['zeta_joint','required_S'],['Joint domination factor ζ','Required scenarios Smin']):
  plot=bands(scaling,'V',field,groups);sc.append(plot);drawbands(ax,plot,'V');ax.set(xlabel='Number of obstacles V',ylabel=label,yscale='log');ax.set_xticks(range(1,7))
 pd.concat(sc).to_csv(out/'plot_obstacle_scaling.csv',index=False);axes[0].set_title('(a) Multiplicative tightening');axes[1].set_title('(b) Scenario-count consequence');axes[0].legend(fontsize=6);fig.tight_layout()
 save(fig,'figure3_obstacle_scaling','Updated offline construction: 100 independent calibration seed sets, V=1–6, M=3 and g=100 per obstacle. Curves show medians and bands the empirical 5th–95th percentiles. Per-obstacle beta=.05/6 is held fixed across V; total allocated beta never exceeds .05. Prefix pairing across V isolates compounding. Smin uses the production SH bound with epsilon=.05, beta_SH=.01, support cap6 and removal budget2. These points are offline sizing calculations, not controller runtimes.')

 # Calibration sweep adds convergence information without mixing outcome studies.
 cert=data('certificate_instances.csv');cal=cert[cert.M==5];fig,axes=plt.subplots(2,2,figsize=(7.16,4.6));calplots=[]
 for ax,field,label in zip(axes.flat,['minimum_zeta','zeta','rho','required_S'],['Minimum factor Σu','Selected factor ζ','Transport design budget ρ','Required scenarios Smin']):
  sub=cal[cal.arm=='full_proposed'];plot=bands(sub,'g',field,['full_proposed']);calplots.append(plot);drawbands(ax,plot,'g');ax.set(xlabel='Calibration observations g',ylabel=label,xscale='log');ax.set_xticks([10,25,50,100,250,500,1000],['10','25','50','100','250','500','1000'])
  if field in ['minimum_zeta','zeta']:ax.axhline(1,color='black',ls=':',lw=.8)
  if field=='required_S':ax.set_yscale('log')
 pd.concat(calplots).to_csv(out/'plot_calibration_sweep.csv',index=False);fig.tight_layout()
 save(fig,'figure4_calibration_evidence','Updated five-mode construction, 100 independent calibration draws at each g in {10,25,50,100,250,500,1000}. Median and empirical 90% intervals for the envelope lower bound, proposed factor, transport budget, and implied S. Geometry and the underlying categorical law are fixed. The radius-calibration cost includes beta quantiles, vertex enumeration and transport solves; separate timing data are in radius_runtime.csv.')

 # User-supplied historical data: use exact cohort, without assigning new formulation.
 progress=data('historical_figure1_progress.csv');med=data('historical_figure1_progress_medians.csv');latency=data('historical_figure1_latency.csv');fig,axes=plt.subplots(1,2,figsize=(7.16,2.8))
 for arm in ['baseline','full_transfer']:
  for _,d in progress[progress.arm==arm].groupby(['pair','seed']):axes[0].plot(d.step,d.normalized_progress,color=COLORS[arm],alpha=.14,lw=.6)
  d=med[med.arm==arm];axes[0].plot(d.step,d.median_normalized_progress,color=COLORS[arm],lw=1.6,label=LABELS[arm])
  v=np.sort(latency[latency.arm==arm].solve_ms);axes[1].step(v,np.arange(1,len(v)+1)/len(v),where='post',color=COLORS[arm],label=LABELS[arm])
 axes[0].axhline(.95,color='black',ls=':',lw=.8);axes[0].set(xlabel='Control step',ylabel='Normalized route progress',xlim=(0,200),ylim=(0,1.02),title='(a) Historical moderate-case progress')
 axes[1].axvline(100,color='black',ls=':',lw=.8,label='100 ms period');axes[1].set(xlabel='Control-cycle wall time [ms]',ylabel='Empirical CDF',xscale='log',ylim=(0,1.01),title='(b) Historical computational cost')
 for ax in axes:grid(ax);ax.legend(fontsize=6)
 fig.tight_layout();save(fig,'figure5_historical_practicality','User-supplied historical CSVs, earlier outer-ball-envelope formulation. Original nine runs per arm: straight, S-curve and intersection, seeds77–79. Faint trajectories stop at observed endpoints; median traces use the existing completed-terminal-value carry convention. This selected original cohort must be reported alongside the complete historical ablation table, which retains incomplete routes and an execution error. It is not an updated-formulation comparison.')
 mech=data('historical_figure2_mechanism.csv');fig,axes=plt.subplots(2,2,figsize=(7.16,4.2))
 for ax,field,label in zip(axes.flat,['rho_max','zeta','S_over_S0','risk_lift_sum'],['Transport radius ρ','Domination factor ζ','Scenario inflation S/S0','Risk-score lift ⟨q − p̂, r⟩']):
  ax.plot(mech.step,mech[field],color=COLORS['full_proposed'],lw=1.3);ax.set(xlabel='Control step',ylabel=label);grid(ax)
 fig.tight_layout();save(fig,'figure6_historical_mechanism','Historical one-obstacle S-curve full-transfer seed77 from the user-supplied mechanism CSV. These logged values belong to the earlier formulation; they are retained for continuity and must not be described as a new CP-envelope rollout. Risk-score lift is a sampling-selection diagnostic, not true collision probability.')

 # Compact publication tables; corresponding CSVs remain the authoritative precision.
 table1=cert[(cert.M==5)&(cert.g==100)].groupby('arm').agg(ratio=('normalized_zeta','median'),target=('epsilon_SH','median'),S=('required_S','median'),ms=('construction_total_ms','median'),danger=('dangerous_mass_change','median')).reindex(['envelope_optimal','nominal','wasserstein_min_zeta','full_proposed'])
 def tablefigure(name,headers,rows,caption,width=9):
  fig,ax=plt.subplots(figsize=(width,.55+.29*len(rows)));ax.axis('off');tab=ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='center');tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.3)
  for (i,j),cell in tab.get_celld().items():
   cell.set_linewidth(.35)
   if i==0:cell.set_facecolor('#E7EDF2');cell.set_text_props(weight='bold')
   elif i%2==0:cell.set_facecolor('#F6F8FA')
  save(fig,name,caption)
 rows=[[LABELS[arm],f'{r.ratio:.3f}',f'{r.target:.4f}',f'{r.S:,.1f}',f'{r.ms:.3f}',f'{r.danger:.4f}'] for arm,r in table1.iterrows()]
 tablefigure('table1_certificate_construction',['Law','ζ / Σu','ε / ζ','Required S','Total ms','Δ dangerous mass'],rows,'Updated five-mode, g100 calibration construction, medians over100 independent draws. S can have a half-integer median because it summarizes integer counts. C++ and Python/HiGHS runtime implementations differ; timing is not a controlled solver comparison. Total time includes the shared CP-radius construction once.')
 rows=[]
 for _,r in table2.sort_values(['delta','arm']).iterrows():
  rows.append([f'{r.delta:g}',LABELS[r.arm],f'{100*r.mean_true_violation_rollout_weighted:.4f}',f'{int(r.decisions_estimate_exceeds_epsilon)}/{int(r.evaluated_decisions)}',f'{int(r.completed)}/{int(r.attempted)}',f'{int(r.execution_errors)}',f'{r.median_solve_ms_concurrent:.1f}'])
 tablefigure('table2_true_law_results',['Shift δ','Controller','Mean V̂ [%]','V̂ > ε / evaluated','Complete','Errors','Median ms*'],rows,'Updated straight-geometry pilot,30 paired seeds per shift/controller. Mean V is equal-rollout-weighted. Per-decision exact95% Clopper–Pearson intervals are in heldout_decisions.csv. Completion includes all attempts; errors have unknown final collision/completion metrics. *Wall times come from four concurrent processes and exclude independent validation; not an isolated real-time claim.',width=10)
 write=lambda path,rows:pd.DataFrame(rows).to_csv(path,index=False)
 write(out/'figure_captions.csv',captions)
 oldsum=data('historical_ablation_summary.csv')
 newriskmax=float(evaluated.violation_estimate.max());newviol=int(evaluated.violations.sum())
 readme=f'''# Results visuals and CSVs — 2 October 2026

Start with figure1_true_law_validation, figure2_sampling_law_ablation, figure3_obstacle_scaling, and the two tables. Figures are supplied as PDF, editable SVG and 300-dpi PNG. Exact captions and scope are in figure_captions.csv; experiment settings and limitations are in experiment_protocol.csv.

Three distinct evidence sources are kept separate:

1. **Updated formulation, new executions:** the true-law validation and independent calibration/scaling sweeps.
2. **Updated formulation, offline re-solves:** frozen_law_*.csv recalculates alternative laws on archived per-obstacle decision inputs. These q values were not executed on the old trajectories. All 6,870 reconstructable DRO obstacle decisions are included; nominal-only/failed runs lack some required inputs. Calibration counts were reconstructed from logged KT probabilities and sample counts, checked to reproduce the original probabilities to1e-10.
3. **Historical executed data:** historical_*.csv are exact copies of the user-supplied results directory, preserving the earlier outer-ball formulation. Figures5–6 visualize only these historical runs. Do not combine their timing or completion outcomes with new trials as if they shared an algorithm.

The focused shift study has {len(pd.read_csv(out/'shift_rollouts.csv'))} attempted runs, {len(evaluated):,} executable evaluated decisions and {int(evaluated.n_validation.sum()):,} independent validation trajectories. It observed {newviol:,} violating held-out trajectories; the maximum per-decision estimate was {newriskmax:.6g}. These are observed finite-sample results, not proof of safety or superiority. Pointwise intervals do not yield a simultaneous or mission-wide guarantee. Replanning decisions within a rollout are dependent. Calibration draws come from each shifted truth, so the actual nominal/true mismatch is estimated and logged rather than forced to equal delta.

All prior data are retained, including the earlier execution error. Four-way/roundabout evidence must not be inferred from the focused one-geometry shift study. The deleted intermediate three-geometry pilot is not included because its raw validation data were unavailable; execution logs alone were not used as numerical evidence. The new staging data were regenerated.

Construction runtime includes CP endpoints, enumeration and all transport solves; per-vertex OT microtimings were not measured. Radius mode-count sweep M2–8 uses the production catalog through M6 and explicit affine-drift additions for M7–8. See protocol for geometry and seeds. Closed-loop wall times used four concurrent processes and exclude validation time.

All raw numeric records used by the figures are in flat CSV files, including plot_* data. package_csv_manifest.csv records file hashes. SHA256SUMS.txt covers all packaged files except itself. make_paper_results_visuals.py and related experiment scripts are included for reproducibility; dependencies are those of the existing project plus NumPy, SciPy, pandas and matplotlib.
'''
 (out/'README.md').write_text(readme)
 # Exact provenance for the user's supplied inputs.
 write(out/'historical_input_provenance.csv',[dict(source=str(p.resolve()),archive_file='historical_'+p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.legacy.glob('*.csv'))])
 scriptroot=Path(__file__).resolve().parents[1]
 for name in ['tools/make_paper_results_visuals.py','tools/report_paper_families.py','tests/run_paper_certificate_families.py','tests/run_paper_shift_family.py','tests/analyze_paper_frozen_decisions.py','tools/paper_certificate_probe.cpp','tools/paper_shift_experiment.cpp','tools/paper_frozen_allocator.cpp']:
  shutil.copy2(scriptroot/name,out/Path(name).name)
 files=[p for p in out.glob('*.csv') if p.name not in ['csv_manifest.csv','package_csv_manifest.csv']]
 write(out/'package_csv_manifest.csv',[dict(file=p.name,rows=sum(1 for _ in csv.DictReader(p.open())),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
 (out/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
 archive=out.parent/(out.name+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for src in sorted(out.iterdir()):
   if src.is_file():z.write(src,src.name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,figures=len(captions),files=len(list(out.iterdir())),heldout_decisions=len(evaluated),violations=newviol),indent=2))
if __name__=='__main__':main()
