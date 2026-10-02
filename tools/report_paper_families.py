#!/usr/bin/env python3
"""Aggregate the three paper families and create a flat CSV-only results archive."""
import argparse,csv,hashlib,json,math,statistics,zipfile
from collections import defaultdict
from pathlib import Path

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:raise ValueError('empty output '+str(p))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
def percentile(values,p):
 values=sorted(values);i=(len(values)-1)*p;a=math.floor(i);b=math.ceil(i);return values[a]+(values[b]-values[a])*(i-a)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args();root=a.input
 manifest=json.loads((root/'shift_manifest.json').read_text())
 expected=len(manifest['seeds'])*len(manifest['geometries'])*len(manifest['shifts'])*len(manifest['arms'])
 metadata=[json.loads(p.read_text()) for p in sorted((root/'shift_raw').glob('*/run.json'))]
 if not a.partial and len(metadata)!=expected:raise ValueError(f'experiment incomplete: {len(metadata)}/{expected}')
 rollouts=[];decisions=[];laws=[]
 for meta in metadata:
  d=root/'shift_raw'/meta['case'];rows=read(d/'rollout.csv') if (d/'rollout.csv').exists() else []
  if len(rows)==1:
   r=rows[0]
   if None in r:raise ValueError('invalid rollout CSV schema: '+str(d))
  else:
   r=dict(geometry=meta['geometry'],delta=meta['delta'],seed=meta['seed'],arm=meta['arm'],status='ERROR',error=str(meta['exit_code']),decisions='',evaluated_decisions='',completed='',steps='',collision='',min_clearance='',progress='',termination='',median_solve_ms='',p95_solve_ms='')
  r.update(case=meta['case'],process_exit=str(meta['exit_code']),run_wall_seconds=meta['wall_seconds'],concurrent_jobs=manifest['jobs'])
  rollouts.append(r)
  for name,target in [('heldout_decisions.csv',decisions),('laws.csv',laws)]:
   if not (d/name).exists():continue
   for row in read(d/name):
    if None in row or any(v is None for v in row.values()):raise ValueError('invalid CSV schema: '+str(d/name))
    row.update(case=meta['case'],rollout_status=r['status']);target.append(row)
 # Verify paired calibration laws and obstacle-state prefixes, including failed prefixes.
 groups=defaultdict(list)
 for r in rollouts:groups[r['geometry'],str(float(r['delta'])),str(r['seed'])].append(r)
 bycase=defaultdict(list)
 for d in decisions:bycase[d['case']].append(d)
 lawcase=defaultdict(list)
 for d in laws:lawcase[d['case']].append(d)
 pairing=[]
 for key,group in sorted(groups.items()):
  group=sorted(group,key=lambda r:r['arm']);ok=len(group)==3
  base=group[0];base_law=lawcase[base['case']]
  for other in group[1:]:
   for x,y in zip(base_law,lawcase[other['case']]):
    for f in ['mode','p_true','count','g','p_nominal','L','U','u']:assert x[f]==y[f],(key,f)
   for x,y in zip(bycase[base['case']],bycase[other['case']]):
    for f in ['step','obstacle_x','obstacle_y','obstacle_vx','obstacle_vy']:assert x[f]==y[f],(key,f)
  pairing.append(dict(geometry=key[0],delta=key[1],seed=key[2],arms_present=len(group),pairing='passed' if ok else 'incomplete',all_rollouts_ok=int(ok and all(r['status']=='OK' for r in group))))
 summary=[]
 cells=defaultdict(list)
 for r in rollouts:cells[r['geometry'],float(r['delta']),r['arm']].append(r)
 for (geo,delta,arm),runs in sorted(cells.items()):
  cases={r['case'] for r in runs};ds=[r for r in decisions if r['case'] in cases];evaluated=[r for r in ds if int(r['n_validation'])>0];valid=[r for r in runs if r['status']=='OK']
  rates=[float(r['violation_estimate']) for r in evaluated];times=[float(r['solve_ms']) for r in ds]
  per_rollout=[]
  for run in runs:
   vals=[float(r['violation_estimate']) for r in evaluated if r['case']==run['case']]
   if vals:per_rollout.append(statistics.mean(vals))
  transferred=[r for r in evaluated if r['conditional_transfer']=='1']
  summary.append(dict(geometry=geo,delta=delta,arm=arm,attempted=len(runs),finished_with_summary=len(valid),execution_errors=len(runs)-len(valid),completed=sum(int(r['completed']) for r in valid),completion_fraction_attempted=sum(int(r['completed']) for r in valid)/len(runs),collisions_observed=sum(int(r['collision']) for r in valid),collision_outcome_unavailable=len(runs)-len(valid),collision_fraction_known=sum(int(r['collision']) for r in valid)/len(valid) if valid else '',mean_completion_seconds=statistics.mean(float(r['steps'])*.1 for r in valid if r['completed']=='1') if any(r['completed']=='1' for r in valid) else '',observed_decisions=len(ds),refused_decisions=sum(r['success']!='1' for r in ds),evaluated_decisions=len(evaluated),validation_trajectories=sum(int(r['n_validation']) for r in evaluated),mean_true_violation_decision_weighted=statistics.mean(rates) if rates else '',mean_true_violation_rollout_weighted=statistics.mean(per_rollout) if per_rollout else '',decisions_estimate_exceeds_epsilon=sum(r['estimate_exceeds_epsilon']=='1' for r in evaluated),decisions_CP95_lower_exceeds_epsilon=sum(r['lower_exceeds_epsilon']=='1' for r in evaluated),decisions_CP95_upper_at_most_epsilon=sum(r['upper_below_epsilon']=='1' for r in evaluated),transferred_evaluated_decisions=len(transferred),transferred_estimate_exceeds_epsilon=sum(r['estimate_exceeds_epsilon']=='1' for r in transferred),covered_transferred_lower_exceeds_epsilon=sum(r['lower_exceeds_epsilon']=='1' and r['cp_covered']=='1' for r in transferred),median_solve_ms_concurrent=statistics.median(times) if times else '',p95_solve_ms_concurrent=percentile(times,.95) if times else '',max_S=max((int(r['S']) for r in ds),default=''),max_zeta=max((float(r['zeta']) for r in ds),default='')))
 for name,data in [('shift_rollouts.csv',rollouts),('heldout_decisions.csv',decisions),('shift_laws.csv',laws),('shift_pairing.csv',pairing),('table2_shift_summary.csv',summary)]:write(root/name,data)
 protocol={
 'formulation':'Updated direct Clopper-Pearson polytope envelope; controller implementation commit 61aaa7de.',
 'family1':'M=2,3,5; g=10,25,50,100,250,1000; 100 independently seeded calibrations per cell; four laws per identical instance.',
 'family1_truth':'Fixed normalized weights proportional to 0.65**mode_index; raw counts and true probabilities in certificate_modes.csv.',
 'family1_geometry':'Production affine mode library, dt=.1,N=20; obstacle (3,1.5,-.6,0); frozen ego x=.15*k,y=0; 3 discs; length4; disc radius.6; obstacle radius.35; margin0.',
 'family1_mode_order':'constant_velocity,turn_left,turn_right,decelerating,accelerating,stop; M7/M8 runtime-only libraries append constant-velocity modes with affine y drifts .02/.04 per step.',
 'water_min_zeta_solver':'Independent SciPy HiGHS LP maximizes t=1/zeta with q_j>=t*u_j, source marginal p_hat and transport cost<=rho. Production full-risk arm uses existing C++ LP.',
 'family1_S':'All four laws sized for epsilon/zeta using production nonconvex SH formula, support cap6+removal2, beta_SH=.01, epsilon=.05. This differs from nominal standard-S arm in family2.',
 'runtime_scope':'cp_radius_total_ms includes CP endpoints, direct envelope, vertex enumeration and all transport solves; individual per-vertex OT times were not instrumented. envelope_formula_ms is a separate direct-formula timing. CP time is not double-counted in construction_total_ms.',
 'runtime_fairness':'C++ full-law allocation and Python/HiGHS Wasserstein-only timings use different implementations; compare mathematical tightness separately from runtime. Certificate/radius sweeps ran serially.',
 'family2':f"Closed-loop pilot: 3 geometries x 4 shifts x 3 arms x {len(manifest['seeds'])} paired seeds; up to 200 steps; N=20; dt=.1; g=25 frozen calibration labels; 1000 fresh held-out trajectories per executable returned decision.",
 'family2_geometries':'Straight highway, S-curve, roundabout (radius4); 25m road setting; one obstacle with three production modes in each geometry. Frozen initial path risk score selects a fixed dangerous shift target before calibration or control.',
 'family2_truth':'p_true(delta)=(1-delta)*(.8,.15,.05)+delta*e_risk, delta=0,.2,.4,.6; mode order constant_velocity,turn_left,turn_right. Calibration is drawn from this same stationary truth, so mismatch is finite-sample estimation error, not intentionally stale data.',
 'family2_arms':'Nominal: q=p_hat and standard S; envelope: q=u/sum(u) with transfer-sized S; full: production constrained risk-aware law and automatic transfer sizing. No oracle arm.',
 'family2_budget':'epsilon=.05,beta_DRO=.05,beta_SH=.01,support_cap6,removal_budget2. Envelope support law is set through existing custom mode weights; conditional transfer is checked from known factor and actual SH acceptance.',
 'heldout_event':'Joint geometric collision-avoidance violation of the frozen returned ego plan: any predicted step k=1..N and any ego disc has distance strictly below combined radius. This evaluates horizon risk under the declared held-mode Gaussian mixture, not realized one-step collision rate. It does not claim equivalence to every recovery-dependent linearized solver halfspace.',
 'heldout_CI':'Per-decision exact two-sided 95% Clopper-Pearson intervals. They are pointwise, not simultaneous across all decisions. Refused/nonexecutable plans have n_validation=0 and blank violation metrics.',
 'independence':'Calibration, plant-mode, controller and validation RNG streams are separate. Validation never changes the live controller or plant RNG. Arms share calibration draws and validation seeds; plant-state prefixes are checked.',
 'world_kernel_limit':'Plant modes are redrawn each world step; held-out horizon trajectories hold one mode fixed. Plant process_noise=1 and speed_cap=1e6. The known held-out distribution matches the declared held-mode kernel, but the receding plant future need not do so.',
 'dependent_decisions':'Do not treat decisions within a rollout as independent experimental replicates or pooled successes as one binomial experiment. Summaries report both decision-weighted and rollout-weighted means; primary independent replication unit is the paired seed.',
 'execution_failures':'Every attempted run is retained. Prefix decisions from execution-error runs remain labeled with rollout_status; missing final collision/completion outcomes are not zero-filled. No unsuccessful seed is replaced.',
 'timing_concurrency':f"Closed-loop runs use {manifest['jobs']} concurrent processes; solve_ms excludes independent validation time. These wall times do not establish isolated or hard real-time performance.",
 'family3':'V=1..6; 100 independent seed sets; each obstacle M3,g100; beta_v=.05/6 fixed across V; prefix pairing across V. Joint beta used=V*.05/6<=.05. No controller solves are performed for this offline scaling family.',
 'sample_bound_limit':'S_bound_satisfied must equal1 to interpret required_S as a verified inverse of the existing implementation bound; all packaged construction/scaling rows are checked.',
 'optional_not_run':'Secondary shift toward a different geometric mode, oracle arm, and per-vertex OT microtimings were not included. This is the requested 30-seed pilot option, not a 100-seed final closed-loop study.',
 'status':f"{len(metadata)}/{expected} attempted closed-loop runs collected; all summaries retain observed limitations. No safety/performance superiority is asserted.",
 'shift_runner_sha256':manifest['runner_sha256'],
 }
 write(root/'experiment_protocol.csv',[dict(key=k,value=v) for k,v in protocol.items()])
 if a.partial:print(f'Partial summary: {len(metadata)}/{expected}');return
 names=['experiment_protocol.csv','table1_certificate_summary.csv','certificate_instances.csv','certificate_modes.csv','certificate_ground_costs.csv','radius_runtime.csv','radius_runtime_summary.csv','table2_shift_summary.csv','shift_rollouts.csv','shift_laws.csv','heldout_decisions.csv','shift_pairing.csv','figure1_scaling_summary.csv','scaling_instances.csv','scaling_obstacles.csv']
 for name in ['certificate_instances.csv','scaling_instances.csv']:
  assert all(r['S_bound_satisfied']=='1' for r in read(root/name))
 for r in decisions:
  if int(r['n_validation']):assert 0<=float(r['cp95_lower'])<=float(r['violation_estimate'])<=float(r['cp95_upper'])<=1
 write(root/'csv_manifest.csv',[dict(file=n,rows=len(read(root/n)),sha256=hashlib.sha256((root/n).read_bytes()).hexdigest()) for n in names]);names.append('csv_manifest.csv')
 archive=root.parent/(root.name+'-csvs.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names:z.write(root/n,n)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  assert all('/' not in n and n.endswith('.csv') for n in z.namelist())
 print(json.dumps(dict(archive=str(archive),csv_files=len(names),bytes=archive.stat().st_size,rollouts=len(rollouts),errors=sum(r['status']!='OK' for r in rollouts),decisions=len(decisions),validation_trajectories=sum(int(r['n_validation']) for r in decisions)),indent=2))
if __name__=='__main__':main()
