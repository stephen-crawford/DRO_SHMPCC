#!/usr/bin/env python3
"""Paired closed-loop pilot; retain failures, never replace outcomes selectively."""
import argparse,csv,hashlib,itertools,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=30);p.add_argument('--jobs',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 binary=ROOT/'build-base/paper_shift_experiment'
 manifest=dict(seeds=list(range(500,500+a.seeds)),geometries=['straight','s_curve','roundabout'],shifts=[0,.2,.4,.6],arms=['nominal','envelope_optimal','full_proposed'],jobs=a.jobs,g=25,N=20,n_validation=1000,max_steps=200,epsilon=.05,beta_DRO=.05,beta_SH=.01,runner_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),timeout_seconds=180)
 path=a.output/'shift_manifest.json'
 if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('manifest changed')
 path.write_text(json.dumps(manifest,indent=2)+'\n')
 tasks=list(itertools.product(manifest['geometries'],manifest['shifts'],manifest['seeds'],manifest['arms']))
 def run(task):
  geometry,delta,seed,arm=task;name=f'{geometry}_d{delta:g}_s{seed}_{arm}';out=a.output/'shift_raw'/name;out.mkdir(parents=True,exist_ok=True);meta=out/'run.json'
  if a.resume and meta.exists():return json.loads(meta.read_text())
  command=[str(binary),str(out),geometry,str(delta),str(seed),arm];start=time.monotonic()
  try:
   code=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=manifest['timeout_seconds']).returncode
  except subprocess.TimeoutExpired:code='timeout'
  result=dict(case=name,geometry=geometry,delta=delta,seed=seed,arm=arm,exit_code=code,wall_seconds=time.monotonic()-start)
  meta.write_text(json.dumps(result,indent=2)+'\n');return result
 completed=[]
 with ThreadPoolExecutor(max_workers=a.jobs) as executor:
  futures=[executor.submit(run,t) for t in tasks]
  for future in as_completed(futures):
   r=future.result();completed.append(r)
   if len(completed)%12==0 or r['exit_code']!=0:print(f"{len(completed)}/{len(tasks)} {r['case']} exit={r['exit_code']} elapsed={r['wall_seconds']:.1f}s",flush=True)
 (a.output/'shift_runs.json').write_text(json.dumps(completed,indent=2)+'\n')
 print('Complete',len(completed),'errors',sum(r['exit_code']!=0 for r in completed),flush=True)
if __name__=='__main__':main()
