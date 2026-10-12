#!/usr/bin/env python3
"""Reproducible heterogeneous paired budget screen; no model training."""
import argparse, concurrent.futures, hashlib, json, subprocess, time
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    base=yaml.safe_load(Path('results/fixed-normal-tightening/config.yaml').read_text());jobs=[]
    for i in range(42):
        cfg=dict(base);snapshot=i>=40
        family=i//10 if not snapshot else 0
        j=i%10
        cfg.update(road_length=25.,road_width=[5.,7.,9.][j%3],ego_initial_v=[1.,1.5,2.5][j%3],metrics_v_ref=[1.,1.5,2.5][j%3],obstacle_prediction_noise=j%5!=0,num_obstacles=[1,1,2,3][family],num_classes=1,obs_modes=[['constant_velocity'],['constant_velocity','decelerating'],['constant_velocity','turn_left','turn_right'],['constant_velocity','decelerating','turn_left','turn_right']][family],obstacle_place_on_path=True,obstacle_initial_states=[],obs_path_fraction=[.2,.35,.55,.75,.9][j%5],obs_arc_fractions=[.25+.04*(j%5),.55,.8],scenario_tag=f'oracle_{i}')
        cfg['num_modes']=len(cfg['obs_modes']);cfg['environment']='s_curve' if j%2 else 'two_lane_highway';cfg['s_curve_amplitude']=[1.,2.,3.][j%3]
        if family==0 and not snapshot:
            cfg['environment']='two_lane_highway';cfg['obstacle_initial_states']=[[5.+1.5*j,(-1 if j%2 else 1)*(.3+.2*(j%3)),.4 if j%2 else 0.,.15*(j%3)]]
        if snapshot:
            cfg.update(environment='two_lane_highway' if i==40 else 's_curve',num_obstacles=1,num_modes=1,obs_modes=['constant_velocity'],obstacle_prediction_noise=True,ego_initial_v=1.5,metrics_v_ref=1.5,obs_path_fraction=.8,obstacle_initial_states=[],road_width=7.)
        config=root/'configs'/f'context_{i:02}.yaml';config.write_text(''.join(str(k)+': '+json.dumps(v)+'\n' for k,v in cfg.items()))
        folder=root/f'context_{i:02}';folder.mkdir();seed=17000+i
        command=[args.runner,'--oracle-screen',str(config),str(folder),str(seed),str(args.repeats),str(int(snapshot))]
        jobs.append(dict(context=i,family=family,source='encounter' if snapshot else 'synthetic',seed=seed,command=command,config=str(config)))
    def run(job):
        start=time.time()
        with (root/f"context_{job['context']:02}"/'execution.log').open('w') as log:
            result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT)
        job.update(returncode=result.returncode,elapsed_seconds=time.time()-start)
        print(job['context'],job['returncode'],round(job['elapsed_seconds'],2),flush=True);return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,jobs))
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.runner,'tools/week2_budget_response.cpp',__file__,'src/mpc_controller.cpp','include/mpc_controller.hpp']}
    (root/'manifest.json').write_text(json.dumps(dict(jobs=results,source_hashes=hashes,repeats=args.repeats,workers=args.workers),indent=2))
    if any(j['returncode'] for j in results):raise SystemExit(1)
if __name__=='__main__':main()
