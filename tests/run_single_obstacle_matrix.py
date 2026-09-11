#!/usr/bin/env python3
"""Run the 32 single-obstacle fixtures twice and retain honest outcome artifacts."""
import argparse
import concurrent.futures
import csv
import html
import json
import math
from pathlib import Path
import subprocess
import yaml
import sys
from test_base_scenario import gif_frames, trace

MODES = {'constant_velocity', 'accelerating', 'turn_left_sharp', 'turn_right_sharp'}

def validate(bundle): 
    errors = []
    with (bundle / 'rollout.csv').open() as stream:
        result = next(csv.DictReader(stream))
    rows = trace(bundle / 'trace.csv')
    ego = [r for r in rows if r['actor'] == 'ego']
    obstacles = [r for r in rows if r['actor'] == 'obstacle']
    steps = int(result['total_steps'])
    if len(ego) != steps + 1 or len(obstacles) != steps + 1:
        errors.append('incorrect actor/frame count')
    if result['effective_available_mode_counts'] != '4':
        errors.append('obstacle does not have four modes')
    # Check actual plant propagation against the reported mode. No noise is
    # injected into the plant in these fixtures; predictor noise remains enabled.
    for previous, current in zip(obstacles, obstacles[1:]):
        mode = current['mode']
        if mode not in MODES:
            errors.append('unexpected mode'); continue
        dt = float(current['time_s']) - float(previous['time_s'])
        x, y, vx, vy = (float(previous[k]) for k in ('x','y','vx','vy'))
        if mode == 'accelerating':
            px, py = x + dt*vx, y + dt*vy
            vx, vy = 1.08*vx, 1.08*vy
        elif mode.startswith('turn_'):
            angle = (.65 if mode == 'turn_left_sharp' else -.65)*dt
            vx, vy = math.cos(angle)*vx-math.sin(angle)*vy, math.sin(angle)*vx+math.cos(angle)*vy
            px, py = x + dt*vx, y + dt*vy
        else:
            px, py = x + dt*vx, y + dt*vy
        speed = math.hypot(vx,vy)
        if speed > 2: vx,vy = vx*2/speed,vy*2/speed
        if any(abs(float(current[k])-v) > 1e-9 for k,v in zip(('x','y','vx','vy'),(px,py,vx,vy))):
            errors.append('plant trace differs from reported mode dynamics'); break
    if 'path_following' in bundle.name:
        with (bundle/'scene.csv').open() as stream:
            route=[(float(r['x']),float(r['y'])) for r in csv.DictReader(stream) if r['kind']=='route']
        def segment_distance(x,y,a,b):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=dx*dx+dy*dy
            t=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/length)) if length else 0
            return math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)
        maximum=max(min(segment_distance(float(r['x']),float(r['y']),a,b)
                        for a,b in zip(route,route[1:])) for r in obstacles)
        result['obstacle_max_reference_distance']=maximum
        if maximum > 0.5:
            errors.append('path-following obstacle departed reference by over 0.5 m')
    if gif_frames(bundle / 'rollout.gif') != steps+1:
        errors.append('GIF frame count mismatch')
    for filename in ('rollout.rviz','scene.csv','geometry.csv','resolved_config.yaml','sampled_scenarios.csv'):
        if not (bundle/filename).is_file(): errors.append('missing '+filename)
    if result['collision'] != '0': errors.append('collision recorded')
    with (bundle / 'resolved_config.yaml').open() as stream:
        resolved_config = yaml.safe_load(stream)

    required_progress = float(
        resolved_config['path_completion_fraction']
    )

    if float(result['total_progress']) < required_progress:
        errors.append('path not completed')
        
    return result, rows, errors

def run_case(config, runner, output, timeout, linearized_constraints=True):
    name = config.stem
    directory = output/name
    directory.mkdir(parents=True, exist_ok=True)
    report = {'name':name, 'config':str(config), 'status':'ERROR'}
    try:
        results=[]
        for suffix in ('','_repeat'):
            label=name+suffix
            command=[str(runner),'--config',str(config),'--seed','77','--output',str(directory),'--label',label]
            command.append('--linearized-constraints' if linearized_constraints else '--no-linearized-constraints')
            with (directory/(label+'.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
            results.append(validate(directory/label))
        first,second=results
        errors=first[2]+second[2]
        reproducible=first[1]==second[1]
        if not reproducible: errors.append('same-seed numerical traces differ')
        if (directory/name/'sampled_scenarios.csv').read_bytes() != (directory/(name+'_repeat')/'sampled_scenarios.csv').read_bytes():
            errors.append('same-seed forecasts differ')
        report.update({k:first[0][k] for k in ('total_steps','total_progress','collision','min_clearance','S','certified_decisions','safe_horizon_decisions','certificate_rate')})
        if 'obstacle_max_reference_distance' in first[0]:
            report['obstacle_max_reference_distance']=first[0]['obstacle_max_reference_distance']
        report.update(status='FAIL' if errors else 'PASS',reproducible=reproducible,errors=sorted(set(errors)))
    except Exception as error:
        report['errors']=[str(error)]
    (directory/'test_result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def gallery(output, reports, total):
    existing={}
    for result_file in output.glob('*/test_result.json'):
        try:
            item=json.loads(result_file.read_text());existing[item['name']]=item
        except json.JSONDecodeError:
            continue
    existing.update({r['name']:r for r in reports})
    reports=sorted(existing.values(),key=lambda r:r['name'])
    if (output/'matrix.json').exists():
        total=len(json.loads((output/'matrix.json').read_text()))
    (output/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
    parts=['<!doctype html><html><meta charset="utf-8"><title>Single obstacle matrix</title>',
           '<style>body{font:16px system-ui;margin:2rem;background:#f2f5f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}article{background:white;padding:1rem}img{width:100%}</style>',
           f'<h1>Single obstacle matrix — {len(reports)}/{total} results</h1>',
           '<p>SH-MPCC, with/without DRO; four paths; four obstacle policies. Seed 77, repeated twice. Orange lines show eight actual sampled forecasts. When enabled, white segments show retained disc-space collision boundaries; ticks point into the feasible side. Plant noise is zero; prediction noise is enabled.</p>',
           '<p>FAIL includes collision or incomplete path; inspect retained trajectories. Certificate counts are solver reports, not a guarantee for the feedback-driven plant policy.</p><main>']
    for r in reports:
        n=r['name'];link=f'{n}/{n}'
        parts.append(f'<article><h2>{html.escape(n)} — {r["status"]}</h2>')
        if (output/link/'rollout.gif').exists():
            parts.append(f'<img loading="lazy" src="{link}/rollout.gif"><p><a href="{link}/rollout.gif">GIF</a> · <a href="{link}/rollout.svg">Final path</a> · <a href="{link}/rollout.rviz">RViz</a> · <a href="{link}/resolved_config.yaml">Config</a></p>')
        parts.append(f'<p>Progress: {r.get("total_progress","—")}; collision: {r.get("collision","—")}; certified: {r.get("certified_decisions","—")}/{r.get("safe_horizon_decisions","—")}.</p><p>{html.escape("; ".join(r.get("errors",[])))}</p><a href="{n}/test_result.json">Test result</a></article>')
    (output/'index.html').write_text('\n'.join(parts)+ '</main></html>')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--configs',type=Path,default=Path(__file__).resolve().parents[1]/'configs/single_obstacle_tests')
    parser.add_argument('--case')
    parser.add_argument('--prefix', help='Run case names beginning with this prefix')
    parser.add_argument('--suffix', help='Run case names ending with this suffix')
    parser.add_argument('--resume',action='store_true',help='Retain finished PASS/FAIL cases and retry errors')
    parser.add_argument('--linearized-constraints', action=argparse.BooleanOptionalAction, default=True,
                        help='Draw retained collision half-spaces (default: enabled)')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args()
    configs=sorted(args.configs.resolve().glob('*.yaml'))
    if args.case: configs=[c for c in configs if c.stem==args.case]
    if args.prefix: configs=[c for c in configs if c.stem.startswith(args.prefix)]
    if args.suffix: configs=[c for c in configs if c.stem.endswith(args.suffix)]
    if not configs: parser.error('no matching cases')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    reports=[]
    pending=[]
    for config in configs:
        result_file=output/config.stem/'test_result.json'
        if args.resume and result_file.exists():
            prior=json.loads(result_file.read_text())
            if prior['status'] in ('PASS','FAIL'):
                reports.append(prior);continue
        pending.append(config)
    gallery(output,reports,len(configs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run_case,c,args.runner.resolve(),output,args.timeout,args.linearized_constraints) for c in pending]
        for future in concurrent.futures.as_completed(futures):
            report=future.result();reports.append(report);gallery(output,reports,len(configs))
            print(f'{len(reports)}/{len(configs)} {report["name"]}: {report["status"]} {report.get("errors",[])}',flush=True)
    sys.exit(any(r['status']!='PASS' for r in reports))
