"""Render recorded simulation geometry and matched reports; no simulation changes."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dro-paper-matplotlib')
from pathlib import Path
import csv
import json
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'build-base/analysis-matrix'
ENVS = ['straight', 's_curve', 'four_way_intersection', 'roundabout']
NAMES = ['Straight highway', 'S-curve', 'Four-way intersection', 'Roundabout']
STYLES = ['sh_mpcc', 'sh_mpcc_dro', 'sh_mpcc_dro_fallback']
COLORS = ['#667B8B', '#347CC2', '#087F78']
ORANGE, PURPLE, INK = '#D86B32', '#9C5897', '#183144'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
SOURCES = set()


def rows(path):
    SOURCES.add(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def xy(rr):
    return np.array([[float(r['x']), float(r['y'])] for r in rr])


def load_scene(env):
    p = DATA / f'sh_mpcc_dro_{env}_o2_c2_m3/seed_77/repeat_0'
    trace = rows(p / 'trace.csv')
    f = p / 'rollout.svg'
    SOURCES.add(f)
    lines = ET.parse(f).getroot().findall('{http://www.w3.org/2000/svg}polyline')
    parse = lambda e: np.array([[float(v) for v in pair.split(',')] for pair in e.attrib['points'].split()])
    world = xy([r for r in trace if r['actor'] == 'ego'])
    pixels = parse(next(e for e in lines if e.attrib.get('class') == 'ego'))
    assert len(world) == len(pixels)
    # Recover the saved renderer's uniform world-to-screen transform from its
    # recorded ego polyline, then invert it for road and reference geometry.
    design = np.zeros((2*len(world), 3))
    design[0::2, 0], design[1::2, 0] = world[:, 0], -world[:, 1]
    design[0::2, 1], design[1::2, 2] = 1, 1
    scale, ox, oy = np.linalg.lstsq(design, pixels.ravel(), rcond=None)[0]
    residual = np.max(np.abs(design @ [scale, ox, oy] - pixels.ravel()))
    assert residual < 1e-5, residual
    paths = defaultdict(list)
    for e in lines:
        kind = e.attrib.get('class')
        if kind in ['road', 'route']:
            q = parse(e)
            paths[kind].append(np.column_stack(((q[:, 0]-ox)/scale, (oy-q[:, 1])/scale)))
    return p, trace, paths


def roads(ax, paths, width=7):
    # Stroke widths are in world coordinates; overlap depicts the road network.
    for q in paths['road']:
        d = np.gradient(q, axis=0)
        n = np.column_stack((-d[:,1], d[:,0])) / np.linalg.norm(d,axis=1)[:,None] * width/2
        band = np.vstack((q+n, (q-n)[::-1]))
        ax.fill(*band.T, color='#E3E9EB', lw=0, zorder=0)
        ax.plot(*q.T, color='white', lw=.9, ls=(0, (4, 4)), zorder=1)
    for q in paths['route']:
        ax.plot(*q.T, color='#879DA4', lw=1.6, ls='--', zorder=2)
    ax.set_aspect('equal')
    ax.set_facecolor('#F8FAF8')
    for s in ax.spines.values():
        s.set_visible(False)


def snapshot(ax, trace, paths, step, annotate=False):
    roads(ax, paths)
    past = [r for r in trace if r['actor']=='ego' and int(r['step'])<=step]
    ax.plot(*xy(past).T, color=COLORS[1], lw=2.7, zorder=4)
    now = [r for r in trace if int(r['step']) == step]
    for r in now:
        pos = xy([r])[0]
        if r['actor']=='ego':
            heading = float(r['theta'])
            for offset in [-2, 0, 2]:
                center = pos+offset*np.array([np.cos(heading), np.sin(heading)])
                ax.add_patch(Circle(center, .5, facecolor=COLORS[1], alpha=.85, edgecolor='white', lw=.8, zorder=6))
            ax.arrow(*pos, np.cos(heading)*1.3, np.sin(heading)*1.3,
                     width=.05, head_width=.3, color=INK, zorder=7)
            if annotate:
                ax.annotate('Ego: three collision discs', pos, xytext=(0.1, .12),
                            textcoords='axes fraction', fontsize=10,
                            arrowprops={'arrowstyle':'-', 'color':INK}, zorder=8)
        else:
            color = [ORANGE, PURPLE][int(r['obstacle_id'])]
            ax.add_patch(Circle(pos, .35, facecolor=color, edgecolor='white', zorder=7))
            ax.add_patch(Circle(pos, .45, fill=False, edgecolor=color, ls=':', zorder=7))
            if annotate:
                ax.annotate(f"Obstacle {int(r['obstacle_id'])+1}", pos,
                            xytext=(-85, 14), textcoords='offset points', color=color,
                            arrowprops={'arrowstyle':'-', 'color':color}, fontsize=10)
    return now


def main():
    rr = rows(DATA / 'rollouts.csv')
    groups = defaultdict(list)
    for r in rr:
        if r['repeat'] in ['', '0']:
            groups[tuple(r[k] for k in ['environment', 'obstacles', 'classes', 'modes_per_class', 'seed'])].append(r)
    matched = [v for v in groups.values() if len(v)==3 and
               {r['solver_style'] for r in v}==set(STYLES) and all(r['status']=='OK' for r in v)]
    report = []
    for env in ENVS + ['overall']:
        for style in STYLES:
            subset = [r for group in matched for r in group if r['solver_style']==style
                      and (env=='overall' or r['environment']==env)]
            n = len(subset)
            refused = sum(r['termination_reason']=='no_admissible_control' for r in subset)
            report.append(dict(environment=env, controller=style, matched_trials=n,
                               no_admissible_control=refused, rate_percent=100*refused/n))
    with (OUT/'matched_report.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    fig=plt.figure(figsize=(17, 12), facecolor='white')
    fig.text(.045,.955,'What the controller sees — and where it is tested',fontsize=23,weight='bold')
    fig.text(.045,.918,'Recorded simulation scenes • four road layouts • seed-matched controller comparison',fontsize=12,color='#59717D')
    fig.text(.045,.874,'1–4 obstacles     /     1–4 shared-history classes (C ≤ O)     /     1–6 modes     /     seeds 77–86',fontsize=12,weight='bold')
    scenes = {env: load_scene(env) for env in ENVS}
    for i, env in enumerate(ENVS):
        ax=fig.add_axes([.045+i*.239,.622,.205,.202])
        _, trace, paths=scenes[env]
        snapshot(ax,trace,paths,0)
        ax.set_xticks([]);ax.set_yticks([])
        fig.text(.045+i*.239,.838,f'{i+1:02d}  {NAMES[i]}',fontsize=12,weight='bold')
    fig.text(.045,.592,'Same seed and configuration in each map: 2 obstacles, 2 classes, 3 modes. Initial positions shown; dashed line = reference route.',fontsize=10,color='#59717D')
    fig.text(.045,.544,'A recorded planning scene',fontsize=17,weight='bold')
    fig.text(.045,.514,'Roundabout · WDRO · seed 77 · t = 6.0 s',fontsize=11,color='#59717D')
    p, trace, paths=scenes['roundabout']
    ax=fig.add_axes([.045,.135,.31,.354])
    snapshot(ax,trace,paths,60,True)
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.tick_params(labelsize=8)
    ax.set_xlim(-11,13);ax.set_ylim(-5,21)
    # A second view enlarges the actual close encounter and stored predictions.
    zoom=fig.add_axes([.37,.272,.21,.21])
    snapshot(zoom,trace,paths,60)
    samples=rows(p/'sampled_scenarios.csv')
    selected=defaultdict(list)
    for r in samples:
        if r['step']=='60' and r['obstacle_id']=='0':
            selected[r['scenario_id']].append(r)
    for seq in selected.values():
        q=xy(sorted(seq,key=lambda r:int(r['horizon_step'])))
        zoom.plot(*q.T,color=ORANGE,lw=1.4,alpha=.9,zorder=8)
        zoom.scatter(*q[-1],color=ORANGE,s=8,zorder=8)
    zoom.set(xlim=(5.3,10.3),ylim=(3.2,8.2))
    zoom.set_title('Encounter detail · metres',fontsize=11,loc='left')
    zoom.tick_params(labelsize=8)
    ax.add_patch(plt.Rectangle((5.3,3.2),5,5,fill=False,edgecolor=ORANGE,lw=1.2))
    fig.text(.37,.218,'Solid blue: executed ego path\nOrange: 3 stored prediction previews\nDiscs: recorded collision geometry',fontsize=9,linespacing=1.7)
    fig.text(.37,.144,'0.1 s control step • 20-step horizon\nEgo discs: r = 0.50 m; offsets −2, 0, 2 m\nObstacle radius: 0.35 m\nDotted ring: +0.10 m safety margin',fontsize=9,linespacing=1.7)
    fig.text(.63,.544,'No-admissible-control outcomes',fontsize=17,weight='bold')
    fig.text(.63,.514,f'{len(matched):,} matched trials per controller · lower is better',fontsize=11,color='#59717D')
    chart=fig.add_axes([.72,.248,.23,.224])
    for j, style in enumerate(STYLES):
        sub=[r for r in report if r['controller']==style and r['environment']!='overall']
        chart.scatter([r['rate_percent'] for r in sub],np.arange(4)+(j-1)*.19,
                      color=COLORS[j],s=48,zorder=3)
    chart.set_yticks(range(4),[f'{name} (n={next(r["matched_trials"] for r in report if r["environment"]==env)})' for env,name in zip(ENVS,['Straight','S-curve','Intersection','Roundabout'])])
    chart.invert_yaxis();chart.set_xlim(0,40);chart.set_xticks([0,10,20,30,40],['0%','10%','20%','30%','40%'])
    chart.grid(axis='x',alpha=.2);chart.tick_params(length=0,pad=7)
    for s in chart.spines.values():s.set_visible(False)
    for j, (style,name) in enumerate(zip(STYLES,['SH-MPCC','WDRO','Hybrid'])):
        r=next(r for r in report if r['controller']==style and r['environment']=='overall')
        x=.63+j*.115
        fig.text(x,.179,f"{r['rate_percent']:.2f}%",fontsize=24,weight='bold',color=COLORS[j])
        fig.text(x,.151,name,fontsize=11,color=COLORS[j])
        fig.text(x,.128,f"{r['no_admissible_control']} / {r['matched_trials']}",fontsize=9,color='#59717D')
    fig.text(.045,.077,'DESIGN  720 configurations × 10 seeds = 7,200 scheduled trials. Matched analysis retains 2,263 / 2,400 configuration–seed triplets.',fontsize=10,weight='bold')
    fig.text(.045,.049,'137 triplets excluded because at least one controller run was not OK. Maps illustrate one setup; rates aggregate all matched setups. No new simulations.',fontsize=9,color='#59717D')
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'simulation_setup.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    SOURCES.add(DATA/'matrix.json')
    SOURCES.add(p/'geometry.csv');SOURCES.add(p/'resolved_config.yaml')
    (OUT/'provenance.json').write_text(json.dumps({
        'matched_triplets':len(matched), 'excluded_triplets':len(groups)-len(matched),
        'snapshot_step':60, 'snapshot_time_s':6, 'stored_obstacle_0_preview_paths':len(selected),
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(SOURCES)}
    },indent=2)+'\n')
    print(json.dumps(report[-3:],indent=2))
    print(f'Exported PNG, PDF, SVG; {len(selected)} recorded obstacle-0 preview paths in zoom.')


if __name__=='__main__':
    main()
