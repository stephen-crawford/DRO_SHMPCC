#!/usr/bin/env python3
"""Run or independently verify the single-obstacle bundle comparison fixture."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('noise','seed','repeat','c0','extras','cycle')
def key(r):return tuple(r[x] for x in FIELDS)
def read(p):
    with p.open() as f: return list(csv.DictReader(f))

def verify(folder,backend):
    manifest=json.loads((folder/'manifest.json').read_text())
    names=sorted(manifest['history'])
    cp=[list(map(float,line.split())) for line in subprocess.check_output(
        [str(backend),'intervals',str(manifest['beta_cp']),*[str(manifest['history'][m]) for m in names]],text=True).splitlines()]
    upper={m:min(cp[i][1],1-sum(row[0] for j,row in enumerate(cp) if j!=i)) for i,m in enumerate(names)}
    @lru_cache(None)
    def required(eta):
        return int(float(subprocess.check_output([str(backend),'size',str(eta),str(manifest['beta_cert']),
            str(manifest['support_cap']),str(manifest['removal_budget'])],text=True).split()[0]))
    cycles={key(r):r for r in read(folder/'cycles.csv')}
    modes=defaultdict(dict)
    for r in read(folder/'modes.csv'):modes[key(r)][r['mode']]=(int(r['K']),float(r['U']))
    plans={(key(r),int(r['stage'])):(float(r['x']),float(r['y'])) for r in read(folder/'plans.csv')}
    counts=defaultdict(Counter);raw_ids=defaultdict(set);stage_masks={};errors=[]
    for name in ['cycles','modes','plans','draws']:
        hashes=[hashlib.sha256(),hashlib.sha256()]
        with (folder/(name+'.csv')).open() as f:
            for r in csv.DictReader(f):
                rep=int(r['repeat'])
                hashes[rep].update(json.dumps({k:v for k,v in r.items() if k not in ['repeat','solve_seconds']},sort_keys=True).encode())
                if name!='draws':continue
                identity=key(r);stage=int(r['stage']);raw=int(r['raw_id']);group=int(r['bundle_id'])
                maskkey=(identity,raw);bit=1<<stage
                if stage_masks.get(maskkey,0)&bit:errors.append('duplicate raw stage')
                stage_masks[maskkey]=stage_masks.get(maskkey,0)|bit
                if stage==0:
                    raw_ids[identity].add(raw);counts[identity][(group,r['mode'])]+=1
                if cycles[identity]['success']=='1' and stage>0:
                    center=plans[(identity,stage)]
                    if math.dist(center,(float(r['x']),float(r['y'])))<.95:errors.append('accepted raw collision')
        if hashes[0].digest()!=hashes[1].digest():errors.append(name+' repeat mismatch')
    if any(mask!=31 for mask in stage_masks.values()):errors.append('incomplete raw horizon')
    for identity,r in cycles.items():
        if len(raw_ids[identity])!=int(r['raw_draws']):errors.append('raw count mismatch')
        groups={g for g,m in counts[identity]}
        if len(groups)!=int(r['bundles']):errors.append('bundle count mismatch')
        if float(r['c0'])>0:
            km=modes[identity]
            if set(km)!={'cut','safe'}:errors.append('missing mode allocation');continue
            if any(abs(U-upper[m])>1e-14 for m,(K,U) in km.items()):errors.append('CP upper mismatch')
            for group in groups:
                for mode,(K,U) in km.items():
                    if counts[identity][(group,mode)]!=K:errors.append('bundle multiplicity mismatch')
            c=min(K/U for K,U in km.values() if U>0)
            if c<float(r['c0']) or c!=float(r['c_K']):errors.append('amplification mismatch')
            eta=-math.expm1(c*math.log1p(-.05))
            if abs(eta-float(r['eta']))>1e-15:errors.append('threshold mismatch')
            if int(r['bundles'])!=int(r['required']):errors.append('automatic count mismatch')
            if int(r['required'])!=required(eta):errors.append('sample formula mismatch')
        elif int(r['required'])!=required(.05):errors.append('baseline formula mismatch')
        if r['certificate']=='certified' and int(r['support'])>6:errors.append('support cap exceeded')
    cohorts=defaultdict(list)
    for k,r in cycles.items():cohorts[k[:-1]].append(r)
    expected=2*manifest['seeds']*2*5
    if len(cohorts)!=expected:errors.append('missing cohort')
    for rows in cohorts.values():
        rows.sort(key=lambda r:int(r['cycle']))
        if [int(r['cycle']) for r in rows]!=list(range(len(rows))) or (len(rows)!=2 and rows[-1]['success']!='0'):
            errors.append('incomplete cohort')
    result=dict(status='PASS' if not errors else 'FAIL',decisions=len(cycles),accepted=sum(r['success']=='1' for r in cycles.values()),
        violations=dict(Counter(errors)),repeat_exact_except_timing=not any('repeat mismatch' in x for x in errors),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
        assumption='Synthetic fixed history; conditional CP confidence and scenario-support hypotheses not established by this simulation.')
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    summary=[]
    for noise,c0,extras in sorted({(r['noise'],r['c0'],r['extras']) for r in cycles.values()}):
        rows=[r for r in cycles.values() if (r['noise'],r['c0'],r['extras'])==(noise,c0,extras) and r['repeat']=='0']
        summary.append(dict(noise=noise,c0=c0,extras=extras,decisions=len(rows),accepted=sum(r['success']=='1' for r in rows),
            bundles_min=min(int(r['bundles']) for r in rows),bundles_max=max(int(r['bundles']) for r in rows),
            raw_draws_min=min(int(r['raw_draws']) for r in rows),raw_draws_max=max(int(r['raw_draws']) for r in rows),
            mean_retained_facets=sum(int(r['retained_facets']) for r in rows)/len(rows),
            mean_solve_seconds=sum(float(r['solve_seconds']) for r in rows)/len(rows)))
    with (folder/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(result,indent=2))
    return not errors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path,help='new directory for a fresh reproducible comparison')
    group.add_argument('--verify-existing',type=Path,help='check every saved raw candidate, group count and repeated artifact without rerunning MPC')
    p.add_argument('--seeds',type=int,default=3,help='paired seeds starting at 77, each repeated twice')
    p.add_argument('--runner',type=Path,default=ROOT/'build-bundles/adversarial_bundle_experiment')
    p.add_argument('--backend',type=Path,default=ROOT/'build-bundles/certificate_numeric_backend')
    a=p.parse_args()
    if a.verify_existing:return 0 if verify(a.verify_existing,a.backend.resolve()) else 1
    if a.output.exists():p.error('use a new output directory')
    if a.seeds<1:p.error('--seeds must be positive')
    files=[a.runner.resolve(),a.backend.resolve(),Path(__file__),ROOT/'tools/adversarial_bundle_experiment.cpp',ROOT/'src/adversarial_bundles.cpp',
           ROOT/'src/mpc_controller.cpp',ROOT/'src/scenario_sampler.cpp',ROOT/'src/collision_constraints.cpp',ROOT/'include/config.hpp']
    manifest=dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),seeds=a.seeds,
        history={'safe':95,'cut':5},horizon=4,cycles=2,noise=[0,.02],c0=[0,2,4],extras=[0,2],repeats=2,
        support_cap=6,removal_budget=0,beta_cp=.05,beta_cert=.01,status='running',
        sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.mkdir(parents=True)
    def save():(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    with (a.output/'experiment.log').open('w') as log:
        run=subprocess.run([str(a.runner.resolve()),str(a.output.resolve()),str(a.seeds)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    manifest['returncode']=run.returncode
    if run.returncode:manifest['status']='failed';save();return run.returncode
    ok=verify(a.output,a.backend.resolve());manifest['status']='complete' if ok else 'validation_failed';save()
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
