#!/usr/bin/env python3
import argparse,csv,json,subprocess
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--bank',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 root=Path(args.output);root.mkdir(parents=True,exist_ok=True);bank=Path(args.bank);jobs=json.loads((bank/'manifest.json').read_text())['jobs'];exports=[]
 for job in jobs:
  folder=root/f"context_{job['context']:02}";folder.mkdir(exist_ok=True);snapshot='-'
  if job['source']=='encounter':
   traces=list(csv.DictReader(next((bank/f"context_{job['context']:02}"/'encounter').glob('*/trace.csv')).open()));features=list(csv.DictReader((bank/f"context_{job['context']:02}"/'contexts.csv').open()))
   snapshot=folder/'stored_snapshots.csv'
   with snapshot.open('w') as out:
    writer=csv.writer(out);writer.writerow(['step','ego_x','ego_y','theta','v','s','id','x','y','vx','vy','vref'])
    for feature in features:
     step=feature['step'];ego=next(r for r in traces if r['step']==step and r['actor']=='ego');obs=next(r for r in traces if r['step']==step and r['actor']=='obstacle')
     writer.writerow([step,ego['x'],ego['y'],ego['theta'],ego['v'],ego['path_progress'],obs['obstacle_id'],obs['x'],obs['y'],obs['vx'],obs['vy'],1.5])
  command=[args.runner,'--transfer-export',job['config'],str(folder),str(job['seed']),'0',str(snapshot)]
  with (folder/'export.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,(job['context'],(folder/'export.log').read_text())
  features=list(csv.DictReader((bank/f"context_{job['context']:02}"/'contexts.csv').open()))
  for feature in features:
   data=json.loads((folder/f"step_{feature['step']}.json").read_text())
   for actual,key in zip([*data['ego'][:2],data['ego'][3]],['ego_x','ego_y','ego_speed']):assert abs(actual-float(feature[key]))<1e-12
   import numpy as np
   distance=min(np.linalg.norm(np.array(obs['state'][:2])-np.array(data['ego'][:2])) for obs in data['obstacles'])
   assert abs(distance-float(feature['observed_distance']))<1e-10
   entropy=sum(-m['p']*__import__('math').log(m['p']) for o in data['obstacles'] for m in o['modes'])
   assert abs(entropy-float(feature['mode_entropy']))<1e-10
   exports.append(dict(context=f"{job['context']:02}_{feature['step']}",source=job['source'],family=job['family'],file=str(folder/f"step_{feature['step']}.json")))
 (root/'bank.json').write_text(json.dumps(exports,indent=2));print('Exported',len(exports),'contexts; ego state, obstacle distance and nominal entropy match stored bank. No controller solve or plant step executed.')
if __name__=='__main__':main()
