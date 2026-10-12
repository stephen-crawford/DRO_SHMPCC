import csv,json,subprocess
from pathlib import Path
root=Path('results/coreset-kill-test');indices=[i for f in range(4) for i in range(f*10,f*10+5)];jobs=json.load(open('results/oracle-gap-screen-v2/manifest.json'))['jobs'];records=[]
for i in indices:
 job=jobs[i];folder=root/'replays'/f'context_{i:02}';folder.mkdir(parents=True,exist_ok=True)
 command=['/tmp/dro-shmpcc-week1-build/week2_budget_response','--coreset-plan-export',job['config'],str(folder),str(job['seed']),'1','0']
 with (folder/'run.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(i,result.returncode)
 old=list(csv.DictReader(open(f'results/oracle-gap-screen-v2/context_{i:02}/step_0/h_10/screen.csv')))[0]
 new=list(csv.DictReader((folder/'step_0/h_10/screen.csv').open()))[0]
 for k in old:
  if k!='solve_ms':assert old[k]==new[k],(i,k,old[k],new[k])
 records.append(dict(context=f'{i:02}_0',family=i//10,plan=str(folder/'step_0/h_10/plans.csv'),parameters=f'results/oracle-transfer/contexts/context_{i:02}/step_0.json',original_screen=old,command=command))
(root/'contexts.json').write_text(json.dumps(records,indent=2));print('20 original frozen outputs match except timing; complete plans saved.')
