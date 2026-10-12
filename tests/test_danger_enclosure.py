#!/usr/bin/env python3
import importlib.util,math
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('transfer_screen',Path('tools/run_oracle_transfer.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(44109);checked=0
for heading in [0,.8,2.,-2.9]:
 c=dict(ego=[0,0,heading,1.5],vmax=4.,amin=-5.,amax=3.,omega=.8,dt=.2,H=20)
 boxes=module.reachable_boxes(c)
 for _ in range(250):
  x=y=0.;v=1.5;theta=heading
  for k,box in enumerate(boxes):
   # Draw only controls preserving hard endpoint speed bounds.
   a=rng.uniform(max(-5.,-v/.2),min(3.,(4.-v)/.2));w=rng.uniform(-.8,.8);dt=.2
   for fraction,weight in [(0,1/6),(.5,1/3),(.5,1/3),(1,1/6)]:
    speed=v+a*fraction*dt;angle=theta+w*fraction*dt
    x+=dt*weight*speed*math.cos(angle);y+=dt*weight*speed*math.sin(angle)
   v+=a*dt;theta+=w*dt
   assert box[0]<=x<=box[2] and box[1]<=y<=box[3],(heading,k,x,y,box)
   checked+=1
# Conditional deterministic danger tests, both always-outside and always-inside.
model=dict(A=np.eye(4).tolist(),G=np.zeros((4,2)).tolist(),b=[0,0,0,0])
c=dict(ego=[0,0,0,0],vmax=0.,amin=0.,amax=0.,omega=0.,dt=.2,H=3,radius=.625,length=0)
assert module.danger(c,dict(state=[10,0,0,0]),model,100,1)[0]==0
assert module.danger(c,dict(state=[.1,0,0,0]),model,100,1)[0]==100
print('PASS:',checked,'RK4 stage-enclosure comparisons and two deterministic danger fixtures; empirical tests do not prove enclosure rounding.')
