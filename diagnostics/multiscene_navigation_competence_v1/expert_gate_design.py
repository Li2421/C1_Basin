from common import *
from geometry import GeneratedGeometry,task
from bottleneck_family.scenario import Config
from design_generated import identities
from registry import register
import math

def prepare(stage='expert_gate_v2'):
 catalog=read(HERE/'PILOT_GEOMETRY_FROZEN.json');cases=[];rows=[];rejected=[]
 for f,ss in catalog.items():
  train=[s for s in ss if s['role'] in ['train','train_new']]
  for m,s in enumerate(train):
   geo=GeneratedGeometry(Config(**s['config']),s['rectangles'])
   for j in range(32):
    base=202660000000+'ABCDEFG'.index(f)*100000000+m*100000+j*100
    for k in range(100):
     try:p,g,paths,_=task(geo,f,base+k);break
     except ValueError as e:rejected.append(dict(family=f,map=m,episode=j,seed=base+k,reason=str(e)))
    else:raise RuntimeError('task draw failed')
    i=j%4;H=int(math.ceil(2.5*np.linalg.norm(np.diff(paths[i],axis=0),axis=1).sum()/.5/.05/100)*100)
    single=dict(s,role='expert_gate',map_index=m,index=j,task_seed=base+k,alias=f'round40_expert_{f}_{m}_{j}',positions=p[i:i+1].tolist(),goals=g[i:i+1].tolist(),velocities=np.zeros((1,2)).tolist(),paths=[paths[i].tolist()],H=H,config=dict(s['config'],num_agents=1,max_steps=H))
    c,r=identities(single,'navigation_field','navigation');cases.append(c);rows+=r
 write(HERE/'EXPERT_TASK_REJECTIONS.json',rejected);register(stage,cases,rows)
 seal('EXPERT_GATE_IMPLEMENTATION_V2.receipt.json',[HERE/n for n in ['navigation_field.py','observation39.py','geometry.py','single_agent.py','expert_gate_design.py','expert_worker.py','NAVIGATION_UNIT_TESTS.json']],before_expert_labels=True)
if __name__=='__main__':prepare()
