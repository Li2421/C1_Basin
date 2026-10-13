import json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.collect_competence import teacher_reference
from bottleneck_family.scenario import Config
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity,uid
from shared_rollout_db.src.planner import preflight
root=Path('diagnostics/gap_flow_scale_20261007')
m=json.loads(Path('diagnostics/gap_flow_v1/n10_wide_recovery/dataset/manifest.json').read_text());c=Config(**m['scenario_config'])
base=BottleneckEnv(replace(c,seed=314159,split='dev'));p=base.positions.copy();g=base.goals.copy();p[1::2]=g[1::2];g[1::2]=base.positions[1::2]
orders={'index':np.arange(10),'near_gate':np.argsort(-p[:,0]),'far_gate':np.argsort(p[:,0]),'near_goal':np.argsort(np.linalg.norm(g-p,axis=1))}
requests=[]
for name,order in orders.items():
 requests.append(dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':g.tolist(),'episode_seed':314159,'split':'dev'}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'nonopposing_one_way_sequential_teacher','order':order.tolist(),'horizon':c.max_steps}),seed_keys=['314159']))
plan=root/'queue_n10_plan.json';plan.write_text(json.dumps({'requests':requests},indent=2));cache=preflight(plan);(root/'queue_n10_preflight.json').write_text(json.dumps(cache,indent=2));assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==len(requests)
f=CertifiedHardSafetyFilter(HardProjectionConfig());rows=[]
for name,order in orders.items():
 e=BottleneckEnv(replace(c,seed=314159,split='dev'));e.reset(p,g);turn=0;t=time.monotonic();corr=[]
 for step in range(c.max_steps):
  if turn<len(order) and np.linalg.norm(e.positions[order[turn]]-g[order[turn]])<=c.goal_tolerance:turn+=1
  u=teacher_reference(e);active=int(order[min(turn,len(order)-1)]);u[np.arange(10)!=active]=0
  r=f(e.snapshot(),u);corr.append(float(np.linalg.norm(r.velocity-u)));_,done,info=e.step(r.velocity,diagnose_stalls=False)
  if done:break
 row={'order':name,'success':e.termination=='success','termination':e.termination,'steps':e.step_count,'turn':turn,'mean_correction':float(np.mean(corr)),'final_distances':np.linalg.norm(e.positions-g,axis=1).tolist(),'seconds':time.monotonic()-t};rows.append(row);print(json.dumps(row),flush=True)
(root/'queue_n10_result.json').write_text(json.dumps(rows,indent=2))
