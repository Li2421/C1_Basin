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
root=Path('diagnostics/gap_flow_scale_20261007');m=json.loads(Path('diagnostics/gap_flow_v1/n10_wide_recovery/dataset/manifest.json').read_text());c=Config(**m['scenario_config']);seeds=[314159,314160,314161]
requests=[];cases=[]
for seed in seeds:
 base=BottleneckEnv(replace(c,seed=seed,split='dev'));p=base.positions.copy();g=base.goals.copy();p[1::2]=g[1::2];g[1::2]=base.positions[1::2];cases.append((seed,p,g))
 requests.append(dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':g.tolist(),'episode_seed':seed,'split':'dev'}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'state_only_same_direction_nearest_gate_teacher_v1','horizon':c.max_steps}),seed_keys=[str(seed)]))
plan=root/'state_queue_n10_plan.json';plan.write_text(json.dumps({'requests':requests},indent=2));cache=preflight(plan);(root/'state_queue_n10_preflight.json').write_text(json.dumps(cache,indent=2));assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==len(requests)
f=CertifiedHardSafetyFilter(HardProjectionConfig());rows=[]
for seed,p,g in cases:
 e=BottleneckEnv(replace(c,seed=seed,split='dev'));e.reset(p,g);t=time.monotonic();maximum=0
 for step in range(c.max_steps):
  pending=np.flatnonzero(e.positions[:,0]<=.8)
  active=np.ones(c.num_agents,dtype=bool)
  active[pending]=False
  if len(pending): active[pending[np.argmax(e.positions[pending,0])]]=True
  u=teacher_reference(e);u[~active]=0
  maximum=max(maximum,int(np.sum(np.linalg.norm(u,axis=1)>.1)))
  r=f(e.snapshot(),u);_,done,info=e.step(r.velocity,diagnose_stalls=False)
  if done:break
 row={'seed':seed,'success':e.termination=='success','termination':e.termination,'steps':e.step_count,'maxmoving':maximum,'dist':np.linalg.norm(e.positions-g,axis=1).tolist(),'seconds':time.monotonic()-t};rows.append(row);print(json.dumps(row),flush=True)
(root/'state_queue_n10_result.json').write_text(json.dumps(rows,indent=2))
