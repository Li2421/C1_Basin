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
root=Path('diagnostics/gap_flow_scale_20261007');m=json.loads(Path('diagnostics/gap_flow_v1/n10_wide_recovery/dataset/manifest.json').read_text());c=Config(**m['scenario_config']);base=BottleneckEnv(replace(c,seed=314160,split='dev'));p=base.positions.copy();g=base.goals.copy();p[1::2]=g[1::2];g[1::2]=base.positions[1::2];order=np.argsort(-p[:,0]);thresholds=(.8,1.2,2.)
requests=[dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':g.tolist(),'episode_seed':314160,'split':'dev'}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'nonopposing_one_way_pipeline_teacher','threshold':x,'horizon':c.max_steps}),seed_keys=['314160']) for x in thresholds]
plan=root/'pipelined_n10_plan.json';plan.write_text(json.dumps({'requests':requests},indent=2));cache=preflight(plan);(root/'pipelined_n10_preflight.json').write_text(json.dumps(cache,indent=2));assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==len(requests)
f=CertifiedHardSafetyFilter(HardProjectionConfig());rows=[]
for threshold in thresholds:
 e=BottleneckEnv(replace(c,seed=314160,split='dev'));e.reset(p,g);released=1;t=time.monotonic();maxmoving=0
 for step in range(c.max_steps):
  if released<len(order) and e.positions[order[released-1],0]>threshold:released+=1
  u=teacher_reference(e);u[order[released:]]=0
  maxmoving=max(maxmoving,int(np.sum(np.linalg.norm(u,axis=1)>.1)))
  r=f(e.snapshot(),u);_,done,info=e.step(r.velocity,diagnose_stalls=False)
  if done:break
 row={'threshold':threshold,'success':e.termination=='success','termination':e.termination,'steps':e.step_count,'released':released,'maxmoving':maxmoving,'remaining':np.linalg.norm(e.positions-g,axis=1).tolist(),'seconds':time.monotonic()-t};rows.append(row);print(json.dumps(row),flush=True)
(root/'pipelined_n10_result.json').write_text(json.dumps(rows,indent=2))
