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
root=Path('diagnostics/gap_flow_scale_20261007');trace=root/'n10_set30k_dev_full/traces/dev_scale_0000_full_LR_perm0.npz';d=np.load(trace);meta=json.loads(str(d['metadata_json'].item()));c=Config(**meta['config']);goals=d['goals'];anchors=[100,500,1000,2000]
requests=[]
for t in anchors:
 p=d['positions'][t]
 requests.append(dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':goals.tolist(),'source_trace':str(trace),'anchor':t}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'state_only_same_direction_nearest_gate_teacher_recovery_v1','horizon':c.max_steps}),seed_keys=[str(meta['seed'])]))
plan=root/'state_queue_recovery_plan.json';plan.write_text(json.dumps({'requests':requests},indent=2));cache=preflight(plan);(root/'state_queue_recovery_preflight.json').write_text(json.dumps(cache,indent=2));assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==len(requests)
f=CertifiedHardSafetyFilter(HardProjectionConfig());rows=[]
for t in anchors:
 p=d['positions'][t];e=BottleneckEnv(replace(c,seed=int(meta['seed']),split='dev'));e.reset(p,goals);start=time.monotonic()
 for step in range(c.max_steps):
  pending=np.flatnonzero(e.positions[:,0]<=.8)
  active=np.ones(c.num_agents,dtype=bool);active[pending]=False
  if len(pending):active[pending[np.argmax(e.positions[pending,0])]]=True
  u=teacher_reference(e);u[~active]=0
  r=f(e.snapshot(),u);_,done,info=e.step(r.velocity,diagnose_stalls=False)
  if done:break
 row={'anchor':t,'success':e.termination=='success','steps':e.step_count,'termination':e.termination,'final_mean_dist':float(np.mean(np.linalg.norm(e.positions-goals,axis=1))),'seconds':time.monotonic()-start};rows.append(row);print(json.dumps(row),flush=True)
(root/'state_queue_recovery_result.json').write_text(json.dumps(rows,indent=2))
