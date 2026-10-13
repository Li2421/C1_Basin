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
m=json.loads(Path('diagnostics/gap_flow_v1/n50_wide_recovery/dataset/manifest.json').read_text());c=Config(**m['scenario_config'])
base=BottleneckEnv(replace(c,seed=314159,split='dev'));p=base.positions.copy();g=base.goals.copy();p[1::2]=g[1::2];g[1::2]=base.positions[1::2]
order=np.argsort(-p[:,0]);req=dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':g.tolist(),'episode_seed':314159,'split':'dev'}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'nonopposing_one_way_pipeline_teacher_0p8','horizon':c.max_steps}),seed_keys=['314159'])
plan=root/'pipelined_n50_plan.json';plan.write_text(json.dumps({'requests':[req]},indent=2));cache=preflight(plan);(root/'pipelined_n50_preflight.json').write_text(json.dumps(cache,indent=2));assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==1
f=CertifiedHardSafetyFilter(HardProjectionConfig());e=BottleneckEnv(replace(c,seed=314159,split='dev'));e.reset(p,g);turn=0;t=time.monotonic();corr=[]
for step in range(c.max_steps):
 if turn<len(order)-1 and e.positions[order[turn],0]>0.8:turn+=1
 u=teacher_reference(e);u[order[turn+1:]]=0
 r=f(e.snapshot(),u);corr.append(float(np.linalg.norm(r.velocity-u)));_,done,info=e.step(r.velocity,diagnose_stalls=False)
 if step%2000==0:print(json.dumps({'step':step,'turn':turn,'elapsed':time.monotonic()-t,'maxdist':float(np.max(np.linalg.norm(e.positions-g,axis=1)))}),flush=True)
 if done:break
row={'success':e.termination=='success','termination':e.termination,'steps':e.step_count,'turn':turn,'mean_correction':float(np.mean(corr)),'final_distances':np.linalg.norm(e.positions-g,axis=1).tolist(),'seconds':time.monotonic()-t};print(json.dumps(row),flush=True);(root/'pipelined_n50_result.json').write_text(json.dumps(row,indent=2))
