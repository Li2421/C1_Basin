import json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from bottleneck_family.collect_competence import rollout
from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.scenario import Config
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity,uid
from shared_rollout_db.src.planner import preflight
root=Path('diagnostics/gap_flow_scale_20261007')
m=json.loads(Path('diagnostics/gap_flow_v1/n10_wide_recovery/dataset/manifest.json').read_text())
c=Config(**m['scenario_config']);e=BottleneckEnv(replace(c,seed=314159,split='dev'))
p=e.positions.copy();g=e.goals.copy();p[1::2]=g[1::2];g[1::2]=e.positions[1::2]
req=dict(state_uid=uid('state',{'physical_fingerprint':c.physical_fingerprint,'positions':p.tolist(),'goals':g.tolist(),'episode_seed':314159,'split':'dev'}),eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=uid('ctl',{'controller':'public_waypoint_v2_accepted_hard_safety','horizon':c.max_steps,'purpose':'n10_nonopposing_teacher_feasibility'}),seed_keys=['314159'])
plan=root/'reference_n10_plan.json';plan.write_text(json.dumps({'requests':[req]},indent=2))
cache=preflight(plan);(root/'reference_n10_preflight.json').write_text(json.dumps(cache,indent=2))
assert cache['summary']['ambiguous']==0 and cache['summary']['genuinely_missing']==1,cache['summary']
t=time.monotonic();r=rollout(replace(c,seed=314159,split='dev'),p,g,CertifiedHardSafetyFilter(HardProjectionConfig()))
result={'success':r['success'],'termination':r['termination'],'steps':len(r['actions']),'elapsed_seconds':time.monotonic()-t,'final_distances':np.linalg.norm(r['states'][-1]-g,axis=1).tolist(),'mean_projection_norm':r['mean_projection_norm'],'wall_active_fraction':r['wall_active_fraction']}
result['final_positions']=r['states'][-1].tolist(); result['goals']=g.tolist()
(root/'reference_n10_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
