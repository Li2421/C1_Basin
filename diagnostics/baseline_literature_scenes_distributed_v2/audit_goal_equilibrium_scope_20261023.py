"""Static goal-configuration checks; do not change physics, tasks or correction."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from projection_adapter import constraints
from distributed_safety import LocalSafety
from correction_revision_20261021 import contact_mask,contact_radius,safe_distance
from states import LiteratureConfig,walls
from shared_control.hard_projection import HardProjectionConfig
import numpy as np
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
rows=[];hp=HardProjectionConfig()
for scene in NAMES:
 c=read(HERE/f'distributed_baseline_{scene}_CASES.json')[0];cfg=LiteratureConfig(**c['scenario_payload']['config']);p=np.asarray(c['state']['goals'],float);snapshot=dict(positions=p,walls=walls(scene),config=c['scenario_payload']['config']);problem=constraints(snapshot,hp)
 A,b,kinds=problem;mask=contact_mask(p,p,cfg.goal_tolerance,contact_radius(cfg));assert not mask.any()
 dd=np.linalg.norm(p[:,None]-p[None,:],axis=-1);dd[np.eye(len(p),dtype=bool)]=np.inf
 margins=[float((-LocalSafety(snapshot,i,hp,problem=problem).lower).min()) for i in range(len(p))]
 rows.append(dict(scene=scene,N=len(p),all_arrived_contact_mask=mask.tolist(),relative_component_gate_closed=True,minimum_goal_center_separation=float(dd.min()),d_safe=safe_distance(cfg),Rc=contact_radius(cfg),joint_zero_input_minimum_linear_margin=float((-b).min()),distributed_zero_input_minimum_linear_margins=margins,strict_zero_feasible=bool(np.min(-b)>=0 and min(margins)>=0),source_case_sha256=sha(HERE/f'distributed_baseline_{scene}_CASES.json')))
write(HERE/'GOAL_EQUILIBRIUM_SCOPE_STATIC_20261023.json',dict(utc=utc(),scope='First saved baseline state per scene,positions set analytically to its goals; constraints only, no actor or physics step',rows=rows,interpretation='All-arrived Brel=0 is guaranteed. Exact-goal full closed-loop equilibrium additionally requires zero velocity to satisfy the original safety constraints; arrival-region success remains the original unchanged criterion.',no_radius_or_tolerance_tuning=True,new_rollouts=0))
seal('GOAL_EQUILIBRIUM_SCOPE_STATIC_20261023.receipt.json',[Path(__file__),HERE/'GOAL_EQUILIBRIUM_SCOPE_STATIC_20261023.json',HERE/'CORRECTION_REVISION.receipt.json'],static_only=True,original_semantics_unchanged=True)
print(json.dumps(rows),flush=True)
