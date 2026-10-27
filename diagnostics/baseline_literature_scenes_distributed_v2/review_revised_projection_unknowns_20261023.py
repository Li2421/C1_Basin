"""Static feasibility review of numerical unknowns; never replay or relabel results."""
from experiment import *
from atomic_write import write
from projection_adapter import constraints
from distributed_safety import LocalSafety
from shared_control.hard_projection import HardProjectionConfig
import numpy as np,collections
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
review=[];hp=HardProjectionConfig();live=read(HERE/'LIVE_REVISED_FIXED_SUCCESS.json')
for entry in live['unknown_examples']:
 raw=read(entry['raw']);c=raw['case'];cfg=c['scenario_payload']['config']
 with np.load(raw['trace_path']) as z:
  p=z['positions'][-1];snap=dict(positions=p,walls=np.zeros((0,2,2)) if c['state']['scene']=='lit_crowded_random' else __import__('states').walls(c['state']['scene']),config=cfg);problem=constraints(snap,hp)
  filters=[LocalSafety(snap,i,hp,problem=problem) for i in range(len(p))]
  margins=[float((-f.lower).min()) for f in filters]
  distances=np.linalg.norm(p[:,None]-p[None,:],axis=-1);distances[np.eye(len(p),dtype=bool)]=np.inf
  ref=entry['error'];review.append(dict(raw=entry['raw'],raw_sha256=sha(entry['raw']),scene=c['state']['scene'],controller=c['controller_payload']['nominal']['controller'],method=c['method'],N=c['state']['N'],error_step=ref['step'],saved_steps=len(z['actions']),failure_type=ref['type'],max_abs_position=float(np.abs(p).max()),positions_finite=bool(np.isfinite(p).all()),minimum_pair_center_distance=float(distances.min()),d_safe=2*cfg['agent_radius']+cfg['agent_collision_margin']+hp.separation_buffer,per_robot_zero_input_constraint_margins=margins,zero_is_feasible_for_all_local_sets=bool(min(margins)>=0),last_executed_actions_finite=bool(np.isfinite(z['actions']).all()),last_executed_max_speed=float(np.linalg.norm(z['actions'],axis=-1).max()),unknown_retained_unchanged=True))
write(HERE/'REVISED_PROJECTION_UNKNOWN_STATIC_REVIEW_20261023.json',dict(utc=utc(),live_landed=live['landed'],live_unknown=live['unknown'],sample_rule='First20 numerical unknown examples in the live monitor',reviews=review,scientific_rollouts=0,relabelled=0,solver_unchanged=True))
print(json.dumps(dict(live_landed=live['landed'],live_unknown=live['unknown'],reviewed=len(review),all_zero_feasible=sum(x['zero_is_feasible_for_all_local_sets'] for x in review),scenes=dict(collections.Counter(x['scene'] for x in review)),sample=review[:2])),flush=True)
