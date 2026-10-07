#!/usr/bin/env python3
"""Reproduce every historical Ring B0 collision under legacy and fixed safety."""
from __future__ import annotations
from pathlib import Path
import glob,json,math
import jax
import numpy as np

from diagnostics.orthoflow3_generator_critic_frozen_test_v1.run_frozen_test import FrozenNewRuntime
from ring_exchange.safety import central_obstacle_polygon,polygonal_obstacle_snapshot
from shared_control.hard_projection import barrier_constraints

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def legacy_snapshot(env):
 s=env.snapshot();s['walls']=central_obstacle_polygon(env.config.obstacle_radius,48)
 s['wall_names']=tuple('central_obstacle' for _ in range(48))
 s['config']={**s['config'],'wall_radius':0.,'wall_collision_margin':env.config.collision_margin}
 return s

def replay(runtime,state,future_index,fixed):
 env=runtime.make_env();runtime.reset(env,state);root=runtime._root(state,future_index,'hard_safety_q16')
 trace=None;termination='timeout';error=None
 for step in range(runtime.config.max_steps):
  before=env.positions.copy();flow=runtime.flow_world(env,jax.random.fold_in(root,step))
  snap=polygonal_obstacle_snapshot(env, sides=48) if fixed else legacy_snapshot(env)
  A,lo,geom=barrier_constraints(snap,runtime.cbf)
  try:
   result=runtime.projector(snap,flow);safe=np.asarray(result.velocity,float);diag=dict(result.diagnostics)
  except Exception as exc:
   error=f'{type(exc).__name__}: {exc}';termination='numerical_failure';break
  _,_,done,info=env.step(safe);termination=info['termination']
  if done:
   trace={'step':step+1,'before':before.tolist(),'flow_action':flow.tolist(),'projected_action':safe.tolist(),
          'solver_status':diag.get('solver_status',result.status),'projection_status':result.status,
          'encoded_min_residual':float(np.min(A@safe.reshape(-1)-lo)),
          'encoded_max_violation':diag.get('max_constraint_violation'),
          'encoded_wall_count':int(np.prod(geom['wall_h'].shape)),
          'next_state':env.positions.tolist(),'collision_checker':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in info.items() if k in ('obstacle_collision','outer_collision','agent_collision','wall_distances','swept_obstacle_distance','swept_outer_distance','swept_agent_distances','termination')}}
   break
 summary=env.summary()
 return {'fixed':fixed,'termination':termination,'error':error,'summary':summary,'terminal_trace':trace}

def main():
 collisions={}
 for path in glob.glob(str(ROOT/'diagnostics/orthoflow3_generator_critic_frozen_test_v1/raw/*.jsonl')):
  for line in open(path):
   try:r=json.loads(line)
   except:continue
   if r.get('scenario')=='ring_exchange' and r.get('controller_chain')=='hard_safety_q16' and r.get('collision'):
    collisions[(r['state_uid'],int(r['future_index']))]=r
 runtime=FrozenNewRuntime('ring_exchange');by={s['uid']:s for s in runtime.states};rows=[]
 for (uid,seed),historical in sorted(collisions.items()):
  old=replay(runtime,by[uid],seed,False);new=replay(runtime,by[uid],seed,True)
  rows.append({'state_uid':uid,'future_index':seed,'historical':historical,'legacy_reproduction':old,'fixed_replay':new,
               'classification':'S5_OMITTED_OUTER_BOUNDARY_CONSTRAINT'})
 summary={'historical_collisions':len(rows),'types':{'outer_boundary':sum(x['legacy_reproduction']['summary']['outer_collision'] for x in rows),
          'central_obstacle':sum(x['legacy_reproduction']['summary']['obstacle_collision'] for x in rows),
          'agent_agent':sum(x['legacy_reproduction']['summary']['agent_collision'] for x in rows)},
          'legacy_all_encoded_constraints_satisfied':all(x['legacy_reproduction']['terminal_trace']['encoded_min_residual']>=-1e-9 for x in rows),
          'fixed_collision_count':sum(any(x['fixed_replay']['summary'][k] for k in ('obstacle_collision','outer_collision','agent_collision')) for x in rows),
          'root_cause':'S5: Ring safety adapter omitted the physical outer boundary; projection satisfied every encoded central/pair constraint, but no outer-containment row existed.',
          'fix':'Add a conservative inscribed 48-edge outer polygon to the same generic wall-CBF snapshot; tolerances and hard-safety equations unchanged.'}
 (OUT/'ring_safety_collision_traces.json').write_text(json.dumps({'summary':summary,'collisions':rows},indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
