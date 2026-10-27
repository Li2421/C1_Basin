"""Unfiltered nominal execution for the common-protocol transition reference."""
from experiment import *
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from controllers import bound
from certified_planning import MemoNominal as Nominal
import jax
import numpy as np
import time
jax.config.update('jax_enable_x64',False)
def run(c):
    start=time.monotonic();env=LiteratureEnv(c['state'],LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config;actor=Nominal(c['controller_payload']['nominal'])
    root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(c['state_uid'].encode()).hexdigest()[:8],16)),c['seed'])
    p=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];clear=[];error=None;term='timeout'
    for t in range(cfg.max_steps):
        try:
            a=bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64);a[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
            _,done,info=env.step(a,diagnose_stalls=False);p.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(a.copy());clear.append(info['swept_clearance']);term=info['termination']
            if done:break
        except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e));term='unknown';break
    arrays=dict(positions=np.asarray(p),velocities=np.asarray(vel),actions=np.asarray(actions).reshape(-1,env.num_agents,2),goals=env.goals,swept_clearance=np.asarray(clear))
    return dict(success=int(term=='success' and not env.collided),collision=int(env.collided),timeout=int(term=='timeout'),numerical_failure=int(error is not None),outcome=term,episode_length=len(actions),error=error,safety_applied=False,elapsed_seconds=time.monotonic()-start,jax_enable_x64=bool(jax.config.x64_enabled)),arrays
