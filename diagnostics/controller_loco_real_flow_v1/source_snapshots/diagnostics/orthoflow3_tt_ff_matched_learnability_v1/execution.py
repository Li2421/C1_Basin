"""Frozen matched task execution. No eta/basis/lifting or solver modification."""
from __future__ import annotations
import hashlib
import time
import numpy as np


def key_for(state, seed, step):
    import jax
    token=int(hashlib.sha256(state['uid'].encode()).hexdigest()[:8],16)
    key=jax.random.fold_in(jax.random.PRNGKey(2026100403),token)
    return jax.random.fold_in(jax.random.fold_in(key,seed),step)


def rollout_tt(rt,c):
    """Exact native terminal/terminal comparator for the archived FF runner."""
    env=rt.make_env(c['state']);eta=np.asarray(c['eta'],float)
    error=None;term='timeout';calls=0;steps=0;first=None;t0=time.monotonic()
    for step in range(rt.config.max_steps):
        try:
            action,cert,n=rt.action(env,key_for(c['state'],c['seed'],step),eta,'old')
            calls+=n;steps+=1
            if first is None:first=np.asarray(action).tolist()
            _,_,done,info=env.step(action);term=info['termination']
            if done:break
        except Exception as ex:
            error={'step':step,'type':type(ex).__name__,'message':str(ex)[:600]}
            term='numerical_failure';break
    summary=env.summary()
    collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
    return dict(id=c['id'],state_uid=c['state']['uid'],scenario=rt.scenario,eta_index=c['eta_index'],eta=c['eta'],
        seed=c['seed'],phase='matched_TT',success=bool(summary['collision_free_success'] and term=='success' and error is None),
        collision=bool(collision),termination=term,error=error,steps=steps,first_action=first,
        projection_calls=calls,elapsed_seconds=time.monotonic()-t0)


def rollout(rt,c,chain):
    if chain=='TT':return rollout_tt(rt,c)
    assert chain=='FF'
    if rt.scenario=='toy_give_way':
        from field_pipeline_v1.external_toy_runner import rollout as original
    else:
        assert rt.scenario=='ring_exchange'
        from field_pipeline_v1.ring_t0_source import rollout as original
    return original(rt,c)
