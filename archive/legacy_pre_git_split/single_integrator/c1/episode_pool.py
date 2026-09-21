"""Parallel independent rollouts with unchanged scalar physics and derivatives.

Each spawned worker keeps the original one-trajectory JAX/CPU callback path.
Batch losses and gradients are averaged in input order. This changes scheduling,
not the trajectory sampler, safety solver, derivative, or C1 objective.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np


def initialize(risk_name,baseline_seed,seed,affinity):
    if affinity:
        index=(mp.current_process()._identity[-1]-1)%len(affinity)
        os.sched_setaffinity(0,{affinity[index]})
    from single_integrator.c1.train_deadlock_union import setup
    from single_integrator.c1.rollout_deadlock_union import rollout
    from single_integrator.environment import GiveWayEnv
    if risk_name=='ordered':
        from single_integrator.c1.risk.ordered_guidance import trajectory as risk
    elif risk_name=='exact':
        from single_integrator.c1.risk.exact_margin import trajectory as risk
    else:raise ValueError('unknown risk')
    global FIELD,PLANT,CBF,FN,VG,SCORE
    _,FIELD,PLANT,CBF,_=setup(seed,baseline_seed)
    goals=jnp.asarray(GiveWayEnv(PLANT).goals)
    kw=dict(dt=PLANT.dt,max_speed=PLANT.max_speed,goal_tolerance=PLANT.goal_tolerance,
        hold_seconds=PLANT.deadlock_hold_seconds,progress_window_seconds=PLANT.progress_window_seconds,
        progress_epsilon=PLANT.progress_epsilon,speed_epsilon_fraction=PLANT.speed_epsilon_fraction)
    SCORE=jax.jit(lambda b,a,u,m,to:risk(b,a,u,goals,m,terminal_timeout=to,**kw)['J_live'])
    @jax.jit
    def fn(phi,x,draws):
        terms,t=rollout(phi,FIELD,x,draws,PLANT,CBF)
        r=SCORE(t['before'],t['after'],t['applied'],t['alive'],terms['timeout'])
        return dict(J_live=r,J_def=terms['J_def'],deadlock=terms['deadlock'],
            either_deadlock=terms['either_deadlock'],success=terms['success'],timeout=terms['timeout'])
    def loss(phi,x,draws,dual,epsilon):
        v=fn(phi,x,draws)
        return v['J_def']+dual*(v['J_live']-epsilon),v
    FN=fn;VG=jax.jit(jax.value_and_grad(loss,has_aux=True))


def task(payload):
    from single_integrator.c1.train_deadlock_union import noise
    params,item,seed,dual,epsilon,gradient=payload
    phi=jax.tree_util.tree_map(jnp.asarray,params)
    x=jnp.asarray(item['initial']);draws=noise(seed,item['rid'])
    if gradient:
        (value,aux),g=VG(phi,x,draws,jnp.asarray(dual),jnp.asarray(epsilon))
        return float(value),{k:float(v) for k,v in aux.items()},jax.device_get(g)
    aux=FN(phi,x,draws)
    value=aux['J_def']+dual*(aux['J_live']-epsilon)
    return float(value),{k:float(v) for k,v in aux.items()},None


class EpisodePool:
    def __init__(self,risk_name,workers=4,baseline_seed=0,seed=0):
        if workers<1:raise ValueError('positive workers required')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            os.environ[name]='1'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        cpus=sorted(os.sched_getaffinity(0))
        self.executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(risk_name,baseline_seed,seed,cpus))

    def rows(self,params,items,seeds,dual=1.,epsilon=.01,gradient=False):
        host=jax.device_get(params)
        payloads=[(host,item,int(seed),float(dual),float(epsilon),gradient) for item,seed in zip(items,seeds)]
        if len(items)!=len(seeds):raise ValueError('one seed per episode required')
        return list(self.executor.map(task,payloads,chunksize=1))

    def evaluate(self,params,items,seed,dual,epsilon,gradient=False):
        rows=self.rows(params,items,[seed]*len(items),dual,epsilon,gradient)
        aux={k:jnp.asarray(np.mean([r[1][k] for r in rows])) for k in ('J_live','J_def')}
        value=jnp.asarray(np.mean([r[0] for r in rows]))
        result=(value,aux)
        if not gradient:return result
        g=jax.tree_util.tree_map(lambda *xs:jnp.asarray(np.mean(np.stack(xs),axis=0)),*[r[2] for r in rows])
        return result,g

    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)
