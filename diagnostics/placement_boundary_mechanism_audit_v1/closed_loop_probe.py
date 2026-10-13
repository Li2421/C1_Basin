"""Paired, fixed-key TT/FF physical continuations for trajectory/margin probes.

This reads a successful shared-cache preflight before any env.step. The finite
horizon mode is diagnostic only and never creates a binary success label.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
for path in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts')):sys.path.insert(0,path)
# The active repository owns the authoritative shared rollout cache. The
# frozen field source also contains an old package with the same import name.
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity

import jax
import numpy as np
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action


def key_for(uid,seed,step,future_root):
    token=int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)
    key=jax.random.fold_in(jax.random.PRNGKey(future_root),token)
    return jax.random.fold_in(jax.random.fold_in(key,seed),step)


def margin(scene,env,info):
    goal=float(np.min(env.config.goal_tolerance-np.asarray(info['goal_errors'])))
    if scene=='toy_give_way':
        safety=float(min(info['min_swept_wall_distance']-env.config.wall_collision_margin,
                         info['min_swept_agent_distance']-env.config.agent_collision_margin))
        if np.any(info['outside_workspace']): safety=min(safety,-1e-12)
    elif scene=='ring_exchange':
        safety=float(min(np.min(info['swept_obstacle_distance'])-env.config.collision_margin,
                         np.min(info['swept_outer_distance'])-env.config.collision_margin,
                         np.min(info['swept_agent_distances'])-env.config.agent_collision_margin))
    else:raise ValueError(scene)
    return goal,safety


def simulate(rt,state,placement,eta,seed,future_root,horizon):
    env=rt.make_env(state);p=[env.positions.copy()];v=[env.velocities.copy()];u=[];keys=[];gm=[];sm=[]
    term='truncated';error=None
    for step in range(min(horizon,rt.config.max_steps)):
        key=key_for(state['uid'],seed,step,future_root)
        try:
            a,cert,_,_=action(rt,env,key,np.asarray(eta,dtype=np.float64),placement)
            _,_,done,info=env.step(a)
        except Exception as exc:
            term='numerical_failure';error={'step':step,'type':type(exc).__name__,'message':str(exc)[:1000]};break
        g,s=margin(state['scenario'],env,info)
        p.append(env.positions.copy());v.append(env.velocities.copy());u.append(a.copy());keys.append(np.asarray(key))
        gm.append(g);sm.append(s)
        if done:term=info['termination'];break
    goal_ever=float(np.max(gm)) if gm else None
    safety_always=float(np.min(sm)) if sm else None
    rho=min(goal_ever,safety_always) if goal_ever is not None and safety_always is not None else None
    summary=env.summary()
    success=bool(summary['collision_free_success'] and term=='success' and error is None) if term!='truncated' else None
    return dict(positions=np.asarray(p),velocities=np.asarray(v),actions=np.asarray(u),noise_keys=np.asarray(keys),
                goal_margins=np.asarray(gm),safety_margins=np.asarray(sm)),dict(
        placement=placement,state_uid=state['uid'],eta=eta,seed=seed,horizon=horizon,termination=term,error=error,
        steps=len(u),rho=rho,goal_ever_margin=goal_ever,safety_always_margin=safety_always,
        goal_active_step=int(np.argmax(gm))+1 if gm else None,safety_active_step=int(np.argmin(sm))+1 if sm else None,
        success=success,plant_summary=summary)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state-uid',required=True)
    ap.add_argument('--center',choices=['zero','candidate4'],required=True)
    ap.add_argument('--horizon',type=int,default=80)
    ap.add_argument('--only-center',action='store_true')
    args=ap.parse_args()
    protocol=json.loads((HERE/'closed_loop_protocol.json').read_text())
    st=next(s for s in protocol['selected_states'] if s['uid']==args.state_uid)
    pre=json.loads((HERE/'cache_preflight.json').read_text())
    if pre['summary']['ambiguous'] or not pre['details']:raise RuntimeError('shared-cache preflight unresolved')
    planned=json.loads((HERE/'planned_rollouts.json').read_text())['requests']
    future_root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    rt=make_runtime(st['scenario'])
    outdir=HERE/('results_full' if args.horizon>=rt.config.max_steps else f'results_h{args.horizon}')
    outdir.mkdir(exist_ok=True)
    points=[p for p in protocol['points'] if p['center']==args.center and (not args.only_center or p['variant']=='center')]
    for placement in ('TT','FF'):
        for point in points:
            eta=point['eta'];eid=eta_identity(eta)[0]
            if not any(r['state_uid']==st['uid'] and r['eta_uid']==eid for r in planned):raise RuntimeError('unplanned eta')
            stem=f"{st['uid']}_{placement}_{args.center}_{point['variant']}"
            path=outdir/(stem+'.npz');meta=outdir/(stem+'.json')
            if path.exists() and meta.exists():continue
            arrays,record=simulate(rt,st,placement,eta,0,future_root,args.horizon)
            record['variant']=point['variant'];record['eta_uid']=eid
            np.savez_compressed(path,**arrays)
            meta.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
            print(st['uid'],placement,args.center,point['variant'],record['steps'],record['termination'],record['rho'],flush=True)


if __name__=='__main__':main()
