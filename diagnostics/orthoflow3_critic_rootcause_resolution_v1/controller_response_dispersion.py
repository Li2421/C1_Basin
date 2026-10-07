"""Retain omitted seed-response dispersion; source-only controller queries."""
import argparse
import copy
import os
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .goal_response import ROOTS,OUT as REST
from .goal_velocity_response import OUT as MOTION
from .function_support import OUT as SOURCE,read,write,sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'controller_response_dispersion'
RULE=ROOT/'controller_response_dispersion_protocol.json'


def measurement(rt,physical,eta,moving):
    import jax
    pos=np.asarray(physical['positions'],float);goal=np.asarray(physical['goals'],float)
    direction=(goal-pos)/np.maximum(np.linalg.norm(goal-pos,axis=1,keepdims=True),1e-12)
    perp=np.stack((-direction[:,1],direction[:,0]),-1)
    values=[]
    for di,offset in enumerate((direction,-direction,perp,-perp)):
        pp=copy.deepcopy(physical);pp['positions']=(goal+.30*offset).tolist()
        pp['velocities']=(-.5*rt.core.cfg.max_speed*offset if moving else np.zeros_like(pos)).tolist()
        pp['flow_committed']=False;env=rt.core.reset(pp)
        wall,pair=env.distances()
        assert np.min(wall)>rt.core.cfg.collision_margin and np.min(pair)>0
        along=(goal-env.positions)/np.linalg.norm(goal-env.positions,axis=1,keepdims=True)
        seeds=[]
        for root in ROOTS:
            key=jax.random.fold_in(jax.random.PRNGKey(root),di)
            raw=np.asarray(rt.alt_flow(env,key),float);safe=rt.core.project(env,raw)
            correction=rt.core.basis.compute(env.positions,env.goals,safe,rt.core.cfg.max_speed).correction(eta)
            executed=rt.core.project(env,safe+correction)
            projection=[np.sum(a*along,axis=1)/rt.core.cfg.max_speed for a in (raw,safe,executed)]
            seeds.append((projection[0].mean(),projection[1].mean(),projection[2].mean(),projection[2].min()))
        values.append(seeds)
    samples=np.asarray(values,float)
    assert samples.shape==(4,4,4) and np.isfinite(samples).all()
    # Population standard deviation of the fixed physical measurement roots;
    # this is a descriptor, not an estimated CI or a new outcome label.
    return samples.mean(1).ravel().astype(np.float32),samples.std(1).ravel().astype(np.float32),samples.astype(np.float32)


def build(index):
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    assert jax.default_backend()=='gpu'
    profiles=read(SOURCE/'protocol.json')['profiles'];p=profiles[index]
    assert sha(p['path'])==p['sha256']
    OUT.mkdir(exist_ok=True);dest=OUT/f'inputs_{p["name"]}.npz'
    assert not dest.exists(),'Preserve measured inputs'
    pairs=read(SOURCE/'pairs.json');physical=read(SOURCE/'physical.json')
    rt=RichRuntime('ring_exchange',p['path'])
    old_rest=np.load(REST/f'inputs_{p["name"]}.npz')['goal_response']
    old_motion=np.load(MOTION/f'inputs_{p["name"]}.npz')['goal_motion_response']
    means=[];stds=[];samples=[];seconds=[]
    for i,pair in enumerate(pairs):
        state=physical[pair['state_index']]
        assert state['state_uid']==pair['state_uid'] and pair['split'] in ('train','validation')
        start=time.perf_counter();r=measurement(rt,state['physical'],np.asarray(pair['eta'],float),False)
        m=measurement(rt,state['physical'],np.asarray(pair['eta'],float),True)
        assert np.allclose(r[0],old_rest[i],atol=3e-6,rtol=3e-6),(p['name'],i,'rest parity')
        assert np.allclose(m[0],old_motion[i],atol=3e-6,rtol=3e-6),(p['name'],i,'motion parity')
        means.append(np.concatenate((r[0],m[0])));stds.append(np.concatenate((r[1],m[1])))
        samples.append(np.stack((r[2],m[2])));seconds.append(time.perf_counter()-start)
    means,stds,samples=np.asarray(means),np.asarray(stds),np.asarray(samples)
    assert np.isfinite(stds).all() and (stds>=0).all()
    np.savez_compressed(dest,response_mean=means,response_std=stds,root_samples=samples,seconds=seconds)
    write(OUT/f'audit_{p["name"]}.json',dict(controller=p['name'],controller_sha256=p['sha256'],
        states_sha256=sha(SOURCE/'states.json'),pairs_sha256=sha(SOURCE/'pairs.json'),
        code_sha256=sha(__file__),rule_sha256=sha(RULE),old_mean_numerical_parity=True,
        new_task_rollouts=0,environment_steps=0,queries_per_pair=32,feature_width=32,
        median_std=float(np.median(stds)),max_std=float(stds.max()),
        warm_seconds_per_pair=float(np.mean(seconds[1:])),
        target_labels_used=False,task_labels_read=False))
    print(dict(controller=p['name'],pairs=len(pairs),mean_parity=True,median_std=float(np.median(stds)),new_task_rollouts=0),flush=True)


def summarize():
    arrays=[];audits=[]
    profiles=read(SOURCE/'protocol.json')['profiles']
    for p in profiles:
        a=read(OUT/f'audit_{p["name"]}.json')
        assert a['code_sha256']==sha(__file__) and a['rule_sha256']==sha(RULE)
        arrays.append(np.load(OUT/f'inputs_{p["name"]}.npz')['response_std']);audits.append(a)
    values=np.asarray(arrays)
    write(OUT/'measurement_audit.json',dict(controllers=12,pairs_per_controller=160,
        new_task_rollouts=0,environment_steps=0,feature_width=32,old_mean_parity=True,
        median_std=float(np.median(values)),max_std=float(values.max()),
        controller_mean_std=values.mean((1,2)).tolist(),
        informative_channels=int(np.sum(values.reshape(-1,32).std(0)>1e-6)),
        protocol_sha256=sha(RULE),code_sha256=sha(__file__),audits=audits))
    print(dict(controllers=12,median_std=float(np.median(values)),max_std=float(values.max()),new_task_rollouts=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();build(a.index) if a.action=='build' else summarize()
