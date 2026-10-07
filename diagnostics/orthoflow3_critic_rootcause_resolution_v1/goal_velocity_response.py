"""One extra physical controller query regime: moving toward the goal.

The zero-velocity endpoint fingerprint misses an explicitly tested controller
degree of freedom. Measure the same four endpoint configurations at v_max/2
toward each goal. No task simulation steps, outcomes, or controller IDs enter
the features. Preserve the original sixteen coordinates and probe RNG roots.
"""
import argparse
import copy
import os
import time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .goal_response import SOURCE,ROOTS,OUT as REST
from .function_support import read,write,sha
OUT=REST.parent/'goal_velocity_response'


def prepare():
    assert not (OUT/'protocol.json').exists(),'Input design frozen'
    write(OUT/'protocol.json',dict(
        question='Does a physically missing near-goal velocity-response regime improve source-controller generalization?',
        motivation='Static goal queries all use zero velocity. Frozen motion-gated intervention keeps existing inputs identical but changes16/16 into5/16 in one source cell; independent confirmation kept separate.',
        measurement='Same four goal offsets0.30m, same four probe roots; set each velocity to0.5*vmax toward its goal. Raw/safe/eta-executed goal projections, mean and executed minimum exactly match original16D coordinate definitions.',
        state_manifest_sha256=sha(SOURCE/'states.json'),pair_manifest_sha256=sha(SOURCE/'pairs.json'),
        feature_choices_frozen_before_new_training=True,target_labels_used=False,
        controller_profiles=read(SOURCE/'protocol.json')['profiles'],code_sha256=sha(__file__),
        training_controls=['rest_only with16zero motion channels','motion_only with16zero rest channels','rest_plus_motion'],
        unchanged='Same frozen source64TRAIN16VAL families,12native+2crossed programs,sourcecontrollerfolds,network dimensions,seeds,batches,W1NLL,optimizer,cadence,source-onlycheckpoint selection',
        no_new_task_rollouts=True,no_environment_steps=True,generator_modified=False,
        promotion_gate='Require source-controllerCV and causal/state-contrast improvement withinput-use evidence. No automatic new target on calibration alone.'))


def measurement(rt,physical,eta):
    import jax
    pos=np.array(physical['positions'],float);goal=np.array(physical['goals'],float)
    direction=(goal-pos)/np.maximum(np.linalg.norm(goal-pos,axis=1,keepdims=True),1e-12)
    perp=np.stack([-direction[:,1],direction[:,0]],-1);values=[]
    for di,offset in enumerate((direction,-direction,perp,-perp)):
        pp=copy.deepcopy(physical)
        pp['positions']=(goal+.30*offset).tolist()
        pp['velocities']=(-.5*rt.core.cfg.max_speed*offset).tolist()
        pp['flow_committed']=False
        env=rt.core.reset(pp)
        wall,pair=env.distances()
        assert np.min(wall)>rt.core.cfg.collision_margin and np.min(pair)>0
        assert np.allclose(np.linalg.norm(env.velocities,axis=1),.5*rt.core.cfg.max_speed,atol=1e-10)
        along=(goal-env.positions)/np.linalg.norm(goal-env.positions,axis=1,keepdims=True)
        root_features=[]
        for root in ROOTS:
            key=jax.random.fold_in(jax.random.PRNGKey(root),di)
            raw=np.asarray(rt.alt_flow(env,key),float);safe=rt.core.project(env,raw)
            correction=rt.core.basis.compute(env.positions,env.goals,safe,rt.core.cfg.max_speed).correction(eta)
            executed=rt.core.project(env,safe+correction)
            projection=[np.sum(a*along,axis=1)/rt.core.cfg.max_speed for a in (raw,safe,executed)]
            root_features.append([projection[0].mean(),projection[1].mean(),projection[2].mean(),projection[2].min()])
        values.extend(np.mean(root_features,axis=0))
    result=np.asarray(values,np.float32)
    assert result.shape==(16,) and np.isfinite(result).all()
    return result


def build(index):
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    assert jax.default_backend()=='gpu'
    protocol=read(OUT/'protocol.json')
    assert protocol['code_sha256']==sha(__file__)
    p=protocol['controller_profiles'][index];dest=OUT/f'inputs_{p["name"]}.npz'
    if dest.exists():
        read(OUT/f'audit_{p["name"]}.json')
        return
    assert sha(p['path'])==p['sha256']
    rt=RichRuntime('ring_exchange',p['path'])
    states,pairs=read(SOURCE/'physical.json'),read(SOURCE/'pairs.json')
    values,timing,errors=[],[],[]
    for i,pair in enumerate(pairs):
        state=states[pair['state_index']]
        assert state['state_uid']==pair['state_uid']
        start=time.perf_counter()
        try:
            values.append(measurement(rt,state['physical'],np.asarray(pair['eta'],float)))
        except Exception as exc:
            errors.append(dict(pair=i,error=f'{type(exc).__name__}: {exc}'))
            values.append(np.full(16,np.nan,np.float32))
        timing.append(time.perf_counter()-start)
    np.savez_compressed(dest,goal_motion_response=np.array(values),valid=np.isfinite(values).all(1),seconds=np.array(timing))
    write(OUT/f'audit_{p["name"]}.json',dict(controller=p['name'],controller_sha256=p['sha256'],pairs=len(pairs),invalid=errors,
        protocol_sha256=sha(OUT/'protocol.json'),code_sha256=sha(__file__),new_task_rollouts=0,
        no_environment_steps=True,task_labels_read=False,measurement_width=16,warm_mean_seconds=float(np.mean(timing[1:]))))
    print(dict(controller=p['name'],pairs=len(pairs),invalid=errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','build'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:build(a.index)
