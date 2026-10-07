"""No-rollout probe of frozen controller response near the known goals."""
import argparse,copy,os,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .function_support import OUT as SOURCE,read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
OUT=SOURCE.parent/'goal_neighborhood_response'
RULE=SOURCE.parent/'goal_response_protocol.json'
ROOTS=(2026100417,2026100499,2026100521,2026100543)


def measurement(rt,physical,eta):
    import jax
    pos=np.array(physical['positions'],float);goal=np.array(physical['goals'],float)
    direction=(goal-pos)/np.maximum(np.linalg.norm(goal-pos,axis=1,keepdims=True),1e-12)
    perp=np.stack([-direction[:,1],direction[:,0]],-1)
    values=[]
    for di,offset in enumerate((direction,-direction,perp,-perp)):
        pp=copy.deepcopy(physical);pp['positions']=(goal+.30*offset).tolist();pp['velocities']=np.zeros_like(pos).tolist();pp['flow_committed']=False
        env=rt.core.reset(pp)
        wall,pair=env.distances()
        assert np.min(wall)>rt.core.cfg.collision_margin and np.min(pair)>0,'Invalid synthetic physical probe'
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
    a=np.array(values,np.float32);assert a.shape==(16,) and np.isfinite(a).all()
    return a


def build(index):
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    assert jax.default_backend()=='gpu'
    profiles=read(SOURCE/'protocol.json')['profiles'];p=profiles[index]
    dest=OUT/f'inputs_{p["name"]}.npz'
    if dest.exists():raise FileExistsError('Frozen derived input already exists')
    physical=read(SOURCE/'physical.json');pairs=read(SOURCE/'pairs.json');rt=RichRuntime('ring_exchange',p['path'])
    assert sha(p['path'])==p['sha256']
    features=[];timings=[];errors=[]
    for i,pair in enumerate(pairs):
        item=physical[pair['state_index']];assert item['state_uid']==pair['state_uid'];start=time.perf_counter()
        try:
            features.append(measurement(rt,item['physical'],np.array(pair['eta'],float)))
        except Exception as exc:
            errors.append(dict(pair=i,error=f'{type(exc).__name__}: {exc}'));features.append(np.full(16,np.nan,np.float32))
        timings.append(time.perf_counter()-start)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(dest,goal_response=np.array(features),valid=np.isfinite(features).all(1),seconds=np.array(timings))
    write(OUT/f'audit_{p["name"]}.json',dict(controller=p['name'],controller_sha256=p['sha256'],pairs=len(pairs),invalid=errors,
        protocol_sha256=sha(RULE),code_sha256=sha(__file__),state_manifest_sha256=sha(SOURCE/'states.json'),pair_manifest_sha256=sha(SOURCE/'pairs.json'),
        no_environment_steps=True,new_task_rollouts=0,task_labels_read=False,measurement_width=16,
        warm_sequential_pair_seconds=float(np.mean(timings[1:])),scope='Ringsource-side inputmeasurement; deploymentbatchlatency notbenchmarked'))
    print(dict(controller=p['name'],pairs=len(pairs),invalid=errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();build(a.index)
