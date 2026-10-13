"""Input-only measurements; no simulator step and no eta-conditioned preview."""
from __future__ import annotations
import argparse
import time
import numpy as np
from .experiment import HERE,read,write,sha
from .runtime import native_runtime,AnalyticField,controller_context,FIELD


def field_context(rt,state):
    import jax
    import jax.numpy as jnp
    env=rt.make_env(state)
    p,g,v=env.positions.copy(),env.goals.copy(),env.velocities.copy()
    e=g-p;e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    lat=np.stack((-e[:,1],e[:,0]),-1);speed=rt.config.max_speed
    poses=[(p,v),(p,.5*speed*e),(p,-.5*speed*e),
           (g-.25*e,np.zeros_like(v)),(g-.25*e,.5*speed*e),
           (g-.25*e,-.5*speed*e),(g-.25*e,.5*speed*lat)]
    vectors=[np.zeros_like(v),.5*speed*e,-.5*speed*e,.5*speed*lat,-.5*speed*lat]
    values=[]
    for position,velocity in poses:
        env.positions[:]=position;env.velocities[:]=velocity
        obs,_,_=rt.nominal(env,jax.random.PRNGKey(2026100908))
        for t in (0.,.5,.9):
            for h in vectors:
                z=(rt.to_local(h,env)-rt.mean)/rt.scale
                fz=np.asarray(rt.velocity_fn(jnp.asarray(obs),jnp.asarray(z),jnp.asarray(t))).reshape(-1,2)
                w=rt.to_world(rt.scale*fz,env)
                values.append(np.stack(((w*e).sum(-1),(w*lat).sum(-1)),-1)/speed)
    result=np.concatenate(values,axis=-1).astype(np.float32)
    assert result.shape==(len(p),210) and np.isfinite(result).all()
    # Analytic fields have known slope: this is an implementation check using
    # controller functions, never a success label or learned feature.
    if isinstance(rt,AnalyticField):
        slope=result[:,2]-result[:,4]
        np.testing.assert_allclose(slope,-rt.parameters[5],rtol=1e-5,atol=1e-5)
    return result


def run(index=0,workers=1):
    from diagnostics.physical_state_critic_formal_v1 import physical_entities as phys
    p=read(HERE/'protocol.json')
    assert p['runtime_sha256']==sha(HERE/'runtime.py')
    assert p['FF_operator_sha256']==sha(FIELD/'poc/field_controller.py')
    assert p['physical_representation_sha256']==sha(phys.__file__)
    bases={};runtimes={};count=0;t0=time.monotonic()
    for i,s in enumerate(p['states']):
        if i%workers!=index:continue
        scene=s['scenario']
        if scene not in bases:bases[scene]=native_runtime(scene)
        x=phys.entities(scene,s['physical'])
        cfg=bases[scene].config
        np.testing.assert_allclose(x['agents'][:,7],cfg.agent_radius/.2)
        np.testing.assert_allclose(x['agents'][:,8],cfg.goal_tolerance/.1)
        for name,ctl in p['controllers'][scene].items():
            # No need to measure held-controller features on TRAIN states.
            ids=(['analytic6','analytic7'] if s['study_split']=='final' else
                 [f'analytic{k}' for k in range(4 if s['study_split']=='train' else 6)]+['flow'])
            if name not in ids:continue
            dest=HERE/'features'/s['state_uid']/(name+'.npz')
            if dest.exists():continue
            if (scene,name) not in runtimes:
                runtimes[scene,name]=bases[scene] if name=='flow' else AnalyticField(bases[scene],ctl['payload']['controller_parameters'])
            rt=runtimes[scene,name]
            nominal=controller_context(rt,s);field=field_context(rt,s)
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**{'entity_'+k:v for k,v in x.items()},nominal=nominal,field=field)
            tmp.replace(dest);count+=1
        if count and count%80==0:print(dict(measurements=count,seconds=time.monotonic()-t0),flush=True)
    write(HERE/'features'/f'complete{index}.json',dict(index=index,workers=workers,measurements=count,
          seconds=time.monotonic()-t0,feature_code_sha256=sha(__file__),success_labels_used=False,
          simulator_steps=0,eta_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();run(a.index,a.workers)
