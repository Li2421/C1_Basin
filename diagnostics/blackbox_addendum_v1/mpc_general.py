"""Only the agent count and native static geometry generalize distributed_shooting."""
from core import *
from legacy_controllers import bound
import jax
jax.config.update('jax_enable_x64',False)

def distributed_shooting(env,key,parameters):
    horizon,samples,weight=parameters;cfg=env.config;answer=[];pos=env.positions;coast=env.velocities
    for i in range(len(pos)):
        token=np.asarray(jax.random.fold_in(key,i),np.uint32).tobytes();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'));mean=bound((env.goals-pos)[i],cfg.max_speed);knots=bound(mean[None,None]+rng.normal(0,.8*cfg.max_speed,(samples,3,2)),cfg.max_speed);acts=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)];acts[0]=mean;acts[1]=0;paths=pos[i]+cfg.dt*np.cumsum(acts,axis=1)
        if hasattr(cfg,'obstacle_radius'):
            radius=np.linalg.norm(paths,axis=-1);inner=np.maximum(cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin-radius,0)**2;outer=np.maximum(radius-(cfg.outer_radius-cfg.agent_radius-cfg.collision_margin),0)**2
        else:
            walls=np.asarray(env.walls,np.float64);start=walls[:,0];segment=walls[:,1]-start;relative=paths[:,:,None,:]-start[None,None,:,:];fraction=np.clip(np.sum(relative*segment[None,None],axis=-1)/np.maximum(np.sum(segment*segment,axis=-1),1e-30),0,1);nearest=start[None,None]+fraction[...,None]*segment[None,None];distance=np.linalg.norm(paths[:,:,None]-nearest,axis=-1);inner=np.square(np.maximum(env.instance.geometry.margin-distance,0)).sum(-1);outer=np.zeros_like(inner)
        neighbors=[j for j in range(len(pos)) if j!=i];forecast=pos[neighbors][None]+cfg.dt*np.arange(1,horizon+1)[:,None,None]*coast[neighbors][None];distance=np.linalg.norm(paths[:,:,None]-forecast[None],axis=-1);pairs=np.maximum(2*cfg.agent_radius+cfg.agent_collision_margin-distance,0)**2;smooth=np.diff(np.concatenate([np.broadcast_to(coast[i],(samples,1,2)),acts],axis=1),axis=1);cost=np.square(paths-env.goals[i]).sum(-1).mean(1)+weight*(inner.mean(1)+outer.mean(1)+pairs.mean((1,2)))+.05*np.square(smooth).sum(-1).mean(1);answer.append(acts[int(cost.argmin()),0])
    return np.asarray(answer)
