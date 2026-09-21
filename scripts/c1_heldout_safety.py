"""The audited swept-distance and applied-control checks, reusable per branch."""
import numpy as np
from single_integrator.environment import segment_distance
from single_integrator.cbf import barrier_constraints

def check(z,env,cfg):
    plant=env.config;walls=env.walls;seg=walls[:,1]-walls[:,0]
    x=z['positions_before'];y=z['positions_after'];u=z['applied'].reshape(-1,2,2)
    rel=x[:,0]-x[:,1];delta=y[:,0]-y[:,1]-rel
    t=np.clip(-np.sum(rel*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-300),0,1)
    pair=np.linalg.norm(rel+t[:,None]*delta,axis=1)-2*plant.agent_radius
    wd=segment_distance(x[:,:,None,:],y[:,:,None,:],walls[None,None,:,0,:],walls[None,None,:,1,:])-plant.agent_radius-plant.wall_radius
    foot=np.clip(np.sum((x[:,:,None,:]-walls[None,None,:,0,:])*seg,axis=-1)/np.sum(seg*seg,axis=1),0,1)
    normal=x[:,:,None,:]-(walls[None,None,:,0,:]+foot[:,:,:,None]*seg)
    distance=np.linalg.norm(normal,axis=-1);normal/=distance[:,:,:,None]
    ph=np.sum(rel*rel,axis=1)-(2*plant.agent_radius+plant.agent_collision_margin+cfg.separation_buffer)**2
    wh=distance-plant.agent_radius-plant.wall_radius-plant.wall_collision_margin-cfg.separation_buffer
    pr=2*np.sum(rel*(u[:,0]-u[:,1]),axis=1)+cfg.gamma*ph
    wr=np.sum(normal*u[:,:,None,:],axis=-1)+cfg.gamma_wall*wh
    res=np.c_[pr,wr.reshape(len(x),-1)]
    for k in [0,len(x)-1]:
        A,b,_=barrier_constraints(dict(positions=x[k],walls=walls,config=plant.to_dict()),cfg)
        np.testing.assert_allclose(res[k],A@u[k].reshape(4)-b,rtol=1e-12,atol=1e-14)
    p=np.concatenate([x,y]);w=plant.corridor_width/2
    inside=((np.abs(p[:,:,0])<=plant.corridor_half_length)&(np.abs(p[:,:,1])<=w))|((np.abs(p[:,:,0])<=w)&(p[:,:,1]>=w)&(p[:,:,1]<=plant.bay_top))
    return dict(steps=len(x),agent_collision_steps=int(np.sum(pair<=plant.agent_collision_margin)),
        wall_collision_steps=int(np.sum(np.any(wd<=plant.wall_collision_margin,axis=(1,2)))),
        outside_endpoints=int(np.sum(~inside)),cbf_violations=int(np.sum(res < -cfg.feasibility_tol)),
        speed_violations=int(np.sum(np.linalg.norm(u,axis=2)>plant.max_speed+cfg.speed_tol)),
        min_center_separation=float(pair.min()+2*plant.agent_radius),min_agent_surface_clearance=float(pair.min()),
        min_wall_surface_clearance=float(wd.min()),min_pair_cbf_residual=float(pr.min()),min_wall_cbf_residual=float(wr.min()),
        min_pair_h=float(ph.min()),min_wall_h=float(wh.min()),max_speed=float(np.linalg.norm(u,axis=2).max()),
        integration_error=float(np.max(np.abs(y-x-plant.dt*u))))
