"""Isolated Phase-B coordinate encoding; no changes to physical execution."""
from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from ring_exchange.local_frame import local_observation, vectors_to_local

VERSION = 'physical_chart_and_agent_order_v1'


def rotate(a, k):
    a=np.asarray(a,dtype=np.float64)
    x,y=a[...,0],a[...,1]
    return np.stack(((x,-y,-x,y)[k%4],(y,x,-y,-x)[k%4]),axis=-1)


def key(x):
    return tuple(np.round(np.asarray(x).ravel(),10))


def four_observation(p,v,g):
    return np.asarray([np.concatenate((p[i],v[i],g[i]-p[i],
        *[q for j in range(4) if j!=i for q in (p[j]-p[i],v[j]-v[i])]))
        for i in range(4)],np.float64)


def encode(scenario, flat, physical, descriptor):
    flat=np.asarray(flat,np.float64)
    if scenario=='double_bottleneck':
        return flat,{'version':VERSION,'transform':'identity_control'}
    p,v,g=(np.asarray(physical[n],np.float64) for n in ('positions','velocities','goals'))
    flow=flat[-8:].reshape(4,2)
    if scenario=='four_way_intersection':
        # Enumerate the four valid square rotations; use only present physical
        # fields to choose a representative. Goal associations travel with agents.
        choices=[]
        for k in range(4):
            pp,vv,gg,ff=(rotate(x,k) for x in (p,v,g,flow))
            order=np.asarray(sorted(range(4),key=lambda i:key(np.r_[gg[i],pp[i],vv[i]])))
            pp,vv,gg,ff=(x[order] for x in (pp,vv,gg,ff))
            choices.append((key(np.r_[gg.ravel(),pp.ravel(),vv.ravel()]),k,order,pp,vv,gg,ff))
        _,k,order,pp,vv,gg,ff=min(choices,key=lambda x:(x[0],x[1]))
        obs=four_observation(pp,vv,gg)
        return np.r_[obs.ravel(),ff.ravel()],{'version':VERSION,'quarter_turns':int(k),'agent_order':order.tolist()}
    if scenario!='ring_exchange':raise ValueError(scenario)
    # Agent signatures are rotation-invariant physical values. Rebuild relative
    # blocks after ordering, so own and other-agent slots stay consistent.
    gv=vectors_to_local(g-p,p);lv=vectors_to_local(v,p)
    order=np.asarray(sorted(range(4),key=lambda i:key(np.r_[np.linalg.norm(p[i]),gv[i],lv[i]])))
    pp,vv,gg,ff=(x[order] for x in (p,v,g,flow))
    cfg=SimpleNamespace(obstacle_radius=float(descriptor['obstacle_radius']),
                        outer_radius=float(descriptor['outer_radius']),
                        agent_radius=float(descriptor['agent_radius']))
    obs=local_observation(pp,vv,gg,cfg).astype(np.float64)
    local_flow=vectors_to_local(ff,pp)
    return np.r_[obs.ravel(),local_flow.ravel()],{'version':VERSION,'chart':'outward_radial_ccw_tangent',
                                               'agent_order':order.tolist()}


def shared_slot_stats(h,scenario):
    h=np.asarray(h,np.float64)
    if scenario=='double_bottleneck':
        mean,std=h.mean(0),h.std(0)
    else:
        width=18 if scenario=='four_way_intersection' else 23
        obs=h[:,:4*width].reshape(-1,4,width)
        action=h[:,-8:].reshape(-1,4,2)
        mean=np.r_[np.tile(obs.mean((0,1)),4),np.tile(action.mean((0,1)),4)]
        std=np.r_[np.tile(obs.std((0,1)),4),np.tile(action.std((0,1)),4)]
    std[std<1e-6]=1.
    return mean,std
