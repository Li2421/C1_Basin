"""Shared physical entities; native parsers have no learned parameters."""
import json,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

ROOT=Path(__file__).resolve().parents[2]
VERSION='physical_entities_goal_frames_v1'
LENGTH=4.;SPEED=.82

def decode(x):
    while isinstance(x,str):x=json.loads(x)
    return x

def parse(row):
    """Only native geometry/controller semantics branch here, never learned routing."""
    sc=row['scenario'];p=decode(row['structured_state']);cond=decode(row['conditioning']);desc=decode(row['environment_descriptor'])
    if sc=='double_bottleneck':
        cfg=desc
        from double_bottleneck.scenario import build_layout
        walls=build_layout(SimpleNamespace(**cfg)).walls
        assert int(p['timestep'])==0,'Intermediate Double needs full monitor snapshot, not fabricated history'
    else:
        path={'four_way_intersection':'four_way_intersection_stage1/base_u_v13_broad_global_dataset',
              'ring_exchange':'ring_exchange_stage1/base_u_v10_local_dataset'}[sc]
        cfg=json.loads((ROOT/'diagnostics'/path/'manifest.json').read_text())['scenario_config']
        if sc=='four_way_intersection':
            from four_way_intersection.scenario import build_layout
            walls=build_layout(SimpleNamespace(**cfg)).walls
        else:walls=[]
    n=len(p['positions']);flow=np.asarray(cond['flat'][-2*n:],float).reshape(n,2)
    obstacles=[{'a':np.asarray(w[0]).tolist(),'b':np.asarray(w[1]).tolist(),'radius':float(cfg.get('wall_radius',0.)),
                'curvature':0.,'interior_sign':0.} for w in walls]
    if sc=='ring_exchange':
        center=desc['obstacle_center']
        obstacles=[{'a':center,'b':center,'radius':float(cfg['obstacle_radius']),'curvature':1./cfg['obstacle_radius'],'interior_sign':1.},
                   {'a':center,'b':center,'radius':float(cfg['outer_radius']),'curvature':-1./cfg['outer_radius'],'interior_sign':-1.}]
    axis_required=sc!='ring_exchange'
    normalized_time=float(p.get('normalized_episode_time',int(p['timestep'])/cfg['max_steps']))
    assert abs(normalized_time-int(p['timestep'])/cfg['max_steps'])<1e-9
    return {'positions':p['positions'],'velocities':p['velocities'],'goals':p['goals'],'flow':flow.tolist(),
        'radius':[cfg['agent_radius']]*n,'goal_tolerance':[cfg['goal_tolerance']]*n,'obstacles':obstacles,
        'policy_origin':[0.,0.],'policy_axis':[1.,0.],'policy_axis_required':axis_required,
        'flow_committed':sc=='double_bottleneck','remaining_fraction':1.-normalized_time,
        'remaining_seconds':(cfg['max_steps']-int(p['timestep']))*cfg['dt'],
        'dt':cfg['dt'],'max_speed':cfg['max_speed'],'max_seconds':cfg['max_steps']*cfg['dt'],
        'wall_margin':cfg.get('wall_collision_margin',cfg.get('collision_margin',0.)),
        'agent_margin':cfg.get('agent_collision_margin',0.),
        'monitor_active':bool(cfg.get('terminate_on_deadlock',False)),
        'progress_window_seconds':cfg.get('progress_window_seconds',0.),'deadlock_hold_seconds':cfg.get('deadlock_hold_seconds',0.),
        'progress_epsilon':cfg.get('progress_epsilon',0.),'speed_epsilon_fraction':cfg.get('speed_epsilon_fraction',0.),
        'monitor_elapsed':0.,'monitor_history_empty':True}

def transform(scene,angle=0.,translation=(0.,0.),permutation=None,obstacle_permutation=None,passive=True):
    s=copy.deepcopy(scene);c,t=np.cos(angle),np.sin(angle);R=np.array([[c,-t],[t,c]]);shift=np.asarray(translation)
    for k in ('positions','goals'):s[k]=(np.asarray(s[k])@R.T+shift).tolist()
    for k in ('velocities','flow'):s[k]=(np.asarray(s[k])@R.T).tolist()
    for o in s['obstacles']:
        for k in ('a','b'):o[k]=(np.asarray(o[k])@R.T+shift).tolist()
    if passive:
        s['policy_origin']=(np.asarray(s['policy_origin'])@R.T+shift).tolist()
        s['policy_axis']=(np.asarray(s['policy_axis'])@R.T).tolist()
    if permutation is not None:
        for k in ('positions','goals','velocities','flow','radius','goal_tolerance'):s[k]=np.asarray(s[k])[permutation].tolist()
    if obstacle_permutation is not None:s['obstacles']=[s['obstacles'][i] for i in obstacle_permutation]
    return s

def entities(scene):
    p,v,g,f=(np.asarray(scene[k],float) for k in ('positions','velocities','goals','flow'))
    n=len(p);goal=g-p;distance=np.linalg.norm(goal,axis=-1);axes=[]
    for i in range(n):
        direction=goal[i]
        if np.linalg.norm(direction)<1e-10:direction=v[i]
        if np.linalg.norm(direction)<1e-10:direction=f[i]
        if np.linalg.norm(direction)<1e-10:direction=p[i]-np.asarray(scene['policy_origin'])
        if np.linalg.norm(direction)<1e-10:direction=np.asarray(scene['policy_axis'])
        e=direction/np.linalg.norm(direction);axes.append(np.stack((e,[-e[1],e[0]])))
    axes=np.asarray(axes);local=lambda x:np.einsum('nij,nj->ni',axes,np.asarray(x))
    bounded=f*np.minimum(1.,scene['max_speed']/np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-30))
    mask=float(scene['policy_axis_required']);r=np.asarray(scene['radius']);tol=np.asarray(scene['goal_tolerance'])
    own=np.column_stack((distance/LENGTH,local(v)/SPEED,local(f)/SPEED,local(bounded)/SPEED,
        r/.2,tol/.1,(distance<=tol).astype(float),np.full(n,mask),
        mask*local(np.asarray(scene['policy_origin'])-p)/LENGTH,
        mask*local(np.tile(scene['policy_axis'],(n,1)))))
    pairs=np.zeros((n,n,9))
    for i in range(n):
        for j in range(n):
            delta=p[j]-p[i];d=np.linalg.norm(delta)
            pairs[i,j]=np.r_[axes[i]@delta/LENGTH,d/LENGTH,axes[i]@(v[j]-v[i])/SPEED,
                axes[i]@(g[j]-p[i])/LENGTH,(d-r[i]-r[j])/LENGTH,(r[i]+r[j])/.4]
    obs=np.zeros((n,len(scene['obstacles']),13))
    for i in range(n):
        for j,o in enumerate(scene['obstacles']):
            a,z=np.asarray(o['a']),np.asarray(o['b']);curve=o['curvature'];rad=o['radius']
            if curve:
                delta=p[i]-a;norm=np.linalg.norm(delta);normal=delta/max(norm,1e-12)*o['interior_sign']
                closest=a+rad*delta/max(norm,1e-12);clear=o['interior_sign']*(norm-rad)-r[i]
                extent=2*rad
            else:
                delta=z-a;u=np.clip(np.dot(p[i]-a,delta)/max(np.dot(delta,delta),1e-30),0,1)
                closest=a+u*delta;norm=np.linalg.norm(p[i]-closest);normal=(p[i]-closest)/max(norm,1e-12)
                clear=norm-rad-r[i];extent=np.linalg.norm(delta)
            endpoints=sorted([(axes[i]@(a-p[i])/LENGTH).tolist(),(axes[i]@(z-p[i])/LENGTH).tolist()],key=lambda x:tuple(np.round(x,12)))
            obs[i,j]=np.r_[axes[i]@(closest-p[i])/LENGTH,clear/LENGTH,axes[i]@normal,
                          curve*LENGTH,rad/LENGTH,extent/LENGTH,*endpoints[0],*endpoints[1],float(bool(curve))]
    glob=np.array([scene['remaining_fraction'],scene['remaining_seconds']/60.,scene['max_seconds']/60.,
        scene['dt']/.05,scene['max_speed']/SPEED,scene['flow_committed'],scene['wall_margin']/.01,
        scene['agent_margin']/.01,scene['monitor_active'],scene['progress_window_seconds']/5.,
        scene['deadlock_hold_seconds']/5.,scene['progress_epsilon']/.01,scene['speed_epsilon_fraction'],
        scene['monitor_elapsed']/5.,scene['monitor_history_empty']],float)
    result={'agents':own,'pairs':pairs,'obstacles':obs,'globals':glob,
            'agent_mask':np.ones(n),'obstacle_mask':np.ones(len(scene['obstacles']))}
    assert all(np.isfinite(v).all() for v in result.values())
    return {k:v.astype(np.float32) for k,v in result.items()}

def batch(items):
    """Padding is masked and shape-only; no padding position has an identity."""
    n=max(len(x['agents']) for x in items);m=max(1,max(len(x['obstacle_mask']) for x in items));out={}
    for key in items[0]:
        values=[]
        for x in items:
            v=x[key]
            if key=='agents':pad=((0,n-len(v)),(0,0))
            elif key=='pairs':pad=((0,n-len(v)),(0,n-len(v)),(0,0))
            elif key=='obstacles':pad=((0,n-len(v)),(0,m-v.shape[1]),(0,0))
            elif key=='agent_mask':pad=((0,n-len(v)),)
            elif key=='obstacle_mask':pad=((0,m-len(v)),)
            else:pad=None
            values.append(np.pad(v,pad) if pad is not None else v)
        out[key]=np.stack(values)
    return out

def pool(x,mask,axis):
    mask=mask[...,None];mean=jnp.sum(x*mask,axis=axis)/jnp.maximum(jnp.sum(mask,axis=axis),1.)
    maximum=jnp.max(jnp.where(mask>0,x,-1e9),axis=axis)
    maximum=jnp.where(jnp.sum(mask,axis=axis)>0,maximum,0.)
    return jnp.concatenate((mean,maximum),axis=-1)

class Encoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        am=x['agent_mask'];n=am.shape[-1]
        pm=am[:,:,None]*am[:,None,:]*(1-jnp.eye(n)[None])
        pair=nn.silu(nn.Dense(32,name='pair1')(x['pairs']));pair=nn.silu(nn.Dense(32,name='pair2')(pair))
        message=pool(pair,pm,2)
        om=am[:,:,None]*x['obstacle_mask'][:,None,:]
        ob=nn.silu(nn.Dense(32,name='obstacle1')(x['obstacles']));ob=nn.silu(nn.Dense(32,name='obstacle2')(ob))
        geometry=pool(ob,om,2)
        own=nn.silu(nn.Dense(32,name='agent1')(x['agents']))
        agent=nn.silu(nn.Dense(64,name='agent2')(jnp.concatenate((own,message,geometry),axis=-1)))
        scene=pool(agent,am,1)
        count=jnp.stack((jnp.log1p(am.sum(-1)),jnp.log1p(x['obstacle_mask'].sum(-1))),axis=-1)
        return nn.silu(nn.Dense(128,name='scene')(jnp.concatenate((scene,x['globals'],count),axis=-1)))

class Generator(nn.Module):
    @nn.compact
    def __call__(self,x):
        h=Encoder(name='physical_encoder')(x)
        z=nn.silu(nn.Dense(128,name='shared1')(h));z=nn.silu(nn.Dense(64,name='shared2')(z))
        return nn.Dense(6,name='out')(z)

class Critic(nn.Module):
    @nn.compact
    def __call__(self,x,eta):
        h=Encoder(name='physical_encoder')(x);e=nn.silu(nn.Dense(32,name='eta_encoder')(eta))
        z=nn.silu(nn.Dense(128,name='shared1')(jnp.concatenate((h,e),axis=-1)))
        z=nn.silu(nn.Dense(64,name='shared2')(z));return nn.Dense(1,name='out')(z)[...,0]
