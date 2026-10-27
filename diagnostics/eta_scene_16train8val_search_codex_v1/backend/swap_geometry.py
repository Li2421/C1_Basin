"""Declared planar geometry; static adapters expose finite segments to frozen actors."""
from experiment import *
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from legacy_states import LiteratureConfig
from single_integrator.environment import point_segment_distance,segment_distance

def static_walls(scene):
    walls=[]
    if scene=='lit_swap8_obstacles':
        for theta in np.deg2rad([22.5,112.5,202.5,292.5]):
            e=np.array([np.cos(theta),np.sin(theta)]);f=np.array([-e[1],e[0]]);corner=.7*e
            for sign in (-1,1):walls.append([corner,corner+.5*(-e+sign*f)/np.sqrt(2)])
    return np.asarray(walls,float).reshape(-1,2,2)

def config(scene,n=8):
    radius=1.4 if scene=='lit_swap8_obstacles' else 1.
    return LiteratureConfig(scene,n,.05,.05,.5,int(round((4*2*radius/.5)/.05)),wall_collision_margin=.005)

class Geometry:
    def __init__(self,c):self.config=c;self.walls=static_walls(c.scene);self.margin=c.agent_radius+c.wall_radius+c.wall_collision_margin
    def valid_points(self,p):
        p=np.asarray(p);ok=np.isfinite(p).all(-1)
        if len(self.walls):ok&=(point_segment_distance(p[...,None,:],self.walls[:,0],self.walls[:,1])>self.margin).all(-1)
        return ok
    def visible(self,a,b):return bool(self.valid_points(np.array([a,b])).all() and (not len(self.walls) or np.min(segment_distance(a,b,self.walls[:,0],self.walls[:,1]))>self.margin))

def moving_path(state,steps):
    # Parent robot identity is independent of controller, seed and solo slicing.
    identity=state.get('parent_physical_identity',uid('swap_robot_state',dict(index=state['index'],positions=state['positions'],goals=state['goals'])))
    rng=np.random.default_rng(np.random.SeedSequence([2026102313,int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)]))
    initial=np.asarray(state.get('parent_positions',state['positions']))
    while True:
        q=rng.uniform(-1.5,1.5,2)
        if np.all(np.linalg.norm(initial-q,axis=-1)>=.45):break
    points=[q.copy()];segments=[];target=rng.uniform(-1.5,1.5,2)
    for t in range(steps):
        left=.05;elapsed=0.;pieces=[]
        while left>1e-14:
            distance=np.linalg.norm(target-q)
            if distance<=1e-12:target=rng.uniform(-1.5,1.5,2);continue
            duration=min(left,distance/.25);end=q+(target-q)*(.25*duration/distance)
            pieces.append([elapsed,elapsed+duration,*q,*end]);q=end;elapsed+=duration;left-=duration
            if distance/.25<=duration+1e-14:target=rng.uniform(-1.5,1.5,2)
        segments.append(pieces);points.append(q.copy())
    return np.asarray(points),segments

class SwapEnv:
    def __init__(self,state,c=None):
        self.config=c or config(state['scene'],state['N']);self.num_agents=state['N'];self.action_shape=(self.num_agents,2)
        self.positions=np.asarray(state['positions'],float).copy();self.goals=np.asarray(state['goals'],float).copy();self.velocities=np.zeros_like(self.positions)
        geom=Geometry(self.config);self.instance=SimpleNamespace(config=self.config,geometry=geom,positions=self.positions.copy(),goals=self.goals.copy());self.walls=geom.walls
        assert geom.valid_points(self.positions).all() and geom.valid_points(self.goals).all()
        self._i,self._j=np.triu_indices(self.num_agents,1);self.pair_indices=tuple(zip(self._i,self._j));self.step_count=0;self.collided=False;self.done=False
        self.mover,self.mover_pieces=moving_path(state,self.config.max_steps) if state['scene']=='lit_swap8_moving' else (None,None)
        self.collision_types=dict(agent=False,static=False,moving=False)
    def snapshot(self):return dict(positions=self.positions.copy(),velocities=self.velocities.copy(),goals=self.goals.copy(),walls=self.walls,config=self.config.to_dict(),moving_position=None if self.mover is None else self.mover[self.step_count].copy())
    def step(self,u,**kw):
        c=self.config;u=np.asarray(u,float);assert u.shape==self.action_shape and np.isfinite(u).all()
        before=self.positions.copy();after=before+c.dt*u
        pair=float(point_segment_distance(np.zeros(2),before[self._i]-before[self._j],after[self._i]-after[self._j]).min()-2*c.agent_radius-c.agent_collision_margin) if len(self._i) else np.inf
        wall=float(segment_distance(before[:,None],after[:,None],self.walls[None,:,0],self.walls[None,:,1]).min()-self.instance.geometry.margin) if len(self.walls) else np.inf
        moving=np.inf
        if self.mover is not None:
            for ta,tb,ax,ay,bx,by in self.mover_pieces[self.step_count]:
                a=before+ta*u-np.array([ax,ay]);b=before+tb*u-np.array([bx,by]);moving=min(moving,float(point_segment_distance(np.zeros(2),a,b).min()-c.agent_radius-.3))
        hits=dict(agent=pair<=0,static=wall<=0,moving=moving<=0)
        for k,v in hits.items():self.collision_types[k]|=v
        self.collided|=any(hits.values());self.positions=after;self.velocities=u.copy();self.step_count+=1
        success=bool(np.all(np.linalg.norm(self.goals-after,axis=-1)<=c.goal_tolerance))
        self.termination='collision' if self.collided else 'success' if success else 'timeout' if self.step_count>=c.max_steps else 'running';self.done=self.termination!='running'
        return None,self.done,dict(termination=self.termination,swept_clearance=min(pair,wall,moving),min_swept_agent_clearance=pair,min_swept_wall_clearance=wall,min_swept_moving_clearance=moving,collision_types=hits)
