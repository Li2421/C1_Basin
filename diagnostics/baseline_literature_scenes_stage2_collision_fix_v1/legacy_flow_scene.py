"""New scene adapters. Protected physical and projection files remain read-only."""
from experiment import *
from states import config,walls
from types import SimpleNamespace
from dataclasses import asdict,replace
import numpy as np
from scipy.ndimage import label
from scipy.optimize import nnls
from single_integrator.environment import point_segment_distance,segment_distance
from bottleneck_family.environment import BottleneckEnv
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig

class LiteratureGeometry:
    def __init__(self,c):
        self.config=c;self.walls=walls(c.scene);self.rectangles=np.asarray([[.9,.3,1.1,1.1],[.9,-1.,1.1,-.2]]) if c.scene=='lit_doorway' else np.zeros((0,4))
        self.margin=c.agent_radius+c.wall_radius+c.wall_collision_margin
    def valid_points(self,p):
        p=np.asarray(p);valid=np.isfinite(p).all(-1)
        if self.config.scene=='lit_crowded_random':valid&=(p>self.margin).all(-1)&(p<2-self.margin).all(-1)
        if self.config.scene=='lit_circle_five':
            normals=np.stack([np.cos(np.arange(128)*2*np.pi/128),np.sin(np.arange(128)*2*np.pi/128)],-1)
            valid&=(p@normals.T<11-self.margin).all(-1)
        for x0,y0,x1,y1 in self.rectangles:
            valid&=~((p[...,0]>=x0)&(p[...,0]<=x1)&(p[...,1]>=y0)&(p[...,1]<=y1))
        if len(self.walls):valid&=(point_segment_distance(p[...,None,:],self.walls[:,0],self.walls[:,1])>self.margin).all(-1)
        return valid
    def visible(self,a,b):
        return self.valid_points(np.asarray([a,b])).all() and (not len(self.walls) or np.min(segment_distance(a,b,self.walls[:,0],self.walls[:,1]))>self.margin)

class LiteratureEnv(BottleneckEnv):
    """Use the native plant; extend its empty wall/pair reductions explicitly."""
    def __init__(self,state,c=None):
        self.config=c or replace(config(state['scene'],state['N']),num_agents=state['N']);self.num_agents=state['N'];self.action_shape=(self.num_agents,2)
        self.instance=SimpleNamespace(config=self.config,geometry=LiteratureGeometry(self.config),positions=np.asarray(state['positions'],float),goals=np.asarray(state['goals'],float))
        self.walls=self.instance.geometry.walls;self._i,self._j=np.triu_indices(self.num_agents,1);self.pair_indices=tuple(zip(self._i,self._j))
        self.reset()
    def reset(self,positions=None,goals=None):
        # Goal centers are task data, not simultaneous occupied disks. Preserve
        # exact paper tangent goals; swept-step collision tests stay unchanged.
        p=np.array(self.instance.positions if positions is None else positions,dtype=float,copy=True)
        g=np.array(self.instance.goals if goals is None else goals,dtype=float,copy=True)
        for points,initial in ((p,True),(g,False)):
            if points.shape!=self.action_shape or not self.instance.geometry.valid_points(points).all():raise ValueError('invalid start/goal geometry')
            sep=np.linalg.norm(points[self._i]-points[self._j],axis=1);diameter=2*self.config.agent_radius+self.config.agent_collision_margin
            if np.any(sep<=diameter) if initial else np.any(sep<diameter):raise ValueError('overlapping start/goal disks')
        self.positions,self.goals=p,g;self.velocities=np.zeros(self.action_shape);self.step_count=0;self.done=False;self.termination='running';self.collided=False;self.history=[p.copy()]
        return self.observation()
    def step(self,velocity,*,diagnose_stalls=False):
        if len(self.walls) and len(self._i):return super().step(velocity,diagnose_stalls=diagnose_stalls)
        # Literal original plant, with minimum(empty)=+inf for absent obstacles.
        if self.done:raise RuntimeError('episode already terminated; reset before stepping')
        u=np.asarray(velocity,dtype=float)
        if u.shape!=self.action_shape or not np.isfinite(u).all():raise ValueError('velocity must be finite [N,2]')
        if np.max(np.linalg.norm(u,axis=1))>self.config.max_speed+1e-10:raise ValueError('speed exceeds max_speed')
        c=self.config;before=self.positions.copy();after=before+c.dt*u
        wall_clearance=float(segment_distance(before[:,None],after[:,None],self.walls[None,:,0],self.walls[None,:,1]).min()-self.instance.geometry.margin) if len(self.walls) else np.inf
        relative_before=before[self._i]-before[self._j];relative_after=after[self._i]-after[self._j]
        pair_clearance=float(point_segment_distance(np.zeros(2),relative_before,relative_after).min()-2*c.agent_radius-c.agent_collision_margin) if len(self._i) else np.inf
        collision=bool(wall_clearance<=0 or pair_clearance<=0 or not self.instance.geometry.valid_points(after).all())
        self.positions,self.velocities=after,u.copy();self.step_count+=1;self.history.append(after.copy());self.collided|=collision
        success=bool(np.all(np.linalg.norm(self.goals-after,axis=1)<=c.goal_tolerance))
        self.termination='collision' if collision else 'success' if success else 'timeout' if self.step_count>=c.max_steps else 'running';self.done=self.termination!='running'
        return self.observation(),self.done,dict(step=self.step_count,termination=self.termination,collision=collision,task_success=success and not self.collided,swept_clearance=float(min(wall_clearance,pair_clearance)),min_swept_wall_clearance=wall_clearance,min_swept_agent_clearance=pair_clearance,positions=after.copy(),velocities=u.copy(),stalled_groups=[])

def validate(s):
    c=config(s['scene'],s['N']);geom=LiteratureGeometry(c);p=np.asarray(s['positions']);g=np.asarray(s['goals']);ii,jj=np.triu_indices(len(p),1)
    starts=bool(geom.valid_points(p).all() and np.all(np.linalg.norm(p[ii]-p[jj],axis=-1)>2*c.agent_radius))
    # Equality is physical tangency. It is reported; goal disks need not be
    # occupied at the exact centers to satisfy the inherited arrival tolerance.
    goals=bool(geom.valid_points(g).all() and np.all(np.linalg.norm(g[ii]-g[jj],axis=-1)>=2*c.agent_radius-1e-12))
    if s['scene']=='lit_crowded_random':bounds=(0,2,0,2);resolution=.025
    elif s['scene']=='lit_circle_five':bounds=(-11.1,11.1,-11.1,11.1);resolution=.1
    else:bounds=(min(p[:,0].min(),g[:,0].min())-1,max(p[:,0].max(),g[:,0].max())+1,-2.,2.);resolution=.025
    x=np.arange(bounds[0],bounds[1]+resolution*.5,resolution);y=np.arange(bounds[2],bounds[3]+resolution*.5,resolution)
    xx,yy=np.meshgrid(x,y);grid=geom.valid_points(np.stack([xx,yy],-1));labels,_=label(grid,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    def components(point):
        ix=int(np.argmin(abs(x-point[0])));iy=int(np.argmin(abs(y-point[1])));result=set()
        for jy in range(max(0,iy-1),min(len(y),iy+2)):
            for jx in range(max(0,ix-1),min(len(x),ix+2)):
                if labels[jy,jx] and geom.visible(point,np.array([x[jx],y[jy]])):result.add(int(labels[jy,jx]))
        return result
    paths=[bool(components(a)&components(b)) for a,b in zip(p,g)]
    # Conservative capacity lower bound from explicit nonoverlapping lattice
    # placements in the inflated, bounded window; no joint-solvability claim.
    space=2*c.agent_radius+2*geom.margin*.001+.01
    gx,gy=np.meshgrid(np.arange(bounds[0],bounds[1],space),np.arange(bounds[2],bounds[3],space));slots=np.stack([gx,gy],-1).reshape(-1,2)
    capacity=int(geom.valid_points(slots).sum())
    return dict(scene=s['scene'],N=s['N'],index=s['index'],valid_initial=starts,valid_goals=goals,single_robot_paths=paths,grid_step=resolution,
                capacity_lower_bound=capacity,capacity_enough=capacity>=s['N'],joint_solvable='not_certified',
                tangent_goal_pairs=[[int(i),int(j)] for i,j in zip(ii,jj) if abs(np.linalg.norm(g[i]-g[j])-2*c.agent_radius)<1e-10],
                eligible=bool(starts and goals and all(paths) and capacity>=s['N']))

def physical12(state):
    c=config(state['scene'],state['N']);p=np.asarray(state['positions']);g=np.asarray(state['goals']);v=np.asarray(state['velocities'])
    L=max(2*np.linalg.norm(g-g.mean(0),axis=-1).max(),2*c.agent_radius);d=np.linalg.norm(g-p,axis=-1)/L
    ii,jj=np.triu_indices(len(p),1);pair=np.linalg.norm(p[ii]-p[jj],axis=-1)/(2*c.agent_radius)
    ww=walls(c.scene)
    wall=(point_segment_distance(p[:,None],ww[:,0],ww[:,1]).min(1)-c.agent_radius-c.wall_radius)/c.agent_radius if len(ww) else np.full(len(p),np.nan)
    speed=np.linalg.norm(v,axis=-1)/c.max_speed
    return np.asarray([len(p),d.mean(),d.std(),d.min(),d.max(),pair.min() if len(pair) else np.nan,pair.mean() if len(pair) else np.nan,np.min(wall),np.mean(wall),speed.mean(),speed.max(),L/c.agent_radius],np.float32)

def main(scene=None):
    prereg();states=read(HERE/'FLOW_STATES_UNCHECKED.json');rows=[];allowed=[]
    if scene:states=[s for s in states if s['scene']==scene]
    for s in states:
        v=validate(s);rows.append(v)
        if v['eligible']:allowed.append(s)
    suffix='_'+scene if scene else ''
    write(HERE/f'GEOMETRY_VALIDATION{suffix}.json',rows);write(HERE/f'FLOW_STATES{suffix}.json',allowed)
    seal(f'GEOMETRY_VALIDATION{suffix}.receipt.json',[HERE/f'GEOMETRY_VALIDATION{suffix}.json',HERE/f'FLOW_STATES{suffix}.json',HERE/'flow_scene.py'],before_flow_rollouts=True)
    print('eligible',len(allowed),'rejected',len(states)-len(allowed))

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else None)
