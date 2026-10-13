"""A single map-aware controller program shared by all four physical domains.

No Flow query, learned checkpoint weights, future trajectory or scene-specific
navigation rule is used. Only the existing adapter supplies physical obstacles.
FF integration, eta lifting, safety and the simulator remain inherited.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse.csgraph import shortest_path
from single_integrator.environment import point_segment_distance,segment_distance
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
from diagnostics.physical_state_critic_formal_v1.physical_entities import templates

FAMILIES=('isotropic','rotate_positive','rotate_negative')


class VisibilityNavigation:
    def __init__(self,physical_obstacles,radius,wall_margin):
        self.obstacles=physical_obstacles
        self.radius=radius
        # Match the physical free-space boundary. An extra blanket clearance
        # incorrectly rejects valid starts near Toy's narrow corridor walls.
        # Roadmap vertices themselves retain the extra .01 offset below.
        self.margin=wall_margin
        nodes=[]
        for o in self.obstacles:
            if o['curvature']<0:continue
            sides=32 if o['curvature'] else 8
            directions=np.column_stack((np.cos(np.arange(sides)*2*np.pi/sides),np.sin(np.arange(sides)*2*np.pi/sides)))
            clearance=o['radius']+radius+self.margin
            offset=(clearance+.01)/np.cos(np.pi/sides)
            for end in (o['a'],o['b']):nodes.extend(np.asarray(end)+offset*directions)
        nodes=np.unique(np.round(nodes,10),axis=0)
        self.nodes=nodes[self.visible(nodes,nodes)]
        a,b=np.broadcast_arrays(self.nodes[:,None],self.nodes[None,:])
        good=self.visible(a,b)
        graph=np.where(good,np.linalg.norm(a-b,axis=-1),np.inf)
        np.fill_diagonal(graph,0)
        self.distance=shortest_path(graph,directed=False)

    def visible(self,a,b):
        a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
        ok=np.ones(a.shape[:-1],bool)
        for o in self.obstacles:
            center=np.asarray(o['a']);required=o['radius']+self.radius+self.margin
            if o['curvature']>0:
                ok &= point_segment_distance(center,a,b)>=required-1e-10
            elif o['curvature']<0:
                radius=o['radius']-self.radius-self.margin
                ok &= np.maximum(np.linalg.norm(a-center,axis=-1),np.linalg.norm(b-center,axis=-1))<=radius+1e-10
            else:
                ok &= segment_distance(a,b,np.asarray(o['a']),np.asarray(o['b']))>=required-1e-10
        return ok

    @lru_cache(maxsize=256)
    def goal_distance(self,goal_tuple):
        goal=np.asarray(goal_tuple);edge=np.linalg.norm(self.nodes-goal,axis=-1)
        edge=np.where(self.visible(self.nodes,goal),edge,np.inf)
        return np.min(self.distance+edge[None,:],axis=1)

    def waypoint(self,position,goal):
        if self.visible(position,goal):return goal,True
        distances=self.goal_distance(tuple(goal))
        direct=np.linalg.norm(self.nodes-position,axis=-1)
        costs=np.where(self.visible(position,self.nodes)&(direct>.015),direct+distances,np.inf)
        best=int(np.argmin(costs))
        if not np.isfinite(costs[best]):return goal,False
        return self.nodes[best],True


class NavigationEndpointField(EndpointMatchedField):
    def __init__(self,base,family):
        assert family in FAMILIES
        super().__init__(base,family)
        n=int(self.agent.config['joint_act_dim'])//2
        # Shared physical noise law, avoiding a different learned mean/scale
        # and a different policy chart in every source/target domain.
        self.mean=np.zeros((n,2));self.scale=np.full((n,2),self.config.max_speed)
        obstacles=templates()[self.scenario]['obstacles']
        self.navigator=VisibilityNavigation(obstacles,self.config.agent_radius,
            getattr(self.config,'wall_collision_margin',getattr(self.config,'collision_margin',0.)))
        self.unroutable_queries=0

    def to_world(self,a,env):return np.asarray(a)
    def to_local(self,a,env):return np.asarray(a)

    def target(self,env):
        waypoints=[]
        for p,g in zip(env.positions,env.goals):
            target,ok=self.navigator.waypoint(p,g)
            self.unroutable_queries+=int(not ok);waypoints.append(target)
        delta=np.asarray(waypoints)-env.positions;dist=np.linalg.norm(delta,axis=1,keepdims=True)
        direction=delta/np.maximum(dist,1e-12)
        remaining=np.linalg.norm(env.goals-env.positions,axis=1,keepdims=True)
        return .85*self.config.max_speed*np.tanh(remaining/.5)*direction
