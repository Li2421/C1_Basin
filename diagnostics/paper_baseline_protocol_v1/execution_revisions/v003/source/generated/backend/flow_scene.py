"""Native v1 plant with the specified unbounded crowded-scene geometry."""
from experiment import *
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from states import config,walls
from legacy_flow_scene import LiteratureGeometry as OriginalGeometry
from legacy_flow_scene import LiteratureEnv as OriginalEnv
from legacy_flow_scene import validate as original_validate,physical12

class LiteratureGeometry(OriginalGeometry):
    def valid_points(self,p):
        if self.config.scene=='lit_crowded_random':return np.isfinite(np.asarray(p)).all(-1)
        return super().valid_points(p)

class LiteratureEnv(OriginalEnv):
    def __init__(self,state,c=None):
        self.config=c or replace(config(state['scene'],state['N']),num_agents=state['N'])
        self.num_agents=state['N'];self.action_shape=(self.num_agents,2)
        self.instance=SimpleNamespace(config=self.config,geometry=LiteratureGeometry(self.config),
            positions=np.asarray(state['positions'],float),goals=np.asarray(state['goals'],float))
        self.walls=self.instance.geometry.walls
        self._i,self._j=np.triu_indices(self.num_agents,1)
        self.pair_indices=tuple(zip(self._i,self._j))
        self.reset()

def validate(s):
    if s['scene']!='lit_crowded_random':return original_validate(s)
    c=config(s['scene'],s['N']);p=np.asarray(s['positions']);g=np.asarray(s['goals'])
    ii,jj=np.triu_indices(s['N'],1)
    starts=bool(np.isfinite(p).all() and np.all(np.linalg.norm(p[ii]-p[jj],axis=-1)>2*c.agent_radius))
    goals=bool(np.isfinite(g).all() and np.all(np.linalg.norm(g[ii]-g[jj],axis=-1)>=2*c.agent_radius))
    return dict(scene=s['scene'],N=s['N'],index=s['index'],valid_initial=starts,valid_goals=goals,
                single_robot_paths=[True]*s['N'],path_proof='Straight segments in the obstacle-free plane',
                capacity_lower_bound=s['N'],capacity_enough=True,unbounded_capacity=True,
                capacity_proof='N points on a line with spacing greater than the physical diameter',
                joint_solvable='not_certified',eligible=starts and goals)
