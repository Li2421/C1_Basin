"""Only layout changes; inherit the exact original swept physical step."""
import bootstrap
from experiment import *
from swap_geometry import SwapEnv as OriginalEnv,Geometry as OriginalGeometry,config,moving_path,LiteratureConfig
from types import SimpleNamespace
from functools import lru_cache
import numpy as np

@lru_cache(None)
def layout():return read(HERE/'LAYOUT_REVISION_DESIGN.json')
def static_walls(scene):
    return np.asarray(layout()['walls'],float) if scene=='lit_swap8_obstacles' else np.zeros((0,2,2))
class GeometryRevision(OriginalGeometry):
    def __init__(self,c):self.config=c;self.walls=static_walls(c.scene);self.margin=c.agent_radius+c.wall_radius+c.wall_collision_margin
class SwapEnvRevision(OriginalEnv):
    def __init__(self,state,c=None):
        self.config=c or config(state['scene'],state['N']);self.num_agents=state['N'];self.action_shape=(self.num_agents,2)
        self.positions=np.asarray(state['positions'],float).copy();self.goals=np.asarray(state['goals'],float).copy();self.velocities=np.zeros_like(self.positions)
        geom=GeometryRevision(self.config);self.instance=SimpleNamespace(config=self.config,geometry=geom,positions=self.positions.copy(),goals=self.goals.copy());self.walls=geom.walls
        assert geom.valid_points(self.positions).all() and geom.valid_points(self.goals).all()
        self._i,self._j=np.triu_indices(self.num_agents,1);self.pair_indices=tuple(zip(self._i,self._j));self.step_count=0;self.collided=False;self.done=False
        self.mover,self.mover_pieces=moving_path(state,self.config.max_steps) if state['scene']=='lit_swap8_moving' else (None,None)
        self.collision_types=dict(agent=False,static=False,moving=False)
