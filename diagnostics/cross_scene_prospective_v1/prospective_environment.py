"""New geometry through the frozen eight-agent SI plant and native planning API."""
from common import *
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
from swap_geometry import SwapEnv
from swap_geometry_user_fix_20261023 import GeometryRevision
from nominal_geometry_user_fix_20261023 import Nominal as OriginalNominal
from controllers_prospective import Nominal as GeometryNominal
from bottleneck_family.scenario import Config, Geometry


@lru_cache(None)
def bottleneck_geometry(config):
    return Geometry(Config(**json.loads(config)))


class NewEnv(SwapEnv):
    """Inherit step and snapshot literally; supply only the new fixed geometry."""
    def __init__(self, state, c):
        self.config = c; self.num_agents = state['N']; self.action_shape = (self.num_agents, 2)
        self.positions = np.asarray(state['positions'], float).copy()
        self.goals = np.asarray(state['goals'], float).copy()
        self.velocities = np.asarray(state['velocities'], float).copy()
        if state.get('bottleneck_config'):
            geom = bottleneck_geometry(canonical(state['bottleneck_config']))
        else:
            # Empty-wall revision has the same geometry contract as the old free scene.
            geom = GeometryRevision(c)
        self.instance = SimpleNamespace(config=c, geometry=geom, positions=self.positions.copy(), goals=self.goals.copy())
        self.walls = geom.walls
        assert geom.valid_points(self.positions).all() and geom.valid_points(self.goals).all()
        self._i, self._j = np.triu_indices(self.num_agents, 1)
        self.pair_indices = tuple(zip(self._i, self._j)); self.step_count = 0; self.collided = False; self.done = False
        self.mover = self.mover_pieces = None
        self.collision_types = dict(agent=False, static=False, moving=False)


class NewNominal(OriginalNominal):
    def action(self, env, key):
        if env.config.scene.startswith(('xsc_','xsp_')) and self.kind not in ('prioritized_astar', 'conflict_based_search'):
            return GeometryNominal.action(self, env, key)
        return super().action(env, key)
