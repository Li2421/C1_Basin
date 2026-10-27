"""Read empty walls through a geometry view; nominal algorithms stay frozen."""
from dataclasses import replace
from tiled_nominal import TiledNominal

class EmptyWallControllerView:
    def __init__(self,env):
        self._env=env
        # The historical cost dispatch calls its zero-wall branch 'narrow'.
        # This view only selects that branch; all physical values and arrays
        # remain the actual crowded-scene values. No module global is patched.
        self.config=replace(env.config,scene='lit_narrow_passage')
    def __getattr__(self,name):return getattr(self._env,name)

class Nominal(TiledNominal):
    def action(self,env,key):
        view=EmptyWallControllerView(env) if env.config.scene=='lit_crowded_random' and not len(env.walls) else env
        return super().action(view,key)
