"""Identical nominal optimizers with an exact-zero interior wall-cost shortcut."""
from controllers_interior import Nominal
from certified_planning import Planner


class InteriorNominal(Nominal):
    def action(self, env, key):
        if self.kind in ('prioritized_astar', 'conflict_based_search'):
            if not hasattr(self, 'planner'):
                self.planner = Planner(self.row)
            return self.planner.action(env)
        return super().action(env, key)
