"""Local-frame action encoding wrapper around the unchanged joint Ring expert."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .expert import CentralizedExpert, CirculationHypothesis, ExpertPlan
from .local_frame import local_observation, world_actions_to_local


class LocalFrameCentralizedExpert:
    """Centralized expert with v6 `[radial,tangential]` trajectory actions."""
    def __init__(self, planner: CentralizedExpert | None = None): self.planner=planner or CentralizedExpert()

    @staticmethod
    def _encode(source_env, plan: ExpertPlan) -> ExpertPlan:
        world_actions=np.asarray(plan.actions,dtype=np.float64)
        positions=np.asarray(plan.positions,dtype=np.float64)
        local_actions=np.stack([world_actions_to_local(world_actions[t],positions[t]) for t in range(len(world_actions))])
        velocities=np.empty_like(positions)
        velocities[0]=source_env.velocities
        velocities[1:]=world_actions
        observations=np.stack([local_observation(positions[t],velocities[t],source_env.goals,source_env.config)
                               for t in range(len(positions))])
        return replace(plan,actions=local_actions,observations=observations)

    def plan(self, env, hypotheses=None, require_success=True) -> ExpertPlan:
        return self._encode(env,self.planner.plan(env,hypotheses,require_success))

    def plan_hypothesis(self, env, hypothesis: CirculationHypothesis) -> ExpertPlan:
        return self._encode(env,self.planner.plan_hypothesis(env,hypothesis))


__all__=("LocalFrameCentralizedExpert",)
