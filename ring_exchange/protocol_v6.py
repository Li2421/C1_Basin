"""v6 rotation-local ScenarioProtocol adapter, isolated from world-frame v1."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .environment import LocalFrameConfig, RingExchangeEnv
from .expert import circulation_signature, infer_circulation_hypothesis
from .expert_v6 import LocalFrameCentralizedExpert
from .local_frame import local_actions_to_world
from .protocol import ExpertContinuation, RingExchangeScenario


@dataclass
class RingExchangeLocalFrameScenario(RingExchangeScenario):
    """Physical state protocol with v6 local observations/actions only."""
    config: LocalFrameConfig = field(default_factory=LocalFrameConfig)
    name: str = "ring_exchange_local_frame_v6"

    def expert(self, initial_state, rng: np.random.Generator) -> ExpertContinuation:
        state=self._state_dict(initial_state); env=RingExchangeEnv(self.config)
        env.reset(state["positions"],velocities=state["velocities"],goals=state["goals"])
        inferred=infer_circulation_hypothesis(state["positions"],state["velocities"])
        try:
            planner=LocalFrameCentralizedExpert()
            if bool(initial_state.get("recovery",False)) and inferred is not None:
                plan=planner.plan_hypothesis(env,inferred); selection="velocity_inferred_continuation"
            else:
                plan=planner.plan(env); selection="shortest_joint_hypothesis_search"
        except RuntimeError:
            return ExpertContinuation(False,"no_valid_recovery",np.empty((0,4,2)),np.empty((0,4,23),dtype=np.float32),np.empty((0,4,2)),{})
        return ExpertContinuation(plan.success,plan.terminal_reason,plan.positions,plan.observations,plan.actions,
                                  {**plan.metadata,"realized_circulation":circulation_signature(plan.positions),
                                   "expert_mode_selection":selection,"representation":"radial_tangential_local_v6"})

    def recovery_state_at(self, trajectory, time: int) -> dict:
        index=int(time); positions=np.asarray(trajectory.states,dtype=np.float64); actions=np.asarray(trajectory.actions,dtype=np.float64)
        if index<0 or index>=len(positions): raise IndexError("recovery time outside trajectory")
        initial=self._state_dict(trajectory.initial_state)
        velocity=np.zeros((4,2)) if index==0 else local_actions_to_world(actions[index-1],positions[index-1])
        return {"positions":positions[index].copy(),"velocities":velocity,"goals":initial["goals"].copy(),
                "split":initial.get("split",trajectory.split),"recovery":True,"source_time":index}


ScenarioV6=RingExchangeLocalFrameScenario
__all__=("RingExchangeLocalFrameScenario","ScenarioV6")
