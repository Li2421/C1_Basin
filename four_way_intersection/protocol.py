"""Adapter from Four-Way's isolated plant to generic Stage-I data utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from new_benchmark_common.protocol import ExpertContinuation
from .environment import Config, FourWayIntersectionEnv
from .expert import CentralizedExpert
from .scenario import AGENT_NAMES, draw_initial_state
from .rollout import crossing_order_signature


@dataclass
class FourWayScenario:
    """ScenarioProtocol implementation; carries physical state only.

    State deliberately omits crossing order.  ``metadata`` returned by
    :meth:`expert` is retained for analysis/audit but never becomes an input.
    """
    config: Config = Config()
    perturb_position_std: float = .075
    perturb_velocity_std: float = .055
    name: str = "four_way_intersection"
    agent_order: tuple[str,...] = AGENT_NAMES
    observation_shape: tuple[int,int] = (4,18)
    action_shape: tuple[int,int] = (4,2)

    @property
    def environment_fingerprint(self): return self.config.fingerprint

    def sample_initial_state(self, split: str, rng: np.random.Generator):
        p,v,draw=draw_initial_state(self.config,rng)
        return {"positions":p,"velocities":v,"split":split,"draw":draw}

    def valid_state(self,state):
        try:
            if isinstance(state, dict):
                p=np.asarray(state["positions"]); v=np.asarray(state.get("velocities",np.zeros((4,2))))
            else:
                p=np.asarray(state); v=np.zeros((4,2))
            env=FourWayIntersectionEnv(self.config); env.reset(p,v)
            return True
        except (ValueError, TypeError, KeyError): return False

    def perturb_state(self,state,rng):
        if isinstance(state, dict):
            base_p=np.asarray(state["positions"],dtype=np.float64); base_v=np.asarray(state.get("velocities",np.zeros((4,2))),dtype=np.float64); split=state.get("split","train")
        else:
            base_p=np.asarray(state,dtype=np.float64); base_v=np.zeros((4,2)); split="train"
        p=base_p+rng.normal(0,self.perturb_position_std,(4,2)); v=base_v+rng.normal(0,self.perturb_velocity_std,(4,2))
        return {"positions":p,"velocities":v,"split":split,"recovery":True}

    def recovery_state_at(self, trajectory, time: int):
        """Reconstruct the Markov anchor used for recovery re-query.

        Dataset arrays store positions as state samples; the observation's
        reported last-applied velocity is restored from the preceding expert
        action (zero only at the initial state).  This avoids quietly changing
        the Stage-I observation at recovery anchors.
        """
        if not 0 <= int(time) < len(trajectory.states):
            raise ValueError("anchor outside trajectory state range")
        previous=np.zeros((4,2),dtype=np.float64) if time == 0 else np.asarray(trajectory.actions[time-1],dtype=np.float64)
        return {"positions":np.asarray(trajectory.states[time],dtype=np.float64).copy(),
                "velocities":previous.copy(),"split":trajectory.split,"recovery_anchor_time":int(time)}

    def expert(self,initial_state,rng):
        env=FourWayIntersectionEnv(self.config)
        if isinstance(initial_state, dict):
            p=np.asarray(initial_state["positions"]); v=np.asarray(initial_state.get("velocities",np.zeros((4,2))))
        else:
            p=np.asarray(initial_state); v=np.zeros((4,2))
        env.reset(p,v)
        # The centralized search ranks orders by the *physical* approach
        # geometry.  That resolves an otherwise arbitrary action-distribution
        # ambiguity at the exact same state while retaining many realized
        # orders across broad starts; it is neither an environment rule nor a
        # policy input.  Recovery falls back to wider search if needed.
        from .expert import all_crossing_orders
        hypotheses=all_crossing_orders()
        planner=CentralizedExpert()
        try:
            plan=planner.plan_hypothesis(env, planner._ranked_candidates(env)[0])
        except RuntimeError:
            # A local perturbation may invalidate this chosen mode while a
            # different legal crossing order remains recoverable.  Exhaust the
            # expert's mode space before declaring a failed recovery query.
            try:
                plan=planner.plan(env,hypotheses)
            except RuntimeError:
                return ExpertContinuation(False,"expert_no_recoverable_crossing_order",
                                          np.empty((0,4,2)),np.empty((0,4,18),dtype=np.float32),np.empty((0,4,2)),
                                          {"expert":"centralized_crossing_order_search_v1","recovery_failure":"no_valid_order"})
        return ExpertContinuation(plan.success,plan.terminal_reason,plan.positions,plan.observations,plan.actions,
                                  {**plan.metadata,"realized_crossing_order":crossing_order_signature(plan.positions)})
