"""ScenarioProtocol adapter for v7 lane-local observations/actions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from new_benchmark_common.protocol import ExpertContinuation
from .environment import Config, FourWayIntersectionEnv
from .expert import CentralizedExpert, all_crossing_orders
from .lane_local import REPRESENTATION, FourWayLaneLocalEnv, representation_fingerprint, world_to_local, local_to_world
from .scenario import AGENT_NAMES, draw_initial_state
from .rollout import crossing_order_signature


@dataclass
class FourWayLaneLocalScenario:
    config: Config = Config()
    perturb_position_std: float=.075
    perturb_velocity_std: float=.055
    name: str="four_way_intersection_lane_local_v7"
    agent_order: tuple[str,...]=AGENT_NAMES
    observation_shape: tuple[int,int]=(4,18)
    action_shape: tuple[int,int]=(4,2)

    @property
    def environment_fingerprint(self): return representation_fingerprint(self.config)
    @property
    def representation(self): return REPRESENTATION

    def sample_initial_state(self,split,rng):
        p,v,draw=draw_initial_state(self.config,rng);return {"positions":p,"velocities":v,"split":split,"draw":draw}

    def valid_state(self,state):
        try:
            if isinstance(state,dict):p=np.asarray(state["positions"]);v=np.asarray(state.get("velocities",np.zeros((4,2))))
            else:p=np.asarray(state);v=np.zeros((4,2))
            FourWayIntersectionEnv(self.config).reset(p,v);return True
        except (ValueError,TypeError,KeyError):return False

    def perturb_state(self,state,rng):
        if isinstance(state,dict):p=np.asarray(state["positions"],dtype=np.float64);v=np.asarray(state.get("velocities",np.zeros((4,2))),dtype=np.float64);split=state.get("split","train")
        else:p=np.asarray(state,dtype=np.float64);v=np.zeros((4,2));split="train"
        return {"positions":p+rng.normal(0,self.perturb_position_std,(4,2)),"velocities":v+rng.normal(0,self.perturb_velocity_std,(4,2)),"split":split,"recovery":True}

    def recovery_state_at(self,trajectory,time):
        if not 0<=int(time)<len(trajectory.states):raise ValueError("anchor outside state range")
        # Stored actions are lane-local; restore the physical prior velocity.
        prior=np.zeros((4,2)) if time==0 else local_to_world(np.asarray(trajectory.actions[time-1]))
        return {"positions":np.asarray(trajectory.states[time],dtype=np.float64).copy(),"velocities":prior,"split":trajectory.split,"recovery_anchor_time":int(time)}

    def expert(self,initial_state,rng):
        if isinstance(initial_state,dict):p=np.asarray(initial_state["positions"]);v=np.asarray(initial_state.get("velocities",np.zeros((4,2))))
        else:p=np.asarray(initial_state);v=np.zeros((4,2))
        env=FourWayLaneLocalEnv(self.config);env.reset(p,v);planner=CentralizedExpert()
        try: plan=planner.plan_hypothesis(env,planner._ranked_candidates(env)[0])
        except RuntimeError:
            try:plan=planner.plan(env,all_crossing_orders())
            except RuntimeError:return ExpertContinuation(False,"expert_no_recoverable_crossing_order",np.empty((0,4,2)),np.empty((0,4,18),dtype=np.float32),np.empty((0,4,2)),{"expert":"centralized_crossing_order_search_v1","recovery_failure":"no_valid_order"})
        return ExpertContinuation(plan.success,plan.terminal_reason,plan.positions,plan.observations,world_to_local(plan.actions).astype(np.float32),{**plan.metadata,"realized_crossing_order":crossing_order_signature(plan.positions),"representation":REPRESENTATION})
