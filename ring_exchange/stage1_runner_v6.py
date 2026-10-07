"""v6 local-frame collection/rollout seam for conventional joint MACFlow."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from new_benchmark_common.dataset import collect_nominal_and_uniform
from new_benchmark_common.protocol import RolloutAdapter
from .environment import LocalFrameConfig, RingExchangeEnv
from .local_frame import local_actions_to_world, local_observation
from .protocol_v6 import RingExchangeLocalFrameScenario


def make_scenario(config: LocalFrameConfig | None=None): return RingExchangeLocalFrameScenario(config or LocalFrameConfig())

def collect_base_u(root: str|Path, *, config: LocalFrameConfig|None=None, nominal_counts=None, uniform_anchors=8, seed=0):
    scenario=make_scenario(config); counts=nominal_counts or {"train":120,"dev":30,"test":60}
    return collect_nominal_and_uniform(scenario,root,scenario_config=asdict(scenario.config),nominal_counts=counts,
                                       uniform_anchors=uniform_anchors,seed=seed)

def rollout_adapter(config: LocalFrameConfig|None=None):
    cfg=config or LocalFrameConfig()
    def reset(env,state): env.reset(state["positions"],velocities=state.get("velocities"),goals=state["goals"])
    def step(env,local_action):
        observation,_,done,info=env.step(local_actions_to_world(local_action,env.positions)); return observation,done,info
    def restore(env,state): env.restore_augmented_state(state)
    def distance(left,right): return float(np.sqrt(np.mean((np.asarray(left["positions"])-np.asarray(right["positions"]))**2)))
    return RolloutAdapter(make_env=lambda:RingExchangeEnv(cfg),reset=reset,
                          observation=lambda env:local_observation(env.positions,env.velocities,env.goals,cfg),step=step,
                          snapshot=lambda env:env.augmented_state(),restore=restore,state_distance=distance)

__all__=("make_scenario","collect_base_u","rollout_adapter")
