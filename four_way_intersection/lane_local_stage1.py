"""v7 lane-local data/evaluation wiring; no mode-conditioned policy path."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from new_benchmark_common.dataset import collect_nominal_and_uniform
from new_benchmark_common.protocol import RolloutAdapter
from .environment import Config
from .lane_local import FourWayLaneLocalEnv, REPRESENTATION, local_to_world
from .lane_local_protocol import FourWayLaneLocalScenario


def make_scenario(config=None):return FourWayLaneLocalScenario(config or Config())

def collect_base_u(root,*,config=None,nominal_counts=None,uniform_anchors=8,seed=0):
    scenario=make_scenario(config);counts=nominal_counts or {"train":120,"dev":30,"test":60}
    payload={**asdict(scenario.config),"observation_action_representation":REPRESENTATION,"representation_fingerprint":scenario.environment_fingerprint}
    return collect_nominal_and_uniform(scenario,root,scenario_config=payload,nominal_counts=counts,uniform_anchors=uniform_anchors,seed=seed)

def rollout_adapter(config=None):
    cfg=config or Config()
    def reset(env,state):env.reset(state["positions"],state.get("velocities")) if isinstance(state,dict) else env.reset(state)
    def step(env,local_action):
        observation,_,done,info=env.step(local_to_world(local_action));return observation,done,info
    def distance(left,right):return float(np.sqrt(np.mean((np.asarray(left["positions"])-np.asarray(right["positions"]))**2)))
    return RolloutAdapter(make_env=lambda:FourWayLaneLocalEnv(cfg),reset=reset,observation=lambda e:e.observation(),step=step,snapshot=lambda e:e.augmented_state(),restore=lambda e,s:e.restore_augmented_state(s),state_distance=distance)
