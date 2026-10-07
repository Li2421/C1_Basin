"""Thin Four-Way wiring for the shared Stage-I collection/evaluation tools.

No training is performed on import.  In particular this module contains no
OrthoFlow3, eta, basin, G_phi, or mode-conditioned policy path.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from new_benchmark_common.dataset import collect_nominal_and_uniform
from new_benchmark_common.protocol import RolloutAdapter
from .environment import Config, FourWayIntersectionEnv
from .protocol import FourWayScenario


def make_scenario(config: Config | None = None) -> FourWayScenario:
    return FourWayScenario(config or Config())


def collect_base_u(root: str | Path, *, config: Config | None = None,
                   nominal_counts: dict[str,int] | None = None, uniform_anchors: int = 8, seed: int = 0):
    """Broad nominal + uniformly anchored recovery acquisition only.

    The independent test split gets nominal expert trajectories but no recovery
    queries, enforcing the untouched-test rule at the collection boundary.
    """
    scenario=make_scenario(config)
    counts=nominal_counts or {"train":120,"dev":30,"test":60}
    return collect_nominal_and_uniform(scenario,root,scenario_config=asdict(scenario.config),
                                       nominal_counts=counts,uniform_anchors=uniform_anchors,seed=seed)


def rollout_adapter(config: Config | None = None) -> RolloutAdapter:
    cfg=config or Config()
    def reset(env, state):
        if isinstance(state,dict): env.reset(state["positions"],state.get("velocities"))
        else: env.reset(state)
    def step(env, action):
        observation,_,done,info=env.step(action); return observation,done,info
    def restore(env,state): env.restore_augmented_state(state)
    def distance(left,right):
        return float(np.sqrt(np.mean((np.asarray(left["positions"])-np.asarray(right["positions"]))**2)))
    return RolloutAdapter(make_env=lambda:FourWayIntersectionEnv(cfg), reset=reset,
                          observation=lambda env:env.observation(), step=step,
                          snapshot=lambda env:env.augmented_state(), restore=restore,
                          state_distance=distance)


def hard_safety_smoke(config: Config | None = None) -> dict:
    """Verify generic projection sees exactly six pairwise Four-Way rows."""
    from .safety import HardSafetyFilter
    env=FourWayIntersectionEnv(config)
    result=HardSafetyFilter()(env.snapshot(),np.zeros((4,2)))
    return {"status":result.status,"num_pair_constraints":result.diagnostics["num_pair_constraints"],
            "pair_indices":result.diagnostics["pair_indices"],"num_wall_constraints":result.diagnostics["num_wall_constraints"]}
