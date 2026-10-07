"""Policy-neutral episode helpers and crossing-order analysis labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .environment import Config, FourWayIntersectionEnv
from .scenario import sample_initial_state


def crossing_order_signature(positions: np.ndarray, *, centre_radius: float = .70) -> str:
    """Analysis-only order based on first entry into the open conflict disk."""
    p=np.asarray(positions, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4,2): raise ValueError("positions must be [T,4,2]")
    entries=[]
    for i in range(4):
        hits=np.flatnonzero(np.linalg.norm(p[:,i],axis=1) <= centre_radius)
        entries.append(int(hits[0]) if len(hits) else 10**9)
    return "->".join("ABCD"[i] for i in np.argsort(entries, kind="stable"))


@dataclass(frozen=True)
class RolloutResult:
    positions: np.ndarray; actions: np.ndarray; observations: np.ndarray; summary: dict; terminal_reason: str; mode_signature: str


def run_rollout(policy, *, positions=None, velocities=None, config: Config | None = None, split="test", seed=0) -> RolloutResult:
    """Run a stateless joint policy mapping env observation to ``[4,2]``."""
    env=FourWayIntersectionEnv(config)
    if positions is None: positions, velocities, _ = sample_initial_state(env.config, split, seed)
    env.reset(positions, velocities)
    states=[env.positions.copy()]; obs=[env.observation().copy()]; actions=[]; info={"termination":"timeout"}
    for _ in range(env.config.max_steps):
        action=np.asarray(policy(env.observation()), dtype=np.float64)
        out,_,done,info=env.step(action); actions.append(action.copy()); states.append(env.positions.copy());obs.append(out.copy())
        if done: break
    return RolloutResult(np.asarray(states),np.asarray(actions),np.asarray(obs),env.summary(),str(info["termination"]),crossing_order_signature(np.asarray(states)))
