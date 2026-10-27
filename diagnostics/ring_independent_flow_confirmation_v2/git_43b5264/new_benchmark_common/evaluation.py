"""Scenario-neutral closed-loop, support/OOD, teacher, and K-step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping

import jax
import numpy as np

from .protocol import RolloutAdapter


@dataclass(frozen=True)
class EvaluationCase:
    rollout_id: str
    initial_state: Any
    split: str = "test"


@dataclass(frozen=True)
class KStepWindow:
    """A reference expert window; states are snapshots accepted by adapter.restore."""
    rollout_id: str
    start_time: int
    states: tuple[Any, ...]


def _policy_action(agent, observation: np.ndarray, key) -> np.ndarray:
    return np.asarray(agent.sample_actions(observation[None], key)[0], dtype=np.float64)


def _ood_distances(observations: np.ndarray, train_observations: np.ndarray, *, chunk: int = 256,
                   reference_cap: int = 4096) -> np.ndarray:
    """Nearest-neighbour standardized distance without using labels or test adaptation."""
    train = train_observations.reshape(len(train_observations), -1).astype(np.float64)
    values = observations.reshape(len(observations), -1).astype(np.float64)
    # Exact all-pairs NN is quadratic and can turn a diagnostic into a
    # multi-gigabyte allocation on the deliberately broad recovery corpus.
    # A deterministic evenly-spaced reference bank is sufficient for this
    # support diagnostic and leaves training labels/actions untouched.
    if len(train) > reference_cap:
        train = train[np.linspace(0, len(train) - 1, reference_cap, dtype=int)]
    mean, scale = train.mean(0), np.maximum(train.std(0), 0.01)
    train, values = (train - mean) / scale, (values - mean) / scale
    result = []
    for begin in range(0, len(values), chunk):
        distances = np.sqrt(((values[begin:begin + chunk, None] - train[None]) ** 2).mean(-1))
        result.extend(distances.min(1))
    return np.asarray(result)


def rollout_case(adapter: RolloutAdapter, agent, case: EvaluationCase, *, seed: int, max_steps: int,
                 action_postprocess: Callable[[np.ndarray], np.ndarray] | None = None):
    env = adapter.make_env()
    adapter.reset(env, case.initial_state)
    stable_id = int.from_bytes(hashlib.sha256(case.rollout_id.encode("utf-8")).digest()[:4], "little")
    key = jax.random.fold_in(jax.random.PRNGKey(seed), stable_id)
    observations, snapshots, infos = [], [adapter.snapshot(env)], []
    terminal = "timeout"
    for step in range(max_steps):
        observation = np.asarray(adapter.observation(env), dtype=np.float32)
        observations.append(observation)
        action = _policy_action(agent, observation, jax.random.fold_in(key, step))
        if action_postprocess is not None:
            action = action_postprocess(action)
        _, done, info = adapter.step(env, action)
        infos.append(dict(info))
        snapshots.append(adapter.snapshot(env))
        terminal = str(info.get("termination", "success" if done else "running"))
        if done:
            break
    last = infos[-1] if infos else {}
    return {"rollout_id": case.rollout_id, "split": case.split, "termination": terminal,
            "success": terminal == "success", "wall_or_obstacle_collision": bool(last.get("wall_collision", False) or last.get("obstacle_collision", False)),
            "agent_collision": bool(last.get("agent_collision", False)), "timeout": terminal == "timeout",
            "episode_length": len(infos), "observations": np.asarray(observations), "snapshots": snapshots,
            "infos": infos}


def teacher_forced_rmse(agent, observations: np.ndarray, expert_actions: np.ndarray, *, seed: int = 0,
                        samples: int = 8, max_states: int = 512) -> Mapping[str, float]:
    indices = np.linspace(0, len(observations) - 1, min(len(observations), max_states), dtype=int)
    expected = expert_actions[indices].reshape(len(indices), -1)
    key = jax.random.PRNGKey(seed)
    errors = []
    for sample in range(samples):
        prediction = np.asarray(agent.sample_actions(observations[indices], jax.random.fold_in(key, sample)))
        errors.append(np.sqrt(np.mean((prediction.reshape(len(indices), -1) - expected) ** 2, axis=-1)))
    errors = np.stack(errors, axis=1)
    return {"states": int(len(indices)), "samples_per_state": samples,
            "mean_sample_action_rmse": float(errors.mean()), "best_of_k_action_rmse": float(errors.min(1).mean())}


def k_step_stability(adapter: RolloutAdapter, agent, windows: Iterable[KStepWindow], *, seed: int,
                     horizons: tuple[int, ...] = (1, 5, 10, 25, 50, 100),
                     action_postprocess: Callable[[np.ndarray], np.ndarray] | None = None):
    """Open-loop reference-window divergence at required horizons, plus K=100 collision."""
    rows = []
    for window_index, window in enumerate(windows):
        for horizon in horizons:
            if len(window.states) <= horizon:
                continue
            env = adapter.make_env()
            adapter.restore(env, window.states[0])
            collision = False
            for step in range(horizon):
                action = _policy_action(agent, np.asarray(adapter.observation(env), dtype=np.float32),
                                        jax.random.fold_in(jax.random.PRNGKey(seed + window_index), step))
                if action_postprocess:
                    action = action_postprocess(action)
                _, done, info = adapter.step(env, action)
                collision |= bool(info.get("wall_collision", False) or info.get("obstacle_collision", False) or info.get("agent_collision", False))
                if done:
                    break
            rows.append({"rollout_id": window.rollout_id, "start_time": window.start_time, "K": horizon,
                         "divergence": float(adapter.state_distance(adapter.snapshot(env), window.states[horizon])),
                         "collision": collision})
    summary = {}
    for horizon in horizons:
        matching = [row for row in rows if row["K"] == horizon]
        summary[str(horizon)] = {"count": len(matching),
                                 "mean_divergence": float(np.mean([r["divergence"] for r in matching])) if matching else None,
                                 "collision_rate": float(np.mean([r["collision"] for r in matching])) if matching else None}
    return summary, rows


def evaluate_closed_loop(adapter: RolloutAdapter, agent, cases: Iterable[EvaluationCase], *, train_observations: np.ndarray,
                         dev_observations: np.ndarray, seed: int = 0, max_steps: int = 500,
                         action_postprocess: Callable[[np.ndarray], np.ndarray] | None = None):
    """Frozen evaluation aggregate with requested failure taxonomy and support flags."""
    rows = [rollout_case(adapter, agent, case, seed=seed + index, max_steps=max_steps,
                         action_postprocess=action_postprocess) for index, case in enumerate(cases)]
    # Avoid a self-nearest-neighbour zero threshold by using disjoint train
    # probes and reference banks.  This remains a train-only calibration.
    train_values = train_observations[1::2]
    train_reference = train_observations[::2]
    train_dist = _ood_distances(train_values, train_reference)
    threshold = float(np.quantile(train_dist, 0.99))
    all_rollout = np.concatenate([row["observations"] for row in rows if len(row["observations"])])
    rollout_dist = _ood_distances(all_rollout, train_observations) if len(all_rollout) else np.empty(0)
    offset = 0
    for row in rows:
        count = len(row["observations"])
        local = rollout_dist[offset:offset + count]
        offset += count
        row["timestep_0_ood"] = bool(len(local) and local[0] > threshold)
        row["rollout_ood"] = bool(np.any(local > threshold))
    count = max(len(rows), 1)
    return {"rollouts": rows, "aggregate": {
        "rollouts": len(rows), "full_task_success": sum(r["success"] for r in rows) / count,
        "wall_or_obstacle_collision": sum(r["wall_or_obstacle_collision"] for r in rows) / count,
        "agent_collision": sum(r["agent_collision"] for r in rows) / count,
        "timeout": sum(r["timeout"] for r in rows) / count,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])) if rows else 0.0,
        "timestep_0_ood": sum(r["timestep_0_ood"] for r in rows) / count,
        "rollout_ood": sum(r["rollout_ood"] for r in rows) / count,
        "ood_threshold_train_nn_p99": threshold,
        "development_states_used_only_for_threshold_check": int(len(dev_observations)),
    }}
