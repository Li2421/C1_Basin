#!/usr/bin/env python3
"""Matched frozen S-XL-128 no-safety and one-pass hard-safety evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import numpy as np

from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _initialize_env, _radial_bound64
from double_bottleneck.expert_dataset import infer_coordination_mode
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from shared_control.hard_projection import HardProjectionConfig, HardSafetyFilter


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
PRIOR_EVALUATION = ROOT / "diagnostics/double_bottleneck_recovery_density_final/evaluation"
SETTINGS = {
    "existing_untouched_test": {
        "dataset": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
        "seeds": (101, 211, 307, 401),
    },
    "fresh_untouched_test": {
        "dataset": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
        "seeds": (503, 607, 701, 809),
    },
}
ACTIVE_THRESHOLD = 1e-6
LARGE_THRESHOLD = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def outcome_category(termination: str, wall: bool, agent: bool) -> str:
    if termination == "collision":
        if wall and agent:
            return "wall_and_agent_collision"
        if wall:
            return "wall_collision"
        if agent:
            return "agent_collision"
        return "unattributed_collision"
    if termination == "success":
        return "success"
    if termination == "deadlock":
        return "strict_deadlock"
    if termination == "timeout":
        return "timeout"
    return "other"


def _candidate_run_update(runs: dict[str, int], maxima: dict[str, int], name: str, value: bool):
    runs[name] = runs.get(name, 0) + 1 if value else 0
    maxima[name] = max(maxima.get(name, 0), runs[name])


def shadow_deadlock(positions, actions, goals, config: Config, termination: str) -> dict:
    """Offline implementation of the unfrozen resource-frontier-v1 audit proposal."""

    positions = np.asarray(positions, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    samples = len(positions)
    if actions.shape != (samples - 1, 4, 2):
        raise ValueError("shadow trajectory shape mismatch")
    window = 40
    required = 101
    epsilon = 0.01
    low_speed = 0.025
    engagement = 1.32
    errors = np.linalg.norm(goals[None] - positions, axis=-1)
    unfinished = errors > config.goal_tolerance
    goal_frontier = np.minimum.accumulate(errors, axis=0)
    directions = np.sign(goals[:, 0] - positions[0, :, 0]).astype(np.int64)
    directions = np.where(directions == 0, np.sign(goals[:, 0]).astype(np.int64), directions)
    if set(directions.tolist()) != {-1, 1}:
        raise ValueError("shadow detector requires both task directions")

    planes = {
        "left_resource": np.where(directions > 0, -0.79, -2.01),
        "right_resource": np.where(directions > 0, 2.01, 0.79),
    }
    q = {
        name: np.maximum(0.0, directions[None] * (plane[None] - positions[:, :, 0]))
        for name, plane in planes.items()
    }
    cleared = {
        name: np.maximum.accumulate(values <= 1e-12, axis=0).astype(bool)
        for name, values in q.items()
    }
    frontier = {name: np.minimum.accumulate(values, axis=0) for name, values in q.items()}

    runs: dict[str, int] = {}
    maxima: dict[str, int] = {}
    first_trigger = None
    first_candidates = {}
    candidate_after_trigger_false = False
    terminal_sample = samples - 1
    for n in range(window, samples):
        current_candidates: dict[str, bool] = {}
        current_unfinished = unfinished[n]
        unchanged = bool(
            np.all(unfinished[n - window : n + 1] == current_unfinished[None])
        )
        global_candidate = bool(
            np.any(current_unfinished)
            and unchanged
            and np.all(np.abs(errors[n - window, current_unfinished] - errors[n, current_unfinished]) < epsilon)
            and np.all(np.linalg.norm(actions[n - 1, current_unfinished], axis=-1) < low_speed)
        )
        current_candidates["global"] = global_candidate

        for name in ("left_resource", "right_resource"):
            cohort_now = ~cleared[name][n]
            cohort_past = ~cleared[name][n - window]
            value = bool(
                np.any(cohort_now)
                and np.array_equal(cohort_now, cohort_past)
                and np.min(q[name][n, cohort_now]) <= engagement
                and np.max(
                    frontier[name][n - window, cohort_now]
                    - frontier[name][n, cohort_now]
                )
                < epsilon
            )
            current_candidates[name] = value

        both_cleared = cleared["left_resource"] & cleared["right_resource"]
        for agent_index in range(4):
            name = f"post_passage_{agent_index}"
            value = bool(
                both_cleared[n - window, agent_index]
                and unfinished[n, agent_index]
                and goal_frontier[n - window, agent_index]
                - goal_frontier[n, agent_index]
                < epsilon
            )
            current_candidates[name] = value

        for name, value in current_candidates.items():
            _candidate_run_update(runs, maxima, name, value)
            if value and name not in first_candidates:
                first_candidates[name] = n

        triggered = [name for name, count in runs.items() if count >= required]
        collision_or_success_priority = n == terminal_sample and termination in (
            "collision",
            "success",
        )
        if first_trigger is None and triggered and not collision_or_success_priority:
            order = ["global", "left_resource", "right_resource"] + [
                f"post_passage_{index}" for index in range(4)
            ]
            selected = min(triggered, key=order.index)
            first_trigger = {"step": n, "type": selected}
        elif first_trigger is not None and not current_candidates.get(first_trigger["type"], False):
            candidate_after_trigger_false = True

    return {
        "detector": "double_bottleneck_resource_frontier_v1",
        "status": "diagnostic_unfrozen_thresholds",
        "shadow_deadlock": first_trigger is not None,
        "first_trigger_step": None if first_trigger is None else first_trigger["step"],
        "first_trigger_time_seconds": (
            None if first_trigger is None else first_trigger["step"] * config.dt
        ),
        "trigger_type": None if first_trigger is None else first_trigger["type"],
        "recovered_after_trigger": bool(first_trigger is not None and candidate_after_trigger_false),
        "max_consecutive_candidate_samples": maxima,
        "first_candidate_steps": first_candidates,
    }


def correction_summary(values: np.ndarray, active: np.ndarray, large: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    large = np.asarray(large, dtype=bool)
    return {
        "steps": int(len(values)),
        "active_steps": int(active.sum()),
        "active_fraction": float(active.mean()) if len(active) else 0.0,
        "mean_norm": float(values.mean()) if len(values) else 0.0,
        "median_norm": float(np.median(values)) if len(values) else 0.0,
        "p90_norm": float(np.percentile(values, 90)) if len(values) else 0.0,
        "p95_norm": float(np.percentile(values, 95)) if len(values) else 0.0,
        "p99_norm": float(np.percentile(values, 99)) if len(values) else 0.0,
        "maximum_norm": float(values.max()) if len(values) else 0.0,
        "large_steps": int(large.sum()),
        "large_fraction": float(large.mean()) if len(large) else 0.0,
        "mean_norm_given_active": float(values[active].mean()) if np.any(active) else 0.0,
    }


def run_episode(policy, dataset, episode, seed: int, rollout_id: int, controller: str):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = HardSafetyFilter() if controller == "hard_safety" else None
    episode_key = jax.random.fold_in(jax.random.PRNGKey(seed), rollout_id)
    positions = [env.positions.copy()]
    raw_actions = []
    flow_actions = []
    executed_actions = []
    corrections = []
    active = []
    large = []
    statuses = []
    min_wall = float(env.distances()[0].min())
    min_pair = float(env.distances()[1].min())
    final_info = None
    started = time.perf_counter()
    for step in range(config.max_steps):
        step_key = jax.random.fold_in(episode_key, step)
        raw = np.asarray(
            policy.sample_actions(env.observation()[None], step_key)[0], dtype=np.float64
        )
        flow = _radial_bound64(raw, config.max_speed)
        if projector is None:
            executed = flow.copy()
            status = "disabled"
        else:
            result = projector(env.snapshot(), flow)
            executed = np.asarray(result.velocity, dtype=np.float64)
            status = str(result.status)
        correction = float(np.linalg.norm(executed - flow))
        _, _, done, info = env.step(executed)
        raw_actions.append(raw)
        flow_actions.append(flow)
        executed_actions.append(executed)
        corrections.append(correction)
        active.append(correction > ACTIVE_THRESHOLD)
        large.append(correction >= LARGE_THRESHOLD)
        statuses.append(status)
        positions.append(env.positions.copy())
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        final_info = info
        if done:
            break
    if final_info is None or not env.done:
        raise AssertionError("episode failed to reach a runtime terminal event")
    positions = np.asarray(positions, dtype=np.float64)
    raw_actions = np.asarray(raw_actions, dtype=np.float64)
    flow_actions = np.asarray(flow_actions, dtype=np.float64)
    executed_actions = np.asarray(executed_actions, dtype=np.float64)
    corrections = np.asarray(corrections, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    large = np.asarray(large, dtype=bool)
    termination = str(final_info["termination"])
    wall = bool(final_info["wall_collision"])
    agent = bool(final_info["agent_collision"])
    outcome = outcome_category(termination, wall, agent)
    mode = infer_coordination_mode(positions, env.goals, config)
    shadow = shadow_deadlock(positions, executed_actions, env.goals, config, termination)
    episode_projection = correction_summary(corrections, active, large)
    episode_projection["solver_status_counts"] = dict(Counter(statuses))
    summary = {
        "set": None,
        "controller": controller,
        "family_id": episode.family_id,
        "regime": episode.regime,
        "seed": int(seed),
        "rollout_id": int(rollout_id),
        "episode_key_data": np.asarray(jax.random.key_data(episode_key)).tolist(),
        "termination": termination,
        "outcome": outcome,
        "task_success": termination == "success",
        "wall_collision": wall,
        "agent_collision": agent,
        "runtime_strict_deadlock": termination == "deadlock",
        "timeout": termination == "timeout",
        "other_termination": outcome == "other",
        "runtime_safe_liveness_failure": bool(
            not wall
            and not agent
            and termination in ("deadlock", "timeout")
        ),
        "episode_steps": int(len(executed_actions)),
        "episode_seconds": float(len(executed_actions) * config.dt),
        "minimum_swept_wall_clearance": min_wall,
        "minimum_swept_agent_clearance": min_pair,
        "final_goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1).tolist(),
        "coordination_mode": mode,
        "shadow_deadlock": shadow,
        "shadow_safe_liveness_failure": bool(
            not wall
            and not agent
            and not (termination == "success")
            and (shadow["shadow_deadlock"] or termination in ("deadlock", "timeout"))
        ),
        "projection": episode_projection,
        "rollout_wall_seconds": time.perf_counter() - started,
        "rng_protocol": "fold_in(fold_in(PRNGKey(seed), rollout_id), absolute_step)",
        "eta": False,
        "g_phi": False,
    }
    arrays = {
        "positions": positions,
        "raw_actions": raw_actions,
        "flow_actions": flow_actions,
        "executed_actions": executed_actions,
        "correction_norms": corrections,
        "projection_active": active,
        "large_correction": large,
        "solver_status": np.asarray(statuses, dtype="U32"),
    }
    return summary, arrays


def aggregate(rows: list[dict], all_corrections, all_active, all_large) -> dict:
    outcomes = Counter(row["outcome"] for row in rows)
    shadow_types = Counter(
        row["shadow_deadlock"]["trigger_type"]
        for row in rows
        if row["shadow_deadlock"]["shadow_deadlock"]
    )
    failures = sum(
        count for outcome, count in outcomes.items() if outcome != "success"
    )
    result = {
        "rollouts": len(rows),
        "outcome_partition": dict(sorted(outcomes.items())),
        "success": sum(row["task_success"] for row in rows),
        "wall_collision": sum(row["wall_collision"] for row in rows),
        "agent_collision": sum(row["agent_collision"] for row in rows),
        "runtime_strict_deadlock": sum(row["runtime_strict_deadlock"] for row in rows),
        "timeout": sum(row["timeout"] for row in rows),
        "other": sum(row["other_termination"] for row in rows),
        "failure": failures,
        "accounting_complete": len(rows) == outcomes.get("success", 0) + failures,
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in rows])),
        "median_episode_steps": float(np.median([row["episode_steps"] for row in rows])),
        "minimum_wall_clearance": float(
            min(row["minimum_swept_wall_clearance"] for row in rows)
        ),
        "minimum_agent_clearance": float(
            min(row["minimum_swept_agent_clearance"] for row in rows)
        ),
        "runtime_safe_liveness_failures": sum(
            row["runtime_safe_liveness_failure"] for row in rows
        ),
        "shadow_deadlocks": sum(
            row["shadow_deadlock"]["shadow_deadlock"] for row in rows
        ),
        "shadow_deadlock_types": dict(sorted(shadow_types.items())),
        "runtime_timeouts_shadow_deadlocked_earlier": sum(
            row["timeout"] and row["shadow_deadlock"]["shadow_deadlock"]
            for row in rows
        ),
        "shadow_safe_liveness_failures": sum(
            row["shadow_safe_liveness_failure"] for row in rows
        ),
        "shadow_triggers_on_eventual_success": sum(
            row["task_success"] and row["shadow_deadlock"]["shadow_deadlock"]
            for row in rows
        ),
        "projection": correction_summary(
            np.concatenate(all_corrections),
            np.concatenate(all_active),
            np.concatenate(all_large),
        ),
    }
    by_outcome = {}
    for outcome in sorted(outcomes):
        selected = [index for index, row in enumerate(rows) if row["outcome"] == outcome]
        by_outcome[outcome] = correction_summary(
            np.concatenate([all_corrections[index] for index in selected]),
            np.concatenate([all_active[index] for index in selected]),
            np.concatenate([all_large[index] for index in selected]),
        )
    result["projection"]["by_episode_outcome"] = by_outcome
    return result


def validate_no_safety_reference(set_name: str, rows: list[dict]) -> dict:
    prior_path = PRIOR_EVALUATION / set_name / "s-xl-128.json"
    prior = json.loads(prior_path.read_text())["rollouts"]
    if len(prior) != len(rows):
        raise AssertionError("no-safety reference rollout count mismatch")
    mismatches = []
    fields = (
        "family_id",
        "seed",
        "rollout_id",
        "episode_steps",
        "wall_collision",
        "agent_collision",
        "timeout",
    )
    for old, new in zip(prior, rows, strict=True):
        for field in fields:
            if old[field] != new[field]:
                mismatches.append(
                    {"rollout_id": new["rollout_id"], "field": field, "old": old[field], "new": new[field]}
                )
        if bool(old["success"]) != bool(new["task_success"]):
            mismatches.append(
                {"rollout_id": new["rollout_id"], "field": "success", "old": old["success"], "new": new["task_success"]}
            )
        if bool(old["deadlock"]) != bool(new["runtime_strict_deadlock"]):
            mismatches.append(
                {"rollout_id": new["rollout_id"], "field": "deadlock", "old": old["deadlock"], "new": new["runtime_strict_deadlock"]}
            )
    if mismatches:
        raise AssertionError(f"no-safety reproduction mismatch: {mismatches[:5]}")
    return {
        "reference": str(prior_path.resolve()),
        "reference_sha256": sha(prior_path),
        "checked_fields": list(fields) + ["success", "deadlock"],
        "matched_rollouts": len(rows),
        "mismatches": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("no_safety", "hard_safety"), required=True)
    parser.add_argument("--set", dest="set_name", choices=tuple(SETTINGS), required=True)
    args = parser.parse_args()
    prereg = STUDY / "PREREGISTRATION.json"
    if sha(prereg) != "1de02ae4e29867b940d8e339fe68bda72b15360ea44a061d068a4cd354bf090b":
        raise RuntimeError("pre-registration changed after freeze")
    setting = SETTINGS[args.set_name]
    dataset = FlowBC4ADataset(setting["dataset"], "val", seed=45 if args.set_name.startswith("existing") else 46)
    policy, metadata = load_checkpoint(CHECKPOINT, dataset.environment_fingerprint)
    if jax.default_backend() != "cpu":
        raise RuntimeError("this matched evaluation is preregistered for CPU")
    result_path = STUDY / f"{args.controller}_{args.set_name}_outcomes.json"
    final_trajectory_dir = STUDY / "trajectories" / args.controller / args.set_name
    partial = STUDY / "trajectories" / f".{args.controller}_{args.set_name}.partial"
    if result_path.exists() or final_trajectory_dir.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.controller}/{args.set_name}")
    partial.mkdir(parents=True, exist_ok=False)
    rows = []
    all_corrections = []
    all_active = []
    all_large = []
    rollout_id = 0
    started = time.perf_counter()
    for family in dataset.family_names:
        episode = dataset.by_family[family][0]
        for seed in setting["seeds"]:
            row, arrays = run_episode(
                policy, dataset, episode, seed, rollout_id, args.controller
            )
            row["set"] = args.set_name
            rows.append(row)
            all_corrections.append(arrays["correction_norms"])
            all_active.append(arrays["projection_active"])
            all_large.append(arrays["large_correction"])
            np.savez_compressed(partial / f"rollout_{rollout_id:03d}.npz", **arrays)
            print(
                json.dumps(
                    {
                        "controller": args.controller,
                        "set": args.set_name,
                        "rollout": rollout_id,
                        "outcome": row["outcome"],
                        "steps": row["episode_steps"],
                        "projection_active": row["projection"]["active_fraction"],
                        "shadow": row["shadow_deadlock"]["trigger_type"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            rollout_id += 1
    aggregate_result = aggregate(rows, all_corrections, all_active, all_large)
    reference_check = (
        validate_no_safety_reference(args.set_name, rows)
        if args.controller == "no_safety"
        else None
    )
    result = {
        "schema": "double_bottleneck_hard_safety_matched_outcomes_v1",
        "preregistration": str(prereg.resolve()),
        "preregistration_sha256": sha(prereg),
        "controller": args.controller,
        "set": args.set_name,
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha(CHECKPOINT),
        "checkpoint_metadata": metadata,
        "seeds": list(setting["seeds"]),
        "protocol": f"{len(dataset.family_names)} frozen initial states x {len(setting['seeds'])} seeds",
        "projection_config": HardProjectionConfig().to_dict(),
        "reference_reproduction": reference_check,
        "aggregate": aggregate_result,
        "rollouts": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    final_trajectory_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(final_trajectory_dir)
    print(json.dumps({"complete": str(result_path), "aggregate": aggregate_result}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
