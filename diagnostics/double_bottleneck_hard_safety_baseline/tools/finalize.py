#!/usr/bin/env python3
"""Freeze paired summaries and post-hoc liveness analysis after evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
SETS = ("existing_untouched_test", "fresh_untouched_test")
CONTROLLERS = ("no_safety", "hard_safety")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value) -> None:
    path = STUDY / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_results():
    results = {}
    for controller in CONTROLLERS:
        for set_name in SETS:
            path = STUDY / f"{controller}_{set_name}_outcomes.json"
            results[(controller, set_name)] = json.loads(path.read_text())
    return results


def combined_outcomes(results, controller: str):
    rows = [
        row
        for set_name in SETS
        for row in results[(controller, set_name)]["rollouts"]
    ]
    outcomes = Counter(row["outcome"] for row in rows)
    directions = Counter(
        row["coordination_mode"]["first_direction"] or "ambiguous"
        for row in rows
    )
    success_directions = Counter(
        row["coordination_mode"]["first_direction"] or "ambiguous"
        for row in rows
        if row["task_success"]
    )
    return {
        "rollouts": len(rows),
        "outcome_partition": dict(sorted(outcomes.items())),
        "success": sum(row["task_success"] for row in rows),
        "wall_collision": sum(row["wall_collision"] for row in rows),
        "agent_collision": sum(row["agent_collision"] for row in rows),
        "runtime_strict_deadlock": sum(row["runtime_strict_deadlock"] for row in rows),
        "timeout": sum(row["timeout"] for row in rows),
        "other": sum(row["other_termination"] for row in rows),
        "failure": sum(not row["task_success"] for row in rows),
        "runtime_safe_liveness_failure": sum(
            row["runtime_safe_liveness_failure"] for row in rows
        ),
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in rows])),
        "median_episode_steps": float(np.median([row["episode_steps"] for row in rows])),
        "minimum_wall_clearance": float(
            min(row["minimum_swept_wall_clearance"] for row in rows)
        ),
        "minimum_agent_clearance": float(
            min(row["minimum_swept_agent_clearance"] for row in rows)
        ),
        "realized_first_direction": dict(sorted(directions.items())),
        "successful_first_direction": dict(sorted(success_directions.items())),
        "successful_complete_signatures": sorted(
            {
                row["coordination_mode"]["signature"]
                for row in rows
                if row["task_success"] and row["coordination_mode"]["complete"]
            }
        ),
    }


def correction_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def projection_analysis(results, pairs):
    corrections = []
    flow_norms = []
    ratios = []
    cosine = []
    active = []
    large = []
    statuses = Counter()
    by_outcome = defaultdict(list)
    by_conversion = defaultdict(list)
    for pair in pairs:
        set_name = pair["set"]
        rollout_id = pair["rollout_id"]
        path = STUDY / "trajectories/hard_safety" / set_name / f"rollout_{rollout_id:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            flow = data["flow_actions"].astype(np.float64).reshape((-1, 8))
            executed = data["executed_actions"].astype(np.float64).reshape((-1, 8))
            corr = data["correction_norms"].astype(np.float64)
            active_values = data["projection_active"].astype(bool)
            large_values = data["large_correction"].astype(bool)
            status_values = data["solver_status"].astype(str)
        fnorm = np.linalg.norm(flow, axis=1)
        enorm = np.linalg.norm(executed, axis=1)
        ratio = corr / np.maximum(fnorm, 1e-12)
        valid = (fnorm > 1e-9) & (enorm > 1e-9)
        cos = np.sum(flow[valid] * executed[valid], axis=1) / (fnorm[valid] * enorm[valid])
        corrections.append(corr)
        flow_norms.append(fnorm)
        ratios.append(ratio)
        cosine.append(cos)
        active.append(active_values)
        large.append(large_values)
        statuses.update(status_values.tolist())
        by_outcome[pair["safety_outcome"]].append(corr)
        by_conversion[f"{pair['no_safety_outcome']} -> {pair['safety_outcome']}"] .append(corr)
    corr = np.concatenate(corrections)
    fnorm = np.concatenate(flow_norms)
    ratio = np.concatenate(ratios)
    cos = np.concatenate(cosine)
    active_values = np.concatenate(active)
    large_values = np.concatenate(large)
    return {
        "definition": {
            "correction": "joint L2 ||u_safe-u_flow||",
            "active_threshold": 1e-6,
            "large_threshold": 0.1,
            "relative_ratio": "||u_safe-u_flow|| / max(||u_flow||, 1e-12); post-hoc continuous diagnostic",
        },
        "all_steps": {
            "correction_norm": correction_stats(corr),
            "flow_action_norm": correction_stats(fnorm),
            "relative_correction_ratio": correction_stats(ratio),
            "flow_safe_cosine_nonzero_steps": correction_stats(cos),
            "active_steps": int(active_values.sum()),
            "active_fraction": float(active_values.mean()),
            "large_steps": int(large_values.sum()),
            "large_fraction": float(large_values.mean()),
            "correction_at_least_half_flow_fraction": float(np.mean(corr >= 0.5 * fnorm)),
            "correction_at_least_flow_fraction": float(np.mean(corr >= fnorm)),
            "solver_status_counts": dict(sorted(statuses.items())),
        },
        "by_safety_outcome": {
            name: correction_stats(np.concatenate(values))
            for name, values in sorted(by_outcome.items())
        },
        "by_outcome_conversion": {
            name: correction_stats(np.concatenate(values))
            for name, values in sorted(by_conversion.items())
        },
    }


def load_goals(set_name: str):
    if set_name == "existing_untouched_test":
        path = ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool/manifest.json"
    else:
        path = ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool/manifest.json"
    return np.asarray(json.loads(path.read_text())["environment"]["goals"], dtype=np.float64)


def timeout_diagnostic(pair):
    set_name = pair["set"]
    rollout_id = pair["rollout_id"]
    goals = load_goals(set_name)
    path = STUDY / "trajectories/hard_safety" / set_name / f"rollout_{rollout_id:03d}.npz"
    with np.load(path, allow_pickle=False) as data:
        positions = data["positions"].astype(np.float64)
        flow = data["flow_actions"].astype(np.float64)
        executed = data["executed_actions"].astype(np.float64)
        corrections = data["correction_norms"].astype(np.float64)
        active = data["projection_active"].astype(bool)
    tail = min(100, len(executed))
    p0 = positions[-tail - 1]
    p1 = positions[-1]
    tail_positions = positions[-tail - 1 :]
    tail_path = np.linalg.norm(np.diff(tail_positions, axis=0), axis=-1).sum(axis=0)
    tail_net = np.linalg.norm(p1 - p0, axis=-1)
    errors0 = np.linalg.norm(goals - p0, axis=-1)
    errors1 = np.linalg.norm(goals - p1, axis=-1)
    before = positions[-tail - 1 : -1]
    goal_delta = goals[None] - before
    goal_unit = goal_delta / np.maximum(np.linalg.norm(goal_delta, axis=-1, keepdims=True), 1e-12)
    flow_goal = np.sum(flow[-tail:] * goal_unit, axis=-1)
    safe_goal = np.sum(executed[-tail:] * goal_unit, axis=-1)
    directions = np.sign(goals[:, 0] - positions[0, :, 0]).astype(int)
    planes_left = np.where(directions > 0, -0.79, -2.01)
    planes_right = np.where(directions > 0, 2.01, 0.79)
    cleared_left = np.any(directions[None] * (positions[:, :, 0] - planes_left[None]) >= 0, axis=0)
    cleared_right = np.any(directions[None] * (positions[:, :, 0] - planes_right[None]) >= 0, axis=0)
    stages = cleared_left.astype(int) + cleared_right.astype(int)
    completed = errors1 <= 0.08
    stationary = tail_net < 0.01
    return {
        "set": set_name,
        "rollout_id": rollout_id,
        "family_id": pair["family_id"],
        "seed": pair["seed"],
        "no_safety_outcome": pair["no_safety_outcome"],
        "safety_outcome": pair["safety_outcome"],
        "completed_agents": int(completed.sum()),
        "completed_mask": completed.tolist(),
        "route_stage_counts_per_agent": stages.tolist(),
        "final_goal_errors": errors1.tolist(),
        "tail_window_steps": tail,
        "tail_net_displacement_per_agent": tail_net.tolist(),
        "tail_path_length_per_agent": tail_path.tolist(),
        "tail_goal_error_reduction_per_agent": (errors0 - errors1).tolist(),
        "tail_stationary_agents_at_0p01m_net": int(stationary.sum()),
        "tail_subset_stationary": bool(0 < stationary.sum() < 4),
        "tail_projection_active_fraction": float(active[-tail:].mean()),
        "tail_mean_correction_norm": float(corrections[-tail:].mean()),
        "tail_mean_flow_goal_component": float(flow_goal.mean()),
        "tail_mean_safe_goal_component": float(safe_goal.mean()),
        "tail_goal_component_removed_by_projection": float((flow_goal - safe_goal).mean()),
        "final_positions": p1.tolist(),
    }


def main() -> int:
    results = load_results()
    source_hashes = {
        "checkpoint": sha(ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"),
        "dataset_manifest": sha(ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json"),
        "macflow_source": sha(ROOT / "double_bottleneck/flowbc_4a_agent.py"),
        "environment_source": sha(ROOT / "double_bottleneck/environment.py"),
        "projection_source": sha(ROOT / "shared_control/hard_projection.py"),
    }
    expected = {
        "checkpoint": "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd",
        "dataset_manifest": "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56",
        "macflow_source": "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8",
        "environment_source": "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc",
        "projection_source": "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79",
    }
    if source_hashes != expected:
        raise RuntimeError(f"frozen source changed: {source_hashes}")

    pairs = []
    matrix_by_set = {}
    for set_name in SETS:
        no_rows = results[("no_safety", set_name)]["rollouts"]
        safe_rows = results[("hard_safety", set_name)]["rollouts"]
        if len(no_rows) != len(safe_rows):
            raise AssertionError("paired rollout count mismatch")
        matrix = Counter()
        cells = defaultdict(list)
        for no, safe in zip(no_rows, safe_rows, strict=True):
            identity = (no["family_id"], no["seed"], no["rollout_id"])
            if identity != (safe["family_id"], safe["seed"], safe["rollout_id"]):
                raise AssertionError("paired rollout identity mismatch")
            key = (no["outcome"], safe["outcome"])
            matrix[key] += 1
            cells[f"{key[0]} -> {key[1]}"].append(no["rollout_id"])
            pairs.append(
                {
                    "set": set_name,
                    "family_id": no["family_id"],
                    "regime": no["regime"],
                    "seed": no["seed"],
                    "rollout_id": no["rollout_id"],
                    "no_safety_outcome": no["outcome"],
                    "safety_outcome": safe["outcome"],
                    "no_safety_steps": no["episode_steps"],
                    "safety_steps": safe["episode_steps"],
                    "safety_projection_active_fraction": safe["projection"]["active_fraction"],
                    "safety_mean_correction_norm": safe["projection"]["mean_norm"],
                    "no_safety_shadow_deadlock": no["shadow_deadlock"]["shadow_deadlock"],
                    "safety_shadow_deadlock": safe["shadow_deadlock"]["shadow_deadlock"],
                }
            )
        matrix_by_set[set_name] = {
            "counts": dict(sorted((f"{a} -> {b}", count) for (a, b), count in matrix.items())),
            "rollout_ids_by_cell": dict(sorted(cells.items())),
        }
    combined_matrix = Counter(
        (pair["no_safety_outcome"], pair["safety_outcome"]) for pair in pairs
    )
    conversion = {
        "schema": "double_bottleneck_matched_failure_conversion_v1",
        "paired_rollouts": len(pairs),
        "sets": matrix_by_set,
        "combined": {
            "counts": dict(
                sorted(
                    (f"{a} -> {b}", count)
                    for (a, b), count in combined_matrix.items()
                )
            ),
            "collision_to_success": sum(
                count
                for (a, b), count in combined_matrix.items()
                if "collision" in a and b == "success"
            ),
            "collision_to_safe_timeout": sum(
                count
                for (a, b), count in combined_matrix.items()
                if "collision" in a and b == "timeout"
            ),
            "success_to_safe_timeout": combined_matrix[("success", "timeout")],
            "timeout_to_success": combined_matrix[("timeout", "success")],
            "timeout_to_timeout": combined_matrix[("timeout", "timeout")],
        },
        "pairs": pairs,
    }
    dump("matched_episode_conversion_matrix.json", conversion)

    with (STUDY / "matched_episode_pairs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    no_summary = {
        "schema": "double_bottleneck_no_safety_frozen_summary_v1",
        "source_hashes": source_hashes,
        "by_set": {
            name: results[("no_safety", name)]["aggregate"] for name in SETS
        },
        "combined": combined_outcomes(results, "no_safety"),
    }
    safety_summary = {
        "schema": "double_bottleneck_hard_safety_frozen_summary_v1",
        "source_hashes": source_hashes,
        "by_set": {
            name: results[("hard_safety", name)]["aggregate"] for name in SETS
        },
        "combined": combined_outcomes(results, "hard_safety"),
    }
    dump("no_safety_summary.json", no_summary)
    dump("hard_safety_summary.json", safety_summary)

    projection = projection_analysis(results, pairs)
    dump("projection_statistics.json", projection)

    shadow_rows = []
    for controller in CONTROLLERS:
        for set_name in SETS:
            for row in results[(controller, set_name)]["rollouts"]:
                if row["shadow_deadlock"]["shadow_deadlock"]:
                    shadow_rows.append(
                        {
                            "controller": controller,
                            "set": set_name,
                            "rollout_id": row["rollout_id"],
                            "runtime_outcome": row["outcome"],
                            "trigger_type": row["shadow_deadlock"]["trigger_type"],
                            "trigger_step": row["shadow_deadlock"]["first_trigger_step"],
                            "recovered_after_trigger": row["shadow_deadlock"]["recovered_after_trigger"],
                        }
                    )
    shadow_summary = {}
    for controller in CONTROLLERS:
        for set_name in SETS:
            rows = results[(controller, set_name)]["rollouts"]
            key = f"{controller}/{set_name}"
            triggered = [row for row in rows if row["shadow_deadlock"]["shadow_deadlock"]]
            successes = [row for row in rows if row["task_success"]]
            timeouts = [row for row in rows if row["timeout"]]
            shadow_summary[key] = {
                "rollouts": len(rows),
                "shadow_deadlocks": len(triggered),
                "by_runtime_outcome": dict(sorted(Counter(row["outcome"] for row in triggered).items())),
                "by_trigger_type": dict(
                    sorted(Counter(row["shadow_deadlock"]["trigger_type"] for row in triggered).items())
                ),
                "successful_rollouts": len(successes),
                "shadow_triggers_on_eventual_success": sum(
                    row["shadow_deadlock"]["shadow_deadlock"] for row in successes
                ),
                "successful_false_positive_fraction": (
                    sum(row["shadow_deadlock"]["shadow_deadlock"] for row in successes) / len(successes)
                    if successes
                    else None
                ),
                "runtime_timeouts": len(timeouts),
                "timeouts_classified_earlier_by_shadow": sum(
                    row["shadow_deadlock"]["shadow_deadlock"] for row in timeouts
                ),
                "recovered_after_trigger": sum(
                    row["shadow_deadlock"]["recovered_after_trigger"] for row in triggered
                ),
            }
    shadow = {
        "schema": "double_bottleneck_shadow_deadlock_analysis_v1",
        "detector": "double_bottleneck_resource_frontier_v1",
        "status": "diagnostic_unfrozen_thresholds",
        "canonical_runtime_unchanged": True,
        "summary": shadow_summary,
        "interpretation": "The candidate detector has a high false-positive rate on eventual successes and is not suitable as canonical ground truth without revision/validation.",
        "triggered_cases": shadow_rows,
    }
    dump("shadow_deadlock_analysis.json", shadow)

    timeout_rows = [timeout_diagnostic(pair) for pair in pairs if pair["safety_outcome"] == "timeout"]
    completed_distribution = Counter(row["completed_agents"] for row in timeout_rows)
    origin_distribution = Counter(row["no_safety_outcome"] for row in timeout_rows)
    timeout_analysis = {
        "schema": "double_bottleneck_safe_timeout_liveness_analysis_v1",
        "timeout_cases": len(timeout_rows),
        "origin_no_safety_outcomes": dict(sorted(origin_distribution.items())),
        "completed_agents_at_horizon": {
            str(key): value for key, value in sorted(completed_distribution.items())
        },
        "cases_with_subset_stationary_in_last_5s": sum(
            row["tail_subset_stationary"] for row in timeout_rows
        ),
        "mean_tail_projection_active_fraction": float(
            np.mean([row["tail_projection_active_fraction"] for row in timeout_rows])
        ),
        "mean_tail_correction_norm": float(
            np.mean([row["tail_mean_correction_norm"] for row in timeout_rows])
        ),
        "mean_tail_goal_component_removed_by_projection": float(
            np.mean([row["tail_goal_component_removed_by_projection"] for row in timeout_rows])
        ),
        "cases": timeout_rows,
    }
    dump("safe_timeout_liveness_analysis.json", timeout_analysis)

    desired_cells = (
        "success -> success",
        "success -> timeout",
        "agent_collision -> success",
        "agent_collision -> timeout",
        "wall_collision -> timeout",
        "timeout -> success",
        "timeout -> timeout",
    )
    representatives = []
    for cell in desired_cells:
        choices = [
            pair
            for pair in pairs
            if f"{pair['no_safety_outcome']} -> {pair['safety_outcome']}" == cell
        ]
        if not choices:
            continue
        pair = choices[0]
        entry = dict(pair)
        if pair["safety_outcome"] == "timeout":
            entry["safety_timeout_diagnostic"] = next(
                row
                for row in timeout_rows
                if row["set"] == pair["set"] and row["rollout_id"] == pair["rollout_id"]
            )
        representatives.append(entry)
    dump(
        "representative_trajectory_analysis.json",
        {
            "schema": "double_bottleneck_hard_safety_representatives_v1",
            "selection": "First rollout in each predeclared available outcome-conversion cell; selected only after all outcomes were frozen.",
            "representatives": representatives,
        },
    )
    print(
        json.dumps(
            {
                "no_safety": no_summary["combined"],
                "hard_safety": safety_summary["combined"],
                "conversion": conversion["combined"],
                "projection": projection["all_steps"],
                "timeouts": {key: value for key, value in timeout_analysis.items() if key != "cases"},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
