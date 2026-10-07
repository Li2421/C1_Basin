#!/usr/bin/env python3
"""Uniform-state raw-basis and post-projection sensitivity audit."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.evaluate_flowbc_4a import _radial_bound64
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import P0_HIGH, P0_LOW, basis_terms, correction, raw_ortho_flow


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}
FD_NORMALIZED = 0.015625


def stats(values) -> dict:
    array = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p10": float(np.percentile(array, 10)) if len(array) else None,
        "p90": float(np.percentile(array, 90)) if len(array) else None,
        "minimum": float(array.min()) if len(array) else None,
        "maximum": float(array.max()) if len(array) else None,
    }


def matrix_stats(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(1e-3, 0.01 * float(singular[0])) if len(singular) else 1e-3
    active = singular[singular >= threshold]
    gram = matrix.T @ matrix
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosine = gram / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    return {
        "singular_values": singular.tolist(),
        "effective_rank": int(len(active)),
        "rank_threshold": threshold,
        "condition_number_effective": float(active[0] / active[-1]) if len(active) else None,
        "gram": gram.tolist(),
        "cosine_gram": cosine.tolist(),
        "column_norms": norms.tolist(),
    }


def pair_cosines(terms: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    names = ("goal_flow", "goal_relation", "flow_relation")
    pairs = ((0, 1), (0, 2), (1, 2))
    output = {}
    for label, (first, second) in zip(names, pairs, strict=True):
        values = []
        degenerate = 0
        for a, b in zip(terms[first], terms[second], strict=True):
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator <= 1e-12:
                degenerate += 1
            else:
                values.append(float(np.dot(a, b) / denominator))
        output[label] = {"values": values, "degenerate_agent_rows": degenerate}
    return output


def snapshot_at(config: Config, positions: np.ndarray, velocities: np.ndarray, regime: str) -> tuple[DoubleBottleneckEnv, dict]:
    env = DoubleBottleneckEnv(config)
    env.reset(positions, regime=regime)
    state = env.augmented_state()
    state["last_applied_velocity"] = np.asarray(velocities, dtype=float)
    env.restore_augmented_state(state)
    return env, env.snapshot()


def sensitivity(projector, snapshot, name, theta, positions, goals, u_safe, max_speed, scale):
    theta = np.asarray(theta, dtype=float)
    raw_columns, executable_columns = [], []
    removal = []
    responses = []
    widths = P0_HIGH - P0_LOW
    raw_center = correction(name, theta, positions, goals, u_safe, max_speed, scale)
    exec_center = np.asarray(projector(snapshot, u_safe + raw_center).velocity)
    for index in range(3):
        delta = FD_NORMALIZED * widths[index]
        lower, upper = theta.copy(), theta.copy()
        lower[index] -= delta
        upper[index] += delta
        raw_low = correction(name, lower, positions, goals, u_safe, max_speed, scale)
        raw_high = correction(name, upper, positions, goals, u_safe, max_speed, scale)
        exec_low = np.asarray(projector(snapshot, u_safe + raw_low).velocity) - u_safe
        exec_high = np.asarray(projector(snapshot, u_safe + raw_high).velocity) - u_safe
        raw_columns.append(((raw_high - raw_low) / (2 * delta)).reshape(-1))
        executable_columns.append(((exec_high - exec_low) / (2 * delta)).reshape(-1))
        raw_step = correction(name, np.eye(3)[index] * delta, positions, goals, u_safe, max_speed, scale)
        exec_step = np.asarray(projector(snapshot, u_safe + raw_center + raw_step).velocity) - exec_center
        raw_norm = float(np.linalg.norm(raw_step))
        removed = float(np.linalg.norm(exec_step - raw_step))
        removal.append(removed / max(raw_norm, 1e-12))
        responses.append(exec_step.reshape(-1))
    raw_matrix = np.stack(raw_columns, axis=1)
    executable_matrix = np.stack(executable_columns, axis=1)
    response_matrix = np.stack(responses, axis=1)
    return {
        "raw": matrix_stats(raw_matrix),
        "executable": matrix_stats(executable_matrix),
        "finite_step_executable_response": matrix_stats(response_matrix),
        "fraction_raw_perturbation_removed_by_dimension": removal,
    }


def main() -> int:
    anchors = json.loads((OUT / "state_anchor_manifest.json").read_text())["anchors"]
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    lookup = {(name, family): dataset.by_family[family][0] for name, dataset in datasets.items() for family in dataset.family_names}
    policy, _ = load_checkpoint(CHECKPOINT, next(iter({dataset.environment_fingerprint for dataset in datasets.values()})))
    projector = CertifiedHardSafetyFilter()
    p0_basin = {row["episode_id"]: row for row in json.loads((ETA3 / "long_run_v2/timeout_basin_matrix.json").read_text())["episodes"]}
    states = []
    u_squared, perp_squared = [], []
    for anchor in anchors:
        path = ROOT / anchor["trajectory_path"]
        with np.load(path, allow_pickle=False) as trace:
            positions_all = np.asarray(trace["positions"], dtype=float)
            step = int(anchor["source_step"])
            positions = positions_all[step]
            episode = lookup[(anchor["set"], anchor["family_id"])]
            config = Config(**datasets[anchor["set"]].config)
            previous_velocity = episode.initial_velocities if step == 0 else (positions_all[step] - positions_all[step - 1]) / config.dt
            env, snapshot = snapshot_at(config, positions, previous_velocity, anchor["regime"])
            if anchor["trajectory_class"] in ("safe_timeout", "baseline_success"):
                u_safe = np.asarray(trace["executed_actions"][step], dtype=float)
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(int(anchor["seed"])), int(anchor["rollout_id"]))
                raw_flow = np.asarray(policy.sample_actions(env.observation()[None], jax.random.fold_in(key, step))[0], dtype=float)
                u_safe = np.asarray(projector(snapshot, _radial_bound64(raw_flow, config.max_speed)).velocity, dtype=float)
            goal, flow, relation = basis_terms("P0-3D", positions, env.goals, u_safe, config.max_speed, 1.0)
            perp_raw = raw_ortho_flow(goal, u_safe, config.max_speed)
            if anchor["trajectory_class"] != "successful_eta_corrected":
                u_squared.extend(np.sum(u_safe * u_safe, axis=1).tolist())
                perp_squared.extend(np.sum(perp_raw * perp_raw, axis=1).tolist())
            states.append({"anchor": anchor, "positions": positions, "goals": env.goals.copy(), "u_safe": u_safe, "snapshot": snapshot, "max_speed": config.max_speed, "p0_terms": (goal, flow, relation), "perp_raw": perp_raw})
    scale = float(np.sqrt(np.mean(u_squared) / np.mean(perp_squared)))
    scale_artifact = {
        "schema": "ortho_flow_global_scale_v1",
        "scale": scale,
        "formula": "sqrt(mean(||u_safe||^2)/mean(||B_flow_perp_raw||^2))",
        "baseline_anchor_states": 96,
        "agent_rows": len(u_squared),
        "u_safe_rms": float(np.sqrt(np.mean(u_squared))),
        "perp_raw_rms": float(np.sqrt(np.mean(perp_squared))),
        "success_results_used": False,
    }
    (OUT / "P1_SCALE.json").write_text(json.dumps(scale_artifact, indent=2, sort_keys=True) + "\n")

    results = []
    for state in states:
        anchor = state["anchor"]
        p0_terms = state["p0_terms"]
        p1_terms = basis_terms("P1-OrthoFlow3", state["positions"], state["goals"], state["u_safe"], state["max_speed"], scale)
        record = {
            **anchor,
            "P0_raw_joint": matrix_stats(np.stack([term.reshape(-1) for term in p0_terms], axis=1)),
            "P1_raw_joint": matrix_stats(np.stack([term.reshape(-1) for term in p1_terms], axis=1)),
            "P0_per_agent_matrix": [matrix_stats(np.stack([term[i] for term in p0_terms], axis=1)) for i in range(4)],
            "P1_per_agent_matrix": [matrix_stats(np.stack([term[i] for term in p1_terms], axis=1)) for i in range(4)],
            "P0_per_agent_cosines": pair_cosines(p0_terms),
            "P1_per_agent_cosines": pair_cosines(p1_terms),
            "basis_norms": {
                "goal": np.linalg.norm(p0_terms[0], axis=1).tolist(),
                "u_safe": np.linalg.norm(p0_terms[1], axis=1).tolist(),
                "relation": np.linalg.norm(p0_terms[2], axis=1).tolist(),
                "flow_perp_raw": np.linalg.norm(state["perp_raw"], axis=1).tolist(),
                "flow_perp_scaled": np.linalg.norm(p1_terms[1], axis=1).tolist(),
            },
            "P1_degenerate_agent_rows": int(np.sum(np.linalg.norm(p1_terms[1], axis=1) <= 1e-6)),
            "sensitivity_eta_zero": {
                name: sensitivity(projector, state["snapshot"], name, np.zeros(3), state["positions"], state["goals"], state["u_safe"], state["max_speed"], scale)
                for name in ("P0-3D", "P1-OrthoFlow3")
            },
        }
        reference = None
        if anchor["trajectory_class"] == "successful_eta_corrected":
            reference = anchor["eta_theta"]
        elif anchor["episode_id"] in p0_basin and p0_basin[anchor["episode_id"]]["eta_rep"] is not None:
            reference = p0_basin[anchor["episode_id"]]["eta_rep"]["theta"]
        if reference is not None:
            record["P0_successful_eta_reference"] = reference
            record["P0_sensitivity_successful_eta"] = sensitivity(projector, state["snapshot"], "P0-3D", np.asarray(reference), state["positions"], state["goals"], state["u_safe"], state["max_speed"], scale)
        results.append(record)
    raw_path = OUT / "offline/basis_state_audit.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))

    def collect_pair(name, representation, trajectory_class):
        values, degenerate = [], 0
        for row in results:
            if row["trajectory_class"] != trajectory_class:
                continue
            item = row[f"{representation}_per_agent_cosines"][name]
            values.extend(item["values"])
            degenerate += item["degenerate_agent_rows"]
        return {"cosine": stats(values), "degenerate_agent_rows": degenerate}

    classes = ("baseline_success", "safe_timeout", "successful_eta_corrected")
    summary = {
        "schema": "eta_basis_geometry_and_projection_audit_v1",
        "states": len(results),
        "ortho_scale": scale_artifact,
        "by_trajectory_class": {},
    }
    for trajectory_class in classes:
        group = [row for row in results if row["trajectory_class"] == trajectory_class]
        summary["by_trajectory_class"][trajectory_class] = {
            "states": len(group),
            "P0_per_agent_cosines": {name: collect_pair(name, "P0", trajectory_class) for name in ("goal_flow", "goal_relation", "flow_relation")},
            "P1_per_agent_cosines": {name: collect_pair(name, "P1", trajectory_class) for name in ("goal_flow", "goal_relation", "flow_relation")},
            "P0_joint_goal_flow_cosine": stats(row["P0_raw_joint"]["cosine_gram"][0][1] for row in group),
            "P1_joint_goal_flow_cosine": stats(row["P1_raw_joint"]["cosine_gram"][0][1] for row in group),
            "P0_raw_rank": stats(row["P0_raw_joint"]["effective_rank"] for row in group),
            "P1_raw_rank": stats(row["P1_raw_joint"]["effective_rank"] for row in group),
            "P0_executable_rank_eta0": stats(row["sensitivity_eta_zero"]["P0-3D"]["executable"]["effective_rank"] for row in group),
            "P1_executable_rank_eta0": stats(row["sensitivity_eta_zero"]["P1-OrthoFlow3"]["executable"]["effective_rank"] for row in group),
            "P0_executable_condition_eta0": stats(row["sensitivity_eta_zero"]["P0-3D"]["executable"]["condition_number_effective"] for row in group),
            "P1_executable_condition_eta0": stats(row["sensitivity_eta_zero"]["P1-OrthoFlow3"]["executable"]["condition_number_effective"] for row in group),
            "P1_degenerate_agent_fraction": sum(row["P1_degenerate_agent_rows"] for row in group) / max(1, 4 * len(group)),
        }
    successful_ref = [row for row in results if "P0_sensitivity_successful_eta" in row]
    summary["P0_successful_eta_sensitivity"] = {
        "states": len(successful_ref),
        "executable_rank": stats(row["P0_sensitivity_successful_eta"]["executable"]["effective_rank"] for row in successful_ref),
        "condition_number": stats(row["P0_sensitivity_successful_eta"]["executable"]["condition_number_effective"] for row in successful_ref),
    }
    (OUT / "basis_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({"state": "basis_audit_complete", "ortho_scale": scale, "basis_audit_states": len(results)})
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"states": len(results), "scale": scale, "classes": {name: summary["by_trajectory_class"][name]["states"] for name in classes}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
