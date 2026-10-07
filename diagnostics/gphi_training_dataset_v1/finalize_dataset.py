"""Construct the immutable pilot dataset from completed oracle rollouts.

No controller, plant, monitor, projection, or event code is modified here.
The script only aggregates isolated diagnostics, audits the stored first step,
and writes deployment-available inputs plus executed-correction labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_v1.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config


OUTCOMES = ("success", "deadlock", "timeout", "collision")
VMAX_EPS = 1e-12
BASELINE_RECORDS = (
    "baseline_normal",
    "baseline_pre_deadlock",
    "baseline_recovery",
)
CANDIDATE_RECORDS = (
    "candidate_normal",
    "candidate_pre_resume_0_gpu_cap10",
    "candidate_pre_resume_1_gpu_cap10",
    "candidate_pre_resume_2_cpu",
    "candidate_pre_resume_3_cpu",
    "candidate_recovery_resume_0_cpu",
    "candidate_recovery_resume_1_gpu_cap10",
    "candidate_recovery_resume_2_cpu",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eta_key(values) -> tuple[float, float, float]:
    return tuple(round(float(value), 12) for value in values)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def same_record(a: dict, b: dict) -> bool:
    if not (
        a["outcome"] == b["outcome"]
        and int(a["steps"]) == int(b["steps"])
        # Independent CPU/GPU replays can accumulate tiny conic-solver roundoff.
        # The audited duplicate maximum is 1.90e-8 with identical outcome/steps.
        and abs(float(a["J_def"]) - float(b["J_def"])) <= 1e-7
    ):
        return False
    first_a, first_b = a.get("first_step"), b.get("first_step")
    if first_a is None or first_b is None:
        return first_a is first_b
    for field in ("u_flow", "u_safe", "g_raw", "u_exec"):
        if not np.allclose(first_a[field], first_b[field], atol=1e-10, rtol=0):
            return False
    return True


def load_effective_records() -> tuple[dict, dict]:
    sources = [HERE / "raw" / stage / "records.jsonl" for stage in BASELINE_RECORDS]
    sources.extend(
        (
            HERE / "candidate_resume_after_gpu_stop_reuse.jsonl",
            HERE / "raw/candidate_normal/records.jsonl",
        )
    )
    sources.extend(HERE / "raw" / stage / "records.jsonl" for stage in CANDIDATE_RECORDS[1:])
    effective = {}
    duplicates = []
    source_counts = {}
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        source_counts[str(path.relative_to(HERE))] = len(rows)
        for row in rows:
            key = (row["state_id"], eta_key(row["eta"]), int(row["seed"]))
            if key in effective:
                if not same_record(effective[key], row):
                    raise AssertionError(("conflicting tuple", key, path))
                duplicates.append(key)
            else:
                effective[key] = row
    return effective, {"source_counts": source_counts, "exact_duplicates": len(duplicates)}


def paired_comparison(candidate: list[dict], best: list[dict]) -> dict:
    a = {int(row["seed"]): row for row in candidate if row["outcome"] == "success"}
    b = {int(row["seed"]): row for row in best if row["outcome"] == "success"}
    seeds = sorted(set(a) & set(b))
    differences = np.asarray([float(a[s]["J_def"]) - float(b[s]["J_def"]) for s in seeds])
    if not len(differences):
        return {
            "paired_n": 0,
            "candidate_minus_best_mean": None,
            "two_sided_paired_t_95_CI": None,
            "indistinguishable_from_zero": False,
        }
    mean = float(differences.mean())
    if len(differences) == 1:
        interval = [mean, mean]
    elif float(differences.std(ddof=1)) == 0.0:
        interval = [mean, mean]
    else:
        half = float(t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / math.sqrt(len(differences)))
        interval = [mean - half, mean + half]
    return {
        "paired_n": len(differences),
        "candidate_minus_best_mean": mean,
        "two_sided_paired_t_95_CI": interval,
        "indistinguishable_from_zero": interval[0] <= 0.0 <= interval[1],
        "common_success_seeds": seeds,
    }


def pairwise_mean(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
    upper = np.triu_indices(len(vectors), 1)
    return float(distances[upper].mean())


def state_oracles(effective: dict, states: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for (state_id, eta, _), row in effective.items():
        grouped[(state_id, eta)].append(row)
    results = []
    lookup = {}
    for state in states:
        cells = []
        for (state_id, eta), rows in sorted(grouped.items()):
            if state_id != state["state_id"]:
                continue
            rows = sorted(rows, key=lambda row: int(row["seed"]))
            counts = Counter(row["outcome"] for row in rows)
            complete = len(rows) == 64 and not any(row.get("execution_error") for row in rows)
            feasible = complete and counts["success"] >= 63
            successful = [row for row in rows if row["outcome"] == "success"]
            cells.append(
                {
                    "eta": list(eta),
                    "evaluated": len(rows),
                    "counts": {outcome: counts[outcome] for outcome in OUTCOMES},
                    "execution_error": counts["execution_error"],
                    "complete_matched_64": complete,
                    "B_63_member": feasible,
                    "mean_J_def_success": float(np.mean([row["J_def"] for row in successful])) if successful else None,
                    "std_J_def_success": float(np.std([row["J_def"] for row in successful], ddof=1)) if len(successful) > 1 else 0.0 if successful else None,
                    "mean_episode_steps_success": float(np.mean([row["steps"] for row in successful])) if successful else None,
                }
            )
        feasible_cells = [cell for cell in cells if cell["B_63_member"]]
        if not feasible_cells:
            result = {
                "state_id": state["state_id"],
                "category": state["category"],
                "B_63_empty": True,
                "eta_best": None,
                "E_near": [],
                "cells": cells,
            }
            results.append(result)
            lookup[state["state_id"]] = result
            continue
        best = min(feasible_cells, key=lambda cell: (cell["mean_J_def_success"], cell["eta"]))
        best_eta = eta_key(best["eta"])
        best_rows = grouped[(state["state_id"], best_eta)]
        comparisons = []
        near = []
        for cell in feasible_cells:
            eta = eta_key(cell["eta"])
            if eta == best_eta:
                comparison = {
                    "eta": list(eta),
                    "paired_n": sum(row["outcome"] == "success" for row in best_rows),
                    "candidate_minus_best_mean": 0.0,
                    "two_sided_paired_t_95_CI": [0.0, 0.0],
                    "indistinguishable_from_zero": True,
                }
            else:
                comparison = {
                    "eta": list(eta),
                    **paired_comparison(grouped[(state["state_id"], eta)], best_rows),
                }
            comparisons.append(comparison)
            if comparison["indistinguishable_from_zero"]:
                near.append(list(eta))
        result = {
            "state_id": state["state_id"],
            "category": state["category"],
            "B_63_empty": False,
            "B_63": [cell["eta"] for cell in feasible_cells],
            "eta_best": best["eta"],
            "J_min": best["mean_J_def_success"],
            "E_near": near,
            "paired_comparisons": comparisons,
            "cells": cells,
        }
        results.append(result)
        lookup[state["state_id"]] = result
    return results, lookup


class FeatureBuilder:
    def __init__(self) -> None:
        self.schema: list[dict] = []
        self._frozen = False

    def build(self, env, first: dict, config: Config, cbf: CBFConfig) -> tuple[np.ndarray, dict]:
        observation = np.asarray(env.observation(), dtype=np.float64)
        positions = np.asarray(env.positions, dtype=np.float64)
        velocities = np.asarray(env.velocities, dtype=np.float64)
        goal_relative = np.asarray(env.goals - env.positions, dtype=np.float64)
        relative_position = np.asarray(observation[:, 6:8], dtype=np.float64)
        relative_velocity = np.asarray(observation[:, 8:10], dtype=np.float64)
        u_flow = np.asarray(first["u_flow"], dtype=np.float64).reshape(2, 2)
        u_safe = np.asarray(first["u_safe"], dtype=np.float64).reshape(2, 2)
        goal_basis = DiagnosticCorrector._bounded_rows(observation[:, 4:6], config.max_speed)
        relative_basis = DiagnosticCorrector._bounded_rows(-observation[:, 6:8], config.max_speed)
        errors = np.linalg.norm(env.goals - env.positions, axis=-1)
        history = np.asarray(env.distance_history[-41:], dtype=np.float64)
        if history.shape != (41, 2):
            raise AssertionError(("history shape", history.shape))
        recent_progress = history[0] - history[-1]
        candidate_active = env.candidate_since is not None
        candidate_age = 0.0 if not candidate_active else (env.step_count - env.candidate_since) * config.dt
        A, lower, geometry = barrier_constraints(env.snapshot(), cbf)
        linear_residuals = A @ u_safe.reshape(4) - lower
        speeds = np.linalg.norm(u_safe, axis=-1)
        projection_delta = u_safe - u_flow

        segments = (
            ("observation", observation, "mixed [m,m/s] in frozen observation column order"),
            ("positions", positions, "m"),
            ("last_executed_velocities", velocities, "m/s"),
            ("goal_relative", goal_relative, "m"),
            ("inter_agent_relative_position", relative_position, "m"),
            ("inter_agent_relative_velocity", relative_velocity, "m/s"),
            ("u_flow", u_flow, "m/s"),
            ("u_safe", u_safe, "m/s"),
            ("B_goal", goal_basis, "m/s"),
            ("B_rel", relative_basis, "m/s"),
            ("physical_timestep", np.asarray([env.step_count]), "step"),
            ("episode_time", np.asarray([env.step_count * config.dt]), "s"),
            ("normalized_episode_step", np.asarray([env.step_count / config.max_steps]), "unitless"),
            ("normalized_remaining_horizon", np.asarray([(config.max_steps - env.step_count) / config.max_steps]), "unitless"),
            ("goal_errors", errors, "m"),
            ("recent_progress_2s", recent_progress, "m"),
            ("window_ready", np.asarray([float(env.step_count >= round(config.progress_window_seconds / config.dt))]), "boolean"),
            ("candidate_active", np.asarray([float(candidate_active)]), "boolean"),
            ("candidate_since_step", np.asarray([-1 if env.candidate_since is None else env.candidate_since]), "step; -1 means inactive"),
            ("candidate_age", np.asarray([candidate_age]), "s"),
            ("stuck_timer", np.asarray([env.stuck_timer]), "s"),
            ("max_stuck_timer", np.asarray([env.max_stuck_timer]), "s"),
            ("ever_candidate_deadlock", np.asarray([float(env.ever_candidate_deadlock)]), "boolean"),
            ("history_start_step", np.asarray([env.step_count - len(history) + 1]), "step"),
            ("goal_error_history_tail_41", history, "m"),
            ("first_projection_delta", projection_delta, "m/s"),
            ("first_projection_delta_norm", np.asarray([np.linalg.norm(projection_delta)]), "m/s"),
            ("pairwise_barrier_h", np.asarray([geometry["pairwise_h"]]), "m^2"),
            ("wall_barrier_h", np.asarray(geometry["wall_h"]), "m"),
            ("first_projection_linear_residuals", linear_residuals, "frozen CBF residual units"),
            ("first_projection_active_linear", (linear_residuals <= 1e-7).astype(np.float64), "boolean"),
            ("u_safe_agent_speeds", speeds, "m/s"),
            ("first_projection_active_speed", (np.abs(speeds - config.max_speed) <= 1e-7).astype(np.float64), "boolean"),
        )
        flattened = []
        schema = []
        offset = 0
        for name, values, unit in segments:
            values = np.asarray(values, dtype=np.float64)
            flat = values.reshape(-1)
            flattened.append(flat)
            schema.append({
                "name": name,
                "offset": offset,
                "length": len(flat),
                "shape": list(values.shape),
                "unit": unit,
            })
            offset += len(flat)
        if not self._frozen:
            self.schema = schema
            self._frozen = True
        elif schema != self.schema:
            raise AssertionError("feature schema changed across samples")
        structured = {
            "observation": observation,
            "positions": positions,
            "velocities": velocities,
            "goal_relative": goal_relative,
            "relative_position": relative_position,
            "relative_velocity": relative_velocity,
            "u_flow": u_flow,
            "u_safe": u_safe,
            "B_goal": goal_basis,
            "B_rel": relative_basis,
            "goal_error_history": history,
            "recent_progress": recent_progress,
            "projection_delta": projection_delta,
            "linear_residuals": linear_residuals,
        }
        return np.concatenate(flattened), structured


def main() -> None:
    started = time.monotonic()
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = read_jsonl(HERE / "state_manifest.jsonl")
    state_by_id = {row["state_id"]: row for row in states}
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    effective, inventory_audit = load_effective_records()
    oracle_rows, oracle_by_state = state_oracles(effective, states)
    write_jsonl(HERE / "oracle_search_results.jsonl", oracle_rows)

    builder = FeatureBuilder()
    sample_rows = []
    arrays = defaultdict(list)
    label_statistics = {}
    reconstruction = defaultdict(list)
    quarantined = []
    empty_states = []
    max_same_seed_u_safe = 0.0
    max_same_seed_u_flow = 0.0
    max_raw_formula_error = 0.0
    max_first_projection_replay_error = 0.0
    max_second_projection_replay_error = 0.0
    min_target_linear_residual = float("inf")
    max_target_speed_excess = -float("inf")

    for state in states:
        state_id = state["state_id"]
        oracle = oracle_by_state[state_id]
        if oracle["B_63_empty"]:
            empty_states.append(state_id)
            label_statistics[state_id] = {
                "classification": "B63_EMPTY",
                "usable": False,
                "B_63_size": 0,
                "E_near_size": 0,
                "eta_values_evaluated": len(oracle["cells"]),
                "continuation_records_considered": sum(cell["evaluated"] for cell in oracle["cells"]),
            }
            continue
        near = [eta_key(eta) for eta in oracle["E_near"]]
        eta_rows = {
            eta: {
                int(seed): effective[(state_id, eta, int(seed))]
                for seed in sorted(
                    key[2] for key in effective
                    if key[0] == state_id and key[1] == eta
                )
            }
            for eta in near
        }
        common = sorted(set.intersection(*(set(rows) for rows in eta_rows.values())))
        if len(common) != 64:
            raise AssertionError((state_id, "near-optimal cohort is not matched 64", len(common)))
        env = restore_full(HERE / state["state_file"], config)
        observation = np.asarray(env.observation(), dtype=np.float64)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        target_by_seed = {}
        same_seed_distances = []
        same_seed_raw_distances = []
        same_seed_cosines = []
        correction_norms = []
        component_variances = []
        labels_by_eta = defaultdict(list)

        for seed in common:
            first_steps = [eta_rows[eta][seed]["first_step"] for eta in near]
            if any(first is None for first in first_steps):
                raise AssertionError((state_id, seed, "missing first step"))
            safe_stack = np.stack([np.asarray(first["u_safe"], dtype=np.float64) for first in first_steps])
            flow_stack = np.stack([np.asarray(first["u_flow"], dtype=np.float64) for first in first_steps])
            max_same_seed_u_safe = max(max_same_seed_u_safe, float(np.max(np.ptp(safe_stack, axis=0))))
            max_same_seed_u_flow = max(max_same_seed_u_flow, float(np.max(np.ptp(flow_stack, axis=0))))
            safe = safe_stack[0].reshape(2, 2)
            flow = flow_stack[0].reshape(2, 2)
            safe_replay, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            max_first_projection_replay_error = max(
                max_first_projection_replay_error,
                float(np.max(np.abs(safe_replay - safe))),
            )
            executed_labels = []
            raw_labels = []
            for eta, first in zip(near, first_steps):
                raw = np.asarray(first["g_raw"], dtype=np.float64).reshape(2, 2)
                executed = np.asarray(first["u_exec"], dtype=np.float64).reshape(2, 2)
                recomputed_raw = DiagnosticCorrector(DiagnosticPhi(*eta))(observation, safe, config.max_speed)
                max_raw_formula_error = max(max_raw_formula_error, float(np.max(np.abs(recomputed_raw - raw))))
                replay, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
                max_second_projection_replay_error = max(
                    max_second_projection_replay_error,
                    float(np.max(np.abs(replay - executed))),
                )
                g_exec = executed - safe
                executed_labels.append(g_exec.reshape(4))
                raw_labels.append(raw.reshape(4))
                labels_by_eta[eta].append(g_exec.reshape(4))
            executed_stack = np.stack(executed_labels)
            raw_stack = np.stack(raw_labels)
            target = executed_stack.mean(axis=0)
            target_action = safe.reshape(4) + target
            min_target_linear_residual = min(min_target_linear_residual, float(np.min(A @ target_action - lower)))
            max_target_speed_excess = max(
                max_target_speed_excess,
                float(np.max(np.linalg.norm(target_action.reshape(2, 2), axis=-1) - config.max_speed)),
            )
            target_by_seed[seed] = {
                "target": target,
                "u_flow": flow.reshape(4),
                "u_safe": safe.reshape(4),
                "individual": executed_stack,
            }
            component_variances.append(np.var(executed_stack, axis=0))
            correction_norms.extend(np.linalg.norm(executed_stack, axis=1).tolist())
            for i, j in combinations(range(len(near)), 2):
                distance = float(np.linalg.norm(executed_stack[i] - executed_stack[j]))
                raw_distance = float(np.linalg.norm(raw_stack[i] - raw_stack[j]))
                same_seed_distances.append(distance)
                same_seed_raw_distances.append(raw_distance)
                denom = float(np.linalg.norm(executed_stack[i]) * np.linalg.norm(executed_stack[j]))
                if denom > VMAX_EPS:
                    same_seed_cosines.append(float(np.dot(executed_stack[i], executed_stack[j]) / denom))

        typical_norm = float(np.median(correction_norms)) if correction_norms else 0.0
        mean_distance = float(np.mean(same_seed_distances)) if same_seed_distances else 0.0
        max_distance = float(np.max(same_seed_distances)) if same_seed_distances else 0.0
        flow_seed_variability = float(np.mean([
            pairwise_mean(np.stack(labels_by_eta[eta])) for eta in near
        ]))
        mean_over_typical = mean_distance / max(typical_norm, VMAX_EPS)
        if len(near) == 1 or (
            max_distance / config.max_speed <= 0.01
            and mean_over_typical <= 0.05
            and mean_distance <= flow_seed_variability + 1e-15
        ):
            classification = "LABEL_STABLE"
        elif max_distance / config.max_speed <= 0.10 and mean_over_typical <= 0.50:
            classification = "LABEL_MILDLY_AMBIGUOUS"
        else:
            classification = "LABEL_MULTIVALUED"
            quarantined.append(state_id)

        label_statistics[state_id] = {
            "classification": classification,
            "usable": classification != "LABEL_MULTIVALUED",
            "B_63_size": len(oracle["B_63"]),
            "E_near_size": len(near),
            "eta_values_evaluated": len(oracle["cells"]),
            "continuation_records_considered": sum(cell["evaluated"] for cell in oracle["cells"]),
            "eta_best": oracle["eta_best"],
            "E_near": [list(eta) for eta in near],
            "J_min": oracle["J_min"],
            "mean_pairwise_L2_eta_same_seed": mean_distance,
            "max_pairwise_L2_eta_same_seed": max_distance,
            "max_distance_over_vmax": max_distance / config.max_speed,
            "typical_g_exec_norm": typical_norm,
            "mean_distance_over_typical_norm": mean_over_typical,
            "componentwise_variance_due_to_eta_mean_over_seeds": np.mean(component_variances, axis=0).tolist(),
            "cosine_similarity_mean": float(np.mean(same_seed_cosines)) if same_seed_cosines else 1.0,
            "cosine_similarity_min": float(np.min(same_seed_cosines)) if same_seed_cosines else 1.0,
            "mean_pairwise_raw_L2_eta_same_seed": float(np.mean(same_seed_raw_distances)) if same_seed_raw_distances else 0.0,
            "flow_seed_mean_pairwise_L2_within_eta": flow_seed_variability,
            "eta_choice_mean_distance_over_flow_seed_mean_distance": mean_distance / max(flow_seed_variability, VMAX_EPS),
            "classification_rule": "Prior audited scale rule: STABLE if singleton or max/vmax<=.01, mean/typical<=.05, eta variability<=Flow variability; MILD if max/vmax<=.10 and mean/typical<=.50; otherwise MULTIVALUED.",
        }
        if classification == "LABEL_MULTIVALUED":
            continue

        for seed in common:
            source = target_by_seed[seed]
            first = {
                "u_flow": source["u_flow"],
                "u_safe": source["u_safe"],
            }
            features, structured = builder.build(env, first, config, cbf)
            target = source["target"]
            sample_id = f"{state_id}__flow{seed}"
            sample_rows.append({
                "sample_id": sample_id,
                "state_id": state_id,
                "state_index": state["state_index"],
                "source_trajectory": state["source_trajectory"],
                "leakage_group": state["leakage_group"],
                "category": state["category"],
                "split": state["split"],
                "flow_seed": seed,
                "target_dimension": 4,
                "target_semantics": "mean executed correction u_exec-u_safe across compact E_near at identical state/Flow seed",
                "eta_best_metadata_only": oracle["eta_best"],
                "E_near_metadata_only": oracle["E_near"],
                "oracle_J_min_metadata_only": oracle["J_min"],
                "label_classification": classification,
                "zero_label": bool(np.linalg.norm(target) <= 1e-14),
            })
            arrays["features"].append(features)
            arrays["targets"].append(target)
            arrays["target_actions"].append(source["u_safe"] + target)
            arrays["state_index"].append(state["state_index"])
            arrays["flow_seed"].append(seed)
            arrays["observation"].append(structured["observation"])
            arrays["positions"].append(structured["positions"])
            arrays["velocities"].append(structured["velocities"])
            arrays["goal_relative"].append(structured["goal_relative"])
            arrays["relative_position"].append(structured["relative_position"])
            arrays["relative_velocity"].append(structured["relative_velocity"])
            arrays["u_flow"].append(structured["u_flow"])
            arrays["u_safe"].append(structured["u_safe"])
            arrays["B_goal"].append(structured["B_goal"])
            arrays["B_rel"].append(structured["B_rel"])
            arrays["goal_error_history"].append(structured["goal_error_history"])
            arrays["recent_progress"].append(structured["recent_progress"])
            arrays["first_projection_delta"].append(structured["projection_delta"])
            arrays["first_projection_linear_residuals"].append(structured["linear_residuals"])

    if not sample_rows:
        raise RuntimeError("no usable samples")
    np_arrays = {name: np.asarray(values) for name, values in arrays.items()}
    np_arrays.update(
        state_id=np.asarray([row["state_id"] for row in sample_rows]),
        split=np.asarray([row["split"] for row in sample_rows]),
        category=np.asarray([row["category"] for row in sample_rows]),
        sample_id=np.asarray([row["sample_id"] for row in sample_rows]),
    )
    np.savez_compressed(HERE / "samples.npz", **np_arrays)
    write_jsonl(HERE / "sample_metadata.jsonl", sample_rows)
    (HERE / "feature_schema.json").write_text(json.dumps({
        "feature_dimension": int(np_arrays["features"].shape[1]),
        "target_dimension": 4,
        "target_semantics": "g*_exec = mean_E_near[Pi_safe(u_safe+g_raw_eta)-u_safe] for identical z and Flow sample",
        "segments": builder.schema,
        "normalization": "none; raw SI units",
        "future_information_in_features": False,
    }, indent=2) + "\n")

    # Leakage and data-integrity checks.
    split_groups = {
        split: {row["leakage_group"] for row in sample_rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    group_overlaps = {
        f"{a}_{b}": sorted(split_groups[a] & split_groups[b])
        for a, b in combinations(("train", "validation", "test"), 2)
    }
    feature_names = [segment["name"] for segment in builder.schema]
    forbidden = ("future", "outcome", "eta", "j_def", "deadlock_time", "success_count")
    future_name_hits = [name for name in feature_names if any(token in name.lower() for token in forbidden)]
    duplicate_sample_ids = len(sample_rows) - len({row["sample_id"] for row in sample_rows})
    state_hashes = [row["state_sha256"] for row in states]
    duplicate_state_snapshots = len(state_hashes) - len(set(state_hashes))
    leakage_checks = {
        "assignment_unit": "source state / leakage_group; never Flow-seed sample",
        "split_group_counts": {split: len(groups) for split, groups in split_groups.items()},
        "split_group_overlaps": group_overlaps,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_state_snapshots": duplicate_state_snapshots,
        "future_information_feature_name_hits": future_name_hits,
        "metadata_separated_from_samples_npz_inputs": True,
        "features_all_finite": bool(np.isfinite(np_arrays["features"]).all()),
        "targets_all_finite": bool(np.isfinite(np_arrays["targets"]).all()),
        "passed": not any(group_overlaps.values()) and duplicate_sample_ids == 0 and duplicate_state_snapshots == 0 and not future_name_hits and bool(np.isfinite(np_arrays["features"]).all()) and bool(np.isfinite(np_arrays["targets"]).all()),
    }
    (HERE / "leakage_checks.json").write_text(json.dumps(leakage_checks, indent=2) + "\n")

    class_counts = Counter(info["classification"] for info in label_statistics.values())
    state_sample_counts = Counter(row["state_id"] for row in sample_rows)
    usable_states = sorted(state_sample_counts)
    category_states = Counter(state_by_id[state_id]["category"] for state_id in usable_states)
    zero_states = sorted({row["state_id"] for row in sample_rows if row["zero_label"]})
    nonzero_states = sorted(set(usable_states) - set(zero_states))
    label_summary = {
        "unique_selected_states": len(states),
        "usable_labeled_states": len(usable_states),
        "supervised_samples": len(sample_rows),
        "samples_per_usable_state": dict(Counter(state_sample_counts.values())),
        "selected_category_counts": dict(Counter(row["category"] for row in states)),
        "usable_category_counts": dict(category_states),
        "zero_label_state_count": len(zero_states),
        "nonzero_label_state_count": len(nonzero_states),
        "zero_label_sample_count": sum(row["zero_label"] for row in sample_rows),
        "nonzero_label_sample_count": sum(not row["zero_label"] for row in sample_rows),
        "B_63_empty_state_count": len(empty_states),
        "B_63_empty_state_ids": empty_states,
        "classification_counts_over_selected_states": dict(class_counts),
        "E_near_size_distribution": dict(Counter(str(len(row["E_near"])) for row in oracle_rows)),
        "quarantined_multivalued_state_ids": quarantined,
        "target_norm": {
            "min": float(np.linalg.norm(np_arrays["targets"], axis=1).min()),
            "median": float(np.median(np.linalg.norm(np_arrays["targets"], axis=1))),
            "mean": float(np.linalg.norm(np_arrays["targets"], axis=1).mean()),
            "max": float(np.linalg.norm(np_arrays["targets"], axis=1).max()),
        },
        "target_component_mean": np_arrays["targets"].mean(axis=0).tolist(),
        "target_component_std": np_arrays["targets"].std(axis=0, ddof=1).tolist(),
        "oracle_J_min": {
            "min": float(min(row["J_min"] for row in oracle_rows if not row["B_63_empty"])),
            "median": float(np.median([row["J_min"] for row in oracle_rows if not row["B_63_empty"]])),
            "max": float(max(row["J_min"] for row in oracle_rows if not row["B_63_empty"])),
        },
        "per_state": label_statistics,
    }
    (HERE / "label_statistics.json").write_text(json.dumps(label_summary, indent=2) + "\n")

    restoration = json.loads((HERE / "restoration_checks.json").read_text())
    frozen_after = {
        "environment": sha(SYSROOT / "single_integrator/environment.py"),
        "projection": sha(SYSROOT / "single_integrator/cbf.py"),
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "retry": sha(ROOT / "diagnostics/success_basin_multimodality/exact_projector.py"),
        "checkpoint": sha(Path(protocol["checkpoint"])),
    }
    frozen_unchanged = frozen_after == protocol["frozen_hashes"]
    stable_or_mild = class_counts["LABEL_STABLE"] + class_counts["LABEL_MILDLY_AMBIGUOUS"]
    usable_fraction = len(usable_states) / len(states)
    categories_nontrivial = all(category_states[category] >= 3 for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"))
    acceptance_checks = {
        "exact_state_restoration": restoration.get("status") == "PASS",
        "usable_state_fraction_at_least_75_percent": usable_fraction >= 0.75,
        "all_categories_have_at_least_3_usable_states": categories_nontrivial,
        "both_zero_and_nonzero_state_labels": bool(zero_states and nonzero_states),
        "stable_or_mild_large_majority_at_least_90_percent_of_nonempty": stable_or_mild >= 0.9 * max(1, len(states) - len(empty_states)),
        "no_leakage_or_nonfinite_data": leakage_checks["passed"],
        "frozen_hashes_unchanged": frozen_unchanged,
        "target_actions_feasible": min_target_linear_residual >= -cbf.feasibility_tol and max_target_speed_excess <= cbf.speed_tol,
    }
    readiness = "READY_FOR_PILOT_TRAINING" if all(acceptance_checks.values()) else "DATASET_NEEDS_REVISION"

    sanity = {
        "same_state_same_Flow_max_u_safe_difference_across_eta": max_same_seed_u_safe,
        "same_state_same_Flow_max_u_flow_difference_across_eta": max_same_seed_u_flow,
        "max_DiagnosticCorrector_raw_formula_error": max_raw_formula_error,
        "max_first_projection_replay_error": max_first_projection_replay_error,
        "max_second_projection_replay_error": max_second_projection_replay_error,
        "averaged_target_min_linear_residual": min_target_linear_residual,
        "averaged_target_max_speed_excess": max_target_speed_excess,
        "inventory_audit": inventory_audit,
        "frozen_hashes_before": protocol["frozen_hashes"],
        "frozen_hashes_after": frozen_after,
        "frozen_hashes_unchanged": frozen_unchanged,
        "acceptance_checks": acceptance_checks,
        "readiness": readiness,
    }
    (HERE / "sanity_checks.json").write_text(json.dumps(sanity, indent=2) + "\n")

    # Count every newly generated raw tuple exactly once; old compatible reuse is separate.
    raw_records = sorted((HERE / "raw").glob("*/records.jsonl"))
    new_by_key = {}
    new_rollout_attempts = 0
    attempted_physical_steps = 0
    duplicate_new_attempts = 0
    for path in raw_records:
        for row in read_jsonl(path):
            key = (row["state_id"], eta_key(row["eta"]), int(row["seed"]))
            new_rollout_attempts += 1
            attempted_physical_steps += int(row["steps"])
            if key in new_by_key:
                if not same_record(new_by_key[key], row):
                    raise AssertionError(("conflicting repeated new rollout", key, path))
                duplicate_new_attempts += 1
            else:
                new_by_key[key] = row
    unique_physical_steps = sum(int(row["steps"]) for row in new_by_key.values())
    # protocol.json was frozen when state construction completed and was never
    # rewritten; unlike state_manifest.jsonl, its mtime is not affected by the
    # later label-blind split rebalance.
    initial_time = (HERE / "protocol.json").stat().st_mtime
    runtime = {
        "new_full_continuation_attempts": new_rollout_attempts,
        "unique_new_state_eta_seed_tuples": len(new_by_key),
        "duplicate_new_attempts_from_interrupted_resume_race": duplicate_new_attempts,
        "attempted_new_physical_steps": attempted_physical_steps,
        "unique_new_tuple_physical_steps": unique_physical_steps,
        "compatible_old_rollouts_reused": sum(1 for _ in (HERE / "candidate_reuse.jsonl").open()),
        "end_to_end_wall_seconds_from_frozen_protocol_to_finalization": time.time() - initial_time,
        "finalization_wall_seconds": time.monotonic() - started,
        "Flow_microbenchmark_samples_per_second": {
            "cpu": json.loads((HERE / "benchmark_cpu.json").read_text())["samples_per_s"],
            "gpu": json.loads((HERE / "benchmark_gpu.json").read_text())["samples_per_s"],
        },
        "resource_policy": {
            "current_gpu_concurrency": 1,
            "current_gpu_shards": 1,
            "JAX_memory_fraction": 0.10,
            "observed_task_gpu_memory_MiB": 10314,
            "total_gpu_memory_MiB": 97887,
            "laboratory_sharing_rule": "normal=1; confirmed low use<=2; overnight confirmed idle<=4; every process memory-limited",
        },
        "savings": {
            "eta_zero_states_skipped_nonzero_search": sum(oracle_by_state[row["state_id"]]["eta_best"] == [0.0, 0.0, 0.0] for row in states if not oracle_by_state[row["state_id"]]["B_63_empty"]),
            "valid_tuple_reuse_and_adaptive_two_failure_stop": True,
            "batched_Flow_sampling": True,
            "state_parallel_CPU_and_single_capped_GPU": True,
        },
    }
    (HERE / "runtime_statistics.json").write_text(json.dumps(runtime, indent=2) + "\n")

    report = f"""# G_phi training dataset v1

## Result

**{readiness}**

- Selected exact augmented states: **{len(states)}**; usable labeled states: **{len(usable_states)}**.
- Supervised samples: **{len(sample_rows)}** (64 matched Flow samples per usable state).
- Selected categories: NORMAL 24, PRE_DEADLOCK 23, RECOVERY 20; usable: {dict(category_states)}.
- Zero/nonzero oracle states: **{len(zero_states)} / {len(nonzero_states)}**.
- B_63 empty states: **{len(empty_states)}**; multivalued quarantines: **{len(quarantined)}**.
- Label classes: {dict(class_counts)}.

## Exact stored example

Input is a {np_arrays['features'].shape[1]}-D unnormalized deployment-available vector. Its exact ordered segment schema and units are in `feature_schema.json`; structured arrays are also stored in `samples.npz`. The 4-D target is

`g*_exec(z,xi) = mean_{{eta in E_near}}[Pi_safe(u_safe + g_raw_eta) - u_safe]`

at the identical restored state and identical current Flow sample. Eta, J_def, and future continuation outcomes occur only in metadata, never in the feature vector.

## Integrity

- Restoration audit: {restoration.get('status')}.
- Frozen hashes unchanged: {frozen_unchanged}.
- Maximum same-seed u_safe discrepancy across eta: {max_same_seed_u_safe:.3e}.
- First/second projection replay errors: {max_first_projection_replay_error:.3e} / {max_second_projection_replay_error:.3e}.
- Averaged target minimum CBF residual: {min_target_linear_residual:.3e}; maximum speed excess: {max_target_speed_excess:.3e}.
- Split leakage check: {leakage_checks['passed']} (split by source/leakage group, never Flow seed).

## Runtime

- New continuation attempts: {new_rollout_attempts}; unique state/eta/seed tuples: {len(new_by_key)}; duplicate interrupted-resume attempts: {duplicate_new_attempts}.
- Attempted physical steps: {attempted_physical_steps}; unique-tuple physical steps: {unique_physical_steps}.
- Compatible old continuation tuples reused: {runtime['compatible_old_rollouts_reused']}.
- End-to-end measured wall time: {runtime['end_to_end_wall_seconds_from_frozen_protocol_to_finalization']:.1f} s.
- Resource use was batched and shared: four CPU workers during the largest phase plus one GPU shard capped at 10% JAX memory; no full-card preallocation.

This task generated data only. **G_phi was not trained.**
"""
    (HERE / "dataset_report.md").write_text(report)

    output_names = (
        "dataset_report.md", "state_manifest.jsonl", "oracle_search_results.jsonl",
        "samples.npz", "sample_metadata.jsonl", "split_manifest.json",
        "feature_schema.json", "label_statistics.json", "restoration_checks.json",
        "leakage_checks.json", "runtime_statistics.json", "sanity_checks.json",
        "loader.py", "protocol.json",
    )
    manifest = {
        "study": "deterministic G_phi pilot dataset construction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "readiness": readiness,
        "unique_states": len(states),
        "usable_labeled_states": len(usable_states),
        "supervised_samples": len(sample_rows),
        "feature_dimension": int(np_arrays["features"].shape[1]),
        "target_dimension": 4,
        "files_sha256": {name: sha(HERE / name) for name in output_names},
        "diagnostic_source_sha256": {
            name: sha(HERE / name)
            for name in (
                "build_states.py", "audit_restoration.py", "adaptive_run.py",
                "assess_oracle.py", "rebalance_splits.py", "finalize_dataset.py", "loader.py",
            )
        },
        "frozen_source_hashes": frozen_after,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "readiness": readiness,
        "states": len(states),
        "usable_states": len(usable_states),
        "samples": len(sample_rows),
        "feature_dimension": int(np_arrays["features"].shape[1]),
        "B63_empty": len(empty_states),
        "quarantined": len(quarantined),
        "classes": dict(class_counts),
        "finalization_seconds": time.monotonic() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
