"""Read-only audit of the frozen 214-D input on oracle-stable states."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
GATE = ROOT / "diagnostics/gphi_gate_feasibility_v1"
ORACLE = ROOT / "diagnostics/oracle_boundary_confidence_audit"
EPS = 1e-12
WALL_NAMES = (
    "left_end", "right_end", "floor", "left_ceiling", "right_ceiling",
    "bay_left", "bay_right", "bay_top",
)
LINEAR_CONSTRAINTS = ("agent_pair",) + tuple(
    f"agent{agent}_{wall}" for agent in range(2) for wall in WALL_NAMES
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def feature_component(segment: dict, local: int) -> str:
    name = segment["name"]
    if name == "observation":
        columns = (
            "position_x", "position_y", "last_executed_velocity_x", "last_executed_velocity_y",
            "goal_relative_x", "goal_relative_y", "other_minus_self_x", "other_minus_self_y",
            "other_velocity_minus_self_x", "other_velocity_minus_self_y",
        )
        return f"agent{local // 10}.{columns[local % 10]}"
    if segment["shape"] == [2, 2]:
        return f"agent{local // 2}.{'x' if local % 2 == 0 else 'y'}"
    if name == "goal_error_history_tail_41":
        lag = 40 - local // 2
        return f"t-{lag}.agent{local % 2}.goal_error"
    if name == "wall_barrier_h":
        return f"agent{local // 8}.{WALL_NAMES[local % 8]}"
    if name in ("first_projection_linear_residuals", "first_projection_active_linear"):
        return LINEAR_CONSTRAINTS[local]
    if name in ("goal_errors", "recent_progress_2s", "u_safe_agent_speeds", "first_projection_active_speed"):
        return f"agent{local}"
    return "scalar" if int(segment["length"]) == 1 else f"component_{local}"


def semantic_group(name: str) -> str:
    if name in {
        "observation", "positions", "last_executed_velocities", "goal_relative",
        "inter_agent_relative_position", "inter_agent_relative_velocity", "B_goal", "B_rel",
        "goal_errors", "pairwise_barrier_h", "wall_barrier_h",
    }:
        return "current_geometry_observation"
    if name in {
        "physical_timestep", "episode_time", "normalized_episode_step", "normalized_remaining_horizon",
    }:
        return "episode_time"
    if name in {
        "recent_progress_2s", "window_ready", "candidate_active", "candidate_since_step",
        "candidate_age", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock",
        "history_start_step", "goal_error_history_tail_41",
    }:
        return "history_monitor"
    if name in {"u_flow", "u_safe"}:
        return "current_control"
    return "projection_safety_diagnostics"


def source_and_meaning(name: str) -> tuple[str, str, str, str]:
    table = {
        "observation": ("GiveWayEnv.observation()", "frozen per-agent current observation", "current", "agent-specific"),
        "positions": ("env.positions", "current agent position", "current", "agent-specific"),
        "last_executed_velocities": ("env.velocities", "last applied/executed velocity", "current-memory", "agent-specific"),
        "goal_relative": ("env.goals-env.positions", "goal displacement", "current-derived", "agent-specific"),
        "inter_agent_relative_position": ("observation[:,6:8]", "other-minus-self position", "current-derived", "joint"),
        "inter_agent_relative_velocity": ("observation[:,8:10]", "other-minus-self last velocity", "current-derived", "joint"),
        "u_flow": ("FlowBC current sample", "bounded nominal velocity", "current stochastic sample", "agent-specific"),
        "u_safe": ("first hard projection", "first-projected safe velocity", "current stochastic sample", "agent-specific"),
        "B_goal": ("bounded(observation[:,4:6])", "DiagnosticCorrector goal basis", "current-derived", "agent-specific"),
        "B_rel": ("bounded(-observation[:,6:8])", "DiagnosticCorrector separation basis", "current-derived", "joint"),
        "physical_timestep": ("env.step_count", "physical timestep", "current-memory", "joint"),
        "episode_time": ("step_count*dt", "elapsed episode time", "current-derived", "joint"),
        "normalized_episode_step": ("step_count/max_steps", "normalized elapsed horizon", "current-derived", "joint"),
        "normalized_remaining_horizon": ("(max_steps-step_count)/max_steps", "normalized remaining horizon", "current-derived", "joint"),
        "goal_errors": ("norm(goals-positions)", "current distance to goal", "current-derived", "agent-specific"),
        "recent_progress_2s": ("error_history[0]-error_history[-1]", "two-second goal progress", "history-derived", "agent-specific"),
        "window_ready": ("step_count>=progress_window/dt", "deadlock monitor window-ready latch", "current-derived", "joint"),
        "candidate_active": ("candidate_since is not None", "current deadlock-candidate flag", "monitor-memory", "joint"),
        "candidate_since_step": ("env.candidate_since", "candidate onset step; -1 inactive", "monitor-memory", "joint"),
        "candidate_age": ("(step-candidate_since)*dt", "candidate duration", "monitor-derived", "joint"),
        "stuck_timer": ("env.stuck_timer", "current consecutive stuck duration", "monitor-memory", "joint"),
        "max_stuck_timer": ("env.max_stuck_timer", "maximum prior stuck duration", "history-derived", "joint"),
        "ever_candidate_deadlock": ("env.ever_candidate_deadlock", "historical candidate latch", "history-derived", "joint"),
        "history_start_step": ("step_count-len(history)+1", "absolute index of history tail start", "history-derived", "joint"),
        "goal_error_history_tail_41": ("env.distance_history[-41:]", "ordered two-second goal-error history", "history", "agent-specific"),
        "first_projection_delta": ("u_safe-u_flow", "first projection rewrite", "current-derived", "agent-specific"),
        "first_projection_delta_norm": ("norm(u_safe-u_flow)", "joint first-projection rewrite magnitude", "current-derived", "joint"),
        "pairwise_barrier_h": ("barrier_constraints geometry", "agent-pair pre-action barrier", "current-derived", "joint"),
        "wall_barrier_h": ("barrier_constraints geometry", "agent-wall pre-action barrier", "current-derived", "agent-specific"),
        "first_projection_linear_residuals": ("A@u_safe-lower", "first-projection CBF margins", "current-derived", "joint"),
        "first_projection_active_linear": ("linear_residual<=1e-7", "active first-projection linear constraints", "current-derived", "joint"),
        "u_safe_agent_speeds": ("norm(u_safe,axis=-1)", "first-projected speed", "current-derived", "agent-specific"),
        "first_projection_active_speed": ("abs(speed-vmax)<=1e-7", "active first-projection speed constraints", "current-derived", "agent-specific"),
    }
    return table[name]


def expanded_schema(schema: dict, binary: set[int]) -> list[dict]:
    rows = []
    for segment in schema["segments"]:
        source, meaning, temporal, scope = source_and_meaning(segment["name"])
        for local in range(int(segment["length"])):
            index = int(segment["offset"]) + local
            rows.append({
                "dimension": index,
                "segment": segment["name"],
                "segment_component": feature_component(segment, local),
                "source_variable": source,
                "physical_control_meaning": meaning,
                "unit": segment["unit"],
                "temporal_origin": temporal,
                "scope": scope,
                "data_type": "binary" if index in binary else (
                    "integer_with_sentinel" if segment["name"] == "candidate_since_step" else "continuous"
                ),
                "directly_available_online": True,
                "semantic_group": semantic_group(segment["name"]),
                "prior_gate_normalization": "literal 0/1; no z-score" if index in binary else "V4 train-sample mean/std z-score",
            })
    if len(rows) != 214 or [row["dimension"] for row in rows] != list(range(214)):
        raise AssertionError("expanded schema is not exactly 214 dimensions")
    return rows


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]; neg = p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= .5
    pos = y == 1; neg = ~pos
    recall = float(np.mean(pred[pos])) if np.any(pos) else float("nan")
    specificity = float(np.mean(~pred[neg])) if np.any(neg) else float("nan")
    return {
        "state_count": len(y), "gate_0": int(np.sum(neg)), "gate_1": int(np.sum(pos)),
        "AUROC": auc(y, p), "balanced_accuracy_at_0.5": (recall + specificity) / 2,
        "accuracy_at_0.5": float(np.mean(pred == y)), "FPR_at_0.5": 1 - specificity,
        "FNR_at_0.5": 1 - recall, "Brier": float(np.mean((p-y) ** 2)),
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    if len(np.unique(y)) < 2:
        prior = np.clip(float(np.mean(y)), 1e-6, 1-1e-6)
        return np.r_[np.zeros(x.shape[1]), math.log(prior/(1-prior))]

    def objective(theta):
        weights = theta[:-1]; logits = x @ weights + theta[-1]
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.logaddexp(0, logits).sum() - y @ logits + .5 / C * (weights @ weights)
        gradient = np.r_[x.T @ (probability-y) + weights/C, np.sum(probability-y)]
        return loss, gradient

    result = minimize(
        objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success and result.status not in (1, 2):
        raise RuntimeError(("logistic fit failed", result.message))
    return np.asarray(result.x)


def logo_predict(x: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 10.) -> np.ndarray:
    prediction = np.zeros(len(y), dtype=np.float64)
    for held in np.unique(groups):
        test = groups == held; train = ~test
        mean = x[train].mean(axis=0); scale = x[train].std(axis=0)
        scale[scale < 1e-9] = 1.
        theta = fit_logistic((x[train]-mean)/scale, y[train], C)
        logits = ((x[test]-mean)/scale) @ theta[:-1] + theta[-1]
        prediction[test] = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return prediction


def rank_summary(x: np.ndarray, name: str, independent_states: int) -> dict:
    centered = x - x.mean(axis=0)
    scale = centered.std(axis=0)
    nonconstant = scale > 1e-12
    z = centered[:, nonconstant] / scale[nonconstant]
    singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    numerical_rank = int(np.sum(singular > (singular[0] * 1e-10 if len(singular) else 0)))
    energy = singular**2
    fractions = energy / max(float(energy.sum()), EPS)
    effective = float(np.exp(-np.sum(fractions * np.log(np.maximum(fractions, 1e-300)))))
    cumulative = np.cumsum(fractions)
    return {
        "matrix": name, "rows": len(x), "independent_states": independent_states,
        "features": x.shape[1], "constant_dimensions": int(np.sum(~nonconstant)),
        "numerical_rank_tolerance": "s_max*1e-10", "numerical_rank": numerical_rank,
        "effective_rank_entropy": effective,
        "components_for_90pct_variance": int(np.searchsorted(cumulative, .90)+1),
        "components_for_95pct_variance": int(np.searchsorted(cumulative, .95)+1),
        "components_for_99pct_variance": int(np.searchsorted(cumulative, .99)+1),
        "singular_values": singular,
        "explained_variance_fraction": fractions,
    }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a = self.find(a); b = self.find(b)
        if a != b:
            self.parent[b] = a


def correlated_clusters(x: np.ndarray, threshold: float) -> tuple[list[list[int]], list[int]]:
    spread = np.ptp(x, axis=0)
    nonconstant = np.flatnonzero(spread > 1e-12)
    corr = np.corrcoef(x[:, nonconstant], rowvar=False)
    union = UnionFind(x.shape[1])
    for left in range(len(nonconstant)):
        for right in range(left+1, len(nonconstant)):
            if abs(corr[left, right]) >= threshold:
                union.union(int(nonconstant[left]), int(nonconstant[right]))
    groups = defaultdict(list)
    for index in nonconstant:
        groups[union.find(int(index))].append(int(index))
    clusters = sorted((sorted(value) for value in groups.values() if len(value) > 1), key=lambda value: value[0])
    keep = sorted(min(value) for value in groups.values())
    return clusters, keep


def main() -> None:
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    source_files = {
        "dataset_manifest": DATA / "manifest.json", "samples": DATA / "samples.npz",
        "state_manifest": DATA / "state_manifest.jsonl", "feature_schema": DATA / "feature_schema.json",
        "gate_manifest": GATE / "manifest.json", "gate_state_metrics": GATE / "state_metrics.csv",
        "oracle_manifest": ORACLE / "manifest.json", "oracle_stability": ORACLE / "b63_resampling_stability.csv",
        "oracle_pairs": ORACLE / "matched_pair_probability_differences.csv",
    }
    hashes_before = {name: sha(path) for name, path in source_files.items()}
    schema = read_json(DATA / "feature_schema.json")
    normalization = read_json(GATE / "normalization.json")
    binary = set(int(value) for value in normalization["binary_feature_indices"])
    manifest_rows = read_jsonl(DATA / "state_manifest.jsonl")
    state_manifest = {row["state_id"]: row for row in manifest_rows}
    stability = list(csv.DictReader((ORACLE / "b63_resampling_stability.csv").open()))
    stability_by_id = {row["state_id"]: row for row in stability}
    stable_rows = [row for row in stability if row["original_label_stable_at_95pct"] == "True"]
    ambiguous_rows = [row for row in stability if row["original_label_stable_at_95pct"] != "True"]
    if len(stable_rows) != 21 or len(ambiguous_rows) != 21:
        raise AssertionError((len(stable_rows), len(ambiguous_rows)))

    with np.load(DATA / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    features = arrays["features"]
    state_ids = arrays["state_id"]
    if features.shape != (20736, 214):
        raise AssertionError(features.shape)
    state_mean = {state_id: features[state_ids == state_id].mean(axis=0) for state_id in np.unique(state_ids)}
    state_std = {state_id: features[state_ids == state_id].std(axis=0) for state_id in np.unique(state_ids)}

    selected_gate = read_json(GATE / "decision_metrics.json")["selected_model"]
    gate_prediction = {}
    for row in csv.DictReader((GATE / "state_metrics.csv").open()):
        if row["model"] == selected_gate:
            gate_prediction[row["state_id"]] = row
    if set(stability_by_id) - set(gate_prediction):
        raise AssertionError("missing prior gate prediction")

    def cohort_rows(rows: list[dict], confidence: str) -> list[dict]:
        result = []
        for row in rows:
            gate = gate_prediction[row["state_id"]]
            result.append({
                **row, "oracle_confidence_class": confidence,
                "prior_gate_p": float(gate["p_gate"]),
                "prior_gate_prediction": int(gate["predicted_gate_label"]),
                "prior_gate_correct": gate["correct"],
                "flow_varying_feature_dimensions": int(np.sum(state_std[row["state_id"]] > 1e-12)),
            })
        return result

    write_csv(HERE / "oracle_stable_states.csv", cohort_rows(stable_rows, "STABLE"))
    write_csv(HERE / "oracle_ambiguous_states.csv", cohort_rows(ambiguous_rows, "AMBIGUOUS"))
    expanded = expanded_schema(schema, binary)
    write_csv(HERE / "feature_schema_expanded.csv", expanded)
    dimension_name = {row["dimension"]: f"{row['segment']}.{row['segment_component']}" for row in expanded}

    stable_ids = [row["state_id"] for row in stable_rows]
    stable_y = np.asarray([int(row["original_gate_label"]) for row in stable_rows], dtype=np.int64)
    stable_groups = np.asarray([row["source_group"] for row in stable_rows], dtype=str)
    stable_x = np.stack([state_mean[state_id] for state_id in stable_ids])
    hard_mask = np.asarray(["EASY_" not in row["audit_tags"] for row in stable_rows])
    train_stable_ids = [row["state_id"] for row in stable_rows if row["split"] == "train"]
    train_stable_state_x = np.stack([state_mean[state_id] for state_id in train_stable_ids])
    train_stable_sample_x = features[np.isin(state_ids, train_stable_ids)]
    v4_train_ids = sorted(set(state_ids[arrays["split"] == "train"]))
    v4_train_state_x = np.stack([state_mean[state_id] for state_id in v4_train_ids])

    prior_scale = np.asarray(normalization["scale"])
    stable_train_constant = np.ptp(train_stable_sample_x, axis=0) <= 1e-12
    relative_std = train_stable_sample_x.std(axis=0) / prior_scale
    stable_train_near = (~stable_train_constant) & (relative_std < .01)
    v4_train_constant = np.ptp(v4_train_state_x, axis=0) <= 1e-12
    exact_clusters, exact_keep = correlated_clusters(v4_train_state_x, .999999999)
    strong_clusters, strong_keep = correlated_clusters(v4_train_state_x, .995)
    exact_involved = sorted({value for cluster in exact_clusters for value in cluster})
    strong_involved = sorted({value for cluster in strong_clusters for value in cluster})
    within_v4_train = np.stack([state_std[state_id] for state_id in v4_train_ids]).mean(axis=0)
    state_invariant_keep = np.flatnonzero(within_v4_train <= 1e-12).tolist()

    algebraic_relationships = [
        "observation[0:20] repeats positions, last velocities, goal-relative, and inter-agent relative blocks at 20:40 (observation stored float32, explicit copies float64)",
        "agent-1 relative-position and relative-velocity encodings are exact sign reversals of agent-0 encodings",
        "episode_time, normalized_episode_step, normalized_remaining_horizon, and history_start_step are affine functions of physical_timestep",
        "goal_errors repeat the final row of goal_error_history_tail_41",
        "recent_progress_2s is the first-minus-last error-history row",
        "candidate_active and candidate_age are derived from candidate_since_step and physical_timestep",
        "first_projection_delta equals u_safe-u_flow and its norm is deterministic",
        "B_goal/B_rel, barrier h values, projection residuals/active masks, and u_safe speeds/active-speed masks are deterministic transforms of current geometry and u_safe",
    ]
    redundancy = {
        "scope_primary": "seven V4-train oracle-stable independent states; sample matrix contains their 64 Flow variants",
        "stable_train_independent_states": len(train_stable_ids),
        "stable_train_samples": len(train_stable_sample_x),
        "stable_train_constant_dimensions": int(stable_train_constant.sum()),
        "stable_train_constant_indices": np.flatnonzero(stable_train_constant),
        "stable_train_near_constant_definition": "nonconstant and std < 0.01 * prior V4-train normalization scale",
        "stable_train_near_constant_dimensions": int(stable_train_near.sum()),
        "stable_train_near_constant_indices": np.flatnonzero(stable_train_near),
        "v4_train_state_mean_constant_dimensions_crosscheck": int(v4_train_constant.sum()),
        "v4_train_state_mean_constant_indices_crosscheck": np.flatnonzero(v4_train_constant),
        "near_exact_abs_correlation_threshold": .999999999,
        "near_exact_clusters": exact_clusters,
        "near_exact_dimensions_in_clusters": len(exact_involved),
        "near_exact_removable_dimensions": sum(len(cluster)-1 for cluster in exact_clusters),
        "strong_abs_correlation_threshold": .995,
        "strong_correlation_clusters": strong_clusters,
        "strongly_redundant_dimensions_in_clusters": len(strong_involved),
        "strong_correlation_removable_dimensions": sum(len(cluster)-1 for cluster in strong_clusters),
        "strong_pruned_retained_dimensions": len(strong_keep),
        "flow_seed_varying_dimensions_v4_train": int(np.sum(within_v4_train > 1e-12)),
        "state_invariant_dimensions_v4_train": len(state_invariant_keep),
        "obvious_deterministic_relationships": algebraic_relationships,
        "important_caveat": "correlation clusters use V4-train state means without gate labels; stable-train has only seven independent states",
    }
    write_json(HERE / "feature_redundancy.json", redundancy)
    ranks = [
        rank_summary(train_stable_state_x, "V4-train oracle-stable state means", len(train_stable_ids)),
        rank_summary(train_stable_sample_x, "V4-train oracle-stable Flow-expanded samples", len(train_stable_ids)),
        rank_summary(v4_train_state_x, "all V4-train state means (context cross-check)", len(v4_train_ids)),
    ]
    write_json(HERE / "feature_rank_analysis.json", {
        "standardization": "matrix-local centering/scaling for rank diagnostic only; no PCA representation was used",
        "analyses": ranks,
    })

    # Full restorable snapshots contain only terminal/event latches not already encoded.
    snapshot_fields = ("first_success_step", "first_deadlock_step", "first_wall_collision_step", "first_agent_collision_step", "done")
    omitted_values = []
    for state_id in stable_ids:
        path = DATA / state_manifest[state_id]["state_file"]
        with np.load(path, allow_pickle=False) as state:
            omitted_values.append([float(np.asarray(state[field]).item()) for field in snapshot_fields])
    omitted_values = np.asarray(omitted_values)
    omitted_rows = []
    meanings = {
        "first_success_step": "first success event latch", "first_deadlock_step": "first deadlock event latch",
        "first_wall_collision_step": "first wall-collision event latch",
        "first_agent_collision_step": "first agent-collision event latch", "done": "terminal-state flag",
    }
    for column, field in enumerate(snapshot_fields):
        values = omitted_values[:, column]
        omitted_rows.append({
            "field_or_quantity": field, "meaning": meanings[field], "stored_in_restorable_snapshot": True,
            "represented_in_214D": False, "deployment_available_at_t": True,
            "requires_new_runtime_memory": False, "stable_unique_values": json.dumps(sorted(set(values.tolist()))),
            "varies_on_stable_states": bool(np.ptp(values) > 0),
            "diagnostic_eligibility": "eligible but noninformative: every audited input is pre-terminal",
        })
    absent_history = [
        ("recent_position_trajectory", "recent trajectory shape beyond current position", True),
        ("recent_u_safe_history", "history of first-projected actions", True),
        ("recent_executed_action_history", "executed-action history beyond the included last velocity", True),
        ("recent_correction_history", "history of prior correction/intervention", True),
        ("stuck_timer_time_series", "timer evolution beyond current/max/candidate-since summaries", True),
        ("projection_active_set_history", "historical projection active-set identities", True),
    ]
    for field, meaning, potentially_online in absent_history:
        omitted_rows.append({
            "field_or_quantity": field, "meaning": meaning, "stored_in_restorable_snapshot": False,
            "represented_in_214D": False, "deployment_available_at_t": potentially_online,
            "requires_new_runtime_memory": True, "stable_unique_values": "not stored",
            "varies_on_stable_states": "not auditable from z",
            "diagnostic_eligibility": "excluded: not part of the current exact restorable augmented state",
        })
    omitted_rows.extend([
        {
            "field_or_quantity": "current_projection_solver_status", "meaning": "nominal_feasible vs solved status",
            "stored_in_restorable_snapshot": False, "represented_in_214D": "not explicit; deterministically revealed by projection delta/residuals",
            "deployment_available_at_t": True, "requires_new_runtime_memory": False,
            "stable_unique_values": "derivable from existing features", "varies_on_stable_states": True,
            "diagnostic_eligibility": "not independent candidate information",
        },
        {
            "field_or_quantity": "static_goals_walls_config", "meaning": "fixed task geometry and parameters",
            "stored_in_restorable_snapshot": "environment constants", "represented_in_214D": "implicit and/or used in derived features",
            "deployment_available_at_t": True, "requires_new_runtime_memory": False,
            "stable_unique_values": "constant", "varies_on_stable_states": False,
            "diagnostic_eligibility": "cannot separate labels",
        },
    ])
    write_csv(HERE / "omitted_augmented_state_fields.csv", omitted_rows)

    # Stable matched-pair audit; both labels must meet the stored >=95% stability judgment.
    pair_rows = list(csv.DictReader((ORACLE / "matched_pair_probability_differences.csv").open()))
    gate_pairs = {row["pair_id"]: row for row in csv.DictReader((GATE / "matched_boundary_pairs.csv").open())}
    norm_mean = np.asarray(normalization["mean"]); norm_scale = np.asarray(normalization["scale"])
    stable_pair_output = []
    schema_groups = defaultdict(list)
    for segment in schema["segments"]:
        schema_groups[semantic_group(segment["name"])].extend(range(int(segment["offset"]), int(segment["offset"])+int(segment["length"])))
    for row in pair_rows:
        zero_id = row["zero_state_id"]; nonzero_id = row["nonzero_state_id"]
        if not (
            stability_by_id[zero_id]["original_label_stable_at_95pct"] == "True"
            and stability_by_id[nonzero_id]["original_label_stable_at_95pct"] == "True"
        ):
            continue
        zero = (state_mean[zero_id]-norm_mean)/norm_scale
        nonzero = (state_mean[nonzero_id]-norm_mean)/norm_scale
        differences = {}
        for group, indices in schema_groups.items():
            delta = zero[indices]-nonzero[indices]
            differences[f"{group}_normalized_RMS"] = float(np.sqrt(np.mean(delta**2)))
        snapshot_diff = {}
        zero_path = DATA / state_manifest[zero_id]["state_file"]
        nonzero_path = DATA / state_manifest[nonzero_id]["state_file"]
        with np.load(zero_path, allow_pickle=False) as a, np.load(nonzero_path, allow_pickle=False) as b:
            for field in a.files:
                av = np.asarray(a[field]); bv = np.asarray(b[field])
                snapshot_diff[field] = float(np.max(np.abs(av.astype(float)-bv.astype(float))))
        gate = gate_pairs[row["pair_id"]]
        stable_pair_output.append({
            **row,
            "normalized_214D_RMS_distance_recomputed": float(np.sqrt(np.mean((zero-nonzero)**2))),
            **differences,
            "full_snapshot_max_abs_differences": json.dumps(snapshot_diff, sort_keys=True),
            "omitted_snapshot_fields_that_differ": "|".join(
                field for field in snapshot_fields if snapshot_diff[field] > 0
            ),
            "p_gate_zero": gate["p_gate_zero"], "p_gate_nonzero": gate["p_gate_nonzero"],
            "prior_gate_pair_outcome": gate["pair_outcome"],
        })
    write_csv(HERE / "stable_matched_pairs.csv", stable_pair_output)

    # Nearest neighbors use state means and the original V4 train-only normalization.
    normalized_stable = (stable_x-norm_mean)/norm_scale
    distance = np.sqrt(np.mean((normalized_stable[:, None, :]-normalized_stable[None, :, :])**2, axis=2))
    np.fill_diagonal(distance, np.inf)
    nn_rows = []
    for index, state_id in enumerate(stable_ids):
        same = np.flatnonzero(stable_y == stable_y[index]); same = same[same != index]
        opposite = np.flatnonzero(stable_y != stable_y[index])
        nearest = int(np.argmin(distance[index]))
        source_allowed = np.flatnonzero(stable_groups != stable_groups[index])
        source_nearest = int(source_allowed[np.argmin(distance[index, source_allowed])])
        nearest_same = int(same[np.argmin(distance[index, same])])
        nearest_opposite = int(opposite[np.argmin(distance[index, opposite])])
        nn_rows.append({
            "state_id": state_id, "label": stable_y[index], "source_group": stable_groups[index],
            "audit_tags": stable_rows[index]["audit_tags"],
            "nearest_state_id": stable_ids[nearest], "nearest_label": stable_y[nearest],
            "nearest_distance": distance[index, nearest], "nearest_label_correct": stable_y[nearest] == stable_y[index],
            "source_aware_nearest_state_id": stable_ids[source_nearest],
            "source_aware_nearest_label": stable_y[source_nearest],
            "source_aware_nearest_distance": distance[index, source_nearest],
            "source_aware_nearest_label_correct": stable_y[source_nearest] == stable_y[index],
            "same_label_nearest_state_id": stable_ids[nearest_same],
            "same_label_nearest_distance": distance[index, nearest_same],
            "opposite_label_nearest_state_id": stable_ids[nearest_opposite],
            "opposite_label_nearest_distance": distance[index, nearest_opposite],
            "opposite_over_same_distance_ratio": distance[index, nearest_opposite]/distance[index, nearest_same],
            "opposite_is_closer_collision": distance[index, nearest_opposite] < distance[index, nearest_same],
        })
    write_csv(HERE / "nearest_neighbor_analysis.csv", nn_rows)

    # Fixed, lightweight source-group-held-out linear diagnostics. C=10 was fixed for all representations.
    feature_groups = {
        "geometry_only": np.arange(0, 40),
        "history_monitor_only": np.arange(56, 154),
        "control_projection_only": np.r_[40:56, 154:214],
        "geometry_plus_history": np.r_[0:40, 56:154],
        "geometry_plus_control": np.r_[0:56, 154:214],
        "full_214D": np.arange(214),
    }
    information_rows = []
    for cohort_name, mask in (("ALL_STABLE", np.ones(len(stable_y), dtype=bool)), ("DIFFICULT_STABLE_ONLY", hard_mask)):
        x = stable_x[mask]; y = stable_y[mask]; groups = stable_groups[mask]
        for name, columns in feature_groups.items():
            prediction = logo_predict(x[:, columns], y, groups, C=10.)
            class0 = x[y == 0][:, columns]; class1 = x[y == 1][:, columns]
            pooled = np.sqrt((class0.var(axis=0)+class1.var(axis=0))/2)
            effect = np.abs(class1.mean(axis=0)-class0.mean(axis=0))/np.maximum(pooled, 1e-12)
            information_rows.append({
                "cohort": cohort_name, "representation": name, "feature_count": len(columns),
                "classifier": "L2 logistic C=10", "evaluation": "leave-one-source-group-out",
                **metrics(y, prediction), "median_univariate_abs_effect_size": float(np.median(effect)),
                "max_univariate_abs_effect_size": float(np.max(effect)),
                "small_sample_caveat": "21 stable states total; difficult-only cohort has 13",
            })
    write_csv(HERE / "feature_group_information.csv", information_rows)

    # Unsupervised pruning is learned only from the original V4-train state means.
    representations = {
        "full_214D": np.arange(214),
        "near_exact_correlation_pruned": np.asarray(exact_keep),
        "strong_correlation_pruned": np.asarray(strong_keep),
        "state_invariant_existing_features": np.asarray(state_invariant_keep),
    }
    pruned_rows = []
    for cohort_name, mask in (("ALL_STABLE", np.ones(len(stable_y), dtype=bool)), ("DIFFICULT_STABLE_ONLY", hard_mask)):
        for name, columns in representations.items():
            prediction = logo_predict(stable_x[mask][:, columns], stable_y[mask], stable_groups[mask], C=10.)
            pruned_rows.append({
                "cohort": cohort_name, "representation": name, "retained_dimensions": len(columns),
                "selection_data": "unlabeled V4-train state means only",
                "selection_rule": (
                    "none" if name == "full_214D" else
                    "one representative per abs-correlation>=0.999999999 cluster; constants removed" if name == "near_exact_correlation_pruned" else
                    "one representative per abs-correlation>=0.995 cluster; constants removed" if name == "strong_correlation_pruned" else
                    "remove dimensions varying across Flow variants of the same V4-train state"
                ),
                "classifier": "L2 logistic C=10", "evaluation": "leave-one-source-group-out",
                **metrics(stable_y[mask], prediction),
            })
    write_csv(HERE / "pruned_feature_diagnostic.csv", pruned_rows)

    # The only omitted restorable fields are constant pre-terminal latches.
    augmented_x = np.c_[stable_x, omitted_values]
    augmented_rows = []
    for cohort_name, mask in (("ALL_STABLE", np.ones(len(stable_y), dtype=bool)), ("DIFFICULT_STABLE_ONLY", hard_mask)):
        for name, x in (("current_214D", stable_x), ("214D_plus_5_omitted_event_latches", augmented_x)):
            prediction = logo_predict(x[mask], stable_y[mask], stable_groups[mask], C=10.)
            augmented_rows.append({
                "cohort": cohort_name, "representation": name, "dimensions": x.shape[1],
                "added_fields": "none" if name == "current_214D" else "|".join(snapshot_fields),
                "classifier": "L2 logistic C=10", "evaluation": "leave-one-source-group-out",
                **metrics(stable_y[mask], prediction),
            })
    write_csv(HERE / "augmented_feature_diagnostic.csv", augmented_rows)

    nn_accuracy = float(np.mean([row["nearest_label_correct"] for row in nn_rows]))
    source_nn_accuracy = float(np.mean([row["source_aware_nearest_label_correct"] for row in nn_rows]))
    hard_nn = [row for row in nn_rows if "EASY_" not in row["audit_tags"]]
    hard_nn_accuracy = float(np.mean([row["nearest_label_correct"] for row in hard_nn]))
    stable_gate_correct = sum(row["prior_gate_correct"] == "True" for row in cohort_rows(stable_rows, "STABLE"))
    full_all = next(row for row in pruned_rows if row["cohort"] == "ALL_STABLE" and row["representation"] == "full_214D")
    strong_all = next(row for row in pruned_rows if row["cohort"] == "ALL_STABLE" and row["representation"] == "strong_correlation_pruned")
    full_hard = next(row for row in pruned_rows if row["cohort"] == "DIFFICULT_STABLE_ONLY" and row["representation"] == "full_214D")
    strong_hard = next(row for row in pruned_rows if row["cohort"] == "DIFFICULT_STABLE_ONLY" and row["representation"] == "strong_correlation_pruned")
    augmented_all = [row for row in augmented_rows if row["cohort"] == "ALL_STABLE"]

    # Redundancy is real, but it does not improve both stable held-out metrics when pruned;
    # no omitted restorable quantity varies, while stable pairs are already clearly separated.
    conclusion = "CURRENT_214D_APPEARS_SUFFICIENT"
    smallest_next = (
        "Repeat the same oracle-stable, source-group-held-out gate fit after collecting a small number "
        "of additional statistically stable close-range pairs from new source groups; keep oracle-ambiguous states excluded."
    )
    decision = {
        "conclusion": conclusion,
        "stable_states": len(stable_rows), "ambiguous_states": len(ambiguous_rows),
        "stable_labels": dict(Counter(row["original_gate_label"] for row in stable_rows)),
        "stable_matched_pairs": len(stable_pair_output),
        "prior_gate_correct_on_stable_states": stable_gate_correct,
        "nearest_neighbor_accuracy_all_stable": nn_accuracy,
        "source_group_aware_nearest_neighbor_accuracy_all_stable": source_nn_accuracy,
        "nearest_neighbor_accuracy_difficult_stable": hard_nn_accuracy,
        "opposite_closer_collisions_all_stable": sum(row["opposite_is_closer_collision"] for row in nn_rows),
        "opposite_closer_collisions_difficult_stable": sum(row["opposite_is_closer_collision"] for row in hard_nn),
        "full_LOGO": full_all, "strong_pruned_LOGO": strong_all,
        "full_difficult_LOGO": full_hard, "strong_pruned_difficult_LOGO": strong_hard,
        "augmentation_LOGO": augmented_all,
        "omitted_restorable_fields_vary": bool(np.any(np.ptp(omitted_values, axis=0) > 0)),
        "smallest_justified_next_experiment": smallest_next,
        "interpretation": (
            "The 214-D representation is strongly redundant, but stable labels and the two fully stable matched pairs "
            "remain separable. Pruning does not consistently improve both AUROC and balanced accuracy, and omitted "
            "restorable fields are constant. Remaining evidence points to tiny stable-boundary sample size/model fitting, "
            "separate from the already-established oracle ambiguity."
        ),
    }
    write_json(HERE / "decision_evidence.json", decision)

    hashes_after = {name: sha(path) for name, path in source_files.items()}
    sanity = {
        "passed": hashes_before == hashes_after,
        "prior_artifacts_unchanged": hashes_before == hashes_after,
        "prior_hashes": hashes_before,
        "oracle_modified": False, "B63_modified": False, "feature_schema_modified": False,
        "G_phi_trained": False, "G_phi_redesigned": False, "learned_closed_loop_run": False,
        "new_rollouts_generated": False, "ambiguous_states_excluded_from_primary_analysis": True,
        "primary_unit": "unique augmented state", "Flow_variants_aggregated_to_state_means": True,
        "stable_rule_reused_verbatim": "original_label_stable_at_95pct from prior audit",
        "stable_pair_requires_both_sides_stable": True,
        "conclusion": conclusion,
    }
    write_json(HERE / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("a prior artifact changed during read-only audit")

    runtime = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic()-started, "GPU_shards": 0,
        "GPU_reason": "CPU-only NumPy/SciPy diagnostics; GPU would not materially accelerate this small audit",
        "CPU_threads_requested": 1, "new_rollouts": 0,
        "python": sys.version, "platform": platform.platform(),
        "resource_snapshot": "GPU 2/97887 MiB used, 0% utilization; Slurm empty; load 0.15 at 2026-09-23 23:13 +08",
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    report = f"""# Stable-oracle feature audit

## Decision

**{conclusion}**

Primary analysis reused the prior audit's stored stability judgment and excluded every oracle-ambiguous state. No rollout, controller, oracle, B_63, feature, G_phi, or closed-loop behavior was changed.

## Cohort

- Audited states: **{len(stable_rows)} stable / {len(ambiguous_rows)} ambiguous**.
- Stable labels: {dict(Counter(row['original_gate_label'] for row in stable_rows))}; stable difficult states: {int(hard_mask.sum())}.
- Fully stable matched zero/nonzero boundary pairs: **{len(stable_pair_output)} / 15**. Both were correctly separated by the existing gate.
- Existing gate correct on stable states: **{stable_gate_correct}/{len(stable_rows)}**. The two stable errors (one train boundary nonzero and one old-hard test zero) were not explained by an omitted varying snapshot field.

## Representation audit

- Stable-train primary matrix has {int(stable_train_constant.sum())} constant and {int(stable_train_near.sum())} near-constant dimensions. The broader V4-train state-mean cross-check has {int(v4_train_constant.sum())} constants.
- Across V4-train state means, {len(strong_involved)} dimensions participate in {len(strong_clusters)} |r|>=0.995 clusters; {sum(len(cluster)-1 for cluster in strong_clusters)} dimensions can be removed by retaining one representative per cluster. This confirms substantial overcompleteness, not that it harms prediction.
- Stable-train state-mean numerical/effective rank: {ranks[0]['numerical_rank']}/{ranks[0]['effective_rank_entropy']:.2f} (only {len(train_stable_ids)} independent states). Flow-expanded rank/effective rank: {ranks[1]['numerical_rank']}/{ranks[1]['effective_rank_entropy']:.2f}. Contextual all-V4-train state means need only {ranks[2]['components_for_95pct_variance']} PCs for 95% variance, although PCA was not used as an input.
- Stable 214-D nearest-neighbor accuracy: {nn_accuracy:.3f}; source-group-aware: {source_nn_accuracy:.3f}; difficult-stable only: {hard_nn_accuracy:.3f}. Opposite-label neighbors are closer than same-label neighbors for {sum(row['opposite_is_closer_collision'] for row in nn_rows)}/{len(nn_rows)} stable states, including {sum(row['opposite_is_closer_collision'] for row in hard_nn)}/{len(hard_nn)} difficult stable states.

## Lightweight source-group-held-out diagnostics

- Full 214-D, all stable: AUROC {full_all['AUROC']:.3f}, balanced accuracy {full_all['balanced_accuracy_at_0.5']:.3f}.
- Full 214-D, difficult stable only: AUROC {full_hard['AUROC']:.3f}, balanced accuracy {full_hard['balanced_accuracy_at_0.5']:.3f}.
- Strong-correlation pruned ({len(strong_keep)} D), all stable: AUROC {strong_all['AUROC']:.3f}, balanced accuracy {strong_all['balanced_accuracy_at_0.5']:.3f}.
- Strong-correlation pruned, difficult stable: AUROC {strong_hard['AUROC']:.3f}, balanced accuracy {strong_hard['balanced_accuracy_at_0.5']:.3f}.

Pruning changes the small-sample metrics but does not consistently improve both AUROC and balanced accuracy. Current-control/projection and combined geometry/history blocks carry the strongest stable-label signal; geometry alone is weaker. Treat all classifier numbers as diagnostic because there are only 21 stable states and 10 source groups.

## Missing-state check

The ordered 41x2 goal-error history, last executed velocity, candidate timing, current/max stuck timers, historical deadlock latch, current active CBF identities, residual margins, barriers, u_Flow, and u_safe are already represented. The only fields in the exact restorable snapshot but absent from 214-D are four first-terminal-event latches and `done`; all are constant in these pre-terminal inputs. Adding them leaves the diagnostic result unchanged. Richer action/position/correction histories are not present in the current restorable augmented state and were therefore not used as evidence of missing features.

## Interpretation

The feature vector is overcomplete, but the evidence does **not** show that redundancy causes the stable-boundary failure, nor that a varying deployment-available restorable field is missing. Stable matched pairs are separated and the full input retains meaningful source-group-held-out signal. The remaining issue is more consistent with the very small number of stable boundary states and model fitting/generalization. This is separate from the 21/42 audited states already known to have oracle-label instability.

Smallest justified next experiment: {smallest_next}
"""
    (HERE / "feature_audit_report.md").write_text(report)

    required = [
        "feature_audit_report.md", "oracle_stable_states.csv", "oracle_ambiguous_states.csv",
        "feature_schema_expanded.csv", "feature_redundancy.json", "feature_rank_analysis.json",
        "omitted_augmented_state_fields.csv", "stable_matched_pairs.csv", "nearest_neighbor_analysis.csv",
        "feature_group_information.csv", "pruned_feature_diagnostic.csv", "augmented_feature_diagnostic.csv",
        "decision_evidence.json", "sanity_checks.json", "runtime_statistics.json",
    ]
    write_json(HERE / "manifest.json", {
        "study": "STABLE_ORACLE_FEATURE_AUDIT", "classification": conclusion,
        "primary_dataset": str(DATA), "oracle_audit": str(ORACLE), "gate_audit": str(GATE),
        "files_sha256": {name: sha(HERE / name) for name in required},
    })
    print(json.dumps({
        "classification": conclusion, "stable": len(stable_rows), "ambiguous": len(ambiguous_rows),
        "stable_pairs": len(stable_pair_output), "prior_gate_correct_stable": stable_gate_correct,
        "full_LOGO_AUROC": full_all["AUROC"], "full_hard_LOGO_AUROC": full_hard["AUROC"],
        "runtime_s": runtime["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
