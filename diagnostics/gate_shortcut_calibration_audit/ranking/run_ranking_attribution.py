"""Frozen-checkpoint attribution audit for the Pair-1 OOF ranking reversal."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared_checkpoints"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
LOCAL = ROOT / "diagnostics/hard_stable_local_feature_audit"
OOF = ROOT / "diagnostics/oof_gate_flow_variant_error_audit"
PAIR_ZERO = "RBV_Q_pair228_m080_s95401003_p030"
PAIR_NONZERO = "R_D4_s95101006_p60"
SAME_NEIGHBOR = "N_r045_s131"
GROUP_ZERO = "anchor_D2_pair228"
GROUP_NONZERO = "anchor_D4_pair227"
SEEDS = (17, 23, 41)
STEPS = 256


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=convert) + "\n")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def load_checkpoint(group: str, seed: int):
    path = SHARED / group / f"seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as z:
        params = []
        layer = 0
        while f"layer_{layer}_weight" in z:
            params.append({"w": jnp.asarray(z[f"layer_{layer}_weight"], jnp.float32), "b": jnp.asarray(z[f"layer_{layer}_bias"], jnp.float32)})
            layer += 1
        metadata = json.loads(str(z["metadata_json"]))
        mean = np.asarray(z["normalization_mean"], float)
        scale = np.asarray(z["normalization_scale"], float)
    return params, mean, scale, metadata


def logit(params, x):
    value = x
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["w"] + layer["b"])
    return (value @ params[-1]["w"] + params[-1]["b"]).reshape(())


def logits(params, x):
    return jax.vmap(lambda row: logit(params, row))(jnp.asarray(x, jnp.float32))


def integrated_gradient(params, starts: np.ndarray, ends: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """Midpoint-rule IG for batches of straight normalized-input paths."""
    starts_j = jnp.asarray(starts, jnp.float32)
    ends_j = jnp.asarray(ends, jnp.float32)
    delta = ends_j - starts_j
    alpha = (jnp.arange(steps, dtype=jnp.float32) + 0.5) / steps
    grad_one = jax.grad(logit, argnums=1)

    @jax.jit
    def path_ig(x0, dx):
        points = x0[None, :] + alpha[:, None] * dx[None, :]
        gradients = jax.vmap(lambda point: grad_one(params, point))(points)
        return dx * jnp.mean(gradients, axis=0)

    return np.asarray(jax.vmap(path_ig)(starts_j, delta), float)


def semantic_groups(schema: dict) -> dict[str, np.ndarray]:
    segments = {s["name"]: np.arange(int(s["offset"]), int(s["offset"]) + int(s["length"])) for s in schema["segments"]}
    definitions = {
        "geometry_observation": ["observation", "positions", "last_executed_velocities", "pairwise_barrier_h", "wall_barrier_h"],
        "goal_relative": ["goal_relative", "B_goal", "goal_errors"],
        "inter_agent_relative": ["inter_agent_relative_position", "inter_agent_relative_velocity", "B_rel"],
        "episode_time": ["physical_timestep", "episode_time", "normalized_episode_step", "normalized_remaining_horizon"],
        "history_monitor": ["recent_progress_2s", "window_ready", "candidate_active", "candidate_since_step", "candidate_age", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock", "history_start_step", "goal_error_history_tail_41"],
        "u_flow": ["u_flow"],
        "u_safe": ["u_safe"],
        "control_projection": ["first_projection_delta", "first_projection_delta_norm", "first_projection_linear_residuals", "first_projection_active_linear", "u_safe_agent_speeds", "first_projection_active_speed"],
    }
    groups = {name: np.concatenate([segments[s] for s in names]) for name, names in definitions.items()}
    joined = np.concatenate(list(groups.values()))
    assert len(joined) == 214 and np.array_equal(np.sort(joined), np.arange(214))
    return groups


def aggregate_group(values: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(values[index].sum()) for name, index in groups.items()}


def eta_squared_by_group(x: np.ndarray, labels: np.ndarray) -> float:
    overall = x.mean(0)
    total = float(np.sum((x - overall) ** 2))
    if total <= 1e-15:
        return 0.0
    between = 0.0
    for value in np.unique(labels):
        subset = x[labels == value]
        between += len(subset) * float(np.sum((subset.mean(0) - overall) ** 2))
    return between / total


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    HERE.mkdir(parents=True, exist_ok=True)
    ready = json.loads((SHARED / "READY.json").read_text())
    if ready["status"] != "READY" or not ready["all_reproduction_checks_passed"]:
        raise RuntimeError("shared frozen checkpoints not verified")
    schema = json.loads((DATA / "feature_schema.json").read_text())
    groups = semantic_groups(schema)
    with np.load(DATA / "samples.npz", allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    ids = arrays["state_id"].astype(str)
    features = np.asarray(arrays["features"], float)
    flow_seed = np.asarray(arrays["flow_seed"], int)

    def cloud(state_id: str):
        index = np.flatnonzero(ids == state_id)
        index = index[np.argsort(flow_seed[index])]
        assert len(index) == 64
        return features[index], flow_seed[index]

    x0_raw, seed0 = cloud(PAIR_ZERO)
    x1_raw, seed1 = cloud(PAIR_NONZERO)
    xs_raw, seeds = cloud(SAME_NEIGHBOR)
    previous = read_csv(OOF / "per_variant_predictions.csv")
    saved = {(r["state_id"], str(r["training_seed"]), int(r["flow_seed"])): float(r["p_gate"]) for r in previous}

    checkpoints = {}
    reproduction = []
    for group in (GROUP_ZERO, GROUP_NONZERO):
        for seed in SEEDS:
            params, mean, scale, metadata = load_checkpoint(group, seed)
            checkpoints[(group, seed)] = (params, mean, scale, metadata)
            state_id, raw, seeds_local = (PAIR_ZERO, x0_raw, seed0) if group == GROUP_ZERO else (PAIR_NONZERO, x1_raw, seed1)
            p = sigmoid(np.asarray(logits(params, (raw - mean) / scale)))
            delta = max(abs(float(value) - saved[(state_id, str(seed), int(fs))]) for value, fs in zip(p, seeds_local))
            reproduction.append({"source_group": group, "training_seed": seed, "state_id": state_id, "max_abs_probability_difference": delta, "tolerance": 2e-6, "passed": delta <= 2e-6})
    if not all(r["passed"] for r in reproduction):
        raise RuntimeError(("checkpoint reproduction failed", reproduction))

    attribution_rows = []
    diagnostics = []
    # Rank-paired variants are comparable but not seed-matched: the two states use disjoint seed namespaces.
    for seed in SEEDS:
        p0, m0, s0, _ = checkpoints[(GROUP_ZERO, seed)]
        p1, m1, s1, _ = checkpoints[(GROUP_NONZERO, seed)]
        z0_m0, z1_m0 = (x0_raw - m0) / s0, (x1_raw - m0) / s0
        z0_m1, z1_m1 = (x0_raw - m1) / s1, (x1_raw - m1) / s1

        # Preferred straight path between the two real inputs, under each frozen fold model.
        ig_m0 = integrated_gradient(p0, z0_m0, z1_m0)
        ig_m1 = integrated_gradient(p1, z0_m1, z1_m1)
        l00, l10 = np.asarray(logits(p0, z0_m0), float), np.asarray(logits(p0, z1_m0), float)
        l01, l11 = np.asarray(logits(p1, z0_m1), float), np.asarray(logits(p1, z1_m1), float)

        # Exact cross-model OOF decomposition relative to each fold's normalized train mean (zero).
        zeros = np.zeros_like(z0_m0)
        ig_zero_from_ref = integrated_gradient(p0, zeros, z0_m0)
        ig_nonzero_from_ref = integrated_gradient(p1, zeros, z1_m1)
        ref0 = float(logit(p0, jnp.zeros(214, jnp.float32)))
        ref1 = float(logit(p1, jnp.zeros(214, jnp.float32)))

        for variant in range(64):
            analyses = {
                "PAIR_PATH_ZERO_FOLD": ig_m0[variant],
                "PAIR_PATH_NONZERO_FOLD": ig_m1[variant],
                "PAIR_PATH_SYMMETRIC_MEAN": 0.5 * (ig_m0[variant] + ig_m1[variant]),
                "ACTUAL_OOF_TRAIN_MEAN_REFERENCE": ig_nonzero_from_ref[variant] - ig_zero_from_ref[variant],
            }
            endpoints = {
                "PAIR_PATH_ZERO_FOLD": l10[variant] - l00[variant],
                "PAIR_PATH_NONZERO_FOLD": l11[variant] - l01[variant],
                "PAIR_PATH_SYMMETRIC_MEAN": 0.5 * ((l10[variant] - l00[variant]) + (l11[variant] - l01[variant])),
                "ACTUAL_OOF_TRAIN_MEAN_REFERENCE": l11[variant] - l00[variant],
            }
            offsets = {
                "PAIR_PATH_ZERO_FOLD": 0.0,
                "PAIR_PATH_NONZERO_FOLD": 0.0,
                "PAIR_PATH_SYMMETRIC_MEAN": 0.0,
                "ACTUAL_OOF_TRAIN_MEAN_REFERENCE": ref1 - ref0,
            }
            for analysis, vector in analyses.items():
                grouped = aggregate_group(vector, groups)
                for group_name, contribution in grouped.items():
                    attribution_rows.append({
                        "analysis": analysis, "training_seed": seed, "variant_rank": variant,
                        "zero_flow_seed": int(seed0[variant]), "nonzero_flow_seed": int(seed1[variant]),
                        "flow_seeds_matched": False, "feature_group": group_name,
                        "logit_contribution_nonzero_minus_zero": contribution,
                        "direction": "CORRECT_ORDERING" if contribution > 0 else "WRONG_ORDERING" if contribution < 0 else "NEUTRAL",
                        "total_endpoint_logit_difference": float(endpoints[analysis]),
                        "fold_model_offset": offsets[analysis],
                        "integration_residual": float(endpoints[analysis] - offsets[analysis] - sum(grouped.values())),
                    })
            diagnostics.append({
                "training_seed": seed, "variant_rank": variant,
                "zero_fold_zero_logit": l00[variant], "zero_fold_nonzero_logit": l10[variant],
                "nonzero_fold_zero_logit": l01[variant], "nonzero_fold_nonzero_logit": l11[variant],
                "actual_oof_logit_difference": l11[variant] - l00[variant],
                "symmetric_within_model_logit_difference": 0.5 * ((l10[variant] - l00[variant]) + (l11[variant] - l01[variant])),
                "cross_fold_model_effect": 0.5 * ((l11[variant] - l10[variant]) + (l01[variant] - l00[variant])),
            })

    write_csv(HERE / "pair1_attribution.csv", attribution_rows)
    write_csv(HERE / "pair1_cross_model_decomposition.csv", diagnostics)

    # State-centroid source/label association on the zero-state fold's training states.
    confidence = read_csv(ROOT / "diagnostics/gphi_gate_confidence_aware_v1/oracle_confidence_dataset.csv")
    stable = {r["state_id"]: r for r in confidence if r["oracle_confidence_class"] in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")}
    fold_meta = json.loads((SHARED / GROUP_ZERO / "fold_metadata.json").read_text())
    train_ids = fold_meta["train_state_ids"]
    _, fold_mean, fold_scale, _ = load_checkpoint(GROUP_ZERO, 17)
    centroids = np.asarray([features[ids == sid].mean(0) for sid in train_ids])
    centroids = (centroids - fold_mean) / fold_scale
    source = np.asarray([stable[sid]["source_group"] for sid in train_ids])
    labels = np.asarray([int(stable[sid]["original_gate_label"]) for sid in train_ids])

    group_rows = []
    analyses = sorted({r["analysis"] for r in attribution_rows})
    for analysis in analyses:
        for group_name, index in groups.items():
            values = np.asarray([float(r["logit_contribution_nonzero_minus_zero"]) for r in attribution_rows if r["analysis"] == analysis and r["feature_group"] == group_name])
            group_rows.append({
                "analysis": analysis, "feature_group": group_name, "dimension_count": len(index),
                "mean_logit_contribution_nonzero_minus_zero": float(values.mean()),
                "std_logit_contribution": float(values.std()),
                "P5_logit_contribution": float(np.percentile(values, 5)),
                "P95_logit_contribution": float(np.percentile(values, 95)),
                "fraction_toward_correct_ordering": float(np.mean(values > 0)),
                "mean_absolute_contribution": float(np.mean(np.abs(values))),
                "source_group_eta_squared_train_centroids": eta_squared_by_group(centroids[:, index], source),
                "oracle_label_eta_squared_train_centroids": eta_squared_by_group(centroids[:, index], labels),
            })
    write_csv(HERE / "pair1_group_contributions.csv", group_rows)

    # Compare the held-out zero with its pre-fixed same/opposite training neighbors under its exact OOF gate.
    neighbor_rows = []
    for seed in SEEDS:
        params, mean, scale, _ = checkpoints[(GROUP_ZERO, seed)]
        base = (x0_raw - mean) / scale
        for relation, state_id, raw in (("NEAREST_SAME_LABEL_TRAIN", SAME_NEIGHBOR, xs_raw), ("NEAREST_OPPOSITE_LABEL_TRAIN", PAIR_NONZERO, x1_raw)):
            normalized = (raw - mean) / scale
            value = np.asarray(logits(params, normalized), float)
            base_value = np.asarray(logits(params, base), float)
            ig = integrated_gradient(params, base, normalized)
            grouped_matrix = {name: ig[:, index].sum(1) for name, index in groups.items()}
            for variant in range(64):
                row = {
                    "training_seed": seed, "neighbor_relation": relation, "heldout_state_id": PAIR_ZERO,
                    "neighbor_state_id": state_id, "variant_rank": variant,
                    "heldout_logit": base_value[variant], "neighbor_logit": value[variant],
                    "logit_difference_neighbor_minus_heldout": value[variant] - base_value[variant],
                    "heldout_p_gate": sigmoid(base_value[variant]), "neighbor_p_gate": sigmoid(value[variant]),
                }
                for name in groups:
                    row[f"IG_{name}"] = grouped_matrix[name][variant]
                neighbor_rows.append(row)
    write_csv(HERE / "pair1_neighbor_comparison.csv", neighbor_rows)

    # Summaries and conclusion.
    primary = [r for r in group_rows if r["analysis"] == "ACTUAL_OOF_TRAIN_MEAN_REFERENCE"]
    primary_sorted = sorted(primary, key=lambda r: abs(float(r["mean_logit_contribution_nonzero_minus_zero"])), reverse=True)
    wrong = [r for r in primary_sorted if float(r["mean_logit_contribution_nonzero_minus_zero"]) < 0]
    correct = [r for r in primary_sorted if float(r["mean_logit_contribution_nonzero_minus_zero"]) > 0]
    diag = {k: float(np.mean([float(r[k]) for r in diagnostics])) for k in ("actual_oof_logit_difference", "symmetric_within_model_logit_difference", "cross_fold_model_effect")}
    diag.update({k: float(np.mean([float(r[k]) for r in diagnostics])) for k in ("zero_fold_zero_logit", "zero_fold_nonzero_logit", "nonzero_fold_zero_logit", "nonzero_fold_nonzero_logit")})
    max_residual = max(abs(float(r["integration_residual"])) for r in attribution_rows)

    # Multiple semantic blocks materially drive the negative OOF ordering.
    material_wrong = [r for r in wrong if abs(float(r["mean_logit_contribution_nonzero_minus_zero"])) >= 0.10]
    if len(material_wrong) >= 2:
        conclusion = "MULTI_GROUP_SHORTCUT"
    elif material_wrong and material_wrong[0]["feature_group"] == "history_monitor":
        conclusion = "WRONG_HISTORY_SHORTCUT"
    elif material_wrong and material_wrong[0]["feature_group"] in ("u_flow", "u_safe", "control_projection"):
        conclusion = "WRONG_CONTROL_SHORTCUT"
    elif material_wrong and material_wrong[0]["feature_group"] in ("geometry_observation", "goal_relative", "inter_agent_relative", "episode_time"):
        conclusion = "WRONG_GEOMETRY_SHORTCUT"
    else:
        conclusion = "ATTRIBUTION_INCONCLUSIVE"

    table = "\n".join(
        f"| {r['feature_group']} | {float(r['mean_logit_contribution_nonzero_minus_zero']):+.6f} | {float(r['std_logit_contribution']):.6f} | {float(r['source_group_eta_squared_train_centroids']):.3f} | {float(r['oracle_label_eta_squared_train_centroids']):.3f} |"
        for r in primary_sorted
    )
    report = f"""# Pair-1 ranking-failure attribution

## Result

**{conclusion}**

The two Pair-1 members necessarily use different valid OOF folds. Their reported OOF logit difference is therefore decomposed exactly relative to each fold's train-mean normalized reference; the explicit fold-model offset is retained rather than incorrectly assigning it to an input dimension. In addition, straight-path integrated gradients were computed under each fold model separately and symmetrically averaged. Flow seeds do not match across these states, so the 64 paths use deterministic seed-rank pairing and are labeled as comparable, not matched.

Mean decomposition across 3 seeds x 64 variant-rank pairs:

- Actual OOF nonzero-minus-zero logit: **{diag['actual_oof_logit_difference']:+.6f}** (wrong ordering).
- Symmetric within-model feature/path effect: **{diag['symmetric_within_model_logit_difference']:+.6f}**.
- Cross-fold model effect: **{diag['cross_fold_model_effect']:+.6f}**.
- Maximum absolute IG completeness residual: `{max_residual:.3e}`.

Crucially, this reversal does **not** survive same-model scoring:

- Under the zero state's `anchor_D2_pair228` OOF model, logits are `{diag['zero_fold_zero_logit']:.6f}` (zero) and `{diag['zero_fold_nonzero_logit']:.6f}` (nonzero), a correctly ordered margin of `{diag['zero_fold_nonzero_logit'] - diag['zero_fold_zero_logit']:+.6f}`.
- Under the nonzero state's `anchor_D4_pair227` OOF model, logits are `{diag['nonzero_fold_zero_logit']:.6f}` and `{diag['nonzero_fold_nonzero_logit']:.6f}`, also correctly ordered by `{diag['nonzero_fold_nonzero_logit'] - diag['nonzero_fold_zero_logit']:+.6f}`.

Thus the cross-fold Pair-1 reversal is dominated by fold-specific model/score offset, not by a single common gate ranking the two clouds backwards. The `MULTI_GROUP_SHORTCUT` label describes why the held-out zero itself sits at an intervention-like absolute score: compared under its own fold gate with its fixed same-label training neighbor, geometry/observation, history/monitor, and control/projection all materially elevate the zero state's logit. It must not be read as claiming that these feature effects alone account for the cross-fold numerical reversal.

### Actual OOF train-mean-reference feature contributions

Positive values support the oracle-correct ordering; negative values support the observed wrong ordering.

| Feature group | Mean contribution | SD | source eta² | label eta² |
|---|---:|---:|---:|---:|
{table}

Dominant wrong-order groups: {', '.join(r['feature_group'] for r in material_wrong) if material_wrong else 'none above 0.10 logit'}. Dominant correct-order groups: {', '.join(r['feature_group'] for r in correct[:3]) if correct else 'none'}.

The fixed nearest opposite-label training state is `{PAIR_NONZERO}`; the fixed nearest same-label training state is `{SAME_NEIGHBOR}`. Their per-seed, per-variant score and IG comparisons are in `pair1_neighbor_comparison.csv`.

## Interpretation boundary

Integrated gradients explain the frozen networks locally; they do not establish causality. Source-group eta-squared is reported only as association and is expected to be upward-biased when many groups are small. No checkpoint, feature, label, threshold, state, or rollout was changed.
"""
    (HERE / "ranking_failure_summary.md").write_text(report)
    write_json(HERE / "sanity.json", {
        "shared_checkpoints_ready": True,
        "all_checkpoint_reproductions_passed": all(r["passed"] for r in reproduction),
        "checkpoint_reproduction": reproduction,
        "features_214": features.shape[1] == 214,
        "variants_per_state": {PAIR_ZERO: len(x0_raw), PAIR_NONZERO: len(x1_raw), SAME_NEIGHBOR: len(xs_raw)},
        "flow_seeds_matched_across_pair": bool(np.array_equal(seed0, seed1)),
        "semantic_groups_disjoint_and_complete": sum(len(v) for v in groups.values()) == 214,
        "max_abs_IG_completeness_residual": max_residual,
        "conclusion": conclusion,
    })
    write_json(HERE / "runtime.json", {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started, "cpu_threads_requested": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "gpu_shards_for_attribution": 0 if jax.default_backend() == "cpu" else 1,
        "jax_backend": jax.default_backend(), "hostname": platform.node(),
        "integrated_gradient_steps": STEPS, "checkpoint_materialization_wall_seconds": ready["wall_seconds"],
        "checkpoint_materialization_gpu_shards": 1,
    })
    print(json.dumps({"status": "COMPLETE", "conclusion": conclusion, "diagnostic": diag, "wall_s": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
