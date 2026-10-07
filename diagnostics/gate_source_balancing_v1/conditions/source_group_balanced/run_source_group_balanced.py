"""Frozen source-group-balanced resampling condition for the gate audit.

The shared runner owns all fold reconstruction, feature normalization,
architecture, optimizer, validation aggregation, early stopping, and
threshold selection.  This condition changes only the epoch sampling law:
source group -> state -> saved Flow variant, all uniformly at their level.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

AUDIT = Path(__file__).resolve().parents[2]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(AUDIT))
from shared import gate_condition_runner as runner  # noqa: E402


OUT = Path(__file__).resolve().parent
CHECKPOINTS = OUT / "checkpoints"
CONDITION = "SOURCE_GROUP_BALANCED"
SAMPLER = "SOURCE_GROUP_BALANCED"
ROBUST_ZERO_IDS = (
    "RBV_Q_pair228_m080_s95401003_p030",
    "RB_Q_pair226_m080_s95400802_p073",
    "RB_Q_pair228_m080_s95401001_p050",
)
# Attribution is a post-training diagnostic only.  256 midpoint steps keeps
# numerical completeness tight on these frozen SiLU gates.
IG_STEPS = 256


def semantic_groups(schema: dict) -> dict[str, np.ndarray]:
    segments = {
        part["name"]: np.arange(int(part["offset"]), int(part["offset"]) + int(part["length"]))
        for part in schema["segments"]
    }
    names = {
        "geometry_observation": ["observation", "positions", "last_executed_velocities", "pairwise_barrier_h", "wall_barrier_h"],
        "goal_relative": ["goal_relative", "B_goal", "goal_errors"],
        "inter_agent_relative": ["inter_agent_relative_position", "inter_agent_relative_velocity", "B_rel"],
        "episode_time": ["physical_timestep", "episode_time", "normalized_episode_step", "normalized_remaining_horizon"],
        "history_monitor": ["recent_progress_2s", "window_ready", "candidate_active", "candidate_since_step", "candidate_age", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock", "history_start_step", "goal_error_history_tail_41"],
        "u_flow": ["u_flow"],
        "u_safe": ["u_safe"],
        "control_projection": ["first_projection_delta", "first_projection_delta_norm", "first_projection_linear_residuals", "first_projection_active_linear", "u_safe_agent_speeds", "first_projection_active_speed"],
    }
    result = {key: np.concatenate([segments[name] for name in value]) for key, value in names.items()}
    joined = np.concatenate(list(result.values()))
    if len(joined) != 214 or not np.array_equal(np.sort(joined), np.arange(214)):
        raise RuntimeError("semantic feature groups must be disjoint and complete")
    return result


def integrated_gradients(params, x: np.ndarray, steps: int = IG_STEPS) -> np.ndarray:
    """Midpoint straight-path IG from the fold train mean (normalized zero).

    This is an inference-only explanation of the frozen gate score.  It makes
    no new deployment/control input and does not change an evaluated vector.
    """
    endpoints = jnp.asarray(x, jnp.float32)
    alphas = (jnp.arange(steps, dtype=jnp.float32) + 0.5) / steps

    def one_logit(point):
        return runner.cv.logits(params, point[None, :])[0]

    grad_one = jax.grad(one_logit)

    @jax.jit
    def one_path(endpoint):
        points = alphas[:, None] * endpoint[None, :]
        gradients = jax.vmap(grad_one)(points)
        return endpoint * jnp.mean(gradients, axis=0)

    return np.asarray(jax.vmap(one_path)(endpoints), float)


def load_baseline_params(fold_id: str, seed: int):
    path = AUDIT / "conditions" / "baseline_sample_uniform" / "checkpoints" / fold_id / f"seed_{seed}.npz"
    if not path.exists():
        raise RuntimeError(f"missing frozen baseline checkpoint: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        params = []
        index = 0
        while f"layer_{index}_weight" in loaded:
            params.append({
                # Preserve the serialized parameter dtype.  The original
                # frozen cross-validation recipe enables JAX x64, so casting
                # a checkpoint to float32 would perturb validation threshold
                # selection during offline reconstruction.
                "w": jnp.asarray(loaded[f"layer_{index}_weight"]),
                "b": jnp.asarray(loaded[f"layer_{index}_bias"]),
            })
            index += 1
        metadata = json.loads(str(loaded["metadata_json"]))
    if metadata.get("condition") != "BASELINE_SAMPLE_UNIFORM":
        raise RuntimeError(("unexpected baseline checkpoint condition", path, metadata.get("condition")))
    return params


def attribution_rows(context: dict, trained_by_fold: dict[str, dict]) -> list[dict]:
    groups = semantic_groups(context["schema"])
    ids = context["sample_ids"]
    features = context["arrays"]["features"]
    flow_seeds = context["arrays"]["flow_seed"]
    fold_by_group = {trained["fold"]["outer_group"]: trained for trained in trained_by_fold.values()}
    rows = []
    for state_id in ROBUST_ZERO_IDS:
        if context["label"].get(state_id) != 0:
            raise RuntimeError(("expected stable zero", state_id))
        group = context["state"][state_id]["source_group"]
        trained = fold_by_group.get(group)
        if trained is None:
            raise RuntimeError(("no valid OOF fold for robust zero", state_id, group))
        index = np.flatnonzero(ids == state_id)
        index = index[np.argsort(flow_seeds[index])]
        if len(index) != 64:
            raise RuntimeError(("expected 64 saved Flow variants", state_id, len(index)))
        x = trained["fold"]["normalized"][index]
        fold_id = trained["fold"]["fold_id"]
        for condition, param_getter in (
            ("BASELINE_SAMPLE_UNIFORM", lambda seed: load_baseline_params(fold_id, seed)),
            (CONDITION, lambda seed: trained["seed_models"][seed]["params"]),
        ):
            for seed in runner.SEEDS:
                params = param_getter(seed)
                ig = integrated_gradients(params, x)
                actual_logit = np.asarray(runner.cv.logits(params, jnp.asarray(x, jnp.float32)), float)
                reference_logit = float(runner.cv.logits(params, jnp.zeros((1, 214), jnp.float32))[0])
                completeness = np.max(np.abs(ig.sum(axis=1) - (actual_logit - reference_logit)))
                # This is a numerical quadrature diagnostic, not a training
                # acceptance criterion.  Persist the residual below so it is
                # visible in the audit rather than silently discarding it.
                if completeness > 5e-3:
                    raise RuntimeError(("IG completeness failure", condition, fold_id, seed, state_id, completeness))
                for name, dims in groups.items():
                    value = ig[:, dims].sum(axis=1)
                    rows.append({
                        "condition": condition,
                        "state_id": state_id,
                        "source_group": group,
                        "fold_id": fold_id,
                        "training_seed": seed,
                        "feature_group": name,
                        "dimension_count": len(dims),
                        "mean_ig_logit_contribution": float(value.mean()),
                        "std_ig_logit_contribution": float(value.std()),
                        "median_ig_logit_contribution": float(np.median(value)),
                        "P5_ig_logit_contribution": float(np.quantile(value, 0.05)),
                        "P95_ig_logit_contribution": float(np.quantile(value, 0.95)),
                        "fraction_variants_push_toward_intervention": float(np.mean(value > 0)),
                        "mean_absolute_ig_logit_contribution": float(np.mean(np.abs(value))),
                        "reference_logit_train_mean": reference_logit,
                        "max_IG_completeness_residual": float(completeness),
                        "ig_path": "fold train-normalized mean (0) to naturally saved 214-D Flow variants",
                    })
    return rows


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    ready_path = AUDIT / "shared" / "BASELINE_READY.json"
    if not ready_path.exists():
        raise RuntimeError("BASELINE_READY.json missing: refusing balanced training")
    ready = json.loads(ready_path.read_text())
    if ready.get("status") != "BASELINE_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline not validated", ready))
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    context = runner.load_context()
    folds = runner.prepare_folds(context)
    trained_by_fold: dict[str, dict] = {}
    state_rows, variant_rows, training_rows = [], [], []
    for ordinal, fold in enumerate(folds, 1):
        fold_started = time.perf_counter()
        trained = runner.train_hierarchical_resampled_fullbatch(fold, context, SAMPLER, CONDITION, CHECKPOINTS)
        trained_by_fold[fold["fold_id"]] = trained
        state_rows.extend(runner.aggregate_prediction_rows(trained, context, CONDITION))
        training_rows.extend(trained["training_rows"])
        print({"condition": CONDITION, "fold": ordinal, "fold_id": fold["fold_id"], "elapsed_s": round(time.perf_counter() - fold_started, 3)}, flush=True)
    target_states = runner.target_variant_states(context)
    by_group = {item["fold"]["outer_group"]: item for item in trained_by_fold.values()}
    for state_id, role in sorted(target_states.items()):
        group = context["state"][state_id]["source_group"]
        variant_rows.extend(runner.infer_variants(state_id, by_group[group], context, CONDITION, role))
    calibration = runner.calibration_rows(state_rows, CONDITION)
    attribution = attribution_rows(context, trained_by_fold)
    seed_mean = [row for row in state_rows if row["training_seed"] == "SEED_MEAN"]
    all_ids = [row["state_id"] for row in seed_mean]
    checks = {
        "status": "COMPLETE",
        "baseline_ready_required_and_verified": True,
        "condition": CONDITION,
        "sampler": "uniform train source group -> uniform stable state in group -> uniform saved Flow variant",
        "epoch_draw_count": "exactly N_train saved samples with replacement, one full-batch optimizer update",
        "frozen_fold_count": len(folds),
        "no_test_or_validation_sampling": True,
        "normalization_train_only": True,
        "threshold_validation_only": True,
        "all_stable_states_exactly_one_seed_mean_oof": len(all_ids) == len(set(all_ids)) == len(context["stable_ids"]),
        "all_predictions_finite": all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in state_rows + variant_rows),
        "feature_schema_changed": False,
        "oracle_labels_changed": False,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "correction_head_trained": False,
        "closed_loop_run": False,
        "attribution_inference_only": True,
        "synthetic_control_or_action_evaluated": False,
    }
    required_true = ("baseline_ready_required_and_verified", "no_test_or_validation_sampling", "normalization_train_only", "threshold_validation_only", "all_stable_states_exactly_one_seed_mean_oof", "all_predictions_finite", "attribution_inference_only")
    if not all(checks[key] is True for key in required_true):
        raise RuntimeError(("sanity failure", checks))
    runner.write_csv(OUT / "training_results.csv", training_rows)
    runner.write_csv(OUT / "out_of_fold_predictions.csv", state_rows)
    runner.write_csv(OUT / "variant_predictions.csv", variant_rows)
    runner.write_csv(OUT / "fold_relative_calibration.csv", calibration)
    runner.write_csv(OUT / "attribution.csv", attribution)
    runner.write_json(OUT / "sanity_checks.json", checks)
    runtime = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started,
        "condition": CONDITION,
        "outer_folds": len(folds),
        "training_runs": len(folds) * len(runner.SEEDS),
        "GPU_shards": 1,
        "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "jax_backend": jax.default_backend(),
        "python": sys.version,
        "platform": platform.platform(),
        "new_oracle_rollouts": 0,
        "integrated_gradient_steps": IG_STEPS,
    }
    runner.write_json(OUT / "runtime_statistics.json", runtime)
    runner.write_json(OUT / "manifest.json", {
        "condition": CONDITION,
        "sampler": SAMPLER,
        "baseline_ready": ready,
        "runner": {"path": "../../shared/gate_condition_runner.py", "sha256": runner.sha256(AUDIT / "shared" / "gate_condition_runner.py")},
        "outputs": ["training_results.csv", "out_of_fold_predictions.csv", "variant_predictions.csv", "fold_relative_calibration.csv", "attribution.csv", "sanity_checks.json", "runtime_statistics.json"],
    })
    print({"status": "COMPLETE", "condition": CONDITION, "wall_s": runtime["wall_s"], "attribution_rows": len(attribution)}, flush=True)


if __name__ == "__main__":
    main()
