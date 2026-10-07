"""Offline integrity audit for the causal startup-aware 214-D wrapper.

This creates no oracle labels or dataset samples.  It advances three short,
deterministic diagnostic traces only to exercise steps 0..40.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
for path in (ROOT, SYSROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from startup_feature_builder import (  # noqa: E402
    FEATURE_DIMENSION,
    HISTORY_LENGTH,
    SOURCE_AUDIT,
    StartupAwareFeatureBuilder,
    _HistoryOverride,
    assert_authoritative_sources,
    left_pad_goal_error_history,
)
from diagnostics.gphi_training_dataset_v2.finalize_dataset import FeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal  # noqa: E402


HISTORY_SEGMENTS = {"recent_progress_2s", "history_start_step", "goal_error_history_tail_41"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def slices(schema: list[dict], names: set[str]) -> np.ndarray:
    selected: list[int] = []
    for segment in schema:
        if segment["name"] in names:
            selected.extend(range(int(segment["offset"]), int(segment["offset"] + segment["length"])))
    return np.asarray(selected, dtype=np.int64)


def nominal_action(trace_index: int, step: int, env: GiveWayEnv, config: Config) -> np.ndarray:
    to_goal = env.goals - env.positions
    directions = to_goal / np.maximum(np.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-30)
    speeds = (0.14, 0.18, 0.22)
    nominal = speeds[trace_index] * directions
    # Small deterministic lateral variation makes the audit exercise geometry
    # without selecting a failure/bottleneck/recovery condition.
    lateral = 0.008 * math.sin(0.23 * step + trace_index)
    nominal[:, 1] += np.asarray([lateral, -lateral])
    return bounded_nominal(nominal, config.max_speed)


def main() -> None:
    source_audit = assert_authoritative_sources()
    protocol = json.loads((ROOT / "diagnostics/gphi_training_dataset_v3/protocol.json").read_text())
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    initial_conditions = (
        [[-0.85, 0.0], [0.85, 0.0]],
        [[-0.90, -0.01], [0.90, 0.01]],
        [[-0.80, 0.015], [0.95, -0.015]],
    )

    trace_rows = []
    max_wrapper_reference_error = 0.0
    max_post_warmup_error = 0.0
    max_nonhistory_sensitivity = 0.0
    max_history_reconstruction_error = 0.0
    total_features = 0
    all_finite = True
    transition_steps: list[int] = []

    for trace_index, positions in enumerate(initial_conditions):
        env = GiveWayEnv(config)
        env.reset(positions=np.asarray(positions, dtype=np.float64))
        wrapper = StartupAwareFeatureBuilder()
        reference_builder = FeatureBuilder()
        alternate_builder = FeatureBuilder()
        original_builder = FeatureBuilder()
        transition_step = None

        for step in range(41):
            if env.step_count != step or len(env.distance_history) != step + 1:
                raise AssertionError((trace_index, step, env.step_count, len(env.distance_history)))
            flow = nominal_action(trace_index, step, env, config)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, status, retried, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            first = {"u_flow": flow, "u_safe": safe}

            vector, structured = wrapper.build(env, first, config, cbf)
            padded = left_pad_goal_error_history(env.distance_history)
            reference, reference_structured = reference_builder.build(
                _HistoryOverride(env, padded), first, config, cbf
            )
            wrapper_reference_error = float(np.max(np.abs(vector - reference)))
            max_wrapper_reference_error = max(max_wrapper_reference_error, wrapper_reference_error)
            max_history_reconstruction_error = max(
                max_history_reconstruction_error,
                float(np.max(np.abs(structured["goal_error_history"] - padded))),
            )

            history_indices = slices(wrapper.schema, HISTORY_SEGMENTS)
            nonhistory_mask = np.ones(FEATURE_DIMENSION, dtype=bool)
            nonhistory_mask[history_indices] = False
            alternate_history = padded.copy()
            alternate_history[:, 0] += np.linspace(0.0, 0.013, HISTORY_LENGTH)
            alternate_history[:, 1] -= np.linspace(0.0, 0.009, HISTORY_LENGTH)
            alternate, _ = alternate_builder.build(
                _HistoryOverride(env, alternate_history), first, config, cbf
            )
            nonhistory_sensitivity = float(np.max(np.abs(reference[nonhistory_mask] - alternate[nonhistory_mask])))
            max_nonhistory_sensitivity = max(max_nonhistory_sensitivity, nonhistory_sensitivity)

            post_warmup_error = None
            if len(env.distance_history) >= HISTORY_LENGTH:
                original, original_structured = original_builder.build(env, first, config, cbf)
                post_warmup_error = float(np.max(np.abs(vector - original)))
                max_post_warmup_error = max(max_post_warmup_error, post_warmup_error)
                if transition_step is None:
                    transition_step = step
                if not np.array_equal(
                    structured["goal_error_history"], original_structured["goal_error_history"]
                ):
                    raise AssertionError((trace_index, step, "post-warmup structured history mismatch"))

            finite = bool(np.isfinite(vector).all())
            all_finite = all_finite and finite
            total_features += 1
            trace_rows.append(
                {
                    "trace": trace_index,
                    "step": step,
                    "dimension": int(vector.size),
                    "finite": finite,
                    "available_history_points": len(env.distance_history),
                    "left_padding_points": max(0, HISTORY_LENGTH - len(env.distance_history)),
                    "wrapper_reference_max_abs_error": wrapper_reference_error,
                    "nonhistory_max_abs_sensitivity_to_history": nonhistory_sensitivity,
                    "post_warmup_original_max_abs_error": post_warmup_error,
                    "first_projection_status": status,
                    "first_projection_retried": bool(retried),
                }
            )
            if vector.shape != (FEATURE_DIMENSION,):
                raise AssertionError((trace_index, step, vector.shape))
            if step < 40:
                _, _, done, info = env.step(safe)
                if done:
                    raise RuntimeError(("diagnostic trace ended before step 40", trace_index, step, info["termination"]))

        if transition_step is None:
            raise AssertionError((trace_index, "no full-history transition"))
        transition_steps.append(transition_step)

    checks = {
        "status": "PASS",
        "scope": "feature-wrapper integrity only; no oracle rollout, label generation, merge, or training",
        "source_audit": source_audit,
        "startup_rule": "left-pad the available goal-error history with its earliest value to length 41",
        "feature_dimension": FEATURE_DIMENSION,
        "traces": len(initial_conditions),
        "steps_per_trace": 41,
        "feature_vectors_checked": total_features,
        "all_vectors_finite": all_finite,
        "transition_steps": transition_steps,
        "first_complete_history_timestep": min(transition_steps),
        "max_wrapper_vs_authoritative_padded_error": max_wrapper_reference_error,
        "max_post_warmup_wrapper_vs_original_error": max_post_warmup_error,
        "max_nonhistory_change_when_only_history_is_perturbed": max_nonhistory_sensitivity,
        "max_padded_history_reconstruction_error": max_history_reconstruction_error,
        "post_warmup_exact": max_post_warmup_error == 0.0,
        "nonhistory_fields_unchanged": max_nonhistory_sensitivity == 0.0,
        "trace_rows": trace_rows,
    }
    required = (
        checks["all_vectors_finite"],
        checks["first_complete_history_timestep"] == 40,
        checks["post_warmup_exact"],
        checks["nonhistory_fields_unchanged"],
        checks["max_wrapper_vs_authoritative_padded_error"] == 0.0,
        checks["max_padded_history_reconstruction_error"] == 0.0,
        SOURCE_AUDIT["status"] == "PASS",
    )
    if not all(required):
        checks["status"] = "FAIL"
    write_json(HERE / "startup_feature_continuity_checks.json", checks)
    write_json(
        HERE / "restoration_checks.json",
        {
            "status": checks["status"],
            "scope": checks["scope"],
            "startup_feature_continuity_checks": "startup_feature_continuity_checks.json",
            "authoritative_source_lock_passed": source_audit["status"] == "PASS",
            "steps_0_through_40_valid": all_finite and total_features == 3 * 41,
            "post_warmup_exact": checks["post_warmup_exact"],
            "nonhistory_fields_unchanged": checks["nonhistory_fields_unchanged"],
        },
    )
    if checks["status"] != "PASS":
        raise RuntimeError("startup feature continuity audit failed")


if __name__ == "__main__":
    main()
