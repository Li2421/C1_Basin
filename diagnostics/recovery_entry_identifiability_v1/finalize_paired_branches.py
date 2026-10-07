"""Validate and aggregate matched N/R branch evidence without training a probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from branch_common import atomic_csv, atomic_json, canonical_hash, file_hash, verify_semantic_hash


HORIZON = 850
INITIAL_FUTURES = 16
CONFIRMATION_FUTURES = 16
MAX_CONTINUATIONS = 12_000
ESCALATION_CUTS = (-0.05, 0.0, 0.05)
Z90 = 1.6448536269514722
Z95 = 1.959963984540054
Z975 = 2.241402727604947


PAIRED_FIELDS = (
    "state_id", "root_source_id", "future_index", "future_stream_id",
    "future_rollout_id", "future_rng_state_hash", "absolute_step",
    "state_sha256", "current_flow_realization_id", "wave",
    "success_n", "success_r", "rescue", "break", "both_success", "both_failure",
    "outcome_n", "outcome_r", "terminal_global_step_n", "terminal_global_step_r",
    "remaining_physical_steps_n", "remaining_physical_steps_r",
    "recovery_exit_step", "recovery_returned_to_safety", "recovery_transitions",
    "eta_latched_hash",
)

STATE_FIELDS = (
    "state_id", "root_source_id", "absolute_step", "state_sha256",
    "matched_futures", "escalated", "success_count_n", "success_count_r",
    "q_n", "q_r", "q_safety", "q_recovery", "delta_q",
    "rescue_count", "break_count", "both_success_count", "both_failure_count",
    "rescue", "break", "delta_identity_error",
)

UNCERTAINTY_FIELDS = (
    "state_id", "root_source_id", "matched_futures", "escalated",
    "q_n_ci90_low", "q_n_ci90_high", "q_r_ci90_low", "q_r_ci90_high",
    "rescue_ci90_low", "rescue_ci90_high", "break_ci90_low", "break_ci90_high",
    "delta_q_ci90_low", "delta_q_ci90_high", "delta_q_ci90_width",
    "q_n_ci95_low", "q_n_ci95_high", "q_r_ci95_low", "q_r_ci95_high",
    "rescue_ci95_low", "rescue_ci95_high", "break_ci95_low", "break_ci95_high",
    "delta_q_ci95_low", "delta_q_ci95_high", "delta_q_ci95_width",
    "ci90_crosses_minus_0_05", "ci90_crosses_zero", "ci90_crosses_plus_0_05",
    "escalation_eligible", "escalation_rule", "confirmation_closed_after_this_wave",
)


def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError((successes, total))
    estimate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (estimate + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def paired_delta_interval(rescue_count: int, break_count: int, total: int,
                          z: float) -> tuple[float, float]:
    """Difference bounds from simultaneous Wilson discordance intervals.

    The caller supplies a Bonferroni component quantile: 95% components for a
    90% DeltaQ interval, or 97.5% components for a 95% DeltaQ interval.
    """

    rescue_low, rescue_high = wilson_interval(rescue_count, total, z)
    break_low, break_high = wilson_interval(break_count, total, z)
    return max(-1.0, rescue_low - break_high), min(1.0, rescue_high - break_low)


def _exact(result: Mapping[str, Any], task: Mapping[str, Any], key: str) -> None:
    if result.get(key) != task.get(key):
        raise RuntimeError((task["task_id"], "result/task mismatch", key,
                            result.get(key), task.get(key)))


def validate_result(result: Mapping[str, Any], task: Mapping[str, Any],
                    manifest: Mapping[str, Any]) -> None:
    task_id = task["task_id"]
    if result.get("schema") != "recovery_entry_paired_branch_result_v1":
        raise RuntimeError((task_id, "result schema mismatch"))
    if result.get("record_complete") is not True or result.get("execution_error") is not None:
        raise RuntimeError((task_id, "incomplete branch result", result.get("execution_error")))
    if result.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise RuntimeError((task_id, "result/manifest lineage mismatch"))
    for key in (
        "task_id", "task_index", "state_id", "root_source_id", "absolute_step", "branch",
        "future_index", "future_stream_id", "future_rollout_id", "future_rng_state_hash",
        "current_flow_realization_id",
    ):
        _exact(result, task, key)
    if result.get("state_sha256") != task["state_sha256"]:
        raise RuntimeError((task_id, "state hash mismatch"))
    if result.get("wave") != manifest["wave"]:
        raise RuntimeError((task_id, "wave mismatch"))
    outcome = result.get("outcome")
    if outcome not in {"success", "deadlock", "timeout", "collision"}:
        raise RuntimeError((task_id, "invalid terminal outcome", outcome))
    flags = {name: bool(result.get(name)) for name in ("success", "deadlock", "timeout", "collision")}
    if sum(flags.values()) != 1 or not flags[outcome]:
        raise RuntimeError((task_id, "terminal flags are not one-hot"))
    transitions = int(result["physical_transition_count"])
    if (transitions != int(result["remaining_physical_steps"])
            or transitions != int(result["flow_sample_count"])
            or transitions <= 0
            or transitions > HORIZON - int(task["absolute_step"])):
        raise RuntimeError((task_id, "physical/Flow/horizon accounting mismatch", transitions))
    if int(result["terminal_global_step"]) != int(task["absolute_step"]) + transitions:
        raise RuntimeError((task_id, "absolute time was not preserved"))
    if outcome == "timeout" and int(result["terminal_global_step"]) != HORIZON:
        raise RuntimeError((task_id, "timeout before official absolute horizon"))
    replay = result.get("current_replay_max_abs", {})
    required_replay = {"current_u_flow", "current_u_safe", "current_feature", "current_flow_key"}
    if set(replay) != required_replay or max(float(replay[key]) for key in replay) > 1e-12:
        raise RuntimeError((task_id, "decision replay mismatch", replay))
    for key in ("invalid_actions", "nan_inf_events", "projection_failures"):
        if int(result.get(key, -1)) != 0:
            raise RuntimeError((task_id, "controller/numeric failure", key))
    if task["branch"] == "N":
        expected_zero = ("recovery_transitions", "entry_query_count", "exit_query_count", "eta_query_count")
        if any(int(result[key]) != 0 for key in expected_zero):
            raise RuntimeError((task_id, "Safety branch invoked recovery machinery"))
        if (result.get("entry_step") is not None or result.get("exit_step") is not None
                or result.get("eta_latched_hash") is not None or result.get("returned_to_safety")):
            raise RuntimeError((task_id, "Safety branch contains recovery state"))
    else:
        if (int(result["entry_query_count"]) != 1 or int(result["eta_query_count"]) != 1
                or int(result["recovery_transitions"]) < 1
                or int(result["entry_step"]) != int(task["absolute_step"])
                or result.get("first_step_exit_queried") is not False
                or not isinstance(result.get("eta_latched_hash"), str)):
            raise RuntimeError((task_id, "persistent-eta entry/latch semantics violated"))
        exited = result.get("exit_step") is not None
        if bool(result.get("returned_to_safety")) != exited:
            raise RuntimeError((task_id, "exit/return-to-Safety mismatch"))
        if exited and int(result["exit_step"]) <= int(result["entry_step"]):
            raise RuntimeError((task_id, "exit occurred without one recovery transition"))


def load_wave(manifest_path: Path, result_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    verify_semantic_hash(manifest, manifest_path)
    if (manifest.get("schema") != "recovery_entry_paired_branch_manifest_v1"
            or manifest.get("status") != "FROZEN_READY_FOR_EXECUTION"):
        raise RuntimeError((manifest_path, "invalid/frozen branch manifest"))
    tasks = list(manifest.get("tasks", []))
    if len(tasks) != int(manifest.get("task_count", -1)):
        raise RuntimeError((manifest_path, "task count mismatch"))
    expected_names = {f"task_{int(task['task_index']):05d}.json" for task in tasks}
    actual_names = {path.name for path in result_directory.glob("task_*.json")}
    if actual_names != expected_names:
        raise RuntimeError((result_directory, "result coverage mismatch",
                            sorted(expected_names - actual_names)[:5],
                            sorted(actual_names - expected_names)[:5]))
    by_index = {int(task["task_index"]): task for task in tasks}
    if set(by_index) != set(range(len(tasks))):
        raise RuntimeError((manifest_path, "task indices are not unique/contiguous"))
    results = []
    for index in range(len(tasks)):
        result = json.loads((result_directory / f"task_{index:05d}.json").read_text())
        validate_result(result, by_index[index], manifest)
        results.append(result)
    return manifest, results


def _validate_waves(manifests: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> None:
    if not manifests or manifests[0].get("wave") != "initial":
        raise RuntimeError("the first input must be the initial branch wave")
    if len(manifests) > 2:
        raise RuntimeError("only one initial and one confirmation wave are permitted")
    if len(manifests) == 2:
        confirmation = manifests[1]
        if confirmation.get("wave") != "confirmation":
            raise RuntimeError("second wave is not the one-shot confirmation block")
        parent = confirmation.get("parent_initial_manifest", {})
        if parent.get("content_sha256") != manifests[0]["content_sha256"]:
            raise RuntimeError("confirmation wave has wrong initial parent")
    if len(results) > MAX_CONTINUATIONS:
        raise RuntimeError(("global branch continuation budget exceeded", len(results)))
    keys = [(row["state_id"], int(row["future_index"]), row["branch"]) for row in results]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate state/future/branch across waves")
    rollout_owners: dict[int, tuple[str, int]] = {}
    for row in results:
        rollout_id = int(row["future_rollout_id"])
        owner = (str(row["state_id"]), int(row["future_index"]))
        previous = rollout_owners.setdefault(rollout_id, owner)
        if previous != owner:
            raise RuntimeError(("future rollout id reused by distinct matched pairs",
                                rollout_id, previous, owner))


def pair_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault((str(result["state_id"]), int(result["future_index"])), {})[
            str(result["branch"])
        ] = result
    paired: list[dict[str, Any]] = []
    for key in sorted(grouped):
        pair = grouped[key]
        if set(pair) != {"N", "R"}:
            raise RuntimeError((key, "missing N/R matched branch"))
        n, r = pair["N"], pair["R"]
        for shared in (
            "state_id", "root_source_id", "state_sha256", "absolute_step",
            "current_flow_realization_id", "future_index", "future_stream_id",
            "future_rollout_id", "future_rng_state_hash", "wave",
        ):
            if n[shared] != r[shared]:
                raise RuntimeError((key, "N/R pairing mismatch", shared))
        success_n, success_r = int(bool(n["success"])), int(bool(r["success"]))
        paired.append({
            "state_id": n["state_id"], "root_source_id": n["root_source_id"],
            "future_index": int(n["future_index"]), "future_stream_id": n["future_stream_id"],
            "future_rollout_id": int(n["future_rollout_id"]),
            "future_rng_state_hash": n["future_rng_state_hash"],
            "absolute_step": int(n["absolute_step"]), "state_sha256": n["state_sha256"],
            "current_flow_realization_id": n["current_flow_realization_id"], "wave": n["wave"],
            "success_n": success_n, "success_r": success_r,
            "rescue": int(not success_n and success_r),
            "break": int(success_n and not success_r),
            "both_success": int(success_n and success_r),
            "both_failure": int(not success_n and not success_r),
            "outcome_n": n["outcome"], "outcome_r": r["outcome"],
            "terminal_global_step_n": int(n["terminal_global_step"]),
            "terminal_global_step_r": int(r["terminal_global_step"]),
            "remaining_physical_steps_n": int(n["remaining_physical_steps"]),
            "remaining_physical_steps_r": int(r["remaining_physical_steps"]),
            "recovery_exit_step": r["exit_step"],
            "recovery_returned_to_safety": bool(r["returned_to_safety"]),
            "recovery_transitions": int(r["recovery_transitions"]),
            "eta_latched_hash": r["eta_latched_hash"],
        })
    return paired


def summarize_states(paired: Sequence[Mapping[str, Any]], *, confirmation_closed: bool
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in paired:
        grouped.setdefault(str(row["state_id"]), []).append(row)
    state_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for state_id in sorted(grouped):
        values = sorted(grouped[state_id], key=lambda row: int(row["future_index"]))
        n = len(values)
        futures = [int(row["future_index"]) for row in values]
        if futures not in (list(range(INITIAL_FUTURES)),
                           list(range(INITIAL_FUTURES + CONFIRMATION_FUTURES))):
            raise RuntimeError((state_id, "future-index block is incomplete/noncontiguous", futures))
        counts = {
            name: sum(int(row[name]) for row in values)
            for name in ("success_n", "success_r", "rescue", "break", "both_success", "both_failure")
        }
        q_n, q_r = counts["success_n"] / n, counts["success_r"] / n
        rescue, break_rate = counts["rescue"] / n, counts["break"] / n
        delta = q_r - q_n
        identity_error = abs(delta - (rescue - break_rate))
        if identity_error > 1e-15:
            raise RuntimeError((state_id, "DeltaQ != rescue-break", identity_error))
        common = {
            "state_id": state_id, "root_source_id": values[0]["root_source_id"],
            "absolute_step": int(values[0]["absolute_step"]),
            "state_sha256": values[0]["state_sha256"], "matched_futures": n,
            "escalated": n > INITIAL_FUTURES,
        }
        state_rows.append({
            **common, "success_count_n": counts["success_n"],
            "success_count_r": counts["success_r"], "q_n": q_n, "q_r": q_r,
            "q_safety": q_n, "q_recovery": q_r, "delta_q": delta,
            "rescue_count": counts["rescue"], "break_count": counts["break"],
            "both_success_count": counts["both_success"],
            "both_failure_count": counts["both_failure"],
            "rescue": rescue, "break": break_rate, "delta_identity_error": identity_error,
        })
        ci_values: dict[str, Any] = {}
        for label, z, delta_component_z in (("90", Z90, Z95), ("95", Z95, Z975)):
            for name, count_key in (("q_n", "success_n"), ("q_r", "success_r"),
                                    ("rescue", "rescue"), ("break", "break")):
                low, high = wilson_interval(counts[count_key], n, z)
                ci_values[f"{name}_ci{label}_low"] = low
                ci_values[f"{name}_ci{label}_high"] = high
            low, high = paired_delta_interval(
                counts["rescue"], counts["break"], n, delta_component_z
            )
            ci_values[f"delta_q_ci{label}_low"] = low
            ci_values[f"delta_q_ci{label}_high"] = high
            ci_values[f"delta_q_ci{label}_width"] = high - low
        low90, high90 = ci_values["delta_q_ci90_low"], ci_values["delta_q_ci90_high"]
        crossings = [low90 <= cut <= high90 for cut in ESCALATION_CUTS]
        uncertainty_rows.append({
            "state_id": state_id, "root_source_id": common["root_source_id"],
            "matched_futures": n, "escalated": common["escalated"], **ci_values,
            "ci90_crosses_minus_0_05": crossings[0], "ci90_crosses_zero": crossings[1],
            "ci90_crosses_plus_0_05": crossings[2],
            "escalation_eligible": any(crossings) and not confirmation_closed,
            "escalation_rule": "paired 90% CI crosses any of {-0.05,0,+0.05}",
            "confirmation_closed_after_this_wave": confirmation_closed,
        })
    return state_rows, uncertainty_rows


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "std_population": float(np.std(array)), "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)), "max": float(np.max(array)),
    }


def advantage_distribution(state_rows: Sequence[Mapping[str, Any]],
                           uncertainty_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta = np.asarray([row["delta_q"] for row in state_rows], dtype=np.float64)
    q_n = np.asarray([row["q_n"] for row in state_rows], dtype=np.float64)
    q_r = np.asarray([row["q_r"] for row in state_rows], dtype=np.float64)
    rescue = np.asarray([row["rescue"] for row in state_rows], dtype=np.float64)
    break_rate = np.asarray([row["break"] for row in state_rows], dtype=np.float64)
    return {
        "schema": "recovery_entry_advantage_distribution_v1",
        "state_count": len(state_rows),
        "root_source_count": len({row["root_source_id"] for row in state_rows}),
        "q_safety": _distribution(q_n), "q_recovery": _distribution(q_r),
        "delta_q": _distribution(delta), "rescue": _distribution(rescue),
        "break": _distribution(break_rate),
        "descriptive_sensitivity_fractions": {
            "delta_q_gt_plus_0_10": float(np.mean(delta > 0.10)),
            "delta_q_gt_plus_0_05": float(np.mean(delta > 0.05)),
            "abs_delta_q_le_0_05": float(np.mean(np.abs(delta) <= 0.05)),
            "delta_q_lt_minus_0_05": float(np.mean(delta < -0.05)),
            "delta_q_lt_minus_0_10": float(np.mean(delta < -0.10)),
        },
        "trivial_and_hindsight_values": {
            "always_safety": float(np.mean(q_n)),
            "always_recovery": float(np.mean(q_r)),
            "hindsight_statewise_oracle_upper_bound": float(np.mean(np.maximum(q_n, q_r))),
            "oracle_gain_over_best_trivial": float(
                np.mean(np.maximum(q_n, q_r)) - max(np.mean(q_n), np.mean(q_r))
            ),
        },
        "uncertainty": {
            "paired_interval_method": (
                "Bonferroni simultaneous Wilson intervals for paired rescue/break rates; "
                "DeltaQ bounds are [rescue_low-break_high,rescue_high-break_low]"
            ),
            "states_with_ci90_crossing_any_predeclared_cut": sum(
                bool(row["ci90_crosses_minus_0_05"])
                or bool(row["ci90_crosses_zero"])
                or bool(row["ci90_crosses_plus_0_05"])
                for row in uncertainty_rows
            ),
        },
    }


def finalize(manifest_paths: Sequence[Path], result_directories: Sequence[Path],
             output_directory: Path) -> dict[str, Any]:
    if len(manifest_paths) != len(result_directories) or not manifest_paths:
        raise ValueError("provide one result directory per manifest")
    outputs = [
        output_directory / "paired_branch_outcomes.csv",
        output_directory / "statewise_recovery_advantage.csv",
        output_directory / "label_uncertainty.csv",
        output_directory / "advantage_distribution.json",
        output_directory / "branch_finalization_manifest.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(("refusing to overwrite finalized branch artifacts", existing))
    manifests: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for manifest_path, result_directory in zip(manifest_paths, result_directories, strict=True):
        manifest, wave_results = load_wave(manifest_path, result_directory)
        manifests.append(manifest)
        results.extend(wave_results)
    _validate_waves(manifests, results)
    paired = pair_results(results)
    state_rows, uncertainty_rows = summarize_states(
        paired, confirmation_closed=len(manifests) == 2
    )
    distribution = advantage_distribution(state_rows, uncertainty_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs[0], paired, PAIRED_FIELDS)
    atomic_csv(outputs[1], state_rows, STATE_FIELDS)
    atomic_csv(outputs[2], uncertainty_rows, UNCERTAINTY_FIELDS)
    distribution["content_sha256"] = canonical_hash(distribution)
    atomic_json(outputs[3], distribution)
    manifest_out = {
        "schema": "recovery_entry_branch_finalization_manifest_v1",
        "status": "COMPLETE",
        "waves": [manifest["wave"] for manifest in manifests],
        "source_manifests": [
            {"path": str(path.resolve()), "sha256": file_hash(path),
             "content_sha256": manifest["content_sha256"]}
            for path, manifest in zip(manifest_paths, manifests, strict=True)
        ],
        "result_directories": [str(path.resolve()) for path in result_directories],
        "branch_continuation_count": len(results), "matched_pair_count": len(paired),
        "queried_state_count": len(state_rows),
        "root_source_count": len({row["root_source_id"] for row in state_rows}),
        "states_at_16_futures": sum(int(row["matched_futures"]) == 16 for row in state_rows),
        "states_at_32_futures": sum(int(row["matched_futures"]) == 32 for row in state_rows),
        "global_continuation_budget": MAX_CONTINUATIONS,
        "budget_respected": len(results) <= MAX_CONTINUATIONS,
        "all_results_complete": True, "all_pairs_matched": True,
        "delta_identity_max_abs_error": max(float(row["delta_identity_error"]) for row in state_rows),
        "controller_or_probe_modified": False, "probe_training_launched": False,
        "outputs": {},
    }
    for name, path in (
        ("paired_branch_outcomes", outputs[0]), ("statewise_recovery_advantage", outputs[1]),
        ("label_uncertainty", outputs[2]), ("advantage_distribution", outputs[3]),
    ):
        manifest_out["outputs"][name] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    manifest_out["content_sha256"] = canonical_hash(manifest_out)
    atomic_json(outputs[4], manifest_out)
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--result-directory", action="append", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.manifest, args.result_directory, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
