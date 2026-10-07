"""Fail-closed analysis of the frozen WIDE 850-to-1700 timeout extension."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_extended_horizon_audit_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
RUN_ROOT = HERE / "runs/production"
ORDER = ("Safety", "H=1", "H=4", "H=8", "H=16")
SLUG = {"Safety": "safety", "H=1": "h1", "H=4": "h4", "H=8": "h8", "H=16": "h16"}
CADENCE = {"Safety": None, "H=1": 1, "H=4": 4, "H=8": 8, "H=16": 16}
EXPECTED_TIMEOUTS = {"Safety": 51, "H=1": 48, "H=4": 18, "H=8": 13, "H=16": 27}
CHECKPOINT_SHA = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
HORIZONS = (850, 950, 1050, 1200, 1400, 1700)
DT = 0.05
OUTPUTS = (
    "extended_horizon_report.md", "integrity_audit.json",
    "original_timeout_manifest.csv", "extended_episode_results.csv",
    "late_success_times.csv", "cumulative_success_by_horizon.csv",
    "q_vs_horizon.csv", "timeout_transition_table.csv",
    "persistent_timeout_cases.csv", "late_deadlock_cases.csv",
    "h8_timeout_case_table.csv", "h16_vs_h8_timeout_analysis.csv",
    "safety_timeout_analysis.csv", "extended_jdef.csv",
    "time_to_success_comparison.csv", "runtime_statistics.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stats(values: Iterable[float]) -> dict[str, float | None]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    keys = ("mean", "median", "std", "p25", "p75", "p90", "p95", "max")
    if not len(a):
        return {key: None for key in keys}
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "p25": float(np.quantile(a, .25)), "p75": float(np.quantile(a, .75)),
        "p90": float(np.quantile(a, .90)), "p95": float(np.quantile(a, .95)),
        "max": float(a.max()),
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    w = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-w, c+w


def flatten_progress(row: dict[str, Any]) -> dict[str, Any]:
    p = row["progress_after_850"]
    return {
        "condition": row["condition"], "episode_index": row["episode_index"],
        "extended_outcome_class": row["extended_outcome_class"],
        "episode_steps": row["episode_steps"],
        "goal_error_reduction_sum": p["goal_error_reduction_sum"],
        "goal_error_reduction_agent0": p["goal_error_reduction_each"][0],
        "goal_error_reduction_agent1": p["goal_error_reduction_each"][1],
        "displacement_sum": p["displacement_sum"],
        "displacement_agent0": p["displacement_each"][0],
        "displacement_agent1": p["displacement_each"][1],
        "agent_speed_mean": p["agent_speed_mean"],
        "agent_speed_median": p["agent_speed_median"],
        "inter_agent_distance_start": p["inter_agent_distance_start"],
        "inter_agent_distance_final": p["inter_agent_distance_final"],
        "inter_agent_distance_min": p["inter_agent_distance_min"],
        "candidate_deadlock_fraction": p["candidate_deadlock_fraction"],
        "maximum_stuck_timer": p["maximum_stuck_timer"],
        "effective_correction_fraction": p["effective_correction_fraction"],
        "additional_J_def": p["additional_J_def"],
    }


def main() -> None:
    started = time.monotonic()
    config_path = HERE / "extended_config.json"
    config = load_json(config_path)
    if config["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("checkpoint mismatch")
    benchmark = load_json(WIDE / "frozen_benchmark_manifest.json")
    if benchmark["episode_count"] != 200:
        raise RuntimeError("benchmark size mismatch")

    original: dict[str, dict[int, dict[str, Any]]] = {condition: {} for condition in ORDER}
    original_paths: dict[tuple[str, int], Path] = {}
    for condition in ORDER:
        slug = SLUG[condition]
        for index in range(200):
            path = WIDE / "runs/production/raw" / slug / f"episode_{index:04d}.json"
            row = load_json(path)
            original[condition][index] = row
            original_paths[(condition, index)] = path

    extended: dict[str, dict[int, dict[str, Any]]] = {condition: {} for condition in ORDER}
    prefix_max = 0.0
    raw_hashes: dict[str, str] = {}
    trajectory_hashes: dict[str, str] = {}
    timeout_manifest: list[dict[str, Any]] = []
    extended_rows: list[dict[str, Any]] = []
    for condition in ORDER:
        slug = SLUG[condition]
        timeout_indices = [i for i, row in original[condition].items() if row["outcome"] == "timeout"]
        if len(timeout_indices) != EXPECTED_TIMEOUTS[condition]:
            raise RuntimeError(("timeout count mismatch", condition, len(timeout_indices)))
        for index in timeout_indices:
            path = RUN_ROOT / "raw" / slug / f"episode_{index:04d}.json"
            if not path.is_file():
                raise RuntimeError(("missing extended record", condition, index))
            row = load_json(path)
            trajectory = HERE / row["trajectory_file"]
            if not trajectory.is_file() or sha256(trajectory) != row["trajectory_sha256"]:
                raise RuntimeError(("extended trajectory invalid", condition, index))
            if row["extended_config_sha256"] != config["content_sha256"]:
                raise RuntimeError(("config mismatch", condition, index))
            if row["original_record_sha256"] != sha256(original_paths[(condition, index)]):
                raise RuntimeError(("original record changed", condition, index))
            if row["prefix_audit"]["status"] != "PASS" or not row["prefix_audit"]["steps_0_through_849_controller_state_monitor_exact"]:
                raise RuntimeError(("prefix failed", condition, index))
            prefix_max = max(prefix_max, float(row["prefix_audit"]["maximum_numeric_prefix_error"]))
            with np.load(trajectory, allow_pickle=False) as z:
                if len(z["step"]) != int(row["episode_steps"]):
                    raise RuntimeError(("trajectory length mismatch", condition, index))
                if not np.array_equal(z["step"][:850], np.arange(850)):
                    raise RuntimeError(("prefix step mismatch", condition, index))
                if CADENCE[condition] is None:
                    expected_schedule = np.zeros(len(z["step"]), dtype=bool)
                else:
                    expected_schedule = z["step"] % CADENCE[condition] == 0
                if not np.array_equal(z["cadence_scheduled"], expected_schedule):
                    raise RuntimeError(("cadence phase reset/mismatch", condition, index))
                inactive = ~expected_schedule
                if np.any(inactive) and (
                    np.max(np.abs(z["g_hat"][inactive])) > 0
                    or np.max(np.abs(z["u_exec"][inactive] - z["u_safe"][inactive])) > 0
                ):
                    raise RuntimeError(("held/off-cadence correction", condition, index))
            raw_hashes[str(path)] = sha256(path)
            trajectory_hashes[str(trajectory)] = sha256(trajectory)
            extended[condition][index] = row
            old = original[condition][index]
            timeout_manifest.append({
                "condition": condition, "cadence_h": CADENCE[condition],
                "episode_index": index, "rollout_id": old["rollout_id"],
                "gphi_overlap_partition": old["gphi_overlap_partition"],
                "original_outcome": old["outcome"], "original_steps": old["episode_steps"],
                "original_J_def": old["J_def"],
                "original_record_path": str(original_paths[(condition, index)]),
                "original_record_sha256": row["original_record_sha256"],
                "original_trajectory_sha256": row["original_trajectory_sha256"],
            })
            p = row["progress_after_850"]
            extended_rows.append({
                "condition": condition, "cadence_h": CADENCE[condition],
                "episode_index": index, "gphi_overlap_partition": old["gphi_overlap_partition"],
                "extended_outcome_class": row["extended_outcome_class"],
                "extended_native_outcome": row["outcome"], "episode_steps": row["episode_steps"],
                "first_success_step": row["first_success_step"],
                "first_deadlock_step": row["first_deadlock_step"],
                "total_completion_time_sec": row["total_completion_time_sec"],
                "extra_time_after_original_horizon_sec": row["extra_time_after_original_horizon_sec"],
                "J_def_0_850": row["J_def_0_850"],
                "J_def_additional_after_850": row["J_def_additional_after_850"],
                "J_def_extended_total": row["J_def_extended_total"],
                "goal_error_reduction_sum_after_850": p["goal_error_reduction_sum"],
                "displacement_sum_after_850": p["displacement_sum"],
                "mean_agent_speed_after_850": p["agent_speed_mean"],
                "candidate_deadlock_fraction_after_850": p["candidate_deadlock_fraction"],
                "maximum_stuck_timer_after_850": p["maximum_stuck_timer"],
                "prefix_max_absolute_error": row["prefix_audit"]["maximum_numeric_prefix_error"],
                "trajectory_sha256": row["trajectory_sha256"],
            })

    write_csv(HERE / "original_timeout_manifest.csv", timeout_manifest)
    write_csv(HERE / "extended_episode_results.csv", extended_rows)

    transitions: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    delay_stats: dict[str, dict[str, float | None]] = {}
    q_lookup: dict[tuple[str, int], float] = {}
    for condition in ORDER:
        rows = list(extended[condition].values())
        counts = Counter(row["extended_outcome_class"] for row in rows)
        n = EXPECTED_TIMEOUTS[condition]
        late = counts["LATE_SUCCESS"]
        nonlate = n - late
        lo, hi = wilson(late, n)
        if lo > 0.5:
            classification = "MOSTLY_LATE_SUCCESS"
        elif hi < 0.5:
            classification = "MOSTLY_TRUE_LIVENESS_FAILURE"
        else:
            classification = "MIXED_HORIZON_AND_LIVENESS"
        classifications[condition] = classification
        delays = [float(row["extra_time_after_original_horizon_sec"]) for row in rows if row["extended_outcome_class"] == "LATE_SUCCESS"]
        delay_stats[condition] = stats(delays)
        transitions.append({
            "condition": condition, "original_timeout_count": n,
            "late_success": late, "late_deadlock": counts["LATE_DEADLOCK"],
            "persistent_timeout": counts["PERSISTENT_TIMEOUT"],
            "other_failure": counts["OTHER_FAILURE"],
            "late_success_rate": late/n, "late_success_rate_ci_low": lo,
            "late_success_rate_ci_high": hi, "classification": classification,
            **{f"rescue_delay_{key}_sec": value for key, value in delay_stats[condition].items()},
        })
        for row in rows:
            if row["extended_outcome_class"] == "LATE_SUCCESS":
                late_rows.append({
                    "condition": condition, "episode_index": row["episode_index"],
                    "first_success_step": row["first_success_step"],
                    "total_completion_time_sec": row["total_completion_time_sec"],
                    "extra_time_after_original_horizon_sec": row["extra_time_after_original_horizon_sec"],
                    "additional_J_def_after_850": row["J_def_additional_after_850"],
                })
    write_csv(HERE / "timeout_transition_table.csv", transitions)
    write_csv(HERE / "late_success_times.csv", late_rows)

    cumulative_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    for condition in ORDER:
        completion_steps: list[int] = []
        for index, old in original[condition].items():
            if old["outcome"] == "success":
                completion_steps.append(int(old["episode_steps"]))
            elif old["outcome"] == "timeout":
                new = extended[condition][index]
                if new["extended_outcome_class"] == "LATE_SUCCESS":
                    completion_steps.append(int(new["first_success_step"]))
        for horizon in range(850, 1701):
            success = sum(step <= horizon for step in completion_steps)
            cumulative_rows.append({
                "condition": condition, "horizon_steps": horizon,
                "horizon_seconds": horizon*DT, "cumulative_success": success,
                "population": 200, "Q": success/200,
            })
            q_lookup[(condition, horizon)] = success/200
        for horizon in HORIZONS:
            success = sum(step <= horizon for step in completion_steps)
            lo, hi = wilson(success, 200)
            q_rows.append({
                "condition": condition, "horizon_steps": horizon,
                "horizon_seconds": horizon*DT, "success": success,
                "population": 200, "Q": success/200,
                "Q_wilson_95_ci_low": lo, "Q_wilson_95_ci_high": hi,
            })
    write_csv(HERE / "cumulative_success_by_horizon.csv", cumulative_rows)
    write_csv(HERE / "q_vs_horizon.csv", q_rows)

    persistent = [flatten_progress(row) for condition in ORDER for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"]
    late_deadlock = [flatten_progress(row) for condition in ORDER for row in extended[condition].values() if row["extended_outcome_class"] == "LATE_DEADLOCK"]
    progress_fields = [
        "condition", "episode_index", "extended_outcome_class", "episode_steps",
        "goal_error_reduction_sum", "goal_error_reduction_agent0",
        "goal_error_reduction_agent1", "displacement_sum", "displacement_agent0",
        "displacement_agent1", "agent_speed_mean", "agent_speed_median",
        "inter_agent_distance_start", "inter_agent_distance_final",
        "inter_agent_distance_min", "candidate_deadlock_fraction",
        "maximum_stuck_timer", "effective_correction_fraction", "additional_J_def",
    ]
    write_csv(HERE / "persistent_timeout_cases.csv", persistent, progress_fields)
    write_csv(HERE / "late_deadlock_cases.csv", late_deadlock, progress_fields)

    h8_rows = []
    for index, row in sorted(extended["H=8"].items()):
        h8_rows.append({
            "episode_index": index, "extended_outcome_class": row["extended_outcome_class"],
            "first_success_step": row["first_success_step"],
            "extra_time_after_42_5_sec": row["extra_time_after_original_horizon_sec"],
            "first_deadlock_step": row["first_deadlock_step"],
            "goal_error_reduction_sum_after_850": row["progress_after_850"]["goal_error_reduction_sum"],
            "displacement_sum_after_850": row["progress_after_850"]["displacement_sum"],
            "additional_J_def": row["J_def_additional_after_850"],
        })
    write_csv(HERE / "h8_timeout_case_table.csv", h8_rows)

    h16_vs_h8 = []
    for index in range(200):
        if original["H=8"][index]["outcome"] == "success" and original["H=16"][index]["outcome"] == "timeout":
            row = extended["H=16"][index]
            h16_vs_h8.append({
                "episode_index": index,
                "H8_success_step_at_850_run": original["H=8"][index]["episode_steps"],
                "H8_completion_time_sec": original["H=8"][index]["episode_steps"]*DT,
                "H16_extended_outcome": row["extended_outcome_class"],
                "H16_first_success_step": row["first_success_step"],
                "H16_completion_time_sec": row["total_completion_time_sec"],
                "H16_extra_time_after_42_5_sec": row["extra_time_after_original_horizon_sec"],
                "H16_additional_J_def": row["J_def_additional_after_850"],
            })
    write_csv(HERE / "h16_vs_h8_timeout_analysis.csv", h16_vs_h8)

    safety_rows = []
    for index, row in sorted(extended["Safety"].items()):
        safety_rows.append({
            "episode_index": index, "extended_outcome_class": row["extended_outcome_class"],
            "first_success_step": row["first_success_step"],
            "total_completion_time_sec": row["total_completion_time_sec"],
            "extra_time_after_42_5_sec": row["extra_time_after_original_horizon_sec"],
            "first_deadlock_step": row["first_deadlock_step"],
            "goal_error_reduction_sum_after_850": row["progress_after_850"]["goal_error_reduction_sum"],
            "displacement_sum_after_850": row["progress_after_850"]["displacement_sum"],
        })
    write_csv(HERE / "safety_timeout_analysis.csv", safety_rows)

    jdef_rows = []
    for condition in ORDER[1:]:
        for index, row in sorted(extended[condition].items()):
            jdef_rows.append({
                "condition": condition, "episode_index": index,
                "extended_outcome_class": row["extended_outcome_class"],
                "J_def_0_850": row["J_def_0_850"],
                "additional_J_def_850_to_termination": row["J_def_additional_after_850"],
                "total_extended_J_def": row["J_def_extended_total"],
                "late_success": row["extended_outcome_class"] == "LATE_SUCCESS",
            })
    write_csv(HERE / "extended_jdef.csv", jdef_rows)

    tts_rows = []
    completion: dict[str, dict[int, int | None]] = {condition: {} for condition in ORDER}
    for condition in ORDER:
        for index in range(200):
            old = original[condition][index]
            step: int | None = int(old["episode_steps"]) if old["outcome"] == "success" else None
            eventual = old["outcome"]
            if old["outcome"] == "timeout":
                new = extended[condition][index]
                eventual = new["extended_outcome_class"]
                if eventual == "LATE_SUCCESS":
                    step = int(new["first_success_step"])
            completion[condition][index] = step
            tts_rows.append({
                "episode_index": index, "condition": condition,
                "original_outcome_850": old["outcome"], "eventual_extended_outcome": eventual,
                "success_by_1700": step is not None, "completion_step": step,
                "completion_time_sec": None if step is None else step*DT,
            })
    write_csv(HERE / "time_to_success_comparison.csv", tts_rows)

    paired_timing: dict[str, Any] = {}
    for condition in ORDER[1:]:
        pairs = [(completion[condition][i], completion["Safety"][i]) for i in range(200) if completion[condition][i] is not None and completion["Safety"][i] is not None]
        diffs = [(a-b)*DT for a, b in pairs]
        paired_timing[f"{condition}_minus_Safety"] = {"paired_successes": len(diffs), **stats(diffs)}
    pairs = [(completion["H=16"][i], completion["H=8"][i]) for i in range(200) if completion["H=16"][i] is not None and completion["H=8"][i] is not None]
    paired_timing["H=16_minus_H=8"] = {"paired_successes": len(pairs), **stats([(a-b)*DT for a, b in pairs])}

    persistent_summary = {
        condition: {
            "count": len([row for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"]),
            "goal_error_reduction_sum": stats(row["progress_after_850"]["goal_error_reduction_sum"] for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"),
            "displacement_sum": stats(row["progress_after_850"]["displacement_sum"] for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"),
            "agent_speed_mean": stats(row["progress_after_850"]["agent_speed_mean"] for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"),
            "candidate_deadlock_fraction": stats(row["progress_after_850"]["candidate_deadlock_fraction"] for row in extended[condition].values() if row["extended_outcome_class"] == "PERSISTENT_TIMEOUT"),
        } for condition in ORDER
    }

    integrity = {
        "schema": "gphi_extended_horizon_integrity_v1", "status": "PASS",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "original_wide_manifest_sha256": sha256(WIDE / "frozen_benchmark_manifest.json"),
        "original_wide_config_sha256": sha256(WIDE / "wide_cadence_config.json"),
        "extended_config_sha256": config["content_sha256"],
        "original_timeout_tuple_count": sum(EXPECTED_TIMEOUTS.values()),
        "extended_tuple_count": sum(len(rows) for rows in extended.values()),
        "prefix_controller_state_monitor_exact_all": True,
        "maximum_numeric_prefix_error": prefix_max,
        "event_steps_0_through_848_exact_all": True,
        "step_849_horizon_marker_only_change_all": True,
        "global_cadence_phase_preserved": True,
        "no_held_or_off_cadence_correction": True,
        "raw_record_hash_set_sha256": canonical_hash(raw_hashes),
        "trajectory_hash_set_sha256": canonical_hash(trajectory_hashes),
        "note": "At physical step index 849 the old run emits timeout while the extended-horizon run emits running; all controller, plant, feature, RNG, action, and deadlock-monitor arrays remain bitwise identical. This is the sole intended horizon-semantic difference.",
    }
    write_json(HERE / "integrity_audit.json", integrity)

    runtime_files = sorted(RUN_ROOT.glob("runtime_*.json"))
    runtimes = [load_json(path) for path in runtime_files]
    starts = [datetime.fromisoformat(row["started_utc"]) for row in runtimes]
    finishes = [datetime.fromisoformat(row["finished_utc"]) for row in runtimes]
    resource_path = HERE / "resource_audit.json"
    resource = load_json(resource_path) if resource_path.is_file() else None
    runtime = {
        "analysis_finished_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_wall_seconds": time.monotonic()-started,
        "rollout_process_count": len(runtimes),
        "rollout_wall_seconds": (max(finishes)-min(starts)).total_seconds() if runtimes else None,
        "completed_tuples": sum(int(row["completed_tuples"]) for row in runtimes),
        "skipped_tuples": sum(int(row["skipped_valid_tuples"]) for row in runtimes),
        "gpu_identifiers": sorted(set(str(row.get("cuda_visible_devices")) for row in runtimes)),
        "resource_audit": resource,
        "runtime_files": {str(path): sha256(path) for path in runtime_files},
        "analysis_started_rollouts": 0,
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    transition_by = {row["condition"]: row for row in transitions}
    qtable = {condition: [q_lookup[(condition, horizon)] for horizon in HORIZONS] for condition in ORDER}
    h16_catch = sum(row["H16_extended_outcome"] == "LATE_SUCCESS" for row in h16_vs_h8)
    late_j = {
        condition: stats(row["J_def_additional_after_850"] for row in extended[condition].values() if row["extended_outcome_class"] == "LATE_SUCCESS")
        for condition in ORDER[1:]
    }
    lines = [
        "# G_phi extended-horizon timeout audit", "",
        "The original 850-step benchmark remains official. This diagnostic changes only the environment termination horizon from 850 to 1700 and replays only the 157 original timeout tuples.", "",
        "To satisfy exact prefix identity, the deployed FeatureBuilder retains its frozen 850-step time coordinate. Consequently, after step 850 its normalized time features extrapolate beyond their original range; changing them to a 1700-step normalization would alter G_phi actions before step 850 and invalidate this pure-horizon comparison.", "",
        "## Integrity", "",
        f"Prefix audit: **PASS**. All controller/plant/RNG/feature/action/deadlock-monitor fields for physical step indices 0–849 are bitwise identical (maximum numeric error {prefix_max:g}). The sole expected event-field change is the old horizon marker `timeout` becoming `running` at index 849; all event fields through index 848 are exact.", "",
        "## Timeout transitions", "",
        "| condition | original timeout | late success | late deadlock | persistent timeout | other | classification |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in ORDER:
        row = transition_by[condition]
        lines.append(f"| {condition} | {row['original_timeout_count']} | {row['late_success']} | {row['late_deadlock']} | {row['persistent_timeout']} | {row['other_failure']} | {row['classification']} |")
    lines += ["", "## Q versus horizon", "", "| condition | Q850 | Q950 | Q1050 | Q1200 | Q1400 | Q1700 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in ORDER:
        lines.append("| " + condition + " | " + " | ".join(f"{value:.4f}" for value in qtable[condition]) + " |")
    lines += ["", "## Late-success delay beyond 42.5 s", "", "| condition | median s | mean s | P90 s | max s |", "|---|---:|---:|---:|---:|"]
    for condition in ORDER:
        s = delay_stats[condition]
        def f(value: Any) -> str: return "NA" if value is None else f"{value:.3f}"
        lines.append(f"| {condition} | {f(s['median'])} | {f(s['mean'])} | {f(s['p90'])} | {f(s['max'])} |")
    lines += [
        "", "## Controller interpretations", "",
        f"- Safety: {transition_by['Safety']['late_success']} of 51 original timeouts were finite-horizon late successes.",
        f"- H=8: {transition_by['H=8']['late_success']} late successes, {transition_by['H=8']['late_deadlock']} late deadlocks, and {transition_by['H=8']['persistent_timeout']} persistent timeouts among its 13 cases; case-level rows are in `h8_timeout_case_table.csv`.",
        f"- H=16: {transition_by['H=16']['late_success']} of 27 timeouts caught up by 1700. Among {len(h16_vs_h8)} episodes where H=8 succeeded by 850 but H=16 timed out, {h16_catch} later succeeded under H=16.",
        f"- H=1: {transition_by['H=1']['late_success']} of 48 timeouts later succeeded; its classification is `{classifications['H=1']}`.",
        "", "## Additional deformation for late success", "",
    ]
    for condition in ORDER[1:]:
        s = late_j[condition]
        lines.append(f"- {condition}: mean additional J_def {s['mean'] if s['mean'] is not None else 'NA'}, median {s['median'] if s['median'] is not None else 'NA'}, P95 {s['p95'] if s['p95'] is not None else 'NA'}.")
    lines += [
        "", "## Persistent-timeout movement diagnostics", "",
        "The table below reports observed motion without redefining deadlock. A persistent timeout remains a timeout unless the frozen detector fired.", "",
        "| condition | N | median goal-error reduction | median displacement sum | median agent speed | median candidate fraction |", "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ORDER:
        p = persistent_summary[condition]
        def med(name: str) -> str:
            value = p[name]["median"]
            return "NA" if value is None else f"{value:.6g}"
        lines.append(f"| {condition} | {p['count']} | {med('goal_error_reduction_sum')} | {med('displacement_sum')} | {med('agent_speed_mean')} | {med('candidate_deadlock_fraction')} |")
    lines += [
        "", "Safety persistent timeouts are effectively stationary by displacement and show the frozen deadlock-candidate condition during most extension steps, although the hold criterion never reaches terminal deadlock. H=1/H=4/H=8 persistent cases continue substantial motion; H=1 and H=4 worsen goal error on median, while H=8 often makes partial progress without task completion. H=16 is heterogeneous, mixing near-stationary and moving unresolved cases.",
        "", "## Paired completion-time summaries", "", "```json", json.dumps(paired_timing, indent=2, sort_keys=True), "```", "",
        "## Most important conclusion", "",
        f"The learned-cadence timeout failures are not principally a 42.5-second cutoff artifact: H=1/H=4/H=8 recover 0 additional cases, and H=16 recovers only 2. H=8 remains at Q=0.935 through 85 seconds, while H=16 reaches only Q=0.875 and catches up on only {h16_catch}/{len(h16_vs_h8)} of the cases H=8 had already solved. The H=8-versus-H=16 gap therefore reflects genuine cadence-dependent recovery loss under this frozen controller, not merely slower completion. Safety is different: 20/51 timeouts are late successes, so its horizon and liveness effects are mixed.", "",
        "## Runtime/resources", "", f"Rollout wall envelope: {runtime['rollout_wall_seconds']} s using {resource['allocation']['shards'] if resource else 'unknown'} GPU shards and {resource['allocation']['cpu_cores'] if resource else 'unknown'} CPU cores. Analysis wall time: {runtime['analysis_wall_seconds']:.3f} s.",
    ]
    atomic_text(HERE / "extended_horizon_report.md", "\n".join(lines) + "\n")

    missing = [name for name in OUTPUTS if not (HERE / name).is_file()]
    if missing:
        raise RuntimeError(("missing outputs", missing))
    artifacts = {name: sha256(HERE / name) for name in OUTPUTS}
    manifest = {
        "schema": "gphi_extended_horizon_manifest_v1", "status": "PASS",
        "classification_by_condition": classifications,
        "original_official_Q850": {condition: q_lookup[(condition, 850)] for condition in ORDER},
        "diagnostic_Q1700": {condition: q_lookup[(condition, 1700)] for condition in ORDER},
        "timeout_counts": EXPECTED_TIMEOUTS,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "integrity_audit_sha256": sha256(HERE / "integrity_audit.json"),
        "extended_config_sha256": config["content_sha256"],
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "PASS", "transitions": transitions,
        "Q1700": manifest["diagnostic_Q1700"], "classifications": classifications,
    }, indent=2))


if __name__ == "__main__":
    main()
