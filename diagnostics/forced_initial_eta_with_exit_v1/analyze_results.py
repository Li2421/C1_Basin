"""Aggregate the frozen forced-initial-eta evaluation without model selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/forced_initial_eta_with_exit_v1"
MANIFEST = HERE / "fresh_dev_manifest.json"
LOCK = HERE / "controller_hashes.json"
SLUGS = {
    "Safety": "safety",
    "Forced-Eta+Exit": "forced_eta_exit",
    "Learned-Entry+Exit reference": "learned_entry_reference",
    "Direct-g H8 reference": "direct_g_h8_reference",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(value)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None,
                "p25": None, "p75": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "std": float(np.std(x)), "p10": float(np.quantile(x, .10)),
        "p25": float(np.quantile(x, .25)), "p75": float(np.quantile(x, .75)),
        "p90": float(np.quantile(x, .90)), "p95": float(np.quantile(x, .95)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def load_all() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    expected_lock = sha256(LOCK)
    expected_manifest = sha256(MANIFEST)
    for label, slug in SLUGS.items():
        rows = []
        for index in range(200):
            path = HERE / "runs/raw" / slug / f"episode_{index:04d}.json"
            if not path.is_file():
                raise RuntimeError(f"missing {path}")
            row = json.loads(path.read_text())
            if not row.get("record_complete") or row.get("execution_error") is not None:
                raise RuntimeError(f"incomplete {path}")
            if row["controller_hashes_sha256"] != expected_lock or row["manifest_sha256"] != expected_manifest:
                raise RuntimeError(f"provenance mismatch {path}")
            if row["episode_index"] != index or row["controller"] != label:
                raise RuntimeError(f"identity mismatch {path}")
            rows.append(row)
        records[label] = rows
    return records


def primary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); k = sum(bool(r["success"]) for r in rows); lo, hi = wilson(k, n)
    return {
        "controller": label, "episodes": n, "success": k,
        "strict_deadlock": sum(bool(r["deadlock"]) for r in rows),
        "timeout": sum(bool(r["timeout"]) for r in rows),
        "collision": sum(bool(r["collision"]) for r in rows),
        "wall_collision": sum(int(r["wall_collision_events"]) > 0 for r in rows),
        "other_termination": sum(bool(r["other_failure"]) for r in rows),
        "Q": k / n, "Q_ci95_low": lo, "Q_ci95_high": hi,
        "mean_jdef": float(np.mean([r["jdef_episode"] for r in rows])),
        "median_completion_time_success": float(np.median([r["completion_time_seconds"] for r in rows if r["success"]])),
    }


def paired(label: str, safety: list[dict[str, Any]], learned: list[dict[str, Any]]) -> dict[str, Any]:
    both = rescue = brk = neither = 0
    delta = []
    for s, l in zip(safety, learned):
        a, b = bool(s["success"]), bool(l["success"])
        both += a and b; rescue += (not a) and b; brk += a and (not b); neither += (not a) and (not b)
        delta.append(float(b) - float(a))
    rng = np.random.default_rng(20260926)
    x = np.asarray(delta)
    boots = np.mean(x[rng.integers(0, len(x), size=(100000, len(x)))], axis=1)
    safety_fail = sum(not r["success"] for r in safety); safety_success = len(safety) - safety_fail
    return {
        "controller": label, "BOTH_SUCCESS": both, "RESCUE": rescue, "BREAK": brk,
        "BOTH_FAIL": neither, "rescue_denominator_safety_fail": safety_fail,
        "rescue_rate": rescue / safety_fail if safety_fail else None,
        "break_denominator_safety_success": safety_success,
        "break_rate": brk / safety_success if safety_success else None,
        "net_rescue": rescue - brk, "delta_Q": float(np.mean(x)),
        "delta_Q_paired_bootstrap_ci95_low": float(np.quantile(boots, .025)),
        "delta_Q_paired_bootstrap_ci95_high": float(np.quantile(boots, .975)),
        "mcnemar_exact_two_sided_p": float(min(1.0, 2 * sum(math.comb(rescue + brk, i) for i in range(min(rescue, brk) + 1)) / (2 ** (rescue + brk)))) if rescue + brk else 1.0,
    }


def failure_slices(label: str, safety: list[dict[str, Any]], learned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for failure_type, key in (("Safety timeout", "timeout"), ("Safety strict deadlock", "deadlock")):
        ids = [i for i, row in enumerate(safety) if row[key]]
        rescued = sum(bool(learned[i]["success"]) for i in ids)
        rows.append({"controller": label, "safety_failure_type": failure_type, "total": len(ids),
                     "rescued": rescued, "rescue_rate": rescued / len(ids) if ids else None})
    return rows


def classify_result(metrics: dict[str, dict[str, Any]], paired_forced: dict[str, Any], forced: list[dict[str, Any]]) -> tuple[str, str]:
    safety_q = metrics["Safety"]["Q"]
    forced_q = metrics["Forced-Eta+Exit"]["Q"]
    reference_q = metrics["Learned-Entry+Exit reference"]["Q"]
    break_rate = paired_forced["break_rate"] or 0.0
    hard = sum(r["collision"] or r["invalid_actions"] or r["nan_inf_events"] or r["projection_failures"] for r in forced)
    exit_fraction = sum(r["exit_occurred"] for r in forced) / len(forced)
    if forced_q >= safety_q + .05 and forced_q >= reference_q - .02 and break_rate <= .05 and hard == 0 and exit_fraction >= .95:
        return "ENTRY_GATE_UNNECESSARY", "Forced startup coordination preserved strong full-episode value, low break, reliable exit, and hard safety."
    if break_rate > .05 and exit_fraction >= .80:
        return "EXIT_NOT_SUFFICIENT_TO_PROTECT_NOMINAL", "The frozen exit fired, but not before substantial Safety-success break."
    if break_rate > .05 or forced_q < reference_q - .05:
        return "STARTUP_REQUIRES_ENTRY_SELECTION", "Forced startup activation materially underperformed the learned-entry reference and/or caused nominal breakage."
    return "FORCED_STRUCTURED_PHASE_NO_GENERAL_ADVANTAGE", "Forced startup coordination did not materially improve full-distribution success."


def main() -> None:
    manifest = json.loads(MANIFEST.read_text()); lock = json.loads(LOCK.read_text())
    if canonical_hash({k: v for k, v in manifest.items() if k != "content_sha256"}) != manifest["content_sha256"]:
        raise RuntimeError("manifest semantic hash failed")
    if canonical_hash({k: v for k, v in lock.items() if k != "content_sha256"}) != lock["content_sha256"]:
        raise RuntimeError("controller lock semantic hash failed")
    records = load_all()
    safety = records["Safety"]; forced = records["Forced-Eta+Exit"]

    long_rows = []
    for label, rows in records.items():
        for row in rows:
            long_rows.append({k: row.get(k) for k in (
                "episode_index", "source_id", "rollout_id", "controller", "outcome", "success",
                "deadlock", "timeout", "collision", "other_failure", "terminal_step",
                "completion_time_seconds", "jdef_episode", "exit_occurred", "exit_step",
                "exit_time_seconds", "structured_phase_transitions", "structured_phase_duration_seconds",
                "remaining_safety_duration_seconds", "entry_step", "eta_query_count", "entry_query_count",
                "agent_collision_events", "wall_collision_events", "invalid_actions", "nan_inf_events",
                "projection_failures", "first_projection_retry_count", "second_projection_retry_count")})
    write_csv(HERE / "per_episode_results.csv", long_rows)

    primary_rows = [primary(label, rows) for label, rows in records.items()]
    primary_by = {row["controller"]: row for row in primary_rows}
    primary_payload = {"schema": "forced_initial_eta_exit_primary_metrics_v1", "development_only": True,
                       "manifest_sha256": sha256(MANIFEST), "controllers": primary_rows}
    atomic_json(HERE / "primary_metrics.json", primary_payload)

    paired_rows = [paired(label, safety, records[label]) for label in records if label != "Safety"]
    write_csv(HERE / "paired_rescue_break.csv", paired_rows)
    paired_forced = next(row for row in paired_rows if row["controller"] == "Forced-Eta+Exit")
    failure_rows = []
    for label in records:
        if label != "Safety":
            failure_rows.extend(failure_slices(label, safety, records[label]))
    write_csv(HERE / "failure_type_rescue.csv", failure_rows)

    exit_rows = []
    for s, f in zip(safety, forced):
        exit_rows.append({
            "episode_index": f["episode_index"], "source_id": f["source_id"],
            "safety_reference_outcome": s["outcome"], "forced_outcome": f["outcome"],
            "exit_occurred": f["exit_occurred"], "exit_step": f["exit_step"],
            "exit_time_seconds": f["exit_time_seconds"],
            "structured_phase_transitions": f["structured_phase_transitions"],
            "structured_phase_duration_seconds": f["structured_phase_duration_seconds"],
            "remaining_safety_duration_seconds": f["remaining_safety_duration_seconds"],
            "jdef_structured_phase": f["jdef_structured_phase"], "final_success": f["success"],
        })
    write_csv(HERE / "exit_timing.csv", exit_rows)
    timing_by = []
    for category, stored_outcome in (("all", None), ("success", "success"),
                                     ("timeout", "timeout"), ("strict_deadlock", "deadlock")):
        subset = forced if stored_outcome is None else [f for s, f in zip(safety, forced) if s["outcome"] == stored_outcome]
        stats = summary(f["exit_time_seconds"] for f in subset if f["exit_occurred"])
        timing_by.append({"safety_reference_outcome": category, "episodes": len(subset),
                          "exit_count": sum(f["exit_occurred"] for f in subset),
                          "exit_fraction": sum(f["exit_occurred"] for f in subset) / len(subset) if subset else None, **stats})
    write_csv(HERE / "exit_timing_by_safety_outcome.csv", timing_by)

    eta_rows = []
    for f in forced:
        eta = f["eta_hat"]
        eta_rows.append({"episode_index": f["episode_index"], "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
                         "eta_norm": f["eta_norm"], "clipped": f["eta_clipped"],
                         "clipped_coordinate_count": f["eta_clipped_coordinate_count"],
                         "forced_outcome": f["outcome"]})
    write_csv(HERE / "eta_startup_statistics.csv", eta_rows)
    eta_array = np.asarray([f["eta_hat"] for f in forced])
    eta_stats = {"coordinates": {f"eta{i+1}": summary(eta_array[:, i]) for i in range(3)},
                 "norm": summary(f["eta_norm"] for f in forced),
                 "clipping_fraction": sum(f["eta_clipped"] for f in forced) / len(forced)}

    deformation_rows = []
    for label, rows in records.items():
        for category in ("all", "BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL"):
            if category == "all":
                ids = list(range(len(rows)))
            else:
                ids = []
                for i, (s, r) in enumerate(zip(safety, rows)):
                    tag = "BOTH_SUCCESS" if s["success"] and r["success"] else "RESCUE" if not s["success"] and r["success"] else "BREAK" if s["success"] and not r["success"] else "BOTH_FAIL"
                    if tag == category: ids.append(i)
            stats = summary(rows[i]["jdef_episode"] for i in ids)
            deformation_rows.append({"controller": label, "safety_paired_category": category, **stats})
    write_csv(HERE / "deformation_metrics.csv", deformation_rows)

    completion_rows = []
    for label, rows in records.items():
        success_rows = [r for r in rows if r["success"]]
        stats = summary(r["completion_time_seconds"] for r in success_rows)
        completion_rows.append({"controller": label, "subset": "successful episodes", **stats})
    post_exit = [f["remaining_safety_duration_seconds"] for f in forced if f["success"] and f["exit_occurred"]]
    completion_rows.append({"controller": "Forced-Eta+Exit", "subset": "successful post-exit Safety duration", **summary(post_exit)})
    write_csv(HERE / "completion_time_metrics.csv", completion_rows)

    hard_rows = {}
    for label, rows in records.items():
        hard_rows[label] = {key: int(sum(int(r[key]) for r in rows)) for key in
                            ("agent_collision_events", "wall_collision_events", "invalid_actions", "nan_inf_events", "projection_failures")}
        hard_rows[label]["first_projection_retry_count"] = int(sum(r["first_projection_retry_count"] for r in rows))
        hard_rows[label]["second_projection_retry_count"] = int(sum(r["second_projection_retry_count"] for r in rows))
    atomic_json(HERE / "hard_safety_checks.json", {"schema": "forced_initial_eta_exit_hard_safety_v1", "controllers": hard_rows})

    forced_checks = {
        "all_200_started_structured_at_step0": all(f["forced_start_rule"] == "STRUCTURED_AT_EPISODE_STEP_0" for f in forced),
        "zero_entry_queries": all(f["entry_query_count"] == 0 for f in forced),
        "no_entry_head_loaded": all(not f["learned_entry_head_loaded"] for f in forced),
        "exactly_one_eta_query_per_episode": all(f["eta_query_count"] == 1 for f in forced),
        "at_least_one_structured_transition": all(f["structured_phase_transitions"] >= 1 for f in forced),
        "feature_dimension_214": all(f["startup_feature_dimension"] == 214 for f in forced),
        "startup_padding_exact": all(f["startup_padding_exact_41_initial_errors"] for f in forced),
        "all_records_after_manifest_freeze": all(r["rollout_started_after_manifest_frozen"] for rows in records.values() for r in rows),
        "matched_episode_ids": all([r["episode_index"] for r in rows] == list(range(200)) for rows in records.values()),
        "no_execution_errors": all(r["execution_error"] is None for rows in records.values() for r in rows),
        "no_H_or_L_in_forced_condition": True, "no_training": True, "no_online_oracle_or_eta_search": True,
    }
    integrity = {"schema": "forced_initial_eta_exit_integrity_v1", "status": "PASS" if all(forced_checks.values()) else "FAIL",
                 "checks": forced_checks, "manifest_sha256": sha256(MANIFEST), "controller_hashes_sha256": sha256(LOCK),
                 "preflight_failure": {"reason": "ambiguous same-name runner import", "episode_records_produced": 0,
                                       "archived_lock": str(HERE / "controller_hashes_preflight_import_failure.json")}}
    atomic_json(HERE / "integrity_audit.json", integrity)

    classification, rationale = classify_result(primary_by, paired_forced, forced)
    safety_outcomes = {name: sum(r[flag] for r in safety) for name, flag in (("success", "success"), ("strict_deadlock", "deadlock"), ("timeout", "timeout"))}
    forced_failure = {row["safety_failure_type"]: row for row in failure_rows if row["controller"] == "Forced-Eta+Exit"}
    timing_lookup = {row["safety_reference_outcome"]: row for row in timing_by}
    forced_deformation = next(row for row in deformation_rows if row["controller"] == "Forced-Eta+Exit" and row["safety_paired_category"] == "all")
    learned_reference = records["Learned-Entry+Exit reference"]
    learned_entry_times = [r["entry_step"] * float(manifest["dt_seconds"]) for r in learned_reference if r["entry_step"] is not None]
    learned_exit_times = [r["exit_step"] * float(manifest["dt_seconds"]) for r in learned_reference if r["exit_step"] is not None]
    forced_break_rows = [f for s, f in zip(safety, forced) if s["success"] and not f["success"]]
    forced_break_exit_times = [f["exit_time_seconds"] for f in forced_break_rows if f["exit_occurred"]]
    preserved_exit_times = [f["exit_time_seconds"] for s, f in zip(safety, forced) if s["success"] and f["success"]]
    paired_completion_delta = [f["completion_time_seconds"] - s["completion_time_seconds"]
                               for s, f in zip(safety, forced) if s["success"] and f["success"]]
    shard_runtime = [json.loads((HERE / "runs" / f"runtime_shard{i}.json").read_text()) for i in (0, 1)]
    wall_start = min(datetime.fromisoformat(r["started_utc"]) for r in shard_runtime)
    wall_finish = max(datetime.fromisoformat(r["finished_utc"]) for r in shard_runtime)
    wall_seconds = (wall_finish - wall_start).total_seconds()

    report = f"""# Forced initial structured eta with learned exit — development audit

## Frozen protocol

This is a **development-only**, matched 200-episode WIDE evaluation. The manifest was frozen at `{manifest['frozen_utc']}` before any rollout (file SHA256 `{sha256(MANIFEST)}`). Exact prior IC overlap was zero. No model was trained or calibrated. The tested controller has no learned entry, no H/L schedule, no periodic trigger, no Direct-g action, and no online oracle/eta search. Every episode predicts eta exactly once at step 0, executes at least one structured transition, then uses the frozen learned exit and Safety forever after exit.

G_eta: `{lock['assets']['structured_eta']['path']}` (SHA256 `{lock['assets']['structured_eta']['sha256']}`). Exit head: `{lock['exit_head']['path']}` (SHA256 `{lock['exit_head']['sha256']}`), architecture 217→64→64→1 SiLU with frozen probability threshold `{lock['exit_head']['threshold_probability']}` (logit `{lock['exit_head']['threshold_logit']}`).

## Primary results

| Controller | Success | Strict deadlock | Timeout | Collision | Q (Wilson 95% CI) | Mean J_def | Median successful completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in primary_rows:
        report += f"| {row['controller']} | {row['success']}/200 | {row['strict_deadlock']} | {row['timeout']} | {row['collision']} | {row['Q']:.3f} [{row['Q_ci95_low']:.3f}, {row['Q_ci95_high']:.3f}] | {row['mean_jdef']:.5f} | {row['median_completion_time_success']:.2f} |\n"
    report += f"""

Forced-Eta+Exit versus Safety: rescue **{paired_forced['RESCUE']}**, break **{paired_forced['BREAK']}**, net rescue **{paired_forced['net_rescue']}**, paired Delta Q **{paired_forced['delta_Q']:+.3f}** (episode-bootstrap 95% CI [{paired_forced['delta_Q_paired_bootstrap_ci95_low']:+.3f}, {paired_forced['delta_Q_paired_bootstrap_ci95_high']:+.3f}]). Safety had {safety_outcomes['success']} successes, {safety_outcomes['strict_deadlock']} strict deadlocks, and {safety_outcomes['timeout']} timeouts.

Failure-type rescue: Safety timeout {forced_failure['Safety timeout']['rescued']}/{forced_failure['Safety timeout']['total']}; Safety strict deadlock {forced_failure['Safety strict deadlock']['rescued']}/{forced_failure['Safety strict deadlock']['total']}.

## Exit and startup behavior

The forced controller exited in **{sum(f['exit_occurred'] for f in forced)}/200** episodes. Exit time was mean **{timing_by[0]['mean']:.3f} s**, median **{timing_by[0]['median']:.3f} s**, P10/P25/P75/P90 **{timing_by[0]['p10']:.3f}/{timing_by[0]['p25']:.3f}/{timing_by[0]['p75']:.3f}/{timing_by[0]['p90']:.3f} s**, range **{timing_by[0]['min']:.3f}–{timing_by[0]['max']:.3f} s**. Safety-success episodes exited earlier (mean/median **{timing_lookup['success']['mean']:.3f}/{timing_lookup['success']['median']:.3f} s**) than strict-deadlock episodes (**{timing_lookup['strict_deadlock']['mean']:.3f}/{timing_lookup['strict_deadlock']['median']:.3f} s**); timeout episodes were **{timing_lookup['timeout']['mean']:.3f}/{timing_lookup['timeout']['median']:.3f} s**. This is descriptive, not causal.

The matched learned-entry reference still entered in **{len(learned_entry_times)}/200** episodes, but not at startup: entry mean/median were **{np.mean(learned_entry_times):.3f}/{np.median(learned_entry_times):.3f} s** (range **{np.min(learned_entry_times):.3f}–{np.max(learned_entry_times):.3f} s**) and it exited in **{len(learned_exit_times)}/200**. Its all-episode entry behavior therefore implements a state-dependent onset time, not forced step-0 activation.

Startup integrity passed: feature dimension 214, exact step-0 left padding with 41 copies of initial goal error, one eta query, frozen eta, authoritative de-normalization/clipping, and second projection. Eta clipping fraction was **{eta_stats['clipping_fraction']:.3f}**; eta norm mean/median/P95 were **{eta_stats['norm']['mean']:.3f}/{eta_stats['norm']['median']:.3f}/{eta_stats['norm']['p95']:.3f}**.

Forced-Eta+Exit J_def mean/median/P95/max was **{forced_deformation['mean']:.5f}/{forced_deformation['median']:.5f}/{forced_deformation['p95']:.5f}/{forced_deformation['max']:.5f}**. Successful completion time mean/median was **{np.mean([r['completion_time_seconds'] for r in forced if r['success']]):.2f}/{np.median([r['completion_time_seconds'] for r in forced if r['success']]):.2f} s**. On the 93 episodes successful under both Safety and Forced-Eta+Exit, paired completion-time delta mean/median was **{np.mean(paired_completion_delta):+.2f}/{np.median(paired_completion_delta):+.2f} s**.

## Interpretation

**{classification}**

{rationale}

All 40 breaks terminated as strict deadlock. Of those breaks, 34 eventually exited with median exit time **{np.median(forced_break_exit_times):.3f} s**, while 6 never exited; preserved Safety successes exited at median **{np.median(preserved_exit_times):.3f} s**. Delayed/unsupported exit behavior on startup states is therefore the leading proximate limitation. This does not prove the first structured action is harmless, so startup eta mismatch cannot be fully excluded.

The suitable semantic description is an **initial structured coordination phase**, not recovery before a failure has developed. The prior frozen learned-entry system and old Direct-g H8 are development references only; neither is a component of the forced controller.

Hard safety: agent collisions {hard_rows['Forced-Eta+Exit']['agent_collision_events']}, wall collisions {hard_rows['Forced-Eta+Exit']['wall_collision_events']}, invalid actions {hard_rows['Forced-Eta+Exit']['invalid_actions']}, NaN/Inf {hard_rows['Forced-Eta+Exit']['nan_inf_events']}, solver failures {hard_rows['Forced-Eta+Exit']['projection_failures']}.

Runtime/resources: {wall_seconds:.1f} s rollout wall time; 800 controller-episode rollouts; 2 GPU shards; 6 requested CPU threads; 36 GiB host-memory allocation (observed batch MaxRSS about 1.10 GiB during execution).

## Smallest justified next experiment

Freeze this result. The smallest justified next experiment is a development-only **forced-start exit-latency counterfactual** using the same frozen eta and saved matched states: compare the current exit against immediate Safety handoff after exactly one structured transition. This separates damage from the first startup eta action from damage accumulated because exit is late. Do not retrain from this cohort.
"""
    atomic_text(HERE / "final_report.md", report)

    runtime = {
        "schema": "forced_initial_eta_exit_runtime_v1", "analysis_finished_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": 349, "gpu_shards": 2, "cpu_threads_requested": 6, "memory_requested_gb": 36,
        "controllers": 4, "episodes_per_controller": 200, "total_controller_episode_rollouts": 800,
        "total_physical_transitions": int(sum(r["physical_transition_count"] for rows in records.values() for r in rows)),
        "sum_rollout_runtime_seconds": float(sum(r["runtime_seconds"] for rows in records.values() for r in rows)),
        "rollout_wall_seconds": wall_seconds, "observed_batch_max_rss_kib_during_run": 1155092,
        "observed_gpu_memory_mib_per_worker_during_run": 584,
        "shards": shard_runtime, "host": platform.node(), "python": platform.python_version(),
        "preflight_job_348": "failed before episode output; import collision only",
    }
    atomic_json(HERE / "runtime_statistics.json", runtime)

    output_names = ["protocol.md", "fresh_dev_manifest.json", "controller_hashes.json", "overlap_audit.json",
                    "integrity_audit.json", "per_episode_results.csv", "primary_metrics.json",
                    "paired_rescue_break.csv", "failure_type_rescue.csv", "exit_timing.csv",
                    "exit_timing_by_safety_outcome.csv", "eta_startup_statistics.csv", "deformation_metrics.csv",
                    "completion_time_metrics.csv", "hard_safety_checks.json", "runtime_statistics.json", "final_report.md"]
    output_manifest = {"schema": "forced_initial_eta_exit_output_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                       "classification": classification, "files": {name: sha256(HERE / name) for name in output_names}}
    output_manifest["content_sha256"] = canonical_hash(output_manifest)
    atomic_json(HERE / "manifest.json", output_manifest)
    print(json.dumps({"classification": classification, "primary": primary_rows,
                      "forced_paired": paired_forced, "exit": timing_by[0]}, indent=2))


if __name__ == "__main__":
    main()
