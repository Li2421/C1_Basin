"""Aggregate exact-flow search and 64-seed fixed-eta capacity validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
QUERY_ORDER = {"S0": 0, "S_8s": 1, "S_4s": 2, "S_2s": 3, "S_1s": 4, "S_pre": 5}
ROBUST_THRESHOLD = 63
ROBUST_N = 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)), "min": float(array.min()), "max": float(array.max()),
    }


def eta_text(eta: Any) -> str:
    return json.dumps([float(value) for value in eta], separators=(",", ":"))


def load_mode(mode: str, state_ids: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    directory = HERE / "raw" / mode
    actual = sorted(directory.glob("*.jsonl"))
    if {path.stem for path in actual} != set(state_ids):
        raise RuntimeError(("incomplete state files", mode, len(actual), len(state_ids)))
    for state_id in state_ids:
        rows = read_jsonl(directory / f"{state_id}.jsonl")
        if not rows or rows[-1].get("state_complete") is not True or rows[-1].get("mode") != mode:
            raise RuntimeError(("incomplete state", mode, state_id))
        body = rows[:-1]
        if mode == "robust":
            grouped = Counter(eta_text(row["eta"]) for row in body)
            if not grouped or any(count != ROBUST_N for count in grouped.values()):
                raise RuntimeError(("robust candidate not evaluated on 64 seeds", state_id, grouped))
            for eta in grouped:
                seeds = {int(row["seed"]) for row in body if eta_text(row["eta"]) == eta}
                if seeds != set(range(95310001, 95310065)):
                    raise RuntimeError(("robust seed mismatch", state_id, eta))
        result.extend(body)
    return result


def main() -> None:
    reproduction = json.loads((HERE / "reproduction_audit.json").read_text())
    if reproduction.get("status") != "PASS" or reproduction.get("all_17_exact") is not True or reproduction.get("case_count") != 17:
        raise RuntimeError("17-case exact reproduction did not pass")
    query = read_csv(HERE / "queried_states.csv")
    if len(query) != 102 or Counter(row["case_id"] for row in query) != Counter({row["case_id"]: 6 for row in query}):
        raise RuntimeError("query-state manifest is not 17 x 6")
    query_by_state = {row["state_id"]: row for row in query}
    state_ids = sorted(query_by_state)
    exact = load_mode("exact", state_ids)
    robust = load_mode("robust", state_ids)
    if any(row.get("execution_error") is not None for row in exact + robust):
        raise RuntimeError("capacity audit contains execution errors")
    if any(row["state_id"] not in query_by_state for row in exact + robust):
        raise RuntimeError("unknown query state in search outputs")

    exact_csv = []
    exact_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        exact_by_state[row["state_id"]].append(row)
        exact_csv.append({
            "state_id": row["state_id"], "case_id": row["case_id"], "benchmark": row["benchmark"],
            "query_label": row["query_label"], "query_step": row["query_step"],
            "seconds_before_deadlock": row["seconds_before_deadlock"], "search_stage": row["search_stage"],
            "candidate_id": row["candidate_id"], "eta": eta_text(row["eta"]), "outcome": row["outcome"],
            "J_def": row["J_def"], "steps": row["steps"], "terminal_step": row["terminal_step"],
            "raw_correction_norm_mean": row["raw_correction_norm_mean"],
            "executed_correction_norm_mean": row["executed_correction_norm_mean"],
            "projection_rewrite_norm_mean": row["projection_rewrite_norm_mean"],
            "projection_rewrite_fraction_of_raw": row["projection_rewrite_fraction_of_raw"],
            "first_projection_retries": row["first_projection_retries"], "second_projection_retries": row["second_projection_retries"],
        })
    write_csv(HERE / "eta_search_results.csv", exact_csv)

    robust_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in robust:
        robust_groups[(row["state_id"], eta_text(row["eta"]))].append(row)
    robust_summary: list[dict[str, Any]] = []
    robust_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (state_id, eta), rows in sorted(robust_groups.items()):
        counts = Counter(row["outcome"] for row in rows)
        summary = {
            "state_id": state_id, "case_id": rows[0]["case_id"], "benchmark": rows[0]["benchmark"],
            "query_label": rows[0]["query_label"], "seconds_before_deadlock": rows[0]["seconds_before_deadlock"],
            "eta": eta, "evaluated": len(rows), "success": counts["success"], "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"], "execution_error": counts["execution_error"],
            "B_63_member": counts["success"] >= ROBUST_THRESHOLD,
            "J_def_mean": float(np.mean([row["J_def"] for row in rows])),
            "J_def_median": float(np.median([row["J_def"] for row in rows])),
            "raw_correction_norm_mean": float(np.mean([row["raw_correction_norm_mean"] for row in rows])),
            "executed_correction_norm_mean": float(np.mean([row["executed_correction_norm_mean"] for row in rows])),
            "projection_rewrite_norm_mean": float(np.mean([row["projection_rewrite_norm_mean"] for row in rows])),
            "projection_rewrite_fraction_of_raw": float(np.mean([row["projection_rewrite_fraction_of_raw"] for row in rows])),
        }
        robust_summary.append(summary); robust_by_state[state_id].append(summary)
    write_csv(HERE / "robust_basin_validation.csv", robust_summary)

    exact_rescue: list[dict[str, Any]] = []
    best_states: list[dict[str, Any]] = []
    best_by_state: dict[str, dict[str, Any]] = {}
    for state_id in state_ids:
        meta = query_by_state[state_id]
        rows = exact_by_state[state_id]
        success_rows = sorted((row for row in rows if row["outcome"] == "success"), key=lambda row: (float(row["J_def"]), eta_text(row["eta"])))
        robust_rows = sorted((row for row in robust_by_state[state_id] if row["B_63_member"]), key=lambda row: (float(row["J_def_mean"]), row["eta"]))
        validation_rows = sorted(robust_by_state[state_id], key=lambda row: (-int(row["success"]), float(row["J_def_mean"]), row["eta"]))
        classification = "ROBUST_SUCCESS_BASIN" if robust_rows else "FRAGILE_SUCCESS_BASIN" if success_rows else "NO_SUCCESS_FOUND"
        exact_rescue.append({
            "state_id": state_id, "case_id": meta["case_id"], "benchmark": meta["benchmark"],
            "query_label": meta["query_label"], "seconds_before_deadlock": meta["seconds_before_deadlock"],
            "eta_evaluations": len(rows), "exact_success_candidates": len(success_rows),
            "exact_success_fraction": len(success_rows) / len(rows),
            "any_exact_flow_rescue": bool(success_rows),
            "minimum_exact_J_def": success_rows[0]["J_def"] if success_rows else None,
            "minimum_exact_J_eta": eta_text(success_rows[0]["eta"]) if success_rows else None,
            "classification": classification,
        })
        if robust_rows:
            best = robust_rows[0]
            best_row = {
                "state_id": state_id, "case_id": meta["case_id"], "benchmark": meta["benchmark"],
                "query_label": meta["query_label"], "seconds_before_deadlock": float(meta["seconds_before_deadlock"]),
                "classification": classification, "best_eta": best["eta"], "success_count": best["success"],
                "evaluated": ROBUST_N, "B_63_member": True, "J_def": best["J_def_mean"],
                "validated_candidate_count": len(validation_rows),
            }
        elif success_rows:
            best_validation = validation_rows[0] if validation_rows else None
            best_row = {
                "state_id": state_id, "case_id": meta["case_id"], "benchmark": meta["benchmark"],
                "query_label": meta["query_label"], "seconds_before_deadlock": float(meta["seconds_before_deadlock"]),
                "classification": classification, "best_eta": eta_text(success_rows[0]["eta"]),
                "success_count": best_validation["success"] if best_validation else None,
                "evaluated": ROBUST_N if best_validation else 1, "B_63_member": False,
                "J_def": best_validation["J_def_mean"] if best_validation else success_rows[0]["J_def"],
                "validated_candidate_count": len(validation_rows),
            }
        else:
            best_row = {
                "state_id": state_id, "case_id": meta["case_id"], "benchmark": meta["benchmark"],
                "query_label": meta["query_label"], "seconds_before_deadlock": float(meta["seconds_before_deadlock"]),
                "classification": classification, "best_eta": None, "success_count": None,
                "evaluated": 0, "B_63_member": False, "J_def": None,
                "validated_candidate_count": 0,
            }
        best_states.append(best_row); best_by_state[state_id] = best_row
    write_csv(HERE / "exact_flow_rescue.csv", exact_rescue)
    write_csv(HERE / "best_eta_by_state.csv", best_states)

    manifest = json.loads((HERE / "strict_deadlock_manifest.json").read_text())
    cases = manifest["cases"]
    onset_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for case in cases:
        members = sorted((row for row in best_states if row["case_id"] == case["case_id"]), key=lambda row: QUERY_ORDER[row["query_label"]])
        successful = [row for row in members if row["classification"] in ("ROBUST_SUCCESS_BASIN", "FRAGILE_SUCCESS_BASIN")]
        robust_members = [row for row in members if row["classification"] == "ROBUST_SUCCESS_BASIN"]
        earliest_success = successful[0] if successful else None
        earliest_robust = robust_members[0] if robust_members else None
        if earliest_robust:
            episode_class = "EARLY_ROBUSTLY_RESCUABLE" if earliest_robust["query_label"] in ("S0", "S_8s", "S_4s") else "LATE_ROBUSTLY_RESCUABLE"
            chosen = earliest_robust
        elif earliest_success:
            episode_class = "FRAGILE_ONLY"; chosen = earliest_success
        else:
            episode_class = "NO_BASIN_FOUND"; chosen = None
        onset = {
            "case_id": case["case_id"], "benchmark": case["benchmark"], "episode_index": case["episode_index"],
            "terminal_step": case["terminal_step"],
            "earliest_success_state": earliest_success["state_id"] if earliest_success else None,
            "earliest_success_label": earliest_success["query_label"] if earliest_success else None,
            "earliest_success_seconds_before_deadlock": earliest_success["seconds_before_deadlock"] if earliest_success else None,
            "earliest_robust_state": earliest_robust["state_id"] if earliest_robust else None,
            "earliest_robust_label": earliest_robust["query_label"] if earliest_robust else None,
            "earliest_robust_seconds_before_deadlock": earliest_robust["seconds_before_deadlock"] if earliest_robust else None,
        }
        onset_rows.append(onset)
        episode_rows.append({
            **onset, "episode_capacity_classification": episode_class,
            "best_eta_at_capacity_onset": chosen["best_eta"] if chosen else None,
            "success_count": chosen["success_count"] if chosen else None,
            "evaluated": chosen["evaluated"] if chosen else None,
            "J_def": chosen["J_def"] if chosen else None,
        })
    write_csv(HERE / "basin_onset_by_episode.csv", onset_rows)
    write_csv(HERE / "episode_capacity_classification.csv", episode_rows)

    class_counts = Counter(row["episode_capacity_classification"] for row in episode_rows)
    benchmark_rows = []
    for label, selected in (
        ("historical", [row for row in episode_rows if row["benchmark"] == "historical"]),
        ("fresh_unseen", [row for row in episode_rows if row["benchmark"] == "fresh_unseen"]),
        ("combined", episode_rows),
    ):
        counts = Counter(row["episode_capacity_classification"] for row in selected)
        benchmark_rows.append({
            "benchmark": label, "episodes": len(selected),
            "episodes_with_any_successful_eta": sum(row["earliest_success_state"] is not None for row in selected),
            "episodes_with_robust_B63_basin": sum(row["earliest_robust_state"] is not None for row in selected),
            "EARLY_ROBUSTLY_RESCUABLE": counts["EARLY_ROBUSTLY_RESCUABLE"],
            "LATE_ROBUSTLY_RESCUABLE": counts["LATE_ROBUSTLY_RESCUABLE"],
            "FRAGILE_ONLY": counts["FRAGILE_ONLY"], "NO_BASIN_FOUND": counts["NO_BASIN_FOUND"],
        })
    write_csv(HERE / "old_vs_fresh_summary.csv", benchmark_rows)

    projection_rows = []
    for label, selected in (
        ("exact_success", [row for row in exact if row["outcome"] == "success"]),
        ("exact_non_success", [row for row in exact if row["outcome"] != "success"]),
        ("robust_candidate_rollouts_success", [row for row in robust if row["outcome"] == "success"]),
        ("robust_candidate_rollouts_non_success", [row for row in robust if row["outcome"] != "success"]),
    ):
        projection_rows.append({
            "scope": label, "rollouts": len(selected),
            "raw_correction_norm_mean": float(np.mean([row["raw_correction_norm_mean"] for row in selected])) if selected else None,
            "executed_correction_norm_mean": float(np.mean([row["executed_correction_norm_mean"] for row in selected])) if selected else None,
            "projection_rewrite_norm_mean": float(np.mean([row["projection_rewrite_norm_mean"] for row in selected])) if selected else None,
            "projection_rewrite_fraction_of_raw_mean": float(np.mean([row["projection_rewrite_fraction_of_raw"] for row in selected])) if selected else None,
            "second_projection_retry_count": sum(int(row["second_projection_retries"]) for row in selected),
        })
    write_csv(HERE / "projection_analysis.csv", projection_rows)
    unresolved = [row for row in episode_rows if row["episode_capacity_classification"] == "NO_BASIN_FOUND"]
    write_csv(HERE / "unresolved_cases.csv", unresolved, list(episode_rows[0].keys()))

    robust_episode_count = sum(row["earliest_robust_state"] is not None for row in episode_rows)
    early_count = class_counts["EARLY_ROBUSTLY_RESCUABLE"]
    late_count = class_counts["LATE_ROBUSTLY_RESCUABLE"]
    any_success_count = sum(row["earliest_success_state"] is not None for row in episode_rows)
    if robust_episode_count >= 9:
        classification = "ETA_CAPACITY_PRESENT_FOR_STRICT_DEADLOCK" if early_count >= late_count else "ETA_CAPACITY_PRESENT_BUT_LATE"
    elif any_success_count >= 9:
        classification = "ETA_CAPACITY_FRAGILE"
    else:
        classification = "EVIDENCE_FOR_ETA_CAPACITY_LIMITATION"
    if classification == "ETA_CAPACITY_PRESENT_FOR_STRICT_DEADLOCK":
        bottleneck = "learned G_phi approximation/deployment"
        next_experiment = "Add only the audited early robust-deadlock states and their minimum-J_def B63 targets to the unified supervised set, retrain one unchanged G_phi, then repeat frozen H=8 evaluation on a new held-out wide cohort."
    elif classification == "ETA_CAPACITY_PRESENT_BUT_LATE":
        bottleneck = "temporal localization"
        next_experiment = "Run one oracle-only timing audit that activates each episode's frozen robust eta at the audited onset checkpoints, without retraining or changing the eta family."
    elif classification == "ETA_CAPACITY_FRAGILE":
        bottleneck = "unresolved"
        next_experiment = "Densify validation only around the fragile successful eta cells inside the same frozen domain using fresh Flow seeds."
    else:
        bottleneck = "representation capacity"
        next_experiment = "On these same frozen query states, compare one pre-registered minimally augmented eta basis against the frozen 3-D family offline; do not train G_phi yet."
    onset_seconds = [float(row["earliest_robust_seconds_before_deadlock"]) for row in episode_rows if row["earliest_robust_seconds_before_deadlock"] is not None]
    success_counts = [int(row["success"]) for row in robust_summary if row["B_63_member"]]
    exact_success_fractions = [float(row["exact_success_fraction"]) for row in exact_rescue if row["any_exact_flow_rescue"]]
    projection_lookup = {row["scope"]: row for row in projection_rows}
    projection_limit = False
    if projection_lookup["exact_non_success"]["projection_rewrite_fraction_of_raw_mean"] is not None and projection_lookup["exact_success"]["projection_rewrite_fraction_of_raw_mean"] is not None:
        projection_limit = projection_lookup["exact_non_success"]["projection_rewrite_fraction_of_raw_mean"] > projection_lookup["exact_success"]["projection_rewrite_fraction_of_raw_mean"] + 0.25

    runtime_files = sorted(HERE.glob("runtime_*_shard*.json"))
    runtime_rows = [json.loads(path.read_text()) for path in runtime_files]
    analysis_time = datetime.now(timezone.utc)
    audit_start = datetime.fromisoformat(manifest["created_utc"])
    runtime = {
        "runtime_records": runtime_rows,
        "total_new_continuation_rollouts": len(exact) + len(robust),
        "exact_flow_search_rollouts": len(exact),
        "robust_validation_rollouts": len(robust),
        "total_physical_steps": sum(int(row["steps"]) for row in exact + robust),
        "exact_flow_search_physical_steps": sum(int(row["steps"]) for row in exact),
        "robust_validation_physical_steps": sum(int(row["steps"]) for row in robust),
        "audit_wall_seconds_from_reproduction_freeze_through_analysis": analysis_time.timestamp() - audit_start.timestamp(),
        "sum_worker_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in runtime_rows),
        "maximum_worker_elapsed_seconds_by_stage": {
            mode: max((float(row["elapsed_seconds"]) for row in runtime_rows if row["mode"] == mode), default=0.0)
            for mode in ("exact", "robust")
        },
        "resource_audit": json.loads((HERE / "resource_audit.json").read_text()),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    old = next(row for row in benchmark_rows if row["benchmark"] == "historical")
    fresh = next(row for row in benchmark_rows if row["benchmark"] == "fresh_unseen")
    combined = next(row for row in benchmark_rows if row["benchmark"] == "combined")
    report = [
        "# Strict-deadlock success-basin capacity audit",
        "",
        f"**Aggregate classification: {classification}.** Dominant remaining bottleneck: **{bottleneck}**.",
        "",
        "## Integrity and frozen semantics",
        "",
        "- All 17 authoritative Safety strict deadlocks reproduced exactly; maximum differences for state, Flow key, projected action, and monitor fields were all zero.",
        "- 102 complete augmented query states were restored: S0, 8 s, 4 s, 2 s, 1 s, and immediately pre-detector for every episode.",
        "- Eta remains fixed for the continuation; goal/safe/relative bases are recomputed at every physical step.",
        "- Continuation horizon is the remaining global frozen episode horizon through step 850, matching the G_phi oracle-label pipeline.",
        "- Search is confined to the authoritative Phase-A envelope goal=[0.5,1.25], safe=[-0.5,0.5], relative=[0,0.75].",
        "",
        "## Episode-level capacity",
        "",
        "| Cohort | Episodes | Any exact success | Robust >=63/64 | Early robust | Late robust | Fragile only | No basin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (old, fresh, combined):
        report.append(f"| {row['benchmark']} | {row['episodes']} | {row['episodes_with_any_successful_eta']} | {row['episodes_with_robust_B63_basin']} | {row['EARLY_ROBUSTLY_RESCUABLE']} | {row['LATE_ROBUSTLY_RESCUABLE']} | {row['FRAGILE_ONLY']} | {row['NO_BASIN_FOUND']} |")
    report += [
        "",
        f"Robust capacity before detector onset: **{robust_episode_count}/17 ({robust_episode_count/17:.1%})**.",
        f"Earliest robust-basin lead-time statistics (seconds): `{json.dumps(stats(onset_seconds), sort_keys=True)}`.",
        f"Robust candidate success-count statistics: `{json.dumps(stats(success_counts), sort_keys=True)}`.",
        f"Exact-flow successful-cell fraction across rescuable queried states: `{json.dumps(stats(exact_success_fractions), sort_keys=True)}`.",
        "",
        "## Projection",
        "",
        f"Projection is classified as **{'a major limiting factor' if projection_limit else 'not the primary limiting factor'}** by the predeclared descriptive comparison of rewrite fractions for exact successes versus non-successes.",
        f"Successful exact mean rewrite/raw ratio: {projection_lookup['exact_success']['projection_rewrite_fraction_of_raw_mean']}; non-success ratio: {projection_lookup['exact_non_success']['projection_rewrite_fraction_of_raw_mean']}.",
        "",
        "## Smallest next experiment",
        "",
        next_experiment,
    ]
    (HERE / "capacity_report.md").write_text("\n".join(report) + "\n")

    output_names = [
        "strict_deadlock_manifest.json", "reproduction_audit.json", "queried_states.csv", "eta_search_config.json",
        "eta_search_results.csv", "exact_flow_rescue.csv", "robust_basin_validation.csv", "best_eta_by_state.csv",
        "basin_onset_by_episode.csv", "episode_capacity_classification.csv", "projection_analysis.csv",
        "unresolved_cases.csv", "old_vs_fresh_summary.csv", "capacity_report.md", "runtime_statistics.json",
        "resource_audit.json", "prepare_audit.py", "run_capacity.py", "analyze_capacity.py",
    ]
    output_manifest = {
        "schema": "strict_deadlock_success_basin_capacity_outputs_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "classification": classification, "dominant_bottleneck": bottleneck,
        "single_smallest_next_experiment": next_experiment,
        "files_sha256": {name: sha(HERE / name) for name in output_names},
        "summary": {"historical": old, "fresh_unseen": fresh, "combined": combined},
    }
    output_manifest["content_sha256"] = canonical_hash(output_manifest)
    write_json(HERE / "manifest.json", output_manifest)
    print(json.dumps({"status": "PASS", "classification": classification, "bottleneck": bottleneck, "combined": combined, "onset_seconds": stats(onset_seconds), "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
