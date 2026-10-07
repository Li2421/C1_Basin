"""Synthesize frozen offline and closed-loop coverage-retraining results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
OLD_FRESH = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
OLD_WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_new(suite: str) -> dict[int, dict]:
    directory = HERE / f"runs/{suite}/raw/h8"
    rows = {}
    for path in sorted(directory.glob("episode_*.json")):
        row = json.loads(path.read_text())
        if row.get("record_complete") is not True or sha256(HERE / row["trajectory_file"]) != row["trajectory_sha256"]:
            raise RuntimeError((path, "invalid record/trajectory"))
        index = int(row["episode_index"])
        if index in rows:
            raise RuntimeError((suite, index, "duplicate"))
        rows[index] = row
    return rows


def aggregate_error(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    result = {}
    for cohort in ("historical11", "fresh6_heldout"):
        subset = [row for row in rows if row["cohort"] == cohort]
        value = {"states": len(subset)}
        for key in (
            "target_l2_mean", "executed_action_l2_mean",
            "correction_norm_error_mean", "cosine_similarity_mean",
        ):
            value[key] = float(np.mean([float(row[key]) for row in subset]))
        result[cohort] = value
    return result


def main() -> None:
    new_fresh = load_new("fresh200")
    new_historical = load_new("historical11")
    if len(new_fresh) != 200 or len(new_historical) != 11:
        raise RuntimeError(("incomplete closed-loop", len(new_fresh), len(new_historical)))
    selection = json.loads((HERE / "selected_checkpoint.json").read_text())
    checkpoint_hash = selection["checkpoint_sha256"]
    if {row["checkpoint_sha256"] for row in [*new_fresh.values(), *new_historical.values()]} != {checkpoint_hash}:
        raise RuntimeError("closed-loop checkpoint mismatch")

    old_fresh_rows = read_csv(OLD_FRESH / "per_episode_results.csv")
    safety = {int(row["episode_index"]): row for row in old_fresh_rows if row["condition"] == "Safety"}
    old_h8_fresh = {int(row["episode_index"]): row for row in old_fresh_rows if row["condition"] == "H=8"}
    if len(safety) != 200 or len(old_h8_fresh) != 200:
        raise RuntimeError("old fresh regression table incomplete")
    old_wide_rows = read_csv(OLD_WIDE / "per_episode_results.csv")
    old_h8_wide = {int(row["episode_index"]): row for row in old_wide_rows if row["condition"] == "H=8"}
    capacity_cases = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())["cases"]
    historical_cases = sorted(
        (row for row in capacity_cases if row["benchmark"] == "historical"),
        key=lambda row: int(row["episode_index"]),
    )
    fresh_cases = sorted(
        (row for row in capacity_cases if row["benchmark"] == "fresh_unseen"),
        key=lambda row: int(row["episode_index"]),
    )

    historical_rows = []
    for case in historical_cases:
        index = int(case["episode_index"])
        old = old_h8_wide[index]
        new = new_historical[index]
        historical_rows.append({
            "case_id": case["case_id"], "episode_index": index,
            "old_outcome": old["outcome"], "new_outcome": new["outcome"],
            "old_rescue": old["outcome"] == "success",
            "new_rescue": new["outcome"] == "success",
            "new_episode_steps": new["episode_steps"], "new_J_def": new["J_def"],
            "new_collision": new["collision"], "new_projection_failures": new["projection_failures"],
            "new_invalid_actions": new["invalid_actions"], "new_nan_inf_events": new["nan_inf_events"],
        })
    write_csv(HERE / "historical11_closed_loop.csv", historical_rows)

    fresh6_rows = []
    for case in fresh_cases:
        index = int(case["episode_index"])
        old = old_h8_fresh[index]
        new = new_fresh[index]
        fresh6_rows.append({
            "case_id": case["case_id"], "episode_index": index,
            "old_outcome": old["outcome"], "new_outcome": new["outcome"],
            "old_rescue": old["outcome"] == "success",
            "new_rescue": new["outcome"] == "success",
            "new_episode_steps": new["episode_steps"], "new_J_def": new["J_def"],
            "new_collision": new["collision"], "new_projection_failures": new["projection_failures"],
            "new_invalid_actions": new["invalid_actions"], "new_nan_inf_events": new["nan_inf_events"],
        })
    write_csv(HERE / "fresh6_closed_loop.csv", fresh6_rows)

    pair_counts = Counter()
    safety_outcomes = Counter()
    new_outcomes = Counter()
    old_outcomes = Counter()
    regression_rows = []
    for index in range(200):
        safe = safety[index]
        old = old_h8_fresh[index]
        new = new_fresh[index]
        safe_success = safe["outcome"] == "success"
        new_success = new["outcome"] == "success"
        cell = (
            "BOTH_SUCCESS" if safe_success and new_success else
            "BREAK" if safe_success and not new_success else
            "RESCUE" if not safe_success and new_success else "BOTH_FAIL"
        )
        pair_counts[cell] += 1
        safety_outcomes[safe["outcome"]] += 1
        old_outcomes[old["outcome"]] += 1
        new_outcomes[new["outcome"]] += 1
        regression_rows.append({
            "episode_index": index, "safety_outcome": safe["outcome"],
            "old_H8_outcome": old["outcome"], "new_H8_outcome": new["outcome"],
            "paired_cell_new_vs_safety": cell, "new_J_def": new["J_def"],
        })
    write_csv(HERE / "regression_wide_per_episode.csv", regression_rows)
    safety_timeouts = [index for index, row in safety.items() if row["outcome"] == "timeout"]
    safety_deadlocks = [index for index, row in safety.items() if row["outcome"] == "deadlock"]
    timeout_rescues = sum(new_fresh[index]["outcome"] == "success" for index in safety_timeouts)
    deadlock_rescues = sum(new_fresh[index]["outcome"] == "success" for index in safety_deadlocks)
    jdef = np.asarray([float(row["J_def"]) for row in new_fresh.values()])
    regression = {
        "cohort_status": "DEVELOPMENT_REGRESSION_SET_PREVIOUSLY_OBSERVED",
        "episodes": 200,
        "safety": dict(safety_outcomes),
        "old_H8": {**dict(old_outcomes), "Q": old_outcomes["success"] / 200},
        "new_H8": {**dict(new_outcomes), "Q": new_outcomes["success"] / 200},
        "paired_new_vs_safety": dict(pair_counts),
        "new_timeout_rescue": {
            "rescued": timeout_rescues, "total": len(safety_timeouts),
            "rate": timeout_rescues / len(safety_timeouts),
        },
        "new_strict_deadlock_rescue": {
            "rescued": deadlock_rescues, "total": len(safety_deadlocks),
            "rate": deadlock_rescues / len(safety_deadlocks),
        },
        "new_break_count": pair_counts["BREAK"],
        "new_collision_count": new_outcomes["collision"],
        "new_J_def": {
            "mean": float(jdef.mean()), "median": float(np.median(jdef)),
            "p95": float(np.quantile(jdef, .95)), "max": float(jdef.max()),
        },
    }
    write_json(HERE / "regression_wide_metrics.json", regression)

    all_new = [*new_fresh.values(), *new_historical.values()]
    safety_checks = {
        "passed": not any(
            row["collision"] or row["projection_failures"] or row["invalid_actions"]
            or row["nan_inf_events"] or row.get("execution_error")
            for row in all_new
        ),
        "evaluated_rollouts": len(all_new),
        "agent_collisions": sum(bool(row["agent_collision"]) for row in all_new),
        "wall_collisions": sum(bool(row["wall_collision"]) for row in all_new),
        "projection_failures": sum(int(row["projection_failures"]) for row in all_new),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in all_new),
        "nan_inf_events": sum(int(row["nan_inf_events"]) for row in all_new),
        "execution_errors": sum(row.get("execution_error") is not None for row in all_new),
    }
    write_json(HERE / "hard_safety_checks.json", safety_checks)

    historical_rescues = sum(row["new_rescue"] for row in historical_rows)
    fresh_rescues = sum(row["new_rescue"] for row in fresh6_rows)
    old_q = old_outcomes["success"] / 200
    new_q = new_outcomes["success"] / 200
    strict_rescues = historical_rescues + fresh_rescues
    if historical_rescues >= 6 and fresh_rescues >= 3 and new_q >= old_q - 0.03 and timeout_rescues >= 55:
        classification = "DATA_COVERAGE_HYPOTHESIS_SUPPORTED"
        dominant = "strict-deadlock supervision coverage was the dominant learned-deployment gap"
        next_step = "Freeze this checkpoint and H=8, then run one new untouched 200-episode WIDE Safety-vs-H8 confirmation cohort with no further tuning."
    elif historical_rescues >= 6 and fresh_rescues == 0:
        classification = "HISTORICAL_MEMORIZATION_ONLY"
        dominant = "coverage addition did not transfer to held-out strict-deadlock episodes"
        next_step = "Audit state-to-target geometry across the historical 11 and held-out 6 before adding any more data."
    elif strict_rescues >= 6 and (new_q < old_q - 0.03 or timeout_rescues < 55):
        classification = "DATA_ADDITION_CAUSES_NEGATIVE_TRANSFER"
        dominant = "strict-deadlock repair trades off against the established timeout-recovery behavior"
        next_step = "Run a validation-only mixture-ratio audit without changing architecture or loss."
    else:
        classification = "DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED"
        dominant = "deployment composition/temporal mismatch beyond pointwise onset-state approximation"
        next_step = "Without training or eta search, replay each frozen minimum-J_def robust eta on the 17 onset states using the deployment H=8 one-step cadence; compare with its known dense fixed-eta continuation."

    pre = aggregate_error(HERE / "pre_retrain_target_error.csv")
    post = aggregate_error(HERE / "post_retrain_target_error.csv")
    retention = read_csv(HERE / "startup_warm_retention.csv")
    old_test = {row["cohort"]: row for row in retention if row["model"] == "OLD" and row["split"] == "test"}
    new_test = {row["cohort"]: row for row in retention if row["model"] == "NEW" and row["split"] == "test"}
    report = f"""# Strict-deadlock supervision coverage retraining audit

## Result

**{classification}**

The original dataset contained no exact match to any of the historical 11
earliest robust-onset states.  Adding exactly one standard 64-variant cohort
per historical state reduced mean target L2 from
{pre['historical11']['target_l2_mean']:.6f} to
{post['historical11']['target_l2_mean']:.6f}; on the untouched Fresh-6 target
diagnostic it fell from {pre['fresh6_heldout']['target_l2_mean']:.6f} to
{post['fresh6_heldout']['target_l2_mean']:.6f}.

## Frozen checkpoint

- Seed {selection['selected_seed']}, epoch {selection['selected_epoch']}
- `{selection['checkpoint']}`
- SHA256 `{selection['checkpoint_sha256']}`
- Selection used startup/warm validation only.

## Offline retention

| Test cohort | Old mean L2 | New mean L2 |
|---|---:|---:|
| Startup | {float(old_test['STARTUP']['state_grouped_mean_l2']):.6f} | {float(new_test['STARTUP']['state_grouped_mean_l2']):.6f} |
| Warm V3 | {float(old_test['WARM_V3']['state_grouped_mean_l2']):.6f} | {float(new_test['WARM_V3']['state_grouped_mean_l2']):.6f} |

## Closed loop, frozen H=8

- Historical strict-deadlock rescue: old 0/11, new {historical_rescues}/11.
- Fresh held-out strict-deadlock rescue: old 0/6, new {fresh_rescues}/6.
- Development fresh-WIDE regression: old Q={old_q:.3f}, new Q={new_q:.3f}.
- Safety timeout rescue: {timeout_rescues}/{len(safety_timeouts)}.
- Breaks of Safety successes: {pair_counts['BREAK']}.
- Collisions: {new_outcomes['collision']}.
- Mean J_def: {regression['new_J_def']['mean']:.6f}.

Hard safety {'passed' if safety_checks['passed'] else 'did not pass'} across all
new rollouts.  The single smallest justified next experiment is: {next_step}
"""
    (HERE / "coverage_report.md").write_text(report)

    training_runtime = json.loads((HERE / "training_runtime.json").read_text())
    worker_runtime = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_closed_loop_shard*.json"))]
    def duration(row: dict) -> float:
        return (datetime.fromisoformat(row["finished_utc"]) - datetime.fromisoformat(row["started_utc"])).total_seconds()
    runtime = {
        "dataset_preparation_seconds": 6.8,
        "training": training_runtime,
        "closed_loop_workers": worker_runtime,
        "closed_loop_parallel_wall_seconds": max(duration(row) for row in worker_runtime),
        "total_compute_pipeline_wall_seconds": (
            6.8 + training_runtime["parallel_training_wall_seconds"]
            + training_runtime["aggregation_offline_seconds"]
            + max(duration(row) for row in worker_runtime)
        ),
        "resource_usage": {
            "training_gpu_shards": 3, "closed_loop_gpu_shards": 2,
            "maximum_gpu_shards_concurrent": 3,
            "training_cpu_cores": 6, "closed_loop_cpu_cores": 6,
            "maximum_allocated_memory_gb": 48,
        },
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    outputs = [
        "augmentation_manifest.json", "overlap_audit.json", "pre_retrain_target_error.csv",
        "augmented_dataset_manifest.json", "training_report.md", "checkpoint_pareto.csv",
        "selected_checkpoint.json", "post_retrain_target_error.csv",
        "historical11_closed_loop.csv", "fresh6_closed_loop.csv",
        "regression_wide_metrics.json", "hard_safety_checks.json", "runtime_statistics.json",
        "coverage_report.md", "startup_warm_retention.csv",
    ]
    write_json(HERE / "manifest.json", {
        "experiment": "gphi_strict_deadlock_coverage_retrain_v1",
        "status": "COMPLETE",
        "classification": classification,
        "dominant_remaining_problem": dominant,
        "smallest_next_experiment": next_step,
        "checkpoint_sha256": checkpoint_hash,
        "fresh6_training_contamination": False,
        "generated_files_sha256": {name: sha256(HERE / name) for name in outputs},
    })
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "historical_rescues": historical_rescues, "fresh6_rescues": fresh_rescues,
        "old_Q": old_q, "new_Q": new_q, "timeout_rescues": timeout_rescues,
        "breaks": pair_counts["BREAK"], "collisions": new_outcomes["collision"],
    }, indent=2))


if __name__ == "__main__":
    main()
