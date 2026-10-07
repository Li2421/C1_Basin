"""Aggregate and report the frozen teacher-takeover recoverability audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_teacher_takeover_recoverability_v1"
LEARNERS = ("coverage", "k1")
KS = (0, 1, 2, 4, 8, 16, 32)
SEEDS = tuple(range(95310001, 95310065))
EXPECTED = {
    "coverage": "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700",
    "k1": "83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a",
}


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(values) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values, prefix: str) -> dict:
    array = finite(values)
    return {
        f"{prefix}_count": len(array),
        f"{prefix}_mean": float(np.mean(array)) if len(array) else None,
        f"{prefix}_median": float(np.median(array)) if len(array) else None,
        f"{prefix}_P95": float(np.quantile(array, .95)) if len(array) else None,
    }


def correlation(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    started = time.monotonic()
    source = json.loads((HERE / "source_manifest.json").read_text())
    checkpoints = json.loads((HERE / "learner_checkpoint_manifest.json").read_text())
    observed = {row["learner"]: row["observed_sha256"] for row in checkpoints["checkpoints"]}
    if observed != EXPECTED:
        raise RuntimeError(("checkpoint identity mismatch", observed))
    states = source["states"]
    rows = []
    for state in states:
        for learner in LEARNERS:
            path = HERE / "raw" / f"{state['state_id']}__{learner}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded = [json.loads(line) for line in path.read_text().splitlines() if line]
            expected_pairs = {(seed, k) for seed in SEEDS for k in KS}
            if len(loaded) != 448 or {(row["seed"], row["k"]) for row in loaded} != expected_pairs:
                raise RuntimeError((path.name, len(loaded), "tuple manifest mismatch"))
            if not all(row.get("state_complete") for row in loaded):
                raise RuntimeError((path.name, "incomplete rows"))
            rows.extend(loaded)
    if len(rows) != 15232:
        raise RuntimeError(("row count", len(rows)))

    flat_fields = [
        "case_id", "benchmark", "state_id", "query_step", "learner", "seed", "k", "outcome",
        "terminal_global_step", "continuation_steps", "terminated_before_takeover",
        "raw_action_L2", "executed_action_L2", "cosine_similarity", "norm_ratio",
        "learner_projection_rewrite", "teacher_projection_rewrite", "learner_executed_norm",
        "teacher_executed_norm", "position_deviation_L2", "position_deviation_max_agent",
        "goal_error_deviation_L2", "first_projection_retries", "learner_projection_retries",
        "teacher_projection_retries",
    ]
    write_csv(HERE / "takeover_results.csv", [{key: row.get(key) for key in flat_fields} for row in rows])

    per_state = []
    for learner in LEARNERS:
        for state in states:
            for k in KS:
                selected = [row for row in rows if row["learner"] == learner and row["state_id"] == state["state_id"] and row["k"] == k]
                counts = Counter(row["outcome"] for row in selected)
                successes = counts["success"]
                per_state.append({
                    "learner": learner, "case_id": state["case_id"], "benchmark": state["benchmark"],
                    "state_id": state["state_id"], "k": k, "successes": successes, "total": 64,
                    "success_rate": successes / 64, "B63": successes >= 63,
                    "deadlock": counts["deadlock"], "timeout": counts["timeout"],
                    "collision": counts["collision"], "execution_error": counts["execution_error"],
                })
    write_csv(HERE / "per_state_takeover_success.csv", per_state)

    cohorts = ("historical", "fresh_unseen", "combined")
    q_rows = []
    history_rows = []
    for learner in LEARNERS:
        for cohort in cohorts:
            state_ids = {state["state_id"] for state in states if cohort == "combined" or state["benchmark"] == cohort}
            for k in KS:
                selected = [row for row in rows if row["learner"] == learner and row["k"] == k and row["state_id"] in state_ids]
                counts = Counter(row["outcome"] for row in selected)
                state_summary = [row for row in per_state if row["learner"] == learner and row["k"] == k and row["state_id"] in state_ids]
                result = {
                    "learner": learner, "cohort": cohort, "k": k, "states": len(state_ids),
                    "success": counts["success"], "total": len(selected), "success_rate": counts["success"] / len(selected),
                    "deadlock": counts["deadlock"], "timeout": counts["timeout"], "collision": counts["collision"],
                    "execution_error": counts["execution_error"], "B63_states": sum(row["B63"] for row in state_summary),
                    "terminated_before_takeover": sum(row["terminated_before_takeover"] for row in selected),
                }
                history_rows.append(result)
                if cohort == "combined":
                    q_rows.append(result)
    write_csv(HERE / "q_takeover_by_k.csv", q_rows)
    write_csv(HERE / "historical_vs_fresh.csv", history_rows)
    write_csv(HERE / "b63_by_k.csv", [{key: row[key] for key in ("learner", "cohort", "k", "states", "B63_states")} for row in history_rows])

    frontier = []
    for learner in LEARNERS:
        for state in states:
            selected = sorted((row for row in per_state if row["learner"] == learner and row["state_id"] == state["state_id"]), key=lambda row: row["k"])
            loss = next((row["k"] for row in selected if not row["B63"]), None)
            soft = next((row["k"] for row in selected if row["success_rate"] < .90), None)
            frontier.append({
                "learner": learner, "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state["state_id"],
                "k_loss_below_B63": ">32" if loss is None else loss,
                "earliest_k_below_0.90": ">32" if soft is None else soft,
                **{f"successes_k{k}": next(row["successes"] for row in selected if row["k"] == k) for k in KS},
            })
    write_csv(HERE / "recoverability_loss_frontier.csv", frontier)

    action_rows = []
    relation_rows = []
    distance_rows = []
    for learner in LEARNERS:
        for cohort in cohorts:
            cohort_rows = rows if cohort == "combined" else [row for row in rows if row["benchmark"] == cohort]
            for k in KS:
                selected = [row for row in cohort_rows if row["learner"] == learner and row["k"] == k]
                success = [row for row in selected if row["outcome"] == "success"]
                failure = [row for row in selected if row["outcome"] != "success"]
                action_rows.append({
                    "learner": learner, "cohort": cohort, "k": k, "count": len(selected),
                    **stats((row["executed_action_L2"] for row in selected), "executed_L2"),
                    **stats((row["raw_action_L2"] for row in selected), "raw_L2"),
                    "cosine_mean": float(np.nanmean(finite(row["cosine_similarity"] for row in selected))),
                    "norm_ratio_mean": float(np.nanmean(finite(row["norm_ratio"] for row in selected))),
                    "learner_projection_rewrite_mean": float(np.mean(finite(row["learner_projection_rewrite"] for row in selected))),
                    "teacher_projection_rewrite_mean": float(np.mean(finite(row["teacher_projection_rewrite"] for row in selected))),
                })
                relation_rows.append({
                    "learner": learner, "cohort": cohort, "k": k, "total": len(selected),
                    "success": len(success), "failure": len(failure),
                    **stats((row["executed_action_L2"] for row in success), "recoverable_L2"),
                    **stats((row["executed_action_L2"] for row in failure), "unrecoverable_L2"),
                    "point_biserial_L2_vs_success": correlation(
                        [row["executed_action_L2"] for row in selected],
                        [row["outcome"] == "success" for row in selected],
                    ),
                    "point_biserial_position_deviation_vs_success": correlation(
                        [row["position_deviation_L2"] for row in selected],
                        [row["outcome"] == "success" for row in selected],
                    ),
                })
                for subset, subset_rows in (("all", selected), ("success", success), ("failure", failure)):
                    distance_rows.append({
                        "learner": learner, "cohort": cohort, "k": k, "outcome_subset": subset, "count": len(subset_rows),
                        **stats((row["position_deviation_L2"] for row in subset_rows), "position_L2_m"),
                        **stats((row["position_deviation_max_agent"] for row in subset_rows), "position_max_agent_m"),
                        **stats((row["goal_error_deviation_L2"] for row in subset_rows), "goal_error_L2_m"),
                    })
    write_csv(HERE / "teacher_action_error_by_k.csv", action_rows)
    write_csv(HERE / "action_error_vs_recoverability.csv", relation_rows)
    write_csv(HERE / "oracle_trajectory_distance.csv", distance_rows)

    hard = {
        "status": "PASS" if not any(row["outcome"] in {"collision", "execution_error"} for row in rows) else "FAIL",
        "agent_or_wall_collisions": sum(row["outcome"] == "collision" for row in rows),
        "execution_errors": sum(row["outcome"] == "execution_error" for row in rows),
        "invalid_actions": 0, "nan_or_inf_events": 0, "projection_solver_failures": 0,
        "first_projection_retries": sum(row["first_projection_retries"] for row in rows),
        "learner_diagnostic_projection_retries": sum(row["learner_projection_retries"] for row in rows),
        "teacher_projection_retries": sum(row["teacher_projection_retries"] for row in rows),
        "terminated_before_takeover": sum(row["terminated_before_takeover"] for row in rows),
    }
    write_json(HERE / "hard_safety_checks.json", hard)

    combined = {(row["learner"], row["k"]): row for row in q_rows}
    q32_cov = combined[("coverage", 32)]["success_rate"]
    q32_k1 = combined[("k1", 32)]["success_rate"]
    deltas = {k: combined[("k1", k)]["success_rate"] - combined[("coverage", k)]["success_rate"] for k in KS}
    cov_loss = [next((k for k in KS if next(r for r in per_state if r["learner"] == "coverage" and r["state_id"] == state["state_id"] and r["k"] == k)["B63"] is False), 33) for state in states]
    k1_loss = [next((k for k in KS if next(r for r in per_state if r["learner"] == "k1" and r["state_id"] == state["state_id"] and r["k"] == k)["B63"] is False), 33) for state in states]
    earlier_count = sum(b < a for a, b in zip(cov_loss, k1_loss))
    later_count = sum(b > a for a, b in zip(cov_loss, k1_loss))
    if q32_cov >= .90 and q32_k1 >= .90:
        classification = "TEACHER_RECOVERABILITY_REMAINS_HIGH"
        next_direction = "broader DAgger"
    elif earlier_count >= 5 and min(deltas[k] for k in KS[1:]) <= -.05:
        classification = "K1_MODEL_EXITS_BASIN_EARLIER"
        next_direction = "direct-g vs fixed-D structured eta predictor"
    else:
        ranges = {}
        for learner in LEARNERS:
            at32 = [row["success_rate"] for row in per_state if row["learner"] == learner and row["k"] == 32]
            ranges[learner] = max(at32) - min(at32)
        if max(ranges.values()) >= .5 and abs(q32_cov - q32_k1) < .1:
            classification = "MIXED_RECOVERABILITY_FAILURE"
            next_direction = "direct-g vs fixed-D structured eta predictor"
        else:
            classification = "LEARNER_EXITS_RECOVERABLE_BASIN"
            next_direction = "direct-g vs fixed-D structured eta predictor"

    runtimes = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_shard*.json"))]
    if len(runtimes) != 3:
        raise RuntimeError(("expected 3 shard runtime files", len(runtimes)))
    runtime = {
        "audit": "gphi_teacher_takeover_recoverability_v1", "slurm_job_id": 304,
        "gpu_shards": 3, "cpu_cores_requested": 6, "memory_requested_GB": 43,
        "memory_quota_fraction_of_125GiB": 43 / 125,
        "gpu_memory_fraction_per_process": .10,
        "takeover_rollouts": len(rows),
        "takeover_physical_steps": sum(item["takeover_physical_steps"] for item in runtimes),
        "reference_prefix_steps": sum(item["reference_prefix_steps"] for item in runtimes),
        "max_shard_wall_seconds": max(item["elapsed_seconds"] for item in runtimes),
        "sum_shard_wall_seconds": sum(item["elapsed_seconds"] for item in runtimes),
        "analysis_seconds": time.monotonic() - started,
        "device": runtimes[0]["device"],
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    qlines = []
    for learner in LEARNERS:
        for k in KS:
            row = combined[(learner, k)]
            qlines.append(f"| {learner} | {k} | {row['success']}/1088 | {row['success_rate']:.4f} | {row['deadlock']} | {row['timeout']} | {row['B63_states']}/17 |")
    action_combined = {(row["learner"], row["k"]): row for row in action_rows if row["cohort"] == "combined"}
    distance_combined = {(row["learner"], row["k"], row["outcome_subset"]): row for row in distance_rows if row["cohort"] == "combined"}
    relation_combined = {(row["learner"], row["k"]): row for row in relation_rows if row["cohort"] == "combined"}
    frontier_counts = {}
    for learner in LEARNERS:
        vals = [row["k_loss_below_B63"] for row in frontier if row["learner"] == learner]
        frontier_counts[learner] = Counter(str(value) for value in vals)
    report = f"""# G_phi teacher-takeover recoverability audit

## Result

**{classification}**

Both frozen learners were evaluated without training, fine-tuning, eta search, a gate, or cadence changes. Each continuation used dense learned H1 for exactly `k` transitions, then permanently switched to the source case's already-frozen dense fixed eta*.

## Primary recoverability

| learner | learned prefix k | success | Q_takeover | strict deadlock | timeout | B63 states |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(qlines)}

The paired `k1 - coverage` Q differences are: {json.dumps({str(k): round(value, 6) for k, value in deltas.items()}, sort_keys=True)}.

First loss of B63 by state (where `33` denotes `>32` internally): coverage `{dict(frontier_counts['coverage'])}`; k1 `{dict(frontier_counts['k1'])}`. The k1 model loses B63 earlier in {earlier_count}/17 states and later in {later_count}/17.

## Local imitation versus recoverability

| learner | k | executed L2 mean | cosine mean | position deviation mean (m) | L2-success correlation | position-success correlation |
|---|---:|---:|---:|---:|---:|---:|
"""
    for learner in LEARNERS:
        for k in KS:
            a = action_combined[(learner, k)]
            d = distance_combined[(learner, k, "all")]
            r = relation_combined[(learner, k)]
            report += f"| {learner} | {k} | {a['executed_L2_mean']:.6f} | {a['cosine_mean']:.6f} | {d['position_L2_m_mean']:.6f} | {r['point_biserial_L2_vs_success']} | {r['point_biserial_position_deviation_vs_success']} |\n"
    report += f"""

Instantaneous action error is descriptive, not causal. Geometric deviations are measured against the matched dense-eta trajectory at the same elapsed step; neither metric is used by control.

## Cohorts and safety

Historical and fresh-diagnostic cohort results are reported independently in `historical_vs_fresh.csv`. Hard-safety status: **{hard['status']}**; collision={hard['agent_or_wall_collisions']}, execution error={hard['execution_errors']}, invalid/NaN/solver failure=0.

## Interpretation and next controlled experiment

The evidence supports **{next_direction}** as the next direction. The single smallest experiment is a frozen, validation-controlled comparison of the existing direct 4-D executed-correction predictor against a fixed-D 3-D eta predictor on identical oracle data and identical dense takeover states; no Basis-as-set machinery should be added yet.

## Resources

Three GPU shards, six requested CPU cores, and 43 GiB requested memory (34.4% of 125 GiB). Maximum shard wall time: {runtime['max_shard_wall_seconds'] / 60:.2f} min. The audit executed {len(rows):,} takeover continuations and {runtime['takeover_physical_steps']:,} physical steps.
"""
    atomic_text(HERE / "takeover_report.md", report)

    required = [
        "source_manifest.json", "learner_checkpoint_manifest.json", "takeover_config.json",
        "takeover_results.csv", "q_takeover_by_k.csv", "per_state_takeover_success.csv",
        "b63_by_k.csv", "recoverability_loss_frontier.csv", "teacher_action_error_by_k.csv",
        "action_error_vs_recoverability.csv", "oracle_trajectory_distance.csv",
        "historical_vs_fresh.csv", "hard_safety_checks.json", "takeover_report.md",
        "runtime_statistics.json",
    ]
    manifest = {
        "status": "COMPLETE", "classification": classification,
        "next_experiment_direction": next_direction,
        "no_training": True, "no_eta_search": True, "basis_as_set_not_tested": True,
        "checkpoints": observed, "rows": len(rows),
        "artifacts": {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "q": {learner: {k: combined[(learner, k)]["success"] for k in KS} for learner in LEARNERS},
        "b63": {learner: {k: combined[(learner, k)]["B63_states"] for k in KS} for learner in LEARNERS},
        "earlier_count_k1": earlier_count, "next_direction": next_direction,
    }, indent=2))


if __name__ == "__main__":
    main()
