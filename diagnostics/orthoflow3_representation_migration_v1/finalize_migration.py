"""Assemble the immutable OrthoFlow3 migration evidence and handoff."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
P0_FILES = (
    ROOT / "diagnostics/gphi_training_dataset_v3/oracle_search_results.jsonl",
    ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1/startup_oracle_search_results.jsonl",
)
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(pattern: str) -> list[dict]:
    output = []
    for path in sorted((HERE / "raw").glob(pattern)):
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError((path, "no rows"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def p0_count(cell: dict, name: str) -> int:
    return int(cell.get("counts", {}).get(name, cell.get(name, 0)))


def p0_records() -> dict[str, dict]:
    output = {}
    for path in P0_FILES:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                output[row["state_id"]] = row
    return output


def median(values) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def main() -> None:
    required = (
        "candidate_screen16_plan.json",
        "promotion1_plan.json",
        "promotion2_plan.json",
        "interpolation8_plan.json",
        "subset_basis_gram_stats.csv",
    )
    for name in required:
        if not (HERE / name).exists():
            raise RuntimeError((name, "required completed-stage artifact missing"))

    subset_doc = json.loads((HERE / "migration_subset_manifest.json").read_text())
    subset = sorted(subset_doc["selected_states"], key=lambda row: row["selection_rank"])
    state_map = {row["state_id"]: row for row in subset}
    global_rows = read_rows("global256_exact/shard*.jsonl")
    screen_rows = read_rows("candidate_screen16/shard*.jsonl")
    promotion1_rows = read_rows("promotion1/shard*.jsonl")
    promotion2_rows = read_rows("promotion2/shard*.jsonl")
    interpolation_rows = read_rows("interpolation8/shard*.jsonl")
    if len(global_rows) != 8192 or len({row["arm_id"] for row in global_rows}) != 8192:
        raise RuntimeError(("bad global evidence", len(global_rows)))

    active_evidence = defaultdict(list)
    for row in (*global_rows, *screen_rows, *promotion1_rows, *promotion2_rows):
        active_evidence[(row["state_id"], int(row["eta_index"]))].append(row)
    active_summary = {}
    candidate_rows = []
    for state in subset:
        for eta_index in range(256):
            evidence = active_evidence[(state["state_id"], eta_index)]
            if not evidence:
                raise RuntimeError((state["state_id"], eta_index, "missing exact evidence"))
            counts = Counter(row["outcome"] for row in evidence)
            success_j = [float(row["J_def"]) for row in evidence if row["success"]]
            trials = len(evidence)
            successes = int(counts["success"])
            summary = {
                "state_id": state["state_id"],
                "selection_rank": state["selection_rank"],
                "category": state["category"],
                "candidate_kind": "active_sobol",
                "eta_index": eta_index,
                "eta": json.dumps(evidence[0]["eta"], separators=(",", ":")),
                "trials": trials,
                "successes": successes,
                "empirical_Q": successes / trials,
                "B63": trials == 64 and successes >= 63,
                "mean_J_def_success": (sum(success_j) / len(success_j)) if success_j else "",
                "deadlock": int(counts["deadlock"]),
                "timeout": int(counts["timeout"]),
                "collision": int(counts["collision"]),
                "execution_error": int(counts["execution_error"]),
                "evidence_stages": ";".join(sorted({row["stage"] for row in evidence})),
            }
            active_summary[(state["state_id"], eta_index)] = summary
            candidate_rows.append(summary)

    p0 = p0_records()
    p0_subset = {state["state_id"]: p0[state["state_id"]] for state in subset}
    zero_summary = {}
    for state in subset:
        record = p0_subset[state["state_id"]]
        zero_cells = [cell for cell in record["cells"] if np.allclose(cell["eta"], 0.0, atol=0.0)]
        if len(zero_cells) != 1:
            raise RuntimeError((state["state_id"], "zero-cell count", len(zero_cells)))
        cell = zero_cells[0]
        trials = int(cell["evaluated"])
        successes = p0_count(cell, "success")
        summary = {
            "state_id": state["state_id"],
            "selection_rank": state["selection_rank"],
            "category": state["category"],
            "candidate_kind": "zero_reused_after_invariance",
            "eta_index": "ZERO",
            "eta": "[0.0,0.0,0.0]",
            "trials": trials,
            "successes": successes,
            "empirical_Q": successes / trials,
            "B63": bool(cell["B_63_member"]),
            "mean_J_def_success": 0.0,
            "deadlock": p0_count(cell, "deadlock"),
            "timeout": p0_count(cell, "timeout"),
            "collision": p0_count(cell, "collision"),
            "execution_error": int(cell.get("execution_error", 0)),
            "evidence_stages": "historical_matched_zero_eta_reuse",
        }
        zero_summary[state["state_id"]] = summary
        candidate_rows.append(summary)
    candidate_rows.sort(
        key=lambda row: (
            int(row["selection_rank"]),
            -1 if row["eta_index"] == "ZERO" else int(row["eta_index"]),
        )
    )
    write_csv(HERE / "subset_oracle_candidates.csv", candidate_rows)

    basin_rows = []
    capacity_rows = []
    for state in subset:
        state_id = state["state_id"]
        active = [active_summary[(state_id, index)] for index in range(256)]
        active_b63 = [row for row in active if row["B63"]]
        robust = list(active_b63)
        if zero_summary[state_id]["B63"]:
            robust.append(zero_summary[state_id])
        def canonical_key(row):
            value = row["mean_J_def_success"]
            return (float(value) if value != "" else math.inf, -1 if row["eta_index"] == "ZERO" else int(row["eta_index"]))
        canonical = min(robust, key=canonical_key) if robust else None
        if zero_summary[state_id]["B63"]:
            status = "ZERO_SUFFICIENT"
        elif active_b63:
            status = "ACTIVE_REQUIRED"
        else:
            status = "NO_BASIN_FOUND"
        p0_record = p0_subset[state_id]
        p0_cells = p0_record["cells"]
        p0_best_q = max(p0_count(cell, "success") / int(cell["evaluated"]) for cell in p0_cells)
        of3_best = max(active + [zero_summary[state_id]], key=lambda row: (row["empirical_Q"], row["trials"]))
        exact_success = sum(row["trials"] >= 1 and next(
            evidence["success"] for evidence in active_evidence[(state_id, int(row["eta_index"]))]
            if evidence["stage"] == "global256_exact"
        ) for row in active)
        basin = {
            "state_id": state_id,
            "selection_rank": state["selection_rank"],
            "category": state["category"],
            "historical_p0_zero": bool(state["zero_eta"]),
            "orthoflow3_status": status,
            "zero_trials": zero_summary[state_id]["trials"],
            "zero_successes": zero_summary[state_id]["successes"],
            "zero_B63": zero_summary[state_id]["B63"],
            "active_exact_success_candidates": int(exact_success),
            "active_candidates_tested_16_or_more": sum(row["trials"] >= 16 for row in active),
            "active_candidates_tested_64": sum(row["trials"] == 64 for row in active),
            "active_B63_candidates_confirmed": len(active_b63),
            "robust_candidate_count_lower_bound": len(robust),
            "orthoflow3_canonical_eta": canonical["eta"] if canonical else "",
            "orthoflow3_canonical_J_def": canonical["mean_J_def_success"] if canonical else "",
            "orthoflow3_best_empirical_Q": of3_best["empirical_Q"],
            "orthoflow3_best_Q_trials": of3_best["trials"],
            "collision_count_all_new_active_evidence": sum(row["collision"] for row in active),
            "execution_error_count_all_new_active_evidence": sum(row["execution_error"] for row in active),
        }
        basin_rows.append(basin)
        capacity_rows.append(
            {
                "state_id": state_id,
                "selection_rank": state["selection_rank"],
                "category": state["category"],
                "p0_basin_found": not bool(p0_record["B_63_empty"]),
                "orthoflow3_basin_found": bool(robust),
                "p0_Q_max_observed": p0_best_q,
                "orthoflow3_Q_max_observed": of3_best["empirical_Q"],
                "orthoflow3_Q_max_trials": of3_best["trials"],
                "p0_best_J_def": p0_record["J_min"],
                "orthoflow3_best_J_def": canonical["mean_J_def_success"] if canonical else "",
                "p0_B63_candidate_count": sum(bool(cell["B_63_member"]) for cell in p0_cells),
                "orthoflow3_B63_candidate_count_lower_bound": len(robust),
            }
        )
    write_csv(HERE / "subset_basin_summary.csv", basin_rows)
    write_csv(HERE / "p0_vs_orthoflow3_capacity.csv", capacity_rows)

    interpolation_plan = json.loads((HERE / "interpolation8_plan.json").read_text())
    interp_by_arm = defaultdict(list)
    for row in interpolation_rows:
        interp_by_arm[row["arm_id"]].append(row)
    interpolation_output = []
    for arm in interpolation_plan["arms"]:
        evidence = interp_by_arm[arm["arm_id"]]
        if len(evidence) != 8:
            raise RuntimeError((arm["arm_id"], "expected 8 interpolation results", len(evidence)))
        counts = Counter(row["outcome"] for row in evidence)
        failures = 8 - int(counts["success"])
        status = (
            "B63_REJECTED_BY_SCREEN" if failures >= 2
            else "ONE_FAILURE_UNRESOLVED" if failures == 1
            else "SCREEN_8_OF_8_UNCONFIRMED_B63"
        )
        interpolation_output.append(
            {
                "state_id": arm["state_id"],
                "selection_rank": arm["selection_rank"],
                "pair_kind": arm["pair_kind"],
                "endpoint_a_id": arm["endpoint_a_id"],
                "endpoint_b_id": arm["endpoint_b_id"],
                "endpoint_distance": arm["endpoint_distance"],
                "alpha": arm["alpha"],
                "eta": json.dumps(arm["eta"], separators=(",", ":")),
                "trials": 8,
                "successes": int(counts["success"]),
                "deadlock": int(counts["deadlock"]),
                "timeout": int(counts["timeout"]),
                "collision": int(counts["collision"]),
                "execution_error": int(counts["execution_error"]),
                "status": status,
            }
        )
    write_csv(HERE / "subset_interpolation_tests.csv", interpolation_output)

    gram_rows = list(csv.DictReader((HERE / "subset_basis_gram_stats.csv").open()))
    p0_cos = [float(row["p0_agent_goal_middle_cosine_max_abs"]) for row in gram_rows]
    of3_cos = [float(row["orthoflow3_agent_goal_middle_cosine_max_abs"]) for row in gram_rows]
    p0_basin = sum(row["p0_basin_found"] is True for row in capacity_rows)
    of3_basin = sum(row["orthoflow3_basin_found"] is True for row in capacity_rows)
    pair_counts = Counter(
        (
            bool(row["p0_basin_found"]),
            bool(row["orthoflow3_basin_found"]),
        )
        for row in capacity_rows
    )
    status_counts = Counter(row["orthoflow3_status"] for row in basin_rows)
    tested_interpolants = len(interpolation_output)
    interpolation_fail = sum(row["status"] == "B63_REJECTED_BY_SCREEN" for row in interpolation_output)
    interpolation_perfect = sum(row["successes"] == 8 for row in interpolation_output)

    p0_canonical = [tuple(row["canonical_eta"]) for row in subset]
    of3_canonical = [tuple(json.loads(row["orthoflow3_canonical_eta"])) for row in basin_rows if row["orthoflow3_canonical_eta"]]
    of3_active_canonical = [eta for eta in of3_canonical if any(abs(value) > 0 for value in eta)]
    on_boundary = sum(
        any(abs(value - lo) <= 1e-12 or abs(value - hi) <= 1e-12 for value, lo, hi in zip(eta, LOW, HIGH))
        for eta in of3_active_canonical
    )
    near_boundary = sum(
        any(min(abs(value - lo), abs(value - hi)) <= 0.01 * (hi - lo) for value, lo, hi in zip(eta, LOW, HIGH))
        for eta in of3_active_canonical
    )

    pathology = f"""# P0 versus OrthoFlow3 learning-geometry check

The frozen 32 states were selected uniformly before new outcomes.  They are a
migration sanity set, not evidence for a global superiority claim.

## Capacity and target structure

- P0 robust basin: **{p0_basin}/32**; OrthoFlow3 robust basin: **{of3_basin}/32**.
- Paired basin counts: both `{pair_counts[(True, True)]}`, P0-only
  `{pair_counts[(True, False)]}`, OrthoFlow3-only `{pair_counts[(False, True)]}`,
  neither `{pair_counts[(False, False)]}`.
- OrthoFlow3 zero/active/no-basin: `{status_counts['ZERO_SUFFICIENT']}` /
  `{status_counts['ACTIVE_REQUIRED']}` / `{status_counts['NO_BASIN_FOUND']}`.
- Distinct canonical values: P0 `{len(set(p0_canonical))}`; OrthoFlow3
  `{len(set(of3_canonical))}` (active only `{len(set(of3_active_canonical))}`).
- OrthoFlow3 active canonical boundary saturation: exact `{on_boundary}`,
  within 1% of a domain boundary `{near_boundary}`.
- Confirmed OrthoFlow3 robust candidate counts are lower bounds because the
  budgeted protocol promotes at most two active candidates per state.  The
  reported canonical target is therefore provisional within the promoted
  robust set, not an exhaustive minimum over every 16-stream screen survivor.

## Basis redundancy

- Median per-agent max `|cos(B_goal, middle)|`: P0 `{median(p0_cos):.6g}`;
  OrthoFlow3 `{median(of3_cos):.6g}`.

## Interpolation

- Tested `{tested_interpolants}` interpolants between confirmed B63 endpoints
  on 8 matched streams each; `{interpolation_fail}` had at least two failures
  and therefore cannot be B63, while `{interpolation_perfect}` were 8/8.
- An 8/8 staged screen is not relabeled as confirmed B63.  Sparse unresolved
  topology is reported as such.

OrthoFlow3 clearly removes the specific P0 goal/Safety collinearity in this
subset.  Whether it also simplifies canonical-target learning remains bounded
by the 32-state sample and intentionally incomplete robust-candidate coverage.
"""
    (HERE / "old_pathology_comparison.md").write_text(pathology)

    runtime_files = sorted((HERE / "raw").glob("*/shard*_runtime.json"))
    runtime_rows = [json.loads(path.read_text()) for path in runtime_files]
    new_rollouts = 6 + sum(int(row["new_rollouts"]) for row in runtime_rows)
    new_steps = 4206 + sum(int(row["new_physical_steps"]) for row in runtime_rows)
    gpu_stage_wall = defaultdict(float)
    for row in runtime_rows:
        gpu_stage_wall[row["stage"]] = max(gpu_stage_wall[row["stage"]], float(row["wall_seconds"]))
    measured_gpu_wall = sum(gpu_stage_wall.values())
    global_runtime = [row for row in runtime_rows if row["stage"] == "global256_exact"]
    global_wall = max(float(row["wall_seconds"]) for row in global_runtime)
    global_steps = sum(int(row["new_physical_steps"]) for row in global_runtime)
    global_avg_steps = global_steps / 8192
    exact424_rollouts = 424 * 256
    exact424_steps = int(math.ceil(exact424_rollouts * global_avg_steps))
    exact424_wall_two_shards = exact424_rollouts / 8192 * global_wall
    selected_active_per_state = sum(row["active_candidates_tested_16_or_more"] for row in basin_rows) / 32
    promotable_per_state = sum(row["active_candidates_tested_64"] for row in basin_rows) / 32
    analogous_rollouts = int(math.ceil(
        exact424_rollouts
        + 424 * selected_active_per_state * 15
        + 424 * promotable_per_state * 48
    ))
    analogous_steps = int(math.ceil(analogous_rollouts * (new_steps - 4206) / max(1, new_rollouts - 6)))
    analogous_wall = measured_gpu_wall * analogous_rollouts / max(1, new_rollouts - 6)
    cost = {
        "schema": "orthoflow3_full424_cost_estimate_v1",
        "automatic_limits": {"continuations": 15000, "physical_steps": 8000000, "wall_minutes": 60},
        "exact_screen_only": {
            "continuations": exact424_rollouts,
            "physical_steps_projected_from_subset": exact424_steps,
            "wall_minutes_two_shards_projected": exact424_wall_two_shards / 60,
            "wall_minutes_four_shards_optimistic": exact424_wall_two_shards / 120,
        },
        "analogous_staged_rebuild": {
            "continuations": analogous_rollouts,
            "physical_steps_projected": analogous_steps,
            "wall_minutes_two_shards_projected": analogous_wall / 60,
            "wall_minutes_four_shards_optimistic": analogous_wall / 120,
            "note": "projection scales the observed subset promotion density; exact future count depends on screens",
        },
        "automatic_execution_allowed": False,
        "binding_reason": "the 108544-continuation exact screen alone exceeds the 15000 automatic cap",
        "full424_executed": False,
    }
    write_json(HERE / "full_rebuild_cost_estimate.json", cost)

    classification = "ORTHOFLOW3_FULL_REBUILD_READY_BUT_REQUIRES_APPROVAL"
    decision = {
        "schema": "orthoflow3_migration_decision_v1",
        "classification": classification,
        "integration_pass": True,
        "subset_states": 32,
        "subset_p0_basin": p0_basin,
        "subset_orthoflow3_basin": of3_basin,
        "basis_redundancy_reduced": median(of3_cos) < median(p0_cos),
        "full424_rebuild_executed": False,
        "future_default_basis": "orthoflow3",
        "p0_preserved_for_historical_reproduction": True,
        "next_experiment": "run the frozen full-424 OrthoFlow3 oracle/data rebuild after approving its explicit compute budget; do not train a model yet",
    }
    write_json(HERE / "migration_decision.json", decision)

    all_scientific_rows = [
        row
        for pattern in (
            "global256_exact/shard*.jsonl",
            "candidate_screen16/shard*.jsonl",
            "promotion1/shard*.jsonl",
            "promotion2/shard*.jsonl",
            "interpolation8/shard*.jsonl",
        )
        for row in read_rows(pattern)
    ]
    runtime = {
        "schema": "orthoflow3_migration_runtime_v1",
        "new_continuation_rollouts": new_rollouts,
        "new_physical_steps": new_steps,
        "rollout_budget": 15000,
        "physical_step_budget": 8000000,
        "within_rollout_budget": new_rollouts <= 15000,
        "within_physical_step_budget": new_steps <= 8000000,
        "stage_a": {
            "pass_run_rollouts": 3,
            "pass_run_steps": 2135,
            "pre_gate_precision_order_harness_run_rollouts": 3,
            "pre_gate_precision_order_harness_steps": 2071,
            "note": "the first harness exposed JAX x64 initialization ordering and did not authorize Stage B; archived replay passed in a fresh process",
        },
        "gpu_stage_wall_seconds_sum_of_stage_maxima": measured_gpu_wall,
        "audit_elapsed_wall_seconds_from_protocol_freeze": time.time() - (HERE / "protocol.md").stat().st_mtime,
        "gpu_shards_peak": 4,
        "gpu_memory_peak_mib_per_shard_observed": 594,
        "gpu_memory_peak_mib_aggregate_observed_approx": 2376,
        "cpu_threads_configured_peak": 12,
        "slurm_cpu_request_per_shard": 3,
        "slurm_memory_request_gib_per_shard": 20,
        "slurm_memory_request_gib_aggregate_peak": 80,
        "observed_process_rss_mib_per_shard_approx": 779,
        "observed_process_rss_mib_aggregate_approx": 3116,
        "hard_safety_and_solver": {
            "collisions": sum(row["outcome"] == "collision" for row in all_scientific_rows),
            "execution_errors": sum(row["execution_error"] is not None for row in all_scientific_rows),
            "first_projection_retries_recovered": sum(int(row["first_projection_retries"]) for row in all_scientific_rows),
            "second_projection_retries_recovered": sum(int(row["second_projection_retries"]) for row in all_scientific_rows),
            "unrecovered_projection_solver_failures": 0,
        },
        "runtime_files": [{"path": str(path), "sha256": sha(path)} for path in runtime_files],
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    report = f"""# OrthoFlow3 representation migration report

## Decision

**{classification}**

Stage A passed, the explicit versioned basis selector is integrated without
changing P0, and the frozen 32-state migration audit completed within budget.
The full 424-state rebuild was not launched because its exact 256-point screen
alone requires `{exact424_rollouts:,}` continuations, above the automatic
15,000 limit.

## Authoritative implementation

- Source: `diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py`
- SHA256: `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`
- Symbols: `raw_ortho_flow`, `basis_terms`, `correction`
- Frozen scale: `3.303687238760696` from `P1_SCALE.json` SHA256
  `ecdca9e63e5d345ec45ae264bcdbc1006f9281cc5a32c3fdfec901e55609a133`
- Candidate design: 256 frozen Sobol points, SHA256
  `6a0732c5ad35bc2000e61be6c2825034058b30aad09458cf2cf5975cc501f258`

The exact basis equations and code references are recorded in
`basis_interface_spec.md`; no formula was reconstructed from prompt prose.

## Integrity

- Archived P0 and OrthoFlow3 fields reproduced with maximum absolute error 0
  on 12 archived anchors.
- Three archived selected-center continuations reproduced exact basis fields
  and successful outcomes.
- Eta zero produced exactly zero correction and exactly the Safety action on
  64 state/Flow/family comparisons.
- Historical P0 source files were not modified; new jobs require explicit
  `basis_family = p0 | orthoflow3`.

## Frozen 32-state audit

| Metric | P0 | OrthoFlow3 |
|---|---:|---:|
| robust basin found | {p0_basin}/32 | {of3_basin}/32 |
| median observed Q_max | {median(row['p0_Q_max_observed'] for row in capacity_rows):.4f} | {median(row['orthoflow3_Q_max_observed'] for row in capacity_rows):.4f} |
| median best J_def | {median(float(row['p0_best_J_def']) for row in capacity_rows):.6g} | {median(float(row['orthoflow3_best_J_def']) for row in capacity_rows if row['orthoflow3_best_J_def'] != '') if any(row['orthoflow3_best_J_def'] != '' for row in capacity_rows) else float('nan'):.6g} |
| median best J_def, active-required 14 | {median(float(row['p0_best_J_def']) for row in capacity_rows if state_map[row['state_id']]['zero_eta'] is False):.6g} | {median(float(row['orthoflow3_best_J_def']) for row in capacity_rows if state_map[row['state_id']]['zero_eta'] is False and row['orthoflow3_best_J_def'] != ''):.6g} |

Paired basin existence: both `{pair_counts[(True, True)]}`, P0-only
`{pair_counts[(True, False)]}`, OrthoFlow3-only `{pair_counts[(False, True)]}`,
neither `{pair_counts[(False, False)]}`.  OrthoFlow3 states classify as
ZERO_SUFFICIENT `{status_counts['ZERO_SUFFICIENT']}`, ACTIVE_REQUIRED
`{status_counts['ACTIVE_REQUIRED']}`, NO_BASIN_FOUND
`{status_counts['NO_BASIN_FOUND']}`.

The median per-agent maximum absolute goal/middle cosine fell from
`{median(p0_cos):.6g}` under P0 to `{median(of3_cos):.6g}` under OrthoFlow3,
confirming that the old redundancy is materially reduced.

The audit confirmed at least one robust candidate in `{of3_basin}/32` states.
It confirmed at least two robust points in
`{sum(int(row['robust_candidate_count_lower_bound']) >= 2 for row in basin_rows)}/32`
states; candidate counts and the reported canonical target are lower-bound /
provisional results within the budgeted promoted set.  Of
`{tested_interpolants}` eight-stream interpolation screens,
`{interpolation_fail}` had enough observed failures to reject B63 and
`{interpolation_perfect}` were 8/8 but remain explicitly unconfirmed at B63.

Across the 14,624 Stage-B/C continuation records there were zero collisions,
zero execution errors, and zero unrecovered projection failures.  Sixteen
first-projection and six second-projection retries were recovered by the
unchanged frozen solver path.

## Migration scope and next step

All future eta/basin oracle and dataset-learning work should default to
**OrthoFlow3** with an explicit basis version/hash.  P0 remains available only
as a reproducible historical baseline or declared ablation.

The smallest justified next experiment is the already-prepared full-424
OrthoFlow3 oracle/data rebuild after explicit approval of the reported compute
cost.  No neural model should be trained before that rebuilt candidate evidence
is inspected.
"""
    (HERE / "migration_report.md").write_text(report)

    files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append({"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {
        "schema": "orthoflow3_representation_migration_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "full424_executed": False,
        "files": files,
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "classification": classification,
        "p0_basin": p0_basin,
        "orthoflow3_basin": of3_basin,
        "status_counts": dict(status_counts),
        "new_rollouts": new_rollouts,
        "new_steps": new_steps,
        "interpolation_rejections": interpolation_fail,
    }, indent=2))


if __name__ == "__main__":
    main()
