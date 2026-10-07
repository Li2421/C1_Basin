"""Create the user-requested Stage-1-only handoff and integrity manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
S1 = HERE / "stage1_window"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> None:
    required = [
        "frozen_state_list.csv", "per_state_delay_curves.csv",
        "paired_success_differences.csv", "paired_deformation_differences.csv",
        "recoverability_curves.csv", "deformation_curves.csv",
        "transition_intervals.csv", "pareto_frontier.csv",
        "candidate_windows.json", "stage1_report.md", "sanity_checks.json",
    ]
    missing = [name for name in required if not (S1 / name).exists()]
    if missing:
        raise RuntimeError(f"missing Stage-1 artifacts: {missing}")

    plan = json.loads((S1 / "audit_plan.json").read_text())
    checks = json.loads((S1 / "sanity_checks.json").read_text())
    candidates = json.loads((S1 / "candidate_windows.json").read_text())
    transitions = read_csv(S1 / "transition_intervals.csv")
    qcurve = read_csv(S1 / "recoverability_curves.csv")
    jcurve = read_csv(S1 / "deformation_curves.csv")
    pareto = read_csv(S1 / "pareto_frontier.csv")
    states = read_csv(S1 / "frozen_state_list.csv")

    finite_bad = [int(row["d_first_supported_bad"]) for row in transitions if row["d_first_supported_bad"]]
    first_grid_delay = min(int(value) for value in plan["coarse_delays"] if int(value) > 0)
    later_bad = [value for value in finite_bad if value != first_grid_delay]
    split = median(later_bad) if later_bad else None
    categories = Counter()
    for row in transitions:
        first = int(row["d_first_supported_bad"]) if row["d_first_supported_bad"] else None
        last = int(row["d_last_resolved_safe"])
        unresolved = bool(row["unresolved_delays"])
        if first == first_grid_delay and last == 0:
            category = "IMMEDIATE"
        elif first is not None and split is not None and first <= split:
            category = "SHORT_WINDOW"
        elif first is not None:
            category = "MEDIUM_WINDOW"
        elif not unresolved and last >= 64:
            category = "LONG_WINDOW"
        else:
            category = "UNRESOLVED"
        categories[category] += 1

    last_safe = Counter(int(row["d_last_resolved_safe"]) for row in transitions)
    first_bad = Counter(row["d_first_supported_bad"] or "RIGHT_CENSORED" for row in transitions)
    full_q = [row for row in qcurve if row["scope"] == "FULL_FROZEN_POOL"]
    full_j = [row for row in jcurve if row["scope"] == "FULL_FROZEN_POOL"]
    onset = next((
        row for row in full_q
        if int(row["delay_steps"]) > 0 and float(row["Delta_bootstrap95_lower"]) > 0
    ), None)
    j_directions = Counter(
        "INCREASE" if float(row["Delta_bootstrap95_lower"]) > 0
        else "DECREASE" if float(row["Delta_bootstrap95_upper"]) < 0
        else "UNRESOLVED_OR_PRESERVED"
        for row in full_j if int(row["delay_steps"]) > 0
    )
    pareto_wait = sum(
        int(row["delay_steps"]) > 0
        and row["point_estimate_pareto_nondominated"].lower() == "true"
        for row in pareto
    )
    ylong1 = [row for row in transitions if int(row["y_long"]) == 1]
    ylong1_immediate = sum(row["d_first_supported_bad"] == str(first_grid_delay) for row in ylong1)

    q_lines = [
        f"| {row['delay_steps']} | {float(row['delay_steps']) * float(plan['environment']['dt']):.2f} | "
        f"{float(row['macro_Q']):.6f} | {float(row['macro_Delta_Q0_minus_QH']):.6f} | "
        f"[{float(row['Delta_bootstrap95_lower']):.6f}, {float(row['Delta_bootstrap95_upper']):.6f}] |"
        for row in full_q
    ]
    j_lines = [
        f"| {row['delay_steps']} | {float(row['macro_mean_J_def']):.6f} | "
        f"{float(row['macro_Delta_JH_minus_J0']):.6f} | "
        f"[{float(row['Delta_bootstrap95_lower']):.6f}, {float(row['Delta_bootstrap95_upper']):.6f}] |"
        for row in full_j
    ]
    onset_text = (
        f"d={onset['delay_steps']} ({float(onset['delay_steps']) * float(plan['environment']['dt']):.2f} s)"
        if onset else "not detected on the full-pool coarse grid"
    )
    inventory_lines = [
        f"- H={row['H']} ({int(row['H']) * float(plan['environment']['dt']):.2f} s): "
        f"resolved zero/one={row['resolved_negative']}/{row['resolved_positive']}, "
        f"ambiguous={row['ambiguous']}, strict-LOGO-feasible={row['strict_logo_feasible']}"
        for row in candidates["all_coarse_horizon_inventory"]
        if int(row["H"]) in candidates["candidate_horizons"]
    ]

    report = f"""# Stage-1 intervention-window audit — final handoff

Stage 1 is complete by explicit user instruction. Stage 2 gate training was
not prepared or run.

## Scope and integrity

- Oracle-stable states: **{len(states)}** across **{plan['selection']['source_group_count']}** source groups.
- Coarse delays: `{plan['coarse_delays']}` steps; adaptive extension/refinement used matched randomness.
- Adaptive protocol: transition ambiguity advanced **64 -> 128 -> 256** only; terminal ambiguous points at the cap: **{checks['terminal_ambiguous_points_at_256']}**.
- Downstream limitation: frozen `eta_best` recovery approximation, not arbitrary-state every-step oracle re-query.
- Stage 2: **not run**; no gate or G_phi was trained.

## Empirical windows

- Window categories: `{dict(categories)}`.
- `d_last_safe` distribution: `{dict(sorted(last_safe.items()))}`.
- `d_first_bad` distribution: `{dict(first_bad)}`.
- First aggregate statistically supported recoverability degradation: **{onset_text}**.
- Among old `y_long=1` states: **{ylong1_immediate}/{len(ylong1)}** degrade at the first tested positive delay (`d={first_grid_delay}`); `y_long` remains diagnostic only.

| Delay | Seconds | Macro Q | Macro Q0-Qd | Paired-state bootstrap 95% CI |
|---:|---:|---:|---:|---:|
{chr(10).join(q_lines)}

## Deformation and tradeoff

- Aggregate J_def directions across full-pool nonzero delays: `{dict(j_directions)}`.
- Recoverability-compatible, nonzero-delay Pareto points: **{pareto_wait}**.

| Delay | Macro J_def | Macro Jd-J0 | Paired-state bootstrap 95% CI |
|---:|---:|---:|---:|
{chr(10).join(j_lines)}

Q and J_def are reported separately; no scalarization was used.

## Stage-1-frozen candidate horizons (not trained)

Candidates: **{candidates['candidate_horizons']}**. They were selected only
from Stage-1 evidence and frozen before any Stage-2 model result existed.

{chr(10).join(inventory_lines)}

## Terminal interpretation

This run establishes the empirical intervention-window target evidence only.
Whether any frozen `y_H` target is source-generalizable remains **untested**
because Stage 2 was explicitly deferred. The smallest next experiment, if
resumed later, is the already specified strict source-group LOGO target-validity
probe on the frozen candidate horizons; correction G_phi should remain deferred
until that probe succeeds.
"""
    (HERE / "parent_summary.md").write_text(report)

    raw_manifests = list((S1 / "raw").glob("**/manifest.json"))
    elapsed = new = reused = steps = 0
    devices = set()
    for path in raw_manifests:
        obj = json.loads(path.read_text())
        elapsed += float(obj.get("elapsed_s", obj.get("elapsed_s_this_invocation", 0)))
        new += int(obj.get("new_rollouts", obj.get("new_physical_rollouts", 0)))
        reused += int(obj.get("reused_rollouts", obj.get("reused_or_materialized_records", 0)))
        steps += int(obj.get("physical_steps_new", 0))
        devices.update(map(str, obj.get("device", [])))
    runtime = {
        "scope": "STAGE1_ONLY",
        "sum_shard_seconds": elapsed,
        "new_physical_rollouts": new,
        "reused_or_materialized_tuples": reused,
        "new_physical_steps": steps,
        "devices": sorted(devices),
        "maximum_gpu_shards_total": 2,
        "cpu_cores_per_worker": 4,
        "maximum_cpu_cores_total": 8,
        "stage2_training_jobs": 0,
    }
    (HERE / "runtime_statistics.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    sanity = {
        "stage1_complete": True,
        "adaptive_convergence_certified": bool(checks.get("adaptive_followup_convergence_sha256")),
        "candidate_frozen_before_training": candidates["frozen_before_any_stage2_training"],
        "stage2_run": False,
        "gate_trained": False,
        "gphi_trained": False,
        "learned_closed_loop_run": False,
        "oracle_or_physics_modified": False,
    }
    (HERE / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == HERE / "manifest.json" or "__pycache__" in path.parts:
            continue
        files[str(path.relative_to(HERE))] = sha(path)
    manifest = {
        "status": "complete",
        "scope": "STAGE1_ONLY_BY_USER_INSTRUCTION",
        "stage2_run": False,
        "candidate_horizons": candidates["candidate_horizons"],
        "files_sha256": files,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "scope": "STAGE1_ONLY", "candidate_horizons": candidates["candidate_horizons"]}, indent=2))


if __name__ == "__main__":
    main()
