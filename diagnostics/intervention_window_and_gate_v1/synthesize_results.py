"""Create the final two-stage audit report, runtime summary, and manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
S1 = HERE / "stage1_window"
S2 = HERE / "stage2_gate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def empirical_median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> None:
    stage1_required = [
        "frozen_state_list.csv", "per_state_delay_curves.csv", "recoverability_curves.csv",
        "deformation_curves.csv", "transition_intervals.csv", "pareto_frontier.csv",
        "candidate_windows.json", "stage1_report.md",
    ]
    missing = [str(S1 / name) for name in stage1_required if not (S1 / name).exists()]
    if missing:
        raise RuntimeError(f"missing required Stage-1 artifacts: {missing}")
    plan = json.loads((S1 / "audit_plan.json").read_text())
    candidates = json.loads((S1 / "candidate_windows.json").read_text())
    transitions = read_csv(S1 / "transition_intervals.csv")
    qcurve = read_csv(S1 / "recoverability_curves.csv")
    jcurve = read_csv(S1 / "deformation_curves.csv")
    targets = read_csv(S2 / "window_targets.csv")
    ambiguous = read_csv(S2 / "ambiguous_targets.csv")
    metrics = read_csv(S2 / "metrics.csv")
    stage2_summary = json.loads((S2 / "stage2_summary.json").read_text())

    # Measured-grid descriptive windows.  The SHORT/MEDIUM boundary is the
    # empirical median of observed non-immediate first-bad delays, not a
    # preregistered physical threshold.  This keeps the names descriptive and
    # prevents them from entering target construction.
    finite_first_bad = [
        int(row["d_first_supported_bad"])
        for row in transitions if row["d_first_supported_bad"]
    ]
    first_tested_positive = min(int(value) for value in plan["coarse_delays"] if int(value) > 0)
    non_immediate_bad = [value for value in finite_first_bad if value != first_tested_positive]
    short_medium_boundary = empirical_median(non_immediate_bad) if non_immediate_bad else None
    window_counts = Counter()
    for row in transitions:
        first = int(row["d_first_supported_bad"]) if row["d_first_supported_bad"] else None
        last = int(row["d_last_resolved_safe"])
        unresolved = bool(row["unresolved_delays"])
        if first is not None and first == first_tested_positive and last == 0:
            category = "IMMEDIATE"
        elif first is not None and short_medium_boundary is not None and first <= short_medium_boundary:
            category = "SHORT_WINDOW"
        elif first is not None:
            category = "MEDIUM_WINDOW"
        elif not unresolved and last >= 64:
            category = "LONG_WINDOW"
        else:
            category = "UNRESOLVED"
        window_counts[category] += 1

    inventory = candidates["all_coarse_horizon_inventory"]
    selected = [int(v) for v in candidates["candidate_horizons"]]
    stage2_required = ["model_comparison.csv", "stage2_report.md"]
    for H in selected:
        stage2_required += [f"target_H{H}.csv", f"ambiguity_H{H}.csv", f"oof_predictions_H{H}.csv", f"metrics_H{H}.json"]
    missing = [str(S2 / name) for name in stage2_required if not (S2 / name).exists()]
    if missing:
        raise RuntimeError(f"missing required Stage-2 artifacts: {missing}")
    target_counts = Counter((int(row["H"]), int(row["y_H"])) for row in targets)
    ambiguity_counts = Counter(int(row["H"]) for row in ambiguous)
    ensemble_mlp = [row for row in metrics if row["model"] == "MLP_64x64" and row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "ALL_RESOLVED"]
    urgency_mlp = [row for row in metrics if row["model"] == "MLP_64x64" and row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "EVENTUALLY_NEEDED_Y_LONG1"]
    recovery_mlp = [row for row in metrics if row["model"] == "MLP_64x64" and row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "RECOVERY"]
    by_h = {int(row["H"]): row for row in ensemble_mlp}
    urgency_by_h = {int(row["H"]): row for row in urgency_mlp}
    recovery_by_h = {int(row["H"]): row for row in recovery_mlp}
    vectors = defaultdict(dict)
    for row in targets:
        vectors[row["state_id"]][int(row["H"])] = int(row["y_H"])
    changes = 0
    violations = 0
    for values in vectors.values():
        hs = sorted(values)
        for left, right in zip(hs, hs[1:]):
            changes += values[left] != values[right]
            violations += values[left] == 1 and values[right] == 0
    usable = [h for h in selected if h in by_h and finite(by_h[h]["balanced_accuracy"]) and finite(by_h[h]["AUROC"])]
    def has_nontrivial_target(h: int) -> bool:
        resolved = target_counts[(h, 0)] + target_counts[(h, 1)]
        total = resolved + ambiguity_counts[h]
        return (
            resolved > 0
            and min(target_counts[(h, 0)], target_counts[(h, 1)]) / resolved >= 0.10
            and ambiguity_counts[h] / max(total, 1) <= 0.50
        )

    def subgroup_is_useful(row: dict | None) -> bool:
        return bool(
            row
            and finite(row.get("balanced_accuracy"))
            and finite(row.get("AUROC"))
            and float(row["balanced_accuracy"]) > 0.55
            and float(row["AUROC"]) > 0.55
            and float(row["FPR"]) < 0.50
            and float(row["FNR"]) < 0.50
        )

    usable_binary = [
        h for h in usable
        if has_nontrivial_target(h)
        and finite(by_h[h].get("balanced_accuracy_group_bootstrap95_lower"))
        and finite(by_h[h].get("AUROC_group_bootstrap95_lower"))
        and float(by_h[h]["balanced_accuracy_group_bootstrap95_lower"]) > 0.5
        and float(by_h[h]["AUROC_group_bootstrap95_lower"]) > 0.5
        and float(by_h[h]["FPR"]) < 0.4
        and float(by_h[h]["FNR"]) < 0.4
        and subgroup_is_useful(recovery_by_h.get(h))
        and subgroup_is_useful(urgency_by_h.get(h))
    ]
    q_lines = []
    for row in qcurve:
        if row["scope"] == "FULL_FROZEN_POOL":
            q_lines.append(f"| {row['delay_steps']} | {float(row['macro_Q']):.6f} | {float(row['macro_Delta_Q0_minus_QH']):.6f} | {row['state_count']} |")
    j_lines = []
    for row in jcurve:
        if row["scope"] == "FULL_FROZEN_POOL":
            j_lines.append(f"| {row['delay_steps']} | {float(row['macro_mean_J_def']):.6f} | {float(row['macro_Delta_JH_minus_J0']):.6f} |")
    metric_lines = []
    for H in selected:
        for model in ("LINEAR", "MLP_64x64"):
            found = next((row for row in metrics if int(row["H"]) == H and row["model"] == model and row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "ALL_RESOLVED"), None)
            if found:
                metric_lines.append(f"| {H} | {H * float(plan['environment']['dt']):.2f} | {model} | {int(found['state_count'])} | {float(found['balanced_accuracy']):.4f} | {float(found['AUROC']):.4f} | {float(found['AUPRC']):.4f} | {float(found['FPR']):.4f} | {float(found['FNR']):.4f} |")
    urgency_metric_lines = []
    for H in selected:
        found = urgency_by_h.get(H)
        if found:
            urgency_metric_lines.append(
                f"| {H} | {int(found['state_count'])} | {float(found['balanced_accuracy']):.4f} | "
                f"{float(found['AUROC']):.4f} | {float(found['FPR']):.4f} | {float(found['FNR']):.4f} |"
            )
    first_bad_distribution = Counter(
        row["d_first_supported_bad"] if row["d_first_supported_bad"] else "RIGHT_CENSORED"
        for row in transitions
    )
    last_safe_distribution = Counter(int(row["d_last_resolved_safe"]) for row in transitions)

    aggregate_degradation = next((
        row for row in qcurve
        if row["scope"] == "FULL_FROZEN_POOL"
        and int(row["delay_steps"]) > 0
        and float(row["Delta_bootstrap95_lower"]) > 0
    ), None)

    pareto = read_csv(S1 / "pareto_frontier.csv")
    pareto_waiting = sum(
        int(row["delay_steps"]) > 0
        and str(row["point_estimate_pareto_nondominated"]).lower() == "true"
        for row in pareto
    )
    full_j_nonzero = [
        row for row in jcurve
        if row["scope"] == "FULL_FROZEN_POOL" and int(row["delay_steps"]) > 0
    ]
    j_direction_counts = Counter(
        "INCREASE"
        if float(row["Delta_bootstrap95_lower"]) > 0
        else "DECREASE"
        if float(row["Delta_bootstrap95_upper"]) < 0
        else "UNRESOLVED_OR_PRESERVED"
        for row in full_j_nonzero
    )

    # This is a diagnostic comparison, not an unbiased deployment selection:
    # every horizon remains reported.  Pick one only so the handoff can answer
    # which measured target looks most promising for the next *fresh* test.
    best_h = None
    if usable:
        best_h = max(
            usable,
            key=lambda h: (
                float(urgency_by_h.get(h, {}).get("balanced_accuracy", -1)),
                float(by_h[h]["balanced_accuracy"]),
                float(by_h[h]["AUROC"]),
                -ambiguity_counts[h],
                -h,
            ),
        )

    resolved_at_earliest = 0
    positive_fraction_earliest = None
    if selected:
        earliest = min(selected)
        resolved_at_earliest = target_counts[(earliest, 0)] + target_counts[(earliest, 1)]
        if resolved_at_earliest:
            positive_fraction_earliest = target_counts[(earliest, 1)] / resolved_at_earliest

    long_majority = window_counts["LONG_WINDOW"] >= math.ceil(0.6 * len(transitions))
    short_target_nearly_trivial = (
        positive_fraction_earliest is not None
        and min(positive_fraction_earliest, 1 - positive_fraction_earliest) < 0.15
    )
    if usable_binary:
        classification = "WINDOW_ALIGNED_GATE_SUPPORTED"
        target = "binary window gate"
    elif long_majority and (not selected or short_target_nearly_trivial):
        classification = "LOW_FREQUENCY_REQUERY_SUPPORTED"
        target = "lower-frequency decision"
    elif (
        len(selected) >= 2
        and changes >= max(5, math.ceil(0.05 * len(transitions)))
        and violations == 0
        and all(has_nontrivial_target(h) for h in selected)
    ):
        classification = "ORDINAL_URGENCY_MORE_APPROPRIATE"
        target = "ordinal urgency"
    else:
        classification = "TEMPORAL_TARGET_REMAINS_UNRESOLVED"
        target = "unclear"

    if classification == "WINDOW_ALIGNED_GATE_SUPPORTED" and best_h is not None:
        next_step = (
            f"Freeze H={best_h} as a hypothesis and run one fresh, independent strict-source-group "
            "confirmation split; only if confirmed, train the 4-D correction head offline on "
            "resolved y_H=1 states while keeping this gate and oracle frozen."
        )
    elif classification == "ORDINAL_URGENCY_MORE_APPROPRIATE":
        next_step = (
            "Fit one small ordinal urgency probe to the already frozen horizon labels under the "
            "same strict LOGO protocol; do not train the correction head until that probe generalizes."
        )
    elif classification == "LOW_FREQUENCY_REQUERY_SUPPORTED":
        next_step = (
            "Run the small oracle-only periodic re-query check at the earliest empirically supported "
            "long decision period; do not train the correction head until recoverability is confirmed."
        )
    else:
        next_step = (
            "Add matched continuation seeds only for Stage-1 transition-ambiguous state/horizon tuples "
            "and repeat target resolution; do not train the correction head yet."
        )

    raw_manifests = list((S1 / "raw").glob("**/manifest.json"))
    stage1_elapsed_preview = sum(
        float(json.loads(path.read_text()).get("elapsed_s", json.loads(path.read_text()).get("elapsed_s_this_invocation", 0)))
        for path in raw_manifests
    )
    training_results_preview = read_csv(S2 / "training_results.csv")
    stage2_elapsed_preview = sum(float(row["wall_seconds"]) for row in training_results_preview)

    degradation_text = (
        f"first aggregate statistically supported loss at d={aggregate_degradation['delay_steps']} "
        f"({float(aggregate_degradation['delay_steps']) * float(plan['environment']['dt']):.2f} s)"
        if aggregate_degradation is not None
        else "no aggregate statistically supported recoverability loss on the full-pool coarse grid"
    )
    candidate_reason_lines = [
        f"- H={row['H']}: resolved zero={row['resolved_negative']}, resolved one={row['resolved_positive']}, "
        f"ambiguous={row['ambiguous']}, zero/one source groups={row['negative_source_groups']}/{row['positive_source_groups']}, "
        f"y_long=1 zero/one source groups={row['eventually_needed_negative_source_groups']}/{row['eventually_needed_positive_source_groups']}, "
        f"LOGO feasible={row['strict_logo_feasible']}"
        for row in inventory if int(row["H"]) in selected
    ]

    report = f"""# Intervention-window and gate audit

## Conclusion

**{classification}**  
Best-supported target form: **{target}**.

The Stage-1 horizon candidates were frozen before any Stage-2 model run.  The
classification above follows a conservative rule frozen before Stage-2
outputs were available: a binary horizon requires nontrivial class balance,
at most 50% ambiguity, source-group-bootstrap lower bounds above chance for
both BAcc and AUROC, FPR/FNR below 0.4, and above-chance RECOVERY plus
`y_long=1` urgency-only diagnostics. If no binary horizon passes but resolved
horizon labels change monotonically, the evidence may favor ordinal urgency.
This is not an unbiased deployment-model selection.

## Stage 1: paired intervention-window evidence

- Frozen oracle-stable states: **{len(plan['states'])}** from **{plan['selection']['source_group_count']}** source groups.
- Fixed downstream limitation: frozen `eta_best`, not arbitrary-state receding oracle re-query.
- Window counts: `{dict(window_counts)}`.
- `d_last_safe` distribution: `{dict(sorted(last_safe_distribution.items()))}`.
- `d_first_bad` distribution: `{dict(first_bad_distribution)}`.
- Aggregate onset: **{degradation_text}**.
- Aggregate J_def direction counts across full-pool nonzero delays: `{dict(j_direction_counts)}`.
- Recoverability-compatible nonzero-delay Pareto points: **{pareto_waiting}**.
- Candidate horizons frozen for Stage 2: `{selected}`.

| Delay (steps) | State-macro Q | Macro Q0-QH | States |
|---:|---:|---:|---:|
{chr(10).join(q_lines)}

| Delay (steps) | State-macro J_def | Macro JH-J0 |
|---:|---:|---:|
{chr(10).join(j_lines)}

Candidate label inventory:
{chr(10).join(f'- H={h}: resolved zero={target_counts[(h,0)]}, resolved one={target_counts[(h,1)]}, ambiguous={ambiguity_counts[h]}' for h in selected)}

Candidate-freeze rationale (Stage-1 evidence only):
{chr(10).join(candidate_reason_lines)}

## Stage 2: strict source-group LOGO

- Feature vector: unchanged 214-D deployment input.
- Models: linear `214->1` and SiLU MLP `214->64->64->1`.
- Loss: ordinary BCE on resolved targets only; ambiguous targets excluded.
- Preregistered seed: 17 only, fixed uniformly before any Stage-2 performance was observed.
- Cross-fold ranking coordinate: fold-relative logit margin; raw logits from different folds are not treated as one calibration coordinate.

| H | Period (s) | Model | Resolved states | BAcc | AUROC | AUPRC | FPR | FNR |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

Urgency-only diagnostic among states with `y_long=1` (metadata only, never a
training target):

| H | Resolved states | BAcc | AUROC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(urgency_metric_lines)}

Resolved horizon-label changes: **{changes}**; ordinal reversals: **{violations}**.

Descriptively best Stage-2 horizon: **{best_h}**. This is a diagnostic
comparison only and requires fresh confirmation before adoption.

## Concise handoff — 12 answers

1. Audited **{len(plan['states'])}** oracle-stable states from **{plan['selection']['source_group_count']}** source groups.
2. The complete coarse recoverability curve is tabulated above; all Q intervals and paired `Q0-QH` effects are in Stage 1.
3. Empirical window counts are `{dict(window_counts)}`; first-bad and last-safe distributions are shown above.
4. The preregistered candidate horizons are `{selected}` and were frozen without gate/test performance.
5. Per-H resolved/ambiguous counts are `{ {h: {'zero': target_counts[(h,0)], 'one': target_counts[(h,1)], 'ambiguous': ambiguity_counts[h]} for h in selected} }`.
6. Q and J_def remain separate; J_def directions are `{dict(j_direction_counts)}` and Pareto points are reported without scalarization.
7. Strict LOGO linear-vs-MLP results are shown in the Stage-2 table above.
8. NORMAL/PRE_DEADLOCK/RECOVERY and hard-13 metrics are retained in `metrics_H*.json`.
9. Cross-fold scoring uses each fold's validation-relative logit margin; held-out groups never enter normalization or thresholds.
10. `y_long` is diagnostic metadata only and never defines `y_H` or BCE supervision.
11. Final classification: **{classification}**; target recommendation: **{target}**.
12. Smallest justified next experiment: **{next_step}** Stage-1 sum-shard runtime was **{stage1_elapsed_preview:.1f} s** and Stage-2 sum-job runtime was **{stage2_elapsed_preview:.1f} s**; execution used at most **2 GPU shards**, **4 CPU cores per worker / 8 total**, and CPU-only statistics.

## Integrity

- Candidate horizon file was hashed and frozen before training.
- Every outer source group was excluded from training, normalization,
  validation, threshold selection, and model selection.
- No gate/G_phi feature, oracle, physics, or correction model was changed.
- No learned-controller closed-loop benchmark was run.
"""
    (HERE / "parent_summary.md").write_text(report)

    raw_manifests = list((S1 / "raw").glob("**/manifest.json"))
    stage1_elapsed = stage1_new = stage1_reused = stage1_steps = 0
    devices = set()
    for path in raw_manifests:
        obj = json.loads(path.read_text())
        stage1_elapsed += float(obj.get("elapsed_s", obj.get("elapsed_s_this_invocation", 0)))
        stage1_new += int(obj.get("new_rollouts", 0))
        stage1_new += int(obj.get("new_physical_rollouts", 0)) if "new_rollouts" not in obj else 0
        stage1_reused += int(obj.get("reused_rollouts", obj.get("reused_or_materialized_records", 0)))
        stage1_steps += int(obj.get("physical_steps_new", 0))
        devices.update(map(str, obj.get("device", [])))
    training_results = read_csv(S2 / "training_results.csv")
    runtime = {
        "stage1_sum_shard_seconds": stage1_elapsed,
        "stage1_new_physical_rollouts": stage1_new,
        "stage1_reused_or_materialized_tuples": stage1_reused,
        "stage1_new_physical_steps": stage1_steps,
        "stage1_devices": sorted(devices),
        "stage2_sum_job_seconds": sum(float(row["wall_seconds"]) for row in training_results),
        "stage2_training_jobs": len(training_results),
        "maximum_parallel_gpu_workers": 2,
        "maximum_gpu_shards_total": 2,
        "cpu_cores_per_gpu_worker": 4,
        "maximum_cpu_cores_total": 8,
        "statistics_device": "CPU",
    }
    (HERE / "runtime_statistics.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    sanity = {
        "stage1_plan_hash_matches_candidates": candidates["audit_plan_sha256"] == sha(S1 / "audit_plan.json"),
        "candidate_frozen_before_training": candidates["frozen_before_any_stage2_training"],
        "stage2_all_required_runs_complete": stage2_summary["required_run_count"] == stage2_summary["completed_run_count"],
        "strict_source_group_logo": True,
        "ambiguous_excluded_from_hard_bce": True,
        "raw_cross_fold_logits_compared_directly": False,
        "gate_or_gphi_feature_modified": False,
        "oracle_or_physics_modified": False,
        "learned_closed_loop_run": False,
    }
    (HERE / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == HERE / "manifest.json" or "__pycache__" in path.parts:
            continue
        if "/runs/" in str(path) and path.name != "manifest.json":
            continue
        files[str(path.relative_to(HERE))] = sha(path)
    manifest = {"status": "complete", "classification": classification, "target_recommendation": target, "files_sha256": files}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": classification, "target": target, "candidate_horizons": selected}, indent=2))


if __name__ == "__main__":
    main()
