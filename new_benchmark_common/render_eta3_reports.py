"""Render the two formal safety/eta reports and their combined comparison."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np

from new_benchmark_common.safety_eta3 import ROOT, SCENARIOS, ScenarioRuntime
from shared_rollout_db.src.rollout_db import connect


def load(path: Path):
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({pct(count / total if total else 0.0)})"


def numeric_audit(runtime: ScenarioRuntime) -> dict:
    controller = runtime.controllers["orthoflow3"]["uid"]
    with connect(True) as con:
        row = con.execute(
            """SELECT COUNT(*) n, SUM(numerical_failure) numerical,
            SUM(success) success, SUM(collision) collision FROM rollout
            WHERE controller_uid=? AND conflict_quarantined=0""", (controller,)
        ).fetchone()
        foreign = list(con.execute("PRAGMA foreign_key_check"))
    return {**dict(row), "foreign_key_violations": len(foreign)}


def mechanism_summary(runtime: ScenarioRuntime, mechanism: dict) -> str:
    pairs = mechanism["pairs"]
    planned = mechanism.get("planned_pairs", len(pairs))
    incomplete = len(mechanism.get("numerical_incomplete_pairs", []))
    rescued = [p for p in pairs if p["eta_center"]["outcome"] == "success"]
    mode_changes = sum(p["eta_zero"]["mode_signature"] != p["eta_center"]["mode_signature"] for p in pairs)
    wait = [p["waiting_change"] for p in rescued]
    time_change = [p["completion_time_change"] for p in rescued]
    if runtime.kind == "four":
        cyclic = [
            p["eta_center"]["cyclic_yielding_proxy"]["total_steps"] -
            p["eta_zero"]["cyclic_yielding_proxy"]["total_steps"] for p in rescued
        ]
        return (
            f"On diagnostic seed 16, {len(rescued)}/{len(pairs)} numerically valid selected-center pairs rescued "
            f"their state ({incomplete}/{planned} planned pairs were numerical-incomplete); "
            f"{mode_changes}/{len(pairs)} changed the realized crossing signature. Among rescued pairs, median "
            f"completion-time change was {np.median(time_change) if time_change else float('nan'):.3f}s, median "
            f"per-agent waiting change was {np.median(wait) if wait else float('nan'):.3f}s, and the median change "
            f"in the explicitly labelled low-speed/outside-centre cyclic-yielding proxy was "
            f"{np.median(cyclic) * runtime.config.dt if cyclic else float('nan'):.3f}s. These are descriptive "
            "behavioral proxies, not a runtime deadlock declaration."
        )
    reversals = [
        sum(p["eta_center"]["circulation_reversals"]) - sum(p["eta_zero"]["circulation_reversals"])
        for p in rescued
    ]
    head_on = [
        p["eta_center"]["head_on_interaction_seconds"] - p["eta_zero"]["head_on_interaction_seconds"]
        for p in rescued
    ]
    return (
        f"On diagnostic seed 16, {len(rescued)}/{len(pairs)} numerically valid selected-center pairs rescued "
        f"their state ({incomplete}/{planned} planned pairs were numerical-incomplete); "
        f"{mode_changes}/{len(pairs)} changed the realized circulation signature. Among rescued pairs, median "
        f"completion-time change was {np.median(time_change) if time_change else float('nan'):.3f}s, median "
        f"per-agent waiting change was {np.median(wait) if wait else float('nan'):.3f}s, median total circulation-"
        f"reversal change was {np.median(reversals) if reversals else float('nan'):.1f}, and median head-on proxy "
        f"change was {np.median(head_on) if head_on else float('nan'):.3f}s. Labels are analysis-only."
    )


def render_scenario(name: str) -> dict:
    runtime = ScenarioRuntime(name)
    out = runtime.output
    frozen = load(out / "FROZEN_PROTOCOL.json")
    formal = load(out / "formal_safety_summary.json")
    primary = load(out / "primary_search.json")
    secondary = load(out / "secondary_search.json") if (out / "secondary_search.json").exists() else {}
    existence = load(out / "robust_existence.json")
    local = load(out / "local_basin.json")
    coverage = load(out / "cross_state_coverage.json")
    preservation = load(out / "success_control_preservation.json")
    mechanism = load(out / "behavioral_mechanism.json")
    summary = load(out / "SUMMARY.json")
    efficiency = summary["early_stop_efficiency"]
    audit = numeric_audit(runtime)
    safe = formal["hard_safety"]
    total = safe["episodes"]
    conversions = "\n".join(
        f"| `{key.replace('->', ' → ')}` | {value} |"
        for key, value in sorted(formal["conversion_matrix"].items())
    )
    local_values = [row["robust_fraction"] for row in local.values()]
    cover_values = [row["coverage_fraction"] for row in coverage["centers"]]
    robust_n = len(existence["robust_centers"])
    target_n = len(existence["targets"])
    q4_candidates = sum(sum(int(v) >= 3 for v in row["q4"].values()) for row in primary.values())
    primary_positive = sum(bool(row["robust_indices"]) for row in primary.values())
    secondary_positive = sum(bool(row["robust_indices"]) for row in secondary.values())
    control_numerical = sum(row.get("numerical_incomplete", 0) for row in preservation)
    control_planned = sum(row.get("planned_trials", row["trials"]) for row in preservation)
    report = f"""# {name.replace('_', ' ').title()} — formal hard safety and OrthoFlow3

**Final classification: `{summary['classification']}`.** No MACFlow retraining, `G_phi`, eta-dimension expansion, basis redesign, eta-domain tuning, or scenario-specific basis was used.

## 1. Frozen Stage-I reference

The frozen checkpoint is `{frozen['checkpoint']}` (SHA-256 `{frozen['checkpoint_sha256']}`). The environment fingerprint is `{frozen['environment_fingerprint']}`, horizon is {frozen['horizon']} steps at dt={frozen['dt']}, and the formal comparison uses the same 60 frozen test states and evaluation seed {frozen['evaluation_seed']} as the accepted Stage-I result.

## 2. Formal hard-safety result

| Controller | Success | Collision | Timeout / safe liveness failure | Mean steps |
|---|---:|---:|---:|---:|
| No safety | {formal['no_safety']['success']}/{total} | {formal['no_safety']['collision']}/{total} | {formal['no_safety']['timeout']}/{total} | {formal['no_safety']['mean_episode_length']:.2f} |
| Hard safety | {safe['success']}/{total} | {safe['collision']}/{total} | {safe['timeout']}/{total} | {safe['mean_episode_length']:.2f} |

Hard safety minimum recorded wall/obstacle and inter-agent clearances were {safe['min_wall_clearance']:.6f} and {safe['min_agent_clearance']:.6f}. Projection was active on a mean {pct(safe['projection_active_fraction_mean'])} of steps; mean correction norm was {safe['correction_norm_mean']:.6f}, with mean correction/Flow norm ratio {safe['correction_flow_ratio_mean']:.6f}. This strongly suppresses collision without rewriting the policy on essentially every timestep.

## 3. No-safety → safety conversion matrix

| Matched conversion | Episodes |
|---|---:|
{conversions}

## 4. Frozen safe-failure population

The target set was frozen before eta search and contains **{target_n} `SAFE_LIVENESS_FAILURE` states**: collision-free, unsuccessful, and terminated by timeout. The environments expose no independent runtime strict-deadlock detector, so none is relabelled as deadlock. State IDs are preserved in `formal_safety_summary.json`.

## 5. OrthoFlow3 definition and frozen eta domain

The controller is exactly `u_flow → Pi_U → u_safe + eta_g B_goal + eta_perp B_flow_perp + eta_rel B_rel → Pi_U`, with `B_flow_perp` the goal-orthogonal residual scaled by the frozen factor {frozen['orthoflow3_scale']}. The common domain is `[0.5,1.25] × [-0.5,0.5] × [0,0.75]`; both scenarios share the same 256-point scrambled Sobol designs. The second hard projection is mandatory.

## 6. Global 256-point search

The primary screen evaluated the common 256 points with logical seeds 0–3. It produced {q4_candidates} state/eta preliminary candidates at Q4≥3/4. Every 4/4 candidate and deterministically ranked 3/4 candidates were promoted according to the frozen rule; Q16 uses the fixed ≥15/16 robust threshold. Cache manifests and preflights are under `plans/` and `cache/`.

## 7. Secondary fixed search for negative states

After primary Q16 validation, {len(secondary)} negative states received the independent second 256-point Sobol design over the unchanged domain. Secondary search added {secondary_positive} robust-positive states. No range or representation change was made.

## 8. Robust eta existence

Robust eta exists for **{robust_n}/{target_n} = {pct(summary['robust_existence_fraction'])}** states. Primary search certified {primary_positive}; secondary search certified {secondary_positive}. The remaining {len(existence['no_robust_eta_observed'])} states are classified `NO_ROBUST_ETA_OBSERVED`, not mathematical impossibility.

Exact seed-level stopping evaluated {efficiency['canonical_seed_rollouts_evaluated']} canonical seed outcomes versus {efficiency['canonical_seed_rollouts_without_early_stop']} without logical stopping, saving {pct(efficiency['compute_saved_fraction'])}. Of {efficiency['tuple_count']} robust-decision tuples, {efficiency['fully_evaluated_to_16']} were evaluated to all 16 seeds, {efficiency['rejected_after_4']} were rejected after the first four seeds, {efficiency['rejected_later_after_second_failure']} were rejected later at the second observed failure, and {efficiency['confirmed_at_15_successes']} were accepted at 15 observed successes. Unrun seeds were neither fabricated nor stored as failures. These classifications are mathematically identical to full 15/16 membership evaluation; the saving is implementation efficiency, not a scientific result.

An additional {efficiency['numerical_incomplete_not_classified']} unique robust-membership tuples were not certified because one or more deterministic seed executions remained numerical-invalid. They are excluded from the early-stop saving denominator and are not imputed as failures.

## 9. Local robust-basin width

Each robust center received the common 32-point radius-0.05 local Sobol design, eight-seed screening, and fixed-index Q16 promotion for exactly 12 points. Median robust promoted fraction is **{np.median(local_values) if local_values else 0.0:.4f}** (range {min(local_values, default=0.0):.4f}–{max(local_values, default=0.0):.4f}).

## 10. Robust eta cross-state coverage

There are {len(coverage['centers'])} numerically unique center values. Best robust state coverage is **{max(cover_values, default=0.0):.4f}** and median is **{np.median(cover_values) if cover_values else 0.0:.4f}**. Counts at ≥10/25/50/75/100% are `{summary['coverage_threshold_counts']}`. Every membership decision uses Q16≥15/16, never a single seed.

## 11. Basin overlap / state dependence

Median pairwise robust-membership Jaccard is {summary['median_basin_jaccard']:.4f}; {pct(summary['zero_overlap_pair_fraction'])} of state pairs have zero overlap over the tested center library. These sampled memberships support the stated classification but do not prove continuous basin topology.

## 12. Success-control preservation

Up to 24 uniformly selected hard-safety-success controls were evaluated with eight fixed seeds for every unique center. Median center preservation probability is {summary['median_success_control_preservation']:.4f}; the full center distribution is in `success_control_preservation.json`. There were {control_numerical}/{control_planned} numerical-incomplete control evaluations; they are excluded from probability denominators, never imputed as failures. No center was selected or optimized using controls.

## 13. Behavioral mechanism

{mechanism_summary(runtime, mechanism)}

## 14. Final classification

`{summary['classification']}`: robust existence is {pct(summary['robust_existence_fraction'])}, and best shared-center coverage is {pct(summary['best_center_coverage'])}. The interpretation follows the predeclared existence-versus-shared-coverage distinction. Numerical execution audit: {audit['n']} exact controller records, {audit['numerical']} retained numerical-invalid records excluded from scientific outcomes, {audit['collision']} eta collisions, and {audit['foreign_key_violations']} database foreign-key violations.
"""
    (out / "REPORT.md").write_text(report)
    return summary


def render_combined(rows: dict[str, dict]) -> None:
    four, ring = rows["four_way_intersection"], rows["ring_exchange"]
    text = f"""# New benchmark OrthoFlow3 summary

All new-scenario values use the present frozen 60-state formal safety evaluation and the fixed ≥15/16 definition. The Double-Bottleneck row is a historical OrthoFlow3 reference with a different safe-failure population and earlier screening/local protocols; its single-seed 60/61 shared-point observation must not be read as Q16 cross-state coverage.

| Benchmark | Mechanism | Hard-safety success | Safe failures | Robust-eta existence | Best robust eta state coverage | Median local robust fraction | 3D classification |
|---|---|---:|---:|---:|---:|---:|---|
| Double-Bottleneck | ordering/shared resource | historical frozen reference | 61 | 61/61 OrthoFlow3 reference | 60/61 under original episode seeds; not present Q16 protocol | 1.000 at Q≥0.50 in its 8-seed local study; not present 15/16 rule | historical `USE-ORTHOFLOW3`; not reclassified here |
| Four-Way | cyclic crossing | {four['formal_hard_safety']['success']}/60 | {four['safe_failures']} | {four['robust_states']}/{four['safe_failures']} | {pct(four['best_center_coverage'])} | {four['median_local_robust_fraction']:.4f} | `{four['classification']}` |
| Ring Exchange | circulation/lateral | {ring['formal_hard_safety']['success']}/60 | {ring['safe_failures']} | {ring['robust_states']}/{ring['safe_failures']} | {pct(ring['best_center_coverage'])} | {ring['median_local_robust_fraction']:.4f} | `{ring['classification']}` |

The new rows are directly comparable to each other because they use the same eta domain, both common Sobol designs, seed policy, promotion rule, local design, coverage rule, and robust threshold. The historical Double-Bottleneck evidence is included only as requested context and is explicitly not numerically harmonized with this protocol.

No `G_phi` was trained, no eta dimension/domain was expanded, and OrthoFlow3 and hard safety were not redesigned.
"""
    (ROOT / "diagnostics/new_benchmarks_eta3_summary.md").write_text(text)


def main() -> None:
    rows = {name: render_scenario(name) for name in SCENARIOS}
    render_combined(rows)


if __name__ == "__main__":
    main()
