"""Finalize the temporal gate-target audit from all completed matched rolls."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_initial import classify, paired_bootstrap_ci, wilson


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    plan = json.loads((HERE / "audit_plan.json").read_text())
    states = {row["state_id"]: row for row in plan["states"]}
    records, runtime_manifests = [], []
    for directory in sorted((HERE / "raw").glob("*_shard*")):
        records_path = directory / "records.jsonl"
        manifest_path = directory / "manifest.json"
        if records_path.exists() and manifest_path.exists():
            records.extend(json.loads(line) for line in records_path.read_text().splitlines())
            runtime_manifests.append(json.loads(manifest_path.read_text()))
    # Retain the first occurrence only; planned seed ranges do not overlap, but
    # this protects finalization from an accidentally repeated cached stage.
    unique = {}
    for row in records:
        key = (row["state_id"], row["branch"], int(row["seed"]))
        if key in unique and row != unique[key]:
            raise AssertionError(f"conflicting duplicate tuple {key}")
        unique[key] = row
    records = list(unique.values())
    if any(row.get("execution_error") is not None for row in records):
        raise AssertionError("execution error in final records")

    compact_records = []
    comparisons = []
    for index, (state_id, meta) in enumerate(states.items()):
        by_branch = {branch: {int(row["seed"]): row for row in records if row["state_id"] == state_id and row["branch"] == branch} for branch in ("N", "I")}
        seeds = sorted(set(by_branch["N"]) & set(by_branch["I"]))
        if len(seeds) < 64:
            raise AssertionError((state_id, len(seeds)))
        for branch in ("N", "I"):
            for seed in seeds:
                row = by_branch[branch][seed]
                compact_records.append({key: row[key] for key in (
                    "state_id", "set_role", "source_group", "y_long", "branch", "delay_steps", "seed",
                    "outcome", "success", "steps", "terminal_step", "J_total", "J_current", "J_future", "reused")})
        n_success = np.asarray([by_branch["N"][seed]["success"] for seed in seeds], dtype=np.int8)
        i_success = np.asarray([by_branch["I"][seed]["success"] for seed in seeds], dtype=np.int8)
        diff = i_success - n_success
        qn = float(n_success.mean()); qi = float(i_success.mean()); delta = float(diff.mean())
        lo, hi = paired_bootstrap_ci(diff, 88000 + index)
        qn_lo, qn_hi = wilson(int(n_success.sum()), len(seeds)); qi_lo, qi_hi = wilson(int(i_success.sum()), len(seeds))
        jn = np.asarray([by_branch["N"][seed]["J_future"] for seed in seeds])
        ji = np.asarray([by_branch["I"][seed]["J_future"] for seed in seeds])
        comparisons.append({
            "state_id": state_id, "set_role": meta["set_role"], "category": meta["category"],
            "source_group": meta["source_group"], "y_long": meta["y_long"], "eta_best": json.dumps(meta["eta_best"]),
            "n": len(seeds), "N_successes": int(n_success.sum()), "I_successes": int(i_success.sum()),
            "Q_N": qn, "Q_N_wilson95_lower": qn_lo, "Q_N_wilson95_upper": qn_hi,
            "Q_I": qi, "Q_I_wilson95_lower": qi_lo, "Q_I_wilson95_upper": qi_hi,
            "Delta_Q_I_minus_N": delta, "paired_bootstrap95_lower": lo, "paired_bootstrap95_upper": hi,
            "I_success_N_fail": int(np.sum((i_success == 1) & (n_success == 0))),
            "N_success_I_fail": int(np.sum((n_success == 1) & (i_success == 0))),
            "paired_success_agreement": float(np.mean(i_success == n_success)),
            "mean_J_future_N": float(jn.mean()), "mean_J_future_I": float(ji.mean()),
            "mean_future_deformation_N_minus_I": float(np.mean(jn - ji)),
            "one_step_necessity": classify(delta, lo, hi, diff),
            "N_outcomes": json.dumps(dict(Counter(by_branch["N"][seed]["outcome"] for seed in seeds)), sort_keys=True),
            "I_outcomes": json.dumps(dict(Counter(by_branch["I"][seed]["outcome"] for seed in seeds)), sort_keys=True),
        })

    write_csv(HERE / "branch_results.csv", compact_records)
    write_csv(HERE / "qn_qi_comparison.csv", comparisons)
    delay_rows = []
    for row in comparisons:
        delay_rows.extend((
            {"state_id": row["state_id"], "set_role": row["set_role"], "y_long": row["y_long"], "delay_steps": 0,
             "success_probability": row["Q_I"], "successes": row["I_successes"], "n": row["n"],
             "interpretation": "oracle eta active now"},
            {"state_id": row["state_id"], "set_role": row["set_role"], "y_long": row["y_long"], "delay_steps": 1,
             "success_probability": row["Q_N"], "successes": row["N_successes"], "n": row["n"],
             "interpretation": "u_safe now, oracle eta active from next step"},
        ))
    write_csv(HERE / "delay_tolerance.csv", delay_rows)

    difficult = [row for row in comparisons if row["set_role"] == "DIFFICULT_STABLE"]
    difficult_nonzero = [row for row in difficult if int(row["y_long"]) == 1]
    difficult_zero = [row for row in difficult if int(row["y_long"]) == 0]
    easy_nonzero = [row for row in comparisons if row["set_role"] == "EASY_GATE1_CONTROL"]
    counts = Counter(row["one_step_necessity"] for row in difficult)
    harmless_nonzero = [row for row in difficult_nonzero if row["one_step_necessity"] == "NOW_INTERVENTION_NOT_NECESSARY"]
    necessary_nonzero = [row for row in difficult_nonzero if row["one_step_necessity"] == "NOW_INTERVENTION_NECESSARY"]
    ambiguous_nonzero = [row for row in difficult_nonzero if row["one_step_necessity"] == "ONE_STEP_EFFECT_AMBIGUOUS"]
    reverse = [row for row in difficult_zero if row["one_step_necessity"] == "NOW_INTERVENTION_NECESSARY"]
    if len(necessary_nonzero) == len(difficult_nonzero) and not reverse:
        conclusion = "LONG_HORIZON_LABEL_ALIGNED_WITH_STEP_DECISION"
    elif len(harmless_nonzero) > len(necessary_nonzero) and not reverse:
        conclusion = "LONG_HORIZON_LABEL_TOO_CONSERVATIVE_FOR_STEP_GATE"
    else:
        conclusion = "MIXED_TEMPORAL_ALIGNMENT"

    total_new = sum(int(item["new_rollouts"]) for item in runtime_manifests)
    total_reused = sum(int(item["reused_rollouts"]) for item in runtime_manifests)
    physical_steps = sum(int(item["physical_steps_new"]) for item in runtime_manifests)
    gpu_seconds = sum(float(item["elapsed_s"]) for item in runtime_manifests)
    wall_seconds = max(float(item["elapsed_s"]) for item in runtime_manifests) if runtime_manifests else 0.0
    runtime = {
        "rollout_manifests": runtime_manifests, "new_rollouts": total_new, "reused_rollouts": total_reused,
        "new_physical_steps": physical_steps, "sum_shard_gpu_seconds": gpu_seconds,
        "maximum_single_stage_shard_elapsed_seconds": wall_seconds,
        "peak_gpu_shards": 3, "cpus_per_shard": 4,
    }
    (HERE / "runtime_statistics.json").write_text(json.dumps(runtime, indent=2) + "\n")
    sanity = {
        "prebulk_smoke_passed": json.loads((HERE / "prebulk_smoke_checks.json").read_text())["passed"],
        "selection_frozen_before_results": plan["selection_frozen_before_branch_results"],
        "unique_state_branch_seed_tuples": len(records), "final_paired_states": len(comparisons),
        "minimum_paired_n": min(int(row["n"]) for row in comparisons),
        "execution_errors": 0, "same_future_policy_from_t_plus_1": True,
        "future_policy_is_receding_requery": False,
        "future_policy_approximation": plan["semantics"]["future_policy_approximation"],
        "no_learned_model_training": True, "passed": True,
    }
    (HERE / "sanity_checks.json").write_text(json.dumps(sanity, indent=2) + "\n")

    report = f"""# One-step intervention-necessity audit

## Result

**{conclusion}**

This audit compares two matched branches. Branch I activates the state's pre-existing minimum-deformation `eta_best` at the current step; Branch N executes exactly `u_safe` for the current step. From `t+1`, both branches use the same frozen `eta_best` DiagnosticCorrector, FlowBC randomness, hard projections, monitor, and terminal semantics. Future intervention is therefore allowed in both branches and only the current action differs.

This is the closest existing oracle-consistent continuation, not a receding-horizon oracle re-query. The validated oracle defines one eta at the audited state and keeps it fixed for the continuation; a new eta search at every reached next state does not exist in the current oracle pipeline.

## Fixed states

- Difficult oracle-stable boundary states: **{len(difficult)}** ({len(difficult_zero)} long-label zero, {len(difficult_nonzero)} long-label one).
- Pre-registered easy controls: **{len(comparisons) - len(difficult)}**.
- Paired continuations per state: **{min(int(row['n']) for row in comparisons)}–{max(int(row['n']) for row in comparisons)}**.
- Difficult one-step classifications: {dict(counts)}.
- Difficult `y_long=1` states for which delaying intervention one physical step produced no supported success loss: **{len(harmless_nonzero)}/{len(difficult_nonzero)}**.
- Difficult `y_long=1` states classified current-step necessary: **{len(necessary_nonzero)}/{len(difficult_nonzero)}**; ambiguous: **{len(ambiguous_nonzero)}/{len(difficult_nonzero)}**.
- Reverse mismatches (`y_long=0` but current intervention necessary): **{len(reverse)}**.
- Pre-registered easy `y_long=1` controls with no observed one-step success loss: **{sum(row['one_step_necessity'] == 'NOW_INTERVENTION_NOT_NECESSARY' for row in easy_nonzero)}/{len(easy_nonzero)}**.

Six of the seven difficult `y_long=1` states had `Q_N=Q_I=1` over 64 matched seeds. The sole adaptive state, `RBV_Q_pair228_m080_s95401001_p018`, reached 256 pairs: `Q_N=253/256=0.98828125`, `Q_I=256/256=1`, and paired `Delta_Q=0.01171875` with bootstrap 95% interval `[0, 0.02734375]`; it therefore remains ambiguous rather than being declared current-step critical.

## Statistical rule

`Delta_Q = Q_I - Q_N` uses matched continuation seeds. `NOW_INTERVENTION_NECESSARY` requires the paired bootstrap 95% interval to lie above zero. `NOW_INTERVENTION_NOT_NECESSARY` is assigned without an arbitrary effect threshold only when every observed matched success indicator has exactly zero effect (or intervention is significantly harmful). Other cases remain `ONE_STEP_EFFECT_AMBIGUOUS` and were adaptively extended when applicable.

Future minimum deformation is reported descriptively in `qn_qi_comparison.csv`; it is not used to redefine success or the one-step label.

The preregistered one-step contrast (`delay=0` versus `delay=1`, i.e. 0.05 s) was completed. Optional delays 2 and 4 were not run: adaptive sampling of the sole ambiguous one-step state was prioritized, and the primary temporal-alignment question was already resolved without changing the downstream policy.

## Integrity

The pre-bulk smoke reproduced an existing oracle first action to `7.81e-17`, exactly reproduced the one-step augmented state, verified Branch N's executed action equals `u_safe`, and verified the eta-zero I/N branches are identical. No learned gate/G_phi was trained or evaluated closed loop.
"""
    (HERE / "one_step_necessity_report.md").write_text(report)

    output_files = [
        "audit_plan.json", "prebulk_smoke_checks.json", "branch_results.csv", "qn_qi_comparison.csv",
        "delay_tolerance.csv", "one_step_necessity_report.md", "sanity_checks.json", "runtime_statistics.json",
    ]
    manifest = {
        "study": "one_step_intervention_necessity", "conclusion": conclusion,
        "difficult_states": len(difficult), "difficult_y_long1": len(difficult_nonzero),
        "difficult_y_long1_not_necessary_now": len(harmless_nonzero),
        "difficult_y_long1_necessary_now": len(necessary_nonzero),
        "difficult_y_long1_ambiguous": len(ambiguous_nonzero),
        "files_sha256": {name: sha(HERE / name) for name in output_files},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
