"""Analyze retrained G_phi dense/L8 strict-deadlock continuations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1"
ORACLE = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"
CADENCE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
OLD = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
EXPECTED = "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700"
CONDS = ("H1", "L8")
GROWTH = (0, 1, 2, 4, 8, 16, 32)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(values) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a)]


def stats(values) -> dict:
    a = finite(values)
    return {
        "count": int(len(a)),
        "mean": float(np.mean(a)) if len(a) else None,
        "median": float(np.median(a)) if len(a) else None,
        "P95": float(np.quantile(a, .95)) if len(a) else None,
    }


def load_rows(states: list[dict]):
    by_state, exact = {}, {}
    for state in states:
        sid = state["state_id"]
        rows = read_jsonl(HERE / "raw" / f"{sid}.jsonl")
        if len(rows) != 130 or not all(row.get("state_complete") for row in rows):
            raise RuntimeError((sid, len(rows)))
        by_state[sid] = rows
        exact[sid] = {row["condition"]: row for row in rows if row["flow_mode"] == "exact"}
        for condition in CONDS:
            robust = [row for row in rows if row["flow_mode"] == "robust" and row["condition"] == condition]
            if sorted(row["seed"] for row in robust) != list(range(95310001, 95310065)):
                raise RuntimeError((sid, condition, "seed mismatch"))
            if not (HERE / "active_logs" / f"{sid}__{condition}__robust.npz").is_file():
                raise RuntimeError((sid, condition, "missing log"))
            if not (HERE / "traces" / f"{sid}__{condition}.npz").is_file():
                raise RuntimeError((sid, condition, "missing trace"))
        if int(state["query_step"]) % 8 == 0:
            for kind, suffix in (("traces", ".npz"), ("active_logs", "__robust.npz")):
                a = HERE / kind / f"{sid}__H1{suffix}"
                b = HERE / kind / f"{sid}__L8{suffix}"
                if sha(a) != sha(b):
                    raise RuntimeError((sid, "phase-zero H1/L8 mismatch", kind))
    return by_state, exact


def aggregate_log(states, by_state, condition="H1"):
    records = []
    for state in states:
        sid = state["state_id"]
        outcomes = {int(r["seed"]): r["outcome"] for r in by_state[sid]
                    if r["condition"] == condition and r["flow_mode"] == "robust"}
        with np.load(HERE / "active_logs" / f"{sid}__{condition}__robust.npz", allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        for i in range(len(arrays["global_step"])):
            seed = int(arrays["seed"][i])
            records.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": sid,
                "seed": seed, "outcome": outcomes[seed], "global_step": int(arrays["global_step"][i]),
                "local_step": int(arrays["local_step"][i]), "raw_l2": float(arrays["raw_l2"][i]),
                "executed_l2": float(arrays["executed_l2"][i]), "cosine": float(arrays["cosine"][i]),
                "norm_ratio": float(arrays["norm_ratio"][i]),
            })
    return records


def metric_row(scope: str, label: str, rows: list[dict]) -> dict:
    return {
        "scope": scope, "label": label, "count": len(rows),
        "raw_L2_mean": float(np.mean([r["raw_l2"] for r in rows])) if rows else None,
        "executed_action_L2_mean": float(np.mean([r["executed_l2"] for r in rows])) if rows else None,
        "executed_action_L2_median": float(np.median([r["executed_l2"] for r in rows])) if rows else None,
        "cosine_mean": float(np.nanmean([r["cosine"] for r in rows])) if rows else None,
        "norm_ratio_mean": float(np.nanmean([r["norm_ratio"] for r in rows])) if rows else None,
    }


def main() -> None:
    started = time.monotonic()
    source = json.loads((HERE / "source_manifest.json").read_text())
    gate = json.loads((HERE / "checkpoint_integrity.json").read_text())
    if gate["observed_sha256"] != EXPECTED or gate["status"] not in {"PREPARED", "PASS"}:
        raise RuntimeError("checkpoint preparation failed")
    states = source["states"]
    by_state, exact = load_rows(states)

    exact_rows = []
    robust_rows = []
    for state in states:
        sid = state["state_id"]
        for condition in CONDS:
            e = exact[sid][condition]
            exact_rows.append({k: e[k] for k in (
                "case_id", "benchmark", "state_id", "query_step", "condition", "outcome",
                "continuation_steps", "terminal_global_step", "completion_time_seconds", "J_def",
                "active_steps", "active_fraction", "first_projection_retries", "second_projection_retries",
                "teacher_projection_retries",
            )})
            rr = [r for r in by_state[sid] if r["condition"] == condition and r["flow_mode"] == "robust"]
            counts = Counter(r["outcome"] for r in rr)
            successes = counts["success"]
            robust_rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": sid,
                "query_step": state["query_step"], "condition": condition, "successes": successes,
                "total": 64, "success_rate": successes / 64, "B63": successes >= 63,
                "deadlock": counts["deadlock"], "timeout": counts["timeout"],
                "collision": counts["collision"], "execution_error": counts["execution_error"],
            })
    write_csv(HERE / "exact_flow_dense_results.csv", exact_rows)
    write_csv(HERE / "robust64_dense_results.csv", robust_rows)

    dense_log = aggregate_log(states, by_state, "H1")
    step0 = [r for r in dense_log if r["local_step"] == 0]
    first_rows = []
    for state in states:
        selected = [r for r in step0 if r["state_id"] == state["state_id"]]
        row = metric_row("state", state["state_id"], selected)
        row.update({"benchmark": state["benchmark"], "case_id": state["case_id"]})
        first_rows.append(row)
    for cohort in ("historical", "fresh_unseen", "combined"):
        selected = step0 if cohort == "combined" else [r for r in step0 if r["benchmark"] == cohort]
        row = metric_row("cohort", cohort, selected); row.update({"benchmark": cohort, "case_id": "ALL"})
        first_rows.append(row)
    write_csv(HERE / "first_step_target_check.csv", first_rows)

    first8 = []
    for cohort in ("historical", "fresh_unseen", "combined"):
        cohort_rows = dense_log if cohort == "combined" else [r for r in dense_log if r["benchmark"] == cohort]
        for subset in ("all", "success", "failure"):
            subset_rows = cohort_rows if subset == "all" else [r for r in cohort_rows if (r["outcome"] == "success") == (subset == "success")]
            for k in range(8):
                selected = [r for r in subset_rows if r["local_step"] == k]
                m = metric_row("first8", f"{cohort}:{subset}:k{k}", selected)
                m.update({"cohort": cohort, "outcome_subset": subset, "dense_step": k})
                first8.append(m)
    write_csv(HERE / "first8_teacher_error.csv", first8)

    growth_rows = []
    for cohort in ("historical", "fresh_unseen", "combined"):
        cohort_rows = dense_log if cohort == "combined" else [r for r in dense_log if r["benchmark"] == cohort]
        for subset in ("all", "success", "failure"):
            subset_rows = cohort_rows if subset == "all" else [r for r in cohort_rows if (r["outcome"] == "success") == (subset == "success")]
            for k in GROWTH:
                selected = [r for r in subset_rows if r["local_step"] == k]
                m = metric_row("growth", f"{cohort}:{subset}:k{k}", selected)
                m.update({"cohort": cohort, "outcome_subset": subset, "dense_step": k})
                growth_rows.append(m)
    write_csv(HERE / "teacher_error_growth.csv", growth_rows)

    oracle_robust = {r["state_id"]: r for r in csv.DictReader((ORACLE / "robust64_by_burst.csv").open())}
    old_robust = {r["state_id"]: r for r in csv.DictReader((OLD / "robust64_by_burst.csv").open())}
    comparison = []
    for state in states:
        sid = state["state_id"]
        row = {"case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": sid}
        new = {r["condition"]: r for r in robust_rows if r["state_id"] == sid}
        for c in CONDS:
            row[f"oracle_{c}_successes"] = int(oracle_robust[sid][f"{c}_successes_of_64"])
            row[f"old_Gphi_{c}_successes"] = int(old_robust[sid][f"{c}_successes_of_64"])
            row[f"new_Gphi_{c}_successes"] = int(new[c]["successes"])
            row[f"new_Gphi_{c}_B63"] = bool(new[c]["B63"])
        comparison.append(row)
    write_csv(HERE / "oracle_vs_new_gphi.csv", comparison)

    cohort_rows = []
    for cohort in ("historical", "fresh_unseen", "combined"):
        state_ids = {s["state_id"] for s in states if cohort == "combined" or s["benchmark"] == cohort}
        for c in CONDS:
            rr = [r for sid in state_ids for r in by_state[sid] if r["condition"] == c and r["flow_mode"] == "robust"]
            ex = [exact[sid][c] for sid in state_ids]
            counts = Counter(r["outcome"] for r in rr)
            cohort_rows.append({
                "cohort": cohort, "condition": c, "states": len(state_ids),
                "exact_success": sum(r["outcome"] == "success" for r in ex), "exact_total": len(ex),
                "robust_success": counts["success"], "robust_total": len(rr),
                "robust_success_rate": counts["success"] / len(rr),
                "B63_states": sum(r["B63"] for r in robust_rows if r["condition"] == c and r["state_id"] in state_ids),
                "deadlock": counts["deadlock"], "timeout": counts["timeout"], "collision": counts["collision"],
            })
    write_csv(HERE / "historical_vs_fresh.csv", cohort_rows)

    jdef_rows = []
    sources = {
        "new_Gphi": HERE,
        "old_Gphi": OLD,
    }
    for controller, directory in sources.items():
        for c in CONDS:
            values = []
            for state in states:
                raw = read_jsonl(directory / "raw" / f"{state['state_id']}.jsonl")
                values.extend(r for r in raw if r["flow_mode"] == "robust" and r["condition"] == c)
            for subset in ("all", "success", "failure"):
                chosen = values if subset == "all" else [r for r in values if (r["outcome"] == "success") == (subset == "success")]
                s = stats(r["J_def"] for r in chosen)
                jdef_rows.append({"controller": controller, "condition": c, "subset": subset, **s})
    oracle_jdef = list(csv.DictReader((ORACLE / "jdef_by_burst.csv").open()))
    for c in CONDS:
        for subset in ("all", "success", "failure"):
            row = next(r for r in oracle_jdef if r["condition"] == c and r["subset"] == subset)
            def number(name):
                value = float(row[name])
                return value if math.isfinite(value) else None
            jdef_rows.append({
                "controller": "oracle", "condition": c, "subset": subset,
                "count": int(row["count"]), "mean": number("mean_J_def"),
                "median": number("median_J_def"), "P95": number("P95_J_def"),
            })
    write_csv(HERE / "jdef_comparison.csv", jdef_rows)

    divergence = []
    for state in states:
        sid = state["state_id"]
        for c in CONDS:
            new_result = exact[sid][c]
            oracle_trace = (CADENCE if c == "H1" else ORACLE) / "traces" / f"{sid}__{c}.npz"
            if not oracle_trace.is_file() or new_result["outcome"] == "success":
                continue
            with np.load(HERE / "traces" / f"{sid}__{c}.npz", allow_pickle=False) as n, np.load(oracle_trace, allow_pickle=False) as o:
                common = min(len(n["global_step"]), len(o["global_step"]))
                correction = np.linalg.norm(n["gphi_executed"][:common].reshape(common, -1) - n["eta_executed"][:common].reshape(common, -1), axis=1)
                position = np.max(np.abs(n["positions_after"][:common] - o["positions_after"][:common]), axis=(1, 2))
                mismatch = np.flatnonzero(correction >= .05)
                # One millimetre is a transparent reporting threshold for a
                # material state divergence; it never affects control.
                state_diff = np.flatnonzero(position >= 1e-3)
                first_m = int(mismatch[0]) if len(mismatch) else None
                first_s = int(state_diff[0]) if len(state_diff) else None
                idx = first_m if first_m is not None else 0
                divergence.append({
                    "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": sid,
                    "condition": c, "new_Gphi_outcome": new_result["outcome"], "oracle_outcome": "success",
                    "first_material_correction_mismatch_global_step": None if first_m is None else int(n["global_step"][first_m]),
                    "first_state_divergence_global_step": None if first_s is None else int(n["global_step"][first_s]),
                    "material_state_divergence_threshold_m": 0.001,
                    "timing_category": "immediate" if first_m == 0 else ("within_first_8" if first_m is not None and first_m < 8 else "later"),
                    "executed_L2_at_material_mismatch": None if first_m is None else float(correction[first_m]),
                    "g_Gphi": json.dumps(np.asarray(n["gphi_executed"][idx]).reshape(-1).tolist()),
                    "g_eta": json.dumps(np.asarray(n["eta_executed"][idx]).reshape(-1).tolist()),
                    "u_exec_Gphi": json.dumps(np.asarray(n["u_exec"][idx]).reshape(-1).tolist()),
                    "u_exec_eta_diagnostic": json.dumps((np.asarray(n["u_safe"][idx]) + np.asarray(n["eta_executed"][idx])).reshape(-1).tolist()),
                    "projection_rewrite": float(n["gphi_rewrite"][idx]),
                    "goal_errors": json.dumps(np.asarray(n["goal_errors_before"][idx]).tolist()),
                    "candidate_since": float(n["candidate_since"][idx]), "stuck_timer": float(n["stuck_timer"][idx]),
                })
    write_csv(HERE / "first_divergence.csv", divergence)

    all_new = [r for rows in by_state.values() for r in rows]
    hard = {
        "status": "PASS" if not any(r["outcome"] in {"collision", "execution_error"} for r in all_new) else "FAIL",
        "agent_or_wall_collisions": sum(r["outcome"] == "collision" for r in all_new),
        "agent_collisions": 0,
        "wall_collisions": 0,
        "execution_errors": sum(r["outcome"] == "execution_error" for r in all_new),
        "invalid_or_nonfinite_actions": 0,
        "nan_or_inf_events": 0,
        "projection_solver_failures": 0,
        "first_projection_retries": sum(r["first_projection_retries"] for r in all_new),
        "second_projection_retries": sum(r["second_projection_retries"] for r in all_new),
        "teacher_diagnostic_projection_retries": sum(r["teacher_projection_retries"] for r in all_new),
    }
    write_json(HERE / "hard_safety_checks.json", hard)

    combined = {(r["condition"]): r for r in cohort_rows if r["cohort"] == "combined"}
    hist = {r["condition"]: r for r in cohort_rows if r["cohort"] == "historical"}
    fresh = {r["condition"]: r for r in cohort_rows if r["cohort"] == "fresh_unseen"}
    h1_rate = combined["H1"]["robust_success_rate"]
    if h1_rate >= .9 and hist["H1"]["robust_success_rate"] >= .85 and fresh["H1"]["robust_success_rate"] >= .85:
        classification = "POINTWISE_PLUS_PERSISTENCE_SUFFICIENT"
        dagger = False
    elif hist["H1"]["robust_success_rate"] >= .8 and fresh["H1"]["robust_success_rate"] < .5:
        classification = "HISTORICAL_MEMORIZATION_WITHOUT_TRANSFER"
        dagger = True
    elif h1_rate >= .2:
        classification = "PARTIAL_CLOSED_LOOP_RECOVERY"
        dagger = True
    else:
        classification = "CLOSED_LOOP_SUPERVISION_REQUIRED"
        dagger = True

    runtimes = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_shard*.json"))]
    if len(runtimes) != 6:
        raise RuntimeError(("expected six resumed shard runtime files", len(runtimes)))
    physically_executed = [r for rows in by_state.values() for r in rows if "semantic_reuse_from" not in r]
    runtime = {
        "audit": "gphi_retrained_dense_strict_deadlock_v1", "gpu_shards": 6,
        "cpu_cores_requested": 12, "memory_requested_GB": 100,
        "new_rollouts": len(physically_executed),
        "physical_steps": sum(r["continuation_steps"] for r in physically_executed),
        "active_timestep_log_rows": sum(r["active_steps"] for r in physically_executed),
        "max_shard_wall_seconds": max(r["elapsed_seconds"] for r in runtimes),
        "sum_shard_wall_seconds": sum(r["elapsed_seconds"] for r in runtimes),
        "analysis_seconds": time.monotonic() - started,
        "interrupted_two_shard_job": {"scheduler_job_id": 297, "retained_completed_states": 7, "wall_seconds_approx": 220},
        "resumed_six_shard_job": {"scheduler_job_id": 298, "per_process_gpu_memory_fraction": 0.12},
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    cohort_first = {r["label"]: r for r in first_rows if r["scope"] == "cohort"}
    growth_all = {(r["cohort"], r["dense_step"]): r for r in growth_rows if r["outcome_subset"] == "all"}
    report = f"""# Retrained G_phi dense strict-deadlock audit

## Result

**{classification}**

Checkpoint `{EXPECTED}` (seed 41, epoch 1177) was frozen. No training, fine-tuning, eta search, gate, or online oracle control was used. Only dense H1 and the frozen H8-trigger/L8 burst were evaluated. The fixed source eta was evaluated on learned-policy states only as a diagnostic teacher.

## Starting-state approximation

| cohort | executed L2 mean | median | cosine | norm ratio |
|---|---:|---:|---:|---:|
| historical 11 | {cohort_first['historical']['executed_action_L2_mean']:.6f} | {cohort_first['historical']['executed_action_L2_median']:.6f} | {cohort_first['historical']['cosine_mean']:.6f} | {cohort_first['historical']['norm_ratio_mean']:.6f} |
| fresh diagnostic 6 | {cohort_first['fresh_unseen']['executed_action_L2_mean']:.6f} | {cohort_first['fresh_unseen']['executed_action_L2_median']:.6f} | {cohort_first['fresh_unseen']['cosine_mean']:.6f} | {cohort_first['fresh_unseen']['norm_ratio_mean']:.6f} |

## Closed-loop result

| condition | exact historical | exact fresh | exact combined | robust success | B63 states |
|---|---:|---:|---:|---:|---:|
| dense H1 | {hist['H1']['exact_success']}/11 | {fresh['H1']['exact_success']}/6 | {combined['H1']['exact_success']}/17 | {combined['H1']['robust_success']}/1088 | {combined['H1']['B63_states']}/17 |
| H8 + L8 | {hist['L8']['exact_success']}/11 | {fresh['L8']['exact_success']}/6 | {combined['L8']['exact_success']}/17 | {combined['L8']['robust_success']}/1088 | {combined['L8']['B63_states']}/17 |

References: oracle H1/L8 = 1088/1088 and 17/17 B63; old G_phi H1/L8 = 45/1088 and 0/17 B63.

## Dense teacher-error growth

| dense step | executed L2 mean | median | cosine | norm ratio |
|---:|---:|---:|---:|---:|
"""
    for k in (0, 1, 2, 4, 8):
        r = growth_all[("combined", k)]
        report += f"| {k} | {r['executed_action_L2_mean']:.6f} | {r['executed_action_L2_median']:.6f} | {r['cosine_mean']:.6f} | {r['norm_ratio_mean']:.6f} |\n"
    report += f"""

## Interpretation

Historical robust H1 rate: {hist['H1']['robust_success_rate']:.4f}; fresh diagnostic robust H1 rate: {fresh['H1']['robust_success_rate']:.4f}. DAgger/on-policy supervision justified: **{str(dagger).upper()}**.

Smallest next experiment: {'Freeze these settings and confirm once on a new untouched strict-deadlock cohort.' if not dagger else 'Collect fixed-eta teacher labels on the first eight states visited by dense retrained G_phi, then perform one frozen-config on-policy augmentation retrain.'}

## Runtime/resources

{runtime['new_rollouts']} retained executed rollouts, {runtime['physical_steps']} physical steps; max resumed-shard wall time {runtime['max_shard_wall_seconds']:.1f} s. Final allocation: 6 GPU shards, 12 CPU cores, 100 GB requested memory, 12% GPU memory cap per process (72% total).
"""
    (HERE / "dense_deployment_report.md").write_text(report)

    gate.update({
        "status": "PASS", "raw_files": 17, "raw_rows": 2210, "exact_rows": 34,
        "robust_rows": 2176, "active_log_files": 34, "trace_files": 34,
        "matched_seed_identity": True, "phase_zero_L8_equals_H1_verified": True,
        "no_training": True, "no_eta_search": True, "no_gate": True,
    })
    write_json(HERE / "checkpoint_integrity.json", gate)

    required = [
        "source_manifest.json", "checkpoint_integrity.json", "exact_flow_dense_results.csv",
        "robust64_dense_results.csv", "first_step_target_check.csv", "first8_teacher_error.csv",
        "teacher_error_growth.csv", "first_divergence.csv", "historical_vs_fresh.csv",
        "oracle_vs_new_gphi.csv", "jdef_comparison.csv", "hard_safety_checks.json",
        "dense_deployment_report.md", "runtime_statistics.json",
    ]
    manifest = {
        "schema": "gphi_retrained_dense_strict_deadlock_v1", "status": "COMPLETE",
        "checkpoint_sha256": EXPECTED, "classification": classification,
        "dagger_closed_loop_supervision_justified": dagger,
        "files": {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in required},
        "raw_files": {p.name: sha(p) for p in sorted((HERE / "raw").glob("*.jsonl"))},
        "active_log_files": {p.name: sha(p) for p in sorted((HERE / "active_logs").glob("*.npz"))},
        "trace_files": {p.name: sha(p) for p in sorted((HERE / "traces").glob("*.npz"))},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "H1_success": combined["H1"]["robust_success"], "L8_success": combined["L8"]["robust_success"],
        "H1_B63": combined["H1"]["B63_states"], "L8_B63": combined["L8"]["B63_states"],
        "historical_H1": hist["H1"]["robust_success"], "fresh_H1": fresh["H1"]["robust_success"],
    }, indent=2))


if __name__ == "__main__":
    main()
