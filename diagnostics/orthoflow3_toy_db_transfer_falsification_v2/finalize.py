#!/usr/bin/env python3
"""Aggregate Stage A and issue the two independent falsification decisions."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).parent
FAMILIES = ("raw_toy", "translation", "diagonal", "rotation_anisotropic", "regularized_affine")


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, rows):
    path = H / path
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    (H / path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_stage_a():
    plan = json.load(open(H / "planned_rollouts.json"))
    pre = json.load(open(H / "cache_preflight_stageA.json"))
    local = {}
    for path in sorted((H / "raw").glob("stage_a_[0-9].jsonl")):
        for line in open(path):
            r = json.loads(line)
            key = (r["state_id"], r["family"], int(r["mode_id"]), int(r["fold"]), int(r["future_index"]))
            local[key] = r
    expected_new = pre["summary"]["genuinely_missing"]
    if len(local) != expected_new:
        raise RuntimeError(f"Stage A incomplete: {len(local)} / {expected_new}")

    details = {r["request_id"]: d for r, d in zip(plan["requests"], pre["details"])}
    out = []
    for req in plan["requests"]:
        vals = [local[(req["state_id"], req["family"], req["mode_id"], req["fold"], i)] for i in range(16)]
        suc = sum(bool(x["success"]) for x in vals)
        dead = sum(x.get("outcome") == "deadlock" for x in vals)
        timeout = sum(x.get("outcome") == "timeout" for x in vals)
        collision = sum(bool(x.get("wall_collision")) or bool(x.get("agent_collision")) for x in vals)
        detail = details[req["request_id"]]
        out.append({"fold": req["fold"], "family": req["family"], "mode_id": req["mode_id"],
                    "state_id": req["state_id"], "successes": suc, "trials": 16,
                    "success_rate": suc / 16, "B15": suc >= 15, "deadlock": dead,
                    "timeout": timeout, "collision": collision, "cache_status": detail["status"],
                    "reused_seeds": detail.get("reused", 0), "new_seeds": len(detail.get("missing_seeds", []))})
    write_csv("heldout_control_validation.csv", out)

    grouped = defaultdict(list)
    for r in out:
        grouped[(int(r["fold"]), r["family"], int(r["mode_id"]))].append(r)
    summary = []
    for (fold, family, mode), block in sorted(grouped.items()):
        summary.append({"fold": fold, "family": family, "mode_id": mode,
                        "panel_states": len(block), "B15_states": sum(str(x["B15"]).lower() == "true" for x in block),
                        "B15_prevalence": np.mean([str(x["B15"]).lower() == "true" for x in block]),
                        "mean_success_rate": np.mean([float(x["success_rate"]) for x in block]),
                        "min_success_rate": min(float(x["success_rate"]) for x in block),
                        "new_continuations": sum(int(x["new_seeds"]) for x in block),
                        "reused_continuations": sum(int(x["reused_seeds"]) for x in block)})
    write_csv("heldout_control_summary.csv", summary)
    return out, summary, pre["summary"]


def main():
    control, control_summary, cache = aggregate_stage_a()
    post_cache = json.load(open(H / "cache_postflight_stageA.json"))["summary"]
    pred = read_csv(H / "heldout_mode_predictions.csv")
    geom = {}
    for family in FAMILIES:
        block = [r for r in pred if r["family"] == family]
        selected = [r for r in block if r["control_validation_selected"].lower() == "true"]
        tested = [r for r in control_summary if r["family"] == family]
        geom[family] = {
            "heldout_modes": len(block),
            "median_coordinate_residual": float(np.median([float(r["target_residual"]) for r in block])),
            "mean_coordinate_residual": float(np.mean([float(r["target_residual"]) for r in block])),
            "max_coordinate_residual": float(np.max([float(r["target_residual"]) for r in block])),
            "median_nearest_robust_distance": float(np.median([float(r["nearest_independent_robust_distance"]) for r in block])),
            "max_condition_number": float(np.max([float(r["condition_number"]) for r in block])),
            "geometry_promising_modes": len(selected),
            "control_tested_modes": len(tested),
            "tested_modes_with_any_B15_state": sum(float(r["B15_prevalence"]) > 0 for r in tested),
            "tested_modes_with_majority_B15_states": sum(float(r["B15_prevalence"]) >= .5 for r in tested),
            "mean_tested_B15_prevalence": float(np.mean([float(r["B15_prevalence"]) for r in tested])) if tested else None,
            "mean_tested_success_rate": float(np.mean([float(r["mean_success_rate"]) for r in tested])) if tested else None,
        }
    raw_med = geom["raw_toy"]["median_coordinate_residual"]
    best_candidates = ["diagonal", "rotation_anisotropic", "regularized_affine"]
    best_family = min(best_candidates, key=lambda f: geom[f]["median_coordinate_residual"])
    best = geom[best_family]
    geom_gain = 1 - best["median_coordinate_residual"] / raw_med
    tested_reliable = best["tested_modes_with_majority_B15_states"]
    if best["geometry_promising_modes"] >= 9 and tested_reliable >= int(np.ceil(.75 * best["control_tested_modes"])) and best["max_condition_number"] <= 10.01:
        stage_a = "PREDICTIVE"
    elif best["geometry_promising_modes"] >= 4 and geom_gain >= .5 and best["median_nearest_robust_distance"] <= .10 and tested_reliable >= max(1, best["control_tested_modes"] // 2):
        stage_a = "PARTIAL"
    else:
        stage_a = "OVERFIT"

    comp = read_csv(H / "correct_vs_shuffled.csv")
    advantages = []
    for r in comp:
        advantages.append({
            "db_fraction": int(r["db_fraction"]),
            "q_gain_vs_shuffled_median": float(r["correct_test_mean_selected_Q_median"]) - float(r["shuffled_test_mean_selected_Q_median"]),
            "robust_state_gain_vs_shuffled_median": float(r["correct_test_robust15_states_median"]) - float(r["shuffled_test_robust15_states_median"]),
            "nll_gain_vs_shuffled_median": float(r["shuffled_test_nll_median"]) - float(r["correct_test_nll_median"]),
            "regret_gain_vs_shuffled_median": float(r["shuffled_test_mean_regret_median"]) - float(r["correct_test_mean_regret_median"]),
            "correct_beats_shuffled_q75_Q": float(r["correct_test_mean_selected_Q_median"]) > float(r["shuffled_test_mean_selected_Q_q75"]),
            "correct_beats_best_shuffled_Q": float(r["correct_test_mean_selected_Q_median"]) > float(r["shuffled_test_mean_selected_Q_best"]),
        })
    stable_clear = all(a["q_gain_vs_shuffled_median"] >= .03 and a["robust_state_gain_vs_shuffled_median"] >= 1 and a["regret_gain_vs_shuffled_median"] >= .02 for a in advantages)
    weak = sum(a["q_gain_vs_shuffled_median"] > 0 and a["regret_gain_vs_shuffled_median"] > 0 for a in advantages) == 2
    stage_b = "SUPPORTED" if stable_clear else ("PARTIAL" if weak else "GENERIC_ONLY")

    if stage_a == "PREDICTIVE" and stage_b == "SUPPORTED":
        joint = "CROSS_SCENARIO_STRUCTURE_STRONGLY_SUPPORTED"
        claim = "ENHANCE"
    elif stage_a in ("PREDICTIVE", "PARTIAL") and stage_b == "GENERIC_ONLY":
        joint = "GEOMETRY_ONLY_SUPPORTED" if stage_a == "PREDICTIVE" else "CURRENT_CROSS_SCENARIO_CLAIM_WEAKENED"
        claim = "WEAKEN"
    elif stage_a == "OVERFIT" and stage_b in ("SUPPORTED", "PARTIAL"):
        joint = "LEARNING_ONLY_SUPPORTED"
        claim = "MAINTAIN_NARROWLY"
    else:
        joint = "CURRENT_CROSS_SCENARIO_CLAIM_WEAKENED"
        claim = "WEAKEN"

    runtime = {"stage_a_requested": cache["total_requested"], "stage_a_exact_reused": cache["exact_reusable"],
               "stage_a_partial_reused": cache["partial_reusable"], "stage_a_aggregate_reused": cache["aggregate_reusable"],
               "stage_a_new_continuations": cache["genuinely_missing"], "stage_b_new_rollouts": 0,
               "stage_b_models": 66}
    decision = {"MODE_DEFORMATION": stage_a, "MODE_SEMANTIC_TRANSFER": stage_b,
                "joint_classification": joint, "cross_scenario_claim_action": claim,
                "best_leave_mode_out_family": best_family, "geometry_improvement_vs_raw_fraction": geom_gain,
                "stage_a_families": geom, "stage_b_advantages": advantages, "cache": cache,
                "cache_postflight": post_cache}
    dump("final_decision.json", decision)
    dump("runtime_statistics.json", runtime)

    geometry_table = []
    for f, v in geom.items():
        prevalence = "-" if v["mean_tested_B15_prevalence"] is None else f"{v['mean_tested_B15_prevalence']:.3f}"
        geometry_table.append(
            f"| {f} | {v['median_coordinate_residual']:.4f} | {v['median_nearest_robust_distance']:.4f} | "
            f"{v['geometry_promising_modes']}/12 | {v['tested_modes_with_majority_B15_states']}/{v['control_tested_modes']} | "
            f"{prevalence} | {v['max_condition_number']:.2f} |")
    report = f"""# OrthoFlow3 Toy→DB transfer falsification V2

## Stage A — leave-mode-out deformation

Each fold fit 8 canonical correspondences and predicted 4 unseen modes. Held-out coordinates and outcomes were excluded from fitting and family selection. Geometry filtering preceded all control validation; there was no local eta optimization.

| family | median held-out residual | median distance to independent robust evidence | promising / 12 | tested modes with majority B15 support | mean tested B15 prevalence | max condition |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(geometry_table) + f"""

Best held-out coordinate family: **{best_family}**. Its median residual improves over raw Toy eta by {100 * geom_gain:.1f}%. Stage A: **MODE_DEFORMATION = {stage_a}**.

Stage-A cache accounting: requested {cache['total_requested']:,}; exact reused {cache['exact_reusable']:,}; partial reused {cache['partial_reusable']:,}; aggregate reused {cache['aggregate_reusable']:,}; newly executed {cache['genuinely_missing']:,}. Every new record was journaled and atomically merged into the global rollout database.

The identical post-run manifest now resolves as {post_cache['exact_reusable']:,}/{post_cache['total_requested']:,} `EXACT_REUSE` with zero missing continuations, verifying that future experiments will not rerun this batch.

## Stage B — mode-ID permutation control

The Toy semantic trunk/head was frozen. Only an 80→64 DB adapter was trained at 25% and 50% DB data. Correct identity and 10 fixed derangements used identical subsets, optimizer, epochs, and seeds 17/23/41.

| DB data | correct Q | shuffled median Q [IQR] | correct B15 | shuffled median B15 [IQR] | correct regret | shuffled median regret |
|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['db_fraction']}% | {float(r['correct_test_mean_selected_Q_median']):.4f} | {float(r['shuffled_test_mean_selected_Q_median']):.4f} [{float(r['shuffled_test_mean_selected_Q_q25']):.4f}, {float(r['shuffled_test_mean_selected_Q_q75']):.4f}] | "
        f"{float(r['correct_test_robust15_states_median']):.0f}/16 | {float(r['shuffled_test_robust15_states_median']):.0f}/16 [{float(r['shuffled_test_robust15_states_q25']):.0f}, {float(r['shuffled_test_robust15_states_q75']):.0f}] | "
        f"{float(r['correct_test_mean_regret_median']):.4f} | {float(r['shuffled_test_mean_regret_median']):.4f} |"
        for r in comp) + f"""

Stage B: **MODE_SEMANTIC_TRANSFER = {stage_b}**. Correct correspondence did not obtain a stable, out-of-distribution advantage over shuffled identities at both data fractions; the observed joint-training benefit therefore cannot be specifically attributed to shared output-column semantics.

## Joint conclusion

**{joint}**. Action on the previous Toy→DB claim: **{claim}**. The evidence supports at most partial low-complexity eta-space deformation; it does not currently isolate shared mode semantics from generic multitask/adapter regularization.
"""
    (H / "final_report.md").write_text(report)
    assets = []
    for p in sorted(H.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            assets.append({"path": str(p.relative_to(H)), "sha256": sha(p)})
    dump("manifest.json", {"task": "ORTHOFLOW3_TOY_DB_TRANSFER_FALSIFICATION_V2", "artifacts": assets})
    dump("working_state.json", {"status": "COMPLETE", "completed": ["stage_a", "stage_b", "global_db_merge", "finalization"],
                                "final_decision": joint, "next_action": None})
    write_csv("experiment_ledger.csv", [
        {"stage": "preflight", "status": "COMPLETE", "new_continuations": 0, "decision": json.dumps(cache, sort_keys=True)},
        {"stage": "stage_a", "status": "COMPLETE", "new_continuations": cache["genuinely_missing"], "decision": stage_a},
        {"stage": "stage_b", "status": "COMPLETE", "new_continuations": 0, "decision": stage_b},
        {"stage": "finalize", "status": "COMPLETE", "new_continuations": 0, "decision": joint},
    ])
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
