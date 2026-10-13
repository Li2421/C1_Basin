"""Joint-model cache evaluation and selected-only fresh confirmation.

All fitting/model selection precede these predictions. Historical phase-1
outcomes are DEVELOPMENT; the next 32 untouched states per scene are fresh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from diagnostics.transferable_state_conditioning_v1.joint_four_scene import HERE, ROOT, STAGE, PANEL, SCENES, read, sha, freeze
from diagnostics.transferable_state_conditioning_v1 import evaluate_final as metrics
from diagnostics.transferable_state_conditioning_v1.stage_evaluate_loso import model_logits
from diagnostics.transferable_state_conditioning_v1.predict_frozen import derangement
from diagnostics.physical_state_critic_formal_v1 import physical_entities
from shared_rollout_db.src.rollout_db import connect

FORMAL = HERE.parent / "physical_state_critic_formal_v1"
OLD = STAGE / "final_phase1_32"
FRESH = ROOT / "fresh_confirmation"


def predict():
    joint = read(ROOT / "final_models_frozen.json")["panels"][PANEL]
    historical = read(STAGE / "final_models_frozen.json")["panels"]
    protocol = read(ROOT / "protocol.json")
    for cohort in ("development", "fresh_confirmation"):
        directory = ROOT / cohort
        for scene in SCENES:
            ids = (protocol["fresh_confirmation_state_uids"][scene] if cohort == "fresh_confirmation"
                   else read(OLD / "frozen_splits.json")["final_confirmation"][scene]["state_uids"])
            proposals = {r["state_uid"]: r for r in read(FORMAL / "loso" / scene / "test/proposals.json")["states"]}
            states = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
            x = physical_entities.batch([{"scene": scene, "physical": states[proposals[u]["state_alias"]]["physical"]} for u in ids])
            for bank in (("proposal",) if cohort == "fresh_confirmation" else ("proposal", "anchor")):
                if bank == "proposal":
                    eta = np.asarray([proposals[u]["eta"] for u in ids], np.float32)
                else:
                    fixed = np.asarray(read(OLD / "frozen_splits.json")["common_anchor_eta"], np.float32)
                    eta = np.broadcast_to(fixed, (len(ids), 16, 3)).copy()
                z, ze, mismatch = model_logits(joint, x, eta)
                rng = np.random.default_rng(20261009)
                permutations = np.stack([derangement(len(ids), rng) for _ in range(100)])
                if bank == "anchor":
                    mismatch_choice = z[permutations].argmax(2)
                else:
                    mismatch_choice = np.stack([mismatch(p).argmax(1) for p in permutations])
                zl, _, _ = model_logits(historical["loso_"+scene], x, eta)
                zw, _, _ = model_logits(historical["within_"+scene], x, eta)
                dest = directory / "predictions" / ("within_"+scene) / (bank+".npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = dict(state_uids=np.asarray(ids), eta=eta, state_logits=z, eta_logits=ze,
                            state_choice=z.argmax(1), eta_choice=ze.argmax(1),
                            mismatch_choice=mismatch_choice.astype(np.int16), state_permutations=permutations,
                            historical_loso_choice=zl.argmax(1), historical_within_choice=zw.argmax(1))
                if dest.exists():
                    previous = np.load(dest)
                    for key, val in data.items():
                        np.testing.assert_array_equal(previous[key], val)
                else:
                    np.savez_compressed(dest, **data)
                freeze(dest.with_suffix(".json"), {"model": "joint_four_scenes", "scene": scene,
                       "cohort": cohort, "bank": bank, "source_models_sha256": sha(ROOT / "final_models_frozen.json"),
                       "predictions_sha256": sha(dest), "physical_states": len(ids),
                       "outcomes_previously_opened": cohort == "development",
                       "model_change_from_test_allowed": False})
                print(cohort, scene, bank, "mismatch_choice_change", float(np.mean(mismatch_choice != z.argmax(1))), flush=True)


def panel_metrics(scene, pred, values, full_pool):
    n = len(values)
    choices = {"Safety": np.full(n, 16), "joint_Q_eta": pred["eta_choice"],
               "joint_Q_state": pred["state_choice"], "frozen_LOSO_Q_state": pred["historical_loso_choice"],
               "frozen_within_Q_state": pred["historical_within_choice"]}
    if values.shape[1] == 16:
        choices.pop("Safety")
    if full_pool:
        choices["Oracle"] = values[:, :16, :, 0].sum(2).argmax(1)
    result = {"scene": scene, "physical_states": n, "training_scenes": list(SCENES),
              "full_candidate_pool_evaluated": full_pool, "methods": {}, "paired": {}}
    per = {}
    for name, choice in choices.items():
        assert np.all(values[np.arange(n), choice] >= 0), name
        per[name] = metrics.method_values(values, choice)
        result["methods"][name] = metrics.summarize_method(per[name])
    mismatch = pred["mismatch_choice"]
    selected_outcomes = values[np.arange(n)[None, :], mismatch]
    assert np.all(selected_outcomes >= 0)
    q = selected_outcomes[..., 0].mean(2)
    b15 = selected_outcomes[..., 0].sum(2) >= 15
    unknown = selected_outcomes[..., 4].sum(2)
    result["methods"]["mismatched_state_100"] = {
        "mean_success": float(q.mean()), "mean_B15_count": float(b15.sum(1).mean()),
        "mean_success_upper": float((q+unknown/16.).mean()),
        "mean_collisions_per_permutation": float(selected_outcomes[..., 3].sum((1,2)).mean()),
        "mean_numerical_failures_per_permutation": float(unknown.sum(1).mean())}
    state = per["joint_Q_state"]
    for name, other in per.items():
        if name == "joint_Q_state":
            continue
        result["paired"]["joint_state_minus_"+name] = {
            "mean_success": metrics.paired_bootstrap(state["q_lower"]-other["q_lower"]),
            "B15": metrics.paired_bootstrap(state["B15_confirmed"].astype(float)-other["B15_confirmed"].astype(float)),
            "numerical_worst_case_mean_success": metrics.paired_bootstrap(state["q_lower"]-other["q_upper"]),
            "numerical_best_case_mean_success": metrics.paired_bootstrap(state["q_upper"]-other["q_lower"]),
            "numerical_worst_case_B15": metrics.paired_bootstrap(state["B15_confirmed"].astype(float)-other["B15_possible"].astype(float)),
            "numerical_best_case_B15": metrics.paired_bootstrap(state["B15_possible"].astype(float)-other["B15_confirmed"].astype(float))}
    result["paired"]["joint_state_minus_mismatch"] = {
        "mean_success": metrics.paired_bootstrap(state["q_lower"]-q.mean(0)),
        "B15": metrics.paired_bootstrap(state["B15_confirmed"].astype(float)-b15.mean(0)),
        "numerical_worst_case_mean_success": metrics.paired_bootstrap(state["q_lower"]-(q+unknown/16.).mean(0)),
        "numerical_best_case_mean_success": metrics.paired_bootstrap(state["q_upper"]-q.mean(0)),
        "numerical_worst_case_B15": metrics.paired_bootstrap(state["B15_confirmed"].astype(float)-((selected_outcomes[...,0].sum(2)+unknown)>=15).mean(0)),
        "numerical_best_case_B15": metrics.paired_bootstrap(state["B15_possible"].astype(float)-b15.mean(0))}
    return result


def development():
    results = {}
    for scene in SCENES:
        for bank in ("proposal", "anchor"):
            pred = np.load(ROOT / "development/predictions" / ("within_"+scene) / (bank+".npz"))
            outcomes = np.load(OLD / "final" / bank / scene / "outcomes.npz")
            assert np.array_equal(pred["state_uids"], outcomes["state_uids"])
            result = panel_metrics(scene, pred, outcomes["values"], True)
            result["scope"] = "posthoc DEVELOPMENT, zero new rollout; no target-based model choice"
            if bank == "anchor":
                v = outcomes["values"]
                disc, ev = v[:, :, :8, 0].mean(2), v[:, :, 8:, 0].mean(2)
                row = np.arange(len(v))
                mq = ev[row[None, :], pred["mismatch_choice"]]
                result["switching"] = metrics.reversal_audit(pred["state_logits"], pred["eta_logits"], disc, ev, mq, pred["state_choice"], pred["eta_choice"])
                p = metrics.jax_sigmoid(pred["state_logits"])
                true = v[..., 0].mean(2)
                result["state_effect_correlation"] = metrics.safe_corr(p-p.mean(0), true-true.mean(0))
                chosen = ev[row, pred["state_choice"]]
                best = ev.mean(0).argmax()
                result["fixed_eta_comparison"] = {"all_16_eval_mean_success": ev.mean(0).tolist(),
                    "state_eval_success": float(chosen.mean()), "best_fixed_eval_success": float(ev[:, best].mean()),
                    "state_minus_posthoc_fixed": metrics.paired_bootstrap(chosen-ev[:, best]),
                    "selection_counts": np.bincount(pred["state_choice"], minlength=16).tolist()}
            results[scene+"/"+bank] = result
            print(scene, bank, {k: x.get("B15_confirmed") for k, x in result["methods"].items()}, flush=True)
    (ROOT / "development/results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")


def prepare_fresh():
    protocol = read(ROOT / "protocol.json")
    freeze(FRESH / "frozen_splits.json", {"final_confirmation": {s: {"state_uids": ids}
           for s, ids in protocol["fresh_confirmation_state_uids"].items()}})
    models = read(ROOT / "final_models_frozen.json")
    models["frozen_splits_sha256"] = sha(FRESH / "frozen_splits.json")
    freeze(FRESH / "final_models_frozen.json", models)
    predictions, panels, work = {}, {}, []
    for scene in SCENES:
        predpath = FRESH / "predictions" / ("within_"+scene) / "proposal.npz"
        pred = np.load(predpath)
        ids = pred["state_uids"].tolist()
        assert ids == protocol["fresh_confirmation_state_uids"][scene]
        selections = np.concatenate([pred["state_choice"][None], pred["eta_choice"][None],
                                     pred["historical_loso_choice"][None], pred["historical_within_choice"][None],
                                     pred["mismatch_choice"], np.full((1, len(ids)), 16)], axis=0)
        needed = {(u, int(j)) for i, u in enumerate(ids) for j in selections[:, i]}
        original = read(FORMAL / "loso" / scene / "test/cases.json")
        cases = [c for c in original if (c["state_uid"], c["eta_index"]) in needed]
        assert len(cases) == len(needed)*16
        panel = FRESH / "final/proposal" / scene
        freeze(panel / "cases.json", cases)
        grouped = {}
        for c in cases:
            key = (c["state_uid"], c["eta_uid"], c["controller_uid"])
            grouped.setdefault(key, []).append(c["seed_key"])
        requests = [dict(state_uid=s, eta_uid=e, controller_uid=c, seed_keys=seeds)
                    for (s, e, c), seeds in grouped.items()]
        freeze(panel / "planned_rollouts.json", {"requests": requests})
        subprocess.run([sys.executable, "-m", "shared_rollout_db.plan", "--manifest", str(panel / "planned_rollouts.json"),
                        "--output", str(panel / "cache_preflight.json")], check=True,
                       stdout=(panel / "preflight_stdout.json").open("w"))
        pf = read(panel / "cache_preflight.json")
        assert not pf["summary"]["ambiguous"]
        missing_keys = {(r["state_uid"], r["eta_uid"], r["controller_uid"], seed)
                        for r in pf["details"] for seed in r["missing_seeds"]}
        missing = [c for c in cases if (c["state_uid"], c["eta_uid"], c["controller_uid"], c["seed_key"]) in missing_keys]
        assert len(missing) == pf["summary"]["genuinely_missing"]
        freeze(panel / "missing_cases.json", missing)
        for start in range(0, len(missing), 64):
            work.append({"scene": scene, "start": start, "end": min(start+64, len(missing))})
        panels[scene] = {"states": len(ids), "selected_pairs_including_safety": len(needed),
                         "requested_seeds": len(cases), "missing_seeds": len(missing),
                         "reused_seeds": len(cases)-len(missing), "cases_sha256": sha(panel / "cases.json")}
        predictions["within_"+scene+"/proposal"] = sha(predpath)
    freeze(FRESH / "final/proposal_work_items.json", work)
    freeze(FRESH / "final_protocol.json", {"schema": "joint_four_selected_only_confirmation_v1",
           "final_models_sha256": sha(FRESH / "final_models_frozen.json"),
           "frozen_splits_sha256": sha(FRESH / "frozen_splits.json"), "predictions_sha256": predictions,
           "panels": panels, "full_pool_oracle_available": False, "new_outcomes_opened_before_freeze": False})
    print(json.dumps({"work_shards": len(work), "panels": panels}), flush=True)


def evaluate_fresh():
    from diagnostics.transferable_state_conditioning_v1 import final_worker
    final_worker.HERE = FRESH
    final_worker.gate()
    result = {}
    with connect(True) as db:
        for scene in SCENES:
            pred = np.load(FRESH / "predictions" / ("within_"+scene) / "proposal.npz")
            ids = pred["state_uids"].tolist()
            lookup = {u: i for i, u in enumerate(ids)}
            values = np.full((len(ids), 17, 16, 5), -1, np.int8)
            cases = read(FRESH / "final/proposal" / scene / "cases.json")
            issues = []
            for c in cases:
                row = db.execute("SELECT * FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0", (c["rollout_uid"],)).fetchone()
                assert row is not None and row["compatibility_quality"] == "EXACT_REUSE", c["rollout_uid"]
                assert (row["state_uid"], row["eta_uid"], row["controller_uid"], row["seed_key"]) == (c["state_uid"], c["eta_uid"], c["controller_uid"], c["seed_key"])
                values[lookup[c["state_uid"]], c["eta_index"], c["future_index"]] = [row[k] for k in metrics.FLAGS]
                if row["collision"] or row["numerical_failure"]:
                    issues.append({**c, "collision": int(row["collision"]),
                                   "numerical_failure": int(row["numerical_failure"]),
                                   "original_source_file": row["original_source_file"]})
            np.savez_compressed(FRESH / "final/proposal" / scene / "selected_outcomes.npz",
                                values=values, observed=values[..., 0]>=0, state_uids=np.asarray(ids), eta=pred["eta"])
            result[scene] = panel_metrics(scene, pred, values, False)
            result[scene]["scope"] = "fresh independent state confirmation; selected-only, unmeasured candidates remain -1"
            result[scene]["all_observed_seed_records"] = len(cases)
            result[scene]["all_observed_collisions"] = int(np.sum(values[...,3][values[...,3]>=0]))
            result[scene]["all_observed_numerical_failures"] = int(np.sum(values[...,4][values[...,4]>=0]))
            (FRESH / "final/proposal" / scene / "safety_issues.json").write_text(json.dumps(issues, indent=2)+"\n")
    (FRESH / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Joint four-scene physical-only critic: fresh confirmation", "", "B15 means at least 15/16 successes; every row uses 32 independent physical states.", "",
             "| Scene | Safety | Joint eta-only | Joint state | Same-scene mismatch mean | Frozen LOSO state | Frozen within state |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scene, r in result.items():
        m = r["methods"]
        vals = [str(m[k]["B15_confirmed"])+"/32" for k in ("Safety", "joint_Q_eta", "joint_Q_state")]
        vals += [f"{m['mismatched_state_100']['mean_B15_count']:.2f}/32"]
        vals += [str(m[k]["B15_confirmed"])+"/32" for k in ("frozen_LOSO_Q_state", "frozen_within_Q_state")]
        lines.append("| "+scene+" | "+" | ".join(vals)+" |")
    lines += ["", "All paired state-bootstrap intervals, mean success, collisions and numerical upper/lower bounds are in results.json. Models and 100 same-scene derangements were frozen before labels. This selected-only confirmation has no full-pool oracle; the independent development cache has complete K16 and fixed-bank diagnostics."]
    (FRESH / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("FRESH_CONFIRMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("predict", "development", "prepare_fresh", "evaluate_fresh"))
    a = p.parse_args()
    globals()[a.action]()
