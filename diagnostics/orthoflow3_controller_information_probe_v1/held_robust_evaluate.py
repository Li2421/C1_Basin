"""Open prospective held-controller outcomes once predictions are frozen."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from shared_rollout_db.src.rollout_db import canonical, connect

from .held_robust_benchmark import DEST
from .probe import read, write


def main():
    scores = read(DEST / "frozen_predictions.json")
    assert scores["held_task_outcomes_present_at_score_time"] is False
    pairs = read(DEST / "pairs.json")
    assert scores["pair_keys"] == [{"state_uid": r["state_uid"], "eta_uid": r["eta_uid"]} for r in pairs]
    profile = read(DEST / "protocol.json")["profiles"]["ring_exchange"]
    records = []
    standard = {canonical({"future_index": i}) for i in range(16)}
    with connect(True) as db:
        for row in pairs:
            got = db.execute("""SELECT seed_key, success, numerical_failure, conflict_quarantined,
                compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                (row["state_uid"], row["eta_uid"], profile["alternate_controller_uid"])).fetchall()
            exact = [r for r in got if r["seed_key"] in standard and not r["numerical_failure"]
                     and not r["conflict_quarantined"] and r["compatibility_quality"] == "EXACT_REUSE"]
            numerical = [r for r in got if r["seed_key"] in standard and r["numerical_failure"]]
            s = sum(int(r["success"]) for r in exact)
            f = len(exact) - s
            records.append({"state_uid": row["state_uid"], "eta_uid": row["eta_uid"],
                            "success": s, "failure": f, "numerical": len(numerical),
                            "observed": len(exact), "full_Q16": len(exact) == 16,
                            "Q_lower": s/16, "Q_upper": (16-f)/16,
                            "Q16": s/16 if len(exact) == 16 else None,
                            "B15": s >= 15, "non_B15": f >= 2})
    by_state = defaultdict(list)
    for i, row in enumerate(records):
        by_state[row["state_uid"]].append(i)
    assert len(by_state) == 12 and all(len(ids) == 16 for ids in by_state.values())
    eligibility = {}
    for uid, ids in by_state.items():
        robust = sum(records[i]["B15"] for i in ids)
        fail = sum(records[i]["non_B15"] for i in ids)
        unknown = 16 - robust - fail
        status = ("SELECTION_ELIGIBLE" if robust and fail else
                  "UNDERDISCRIMINATIVE_ALL_B15" if robust == 16 else
                  "COVERAGE_FAILURE" if not robust and not unknown else
                  "UNDERRESOLVED")
        eligibility[uid] = {"status": status, "B15_candidates": robust,
                            "non_B15_candidates": fail, "unknown_candidates": unknown}
    # Each state uses the same 16 eta. Thus opposite true order for the same
    # eta pair across states is direct state-eta interaction evidence.
    state_ids = list(by_state)
    reversal_cases = []
    for ua, ub in combinations(state_ids, 2):
        ia, ib = by_state[ua], by_state[ub]
        for j, k in combinations(range(16), 2):
            a0, a1 = records[ia[j]]["Q16"], records[ia[k]]["Q16"]
            b0, b1 = records[ib[j]]["Q16"], records[ib[k]]["Q16"]
            if None in (a0, a1, b0, b1):
                continue
            da, db = a0-a1, b0-b1
            if abs(da) >= .25 and abs(db) >= .25 and da*db < 0:
                reversal_cases.append((ia[j], ia[k], ib[j], ib[k], int(np.sign(da)), int(np.sign(db))))
    eval_rows = []
    selections = {}
    for item in scores["predictions"]:
        p = np.asarray(item["probabilities"])
        assert p.shape == (len(pairs),)
        key = (item["kind"], item["seed"], item["context"])
        picked = {}
        eligible = []
        for uid, ids in by_state.items():
            chosen = sorted(ids, key=lambda i: (-p[i], i))
            i = chosen[0]
            picked[uid] = i
            if eligibility[uid]["status"] != "SELECTION_ELIGIBLE":
                continue
            eligible.append((uid, i, chosen))
        selected_Q = [records[i]["Q16"] for _, i, _ in eligible]
        true_Q = [records[i]["Q16"] for uid, _, _ in eligible for i in by_state[uid]]
        full = all(q is not None for q in true_Q)
        oracle_regret = (float(np.mean([max(records[i]["Q16"] for i in by_state[uid]) - records[j]["Q16"]
                                        for uid, j, _ in eligible])) if full else None)
        s = np.asarray([r["success"] for r in records], float)
        f = np.asarray([r["failure"] for r in records], float)
        likelihood = float((-s*np.log(np.clip(p,1e-8,1)) - f*np.log(np.clip(1-p,1e-8,1))).sum()
                           / max(1,(s+f).sum()))
        reversal_correct = sum((np.sign(p[aj]-p[ak]) == sa and np.sign(p[bj]-p[bk]) == sb)
                               for aj,ak,bj,bk,sa,sb in reversal_cases)
        reversal_changed = sum((p[aj]-p[ak])*(p[bj]-p[bk]) < 0
                               for aj,ak,bj,bk,_,_ in reversal_cases)
        eval_rows.append({"kind": item["kind"], "seed": item["seed"], "context": item["context"],
                          "observed_NLL": likelihood,
                          "eligible_states": len(eligible),
                          "B15_selected": sum(records[i]["B15"] for _, i, _ in eligible),
                          "non_B15_selected": sum(records[i]["non_B15"] for _, i, _ in eligible),
                          "unknown_selected": sum(not records[i]["B15"] and not records[i]["non_B15"]
                                                  for _, i, _ in eligible),
                          "mean_selected_Q16": float(np.mean(selected_Q)) if full and eligible else None,
                          "mean_selected_Q_lower": float(np.mean([records[i]["Q_lower"] for _,i,_ in eligible])) if eligible else None,
                          "mean_selected_Q_upper": float(np.mean([records[i]["Q_upper"] for _,i,_ in eligible])) if eligible else None,
                          "mean_true_Q_regret": oracle_regret,
                          "top2_contains_B15": sum(any(records[i]["B15"] for i in chosen[:2]) for _, _, chosen in eligible),
                          "top3_contains_B15": sum(any(records[i]["B15"] for i in chosen[:3]) for _, _, chosen in eligible),
                          "top1_agrees_with_empirical_oracle": sum(
                              records[i]["Q16"] is not None and
                              records[i]["Q16"] == max((records[j]["Q16"] for j in by_state[uid]
                                                         if records[j]["Q16"] is not None), default=-1)
                              for uid, i, _ in eligible),
                          "severe_false_positive_top1": sum(p[i]>.9 and records[i]["Q16"] is not None
                                                       and records[i]["Q16"]<=.5 for _, i, _ in eligible),
                          "mean_selected_prediction": float(np.mean([p[i] for _, i, _ in eligible])) if eligible else None,
                          "strong_state_eta_reversals": len(reversal_cases),
                          "strong_reversals_both_rankings_correct": int(reversal_correct),
                          "strong_reversals_predicted_sign_change": int(reversal_changed),
                          "selected_indices": [i for _, i, _ in eligible]})
        selections[key] = picked
    paired = []
    for seed in (17, 23, 41):
        baseline = selections[("eta_only", seed, "none")]
        correct = selections[("physical_context", seed, "correct")]
        wrong = selections[("physical_context", seed, "wrong_base")]
        selected = [uid for uid in by_state if eligibility[uid]["status"] == "SELECTION_ELIGIBLE"]
        outcome = lambda choice, uid: records[choice[uid]]["B15"]
        paired.append({"seed": seed, "eligible_states": len(selected),
                       "correct_vs_eta_rescue": sum(outcome(correct,u) and not outcome(baseline,u) for u in selected),
                       "correct_vs_eta_break": sum(outcome(baseline,u) and not outcome(correct,u) for u in selected),
                       "correct_vs_wrong_rescue": sum(outcome(correct,u) and not outcome(wrong,u) for u in selected),
                       "correct_vs_wrong_break": sum(outcome(wrong,u) and not outcome(correct,u) for u in selected),
                       "correct_wrong_top1_changed": sum(correct[u]!=wrong[u] for u in selected),
                       "correct_eta_top1_changed": sum(correct[u]!=baseline[u] for u in selected)})
    with (DEST / "held_selection_metrics.csv").open("w", newline="") as fh:
        fields = [k for k in eval_rows[0] if k != "selected_indices"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        w.writerows([{k:r[k] for k in fields} for r in eval_rows])
    with (DEST / "paired_selection.csv").open("w", newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    write(DEST / "held_selection_detail.json", {
        "label_semantics": "exact compatible observed seeds; numerical separate, no imputed Q16",
        "candidate_coverage": {"states": len(by_state),
                               "eligible": sum(v["status"]=="SELECTION_ELIGIBLE" for v in eligibility.values()),
                               "underdiscriminative": sum(v["status"]=="UNDERDISCRIMINATIVE_ALL_B15" for v in eligibility.values()),
                               "coverage_failure": sum(v["status"]=="COVERAGE_FAILURE" for v in eligibility.values()),
                               "underresolved": sum(v["status"]=="UNDERRESOLVED" for v in eligibility.values())},
        "state_status": eligibility, "candidate_outcomes": records,
        "strong_state_eta_reversal_cases": len(reversal_cases),
        "evaluation": eval_rows, "paired": paired})
    print(json.dumps({"coverage": read(DEST / "held_selection_detail.json")["candidate_coverage"],
                      "paired": paired}, indent=2))


if __name__ == "__main__": main()
