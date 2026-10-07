#!/usr/bin/env python3
"""Paired hard-cohort analysis after all journals are merged and postflight passes."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
KINDS = ("mac_only", "safety", "fixed_common", "old_selector_anchor",
         "old_mode_generator_mean", "generator_mean", "sample_0", "sample_1", "sample_2", "sample_3",
         "critic_selected", "oracle_best_of_4")


def write_csv(path, rows):
    with (OUT / path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def bootstrap_diff(a, b, seed=20261001):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(50000, len(a)))
    vals = (a - b)[draws].mean(1)
    return {"estimate": float(np.mean(a-b)), "ci95": np.quantile(vals, [.025,.975]).tolist()}


def main():
    post = json.loads((OUT / "cache_postflight.json").read_text())["summary"]
    # Historical incompatible profiles may exist for the same physical states;
    # they are not reused and do not invalidate full exact coverage of this plan.
    if (post["genuinely_missing"] or post["ambiguous"] or
            post["exact_reusable"] != post["total_requested"]):
        raise RuntimeError(f"Postflight incomplete: {post}")
    frozen = json.loads((OUT / "frozen_proposals.json").read_text())
    records = {}
    duplicates = 0
    for path in sorted((OUT / "raw").glob("shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line: continue
            r = json.loads(line)
            if not r["scientific_outcome_valid"]: raise RuntimeError(f"Invalid rollout in {path}")
            for kind in r["candidate_aliases"]:
                key = (r["episode_index"], kind, r["future_index"])
                if key in records:
                    old = records[key]
                    if (old["outcome"], old["eta"]) != (r["outcome"], r["eta"]):
                        raise RuntimeError(f"Conflicting local records: {key}")
                    duplicates += 1
                records[key] = r
    expected = len(frozen["states"]) * 10 * 16
    if len(records) != expected: raise RuntimeError(f"Incomplete local outcome matrix: {len(records)} / {expected}")
    per_eta = []
    per_state = []
    selected = defaultdict(list)
    for state in frozen["states"]:
        i = state["episode_index"]
        q = {}
        for kind in KINDS[:10]:
            rec = [records[(i,kind,s)] for s in range(16)]
            k = sum(r["success"] for r in rec)
            counts = Counter(r["outcome"] for r in rec)
            q[kind] = k / 16
            selected[kind].extend(rec)
            per_eta.append({"episode_index": i, "kind": kind, "successes": k, "trials": 16,
                            "Q16": q[kind], "B15": int(k>=15),
                            "deadlock": counts["deadlock"], "timeout": counts["timeout"],
                            "collision": counts["collision"], "mean_J_def": float(np.mean([r["J_def"] for r in rec if r["J_def"] is not None])) if kind!="mac_only" else None,
                            "eta": json.dumps(rec[0]["eta"])})
        crit = state["critic_selected_sample"]
        oracle = max(range(4), key=lambda j: (q[f"sample_{j}"], -j))
        for label, j in (("critic_selected",crit),("oracle_best_of_4",oracle)):
            kind = f"sample_{j}"
            q[label] = q[kind]
            selected[label].extend(records[(i,kind,s)] for s in range(16))
        row = {"episode_index": i, "critic_choice": crit, "oracle_choice": oracle,
               "oracle_has_B15": int(any(q[f"sample_{j}"]>=15/16 for j in range(4))),
               "critic_scores": json.dumps(state["critic_scores"])}
        row.update({f"Q16_{kind}": q[kind] for kind in KINDS})
        row.update({f"B15_{kind}": int(q[kind]>=15/16) for kind in KINDS})
        per_state.append(row)
    write_csv("candidate_q16.csv", per_eta)
    write_csv("per_state_results.csv", per_state)
    summary = []
    for kind in KINDS:
        rec = selected[kind]
        q = np.asarray([r[f"Q16_{kind}"] for r in per_state], np.float64)
        summary.append({"controller": kind, "states": len(per_state), "B15_states": int(np.sum(q>=15/16)),
                        "mean_Q16": float(q.mean()), "success": sum(r["success"] for r in rec),
                        "trials": len(rec), "deadlock": sum(r["deadlock"] for r in rec),
                        "timeout": sum(r["timeout"] for r in rec), "collision": sum(r["collision"] for r in rec),
                        "J_def": float(np.mean([r["J_def"] for r in rec if r["J_def"] is not None])) if kind!="mac_only" else None,
                        "episode_length": float(np.mean([r["episode_steps"] for r in rec]))})
    write_csv("controller_comparison.csv", summary)
    pairs = {}
    state_by_id = {r["episode_index"]: r for r in per_state}
    def deployed_kind(i, kind):
        if kind == "critic_selected": return f"sample_{state_by_id[i]['critic_choice']}"
        if kind == "oracle_best_of_4": return f"sample_{state_by_id[i]['oracle_choice']}"
        return kind
    for a,b in (("generator_mean","safety"),("generator_mean","fixed_common"),
                ("generator_mean","mac_only"),
                ("generator_mean","old_selector_anchor"),("generator_mean","old_mode_generator_mean"),
                ("sample_0","generator_mean"),("critic_selected","generator_mean"),
                ("critic_selected","sample_0"),("critic_selected","safety"),
                ("critic_selected","mac_only"),("critic_selected","fixed_common"),
                ("oracle_best_of_4","fixed_common"),("oracle_best_of_4","critic_selected"),
                ("oracle_best_of_4","generator_mean")):
        qa = [r[f"Q16_{a}"] for r in per_state]
        qb = [r[f"Q16_{b}"] for r in per_state]
        ba = [r[f"B15_{a}"] for r in per_state]
        bb = [r[f"B15_{b}"] for r in per_state]
        rescue = sum(records[(i,deployed_kind(i,a),s)]["success"] and not records[(i,deployed_kind(i,b),s)]["success"]
                     for i in range(200) for s in range(16))
        breaks = sum(records[(i,deployed_kind(i,b),s)]["success"] and not records[(i,deployed_kind(i,a),s)]["success"]
                     for i in range(200) for s in range(16))
        # State-level effects are primary; continuation rescues are paired by seed.
        pairs[f"{a}_vs_{b}"] = {"mean_Q16_difference": bootstrap_diff(qa,qb),
                                "B15_rate_difference": bootstrap_diff(ba,bb),
                                "B15_rescue_states": sum(x and not y for x,y in zip(ba,bb)),
                                "B15_break_states": sum(y and not x for x,y in zip(ba,bb)),
                                "paired_continuation_rescue": int(rescue),
                                "paired_continuation_break": int(breaks)}
    coverable = [r for r in per_state if r["oracle_has_B15"]]
    recovery = {
        "denominator_oracle_coverable_states": len(coverable),
        "critic_B15_and_mean_nonB15": sum(r["B15_critic_selected"] and not r["B15_generator_mean"] for r in coverable),
        "critic_B15_and_random_nonB15": sum(r["B15_critic_selected"] and not r["B15_sample_0"] for r in coverable),
    }
    recovery["versus_mean_fraction"] = recovery["critic_B15_and_mean_nonB15"] / len(coverable) if coverable else None
    recovery["versus_random_fraction"] = recovery["critic_B15_and_random_nonB15"] / len(coverable) if coverable else None
    stochastic = {
        "state_sample_draws": 4 * len(per_state),
        "B15_draws": sum(r[f"B15_sample_{j}"] for r in per_state for j in range(4)),
        "mean_Q16": float(np.mean([r[f"Q16_sample_{j}"] for r in per_state for j in range(4)])),
    }
    stochastic["B15_draw_rate"] = stochastic["B15_draws"] / stochastic["state_sample_draws"]
    baseline_failures = {}
    for baseline in ("mac_only", "safety", "fixed_common"):
        subset = [r for r in per_state if not r[f"B15_{baseline}"]]
        baseline_failures[baseline] = {
            "nonB15_states": len(subset),
            "mean_rescues_B15": sum(r["B15_generator_mean"] for r in subset),
            "random_sample_rescues_B15": sum(r["B15_sample_0"] for r in subset),
            "critic_rescues_B15": sum(r["B15_critic_selected"] for r in subset),
            "oracle_proposal_rescues_B15": sum(r["B15_oracle_best_of_4"] for r in subset),
        }
    distances = []
    for state in frozen["states"]:
        eta = state["eta"]
        mean = np.asarray(eta["generator_mean"], np.float64)
        fixed = np.asarray(eta["fixed_common"], np.float64)
        samples = np.stack([eta[f"sample_{j}"] for j in range(4)])
        distances.append({"episode_index": state["episode_index"],
                          "mean_to_fixed": float(np.linalg.norm(mean-fixed)),
                          "mean_sample_distance": float(np.linalg.norm(samples-mean, axis=1).mean()),
                          "sample_pairwise_max": float(max(np.linalg.norm(samples[a]-samples[b])
                                                             for a in range(4) for b in range(a+1,4)))})
    write_csv("proposal_geometry.csv", distances)
    stats = {"summary": summary, "paired": pairs, "critic_recovery": recovery,
             "stochastic_single_draw": stochastic,
             "baseline_failure_subgroups": baseline_failures,
             "proposal_geometry": {k: float(np.mean([r[k] for r in distances])) for k in
                                   ("mean_to_fixed", "mean_sample_distance", "sample_pairwise_max")},
             "duplicate_local_records": duplicates, "postflight": post}
    (OUT / "paired_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": {r["controller"]: [r["B15_states"],r["mean_Q16"]] for r in summary},
                      "critic_recovery": recovery}, indent=2))


if __name__ == "__main__": main()
