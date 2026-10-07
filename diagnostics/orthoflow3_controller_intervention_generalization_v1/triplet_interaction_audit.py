"""Certified source-family-heldout eta-ranking reversals under Flow swaps."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from .second_variant import OUT as SECOND
from .train import OUT, read, write


def run():
    data = dict(np.load(SECOND / "triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    exact = {(r["scene"], r["state_uid"], tuple(np.asarray(r["eta"], np.float32))): r
             for r in read(SECOND / "pairs.json")}
    results = {(r["scene"], r["state_uid"], r["eta_uid"]): r
               for r in csv.DictReader((SECOND / "intervention_pairs.csv").open())}
    by = defaultdict(list)
    for idx, r in enumerate(rows):
        if r["split"] != "validation" or r["scene"] == "toy_giveway":
            continue
        pair = exact[(r["scene"], r["state_uid"], tuple(data["eta"][idx]))]
        outcome = results[(r["scene"], r["state_uid"], pair["eta_uid"])]
        if int(outcome["second_valid"]) != 16:
            continue
        by[(r["scene"], r["state_uid"])].append((idx, outcome))
    comparisons = []
    for (scene, uid), items in by.items():
        for (ia, a), (ib, b) in combinations(items, 2):
            a_lo, a_hi = int(a["base_observed_success"]) / 16, (16 - int(a["base_observed_failure"])) / 16
            b_lo, b_hi = int(b["base_observed_success"]) / 16, (16 - int(b["base_observed_failure"])) / 16
            second = (int(a["second_success"]) - int(b["second_success"])) / 16
            if a_lo > b_hi and second <= -.25:
                base_sign = 1
            elif b_lo > a_hi and second >= .25:
                base_sign = -1
            else:
                continue
            comparisons.append({"scene": scene, "state_uid": uid, "pair_a": int(ia), "pair_b": int(ib),
                                "certified_base_sign": int(base_sign), "second_sign": int(-base_sign),
                                "second_Q_difference": float(second)})
    out = []
    kinds = (("full", "triplet_input_predictions.csv"),
             ("additive", "triplet_nominal_additive_input_predictions.csv"),
             ("bounded1", "triplet_bounded1_input_predictions.csv"),
             ("bounded3", "triplet_bounded3_input_predictions.csv"),
             ("bounded3_interaction", "triplet_bounded3_interaction_input_predictions.csv"))
    for kind, path in kinds:
        preds = {(int(r["seed"]), int(r["controller"]), int(r["pair_index"])): float(r["p_correct"])
                 for r in csv.DictReader((SECOND / path).open())}
        for seed in (17, 23, 41):
            for row in comparisons:
                ia, ib = row["pair_a"], row["pair_b"]
                d0 = preds[(seed, 0, ia)] - preds[(seed, 0, ib)]
                d2 = preds[(seed, 2, ia)] - preds[(seed, 2, ib)]
                out.append({**row, "kind": kind, "seed": seed,
                            "predicted_base_difference": float(d0),
                            "predicted_second_difference": float(d2),
                            "predicted_reversal": bool(d0 * d2 < 0),
                            "both_rankings_correct": bool(np.sign(d0) == row["certified_base_sign"] and
                            np.sign(d2) == row["second_sign"])})
    with (SECOND / "certified_heldout_rank_reversals.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader();writer.writerows(out)
    summary = {kind: {"cases_per_seed": len(comparisons),
                      "both_rankings_correct": sum(r["both_rankings_correct"] for r in out if r["kind"] == kind),
                      "total_seed_cases": 3 * len(comparisons),
                      "predicted_reversal": sum(r["predicted_reversal"] for r in out if r["kind"] == kind)}
               for kind, _ in kinds}
    write(SECOND / "certified_heldout_rank_reversals.json",
          {"selection": "true interval-certified reversal only; independent of model prediction",
           "source_val_only": True, "target_labels_used": False,
           "cases": comparisons, "summary": summary})
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
