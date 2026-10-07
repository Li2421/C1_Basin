"""Read-only analysis of why the prospective Ring B15 selection test failed."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .held_robust_benchmark import DEST
from .probe import OUT, read, write


def main():
    detail = read(DEST / "held_selection_detail.json")
    frozen = read(DEST / "frozen_predictions.json")
    rows = detail["candidate_outcomes"]
    pairs = read(DEST / "pairs.json")
    by_state = defaultdict(list)
    for i, r in enumerate(rows): by_state[r["state_uid"]].append(i)
    # The first four were the outcome-blind high-frequency source-TRAIN eta;
    # K8 adds the first four outcome-blind farthest-first eta. This is a
    # diagnostic subset of the already-opened panel, not a new test set.
    subpanels = []
    for k in (4, 8, 16):
        pools = {u: ids[:k] for u, ids in by_state.items()}
        eligible = [u for u, ids in pools.items()
                    if any(rows[i]["B15"] for i in ids) and any(rows[i]["non_B15"] for i in ids)]
        scored = []
        for item in frozen["predictions"]:
            p = item["probabilities"]
            picks = {u: max(pools[u], key=lambda i: p[i]) for u in eligible}
            scored.append({"model": item["kind"], "seed": item["seed"],
                           "context": item["context"], "selected_B15": sum(rows[i]["B15"] for i in picks.values())})
        subpanels.append({"K": k, "eligible_mixed_states": len(eligible), "scores": scored})
    contexts = sorted((r for shard in range(4)
                       for r in read(DEST / f"context_shard{shard}of4.json")),
                      key=lambda r:r["pair_index"])
    held = np.asarray([r["held"] for r in contexts])
    base = np.asarray([r["wrong_base"] for r in contexts])
    norm = read(OUT / "models/ring_exchange/physical_context/seed17/normalization.json")
    center, scale = np.asarray(norm["context_center"]), np.asarray(norm["context_scale"])
    z = (held-center)/scale
    failures = []
    for state_uid, info in detail["state_status"].items():
        ids = by_state[state_uid]
        if info["B15_candidates"] > 3: continue
        summaries = []
        for local_index, i in enumerate(ids):
            summaries.append({"local_index": local_index, "eta_uid": pairs[i]["eta_uid"],
                              "eta": pairs[i]["eta"], "success": rows[i]["success"],
                              "observed": rows[i]["observed"],
                              "scores": [{"model": r["kind"], "seed":r["seed"],
                                          "context":r["context"], "probability":r["probabilities"][i]}
                                         for r in frozen["predictions"]]})
        failures.append({"state_uid":state_uid,"family":pairs[ids[0]]["family"],
                         "B15_candidates":info["B15_candidates"],"candidates":summaries})
    write(DEST / "failure_diagnosis.json", {
        "main_test_K16_frozen": True,
        "K4_K8_are_posthoc_diagnostics_not_independent_confirmation": True,
        "subpanels": subpanels,
        "H20_correct_wrong_median_normalized_L2":float(np.median(np.linalg.norm((held-base)/scale,axis=1))),
        "H20_correct_absolute_z_above3":int((np.abs(z)>3).sum()),
        "H20_total_values":int(z.size),
        "H20_correct_absolute_z_above5":int((np.abs(z)>5).sum()),
        "few_B15_states": failures,
    })
    print(json.dumps({"few_B15_families":[x["family"] for x in failures],
                      "H20_z_above3":int((np.abs(z)>3).sum()),
                      "subpanels":[{"K":x["K"],"eligible":x["eligible_mixed_states"]} for x in subpanels]}))


if __name__ == "__main__":main()
