"""Compare H20 and H100 nominal separation for matched Q16 counterfactuals."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from .train import OUT, read, write


def run():
    d = dict(np.load(OUT / "second_variant/triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    coverage = []
    for path in sorted((OUT / "h100_nominal/coverage").glob("*.json")):
        coverage.extend(read(path))
    data = {(r["state_uid"], r["controller_sha"]): np.asarray(r["nominal"], np.float32)
            for r in coverage if r["valid"]}
    first = read(OUT / "balanced_expansion/protocol.json")["profiles"]
    second = read(OUT / "second_variant/protocol.json")["profiles"]
    toy = read(OUT.parent / "orthoflow3_controller_conditioning_probe_v1/protocol.json")
    results = []
    for scene in ("toy_giveway", "ring_exchange"):
        ids = np.asarray([i for i, r in enumerate(rows) if r["scene"] == scene and r["split"] == "train"], int)
        shas = [None, toy["flow_sha256"]["1"]] if scene == "toy_giveway" else [None, first[scene]["alternate_flow_sha256"], second[scene]["alternate_flow_sha256"]]
        states = sorted({rows[i]["state_uid"] for i in ids})
        train_h100 = np.stack([data[(uid, sha)] for uid in states for sha in shas])
        scale100 = np.maximum(train_h100.std(0), .05)
        train_h20 = np.concatenate([d["base_c"][ids, :10], d["alt_c"][ids, :10]] +
                                   ([] if scene == "toy_giveway" else [d["second_c"][ids, :10]]))
        scale20 = np.maximum(train_h20.std(0), .05)
        train_rich = np.concatenate([d["base_c"][ids], d["alt_c"][ids]] +
                                    ([] if scene == "toy_giveway" else [d["second_c"][ids]]))
        scale_rich = np.maximum(train_rich.std(0), .05)
        labels = ("base", "alt", "second") if scene != "toy_giveway" else ("base", "alt")
        for i, row in enumerate(rows):
            if row["scene"] != scene:
                continue
            uid = row["state_uid"]
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    va, vb = labels[a], labels[b]
                    na, nb = d[va + "_s"][i] + d[va + "_f"][i], d[vb + "_s"][i] + d[vb + "_f"][i]
                    if na != 16 or nb != 16:
                        continue
                    qa, qb = d[va + "_s"][i] / 16, d[vb + "_s"][i] / 16
                    ca100, cb100 = data[(uid, shas[a])], data[(uid, shas[b])]
                    ca20, cb20 = d[va + "_c"][i, :10], d[vb + "_c"][i, :10]
                    car, cbr = d[va + "_c"][i], d[vb + "_c"][i]
                    results.append({"scene": scene, "split": row["split"], "state_uid": uid,
                                    "pair_index": i, "controller_a": va, "controller_b": vb,
                                    "abs_Q16_change": abs(float(qa-qb)),
                                    "h20_nominal_distance": float(np.linalg.norm((ca20-cb20)/scale20)/np.sqrt(10)),
                                    "h20_rich_distance": float(np.linalg.norm((car-cbr)/scale_rich)/np.sqrt(24)),
                                    "h100_nominal_distance": float(np.linalg.norm((ca100-cb100)/scale100)/np.sqrt(10))})
    with (OUT / "h100_nominal/matched_q_distances.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary = {}
    for scene in ("toy_giveway", "ring_exchange"):
        z = [r for r in results if r["scene"] == scene]
        big = [r for r in z if r["abs_Q16_change"] >= .75]
        summary[scene] = {"full_matched_comparisons": len(z), "large_Q_change": len(big),
                          "distance_median_all": {name: float(np.median([r[name] for r in z]))
                                                  for name in ("h20_nominal_distance", "h20_rich_distance", "h100_nominal_distance")},
                          "distance_median_large_Q": {name: float(np.median([r[name] for r in big])) if big else None
                                                      for name in ("h20_nominal_distance", "h20_rich_distance", "h100_nominal_distance")},
                          "large_Q_with_distance_below_0.25": {name: sum(r[name] < .25 for r in big)
                                                                  for name in ("h20_nominal_distance", "h20_rich_distance", "h100_nominal_distance")},
                          "distance_vs_abs_Q_spearman": {name: float(spearmanr([r[name] for r in z],
                                                                               [r["abs_Q16_change"] for r in z]).statistic)
                                                         for name in ("h20_nominal_distance", "h20_rich_distance", "h100_nominal_distance")}}
    write(OUT / "h100_nominal/audit.json",
          {"source_only": True, "target_K16_labels_used": False,
           "horizon_steps": 100, "full_task_fraction_upper_bound": 100/700,
           "eta_independent_nominal_context_once_per_state_controller": True,
           "summary": summary})
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
