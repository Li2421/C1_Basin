"""Non-neural local support diagnostic on matched three-controller evidence.

Validation source families are never neighbors of themselves. Partial-count
early-stop pairs are excluded from Q regression, not assigned invented Q16.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from .second_variant import OUT as SECOND
from .train import OUT, read, write


def run():
    d = dict(np.load(SECOND / "triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    scene = np.asarray([r["scene"] for r in rows])
    split = np.asarray([r["split"] for r in rows])
    state = d["state_index"]
    parts = []
    for name in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask"):
        parts.append(d[name].reshape(len(d[name]), -1).astype(np.float32))
    physical = np.concatenate(parts, -1)[state]
    samples = []
    for c, name in enumerate(("base", "alt", "second")):
        for i in range(len(rows)):
            n = float(d[name + "_s"][i] + d[name + "_f"][i])
            if n != 16:
                continue
            samples.append({"pair": i, "controller": c, "scene": scene[i], "split": split[i],
                            "state_uid": rows[i]["state_uid"],
                            "q": float(d[name + "_s"][i] / 16),
                            "eta": d["eta"][i], "context": d[name + "_c"][i],
                            "physical": physical[i]})
    train = [r for r in samples if r["split"] == "train"]
    val = [r for r in samples if r["split"] == "validation"]
    eta_fit = np.stack([r["eta"] for r in train])
    context_fit = np.stack([r["context"] for r in train])
    physical_fit = np.stack([r["physical"] for r in train])
    stats = [(a.mean(0), np.maximum(a.std(0), floor)) for a, floor in
             ((eta_fit, .1), (context_fit, .05), (physical_fit, .05))]

    def feature(row, kind):
        eta = (row["eta"] - stats[0][0]) / stats[0][1]
        if kind == "eta_only":
            return eta / np.sqrt(len(eta))
        c = (row["context"] - stats[1][0]) / stats[1][1]
        if kind == "eta_context":
            return np.concatenate((eta / np.sqrt(len(eta)), c / np.sqrt(len(c))))
        h = (row["physical"] - stats[2][0]) / stats[2][1]
        return np.concatenate((eta / np.sqrt(len(eta)), c / np.sqrt(len(c)), h / np.sqrt(len(h))))

    output, aggregate = [], []
    for kind in ("eta_only", "eta_context", "eta_context_physical"):
        for scope in ("same_scene", "all_source_scenes"):
            for k in (1, 5):
                X = np.stack([feature(r, kind) for r in train])
                by = defaultdict(list)
                for r in val:
                    pool = np.flatnonzero(np.asarray([u["scene"] == r["scene"] for u in train])) if scope == "same_scene" else np.arange(len(train))
                    distances = np.linalg.norm(X[pool] - feature(r, kind), axis=1)
                    select = pool[np.argsort(distances)[:k]]
                    dist = distances[np.argsort(distances)[:k]]
                    weights = 1 / np.maximum(dist, .05)
                    prediction = float(np.average([train[i]["q"] for i in select], weights=weights))
                    output.append({"kind": kind, "scope": scope, "k": k,
                                   "scene": r["scene"], "state_uid": r["state_uid"],
                                   "pair_index": r["pair"], "controller": r["controller"],
                                   "true_Q16": r["q"], "predicted_Q": prediction,
                                   "nearest_distance": float(dist.min())})
                # Evaluate only already-established source-family validation.
                for scene_name in sorted(set(r["scene"] for r in val)):
                    zz = [r for r in output if r["kind"] == kind and r["scope"] == scope and r["k"] == k and r["scene"] == scene_name]
                    y = np.asarray([r["true_Q16"] for r in zz]);p = np.asarray([r["predicted_Q"] for r in zz])
                    groups = defaultdict(list)
                    for r in zz: groups[(r["state_uid"], r["controller"])].append(r)
                    oracle = selected = 0
                    for group in groups.values():
                        oracle += any(r["true_Q16"] >= 15/16 for r in group)
                        selected += max(group, key=lambda r: r["predicted_Q"])["true_Q16"] >= 15/16
                    aggregate.append({"kind": kind, "scope": scope, "k": k, "scene": scene_name,
                                      "full_Q16_pairs": len(zz), "state_controller_groups": len(groups),
                                      "MAE": float(np.abs(y-p).mean()),
                                      "NLL": float((-y*np.log(np.clip(p,1e-5,1-1e-5))-(1-y)*np.log(np.clip(1-p,1e-5,1-1e-5))).mean()),
                                      "oracle_B15_groups": int(oracle), "selected_B15_groups": int(selected),
                                      "median_nearest_train_distance": float(np.median([r["nearest_distance"] for r in zz]))})
    with (SECOND / "local_source_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    with (SECOND / "local_source_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    write(SECOND / "local_source_audit.json",
          {"full_Q16_train_conditions": len(train), "full_Q16_heldout_source_conditions": len(val),
           "partial_count_labels_not_imputed": True, "target_labels_used": False,
           "metrics": aggregate})
    print(json.dumps({"train_conditions": len(train), "val_conditions": len(val),
                      "ring": [x for x in aggregate if x["scene"] == "ring_exchange"]}))


if __name__ == "__main__":
    run()
