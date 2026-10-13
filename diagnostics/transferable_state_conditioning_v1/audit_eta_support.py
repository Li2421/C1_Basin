"""Label-blind frozen K16 proposal geometry audit for source/target shift."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "physical_state_critic_formal_v1"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")
SCALE = np.array([0.75, 1.0, 0.75])


def pool(scene, split):
    rows = json.loads((FORMAL / "loso" / scene / split / "proposals.json").read_text())["states"]
    return np.asarray([r["eta"] for r in rows], float).reshape(-1, 3)


def main():
    train = {s: pool(s, "train") for s in SCENES}
    test = {s: pool(s, "test") for s in SCENES}
    result = {"schema": "frozen_stochastic_K16_eta_support_v1", "normalization_scale": SCALE.tolist(),
              "scene_proposals": {}, "loso_target_vs_source": {}}
    for scene in SCENES:
        e = train[scene]
        result["scene_proposals"][scene] = {"n": len(e), "mean": e.mean(0).tolist(),
                                             "std": e.std(0).tolist(),
                                             "min": e.min(0).tolist(), "max": e.max(0).tolist(),
                                             "p05": np.quantile(e, .05, axis=0).tolist(),
                                             "p95": np.quantile(e, .95, axis=0).tolist()}
    for target in SCENES:
        source = np.concatenate([train[s] for s in SCENES if s != target])
        held = test[target]
        tree = cKDTree(source / SCALE)
        distance = tree.query(held / SCALE, k=1)[0]
        minval, maxval = source.min(0), source.max(0)
        outside = np.any((held < minval) | (held > maxval), axis=1)
        q05, q95 = np.quantile(source, [.05, .95], axis=0)
        tails = np.any((held < q05) | (held > q95), axis=1)
        result["loso_target_vs_source"][target] = {
            "target_proposals": len(held),
            "fraction_outside_source_axis_bounds": float(outside.mean()),
            "fraction_outside_source_5_95_box": float(tails.mean()),
            "nearest_source_eta_distance_median_normalized": float(np.median(distance)),
            "nearest_source_eta_distance_p95_normalized": float(np.quantile(distance, .95))}
    target = HERE / "eta_support_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    for s, v in result["loso_target_vs_source"].items():
        print(s, v)


if __name__ == "__main__":
    main()
