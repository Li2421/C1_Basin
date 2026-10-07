"""Outcome-blind state-family and spatial eta partition for Four/Ring."""

import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from offline_baselines import ROOT, OUT, dump_json


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    states = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                           columns=["scenario", "split", "state_uid", "source_initial_state_id"]).to_pylist()
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "eta_uid", "eta_raw"]).to_pylist()
    # Ring and Four have an exactly shared geometric probe panel. Coordinates
    # and source IDs alone determine this split, never success outcomes.
    common = set(r["eta_uid"] for r in labels if r["scenario"] == "ring_exchange" and r["split"] == "train") & \
             set(r["eta_uid"] for r in labels if r["scenario"] == "four_way_intersection" and r["split"] == "train")
    coord = {r["eta_uid"]: np.asarray(json.loads(r["eta_raw"]), float) for r in labels if r["eta_uid"] in common}
    eids = sorted(common)
    x = np.stack([coord[e] for e in eids])
    center = (x.min(axis=0) + x.max(axis=0)) / 2
    radius = np.maximum((x.max(axis=0) - x.min(axis=0)) / 2, 1e-6)
    z = (x - center) / radius
    medoid = [0]
    for _ in range(11):
        dist = cdist(z, z[medoid]).min(axis=1)
        dist[medoid] = -1
        medoid.append(int(np.argmax(dist)))
    group = np.argmin(cdist(z, z[medoid]), axis=1)
    group_ids = list(range(12))
    group_ids.sort(key=lambda g: digest(eids[medoid[g]]))
    assignment = {group_ids[0]: "test", group_ids[1]: "test", group_ids[2]: "dev", group_ids[3]: "dev"}
    assignment.update({g: "train" for g in group_ids[4:]})
    eta_split = {part: [eids[i] for i, g in enumerate(group) if assignment[int(g)] == part]
                 for part in ("train", "dev", "test")}
    assert sum(map(len, eta_split.values())) == len(eids)
    dtest = cdist(z[np.isin(eids, eta_split["test"])], z[np.isin(eids, eta_split["train"])]).min(axis=1)
    state_split = {}
    for sc in ("four_way_intersection", "ring_exchange"):
        tr = [r for r in states if r["scenario"] == sc and r["split"] == "train"]
        outer = [r for r in states if r["scenario"] == sc and r["split"] in ("val", "validation")]
        tr = sorted(tr, key=lambda r: digest(r["source_initial_state_id"]))
        state_split[sc] = {"train": [r["state_uid"] for r in tr[16:]],
                           "dev": [r["state_uid"] for r in tr[:16]],
                           "test": [r["state_uid"] for r in outer]}
        assert len(state_split[sc]["train"]) == 48
        assert len(state_split[sc]["dev"]) == 16
        assert len(state_split[sc]["test"]) == 16
    report = {"frozen_before_outcome_inspection": True, "method": "12 farthest-point Voronoi eta groups; 8 train, 2 dev, 2 test by hash of coordinate-only medoid UID",
              "eta_groups": {str(g): {"medoid_eta_uid": eids[medoid[g]], "size": int(np.sum(group == g)), "split": assignment[g]} for g in range(12)},
              "eta_uids": eta_split, "state_uids": state_split,
              "eta_coord_center": center.tolist(), "eta_coord_radius": radius.tolist(),
              "test_to_train_eta_distance_normalized": {"min": float(dtest.min()), "median": float(np.median(dtest)), "p75": float(np.percentile(dtest, 75))},
              "raw_eta_coordinates_shared_between_scenes": True,
              "cross_scene_caveat": "same raw coordinate system, not a claim that identical numeric eta must be effective in both scenes"}
    dump_json("spatial_source_split.json", report)
    print(json.dumps({"eta_counts": {p: len(v) for p, v in eta_split.items()},
                      "state_counts": {sc: {p: len(v) for p, v in parts.items()} for sc, parts in state_split.items()},
                      "eta_test_distance": report["test_to_train_eta_distance_normalized"]}, indent=2))


if __name__ == "__main__":
    main()
