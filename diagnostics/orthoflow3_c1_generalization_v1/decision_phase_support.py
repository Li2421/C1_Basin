"""Quantify mid-trajectory TRAIN versus fresh true-t0 physical state shift."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from offline_baselines import ROOT, OUT, dump_json


def decode(x):
    y = json.loads(x)
    return json.loads(y) if isinstance(y, str) else y


def feature(physical):
    return np.asarray(physical["positions"] + physical["velocities"], float).reshape(-1)


def percentiles(a):
    return {"min": float(np.min(a)), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}


def main():
    states = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                           columns=["scenario", "split", "timestep", "structured_state", "state_uid"]).to_pylist()
    fresh = json.loads((ROOT / "diagnostics/orthoflow3_ring_revision_v1/fresh_test_manifests.json").read_text())["states"]
    out = {}
    for sc in ("double_bottleneck", "four_way_intersection", "ring_exchange"):
        tr = [r for r in states if r["scenario"] == sc and r["split"] == "train"]
        va = [r for r in states if r["scenario"] == sc and r["split"] in ("val", "validation")]
        fr = [r for r in fresh if r["scenario"] == sc]
        x = np.stack([feature(decode(r["structured_state"])) for r in tr])
        y = np.stack([feature(decode(r["structured_state"])) for r in va])
        z = np.stack([feature(r["physical"]) for r in fr])
        mu, sd = x.mean(0), np.maximum(x.std(0), 0.05)
        xn, yn, zn = (x-mu)/sd, (y-mu)/sd, (z-mu)/sd
        val_d = cdist(yn, xn).min(axis=1)
        fresh_d = cdist(zn, xn).min(axis=1)
        out[sc] = {"audited_train_states": len(tr), "audited_val_states": len(va),
                   "fresh_true_t0_states": len(fr),
                   "audited_train_timestep": percentiles([r["timestep"] for r in tr]),
                   "audited_val_timestep": percentiles([r["timestep"] for r in va]),
                   "validation_to_train_nearest_physical16_distance": percentiles(val_d),
                   "fresh_t0_to_train_nearest_physical16_distance": percentiles(fresh_d),
                   "fresh_over_val_median_distance_ratio": float(np.median(fresh_d) / np.median(val_d))}
    dump_json("decision_phase_support.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
