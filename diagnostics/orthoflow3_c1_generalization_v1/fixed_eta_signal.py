"""Does Ring h predict B15 variation for the *same* fixed eta across families?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from offline_baselines import ROOT, OUT, dump_json, ring_matrices
from state_eta_local_control import state_features, train_inner_split


def h_features() -> dict[str, np.ndarray]:
    rows = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                         columns=["scenario", "state_uid", "conditioning"]).to_pylist()
    out = {}
    for r in rows:
        if r["scenario"] == "ring_exchange":
            out[r["state_uid"]] = np.asarray(json.loads(json.loads(r["conditioning"]))["flat"], float)
    return out


def predict_knn(x_train, y_train, x_query, k: int):
    d = cdist(x_query, x_train)
    near = np.argsort(d, axis=1)[:, :k]
    prior = np.nanmean(y_train, axis=0)
    out = np.empty((len(x_query), y_train.shape[1]))
    for i, j in enumerate(near):
        w = 1.0 / np.maximum(d[i, j], 1e-4)
        w /= w.sum()
        vals = y_train[j]
        good = np.isfinite(vals)
        num = np.sum(np.where(good, vals, 0.0) * w[:, None], axis=0)
        den = np.sum(good * w[:, None], axis=0)
        out[i] = np.divide(num, den, out=prior.copy(), where=den > 0)
    return out


def main():
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version"]).to_pylist()
    ring = [r for r in labels if r["scenario"] == "ring_exchange"]
    assert all(r["label_semantics_version"] == "ring_current_safety_v2" for r in ring)
    train, val = ring_matrices(labels)
    prevalence = np.nanmean(train["y"], axis=0)
    panel = np.where((prevalence >= 0.2) & (prevalence <= 0.8))[0]
    assert len(panel) >= 30
    inner_train, inner_dev = train_inner_split(train["states"])
    results = {}
    for key, fmap in (("critic_h100", h_features()), ("physical16", state_features())):
        x = np.stack([fmap[s] for s in train["states"]])
        xv = np.stack([fmap[s] for s in val["states"]])
        mu, sd = x[inner_train].mean(0), np.maximum(x[inner_train].std(0), 0.05)
        xi, xd = (x[inner_train] - mu) / sd, (x[inner_dev] - mu) / sd
        yi, yd = train["y"][inner_train][:, panel], train["y"][inner_dev][:, panel]
        valid = np.isfinite(yd)
        prior = np.nanmean(yi, axis=0)
        scans = {}
        scans["constant"] = float(np.mean((yd[valid] - np.broadcast_to(prior, yd.shape)[valid]) ** 2))
        for k in (1, 3, 5, 10, 20, 48):
            scans[str(k)] = float(np.mean((yd[valid] - predict_knn(xi, yi, xd, k)[valid]) ** 2))
        # Select k via TRAIN-family inner-dev only, then refit 64 TRAIN families.
        best_k = min((1, 3, 5, 10, 20, 48), key=lambda k: (scans[str(k)], k))
        mu, sd = x.mean(0), np.maximum(x.std(0), 0.05)
        ytr, yv = train["y"][:, panel], val["y"][:, panel]
        valid = np.isfinite(yv)
        ptr = np.nanmean(ytr, axis=0)
        p = predict_knn((x - mu) / sd, ytr, (xv - mu) / sd, best_k)
        results[key] = {"feature_dim": x.shape[1], "train_selected_eta_count": len(panel),
                        "inner_dev_brier_by_k": scans, "selected_k": best_k,
                        "val_constant_brier": float(np.mean((yv[valid] - np.broadcast_to(ptr, yv.shape)[valid]) ** 2)),
                        "val_knn_brier": float(np.mean((yv[valid] - p[valid]) ** 2)),
                        "val_pair_count": int(np.sum(valid))}
    summary = {"scenario": "ring_exchange", "label_semantics": "ring_current_safety_v2",
               "probe_selection": "TRAIN-only B15 prevalence 20-80%", "probe_count": len(panel),
               "probe_eta_uids": [train["eta_ids"][i] for i in panel],
               "source_split_overlap": 0, "results": results, "new_rollout": 0}
    dump_json("ring_fixed_eta_state_signal.json", summary)
    print(json.dumps({"probe_count": len(panel), "results": results}, indent=2))


if __name__ == "__main__":
    main()
