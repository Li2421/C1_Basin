"""Small non-neural state×eta control; all tuning uses TRAIN-family inner holdout.

This is a diagnostic comparator, not a replacement architecture. It uses only
physical initial state and exact eta coordinates; no Ring test labels enter fit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from offline_baselines import K16, ROOT, OUT, dump_json, ring_matrices


def features(physical: dict) -> np.ndarray:
    # Four agents, each with (x,y,vx,vy). No outcome-derived features.
    return np.asarray(physical["positions"] + physical["velocities"], float).reshape(-1)


def state_features() -> dict[str, np.ndarray]:
    table = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                          columns=["scenario", "state_uid", "structured_state"]).to_pylist()
    out = {}
    for row in table:
        if row["scenario"] == "ring_exchange":
            out[row["state_uid"]] = features(json.loads(json.loads(row["structured_state"])))
    fresh = json.loads((ROOT / "diagnostics/orthoflow3_ring_revision_v1/fresh_test_manifests.json").read_text())["states"]
    for row in fresh:
        if row["scenario"] == "ring_exchange":
            assert row["state_uid"] not in out, "fresh K16 source leaked into audited dataset"
            out[row["state_uid"]] = features(row["physical"])
    return out


def predict(x_state_train, y_train, z_eta_train, x_state_query, z_eta_query, k_state: int, h_eta: float):
    # Equal eta panel is measured on every train state. Smooth each source
    # state's binary B15 field in eta, then average nearby source states.
    eta_weight = np.exp(-cdist(z_eta_query, z_eta_train, "sqeuclidean") / (2 * h_eta**2))
    eta_weight /= np.maximum(np.sum(eta_weight, axis=1, keepdims=True), 1e-12)
    y_filled = np.where(np.isfinite(y_train), y_train, np.nanmean(y_train, axis=0)[None, :])
    field = eta_weight @ y_filled.T  # query eta × train states
    d = cdist(x_state_query, x_state_train, "euclidean")
    closest = np.argsort(d, axis=1)[:, :k_state]
    result = np.empty((len(x_state_query), len(z_eta_query)), float)
    for i, neighbors in enumerate(closest):
        # Inverse-distance weights preserve a continuous local estimate.
        w = 1.0 / np.maximum(d[i, neighbors], 1e-5)
        w /= w.sum()
        result[i] = field[:, neighbors] @ w
    return result


def train_inner_split(states: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(range(len(states)), key=lambda i: hashlib.sha256(states[i].encode()).hexdigest())
    dev = np.asarray(sorted(ordered[:16]), int)
    tr = np.asarray(sorted(ordered[16:]), int)
    assert len(tr) == 48 and len(dev) == 16
    return tr, dev


def main():
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version"]).to_pylist()
    ring = [r for r in labels if r["scenario"] == "ring_exchange"]
    assert all(r["label_semantics_version"] == "ring_current_safety_v2" for r in ring)
    train, val = ring_matrices(labels)
    feat = state_features()
    tr_ids, va_ids = train["states"], val["states"]
    assert not set(tr_ids) & set(va_ids)
    x = np.stack([feat[s] for s in tr_ids])
    mu, sd = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
    x = (x - mu) / sd
    xv = (np.stack([feat[s] for s in va_ids]) - mu) / sd
    inner_tr, inner_dev = train_inner_split(tr_ids)
    valid = np.isfinite(train["y"][inner_dev])
    scans = []
    for k_state in (1, 3, 5, 10, 20, 48):
        for h_eta in (0.04, 0.07, 0.10, 0.15, 0.22, 0.33, 0.5):
            pred = predict(x[inner_tr], train["y"][inner_tr], train["z"], x[inner_dev], train["z"], k_state, h_eta)
            brier = float(np.mean((pred[valid] - train["y"][inner_dev][valid]) ** 2))
            scans.append({"k_state": k_state, "eta_bandwidth": h_eta, "inner_dev_brier": brier})
    scans.sort(key=lambda r: (r["inner_dev_brier"], r["k_state"], r["eta_bandwidth"]))
    best = scans[0]
    with (OUT / "state_eta_local_inner_scan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scans[0])); w.writeheader(); w.writerows(scans)
    val_pred = predict(x, train["y"], train["z"], xv, train["z"], best["k_state"], best["eta_bandwidth"])
    va_mask = np.isfinite(val["y"])
    val_brier = float(np.mean((val_pred[va_mask] - val["y"][va_mask]) ** 2))
    eta_only = np.load(OUT / "eta_only_kernel_model.npz")
    zeta = eta_only["z_train"]
    kernel = np.exp(-cdist(zeta, zeta, "sqeuclidean") / (2 * float(eta_only["bandwidth"]) ** 2))
    eta_only_pred = np.clip(float(eta_only["prior"]) + kernel @ eta_only["alpha"], 0, 1)
    eta_only_val_brier = float(np.mean((val["y"][va_mask] - np.broadcast_to(eta_only_pred, val["y"].shape)[va_mask]) ** 2))

    proposal_rows = json.loads((K16 / "frozen_proposals.json").read_text())["states"]
    outcomes = {r["state_uid"]: r for r in json.loads((K16 / "per_state_results.json").read_text())}
    detail = []
    for p in proposal_rows:
        sid = p["state_uid"]
        xx = ((feat[sid] - mu) / sd)[None, :]
        xyz = np.asarray([p["mean"]] + p["samples"])
        zz = (xyz - train["center"]) / train["radius"]
        scores = predict(x, train["y"], train["z"], xx, zz, best["k_state"], best["eta_bandwidth"])[0]
        q = np.asarray([outcomes[sid]["candidate_evidence"]["mean"]["Q16"]] +
                       [outcomes[sid]["candidate_evidence"][f"sample_{i}"]["Q16"] for i in range(1, 17)])
        idx = int(np.argmax(scores))
        old = int(p["critic_index"])
        detail.append({"state_uid": sid, "local_index": idx, "local_score": float(scores[idx]),
                       "local_q16": float(q[idx]), "local_b15": bool(q[idx] >= 15 / 16),
                       "old_q16": float(q[old]), "old_b15": bool(q[old] >= 15 / 16),
                       "oracle_b15": bool(np.any(q >= 15 / 16))})
    with (OUT / "ring_k16_state_eta_local.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
    summary = {"training_states": len(tr_ids), "inner_train": len(inner_tr), "inner_dev": len(inner_dev),
               "audited_validation_states": len(va_ids), "source_overlap": 0,
               "physical_feature_dim": len(mu), "model": "physical-state kNN + continuous eta RBF field",
               "selected_hyperparameters": best, "validation_brier_state_eta_local": val_brier,
               "validation_brier_eta_only_kernel": eta_only_val_brier,
               "ring_k16_states": len(detail), "ring_k16_local_b15": sum(r["local_b15"] for r in detail),
               "ring_k16_local_mean_q16": float(np.mean([r["local_q16"] for r in detail])),
               "ring_k16_old_critic_b15": sum(r["old_b15"] for r in detail),
               "ring_k16_rescue_vs_old": sum(r["local_b15"] and not r["old_b15"] for r in detail),
               "ring_k16_break_vs_old": sum(r["old_b15"] and not r["local_b15"] for r in detail),
               "test_status": "previously examined diagnostic only",
               "new_rollout": 0}
    dump_json("state_eta_local_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
