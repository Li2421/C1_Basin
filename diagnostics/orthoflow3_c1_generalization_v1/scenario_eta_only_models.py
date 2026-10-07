"""Freeze source-disjoint, target-trained continuous eta-only controls for all scenes."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, SCENARIOS, dump_json, kernel_predict, kernel_train


def scenario_matrix(labels, scenario):
    rs = [r for r in labels if r["scenario"] == scenario and r["robust_15of16"] is not None]
    if scenario == "ring_exchange":
        assert all(r["label_semantics_version"] == "ring_current_safety_v2" for r in rs)
    tr = [r for r in rs if r["split"] == "train"]
    va = [r for r in rs if r["split"] in ("val", "validation")]
    eids = sorted(set(r["eta_uid"] for r in tr) & set(r["eta_uid"] for r in va))
    coord = {r["eta_uid"]: np.asarray(json.loads(r["eta_raw"]), float) for r in tr}
    x = np.stack([coord[e] for e in eids])
    center = (x.min(axis=0) + x.max(axis=0)) / 2
    radius = np.maximum((x.max(axis=0) - x.min(axis=0)) / 2, 1e-6)
    z = (x - center) / radius
    eidx = {e: j for j, e in enumerate(eids)}

    def make(rows):
        sids = sorted({r["state_uid"] for r in rows})
        sidx = {s: i for i, s in enumerate(sids)}
        mat = np.full((len(sids), len(eids)), np.nan)
        for r in rows:
            if r["eta_uid"] in eidx:
                mat[sidx[r["state_uid"]], eidx[r["eta_uid"]]] = float(r["robust_15of16"])
        return sids, mat

    train_ids, y_train = make(tr)
    val_ids, y_val = make(va)
    assert not set(train_ids) & set(val_ids)
    return eids, z, center, radius, train_ids, val_ids, y_train, y_val


def main():
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version"]).to_pylist()
    report = {}
    for sc in SCENARIOS:
        eids, z, center, radius, train_ids, val_ids, yt, yv = scenario_matrix(labels, sc)
        valid = np.isfinite(yv)
        trials = []
        for bw in (0.04, 0.07, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75):
            for ridge in (0.001, 0.01, 0.1, 1.0):
                prior, alpha = kernel_train(z, yt, bw, ridge)
                p = kernel_predict(z, z, prior, alpha, bw)
                brier = float(np.mean((np.broadcast_to(p, yv.shape)[valid] - yv[valid]) ** 2))
                trials.append((brier, bw, ridge))
        brier, bw, ridge = min(trials)
        prior, alpha = kernel_train(z, yt, bw, ridge)
        np.savez(OUT / f"{sc}_eta_only_model.npz", z_train=z, center=center, radius=radius,
                 prior=prior, alpha=alpha, bandwidth=bw, ridge=ridge)
        report[sc] = {"train_source_states": len(train_ids), "val_source_states": len(val_ids),
                      "repeated_eta_probes": len(eids), "selected_bandwidth": bw,
                      "selected_ridge": ridge, "val_brier": brier,
                      "train_state_uids": train_ids, "val_state_uids": val_ids,
                      "label_target": "logical B15, not empirical Q16",
                      "model_path": str(OUT / f"{sc}_eta_only_model.npz")}
    dump_json("scenario_eta_only_manifest.json", {"scenarios": report, "new_rollout": 0,
                                                "selection": "each scenario's VAL source-family Brier only; no external K16 outcomes"})
    print(json.dumps({s: {k: v for k, v in report[s].items() if k not in ("train_state_uids", "val_state_uids")}
                      for s in SCENARIOS}, indent=2))


if __name__ == "__main__":
    main()
