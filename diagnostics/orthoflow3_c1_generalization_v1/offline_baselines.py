"""Audited, rollout-free eta-only and state-interaction controls.

Run from Basin_C1 with the project venv. Ring v2 labels are binary *logical*
B15 evidence (early stopping is allowed); they are not empirical Q16 targets.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/orthoflow3_basin_dataset_v2_audited"
K16 = ROOT / "diagnostics/orthoflow3_ring_k16_diagnostic_v1"
SCENARIOS = ("double_bottleneck", "four_way_intersection", "ring_exchange")


def dump_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def dump_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_labels() -> tuple[list[dict], list[dict]]:
    state_cols = ["scenario", "split", "state_id", "state_uid", "source_initial_state_id", "parent_episode_id"]
    label_cols = ["scenario", "split", "state_id", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version", "controller_uid", "numerical_failure_count"]
    states = pq.read_table(DATA / "states.parquet", columns=state_cols).to_pylist()
    labels = pq.read_table(DATA / "eta_labels.parquet", columns=label_cols).to_pylist()
    # Canonical pair uniqueness; exact duplicate keys are a dataset construction error.
    keys = [(x["scenario"], x["state_uid"], x["eta_uid"], x["controller_uid"]) for x in labels]
    assert len(keys) == len(set(keys)), "duplicate canonical pair in audited dataset"
    ring = [x for x in labels if x["scenario"] == "ring_exchange"]
    assert ring and all(x["label_semantics_version"] == "ring_current_safety_v2" for x in ring)
    return states, labels


def ring_matrices(labels: list[dict]) -> tuple[dict, dict]:
    train_rows = [x for x in labels if x["scenario"] == "ring_exchange" and x["split"] == "train" and x["robust_15of16"] is not None]
    val_rows = [x for x in labels if x["scenario"] == "ring_exchange" and x["split"] == "validation" and x["robust_15of16"] is not None]
    if not val_rows:
        val_rows = [x for x in labels if x["scenario"] == "ring_exchange" and x["split"] == "val" and x["robust_15of16"] is not None]
    assert train_rows and val_rows
    eta_ids = sorted(set(x["eta_uid"] for x in train_rows))
    coord = {x["eta_uid"]: np.asarray(json.loads(x["eta_raw"]), float) for x in train_rows}
    x = np.stack([coord[e] for e in eta_ids])
    center = (x.min(axis=0) + x.max(axis=0)) / 2
    radius = np.maximum((x.max(axis=0) - x.min(axis=0)) / 2, 1e-6)
    z = (x - center) / radius
    eidx = {e: j for j, e in enumerate(eta_ids)}

    def matrix(rows: list[dict]) -> tuple[list[str], np.ndarray]:
        states = sorted(set(r["state_uid"] for r in rows))
        sidx = {s: j for j, s in enumerate(states)}
        y = np.full((len(states), len(eta_ids)), np.nan)
        for r in rows:
            if r["eta_uid"] in eidx:
                y[sidx[r["state_uid"]], eidx[r["eta_uid"]]] = float(r["robust_15of16"])
        return states, y

    tr_states, tr_y = matrix(train_rows)
    va_states, va_y = matrix(val_rows)
    assert len(tr_states) >= 50 and len(va_states) >= 10
    return {"eta_ids": eta_ids, "x": x, "z": z, "center": center, "radius": radius,
            "states": tr_states, "y": tr_y}, {"states": va_states, "y": va_y}


def kernel_train(z: np.ndarray, y: np.ndarray, bandwidth: float, ridge: float):
    mean_per_eta = np.nanmean(y, axis=0)
    prior = float(np.nanmean(y))
    k = np.exp(-cdist(z, z, "sqeuclidean") / (2 * bandwidth**2))
    alpha = np.linalg.solve(k + ridge * np.eye(len(z)), mean_per_eta - prior)
    return prior, alpha


def kernel_predict(z_train: np.ndarray, z_query: np.ndarray, prior: float, alpha: np.ndarray, bandwidth: float):
    kernel = np.exp(-cdist(z_query, z_train, "sqeuclidean") / (2 * bandwidth**2))
    return np.clip(prior + kernel @ alpha, 0.0, 1.0)


def select_eta_only(train: dict, val: dict) -> dict:
    z, tr_y, va_y = train["z"], train["y"], val["y"]
    valid = np.isfinite(va_y)
    scans = []
    for h in (0.04, 0.07, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75):
        for ridge in (0.001, 0.01, 0.1, 1.0):
            prior, alpha = kernel_train(z, tr_y, h, ridge)
            pred = kernel_predict(z, z, prior, alpha, h)
            brier = float(np.mean((va_y[valid] - np.broadcast_to(pred, va_y.shape)[valid]) ** 2))
            scans.append({"bandwidth": h, "ridge": ridge, "val_brier": brier})
    scans.sort(key=lambda r: (r["val_brier"], r["bandwidth"], r["ridge"]))
    best = scans[0]
    dump_csv("eta_only_val_scan.csv", scans)
    prior, alpha = kernel_train(z, tr_y, best["bandwidth"], best["ridge"])
    return {"bandwidth": best["bandwidth"], "ridge": best["ridge"], "val_brier": best["val_brier"],
            "prior": prior, "alpha": alpha}


def reversal_audit(labels: list[dict], scenario: str) -> dict:
    rows = [r for r in labels if r["scenario"] == scenario and r["robust_15of16"] is not None]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] in ("validation", "val")]
    eta_ids = sorted(set(r["eta_uid"] for r in train))
    s_train = sorted(set(r["state_uid"] for r in train))
    s_val = sorted(set(r["state_uid"] for r in val))
    eidx = {e: i for i, e in enumerate(eta_ids)}

    def make(rs: list[dict], sids: list[str]) -> np.ndarray:
        mat = np.full((len(sids), len(eta_ids)), np.nan)
        idx = {s: i for i, s in enumerate(sids)}
        for r in rs:
            if r["eta_uid"] in eidx:
                mat[idx[r["state_uid"]], eidx[r["eta_uid"]]] = float(r["robust_15of16"])
        return mat

    a, b = make(train, s_train), make(val, s_val)
    # Only exact eta shared over at least eight source-distinct train states and
    # four validation states; outcome-based pair selection uses TRAIN only.
    eligible = np.where((np.isfinite(a).sum(axis=0) >= 8) & (np.isfinite(b).sum(axis=0) >= 4))[0]
    pos = (a[:, eligible] == 1).astype(float)
    neg = (a[:, eligible] == 0).astype(float)
    wins = pos.T @ neg
    ranked = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            forward = int(wins[i, j])
            reverse = int(wins[j, i])
            if min(forward, reverse) >= 2:
                ranked.append((min(forward, reverse), forward + reverse, i, j))
    ranked.sort(reverse=True)
    selected = []
    for score, total, i, j in ranked[:10]:
        ii, jj = int(eligible[i]), int(eligible[j])
        first, second = b[:, ii], b[:, jj]
        observed = np.isfinite(first) & np.isfinite(second)
        fw = int(np.sum((first > second) & observed))
        rv = int(np.sum((first < second) & observed))
        selected.append({"eta_a_uid": eta_ids[ii], "eta_b_uid": eta_ids[jj],
                         "train_a_better": int(wins[i, j]), "train_b_better": int(wins[j, i]),
                         "val_a_better": fw, "val_b_better": rv,
                         "val_both_directions": bool(fw > 0 and rv > 0),
                         "val_states_compared": int(np.sum(observed))})
    dump_csv(f"{scenario}_reversal_pairs.csv", selected)
    return {"scenario": scenario, "train_states": len(s_train), "val_states": len(s_val),
            "repeated_eligible_eta": len(eligible), "train_reversal_pair_count_min2_each": len(ranked),
            "top10_val_reversal_count": int(sum(r["val_both_directions"] for r in selected))}


def evaluate_ring_k16(train: dict, fit: dict) -> dict:
    proposals = json.loads((K16 / "frozen_proposals.json").read_text())["states"]
    results = json.loads((K16 / "per_state_results.json").read_text())
    by_state = {r["state_uid"]: r for r in results}
    assert len(proposals) == len(results) == 60
    center, radius = train["center"], train["radius"]
    ztrain = train["z"]
    mean_per_eta = np.nanmean(train["y"], axis=0)
    rows = []
    for p in proposals:
        r = by_state[p["state_uid"]]
        xyz = np.asarray([p["mean"]] + p["samples"], float)
        z = (xyz - center) / radius
        kscore = kernel_predict(ztrain, z, fit["prior"], fit["alpha"], fit["bandwidth"])
        d = cdist(z, ztrain)
        near = np.argmin(d, axis=1)
        near_score = mean_per_eta[near]
        true = np.asarray([r["candidate_evidence"]["mean"]["Q16"]] +
                          [r["candidate_evidence"][f"sample_{i}"]["Q16"] for i in range(1, 17)], float)
        robust = true >= 15 / 16
        assert len(xyz) == len(true) == len(p["critic_scores"]) == 17
        assert int(np.argmax(p["critic_scores"])) == int(r["critic_index"])
        scores = {"frozen_critic": np.asarray(p["critic_scores"]),
                  "eta_only_kernel": kscore, "eta_only_nearest": near_score}
        chosen = {name: int(np.argmax(sc)) for name, sc in scores.items()}
        rows.append({"state_uid": p["state_uid"], "state_id": p["state_id"],
                     "oracle_b15": bool(np.any(robust)), "oracle_best_q16": float(np.max(true)),
                     "frozen_critic_idx": chosen["frozen_critic"],
                     "frozen_critic_b15": bool(robust[chosen["frozen_critic"]]),
                     "frozen_critic_q16": float(true[chosen["frozen_critic"]]),
                     "eta_only_kernel_idx": chosen["eta_only_kernel"],
                     "eta_only_kernel_b15": bool(robust[chosen["eta_only_kernel"]]),
                     "eta_only_kernel_q16": float(true[chosen["eta_only_kernel"]]),
                     "eta_only_nearest_idx": chosen["eta_only_nearest"],
                     "eta_only_nearest_b15": bool(robust[chosen["eta_only_nearest"]]),
                     "eta_only_nearest_q16": float(true[chosen["eta_only_nearest"]]),
                     "selected_kernel_eta_distance": float(d[chosen["eta_only_kernel"], near[chosen["eta_only_kernel"]]]),
                     "min_candidate_train_eta_distance": float(np.min(d)),
                     "median_candidate_train_eta_distance": float(np.median(np.min(d, axis=1)))})
    dump_csv("ring_k16_eta_only_comparison.csv", rows)
    summary = {"states": len(rows), "oracle_b15": sum(r["oracle_b15"] for r in rows)}
    for name in ("frozen_critic", "eta_only_kernel", "eta_only_nearest"):
        summary[name] = {"b15": sum(r[f"{name}_b15"] for r in rows),
                         "mean_q16": float(np.mean([r[f"{name}_q16"] for r in rows]))}
    for name in ("eta_only_kernel", "eta_only_nearest"):
        summary[name]["rescue_vs_frozen_critic"] = sum(r[f"{name}_b15"] and not r["frozen_critic_b15"] for r in rows)
        summary[name]["break_vs_frozen_critic"] = sum(r["frozen_critic_b15"] and not r[f"{name}_b15"] for r in rows)
    summary["eta_distance_to_train"] = {"min": min(r["min_candidate_train_eta_distance"] for r in rows),
                                        "median_of_state_medians": float(np.median([r["median_candidate_train_eta_distance"] for r in rows]))}
    return summary


def main():
    states, labels = load_labels()
    source_sets = defaultdict(lambda: defaultdict(set))
    for s in states:
        source_sets[s["scenario"]][s["split"]].add(s["source_initial_state_id"])
    source_audit = {}
    for scenario, by_split in source_sets.items():
        splits = list(by_split)
        cross = sum(len(by_split[splits[i]] & by_split[splits[j]]) for i in range(len(splits)) for j in range(i + 1, len(splits)))
        assert cross == 0
        source_audit[scenario] = {"source_groups_per_split": {k: len(v) for k, v in by_split.items()}, "cross_split_overlap": cross}
    train, val = ring_matrices(labels)
    fit = select_eta_only(train, val)
    fit_audit = {k: v for k, v in fit.items() if k != "alpha"}
    fit_audit.update({"training_state_uids": train["states"], "validation_state_uids": val["states"],
                      "eta_uids": train["eta_ids"], "eta_coord_center": train["center"].tolist(),
                      "eta_coord_radius": train["radius"].tolist(), "label_target": "logical B15, not Q16"})
    dump_json("eta_only_model_manifest.json", fit_audit)
    np.savez(OUT / "eta_only_kernel_model.npz", z_train=train["z"], alpha=fit["alpha"], prior=fit["prior"],
             bandwidth=fit["bandwidth"], center=train["center"], radius=train["radius"])
    reversal = [reversal_audit(labels, s) for s in SCENARIOS]
    k16 = evaluate_ring_k16(train, fit)
    summary = {"source_split_audit": source_audit, "reversal_audit": reversal,
               "ring_eta_only_model": {"bandwidth": fit["bandwidth"], "ridge": fit["ridge"], "val_brier": fit["val_brier"]},
               "ring_k16_diagnostic": k16, "new_rollout": 0,
               "ring_k16_status": "previously examined diagnostic; independent confirmation remains pending"}
    dump_json("offline_summary.json", summary)
    print(json.dumps({"reversal_audit": reversal, "ring_k16_diagnostic": k16}, indent=2))


if __name__ == "__main__":
    main()
