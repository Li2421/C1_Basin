"""Controller-response intervention contrast audit, source families only.

This is a low-capacity diagnostic, not the deployment critic. Hyperparameters
are fixed a priori. It asks whether short physical response features predict
the Q change caused by a matched future-Flow intervention.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .intervention import OUT, ROOT, read, write
from .rich_context import key, NAMES


def get_features(scene, state_uid, eta, alt_sha):
    p = OUT / "rich_response_cache" / f"{key(scene, state_uid, eta, alt_sha)}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.asarray(read(p)["features"]["mean"], float)


def rows():
    protocol = read(OUT / "protocol.json")
    pilot = list(csv.DictReader((OUT / "intervention_pairs.csv").open()))
    result = []
    for r in pilot:
        scene = r["scene"]
        eta = json.loads(r["eta"])
        alt_sha = protocol["profiles"][scene]["alternate_flow_sha256"]
        a = get_features(scene, r["state_uid"], eta, None)
        b = get_features(scene, r["state_uid"], eta, alt_sha)
        result.append({"scene": scene, "family": r["family"], "state_uid": r["state_uid"],
                       "eta_uid": r["eta_uid"], "eta": eta, "split": r["split"],
                       "base_q": float(r["base_Q16"]), "alt_q": float(r["alt_Q16"]),
                       "delta_q": float(r["delta_Q16"]), "a": a, "b": b,
                       "valid": int(r["alt_valid"]) == 16})
    swap = ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1"
    toy_protocol = read(swap / "protocol.json")
    toy_rows = list(csv.DictReader((swap / "controller_pair_results.csv").open()))
    family_order = sorted({r["source_group"] for r in toy_rows},
                          key=lambda f: __import__("hashlib").sha256(f"intervention-family-v1|{f}".encode()).hexdigest())
    toy_validation = set(family_order[:4])
    for r in toy_rows:
        eta = json.loads(r["eta"])
        a = get_features("toy_giveway", r["state_uid"], eta, None)
        b = get_features("toy_giveway", r["state_uid"], eta, toy_protocol["flow_sha256"]["1"])
        family = r["source_group"]
        # Exactly four source families are held out by a fixed hash rank.
        split = "validation" if family in toy_validation else "train"
        result.append({"scene": "toy_giveway", "family": family, "state_uid": r["state_uid"],
                       "eta_uid": r["eta_uid"], "eta": eta, "split": split,
                       "base_q": float(r["base_Q16"]), "alt_q": float(r["alternate_Q16"]),
                       "delta_q": float(r["delta_Q16"]), "a": a, "b": b, "valid": True})
    return result


def audit():
    data = [r for r in rows() if r["valid"]]
    train = [r for r in data if r["split"] == "train"]
    val = [r for r in data if r["split"] == "validation"]
    assert not {r["family"] for r in train} & {r["family"] for r in val}
    train_x = np.asarray([r["b"] - r["a"] for r in train])
    val_x = np.asarray([r["b"] - r["a"] for r in val])
    train_y = np.asarray([r["delta_q"] for r in train])
    val_y = np.asarray([r["delta_q"] for r in val])
    center = train_x.mean(0)
    scale = np.maximum(train_x.std(0), .05)
    x = (train_x - center) / scale
    xv = (val_x - center) / scale
    # Fixed ridge strength; no target-outcome model selection.
    lam = 10.
    coef = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ (train_y - train_y.mean()))
    pred = train_y.mean() + xv @ coef
    zero = np.zeros_like(val_y)
    pooled = np.full_like(val_y, train_y.mean())
    def score(a, b):
        return {"mae": float(np.mean(np.abs(a - b))),
                "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                "spearman": float(spearmanr(a, b).statistic) if len(set(np.round(a, 8))) > 1 and len(set(np.round(b, 8))) > 1 else None,
                "strong_sign_accuracy": float(np.mean(np.sign(a[np.abs(b) >= .25]) == np.sign(b[np.abs(b) >= .25]))) if np.any(np.abs(b) >= .25) else None}
    summary = {"train_pairs": len(train), "val_pairs": len(val),
               "train_families": len({r["family"] for r in train}),
               "val_families": len({r["family"] for r in val}),
               "feature_names": NAMES, "ridge_lambda": lam,
               "delta_q_abs_ge_0_25_train": int(np.sum(np.abs(train_y) >= .25)),
               "delta_q_abs_ge_0_25_val": int(np.sum(np.abs(val_y) >= .25)),
               "val": {"zero": score(zero, val_y), "pooled": score(pooled, val_y),
                       "rich_response": score(pred, val_y)},
               "scene_diagnostics": {},
               "interpretation": "source-family-heldout controller contrast only; not frozen LOSO K16"}
    for scene in sorted({r["scene"] for r in val}):
        mask = np.array([r["scene"] == scene for r in val])
        summary["scene_diagnostics"][scene] = {"pairs": int(mask.sum()),
                                                "strong_contrasts": int(np.sum(np.abs(val_y[mask]) >= .25)),
                                                "zero": score(zero[mask], val_y[mask]),
                                                "rich_response": score(pred[mask], val_y[mask])}
    by_state = defaultdict(list)
    for r in data:
        by_state[(r["scene"], r["state_uid"])].append(r)
    reversals = []
    for (scene, uid), group in by_state.items():
        for i in range(len(group)):
            for j in range(i):
                a, b = group[i], group[j]
                base = a["base_q"] - b["base_q"]
                alt = a["alt_q"] - b["alt_q"]
                if base * alt < 0 and min(abs(base), abs(alt)) >= .25:
                    reversals.append({"scene": scene, "state_uid": uid, "family": a["family"],
                                      "eta_a": a["eta_uid"], "eta_b": b["eta_uid"],
                                      "base_delta": base, "alt_delta": alt})
    summary["controller_ranking_reversals"] = len(reversals)
    summary["reversal_families"] = len({r["family"] for r in reversals})
    write("controller_contrast_probe.json", summary)
    write("controller_ranking_reversals.json", reversals)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit()
