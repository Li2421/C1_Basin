"""Does local H20 response separate the fresh Flow-v8 failure contrast?"""
from __future__ import annotations

import json

import numpy as np

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from diagnostics.orthoflow3_controller_intervention_generalization_v1.intervention import OLD
from .fresh_controller import DEST, FLOW
from .probe import load_scene, read, write


def run():
    data = load_scene("ring_exchange")
    fresh = read(DEST / "fresh_v8_detail.json")["candidate_Q16"]
    pairs = read(DEST / "pairs.json")
    states = read(OLD / "states.json")
    mapping = {(r["state_uid"], tuple(np.asarray(r["eta"], np.float32))): r for r in pairs}
    base = h20.rc.RichRuntime("ring_exchange")
    variant = h20.rc.RichRuntime("ring_exchange", FLOW)
    result = []
    val_ix = np.flatnonzero(data["split"] == "validation")
    for i, target in zip(val_ix, fresh):
        uid = data["rows"][i]["state_uid"]
        raw_eta = tuple(np.asarray(data["d"]["eta"][data["ids"][i]], np.float32))
        pair = mapping[(uid, raw_eta)]
        physical = states[pair["state_index"]]["physical"]
        obj = {"state_uid": uid, "physical": physical}
        a = h20.rc.cached(base, obj, pair["eta"])
        b = h20.rc.cached(variant, obj, pair["eta"])
        if not a["valid"] or not b["valid"]:
            result.append({"row_index": int(i), "valid": False,
                           "base_error": a.get("error"), "v8_error": b.get("error")})
            continue
        va = np.asarray(a["features"]["mean"], float)
        vb = np.asarray(b["features"]["mean"], float)
        scale = data["context_scale"]
        base_success = float(data["success"][0, i])
        v8_success = float(target["success"])
        result.append({"row_index": int(i), "valid": True,
                       "base_Q16_lower": base_success / 16,
                       "v8_Q16": v8_success / 16,
                       "absolute_Q_contrast_lower": abs(base_success-v8_success)/16,
                       "H20_normalized_L2": float(np.linalg.norm((va-vb)/scale)),
                       "H20_raw_L2": float(np.linalg.norm(va-vb)),
                       "base_v9_H20_normalized_L2": float(np.linalg.norm(
                           data["context"][0, i]-data["context"][1, i])),
                       "base_v7_H20_normalized_L2": float(np.linalg.norm(
                           data["context"][0, i]-data["context"][2, i]))})
    valid = [r for r in result if r["valid"]]
    strong = [r for r in valid if r["absolute_Q_contrast_lower"] >= .5]
    summary = {"valid_pairs": len(valid), "strong_Q_contrast_pairs": len(strong),
               "H20_distance_median_all": float(np.median([r["H20_normalized_L2"] for r in valid])),
               "H20_distance_median_strong_contrast": float(np.median(
                   [r["H20_normalized_L2"] for r in strong])) if strong else None,
               "strong_contrast_with_H20_distance_le_0_25": sum(r["H20_normalized_L2"] <= .25
                                                            for r in strong)}
    write(DEST / "fresh_H20_alias_audit.json", {"summary": summary, "pairs": result,
                                                 "new_task_rollout": 0})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
