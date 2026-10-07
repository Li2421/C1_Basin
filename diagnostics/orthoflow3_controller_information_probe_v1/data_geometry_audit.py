"""Audit matched-controller interaction evidence without model predictions."""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from .probe import OUT, SCENES, load_scene, write


def certified_sign(data, controller, a, b):
    sa, fa = data["success"][controller, a], data["failure"][controller, a]
    sb, fb = data["success"][controller, b], data["failure"][controller, b]
    lower = (sa - (16 - fb)) / 16
    upper = ((16 - fa) - sb) / 16
    return 1 if lower > 0 else -1 if upper < 0 else 0


def count_reversals(data, split):
    by_state = {}
    for i in np.flatnonzero(data["split"] == split):
        by_state.setdefault(data["rows"][i]["state_uid"], []).append(i)
    total = comparable = 0
    state_hits = set()
    for uid, pairs in by_state.items():
        for a, b in combinations(pairs, 2):
            signs = [certified_sign(data, c, a, b) for c in range(3)]
            for c0, c1 in combinations(range(3), 2):
                comparable += int(signs[c0] != 0 and signs[c1] != 0)
                if signs[c0] * signs[c1] < 0:
                    total += 1
                    state_hits.add(uid)
    return {"certified_reversals": total, "comparable_controller_eta_orders": comparable,
            "states_with_reversals": len(state_hits), "states": len(by_state)}


def run():
    result = {}
    for scene in SCENES:
        data = load_scene(scene)
        tr = data["split"] == "train"
        va = data["split"] == "validation"
        te = data["eta"][tr]
        ve = data["eta"][va]
        distance = np.linalg.norm(ve[:, None, :] - te[None, :, :], axis=-1).min(axis=1)
        tr_uids = {r["state_uid"] for r in np.asarray(data["rows"], dtype=object)[tr]}
        va_uids = {r["state_uid"] for r in np.asarray(data["rows"], dtype=object)[va]}
        assert not tr_uids & va_uids
        full = data["success"] + data["failure"]
        result[scene] = {
            "train": count_reversals(data, "train"),
            "validation": count_reversals(data, "validation"),
            "train_pairs": int(tr.sum()), "validation_pairs": int(va.sum()),
            "train_unique_eta": len({tuple(x) for x in np.round(data["eta"][tr], 8)}),
            "validation_to_nearest_train_eta_norm": {
                "median": float(np.median(distance)), "p90": float(np.percentile(distance, 90)),
                "max": float(distance.max()),
            },
            "valid_controller_pairs_train": int((full[:, tr] > 0).sum()),
            "full16_controller_pairs_train": int((full[:, tr] == 16).sum()),
            "partial_controller_pairs_train": int(((full[:, tr] > 0) & (full[:, tr] < 16)).sum()),
        }
    write(OUT / "data_geometry_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
