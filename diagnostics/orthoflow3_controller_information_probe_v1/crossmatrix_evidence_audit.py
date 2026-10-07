"""Observed Q8 structure of the source TRAIN shared-eta controller matrix."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from shared_rollout_db.src.rollout_db import canonical, connect

from .crossmatrix_ring import DEST
from .probe import read, write


def run():
    pairs = read(DEST / "pairs.json")
    data = {}
    with connect(True) as db:
        for row in pairs:
            key = row["state_uid"], row["eta_uid"]
            item = {"state_uid": key[0], "eta_uid": key[1], "eta": row["eta"]}
            for controller, field in enumerate(("base_controller_uid", "alternate_controller_uid")):
                obs = db.execute("""SELECT seed_key,success,numerical_failure,conflict_quarantined,
                    compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                    (key[0], key[1], row[field])).fetchall()
                valid = [r for r in obs if r["seed_key"] in
                         {canonical({"future_index": i}) for i in range(8)}
                         and not r["numerical_failure"] and not r["conflict_quarantined"]
                         and r["compatibility_quality"] == "EXACT_REUSE"]
                item[f"c{controller}_n"] = len(valid)
                item[f"c{controller}_s"] = sum(int(r["success"]) for r in valid)
                item[f"c{controller}_q"] = item[f"c{controller}_s"] / len(valid) if valid else None
            data[key] = item
    selected = read(DEST / "protocol.json")["selected_eta_uid"]
    states = sorted({r["state_uid"] for r in pairs})
    eta_stats = []
    reversals = {}
    for controller in (0, 1):
        for eta_uid in selected:
            values = [data[(uid, eta_uid)][f"c{controller}_q"] for uid in states]
            x = np.asarray([v for v in values if v is not None])
            eta_stats.append({"controller": controller, "eta_uid": eta_uid,
                              "states_observed": len(x), "q8_mean": float(x.mean()),
                              "q8_std_across_states": float(x.std()),
                              "q8_le_0_5": int((x <= .5).sum()),
                              "q8_ge_0_875": int((x >= .875).sum()),
                              "clear_failure_and_high_success": bool((x <= .5).any() and (x >= .875).any())})
        strong_reversals = 0
        comparable = 0
        for a, b in combinations(selected, 2):
            for u, v in combinations(states, 2):
                du = data[(u, a)][f"c{controller}_q"] - data[(u, b)][f"c{controller}_q"]
                dv = data[(v, a)][f"c{controller}_q"] - data[(v, b)][f"c{controller}_q"]
                if abs(du) >= .25 and abs(dv) >= .25:
                    comparable += 1
                    strong_reversals += int(du * dv < 0)
        reversals[f"controller{controller}"] = {"strong_q8_state_order_reversals": strong_reversals,
                                                 "comparable_state_eta_pair_cases": comparable}
    result = {"source_TRAIN_only": True, "target_controller2_outcomes_read": False,
              "new_rollout": 0, "eta_stats": eta_stats, "reversals": reversals,
              "observed_valid_trials": sum(data[k][f"c{c}_n"] for k in data for c in (0, 1)),
              "matrix_cells": len(data)}
    write(DEST / "matrix_evidence_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
