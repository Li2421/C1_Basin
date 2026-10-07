"""Separate unseen-controller outcomes from seen-controller source validation."""
from __future__ import annotations

import csv
from itertools import combinations

import numpy as np

from .data_geometry_audit import certified_sign
from .probe import OUT, SCENES, load_scene, observed_nll, read


def run():
    rows = []
    for scene in SCENES[1:]:
        data = load_scene(scene)
        ix = np.flatnonzero(data["split"] == "validation")
        by_state = {}
        for i in ix:
            by_state.setdefault(data["rows"][i]["state_uid"], []).append(i)
        for kind in ("fingerprint_eta", "fingerprint_full"):
            for heldout in (False, True):
                label = kind + ("_heldout_controller2" if heldout else "")
                for seed in (17, 23, 41):
                    dest = OUT / "models" / scene / label / f"seed{seed}"
                    if not (dest / "summary.json").exists():
                        raise FileNotFoundError(dest)
                    z = np.load(dest / "validation_predictions.npz")["logits"]
                    nll = observed_nll(z[2, ix], data["success"][2, ix], data["failure"][2, ix])
                    eligible = success = 0
                    for pairs in by_state.values():
                        robust = data["success"][2, pairs] >= 15
                        if robust.any():
                            eligible += 1
                            success += int(robust[np.argmax(z[2, pairs])])
                    reversal_total = reversal_correct = 0
                    for pairs in by_state.values():
                        for a, b in combinations(pairs, 2):
                            for other in (0, 1):
                                t0 = certified_sign(data, other, a, b)
                                t2 = certified_sign(data, 2, a, b)
                                if t0 * t2 < 0:
                                    reversal_total += 1
                                    reversal_correct += int(np.sign(z[other, a] - z[other, b]) == t0 and
                                                            np.sign(z[2, a] - z[2, b]) == t2)
                    rows.append({"scene": scene, "kind": kind, "controller2_labels_seen_in_TRAIN": not heldout,
                                 "seed": seed, "controller2_observed_NLL": nll,
                                 "controller2_B15_selected": success,
                                 "controller2_B15_available": eligible,
                                 "controller2_certified_reversals": reversal_total,
                                 "controller2_reversals_predicted": reversal_correct})
    with (OUT / "held_controller_evaluation.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> held_controller_evaluation.csv")


if __name__ == "__main__":
    run()
