"""Direct test whether a local Flow function signature resolves Q aliases."""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.stats import spearmanr

from .controller_signature import DEST, PROTOCOL
from .train import OUT, read, write


def run():
    d = dict(np.load(OUT / "second_variant/triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    features = []
    for path in sorted((DEST / "features").glob("*.json")):
        features.extend(read(path))
    assert all(r["valid"] for r in features), [r for r in features if not r["valid"]]
    by = {(r["scene"], r["state_uid"], r["controller"]): np.asarray(r["features"]["mean"], np.float32)
          for r in features}
    comparisons = []
    for scene in ("toy_giveway", "ring_exchange"):
        train_states = sorted({r["state_uid"] for r in rows if r["scene"] == scene and r["split"] == "train"})
        names = ("base", "first") if scene == "toy_giveway" else ("base", "first", "second")
        reference = np.stack([by[(scene, uid, controller)] for uid in train_states for controller in names])
        scale = np.maximum(reference.std(0), .05)
        for i, r in enumerate(rows):
            if r["scene"] != scene: continue
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    ca, cb = ("base", "alt", "second")[a], ("base", "alt", "second")[b]
                    if d[ca + "_s"][i] + d[ca + "_f"][i] != 16 or d[cb + "_s"][i] + d[cb + "_f"][i] != 16:
                        continue
                    qa, qb = d[ca + "_s"][i] / 16, d[cb + "_s"][i] / 16
                    signature_distance = np.linalg.norm((by[(scene, r["state_uid"], names[a])] -
                                                         by[(scene, r["state_uid"], names[b])]) / scale) / np.sqrt(24)
                    comparisons.append({"scene": scene, "split": r["split"], "state_uid": r["state_uid"],
                                        "pair_index": i, "controller_a": names[a], "controller_b": names[b],
                                        "abs_Q16_change": abs(float(qa-qb)),
                                        "signature_distance": float(signature_distance)})
    with (DEST / "matched_q_distances.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    summary = {}
    for scene in ("toy_giveway", "ring_exchange"):
        z = [r for r in comparisons if r["scene"] == scene]
        big = [r for r in z if r["abs_Q16_change"] >= .75]
        summary[scene] = {"full_matched_comparisons": len(z), "large_Q_change": len(big),
                          "median_distance_all": float(np.median([r["signature_distance"] for r in z])),
                          "median_distance_large_Q": float(np.median([r["signature_distance"] for r in big])),
                          "large_Q_with_signature_distance_below_0.25": sum(r["signature_distance"] < .25 for r in big),
                          "distance_vs_abs_Q_spearman": float(spearmanr([r["signature_distance"] for r in z],
                                                                       [r["abs_Q16_change"] for r in z]).statistic)}
    write(DEST / "audit.json", {"source_only": True, "target_labels_used": False,
                                "protocol": PROTOCOL, "summary": summary})
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
