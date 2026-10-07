"""Source labels only: matrix completeness and identifiable interaction evidence."""
from collections import Counter
from itertools import combinations
import numpy as np

from .data import OUT, CONTROLLERS, read, write


def main():
    data = dict(np.load(OUT / "dataset.npz"))
    pairs, states = read(OUT / "pairs.json"), read(OUT / "states.json")
    provenance = read(OUT / "dataset_provenance.json")
    rollouts = [u for r in provenance for u in r["rollout_uids"]]
    assert len(rollouts) == len(set(rollouts))
    ss, ff = data["standard_success"], data["standard_failure"]
    lo, hi = ss / 16., (16. - ff) / 16.
    result = {"no_target_labels_read": True, "duplicate_canonical_rollout_uids": 0,
              "source_family_train_val_overlap": 0, "per_controller": {}, "validation": {}}
    for ci, controller in enumerate(CONTROLLERS):
        parts = {}
        for split in ("train", "validation"):
            ii = np.flatnonzero(data["split"] == split)
            s, f = data["success"][ci, ii], data["failure"][ci, ii]
            n = s + f
            rows = []
            for state in states:
                if state["split"] != split:
                    continue
                jj = [i for i in ii if pairs[i]["state_uid"] == state["uid"]]
                assert len(jj) == 16
                positive = int((ss[ci, jj] >= 15).sum())
                negative = int((ff[ci, jj] >= 2).sum())
                rows.append({"state_uid": state["uid"], "source_group": state["source_group"],
                             "pairs": len(jj), "B15_confirmed": positive, "non_B15_confirmed": negative,
                             "B15_unresolved": 16 - positive - negative,
                             "discriminative_eligible": positive > 0 and negative > 0})
            parts[split] = {"states": len(rows), "pairs": len(ii), "observed_trials": int(n.sum()),
                            "trials_per_pair_quantiles": np.quantile(n, [0, .25, .5, .75, 1]).tolist(),
                            "empirical_Q_groups": {"zero": int((s == 0).sum()),
                                "one": int((f == 0).sum()), "intermediate": int(((s > 0) & (f > 0)).sum())},
                            "numerical_trials_excluded": int(data["numerical"][ci, ii].sum()),
                            "per_state": rows}
        result["per_controller"][controller] = parts
    va = np.flatnonzero(data["split"] == "validation")
    groups = {s["uid"]: [i for i in va if pairs[i]["state_uid"] == s["uid"]]
              for s in states if s["split"] == "validation"}
    def ordering(c, i, j):
        return 1 if lo[c, i] - hi[c, j] >= .25 else -1 if hi[c, i] - lo[c, j] <= -.25 else 0
    flips = []
    for state_uid, ii in groups.items():
        for a, b in combinations(range(3), 2):
            for i, j in combinations(ii, 2):
                sa, sb = ordering(a, i, j), ordering(b, i, j)
                if sa * sb < 0:
                    flips.append({"state_uid": state_uid, "controllers": [CONTROLLERS[a], CONTROLLERS[b]],
                                  "pair_indices": [int(i), int(j)], "ordering_signs": [sa, sb]})
    state_flips = Counter()
    for a, b in combinations(groups.values(), 2):
        for c in range(3):
            for i, j in combinations(range(16), 2):
                if ordering(c, a[i], a[j]) * ordering(c, b[i], b[j]) < 0:
                    state_flips[CONTROLLERS[c]] += 1
    result["validation"] = {"controller_reversals": len(flips),
                            "controller_reversal_families": len({r["state_uid"] for r in flips}),
                            "state_reversals_by_controller": dict(state_flips),
                            "margin": .25, "missing_numeric_outcomes": "bounded, never imputed",
                            "not_independent_samples": "reversals share families and eta; do not use their count as statistical sample size"}
    write(OUT / "source_evidence_audit.json", result)
    write(OUT / "certified_controller_reversals.json", flips)
    print(result["validation"])


if __name__ == "__main__":
    main()
