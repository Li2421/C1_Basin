"""Audit natural fixed-eta ranking reversals on source-isolated TEST states.

The eta panels were chosen using TRAIN outcomes and TEST coverage metadata only.
No model predictions or new rollouts enter this calculation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent
SNAP = ROOT.parent / "orthoflow3_fixed_eta_state_to_q_identifiability_v2"
MARGIN = 0.25


def interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def panel(scene: str) -> set[str]:
    with (SNAP / f"fixed_eta_panel_{scene.lower()}.csv").open() as f:
        return {row["eta_uid"] for row in csv.DictReader(f)}


def audit_scene(scene: str, snapshot: list[dict]) -> tuple[dict, list[dict]]:
    eta_keep = panel(scene)
    train_q = defaultdict(list)
    for r in snapshot:
        if (r["scenario"] == scene and r["state_split"] == "train"
                and r["eta_uid"] in eta_keep and r["n_trials"] >= 16
                and r["evidence_class"] == "SEED_EXACT"):
            train_q[r["eta_uid"]].append(r["empirical_q"])
    train_means = {e: sum(v) / len(v) for e, v in train_q.items()}
    data = [
        r for r in snapshot
        if r["scenario"] == scene
        and r["state_split"] == "test"
        and r["eta_uid"] in eta_keep
        and r["n_trials"] >= 16
        and r["seed_identities_known"]
        and r["evidence_class"] == "SEED_EXACT"
    ]
    ctl = {r["controller_uid"] for r in data}
    assert len(ctl) == 1, (scene, ctl)
    by_state = defaultdict(dict)
    state_family = {}
    for r in data:
        s, e = r["state_uid"], r["eta_uid"]
        assert e not in by_state[s], (scene, s, e)
        by_state[s][e] = r
        state_family[s] = r["source_group"]
    assert len(set(state_family.values())) == len(state_family), scene

    pair_signs: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: {-1: [], 1: []})
    pair_signs_cp: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: {-1: [], 1: []})
    informative = 0
    total_state_eta_comparisons = 0
    train_eta_order_correct = 0
    train_eta_order_evaluable = 0
    for state, erows in by_state.items():
        for e1, e2 in combinations(sorted(erows), 2):
            total_state_eta_comparisons += 1
            r1, r2 = erows[e1], erows[e2]
            d = r1["empirical_q"] - r2["empirical_q"]
            if abs(d) < MARGIN:
                continue
            informative += 1
            sign = 1 if d > 0 else -1
            if e1 in train_means and e2 in train_means:
                train_eta_order_evaluable += 1
                pred = train_means[e1] - train_means[e2]
                train_eta_order_correct += float(pred * sign > 0) + 0.5 * float(pred == 0)
            item = {
                "state_uid": state, "source_group": state_family[state],
                "q1": r1["empirical_q"], "q2": r2["empirical_q"],
                "n1": r1["n_trials"], "n2": r2["n_trials"],
                "k1": r1["n_success"], "k2": r2["n_success"],
            }
            pair_signs[(e1, e2)][sign].append(item)
            ci1, ci2 = interval(r1["n_success"], r1["n_trials"]), interval(r2["n_success"], r2["n_trials"])
            certified = ci1[0] > ci2[1] if sign == 1 else ci2[0] > ci1[1]
            if certified:
                pair_signs_cp[(e1, e2)][sign].append(item)

    reversals = []
    reversal_quadruples = 0
    involved_states = set()
    eta_only_correct = 0
    for (e1, e2), signs in pair_signs.items():
        pos, neg = signs[1], signs[-1]
        eta_only_correct += max(len(pos), len(neg))
        if not pos or not neg:
            continue
        reversal_quadruples += len(pos) * len(neg)
        involved_states.update(x["state_uid"] for x in pos + neg)
        pos_witness = max(pos, key=lambda x: x["q1"] - x["q2"])
        neg_witness = max(neg, key=lambda x: x["q2"] - x["q1"])
        cp = pair_signs_cp[(e1, e2)]
        reversals.append({
            "scenario": scene, "eta_uid_1": e1, "eta_uid_2": e2,
            "positive_states": len(pos), "negative_states": len(neg),
            "empirical_quadruples": len(pos) * len(neg),
            "cp_certified_opposite_sides": bool(cp[1] and cp[-1]),
            "cp_positive_states": len(cp[1]), "cp_negative_states": len(cp[-1]),
            "positive_witness": pos_witness, "negative_witness": neg_witness,
        })
    reversals.sort(key=lambda r: (-r["empirical_quadruples"], r["eta_uid_1"], r["eta_uid_2"]))
    summary = {
        "scenario": scene,
        "controller_uid": next(iter(ctl)),
        "independent_test_states": len(by_state),
        "independent_source_groups": len(set(state_family.values())),
        "selected_eta_probes": len(eta_keep),
        "eta_probes_present_on_test": len({r["eta_uid"] for r in data}),
        "full_seed_exact_pairs": len(data),
        "all_within_state_eta_comparisons": total_state_eta_comparisons,
        "informative_eta_comparisons_gap_ge_0_25": informative,
        "eta_pairs_with_any_informative_comparison": len(pair_signs),
        "eta_pairs_with_opposite_order_across_states": len(reversals),
        "eta_pairs_with_cp95_certified_opposite_order": sum(r["cp_certified_opposite_sides"] for r in reversals),
        "independent_states_involved_in_at_least_one_reversal": len(involved_states),
        "empirical_reversal_quadruples": reversal_quadruples,
        "best_fixed_eta_pairwise_ordering_accuracy_on_test_exploratory": eta_only_correct / informative if informative else None,
        "train_mean_eta_only_pairwise_ordering_accuracy_on_test": train_eta_order_correct / train_eta_order_evaluable if train_eta_order_evaluable else None,
        "train_mean_eta_only_evaluable_comparisons": train_eta_order_evaluable,
        "qualification": "Panel chosen without TEST outcomes; CP95 is per-comparison descriptive and not multiplicity corrected.",
    }
    return summary, reversals


def main() -> None:
    snapshot = pq.read_table(SNAP / "canonical_pair_snapshot.parquet").to_pylist()
    summaries, witnesses = [], []
    for scene in ("Toy", "DB"):
        summary, rows = audit_scene(scene, snapshot)
        summaries.append(summary)
        witnesses.extend(rows)
    (ROOT / "heldout_reversal_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (ROOT / "heldout_reversal_witnesses.csv").open("w", newline="") as f:
        fieldnames = ["scenario", "eta_uid_1", "eta_uid_2", "positive_states", "negative_states", "empirical_quadruples", "cp_certified_opposite_sides", "cp_positive_states", "cp_negative_states", "positive_witness", "negative_witness"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in witnesses:
            w.writerow(row)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
