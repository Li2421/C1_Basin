"""Frozen source validation summary, not held-controller or LOSO evidence."""
import csv
import numpy as np
from .data import OUT, read, write, CONTROLLERS
from .train import KINDS, SEEDS


def main():
    d = dict(np.load(OUT / "dataset.npz"))
    results = {(k, seed): read(OUT / "models" / k / f"seed{seed}/summary.json")
               for k in KINDS for seed in SEEDS}
    rows = []
    for (kind, seed), r in results.items():
        for variant in ("correct", "wrong_controller", "state_shuffle", "context_shuffle", "state_response_shuffle", "eta_shuffle"):
            m = r[variant]
            rows.append({"kind": kind, "seed": seed, "variant": variant, "best_step": r["best_step"],
                         "NLL": m["NLL"], "MAE": m["MAE"], "eligible": m["eligible"],
                         "selected_B15": m["selected_B15"], "selected_unknown": m["selected_unknown"],
                         "controller_reversal_accuracy": m["controller_reversals"]["accuracy"],
                         "state_reversal_accuracy": m["state_reversals"]["accuracy"],
                         "severe_false_positive": m["severe_false_positive"]})
    with (OUT / "source_validation_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for seed in SEEDS:
        reference = results["physical_context", seed]["correct"]
        for name, other in ((k, results[k, seed]["correct"]) for k in ("eta_only", "additive")):
            a = {(r["controller"], r["state_uid"]): r for r in reference["picks"]}
            b = {(r["controller"], r["state_uid"]): r for r in other["picks"]}
            assert a.keys() == b.keys()
            rescue = sum(a[k]["B15"] and not b[k]["B15"] for k in a)
            broken = sum(b[k]["B15"] and not a[k]["B15"] for k in a)
            families = sorted({k[1] for k in a})
            net = np.array([sum(int(a[k]["B15"])-int(b[k]["B15"]) for k in a if k[1] == s) for s in families])
            denominators = np.array([sum(k[1] == s for k in a) for s in families])
            rng = np.random.default_rng(20261003)
            if len(families):
                index = rng.integers(0, len(families), (10000, len(families)))
                draws = net[index].sum(1) / denominators[index].sum(1)
                interval = np.quantile(draws, [.025, .975]).tolist()
            else:
                interval = None
            comparisons.append({"seed": seed, "full_vs": name, "rescue": int(rescue), "break": int(broken),
                                "net": int(rescue-broken), "eligible_state_controller_count": len(a),
                                "independent_source_families": len(families),
                                "family_cluster_bootstrap_95pct_rate_difference": interval})
    uncertainty = {"scope": "source-family validation only; no held-controller TEST outcome used",
                   "seed_results": rows, "paired_comparisons": comparisons,
                   "gate": read(OUT / "source_gate.json"),
                   "data_evidence": read(OUT / "source_evidence_audit.json")["validation"],
                   "source_family_count": 6,
                   "warning": "Repeated eta/controller reversals are dependent. Gate is a predeclared readiness check, not a proof of transfer.",
                   "new_source_rollouts": read(OUT / "merge_audit.json")["total_new_records_in_db"],
                   "quarantined_cost_not_training_data": 4825,
                   "held_controller_confirmation": "NOT_STARTED"}
    write(OUT / "source_validation_summary.json", uncertainty)
    print({"gate": uncertainty["gate"]["verdict"], "paired": comparisons})


if __name__ == "__main__":
    main()
