"""Freeze window-aligned targets and strict source-group LOGO folds.

Must run only after Stage 1 has frozen ../candidate_windows.json.  The target
is never derived from y_long or correction magnitude:

  y_H=1  iff the paired Q0-QH 95% CI is strictly above zero;
  y_H=0  iff paired outcomes are exactly unchanged or favor delay;
  otherwise H_AMBIGUOUS and excluded from ordinary BCE.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE1 = ROOT / "stage1_window"
DATA = Path("/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4")
HARD = Path("/home/zhihan/research/Basin_C1/diagnostics/hard_stable_boundary_crossval/difficult_stable_states.csv")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        # A header-only ambiguous table is still an explicit scientific result.
        path.write_text("H,state_id,target_status\n")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def choose_validation_groups(rows: list[dict], outer: str) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        if row["source_group"] != outer:
            grouped[row["source_group"]].append(row)
    groups = sorted(grouped)
    total = [sum(len(v) for v in grouped.values()),
             sum(int(r["y_H"]) == 0 for v in grouped.values() for r in v),
             sum(int(r["y_H"]) == 1 for v in grouped.values() for r in v)]
    target = [0.18 * x for x in total]

    def vector(selection):
        chosen = [r for g in selection for r in grouped[g]]
        return [len(chosen), sum(int(r["y_H"]) == 0 for r in chosen), sum(int(r["y_H"]) == 1 for r in chosen)]

    def valid(selection):
        val = vector(selection)
        train = [total[i] - val[i] for i in range(3)]
        return val[1] > 0 and val[2] > 0 and train[1] > 0 and train[2] > 0

    def score(selection):
        value = vector(selection)
        return sum(((value[i] - target[i]) / max(target[i], 1.0)) ** 2 for i in range(3))

    # Deterministically seed with the best valid one- or two-group set, then
    # greedily approach 18% without sacrificing both classes in train/val.
    seeds = []
    for size in (1, 2):
        for combo in itertools.combinations(groups, size):
            if valid(combo):
                seeds.append(combo)
        if seeds:
            break
    if not seeds:
        raise RuntimeError(f"cannot form class-valid validation for outer group {outer}")
    selected = list(min(seeds, key=lambda x: (score(x), hashlib.sha256((outer + ':' + '|'.join(x)).encode()).hexdigest())))
    remaining = set(groups) - set(selected)
    while remaining:
        candidates = [selected + [group] for group in remaining if valid(selected + [group])]
        if not candidates:
            break
        best = min(candidates, key=lambda x: (score(x), hashlib.sha256((outer + ':' + '|'.join(sorted(x))).encode()).hexdigest()))
        if score(best) >= score(selected):
            break
        added = (set(best) - set(selected)).pop()
        selected.append(added)
        remaining.remove(added)
    return sorted(selected)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    candidates_path = STAGE1 / "candidate_windows.json"
    candidates = json.loads(candidates_path.read_text())
    if not candidates.get("frozen_before_any_stage2_training"):
        raise AssertionError("candidate horizons were not preregistered")
    if candidates["audit_plan_sha256"] != sha(STAGE1 / "audit_plan.json"):
        raise AssertionError("candidate/Stage1 plan hash mismatch")
    if candidates["paired_success_statistics_sha256"] != sha(STAGE1 / "paired_success_differences.csv"):
        raise AssertionError("candidate/paired-statistics hash mismatch")

    meta = {row["state_id"]: row for row in read_csv(STAGE1 / "frozen_state_list.csv")}
    paired = read_csv(STAGE1 / "paired_success_differences.csv")
    hard_ids = {row["state_id"] for row in read_csv(HARD)}
    target_rows, ambiguous_rows = [], []
    for H in candidates["candidate_horizons"]:
        rows = [row for row in paired if int(row["delay_steps"]) == int(H)]
        if len(rows) != len(meta):
            raise AssertionError(f"candidate H={H} is not evaluated on the full frozen state pool")
        for row in rows:
            status = row["q_status"]
            if status == "SUPPORTED_RECOVERABILITY_LOSS":
                label, target_status = 1, "H_RESOLVED_POSITIVE"
            elif status in ("RESOLVED_NO_LOSS_EXACT", "SUPPORTED_RECOVERABILITY_IMPROVEMENT"):
                label, target_status = 0, "H_RESOLVED_NEGATIVE"
            else:
                label, target_status = "", "H_AMBIGUOUS"
            item = {
                "H": int(H), "state_id": row["state_id"], "source_group": row["source_group"],
                "category": row["category"], "is_hard13_anchor": row["state_id"] in hard_ids,
                "y_H": label, "target_status": target_status, "q_status": status,
                "n_matched": int(row["n_matched"]), "Q0": float(row["Q0"]), "QH": float(row["QH"]),
                "Delta_Q0_minus_QH": float(row["Delta_Q0_minus_QH"]),
                "paired_ci_lower": float(row["paired_bootstrap95_lower"]),
                "paired_ci_upper": float(row["paired_bootstrap95_upper"]),
                "y_long_diagnostic_only": int(meta[row["state_id"]]["y_long"]),
            }
            (ambiguous_rows if target_status == "H_AMBIGUOUS" else target_rows).append(item)

    write_csv(HERE / "window_targets.csv", target_rows)
    write_csv(HERE / "ambiguous_targets.csv", ambiguous_rows)
    for H in candidates["candidate_horizons"]:
        write_csv(HERE / f"target_H{H}.csv", [row for row in target_rows if int(row["H"]) == int(H)])
        write_csv(HERE / f"ambiguity_H{H}.csv", [row for row in ambiguous_rows if int(row["H"]) == int(H)])

    folds = []
    inventory = []
    for H in candidates["candidate_horizons"]:
        resolved = [row for row in target_rows if int(row["H"]) == int(H)]
        groups = sorted({row["source_group"] for row in resolved})
        counts = Counter(int(row["y_H"]) for row in resolved)
        by_class_groups = {
            str(y): len({row["source_group"] for row in resolved if int(row["y_H"]) == y}) for y in (0, 1)
        }
        eventual = [row for row in resolved if int(row["y_long_diagnostic_only"]) == 1]
        inventory.append({"H": int(H), "resolved_zero": counts[0], "resolved_one": counts[1], "ambiguous": sum(int(row["H"]) == int(H) for row in ambiguous_rows), "resolved_source_groups": len(groups), "zero_source_groups": by_class_groups["0"], "one_source_groups": by_class_groups["1"], "eventually_needed_resolved_zero": sum(int(row["y_H"]) == 0 for row in eventual), "eventually_needed_resolved_one": sum(int(row["y_H"]) == 1 for row in eventual), "eventually_needed_zero_source_groups": len({row["source_group"] for row in eventual if int(row["y_H"]) == 0}), "eventually_needed_one_source_groups": len({row["source_group"] for row in eventual if int(row["y_H"]) == 1})})
        if min(by_class_groups.values()) < 3:
            raise RuntimeError(f"H={H} does not have three source groups per resolved class")
        for outer in groups:
            validation_groups = choose_validation_groups(resolved, outer)
            test = sorted(row["state_id"] for row in resolved if row["source_group"] == outer)
            validation = sorted(row["state_id"] for row in resolved if row["source_group"] in validation_groups)
            train = sorted(row["state_id"] for row in resolved if row["source_group"] != outer and row["source_group"] not in validation_groups)
            if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
                raise AssertionError("state leakage")
            if set(train) | set(validation) | set(test) != {row["state_id"] for row in resolved}:
                raise AssertionError("fold coverage mismatch")
            folds.append({
                "H": int(H), "fold_id": f"H{H}_LOGO_{outer}", "outer_test_source_group": outer,
                "train_source_groups": sorted({row["source_group"] for row in resolved if row["state_id"] in train}),
                "validation_source_groups": validation_groups, "test_source_groups": [outer],
                "normalization_fit_state_ids": train, "train_state_ids": train,
                "validation_state_ids": validation, "test_state_ids": test,
                "train_counts": dict(Counter(str(row["y_H"]) for row in resolved if row["state_id"] in train)),
                "validation_counts": dict(Counter(str(row["y_H"]) for row in resolved if row["state_id"] in validation)),
                "test_counts": dict(Counter(str(row["y_H"]) for row in resolved if row["state_id"] in test)),
            })

    fold_manifest = {
        "candidate_windows_sha256": sha(candidates_path),
        "window_targets_sha256": sha(HERE / "window_targets.csv"),
        "ambiguous_targets_sha256": sha(HERE / "ambiguous_targets.csv"),
        "samples_npz_sha256": sha(DATA / "samples.npz"),
        "feature_schema_sha256": sha(DATA / "feature_schema.json"),
        "features": 214, "models": {"LINEAR": [214, 1], "MLP_64x64": [214, 64, 64, 1]},
        "loss": "ordinary unweighted BCE on H_RESOLVED states only",
        "training_seeds": [17],
        "seed_preregistration": "single seed 17 fixed uniformly for every H/model/fold before any Stage-2 performance was observed; no performance-triggered seed additions permitted",
        "threshold": "validation-only per fold/model/seed",
        "folds": folds, "horizon_inventory": inventory,
        "per_horizon_target_hashes": {
            str(H): {
                "target": sha(HERE / f"target_H{H}.csv"),
                "ambiguity": sha(HERE / f"ambiguity_H{H}.csv"),
            }
            for H in candidates["candidate_horizons"]
        },
    }
    write_json(HERE / "fold_manifest.json", fold_manifest)
    write_json(HERE / "seed_preregistration.json", {
        "training_seeds": [17],
        "frozen_before_any_training": True,
        "applies_uniformly_to_every_horizon_model_and_fold": True,
        "performance_triggered_seed_additions_allowed": False,
        "reason": "full 117-group strict LOGO across up to three H and two models is already up to 702 jobs with one seed",
    })
    ready = {
        "candidate_windows_sha256": sha(candidates_path),
        "fold_manifest_sha256": sha(HERE / "fold_manifest.json"),
        "window_targets_sha256": sha(HERE / "window_targets.csv"),
        "candidate_frozen_before_training": True, "strict_source_group_logo": True,
        "ambiguous_excluded_from_bce": True, "y_long_not_used_as_target": True,
        "correction_magnitude_not_used_as_target": True,
        "seed_preregistration_sha256": sha(HERE / "seed_preregistration.json"),
    }
    write_json(HERE / "TRAINING_READY.json", ready)
    print(json.dumps({"candidate_horizons": candidates["candidate_horizons"], "resolved_targets": len(target_rows), "ambiguous_targets": len(ambiguous_rows), "folds": len(folds), "training_ready_sha256": sha(HERE / "TRAINING_READY.json")}, indent=2))


if __name__ == "__main__":
    main()
