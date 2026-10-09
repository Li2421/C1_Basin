"""Audit the predeclared global-prefix TRAIN/VAL gate without opening TEST."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def audit(design_dir: Path, prefix: int) -> dict:
    design = json.loads((design_dir / "design_manifest.json").read_text())
    pool = json.loads((design_dir / "eta_pool.json").read_text())
    if prefix not in (32, 64, 128) or prefix > len(pool["eta_uid"]):
        raise ValueError("prefix must be a predeclared global stage")
    result = {"schema": "gap1_eta_v4_train_val_prefix_gate_v1",
              "design": str(design_dir), "prefix": prefix,
              "test_outcomes_read": False, "per_N": {}}
    for n in (10, 20):
        states = [s for s in design["states"] if s["N"] == n]
        expected_hash = design["scenario_info"][str(n)]["controller"]["flow_checkpoint_sha256"]
        counts = {"train": Counter(), "val": Counter()}
        robust_by_state = {s["state_uid"]: 0 for s in states if s["split"] != "test"}
        errors = []
        for state_index, state in enumerate(states):
            if state["split"] == "test":
                continue
            split = state["split"]
            for eta_index in range(prefix):
                pair_index = state_index * len(pool["eta_uid"]) + eta_index
                path = design_dir / f"rollouts/n{n}/pair_{pair_index:04d}/result.json"
                counts[split]["expected"] += 1
                if not path.exists():
                    continue
                pair = json.loads(path.read_text())
                if (pair["state_uid"] != state["state_uid"] or
                        pair["eta_uid"] != pool["eta_uid"][eta_index] or
                        pair["flow_checkpoint_sha256"] != expected_hash):
                    errors.append(str(path))
                    continue
                counts[split]["pair_files"] += 1
                counts[split]["valid_rollouts"] += sum(
                    not row["numerical_failure"] for row in pair["seeds"].values())
                counts[split]["numerical_rollouts"] += sum(
                    row["numerical_failure"] for row in pair["seeds"].values())
                counts[split]["collisions"] += sum(
                    row["collision"] for row in pair["seeds"].values())
                decision = pair.get("logical_robust_decision")
                if decision is None:
                    continue
                valid = [row for row in pair["seeds"].values()
                         if not row["numerical_failure"]]
                successes = sum(bool(row["success"]) for row in valid)
                failures = len(valid) - successes
                if ((decision["robust"] and successes < 15) or
                        (not decision["robust"] and failures < 2)):
                    errors.append(f"invalid B15 decision: {path}")
                    continue
                counts[split]["decided"] += 1
                if decision["robust"]:
                    counts[split]["robust_pairs"] += 1
                    robust_by_state[state["state_uid"]] += 1
        by_split = {}
        for split in ("train", "val"):
            split_states = [s for s in states if s["split"] == split]
            by_split[split] = {
                **counts[split],
                "physical_states": len(split_states),
                "states_with_robust_eta": sum(
                    robust_by_state[s["state_uid"]] > 0 for s in split_states),
            }
        complete = not errors and all(
            by_split[s].get("decided", 0) == by_split[s]["expected"]
            for s in ("train", "val"))
        coverage_pass = (complete and
                         by_split["train"]["states_with_robust_eta"] >= 8 and
                         by_split["val"]["states_with_robust_eta"] >= 2)
        result["per_N"][str(n)] = {
            "complete": complete, "coverage_pass": coverage_pass,
            "by_split": by_split, "identity_errors": errors,
        }
    result["both_N_coverage_pass"] = all(
        result["per_N"][str(n)]["coverage_pass"] for n in (10, 20))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path,
                        default=Path("datasets/gap1_eta_scaling_v4_global128"))
    parser.add_argument("--prefix", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.design, args.prefix)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
