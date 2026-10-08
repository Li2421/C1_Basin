"""Audit exact frozen Gap1 state-eta rollout labels and nested K coverage.

The report distinguishes the exact early B15 decision from a fully observed
Q16 count. Missing future seeds are never treated as failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


DEFAULT_DESIGN = Path("datasets/gap1_eta_scaling_v2")
K_VALUES = (1, 2, 4, 8, 16)


def audit(design_dir: Path) -> dict:
    design = json.loads((design_dir / "design_manifest.json").read_text())
    pool = json.loads((design_dir / "eta_pool.json").read_text())
    eta_uids = pool["eta_uid"]
    result = {
        "schema": "gap1_eta_scaling_audit_v1",
        "design": str(design_dir),
        "full_Q16_rollout_budget": design["rollout_budget_full_Q16"],
        "K": list(K_VALUES),
        "per_N": {},
    }
    all_complete = True
    for n in (2, 10, 50):
        states = [s for s in design["states"] if s["N"] == n]
        spec = design["scenario_info"][str(n)]
        split_counts = Counter(s["split"] for s in states)
        q_distribution = Counter()
        robust_by_state: dict[str, set[int]] = {s["state_uid"]: set() for s in states}
        observed_valid = numerical = collisions = timeouts = successes = 0
        logical_pairs = full_pairs = pair_files = 0
        errors = []
        for state_index, state in enumerate(states):
            for eta_index, eta_uid in enumerate(eta_uids):
                pair_index = state_index * len(eta_uids) + eta_index
                path = design_dir / f"rollouts/n{n}/pair_{pair_index:04d}/result.json"
                if not path.exists():
                    continue
                pair_files += 1
                pair = json.loads(path.read_text())
                if (pair["state_uid"] != state["state_uid"] or
                        pair["eta_uid"] != eta_uid or
                        pair["flow_checkpoint_sha256"] !=
                        spec["controller"]["flow_checkpoint_sha256"]):
                    errors.append(f"identity mismatch: {path}")
                    continue
                rows = pair["seeds"]
                if any(str(i) not in map(str, range(16)) for i in rows):
                    errors.append(f"invalid future index: {path}")
                    continue
                observed_valid += sum(not row["numerical_failure"] for row in rows.values())
                numerical += sum(row["numerical_failure"] for row in rows.values())
                collisions += sum(row["collision"] for row in rows.values())
                timeouts += sum(row["timeout"] for row in rows.values())
                successes += sum(row["success"] for row in rows.values())
                if pair.get("logical_robust_decision") is not None:
                    logical_pairs += 1
                if len(rows) != 16 or any(row["numerical_failure"] for row in rows.values()):
                    continue
                full_pairs += 1
                q16 = sum(bool(row["success"]) for row in rows.values())
                if pair.get("Q16_success_count") != q16:
                    errors.append(f"Q16 count mismatch: {path}")
                q_distribution[str(q16)] += 1
                if q16 >= 15:
                    robust_by_state[state["state_uid"]].add(eta_index)
        expected_pairs = len(states) * len(eta_uids)
        complete = full_pairs == expected_pairs and not errors
        all_complete &= complete
        # Eligibility and K curves are exact only when every candidate for
        # every state has a complete, numerically valid Q16 label.
        per_n = {
            "N": n,
            "physical_states": len(states),
            "split_counts": dict(split_counts),
            "eta_per_state": len(eta_uids),
            "state_eta_pairs_expected": expected_pairs,
            "pair_files": pair_files,
            "logical_robust_pairs": logical_pairs,
            "full_Q16_pairs": full_pairs,
            "full_Q16_rollouts_expected": expected_pairs * 16,
            "observed_valid_rollouts": observed_valid,
            "numerical_failures": numerical,
            "collisions": collisions,
            "timeouts_observed": timeouts,
            "successes_observed": successes,
            "Q16_distribution_full_pairs_only": dict(sorted(q_distribution.items(), key=lambda x: int(x[0]))),
            "complete": complete,
            "errors": errors,
        }
        if complete:
            robust_counts = [len(robust_by_state[s["state_uid"]]) for s in states]
            eligible = sum(x > 0 for x in robust_counts)
            per_n.update({
                "robust_success_pairs": sum(robust_counts),
                "robust_success_prevalence": sum(robust_counts) / expected_pairs,
                "states_with_no_sampled_robust_eta": len(states) - eligible,
                "states_with_at_least_one_robust_eta": eligible,
                "oracle_eligible_fraction": eligible / len(states),
                "robust_eta_count_per_state": {
                    s["state_id"]: len(robust_by_state[s["state_uid"]]) for s in states
                },
                "oracle_K_coverage": {
                    str(k): {
                        "all_states": sum(any(j < k for j in robust_by_state[s["state_uid"]])
                                          for s in states) / len(states),
                        "eligible_states": (
                            sum(any(j < k for j in robust_by_state[s["state_uid"]])
                                for s in states) / eligible if eligible else None
                        ),
                    } for k in K_VALUES
                },
                "split_oracle_eligible": {
                    split: sum(bool(robust_by_state[s["state_uid"]]) for s in states
                               if s["split"] == split)
                    for split in split_counts
                },
            })
        result["per_N"][str(n)] = per_n
    result["complete"] = all_complete
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.design)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized)
    if args.require_complete and not result["complete"]:
        raise SystemExit("Gap1 eta dataset is incomplete")


if __name__ == "__main__":
    main()
