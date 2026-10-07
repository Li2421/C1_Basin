"""Aggregate completed adaptive arms and identify any still-unlabelled states."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
def main() -> None:
    states = [json.loads(line) for line in (HERE / "state_manifest.jsonl").read_text().splitlines()]
    by_state: dict[str, list[dict]] = defaultdict(list)
    stage_counts = {}
    paths = sorted((HERE / "raw").glob("baseline_shard*/manifest.json"))
    paths += sorted((HERE / "raw").glob("candidate_shard*/manifest.json"))
    paths += sorted((HERE / "raw").glob("local_search_shard*/manifest.json"))
    if len(list((HERE / "raw").glob("baseline_shard*/manifest.json"))) != 2:
        raise RuntimeError("both baseline shards must complete before assessment")
    for path in paths:
        stage = path.parent.name
        manifest = json.loads(path.read_text())
        stage_counts[stage] = {
            "arms": len(manifest["arm_results"]),
            "new_rollouts": manifest["new_rollouts"],
            "reused_rollouts": manifest["reused_rollouts"],
        }
        for row in manifest["arm_results"]:
            by_state[row["state_id"]].append({"stage": stage, **row})

    assessment = []
    for state in states:
        arms = by_state[state["state_id"]]
        feasible = [row for row in arms if row["B_63_member"]]
        assessment.append(
            {
                "state_id": state["state_id"],
                "category": state["category"],
                "eta_zero_feasible": any(
                    row["B_63_member"] and row["eta"] == [0.0, 0.0, 0.0]
                    for row in arms
                ),
                "tested_arms": len(arms),
                "feasible_arm_count": len(feasible),
                "feasible_etas": [row["eta"] for row in feasible],
                "best_observed_success_count": max((row["success"] for row in arms), default=0),
                "arms": arms,
            }
        )
    missing = [row for row in assessment if not row["feasible_etas"]]
    summary = {
        "state_count": len(states),
        "covered_state_count": len(assessment) - len(missing),
        "B63_empty_state_count": len(missing),
        "B63_empty_state_ids": [row["state_id"] for row in missing],
        "eta_zero_feasible_count": sum(row["eta_zero_feasible"] for row in assessment),
        "nonzero_oracle_required_count": sum(
            bool(row["feasible_etas"]) and not row["eta_zero_feasible"]
            for row in assessment
        ),
        "category_coverage": {
            category: {
                "states": sum(row["category"] == category for row in assessment),
                "covered": sum(row["category"] == category and bool(row["feasible_etas"]) for row in assessment),
                "empty": sum(row["category"] == category and not row["feasible_etas"] for row in assessment),
            }
            for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")
        },
        "feasible_arm_count_distribution": dict(
            Counter(str(row["feasible_arm_count"]) for row in assessment)
        ),
        "stage_counts": stage_counts,
        "states": assessment,
    }
    (HERE / "oracle_coverage_assessment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "states"}, indent=2))


if __name__ == "__main__":
    main()
