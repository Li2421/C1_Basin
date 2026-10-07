"""Freeze eta-zero first, then nonzero arms only for eta-zero failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from common import HERE, eta_key, load_effective_records, read_jsonl, write_json


def _result(rows: list[dict]) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    complete = len(rows) == 64 and not any(row.get("execution_error") for row in rows)
    return {
        "evaluated": len(rows),
        "success": counts["success"],
        "failure": len(rows) - counts["success"],
        "complete": complete,
        "B_63_member": complete and counts["success"] >= 63,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("zero", "candidates", "assessment"))
    args = parser.parse_args()
    integrity = json.loads((HERE / "startup_state_integrity_checks.json").read_text())
    if integrity["status"] != "PASS":
        raise RuntimeError("startup integrity gate did not pass")
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = read_jsonl(HERE / "startup_state_manifest.jsonl")
    seeds = list(map(int, protocol["oracle_seed_default"]))
    effective, duplicates = load_effective_records()

    if args.phase == "zero":
        arms = [{"arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"], "eta": [0.0, 0.0, 0.0], "seeds": seeds} for row in states]
        write_json(HERE / "eta_zero_arms.json", arms)
        print(json.dumps({"arms": len(arms), "maximum_rollouts": len(arms) * 64}, indent=2)); return

    grouped = defaultdict(list)
    for (state_id, eta, _), row in effective.items():
        grouped[(state_id, eta)].append(row)
    zero = eta_key((0.0, 0.0, 0.0))
    zero_results = {row["state_id"]: _result(grouped[(row["state_id"], zero)]) for row in states}
    if any(not value["complete"] and value["failure"] < 2 for value in zero_results.values()):
        missing = [key for key, value in zero_results.items() if not value["complete"] and value["failure"] < 2]
        raise RuntimeError(("eta-zero stage incomplete", missing[:10]))

    candidates = [list(eta_key(eta)) for eta in protocol["candidate_etas"]]
    needs_nonzero = [state_id for state_id, value in zero_results.items() if not value["B_63_member"]]
    arms = [{"arm_id": f"{state_id}__eta_{index}", "state_id": state_id, "eta": eta, "seeds": seeds, "candidate_priority": index} for state_id in needs_nonzero for index, eta in enumerate(candidates)]
    if args.phase == "candidates":
        write_json(HERE / "candidate_arms.json", arms)
        print(json.dumps({"eta_zero_B63": len(states) - len(needs_nonzero), "nonzero_required": len(needs_nonzero), "candidate_arms": len(arms), "maximum_rollouts": 64 * len(arms)}, indent=2)); return

    assessment = []
    for state in states:
        cells = []
        for eta in [zero] + [eta_key(value) for value in candidates]:
            rows = grouped[(state["state_id"], eta)]
            if rows:
                cells.append({"eta": list(eta), **_result(rows)})
        feasible = [cell for cell in cells if cell["B_63_member"]]
        assessment.append({"state_id": state["state_id"], "zero_B_63": zero_results[state["state_id"]]["B_63_member"], "B_63_empty": not feasible, "feasible_etas": [cell["eta"] for cell in feasible], "cells": cells})
    write_json(HERE / "oracle_coverage_assessment.json", {"states": len(states), "eta_zero_B63": sum(row["zero_B_63"] for row in assessment), "B_63_empty": sum(row["B_63_empty"] for row in assessment), "exact_cache_duplicates": len(duplicates), "state_results": assessment})
    print(json.dumps({"states": len(states), "eta_zero_B63": sum(row["zero_B_63"] for row in assessment), "B_63_empty": sum(row["B_63_empty"] for row in assessment)}, indent=2))


if __name__ == "__main__":
    main()
