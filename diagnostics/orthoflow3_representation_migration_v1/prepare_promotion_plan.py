"""Freeze first B63-promotion wave from completed 16-stream evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1")


def rows(pattern: str) -> list[dict]:
    output = []
    for path in sorted((HERE / "raw").glob(pattern)):
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    output = HERE / "promotion1_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite promotion plan")
    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    state_map = {row["state_id"]: row for row in subset["selected_states"]}
    screen_plan = json.loads((HERE / "candidate_screen16_plan.json").read_text())
    global_rows = rows("global256_exact/shard*.jsonl")
    screen_rows = rows("candidate_screen16/shard*.jsonl")
    global_lookup = {(row["state_id"], int(row["eta_index"])): row for row in global_rows}
    grouped = defaultdict(list)
    for row in screen_rows:
        grouped[(row["state_id"], int(row["eta_index"]))].append(row)
    ranked = defaultdict(list)
    for arm in screen_plan["arms"]:
        key = (arm["state_id"], int(arm["eta_index"]))
        evidence = [global_lookup[key], *grouped[key]]
        if len(evidence) != 16:
            raise RuntimeError((key, len(evidence)))
        success = sum(row["success"] for row in evidence)
        successful_j = [float(row["J_def"]) for row in evidence if row["success"]]
        ranked[arm["state_id"]].append(
            {
                **arm, "screen16_success": success,
                "screen16_mean_J_def_success": sum(successful_j) / len(successful_j) if successful_j else math.inf,
            }
        )
    selected = []
    for state_id, candidates in ranked.items():
        feasible = [row for row in candidates if row["screen16_success"] >= 15]
        feasible.sort(key=lambda row: (-row["screen16_success"], row["screen16_mean_J_def_success"], row["eta_index"]))
        if feasible:
            selected.append(feasible[0])
    arms = []
    for candidate in selected:
        state = state_map[candidate["state_id"]]
        arms.append(
            {
                "arm_id": f"P64A__{state['selection_rank']:02d}__{candidate['eta_index']:03d}",
                "basis_family": "orthoflow3", "state_id": state["state_id"],
                "selection_rank": state["selection_rank"], "state_file": state["state_file"],
                "state_sha256": state["state_sha256"], "absolute_step": state["absolute_step"],
                "rng_namespace": state["rng_namespace"], "eta_index": candidate["eta_index"],
                "parameter_id": candidate["parameter_id"], "eta": candidate["eta"],
                "screen16_success": candidate["screen16_success"],
                "screen16_mean_J_def_success": candidate["screen16_mean_J_def_success"],
                "seeds": state["matched_flow_seeds"][16:64],
            }
        )
    plan = {
        "schema": "orthoflow3_migration_arm_plan_v1", "stage": "promotion1",
        "basis_family": "orthoflow3",
        "selection_rule": "best screen16 candidate per state; >=15/16 required",
        "prior_evidence_per_arm": 16, "arms": arms,
        "maximum_new_continuations": 48 * len(arms),
        "maximum_physical_steps": sum((850 - int(arm["absolute_step"])) * len(arm["seeds"]) for arm in arms),
    }
    plan["content_sha256"] = canonical_hash(plan)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"states_with_promotable_candidate": len(arms), "new_rollouts": 48 * len(arms), "content_sha256": plan["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
