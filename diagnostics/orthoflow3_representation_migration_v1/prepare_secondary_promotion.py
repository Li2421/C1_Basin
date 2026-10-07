"""Freeze a bounded second-endpoint B63 promotion plan.

The primary promotion establishes one active OrthoFlow3 endpoint per
state.  A second active endpoint is useful mainly on states where eta=0
is not already a robust endpoint.  Selection is deterministic and capped
before any second-endpoint 64-stream outcomes are observed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1")
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
WIDTH = HIGH - LOW
MAX_SECONDARY_STATES = 16


def rows(pattern: str) -> list[dict]:
    output = []
    for path in sorted((HERE / "raw").glob(pattern)):
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def evidence16(screen_plan: dict) -> dict[tuple[str, int], dict]:
    global_lookup = {
        (row["state_id"], int(row["eta_index"])): row
        for row in rows("global256_exact/shard*.jsonl")
    }
    screen_lookup = defaultdict(list)
    for row in rows("candidate_screen16/shard*.jsonl"):
        screen_lookup[(row["state_id"], int(row["eta_index"]))].append(row)
    output = {}
    for arm in screen_plan["arms"]:
        key = (arm["state_id"], int(arm["eta_index"]))
        evidence = [global_lookup[key], *screen_lookup[key]]
        if len(evidence) != 16:
            raise RuntimeError((key, "expected 16 screen observations", len(evidence)))
        successful_j = [float(row["J_def"]) for row in evidence if row["success"]]
        output[key] = {
            **arm,
            "screen16_success": sum(bool(row["success"]) for row in evidence),
            "screen16_mean_J_def_success": (
                sum(successful_j) / len(successful_j) if successful_j else math.inf
            ),
        }
    return output


def promotion64(plan: dict, stage: str) -> dict[tuple[str, int], dict]:
    first = {
        (row["state_id"], int(row["eta_index"])): row
        for row in rows("global256_exact/shard*.jsonl")
    }
    screen = defaultdict(list)
    for row in rows("candidate_screen16/shard*.jsonl"):
        screen[(row["state_id"], int(row["eta_index"]))].append(row)
    promoted = defaultdict(list)
    for row in rows(f"{stage}/shard*.jsonl"):
        promoted[(row["state_id"], int(row["eta_index"]))].append(row)
    output = {}
    for arm in plan["arms"]:
        key = (arm["state_id"], int(arm["eta_index"]))
        evidence = [first[key], *screen[key], *promoted[key]]
        if len(evidence) != 64:
            raise RuntimeError((key, "expected 64 primary observations", len(evidence)))
        output[key] = {
            "success": sum(bool(row["success"]) for row in evidence),
            "B63": sum(bool(row["success"]) for row in evidence) >= 63,
        }
    return output


def main() -> None:
    output = HERE / "promotion2_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite secondary promotion plan")

    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    state_map = {row["state_id"]: row for row in subset["selected_states"]}
    screen_plan = json.loads((HERE / "candidate_screen16_plan.json").read_text())
    primary_plan = json.loads((HERE / "promotion1_plan.json").read_text())
    screen = evidence16(screen_plan)
    primary64 = promotion64(primary_plan, "promotion1")
    primary_by_state = {arm["state_id"]: arm for arm in primary_plan["arms"]}

    eligible = []
    for state in sorted(subset["selected_states"], key=lambda row: row["selection_rank"]):
        # eta=0 is representation invariant.  A historical zero target means
        # zero already supplies a robust second endpoint, so spend this bounded
        # promotion on the active-required states instead.
        if bool(state["zero_eta"]):
            continue
        primary = primary_by_state.get(state["state_id"])
        if primary is None:
            continue
        primary_key = (state["state_id"], int(primary["eta_index"]))
        if not primary64[primary_key]["B63"]:
            continue
        primary_eta = np.asarray(primary["eta"], dtype=np.float64)
        alternatives = []
        for (state_id, eta_index), candidate in screen.items():
            if state_id != state["state_id"] or eta_index == int(primary["eta_index"]):
                continue
            if candidate["screen16_success"] < 15:
                continue
            eta = np.asarray(candidate["eta"], dtype=np.float64)
            candidate = dict(candidate)
            candidate["normalized_distance_from_primary"] = float(
                np.linalg.norm((eta - primary_eta) / WIDTH)
            )
            alternatives.append(candidate)
        alternatives.sort(
            key=lambda row: (
                -row["normalized_distance_from_primary"],
                -row["screen16_success"],
                row["screen16_mean_J_def_success"],
                row["eta_index"],
            )
        )
        if alternatives:
            eligible.append((state, primary, alternatives[0]))

    eligible = eligible[:MAX_SECONDARY_STATES]
    arms = []
    for state, primary, candidate in eligible:
        arms.append(
            {
                "arm_id": f"P64B__{state['selection_rank']:02d}__{candidate['eta_index']:03d}",
                "basis_family": "orthoflow3",
                "state_id": state["state_id"],
                "selection_rank": state["selection_rank"],
                "state_file": state["state_file"],
                "state_sha256": state["state_sha256"],
                "absolute_step": state["absolute_step"],
                "rng_namespace": state["rng_namespace"],
                "eta_index": candidate["eta_index"],
                "parameter_id": candidate["parameter_id"],
                "eta": candidate["eta"],
                "primary_eta_index": primary["eta_index"],
                "screen16_success": candidate["screen16_success"],
                "normalized_distance_from_primary": candidate["normalized_distance_from_primary"],
                "seeds": state["matched_flow_seeds"][16:64],
            }
        )

    plan = {
        "schema": "orthoflow3_migration_arm_plan_v1",
        "stage": "promotion2",
        "basis_family": "orthoflow3",
        "selection_rule": (
            "among historical active-required states with a confirmed primary B63 endpoint, "
            "choose the farthest normalized-domain candidate having >=15/16 screen success; "
            "tie by screen success, successful J_def, eta index; state-rank cap 16"
        ),
        "maximum_secondary_states": MAX_SECONDARY_STATES,
        "prior_evidence_per_arm": 16,
        "arms": arms,
        "maximum_new_continuations": 48 * len(arms),
        "maximum_physical_steps": sum(
            (850 - int(arm["absolute_step"])) * len(arm["seeds"]) for arm in arms
        ),
    }
    plan["content_sha256"] = canonical_hash(plan)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "eligible_secondary_states": len(eligible),
                "new_rollouts": 48 * len(arms),
                "content_sha256": plan["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
