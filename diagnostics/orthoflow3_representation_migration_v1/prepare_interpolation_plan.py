"""Freeze the bounded endpoint-interpolation topology screen."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1")
ALPHAS = (0.25, 0.5, 0.75)
INITIAL_STREAMS = 8
MAX_PAIRS = 16


def rows(pattern: str) -> list[dict]:
    output = []
    for path in sorted((HERE / "raw").glob(pattern)):
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def combined64(state_id: str, eta_index: int, promotion_stage: str) -> list[dict]:
    output = []
    for pattern in (
        "global256_exact/shard*.jsonl",
        "candidate_screen16/shard*.jsonl",
        f"{promotion_stage}/shard*.jsonl",
    ):
        output.extend(
            row
            for row in rows(pattern)
            if row["state_id"] == state_id and int(row["eta_index"]) == eta_index
        )
    return output


def main() -> None:
    output = HERE / "interpolation8_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite interpolation plan")
    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    primary = json.loads((HERE / "promotion1_plan.json").read_text())
    secondary_path = HERE / "promotion2_plan.json"
    secondary = json.loads(secondary_path.read_text()) if secondary_path.exists() else {"arms": []}
    primary_by_state = {arm["state_id"]: arm for arm in primary["arms"]}
    secondary_by_state = {arm["state_id"]: arm for arm in secondary["arms"]}

    pairs = []
    for state in sorted(subset["selected_states"], key=lambda row: row["selection_rank"]):
        active_a = primary_by_state.get(state["state_id"])
        if active_a is None:
            continue
        evidence_a = combined64(state["state_id"], int(active_a["eta_index"]), "promotion1")
        if len(evidence_a) != 64 or sum(row["success"] for row in evidence_a) < 63:
            continue
        if bool(state["zero_eta"]):
            endpoint_a = [0.0, 0.0, 0.0]
            endpoint_a_id = "ZERO_REUSED_B63"
            endpoint_b = active_a["eta"]
            endpoint_b_id = f"A{int(active_a['eta_index']):03d}"
            pair_kind = "zero_to_active"
        else:
            active_b = secondary_by_state.get(state["state_id"])
            if active_b is None:
                continue
            evidence_b = combined64(state["state_id"], int(active_b["eta_index"]), "promotion2")
            if len(evidence_b) != 64 or sum(row["success"] for row in evidence_b) < 63:
                continue
            endpoint_a = active_a["eta"]
            endpoint_a_id = f"A{int(active_a['eta_index']):03d}"
            endpoint_b = active_b["eta"]
            endpoint_b_id = f"A{int(active_b['eta_index']):03d}"
            pair_kind = "active_to_active"
        distance = float(np.linalg.norm(np.asarray(endpoint_b) - np.asarray(endpoint_a)))
        pairs.append(
            {
                "state": state,
                "endpoint_a": endpoint_a,
                "endpoint_a_id": endpoint_a_id,
                "endpoint_b": endpoint_b,
                "endpoint_b_id": endpoint_b_id,
                "pair_kind": pair_kind,
                "endpoint_distance": distance,
            }
        )

    # Preserve representation of active-required states first, then use the
    # frozen state order.  This ordering is fixed before interpolation outcomes.
    pairs.sort(
        key=lambda row: (
            row["pair_kind"] != "active_to_active",
            row["state"]["selection_rank"],
        )
    )
    pairs = pairs[:MAX_PAIRS]
    arms = []
    for pair_index, pair in enumerate(pairs):
        state = pair["state"]
        a = np.asarray(pair["endpoint_a"], dtype=np.float64)
        b = np.asarray(pair["endpoint_b"], dtype=np.float64)
        for alpha in ALPHAS:
            eta = ((1.0 - alpha) * a + alpha * b).tolist()
            arms.append(
                {
                    "arm_id": f"I8__{pair_index:02d}__{alpha:.2f}",
                    "basis_family": "orthoflow3",
                    "state_id": state["state_id"],
                    "selection_rank": state["selection_rank"],
                    "state_file": state["state_file"],
                    "state_sha256": state["state_sha256"],
                    "absolute_step": state["absolute_step"],
                    "rng_namespace": state["rng_namespace"],
                    "eta_index": None,
                    "parameter_id": f"INTERP_{pair_index:02d}_{alpha:.2f}",
                    "eta": eta,
                    "alpha": alpha,
                    "pair_kind": pair["pair_kind"],
                    "endpoint_a": pair["endpoint_a"],
                    "endpoint_a_id": pair["endpoint_a_id"],
                    "endpoint_b": pair["endpoint_b"],
                    "endpoint_b_id": pair["endpoint_b_id"],
                    "endpoint_distance": pair["endpoint_distance"],
                    "seeds": state["matched_flow_seeds"][:INITIAL_STREAMS],
                }
            )
    plan = {
        "schema": "orthoflow3_migration_arm_plan_v1",
        "stage": "interpolation8",
        "basis_family": "orthoflow3",
        "selection_rule": (
            "confirmed active-to-active B63 pairs first, then confirmed zero-to-active B63 "
            "pairs, each in frozen subset state order; cap 16; alphas 0.25/0.5/0.75"
        ),
        "endpoint_requirement": ">=63/64 for both endpoints",
        "interpolation_interpretation": (
            "8-stream staged screen; two observed failures disprove B63 for that interpolant; "
            "8/8 alone is not a confirmed B63 claim"
        ),
        "maximum_pairs": MAX_PAIRS,
        "initial_matched_streams": INITIAL_STREAMS,
        "arms": arms,
        "maximum_new_continuations": INITIAL_STREAMS * len(arms),
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
                "pairs": len(pairs),
                "active_to_active": sum(row["pair_kind"] == "active_to_active" for row in pairs),
                "zero_to_active": sum(row["pair_kind"] == "zero_to_active" for row in pairs),
                "new_rollouts": INITIAL_STREAMS * len(arms),
                "content_sha256": plan["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
