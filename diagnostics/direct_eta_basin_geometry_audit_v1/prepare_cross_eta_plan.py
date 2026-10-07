"""Freeze Stage-D cross-eta and midpoint screening arms for 20 preselected pairs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    destination = HERE / "cross_eta_screen_plan.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen plan")

    dataset = json.loads((HERE / "dataset_424_manifest.json").read_text())
    states = {row["state_id"]: row for row in dataset["states"]}
    pair_path = HERE / "stage_bc_work/cross_eta_candidate_pairs.csv"
    pairs = list(csv.DictReader(pair_path.open()))
    if len(pairs) != 20:
        raise RuntimeError(f"expected 20 frozen candidate pairs, found {len(pairs)}")

    strict_source = json.loads(
        (ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/source_manifest.json").read_text()
    )
    strict_rng = {row["state_id"]: int(row["rng_namespace"]) for row in strict_source["states"]}

    def seeds_for(state: dict) -> list[int]:
        if state["category"] == "STARTUP":
            return list(range(95710001, 95710017))
        if state["category"] == "STRICT_DEADLOCK":
            return list(range(95310001, 95310017))
        return list(range(95210001, 95210017))

    def namespace_for(state: dict) -> int:
        if state["category"] == "STRICT_DEADLOCK":
            return strict_rng[state["state_id"]]
        return int(state["rng_namespace"])

    arms = []
    for pair in pairs:
        rank = int(pair["priority_rank"])
        eta_i = json.loads(pair["eta_i"])
        eta_j = json.loads(pair["eta_j"])
        midpoint = [(a + b) / 2.0 for a, b in zip(eta_i, eta_j)]
        for side, own_eta, other_eta in (("i", eta_i, eta_j), ("j", eta_j, eta_i)):
            state = states[pair[f"state_{side}"]]
            for assignment, eta in (("cross", other_eta), ("midpoint", midpoint)):
                arms.append(
                    {
                        "arm_id": f"PAIR{rank:02d}__{side.upper()}__{assignment.upper()}",
                        "pair_rank": rank,
                        "pair_state_side": side,
                        "assignment": assignment,
                        "state_id": state["state_id"],
                        "state_file": state["state_file"],
                        "state_sha256": state["state_sha256"],
                        "absolute_step": int(state["absolute_step"]),
                        "rng_namespace": namespace_for(state),
                        "eta": eta,
                        "own_canonical_eta": own_eta,
                        "other_canonical_eta": other_eta,
                        "seeds": seeds_for(state),
                    }
                )

    plan = {
        "schema": "direct_eta_cross_eta_screen_plan_v1",
        "status": "FROZEN_BEFORE_ROLLOUT",
        "selection": "all 20 outcome-blind close-h/large-canonical-jump pairs frozen by Stage B/C",
        "screening_only": True,
        "screening_futures_per_arm": 16,
        "promotion_rule": "no automatic promotion; confirmation block only if budget remains and screen is decision-critical",
        "self_canonical_note": "own canonical B63 outcomes are reused from exact frozen source oracle caches",
        "pair_manifest_path": str(pair_path),
        "pair_manifest_sha256": sha(pair_path),
        "dataset_manifest_sha256": sha(HERE / "dataset_424_manifest.json"),
        "arms": arms,
        "new_continuations": sum(len(arm["seeds"]) for arm in arms),
        "maximum_physical_steps": sum(
            (850 - arm["absolute_step"]) * len(arm["seeds"]) for arm in arms
        ),
    }
    plan["content_sha256"] = semantic_sha(plan)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "arms": len(arms),
                "continuations": plan["new_continuations"],
                "maximum_physical_steps": plan["maximum_physical_steps"],
                "plan_sha256": sha(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
