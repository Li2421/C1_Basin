"""Freeze the 32 x common-256 exact-screen plan before outcomes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
DESIGN = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json"


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> None:
    output = HERE / "global256_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen global plan")
    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    points = json.loads(DESIGN.read_text())["points"]
    arms = []
    for state in subset["selected_states"]:
        for eta_index, point in enumerate(points):
            arms.append(
                {
                    "arm_id": f"G256__{state['selection_rank']:02d}__{eta_index:03d}",
                    "basis_family": "orthoflow3",
                    "state_id": state["state_id"],
                    "selection_rank": state["selection_rank"],
                    "state_file": state["state_file"],
                    "state_sha256": state["state_sha256"],
                    "absolute_step": state["absolute_step"],
                    "rng_namespace": state["rng_namespace"],
                    "eta_index": eta_index,
                    "parameter_id": point["parameter_id"],
                    "eta": point["theta"],
                    "seeds": [state["exact_screen_seed"]],
                }
            )
    plan = {
        "schema": "orthoflow3_migration_arm_plan_v1",
        "stage": "global256_exact",
        "basis_family": "orthoflow3",
        "selection_is_outcome_blind": True,
        "candidate_design_sha256": hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        "arms": arms,
        "maximum_new_continuations": len(arms),
        "maximum_physical_steps": sum(
            850 - int(arm["absolute_step"]) for arm in arms
        ),
    }
    plan["content_sha256"] = canonical_hash(plan)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"arms": len(arms), "maximum_steps": plan["maximum_physical_steps"], "content_sha256": plan["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
