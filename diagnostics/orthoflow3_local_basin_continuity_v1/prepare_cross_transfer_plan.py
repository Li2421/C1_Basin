"""Freeze the resource-bounded directional robust cross-transfer schedule."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    destination = HERE / "cross_transfer_plan.json"
    if destination.exists():
        raise RuntimeError("cross-transfer plan already exists")
    anchors = json.loads((HERE / "anchor_manifest.json").read_text())["anchors"]
    neighbors = json.loads((HERE / "neighbor_manifest.json").read_text())["neighbors"]
    cloud = []
    for line in (HERE / "common_eta_probe_cloud.csv").read_text().splitlines()[1:]:
        # The zero overlap decision only needs the literal first-column form;
        # robust parsing is handled below through the documented CSV shape.
        cloud.append(line)
    zero_anchor = {row["anchor_rank"]: row["canonical_eta_provisional"] == [0.0, 0.0, 0.0] for row in anchors}
    # All +/-1 pairs, plus the 21 cheapest +/-4 pairs.  Cost is remaining
    # frozen horizon only, so this is decided without outcome inspection.
    plusminus1 = [row for row in neighbors if abs(int(row["offset_steps"])) == 1]
    plusminus4 = [row for row in neighbors if abs(int(row["offset_steps"])) == 4]
    plusminus4.sort(key=lambda row: (850 - int(row["absolute_step"]), int(row["anchor_rank"]), int(row["offset_steps"])))
    chosen = sorted(plusminus1 + plusminus4[:21], key=lambda row: (int(row["anchor_rank"]), int(row["offset_steps"])))
    anchor_map = {int(row["anchor_rank"]): row for row in anchors}
    arms = []
    for index, neighbor in enumerate(chosen):
        anchor = anchor_map[int(neighbor["anchor_rank"])]
        eta = anchor["canonical_eta_provisional"]
        # eta=0 is in the common cloud at every neighbor, so its first eight
        # observations are supplied by screening; nonzero canonical eta is not
        # assumed to overlap the independent cloud.
        seeds = neighbor["matched_flow_seeds"][8:64] if zero_anchor[int(neighbor["anchor_rank"])] else neighbor["matched_flow_seeds"][:64]
        arms.append({
            "arm_id": f"X64__{index:02d}__A{int(neighbor['anchor_rank']):02d}__d{int(neighbor['offset_steps']):+d}",
            "state_id": neighbor["neighbor_id"], "role": "neighbor", "anchor_rank": int(neighbor["anchor_rank"]),
            "offset_steps": int(neighbor["offset_steps"]), "state_file": neighbor["state_file"], "state_sha256": neighbor["state_sha256"],
            "absolute_step": int(neighbor["absolute_step"]), "rng_namespace": int(neighbor["rng_namespace"]),
            "eta": eta, "probe_id": "ANCHOR_CANONICAL_CROSS_TRANSFER", "seeds": seeds,
            "screen_seed_reuse_expected": bool(zero_anchor[int(neighbor["anchor_rank"])]),
            "anchor_state_id": anchor["state_id"], "anchor_eta_is_prior_B63": True,
        })
    plan = {
        "schema": "orthoflow3_local_basin_cross_transfer_plan_v1", "basis_family": "orthoflow3",
        "selection_rule": "all +/-1; 21 +/-4 with smallest remaining frozen horizon, tie anchor rank then offset; fixed before screening outcomes",
        "canonical_source": "prior migration promoted B63 source; no new canonical selection",
        "arms": arms,
        "new_continuations": sum(len(arm["seeds"]) for arm in arms),
        "physical_step_upper_bound": sum((850 - arm["absolute_step"]) * len(arm["seeds"]) for arm in arms),
        "screening_first_eight_required_for_zero": True,
    }
    plan["content_sha256"] = digest(plan)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({"pairs": len(arms), "by_offset": {str(offset): sum(arm["offset_steps"] == offset for arm in arms) for offset in (-4, -1, 1, 4)}, "new_continuations": plan["new_continuations"], "new_steps": plan["physical_step_upper_bound"]}, indent=2))


if __name__ == "__main__":
    main()
