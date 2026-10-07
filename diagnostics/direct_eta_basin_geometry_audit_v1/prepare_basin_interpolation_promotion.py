"""Freeze budget-safe 48-seed confirmation blocks for perfect interpolation screens."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> None:
    destination = HERE / "basin_interpolation_promotion_plan.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen plan")
    screen_plan = json.loads((HERE / "basin_interpolation_screen_plan.json").read_text())
    screen_rows = [json.loads(line) for line in (HERE / "raw/basin_interp_screen/records.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in screen_rows:
        grouped[row["arm_id"]].append(row)
    if len(screen_rows) != 640:
        raise RuntimeError(f"screen incomplete: {len(screen_rows)}/640")

    oracle_plan = json.loads((HERE / "fresh32_oracle_plan.json").read_text())
    oracle_rows = [json.loads(line) for line in (HERE / "raw/fresh32_oracle/records.jsonl").read_text().splitlines()]
    oracle_grouped = defaultdict(list)
    for row in oracle_rows:
        oracle_grouped[row["arm_id"]].append(row)
    oracle_upper_bound = 0
    for arm in oracle_plan["arms"]:
        rows = oracle_grouped[arm["arm_id"]]
        failures = sum(row["outcome"] != "success" for row in rows)
        oracle_upper_bound += len(rows) if failures >= 2 else 64

    other_committed = 4096 + 688 + 2048 + 640
    remaining = 15000 - other_committed - oracle_upper_bound
    maximum_promotions = max(0, remaining // 48)
    subset = json.loads((HERE / "basin_subset_manifest.json").read_text())
    state_order = {row["state_id"]: index for index, row in enumerate(subset["selected_states"])}
    screen_arm = {arm["arm_id"]: arm for arm in screen_plan["arms"]}
    eligible = []
    for request in screen_plan["requested_new_assignments"]:
        if request["kind"] != "interpolation":
            continue
        rows = grouped[request["simulation_arm_id"]]
        if len(rows) == 16 and all(row["outcome"] == "success" for row in rows):
            eligible.append(request)
    alpha_priority = {0.5: 0, 0.25: 1, 0.75: 2}
    eligible.sort(key=lambda row: (alpha_priority[float(row["alpha"])], state_order[row["state_id"]]))
    chosen = eligible[:maximum_promotions]
    arms = []
    assignments = []
    for request in chosen:
        base = screen_arm[request["simulation_arm_id"]]
        arm_id = f"PROMOTE{len(arms):02d}__{base['state_id']}"
        arms.append({**base, "arm_id": arm_id, "seeds": base["seeds"][:0] + list(range(base["seeds"][0] + 16, base["seeds"][0] + 64))})
        assignments.append({**request, "promotion_arm_id": arm_id})
    plan = {
        "schema": "direct_eta_basin_interpolation_promotion_v1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "selection": "all 16/16 interpolation screens; midpoint first, then alpha=.25, then alpha=.75, within each by frozen uniform subset order",
        "screen_plan_sha256": sha(HERE / "basin_interpolation_screen_plan.json"),
        "screen_records_sha256": sha(HERE / "raw/basin_interp_screen/records.jsonl"),
        "fresh_oracle_records_sha256_at_freeze": sha(HERE / "raw/fresh32_oracle/records.jsonl"),
        "fresh_oracle_maximum_final_rollouts_from_current_monotone_status": oracle_upper_bound,
        "other_committed_new_rollouts": other_committed,
        "overall_cap": 15000,
        "rollouts_available_for_promotion": remaining,
        "eligible_perfect_screens": len(eligible),
        "selected_promotions": len(arms),
        "unselected_perfect_screens": len(eligible) - len(arms),
        "assignments": assignments,
        "arms": arms,
        "new_continuations": len(arms) * 48,
        "maximum_physical_steps": sum((850 - arm["absolute_step"]) * 48 for arm in arms),
    }
    plan["content_sha256"] = semantic_sha(plan)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({
        "oracle_upper_bound": oracle_upper_bound, "remaining_before_promotion": remaining,
        "eligible": len(eligible), "selected": len(arms), "new_continuations": plan["new_continuations"],
        "worst_case_total_new_rollouts": other_committed + oracle_upper_bound + plan["new_continuations"],
        "plan_sha256": sha(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
