"""Replace the abandoned all-new Stage-D plan with exact cache-aware arms."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def eta_key(value: list[float]) -> tuple[float, float, float]:
    return tuple(round(float(x), 12) for x in value)


def main() -> None:
    destination = HERE / "cross_eta_incremental_plan_v4.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen plan")
    abandoned = json.loads((HERE / "cross_eta_screen_plan.json").read_text())
    cache = {}
    cache_files = (
        ROOT / "diagnostics/gphi_training_dataset_v3/oracle_search_results.jsonl",
        ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1/startup_oracle_search_results.jsonl",
    )
    for path in cache_files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for cell in row["cells"]:
                success = int(cell.get("success", cell.get("counts", {}).get("success", 0)))
                evaluated = int(cell["evaluated"])
                failures = evaluated - success
                # A complete matched64 result establishes either membership or
                # rejection; an adaptive cell with two physical failures also
                # definitively rejects B63 and must not be rerun.
                if evaluated == 64 or failures >= 2:
                    cache[(row["state_id"], eta_key(cell["eta"]))] = {
                        "state_id": row["state_id"],
                        "eta": cell["eta"],
                        "evaluated": evaluated,
                        "success": success,
                        "deadlock": int(cell.get("deadlock", cell.get("counts", {}).get("deadlock", 0))),
                        "timeout": int(cell.get("timeout", cell.get("counts", {}).get("timeout", 0))),
                        "collision": int(cell.get("collision", cell.get("counts", {}).get("collision", 0))),
                        "B_63_member": bool(cell["B_63_member"]),
                        "cache_resolution": "complete64" if evaluated == 64 else "adaptive_two_failure_rejection",
                        "cache_path": str(path),
                        "cache_sha256": sha(path),
                    }
    strict_directory = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/raw/robust"
    strict_files = []
    for path in sorted(strict_directory.glob("*.jsonl")):
        grouped = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "eta" in row:
                grouped[(row["state_id"], eta_key(row["eta"]))].append(row)
        for (state_id, eta), records in grouped.items():
            counts = Counter(record["outcome"] for record in records)
            if len(records) == 64:
                cache[(state_id, eta)] = {
                    "state_id": state_id,
                    "eta": list(eta),
                    "evaluated": 64,
                    "success": counts["success"],
                    "deadlock": counts["deadlock"],
                    "timeout": counts["timeout"],
                    "collision": counts["collision"],
                    "B_63_member": counts["success"] >= 63,
                    "cache_resolution": "complete64",
                    "cache_path": str(path),
                    "cache_sha256": sha(path),
                }
                strict_files.append(path)
    reused = []
    requested_new = []
    for arm in abandoned["arms"]:
        cached = cache.get((arm["state_id"], eta_key(arm["eta"])))
        if cached is None:
            requested_new.append(arm)
        else:
            reused.append(
                {
                    "arm_id": arm["arm_id"],
                    "pair_rank": arm["pair_rank"],
                    "pair_state_side": arm["pair_state_side"],
                    "assignment": arm["assignment"],
                    **cached,
                }
            )
    unique = {}
    assignment_map = []
    for request in requested_new:
        key = (request["state_id"], eta_key(request["eta"]))
        if key not in unique:
            arm = {**request, "arm_id": f"UNIQUE{len(unique):03d}__{request['state_id']}"}
            unique[key] = arm
        assignment_map.append(
            {
                "requested_arm_id": request["arm_id"],
                "simulation_arm_id": unique[key]["arm_id"],
                "pair_rank": request["pair_rank"],
                "pair_state_side": request["pair_state_side"],
                "assignment": request["assignment"],
            }
        )
    arms = list(unique.values())
    plan = {
        "schema": "direct_eta_cross_eta_incremental_plan_v1",
        "status": "FROZEN_BEFORE_ROLLOUT",
        "supersedes_abandoned_unlaunched_plan_sha256": sha(HERE / "cross_eta_screen_plan.json"),
        "reason": "avoid rerunning exact state/eta/seed tuples already available in original oracle caches",
        "cache_files": [{"path": str(path), "sha256": sha(path)} for path in cache_files]
        + [{"path": str(path), "sha256": sha(path)} for path in sorted(set(strict_files))],
        "reused_arms": reused,
        "requested_new_assignments": assignment_map,
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
                "reused_arms": len(reused),
                "new_arms": len(arms),
                "new_continuations": plan["new_continuations"],
                "plan_sha256": sha(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
