"""Freeze sparse, budget-aware interpolation screens on the uniform 32-state basin subset."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
LOW = np.asarray([0.0, -0.53125, -0.125], dtype=np.float64)
HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def eta_key(value) -> tuple[float, float, float]:
    return tuple(round(float(x), 12) for x in value)


def main() -> None:
    destination = HERE / "basin_interpolation_screen_plan.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen plan")
    subset = json.loads((HERE / "basin_subset_manifest.json").read_text())
    states = {row["state_id"]: row for row in subset["selected_states"]}
    b63 = defaultdict(list)
    resolved_cache = {}
    source_paths = (
        ROOT / "diagnostics/gphi_training_dataset_v3/oracle_search_results.jsonl",
        ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1/startup_oracle_search_results.jsonl",
    )
    for path in source_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["state_id"] not in states:
                continue
            for cell in row["cells"]:
                success = int(cell.get("success", cell.get("counts", {}).get("success", 0)))
                evaluated = int(cell["evaluated"])
                failures = evaluated - success
                if bool(cell["B_63_member"]):
                    b63[row["state_id"]].append(list(cell["eta"]))
                if evaluated == 64 or failures >= 2:
                    resolved_cache[(row["state_id"], eta_key(cell["eta"]))] = {
                        "state_id": row["state_id"],
                        "eta": cell["eta"],
                        "evaluated": evaluated,
                        "success": success,
                        "B_63_member": bool(cell["B_63_member"]),
                        "resolution": "complete64" if evaluated == 64 else "adaptive_two_failure_rejection",
                        "cache_path": str(path),
                        "cache_sha256": sha(path),
                    }
    strict_paths = []
    strict_dir = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/raw/robust"
    for state_id, state in states.items():
        if state["category"] != "STRICT_DEADLOCK":
            continue
        path = strict_dir / f"{state_id}.jsonl"
        strict_paths.append(path)
        grouped = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "eta" in row:
                grouped[eta_key(row["eta"])].append(row)
        for eta, records in grouped.items():
            counts = Counter(record["outcome"] for record in records)
            success = counts["success"]
            if len(records) != 64:
                raise RuntimeError((state_id, eta, len(records)))
            if success >= 63:
                b63[state_id].append(list(eta))
            resolved_cache[(state_id, eta)] = {
                "state_id": state_id,
                "eta": list(eta),
                "evaluated": 64,
                "success": success,
                "B_63_member": success >= 63,
                "resolution": "complete64",
                "cache_path": str(path),
                "cache_sha256": sha(path),
            }

    requests = []
    endpoints = []
    for state_id, points in b63.items():
        unique_points = sorted({eta_key(point) for point in points})
        if len(unique_points) < 2:
            continue
        normalized = [(np.asarray(point) - LOW) / (HIGH - LOW) for point in unique_points]
        candidates = [
            (float(np.linalg.norm(normalized[i] - normalized[j])), i, j)
            for i in range(len(unique_points))
            for j in range(i + 1, len(unique_points))
        ]
        distance, left, right = max(candidates)
        eta_a = np.asarray(unique_points[left], dtype=np.float64)
        eta_b = np.asarray(unique_points[right], dtype=np.float64)
        endpoints.append(
            {
                "state_id": state_id,
                "eta_a": eta_a.tolist(),
                "eta_b": eta_b.tolist(),
                "normalized_endpoint_distance": distance,
                "confirmed_robust_points_in_cached_cloud": len(unique_points),
            }
        )
        for alpha in (0.25, 0.5, 0.75):
            eta = ((1.0 - alpha) * eta_a + alpha * eta_b).tolist()
            requests.append({"state_id": state_id, "kind": "interpolation", "alpha": alpha, "eta": eta})
    # The selected strict state came from an active-only oracle domain.  Include
    # zero explicitly as required, but screen it rather than spending 64 futures
    # when two failures already rule out B63.
    for state_id, state in states.items():
        if state["category"] == "STRICT_DEADLOCK" and (state_id, (0.0, 0.0, 0.0)) not in resolved_cache:
            requests.append({"state_id": state_id, "kind": "strict_zero_screen", "alpha": None, "eta": [0.0, 0.0, 0.0]})

    reused = []
    unique = {}
    assignment_map = []
    for request in requests:
        key = (request["state_id"], eta_key(request["eta"]))
        if key in resolved_cache:
            reused.append({**request, **resolved_cache[key]})
            continue
        if key not in unique:
            state = states[request["state_id"]]
            if state["category"] == "STARTUP":
                seeds = list(range(95710001, 95710017))
            elif state["category"] == "STRICT_DEADLOCK":
                seeds = list(range(95310001, 95310017))
            else:
                seeds = list(range(95210001, 95210017))
            if state["category"] == "STRICT_DEADLOCK":
                strict_source = json.loads(
                    (ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/source_manifest.json").read_text()
                )
                rng_namespace = next(
                    int(row["rng_namespace"])
                    for row in strict_source["states"]
                    if row["state_id"] == state["state_id"]
                )
            else:
                rng_namespace = int(state["rng_namespace"])
            unique[key] = {
                "arm_id": f"UNIQUE{len(unique):03d}__{state['state_id']}",
                "state_id": state["state_id"],
                "state_file": state["state_file"],
                "state_sha256": state["state_sha256"],
                "absolute_step": int(state["absolute_step"]),
                "rng_namespace": rng_namespace,
                "eta": request["eta"],
                "seeds": seeds,
            }
        assignment_map.append({**request, "simulation_arm_id": unique[key]["arm_id"]})
    arms = list(unique.values())
    plan = {
        "schema": "direct_eta_basin_interpolation_screen_plan_v1",
        "status": "FROZEN_NOT_YET_LAUNCHED",
        "selection": "farthest pair of cached confirmed B63 points for each uniformly selected state with at least two points",
        "alphas": [0.25, 0.5, 0.75],
        "screening_futures": 16,
        "promotion_rule": "16/16 screen requires a fresh fixed 48-seed confirmation block before B63 membership is asserted",
        "topology_limitation": "128-point Sobol per each of 32 states is omitted because its 32768-rollout screen alone exceeds the explicit 15000 cap",
        "endpoints": endpoints,
        "reused_requests": reused,
        "requested_new_assignments": assignment_map,
        "arms": arms,
        "new_continuations": sum(len(arm["seeds"]) for arm in arms),
        "maximum_physical_steps": sum((850 - arm["absolute_step"]) * len(arm["seeds"]) for arm in arms),
        "subset_manifest_sha256": sha(HERE / "basin_subset_manifest.json"),
        "source_hashes": [{"path": str(path), "sha256": sha(path)} for path in source_paths + tuple(strict_paths)],
    }
    plan["content_sha256"] = semantic_sha(plan)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "states_with_interpolation": len(endpoints),
                "reused_requests": len(reused),
                "new_unique_arms": len(arms),
                "new_continuations": plan["new_continuations"],
                "plan_sha256": sha(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
