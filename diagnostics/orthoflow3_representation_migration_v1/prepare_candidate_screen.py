"""Apply the archived top-eight rule and freeze the 16-stream screen plan."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
DESIGN = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
WIDTH = HIGH - LOW


def read_rows(pattern: str) -> list[dict]:
    output = []
    for path in sorted((HERE / "raw").glob(pattern)):
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidates(successful: list[int], theta: np.ndarray, points: list[dict]) -> list[dict]:
    unit = (theta - LOW) / WIDTH
    successful_set = set(successful)
    output = []
    for eta_index in successful:
        distances = np.linalg.norm(unit - unit[eta_index], axis=1)
        distances[eta_index] = np.inf
        nearest = np.argsort(distances, kind="stable")[:8]
        output.append(
            {
                "eta_index": eta_index,
                "parameter_id": points[eta_index]["parameter_id"],
                "eta": theta[eta_index].tolist(),
                "successful_neighbors_among_8": sum(int(index) in successful_set for index in nearest),
                "normalized_distance_to_eta_zero": float(np.linalg.norm(theta[eta_index] / WIDTH)),
            }
        )
    output.sort(key=lambda row: (-row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
    return output[:8]


def main() -> None:
    output = HERE / "candidate_screen16_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite candidate screen plan")
    rows = read_rows("global256_exact/shard*.jsonl")
    if len(rows) != 32 * 256 or len({row["arm_id"] for row in rows}) != 32 * 256:
        raise RuntimeError(("incomplete global matrix", len(rows)))
    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    states = {row["state_id"]: row for row in subset["selected_states"]}
    points = json.loads(DESIGN.read_text())["points"]
    theta = np.asarray([point["theta"] for point in points], dtype=np.float64)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["state_id"]][int(row["eta_index"])] = row
    arms = []
    summary = []
    for state_id, state in states.items():
        state_rows = grouped[state_id]
        if len(state_rows) != 256:
            raise RuntimeError((state_id, len(state_rows)))
        successful = [index for index in range(256) if state_rows[index]["success"]]
        selected = candidates(successful, theta, points)
        summary.append(
            {
                "state_id": state_id, "selection_rank": state["selection_rank"],
                "successful_exact_candidates": len(successful),
                "exact_Q_max": 1.0 if successful else 0.0,
                "selected_candidate_count": len(selected),
                "selected_eta_indices": ";".join(str(row["eta_index"]) for row in selected),
            }
        )
        for rank, candidate in enumerate(selected):
            arms.append(
                {
                    "arm_id": f"S16__{state['selection_rank']:02d}__{candidate['eta_index']:03d}",
                    "basis_family": "orthoflow3",
                    "state_id": state_id, "selection_rank": state["selection_rank"],
                    "state_file": state["state_file"], "state_sha256": state["state_sha256"],
                    "absolute_step": state["absolute_step"], "rng_namespace": state["rng_namespace"],
                    "eta_index": candidate["eta_index"], "parameter_id": candidate["parameter_id"],
                    "eta": candidate["eta"], "selection_priority": rank,
                    "successful_neighbors_among_8": candidate["successful_neighbors_among_8"],
                    "normalized_distance_to_eta_zero": candidate["normalized_distance_to_eta_zero"],
                    "seeds": state["matched_flow_seeds"][1:16],
                    "reused_exact_seed": state["matched_flow_seeds"][0],
                }
            )
    plan = {
        "schema": "orthoflow3_migration_arm_plan_v1", "stage": "candidate_screen16",
        "basis_family": "orthoflow3", "selection_rule": "archived deterministic top-eight rule",
        "first_seed_reused_from_global256": True, "arms": arms,
        "maximum_new_continuations": 15 * len(arms),
        "maximum_physical_steps": sum((850 - int(arm["absolute_step"])) * len(arm["seeds"]) for arm in arms),
    }
    plan["content_sha256"] = canonical_hash(plan)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    with (HERE / "global256_state_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"states_with_exact_basin": sum(row['successful_exact_candidates'] > 0 for row in summary), "selected_arms": len(arms), "new_screen_rollouts": 15 * len(arms), "selected_per_state": [row['selected_candidate_count'] for row in summary]}, indent=2))


if __name__ == "__main__":
    main()
