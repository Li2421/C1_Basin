"""Compute matched starting-state P0/OrthoFlow3 Gram diagnostics from screen records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def main() -> None:
    records = []
    for path in sorted((HERE / "raw/global256_exact").glob("shard*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    selected = {}
    for row in records:
        if int(row["eta_index"]) == 0:
            selected[row["state_id"]] = row
    if len(selected) != 32:
        raise RuntimeError(("expected 32 state records", len(selected)))
    output = []
    for state_id, row in sorted(selected.items()):
        first = row["first_step"]
        goal, flow_perp, relation = [np.asarray(value, dtype=np.float64).reshape(2, 2) for value in first["basis_values"]]
        safe = np.asarray(first["u_safe"], dtype=np.float64).reshape(2, 2)
        p0_terms = (goal, safe, relation)
        of3_terms = (goal, flow_perp, relation)
        p0_matrix = np.stack([value.reshape(-1) for value in p0_terms], axis=1)
        of3_matrix = np.stack([value.reshape(-1) for value in of3_terms], axis=1)
        p0_agent = [cosine(goal[index], safe[index]) for index in range(2)]
        of3_agent = [cosine(goal[index], flow_perp[index]) for index in range(2)]
        output.append(
            {
                "state_id": state_id,
                "p0_agent_goal_middle_cosine_mean": float(np.nanmean(p0_agent)),
                "p0_agent_goal_middle_cosine_max_abs": float(np.nanmax(np.abs(p0_agent))),
                "orthoflow3_agent_goal_middle_cosine_mean": float(np.nanmean(of3_agent)),
                "orthoflow3_agent_goal_middle_cosine_max_abs": float(np.nanmax(np.abs(of3_agent))),
                "p0_joint_goal_middle_cosine": cosine(p0_matrix[:, 0], p0_matrix[:, 1]),
                "orthoflow3_joint_goal_middle_cosine": cosine(of3_matrix[:, 0], of3_matrix[:, 1]),
                "p0_gram": json.dumps((p0_matrix.T @ p0_matrix).tolist(), separators=(",", ":")),
                "orthoflow3_gram": json.dumps((of3_matrix.T @ of3_matrix).tolist(), separators=(",", ":")),
                "p0_column_norms": json.dumps(np.linalg.norm(p0_matrix, axis=0).tolist(), separators=(",", ":")),
                "orthoflow3_column_norms": json.dumps(np.linalg.norm(of3_matrix, axis=0).tolist(), separators=(",", ":")),
            }
        )
    with (HERE / "subset_basis_gram_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(json.dumps({
        "states": len(output),
        "p0_median_agent_max_abs": float(np.median([row["p0_agent_goal_middle_cosine_max_abs"] for row in output])),
        "orthoflow3_median_agent_max_abs": float(np.median([row["orthoflow3_agent_goal_middle_cosine_max_abs"] for row in output])),
        "p0_median_joint": float(np.median([row["p0_joint_goal_middle_cosine"] for row in output])),
        "orthoflow3_median_joint": float(np.median([row["orthoflow3_joint_goal_middle_cosine"] for row in output])),
    }, indent=2))


if __name__ == "__main__":
    main()
