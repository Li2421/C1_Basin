"""Summarize the complete matched64 ZERO and fixed learned-eta fresh32 rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    raw_path = HERE / "raw/fresh32_zero_prediction.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if len(rows) != 4096:
        raise RuntimeError(f"stage incomplete: {len(rows)}/4096")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode_index"]), row["condition"])].append(row)
    if len(grouped) != 64 or any(len(values) != 64 for values in grouped.values()):
        raise RuntimeError("not exactly 32 episodes x 2 conditions x 64 futures")

    outputs = {"ZERO": [], "CURRENT_G_ETA": []}
    for (episode, condition), values in sorted(grouped.items()):
        counts = Counter(row["outcome"] for row in values)
        eta = values[0]["eta"]
        if any(row["eta"] != eta for row in values):
            raise RuntimeError((episode, condition, "eta not fixed across futures"))
        outputs[condition].append(
            {
                "episode_index": episode,
                "source_id": values[0]["source_id"],
                "eta1": eta[0],
                "eta2": eta[1],
                "eta3": eta[2],
                "eta_norm": values[0]["eta_norm"],
                "eta_clipped": values[0]["eta_clipped"],
                "evaluated": len(values),
                "success": counts["success"],
                "deadlock": counts["deadlock"],
                "timeout": counts["timeout"],
                "collision": counts["collision"],
                "execution_error": counts["execution_error"],
                "success_rate": counts["success"] / 64.0,
                "B_63_member": counts["success"] >= 63,
                "mean_J_def": sum(float(row["J_def"]) for row in values) / 64.0,
                "mean_physical_steps": sum(int(row["physical_steps"]) for row in values) / 64.0,
                "raw_records_sha256": sha(raw_path),
            }
        )
    atomic_csv(HERE / "fresh32_zero_results.csv", outputs["ZERO"])
    atomic_csv(HERE / "fresh32_prediction_results.csv", outputs["CURRENT_G_ETA"])
    summary = {
        "schema": "fresh32_zero_prediction_summary_v1",
        "raw_records": len(rows),
        "raw_records_sha256": sha(raw_path),
        "zero_B63": sum(row["B_63_member"] for row in outputs["ZERO"]),
        "prediction_B63": sum(row["B_63_member"] for row in outputs["CURRENT_G_ETA"]),
        "zero_total_successes": sum(row["success"] for row in outputs["ZERO"]),
        "prediction_total_successes": sum(row["success"] for row in outputs["CURRENT_G_ETA"]),
        "zero_insufficient": sum(not row["B_63_member"] for row in outputs["ZERO"]),
    }
    (HERE / "fresh32_zero_prediction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
