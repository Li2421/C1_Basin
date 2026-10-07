#!/usr/bin/env python3
"""Aggregate expert-fit and projection-coupling state audits."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
NAMES = ("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6")


def stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p25": float(np.percentile(values, 25)) if len(values) else None,
        "p75": float(np.percentile(values, 75)) if len(values) else None,
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def main() -> int:
    rows = []
    for path in sorted((STUDY / "offline").glob("state_audit_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows) != 54 or len({row["state_id"] for row in rows}) != 54:
        raise RuntimeError("state audit incomplete or duplicated")
    valid = [row for row in rows if row["expert_recovery_valid"]]
    representations = {}
    for name in NAMES:
        fits = [row["fits"][name] for row in valid]
        residual = np.asarray([fit["best_executable_action_residual"] for fit in fits])
        zero = np.asarray([row["theta_zero_residual"] for row in valid])
        p0 = np.asarray([fit["p0_embedding_residual"] for fit in fits])
        representations[name] = {
            "best_residual": stats(residual),
            "p0_embedding_residual": stats(p0),
            "absolute_improvement_from_theta_zero": stats(zero - residual),
            "relative_improvement_from_theta_zero": stats((zero - residual) / np.maximum(zero, 1e-12)),
            "relative_improvement_from_P0_best": (
                stats((np.asarray([row["fits"]["P0-3D"]["best_executable_action_residual"] for row in valid]) - residual) /
                      np.maximum(np.asarray([row["fits"]["P0-3D"]["best_executable_action_residual"] for row in valid]), 1e-12))
                if name != "P0-3D" else stats(np.zeros(len(valid)))
            ),
            "raw_correction_norm": stats(fit["raw_correction_norm"] for fit in fits),
            "executable_correction_norm": stats(fit["executable_correction_norm"] for fit in fits),
            "projection_retention_ratio": stats(fit["projection_retention_ratio"] for fit in fits),
            "raw_executable_cosine": stats(fit["raw_executable_cosine"] for fit in fits),
            "rank_at_embedding_raw": stats(fit["sensitivity_at_p0_embedding"]["raw"]["effective_rank"] for fit in fits),
            "rank_at_embedding_executable": stats(fit["sensitivity_at_p0_embedding"]["executable"]["effective_rank"] for fit in fits),
            "rank_at_best_raw": stats(fit["sensitivity_at_best_fit"]["raw"]["effective_rank"] for fit in fits),
            "rank_at_best_executable": stats(fit["sensitivity_at_best_fit"]["executable"]["effective_rank"] for fit in fits),
            "raw_rank_minus_executable_rank_best": stats(
                fit["sensitivity_at_best_fit"]["raw"]["effective_rank"] - fit["sensitivity_at_best_fit"]["executable"]["effective_rank"] for fit in fits
            ),
            "best_fit_singular_values": [fit["sensitivity_at_best_fit"]["executable"]["singular_values"] for fit in fits],
        }
    output = {
        "schema": "double_bottleneck_eta_capacity_offline_summary_v1",
        "states": len(rows),
        "valid_expert_recoveries": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "validity_by_stratum": {
            stratum: {"valid": sum(row["expert_recovery_valid"] for row in group), "states": len(group)}
            for stratum in sorted({row["pilot_stratum"] for row in rows})
            for group in [[row for row in rows if row["pilot_stratum"] == stratum]]
        },
        "expert_failure_types": dict(sorted(Counter(row.get("expert_error_type", "valid") for row in rows).items())),
        "theta_zero_residual": stats(row["theta_zero_residual"] for row in valid),
        "representations": representations,
        "limitations": "Only states with a validated successful continuation from the unchanged centralized expert enter fitting; no invalid state was replaced.",
    }
    path = STUDY / "offline_expert_fit_and_projection_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with (STUDY / "offline_state_fit.csv").open("w", newline="") as handle:
        fields = ["state_id", "pilot_stratum", "source_step", "expert_recovery_valid", "theta_zero_residual"] + [f"{name}_residual" for name in NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields[:5]}
            for name in NAMES:
                flat[f"{name}_residual"] = row["fits"].get(name, {}).get("best_executable_action_residual")
            writer.writerow(flat)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
