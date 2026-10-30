"""Quantify frozen Gap1 N=50 opposing traces without changing audit labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_scale_opposing import audit_trace


def summarize(directories: list[Path], output: Path) -> dict:
    rows = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if len(summary["rollouts"]) != 1:
            raise ValueError(f"expected exactly one rollout: {directory}")
        item = summary["rollouts"][0]
        trace_path = directory / "traces" / f"{item['rollout_id']}.npz"
        label = audit_trace(trace_path)
        with np.load(trace_path, allow_pickle=False) as trace:
            position = np.asarray(trace["positions"], dtype=np.float64)
            goals = np.asarray(trace["goals"], dtype=np.float64)
            flow = np.asarray(trace["u_flow"], dtype=np.float64)
            safe = np.asarray(trace["u_safe"], dtype=np.float64)
            pair = np.asarray(trace["active_pair_count"], dtype=np.int64)
            wall = np.asarray(trace["active_wall_count"], dtype=np.int64)
            swept_wall = np.asarray(trace["swept_wall_clearance"], dtype=np.float64)
            swept_agent = np.asarray(trace["swept_agent_clearance"], dtype=np.float64)
            meta = json.loads(str(trace["metadata_json"].item()))
        if summary["checkpoint_sha256"] not in meta["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {directory}")
        distance = np.linalg.norm(goals[None] - position, axis=2)
        goal_direction = (goals[None] - position[:-1]) / np.maximum(
            distance[:-1, :, None], 1e-12
        )
        flow_goalward = np.sum(flow * goal_direction, axis=2).mean(axis=1)
        safe_goalward = np.sum(safe * goal_direction, axis=2).mean(axis=1)
        correction = np.linalg.norm(safe - flow, axis=2).mean(axis=1)
        flow_speed = np.linalg.norm(flow, axis=2).mean(axis=1)
        safe_speed = np.linalg.norm(safe, axis=2).mean(axis=1)
        interaction = label["interaction_step"]
        tail_start = max(0, len(safe) - max(1000, len(safe) // 4))
        windows = {"pre_interaction": slice(0, interaction),
                   "post_interaction": slice(interaction, None),
                   "last_quarter": slice(tail_start, None)}
        metrics = {}
        for name, window in windows.items():
            if window.stop is not None and window.stop <= window.start:
                continue
            metrics[name] = {
                "flow_goalward_mps": float(np.mean(flow_goalward[window])),
                "safe_goalward_mps": float(np.mean(safe_goalward[window])),
                "flow_speed_mps": float(np.mean(flow_speed[window])),
                "safe_speed_mps": float(np.mean(safe_speed[window])),
                "mean_correction_per_agent_mps": float(np.mean(correction[window])),
                "pair_active_fraction": float(np.mean(pair[window] > 0)),
                "wall_active_fraction": float(np.mean(wall[window] > 0)),
            }
        rows.append({**label,
            "source_directory": str(directory),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "goal_distance_at_interaction_mean_m": float(distance[interaction].mean())
                if interaction is not None else None,
            "near_gate_agents_final_abs_x_le_1p5": int(np.sum(np.abs(position[-1, :, 0]) <= 1.5)),
            "min_swept_wall_clearance_m": float(np.min(swept_wall)),
            "min_swept_agent_clearance_m": float(np.min(swept_agent)),
            "windows": metrics,
        })
    keys = ("pre_interaction_progress_rate", "post_interaction_progress_rate",
            "pre_interaction_mean_speed", "post_interaction_mean_speed",
            "tail_goal_progress_per_agent", "tail_pair_active_fraction")
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    for window in ("pre_interaction", "post_interaction", "last_quarter"):
        for key in rows[0]["windows"][window]:
            aggregate[f"{window}.{key}"] = float(np.mean(
                [row["windows"][window][key] for row in rows]))
    result = {
        "schema": "gap1_n50_frozen_opposing_mechanism_v1",
        "classification": "predeclared audit_scale_opposing screening labels; no threshold changed",
        "runs": len(rows),
        "counts": {key: sum(row["screening_label"] == key for row in rows)
                   for key in sorted({row["screening_label"] for row in rows})},
        "aggregate_means": aggregate,
        "rollouts": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evaluation, args.output)
    print(json.dumps({"counts": result["counts"],
                      "aggregate_means": result["aggregate_means"]}, indent=2))


if __name__ == "__main__":
    main()
