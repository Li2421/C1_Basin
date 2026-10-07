"""Episode-level Stage-1 validation summary; performs no selection or rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from local_common import content_hash, verify_content_hash


def wilson(k: int, n: int) -> list[float | None]:
    if n == 0: return [None, None]
    z = NormalDist().inv_cdf(0.975); p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values: return {"n": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(np.mean(array)), "median": float(np.median(array)), "p95": float(np.quantile(array, .95))}


def analyze(manifest_path: Path, result_directory: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text()); verify_content_hash(manifest, manifest_path)
    tasks = manifest["tasks"]; rows = []
    for task in tasks:
        path = result_directory / f"task_{int(task['task_index']):04d}.json"
        row = json.loads(path.read_text())
        if (not row.get("record_complete") or row.get("manifest_content_sha256") != manifest["content_sha256"]
                or row.get("source_id") != task["source_id"]):
            raise RuntimeError((path, "incomplete/mismatched full-loop result"))
        if int(row["flow_sample_count"]) != int(row["physical_transition_count"]):
            raise RuntimeError((path, "Flow/physical-step mismatch"))
        rows.append(row)
    n = len(rows); successes = sum(row["learned_success"] for row in rows)
    safety_successes = sum(row["safety_success"] for row in rows)
    rescues = sum((not row["safety_success"]) and row["learned_success"] for row in rows)
    breaks = sum(row["safety_success"] and not row["learned_success"] for row in rows)
    timeout_rows = [row for row in rows if row["safety_outcome"] == "timeout"]
    deadlock_rows = [row for row in rows if row["safety_outcome"] == "deadlock"]
    spacings = [float(value) for row in rows for value in row["local_event_spacings"]]
    result = {
        "schema": "semantic_local_full_loop_summary_v1", "controller": manifest["controller"],
        "split": manifest["split"], "policy_hash": manifest["policy_hash"], "episodes": n,
        "safety": {"success": safety_successes, "q": safety_successes/n},
        "learned": {
            "success": successes, "q": successes/n, "q_wilson95": wilson(successes, n),
            "deadlock": sum(row["learned_deadlock"] for row in rows),
            "timeout": sum(row["learned_timeout"] for row in rows),
            "collision": sum(row["learned_collision"] for row in rows),
        },
        "paired": {
            "rescue": rescues, "break": breaks, "net_rescue": rescues-breaks,
            "break_rate_given_safety_success": breaks/safety_successes if safety_successes else None,
            "timeout_rescue": sum(row["learned_success"] for row in timeout_rows),
            "timeout_total": len(timeout_rows),
            "strict_deadlock_rescue": sum(row["learned_success"] for row in deadlock_rows),
            "strict_deadlock_total": len(deadlock_rows),
        },
        "intervention": {
            "events_total": sum(row["local_event_count"] for row in rows),
            "events_per_episode": summary([float(row["local_event_count"]) for row in rows]),
            "event_spacing_steps": summary(spacings),
            "event_spacing_seconds": summary([.05*value for value in spacings]),
        },
        "jdef": summary([float(row["jdef"]) for row in rows]),
        "hard_safety": {
            "agent_collision_events": sum(row["agent_collision_events"] for row in rows),
            "wall_collision_events": sum(row["wall_collision_events"] for row in rows),
            "execution_errors": sum(row["execution_error"] is not None for row in rows),
        },
        "source_episode_level_inference": True, "final_test_used": False,
    }
    result["content_sha256"] = content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.manifest, args.result_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
