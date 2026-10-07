#!/usr/bin/env python3
"""Materialize the pre-registered Stage-B/Stage-C eta job manifests.

The branching rule is entirely determined by frozen earlier-stage outcomes:
Stage B applies the registered local rule after a Stage-A success and otherwise
uses the second half of the nested Sobol design.  Stage C applies the same
local rule only when Stage B is the first stage to observe success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
TOY = np.asarray((1.0, 0.0, 0.25), dtype=np.float64)
STEP = 1.0 / 16.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(STUDY.glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate job rows under {pattern}")
    return rows


def normalized_distance(eta: np.ndarray) -> float:
    return float(np.linalg.norm((eta - TOY) / (HIGH - LOW)))


def local_samples(center: np.ndarray) -> list[dict]:
    offsets = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            value = np.zeros(3, dtype=np.float64)
            value[axis] = sign * STEP
            offsets.append(value)
    offsets.extend(np.asarray(signs, dtype=np.float64) * STEP for signs in product((-1.0, 1.0), repeat=3))
    center_unit = (center - LOW) / (HIGH - LOW)
    values: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for offset in offsets:
        eta = LOW + np.clip(center_unit + offset, 0.0, 1.0) * (HIGH - LOW)
        key = tuple(np.round(eta, 15))
        if key not in seen and not np.allclose(eta, center, atol=1e-14, rtol=0.0):
            seen.add(key)
            values.append(eta)
    return [
        {
            "eta_id": f"L{index:03d}",
            "eta": eta.tolist(),
            "sample_type": "local_registered_neighbor",
        }
        for index, eta in enumerate(values)
    ]


def job(episode: dict, stage: str, sample: dict, center: dict | None = None) -> dict:
    result = {
        **episode,
        "stage": stage,
        "sample_type": sample["sample_type"],
        "eta_id": sample["eta_id"],
        "eta": sample["eta"],
        "job_id": f"{stage}|{episode['episode_id']}|{sample['eta_id']}",
    }
    if center is not None:
        result["refinement_center_eta_id"] = center["eta_id"]
        result["refinement_center_eta"] = center["eta"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("b", "c"), required=True)
    args = parser.parse_args()
    catalog = json.loads((STUDY / "episode_catalog.json").read_text())
    samples = json.loads((STUDY / "eta_samples.json").read_text())
    episodes = {row["episode_id"]: row for row in catalog["targets"]}
    stage_a = load_rows("raw/stage_a_shard*.jsonl")
    expected_stage_a = (len(catalog["targets"]) + len(catalog["controls"])) * 34
    if len(stage_a) != expected_stage_a:
        raise RuntimeError(f"Stage A incomplete: {len(stage_a)} != {expected_stage_a}")
    by_episode_a = {
        episode_id: [row for row in stage_a if f"{row['set']}|{row['rollout_id']:03d}" == episode_id]
        for episode_id in episodes
    }
    jobs: list[dict] = []
    decisions: list[dict] = []
    if args.stage == "b":
        for episode_id, episode in episodes.items():
            successes = [
                row for row in by_episode_a[episode_id]
                if row["eta_id"].startswith("A") and row["success"]
            ]
            if successes:
                center = min(
                    successes,
                    key=lambda row: (normalized_distance(np.asarray(row["eta"])), row["eta_id"]),
                )
                local = local_samples(np.asarray(center["eta"], dtype=np.float64))
                jobs.extend(job(episode, "stage_b", sample, center) for sample in local)
                decisions.append({"episode_id": episode_id, "branch": "local_after_stage_a_success", "center": center["eta_id"], "jobs": len(local)})
            else:
                dense = samples["stage_b_dense_global"]
                jobs.extend(job(episode, "stage_b", sample) for sample in dense)
                decisions.append({"episode_id": episode_id, "branch": "dense_global_after_no_stage_a_success", "jobs": len(dense)})
        stage_name = "stage_b"
    else:
        stage_b = load_rows("raw/stage_b_shard*.jsonl")
        stage_b_manifest = json.loads((STUDY / "jobs/stage_b.json").read_text())
        if len(stage_b) != len(stage_b_manifest["jobs"]):
            raise RuntimeError(f"Stage B incomplete: {len(stage_b)} != {len(stage_b_manifest['jobs'])}")
        for episode_id, episode in episodes.items():
            if any(row["success"] for row in by_episode_a[episode_id] if row["eta_id"].startswith("A")):
                continue
            dense_successes = [
                row for row in stage_b
                if f"{row['set']}|{row['rollout_id']:03d}" == episode_id
                and row["sample_type"] == "stage_b_dense_sobol"
                and row["success"]
            ]
            if not dense_successes:
                decisions.append({"episode_id": episode_id, "branch": "no_success_after_dense_global", "jobs": 0})
                continue
            center = min(
                dense_successes,
                key=lambda row: (normalized_distance(np.asarray(row["eta"])), row["eta_id"]),
            )
            local = local_samples(np.asarray(center["eta"], dtype=np.float64))
            jobs.extend(job(episode, "stage_c", sample, center) for sample in local)
            decisions.append({"episode_id": episode_id, "branch": "local_after_dense_global_success", "center": center["eta_id"], "jobs": len(local)})
        stage_name = "stage_c"
    output = {
        "schema": "double_bottleneck_eta3_jobs_v1",
        "stage": stage_name,
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "decisions": decisions,
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{stage_name}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": stage_name, "episodes": len(decisions), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
