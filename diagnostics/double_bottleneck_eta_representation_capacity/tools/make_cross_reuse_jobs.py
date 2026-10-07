#!/usr/bin/env python3
"""Cross-evaluate every promoted-representation success vector not already global."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin/episode_catalog.json"


def theta_key(theta) -> str:
    canonical = json.dumps([round(float(value), 14) for value in theta], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(slug: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{slug}_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6", "P4-Agent12"), required=True)
    args = parser.parse_args()
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    if not pilot["promotions"].get(args.representation, {}).get("promote", False):
        raise RuntimeError(f"{args.representation} was not promoted")
    slug = args.representation.replace("-", "_")
    rows = [row for row in load_rows(slug) if row.get("stage") not in ("stochastic_robustness", "cross_state_reuse")]
    stage_a_keys = {
        theta_key(row["theta"])
        for row in rows
        if row["stage"] in ("pilot_stage_a", "full_stage_a")
    }
    vectors = {}
    sources = {}
    for row in rows:
        if not row["success"]:
            continue
        key = theta_key(row["theta"])
        if key in stage_a_keys:
            continue
        vectors[key] = row["theta"]
        sources.setdefault(key, set()).add(row["episode_id"])
    already = {(theta_key(row["theta"]), row["episode_id"]) for row in rows}
    catalog = json.loads(PRIOR.read_text())
    episodes = []
    for population, source in (("safe_timeout_target", catalog["targets"]), ("baseline_success_control", catalog["controls"])):
        for row in source:
            episodes.append({**row, "population": population, "pilot_stratum": "full_target" if population == "safe_timeout_target" else "full_control"})
    jobs = []
    for key, theta in sorted(vectors.items()):
        for episode in episodes:
            if (key, episode["episode_id"]) in already:
                continue
            jobs.append({
                **episode,
                "stage": "cross_state_reuse",
                "representation": args.representation,
                "parameter_id": f"X{key}",
                "sample_type": "successful_local_vector_cross_state",
                "theta": theta,
                "source_episode_ids": sorted(sources[key]),
                "job_id": f"cross_state_reuse|{args.representation}|X{key}|{episode['episode_id']}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": file_sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": "cross_state_reuse",
        "unique_success_vectors": len(vectors),
        "vectors": [{"parameter_id": f"X{key}", "theta": vectors[key], "source_episode_ids": sorted(sources[key])} for key in sorted(vectors)],
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{slug}_cross_state_reuse.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "unique_local_success_vectors": len(vectors), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
