#!/usr/bin/env python3
"""Create immutable pilot rollout manifests from the pre-registered designs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True, choices=("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6"))
    parser.add_argument("--part", choices=("stage_a", "stage_c"), default="stage_a")
    args = parser.parse_args()
    designs = json.loads((STUDY / "parameter_designs.json").read_text())
    episodes = json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]
    key = "stage_a" if args.part == "stage_a" else "stage_c_negative_confirmation"
    samples = designs[args.representation][key]
    if args.representation == "P0-3D" and args.part == "stage_a":
        samples = [{"parameter_id": "ZERO", "theta": [0.0, 0.0, 0.0], "sample_type": "baseline_sentinel"}] + samples
    stage = f"pilot_{args.part}"
    jobs = []
    for episode in episodes:
        for sample in samples:
            jobs.append({
                **episode,
                "stage": stage,
                "representation": args.representation,
                **sample,
                "job_id": f"{stage}|{args.representation}|{episode['episode_id']}|{sample['parameter_id']}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": stage,
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{args.representation.replace('-', '_')}_{args.part}.json"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
