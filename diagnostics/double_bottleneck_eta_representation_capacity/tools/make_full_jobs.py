#!/usr/bin/env python3
"""Create full-population Stage-A jobs for a promoted representation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin/episode_catalog.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        required=True,
        choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6", "P4-Agent12"),
    )
    args = parser.parse_args()
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    if not pilot["promotions"].get(args.representation, {}).get("promote", False):
        raise RuntimeError(f"{args.representation} did not pass the pre-registered promotion gate")
    catalog = json.loads(PRIOR.read_text())
    episodes = []
    for population, rows in (("safe_timeout_target", catalog["targets"]), ("baseline_success_control", catalog["controls"])):
        for row in rows:
            episodes.append({**row, "population": population, "pilot_stratum": "full_target" if population == "safe_timeout_target" else "full_control"})
    samples = json.loads((STUDY / "parameter_designs.json").read_text())[args.representation]["stage_a"]
    jobs = []
    for episode in episodes:
        for sample in samples:
            jobs.append({
                **episode,
                "stage": "full_stage_a",
                "representation": args.representation,
                **sample,
                "job_id": f"full_stage_a|{args.representation}|{episode['episode_id']}|{sample['parameter_id']}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": "full_stage_a",
        "episodes": len(episodes),
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{args.representation.replace('-', '_')}_full_stage_a.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "episodes": len(episodes), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
