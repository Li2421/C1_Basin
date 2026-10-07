#!/usr/bin/env python3
"""Apply the registered refinement/negative-confirmation rule to full results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta_representation_capacity.tools.plan_refinement import local_samples
from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin/episode_catalog.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(prefix: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{prefix}_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate full Stage-A job ids")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6", "P4-Agent12"), required=True)
    args = parser.parse_args()
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    if not pilot["promotions"].get(args.representation, {}).get("promote", False):
        raise RuntimeError(f"{args.representation} was not promoted")
    slug = args.representation.replace("-", "_")
    design = json.loads((STUDY / "parameter_designs.json").read_text())[args.representation]
    catalog = json.loads(PRIOR.read_text())
    episodes = []
    for population, source in (("safe_timeout_target", catalog["targets"]), ("baseline_success_control", catalog["controls"])):
        for row in source:
            episodes.append({**row, "population": population, "pilot_stratum": "full_target" if population == "safe_timeout_target" else "full_control"})
    stage_a_rows = load_rows(f"{slug}_full_stage_a")
    expected = len(episodes) * len(design["stage_a"])
    if len(stage_a_rows) != expected:
        raise RuntimeError(f"full Stage A incomplete: {len(stage_a_rows)} != {expected}")
    inheritance_rows = load_rows(f"{slug}_full_p0_inheritance")
    inheritance_manifest = json.loads((STUDY / "jobs" / f"{slug}_full_p0_inheritance.json").read_text())
    if len(inheritance_rows) != len(inheritance_manifest["jobs"]):
        raise RuntimeError("full P0 inheritance control incomplete")
    rows = stage_a_rows + inheritance_rows
    by_episode = {episode["episode_id"]: [row for row in rows if row["episode_id"] == episode["episode_id"]] for episode in episodes}
    rep = REPRESENTATIONS[args.representation]
    embed = rep.embed_p0()
    jobs, decisions = [], []
    for episode in episodes:
        successes = [row for row in by_episode[episode["episode_id"]] if row["success"]]
        if successes and episode["population"] == "safe_timeout_target":
            center = min(successes, key=lambda row: (
                row["correction"]["raw_norm"]["mean"],
                float(np.linalg.norm((np.asarray(row["theta"]) - embed) / (rep.high - rep.low))),
                row["parameter_id"],
            ))
            samples = local_samples(args.representation, center["theta"], design)
            branch = "local_after_success"
        elif not successes:
            center = None
            samples = design["stage_c_negative_confirmation"]
            branch = "dense_same_domain_after_no_success"
        else:
            center = None
            samples = []
            branch = "control_success_no_refinement"
        decisions.append({"episode_id": episode["episode_id"], "branch": branch, "jobs": len(samples), "center": None if center is None else center["parameter_id"]})
        for sample in samples:
            job = {
                **episode,
                "stage": "full_refinement",
                "representation": args.representation,
                **sample,
                "job_id": f"full_refinement|{args.representation}|{episode['episode_id']}|{sample['parameter_id']}",
            }
            if center is not None:
                job["refinement_center_id"] = center["parameter_id"]
                job["refinement_center_theta"] = center["theta"]
            jobs.append(job)
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": "full_refinement",
        "decisions": decisions,
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{slug}_full_refinement.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "jobs": len(jobs), "branches": {branch: sum(d["branch"] == branch for d in decisions) for branch in sorted({d["branch"] for d in decisions})}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
