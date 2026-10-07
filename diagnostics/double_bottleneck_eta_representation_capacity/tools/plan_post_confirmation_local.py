#!/usr/bin/env python3
"""Add the registered local check when dense negative confirmation finds success."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(prefix: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{prefix}_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6"), required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    slug = args.representation.replace("-", "_")
    suffix = "refinement" if args.scope == "pilot" else "full_refinement"
    manifest_path = STUDY / "jobs" / f"{slug}_{suffix}.json"
    manifest = json.loads(manifest_path.read_text())
    rows = load(f"{slug}_{suffix}")
    if len(rows) != len(manifest["jobs"]):
        raise RuntimeError(f"{suffix} incomplete: {len(rows)} != {len(manifest['jobs'])}")
    design = json.loads((STUDY / "parameter_designs.json").read_text())[args.representation]
    rep = REPRESENTATIONS[args.representation]
    embed = rep.embed_p0()
    jobs, decisions = [], []
    for decision in manifest["decisions"]:
        if decision["branch"] != "dense_same_domain_after_no_success":
            continue
        episode_rows = [row for row in rows if row["episode_id"] == decision["episode_id"]]
        successes = [row for row in episode_rows if row["success"]]
        if not successes:
            decisions.append({"episode_id": decision["episode_id"], "branch": "no_dense_success", "jobs": 0, "center": None})
            continue
        center = min(successes, key=lambda row: (
            row["correction"]["raw_norm"]["mean"],
            float(np.linalg.norm((np.asarray(row["theta"]) - embed) / (rep.high - rep.low))),
            row["parameter_id"],
        ))
        samples = local_samples(args.representation, center["theta"], design)
        decisions.append({"episode_id": decision["episode_id"], "branch": "local_after_dense_success", "jobs": len(samples), "center": center["parameter_id"]})
        episode = {key: center[key] for key in ("episode_id", "population", "pilot_stratum", "set", "family_id", "regime", "seed", "rollout_id", "baseline_outcome") if key in center}
        if "baseline_steps" in center:
            episode["baseline_steps"] = center["baseline_steps"]
        stage = f"{args.scope}_post_confirmation_local"
        for sample in samples:
            jobs.append({
                **episode,
                "stage": stage,
                "representation": args.representation,
                **sample,
                "refinement_center_id": center["parameter_id"],
                "refinement_center_theta": center["theta"],
                "job_id": f"{stage}|{args.representation}|{center['episode_id']}|{sample['parameter_id']}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": f"{args.scope}_post_confirmation_local",
        "decisions": decisions,
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{slug}_{args.scope}_post_confirmation_local.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "scope": args.scope, "jobs": len(jobs), "dense_successes": sum(d["branch"] == "local_after_dense_success" for d in decisions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
