#!/usr/bin/env python3
"""Apply the pre-registered local/negative-confirmation pilot branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(prefix: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{prefix}_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def local_samples(name, center, design):
    rep = REPRESENTATIONS[name]
    radius = float(design["local_rule"]["normalized_radius"])
    center_unit = (np.asarray(center) - rep.low) / (rep.high - rep.low)
    offsets = []
    for dimension in range(rep.dimension):
        for sign in (-1.0, 1.0):
            value = np.zeros(rep.dimension)
            value[dimension] = sign * radius
            offsets.append(value)
    sobol = qmc.Sobol(rep.dimension, scramble=True, seed=design["local_rule"]["sobol_seed"]).random_base2(4)
    offsets.extend((2.0 * sobol - 1.0) * radius)
    result, seen = [], set()
    for offset in offsets:
        theta = rep.low + np.clip(center_unit + offset, 0.0, 1.0) * (rep.high - rep.low)
        key = tuple(np.round(theta, 14))
        if key not in seen and not np.allclose(theta, center, atol=1e-14, rtol=0):
            seen.add(key)
            result.append({"parameter_id": f"L{len(result):03d}", "theta": theta.tolist(), "sample_type": "registered_local_neighbor"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6"), required=True)
    args = parser.parse_args()
    slug = args.representation.replace("-", "_")
    designs = json.loads((STUDY / "parameter_designs.json").read_text())
    design = designs[args.representation]
    episodes = json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]
    stage_a_rows = load_rows(f"{slug}_stage_a")
    expected = len(episodes) * len(design["stage_a"])
    if len(stage_a_rows) != expected:
        raise RuntimeError(f"Stage A incomplete: {len(stage_a_rows)} != {expected}")
    inheritance_rows = load_rows(f"{slug}_p0_inheritance")
    inheritance_manifest = json.loads((STUDY / "jobs" / f"{slug}_p0_inheritance.json").read_text())
    if len(inheritance_rows) != len(inheritance_manifest["jobs"]):
        raise RuntimeError("P0 inheritance control incomplete")
    rows = stage_a_rows + inheritance_rows
    by_episode = {episode["episode_id"]: [row for row in rows if row["episode_id"] == episode["episode_id"]] for episode in episodes}
    rep = REPRESENTATIONS[args.representation]
    embed = rep.embed_p0()
    jobs, decisions = [], []
    for episode in episodes:
        erows = by_episode[episode["episode_id"]]
        successes = [row for row in erows if row["success"]]
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
                **episode, "stage": "pilot_refinement", "representation": args.representation,
                **sample,
                "job_id": f"pilot_refinement|{args.representation}|{episode['episode_id']}|{sample['parameter_id']}",
            }
            if center is not None:
                job["refinement_center_id"] = center["parameter_id"]
                job["refinement_center_theta"] = center["theta"]
            jobs.append(job)
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": "pilot_refinement",
        "decisions": decisions,
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{slug}_refinement.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "jobs": len(jobs), "branches": {branch: sum(d["branch"] == branch for d in decisions) for branch in sorted({d["branch"] for d in decisions})}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
