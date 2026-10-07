#!/usr/bin/env python3
"""Build the pre-registered fixed-state MACFlow-seed robustness jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
NEW_SEEDS = (1103, 1201, 1301, 1409)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(slug: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{slug}_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    unique = {}
    for row in rows:
        if row["job_id"] in unique:
            raise RuntimeError(f"duplicate {row['job_id']}")
        unique[row["job_id"]] = row
    return list(unique.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6", "P4-Agent12"), required=True)
    args = parser.parse_args()
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    if not pilot["promotions"].get(args.representation, {}).get("promote", False):
        raise RuntimeError(f"{args.representation} was not promoted")
    slug = args.representation.replace("-", "_")
    catalog = {row["episode_id"]: row for row in json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]}
    rep = REPRESENTATIONS[args.representation]
    embed = rep.embed_p0()
    rows = [row for row in load_rows(slug) if row["population"] == "safe_timeout_target" and row["success"] and row["stage"] in ("pilot_stage_a", "pilot_refinement", "pilot_post_confirmation_local")]
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    centers = []
    for episode_id, success_rows in sorted(by_episode.items()):
        centers.append(min(success_rows, key=lambda row: (
            row["correction"]["raw_norm"]["mean"],
            float(np.linalg.norm((np.asarray(row["theta"]) - embed) / (rep.high - rep.low))),
            row["parameter_id"],
        )))
    if not centers:
        raise RuntimeError("no successful pilot centers")
    indices = np.unique(np.rint(np.linspace(0, len(centers) - 1, min(4, len(centers)))).astype(int))
    selected = [centers[index] for index in indices]
    jobs = []
    for center in selected:
        episode = catalog[center["episode_id"]]
        for seed in NEW_SEEDS:
            jobs.append({
                **episode,
                "stage": "stochastic_robustness",
                "representation": args.representation,
                "parameter_id": f"{center['parameter_id']}_SEED{seed}",
                "sample_type": "fixed_parameter_new_macflow_seed",
                "theta": center["theta"],
                "seed": seed,
                "source_parameter_id": center["parameter_id"],
                "job_id": f"stochastic_robustness|{args.representation}|{episode['episode_id']}|{center['parameter_id']}|{seed}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "representation": args.representation,
        "stage": "stochastic_robustness",
        "new_seeds": list(NEW_SEEDS),
        "selected_centers": [{"episode_id": row["episode_id"], "source_parameter_id": row["parameter_id"], "theta": row["theta"]} for row in selected],
        "jobs": jobs,
    }
    path = STUDY / "jobs" / f"{slug}_stochastic_robustness.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "centers": len(selected), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
