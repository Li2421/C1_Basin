#!/usr/bin/env python3
"""Embed all newly discovered dense-P0 successes into each nested candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import P0_ANCHOR, REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("P1-Agent6", "P2-Pair8", "P3-Temporal6"), required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    p0_rows = []
    for path in sorted((STUDY / "raw").glob("P0_3D_stage_a*_shard*.jsonl")):
        p0_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    values = {}
    for row in p0_rows:
        eta = np.asarray(row["theta"], dtype=np.float64)
        if row["success"] and row["parameter_id"] != "ZERO" and not np.allclose(eta, P0_ANCHOR, atol=1e-14, rtol=0):
            values[tuple(np.round(eta, 14))] = {"source_parameter_id": row["parameter_id"], "eta": eta.tolist()}
    if args.scope == "pilot":
        episodes = json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]
    else:
        prior = json.loads((ROOT / "diagnostics/double_bottleneck_eta3_basin/episode_catalog.json").read_text())
        episodes = []
        for population, source in (("safe_timeout_target", prior["targets"]), ("baseline_success_control", prior["controls"])):
            for row in source:
                episodes.append({**row, "population": population, "pilot_stratum": "full_target" if population == "safe_timeout_target" else "full_control"})
    rep = REPRESENTATIONS[args.representation]
    jobs = []
    for source in sorted(values.values(), key=lambda row: row["source_parameter_id"]):
        theta = rep.embed_p0(source["eta"]).tolist()
        for episode in episodes:
            parameter_id = f"INHERIT_{source['source_parameter_id']}"
            jobs.append({
                **episode,
                "stage": f"{args.scope}_p0_inheritance",
                "representation": args.representation,
                "parameter_id": parameter_id,
                "sample_type": "exact_p0_subspace_inheritance",
                "theta": theta,
                "source_p0_eta": source["eta"],
                "job_id": f"{args.scope}_p0_inheritance|{args.representation}|{episode['episode_id']}|{parameter_id}",
            })
    output = {
        "schema": "double_bottleneck_eta_capacity_jobs_v1",
        "preregistration_sha256": sha(STUDY / "PREREGISTRATION.json"),
        "amendment_sha256": sha(STUDY / "H0_INHERITANCE_AMENDMENT.json"),
        "representation": args.representation,
        "stage": f"{args.scope}_p0_inheritance",
        "inherited_p0_points": len(values),
        "jobs": jobs,
    }
    suffix = "p0_inheritance" if args.scope == "pilot" else "full_p0_inheritance"
    path = STUDY / "jobs" / f"{args.representation.replace('-', '_')}_{suffix}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representation": args.representation, "scope": args.scope, "inherited_p0_points": len(values), "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
