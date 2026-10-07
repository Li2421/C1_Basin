#!/usr/bin/env python3
"""Freeze the triggered full-population OrthoFlow3 common-256 jobs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"


def main() -> int:
    robustness = json.loads((OUT / "pilot_seed_robustness.json").read_text())
    if not robustness["full_trigger"]["full_P1_triggered"]:
        raise RuntimeError("pre-registered full-P1 trigger did not fire")
    catalog = json.loads((ETA3 / "episode_catalog.json").read_text())["episodes"]
    targets = [row for row in catalog if row["population"] == "safe_timeout_target"]
    pilot = json.loads((OUT / "pilot_catalog.json").read_text())["episodes"]
    pilot_target_ids = {row["episode_id"] for row in pilot if row["population"] == "safe_timeout_target"}
    pending = [row for row in targets if row["episode_id"] not in pilot_target_ids]
    points = json.loads((ETA3 / "eta_points.json").read_text())["points"]
    scale = json.loads((OUT / "P1_SCALE.json").read_text())["scale"]
    jobs = []
    for episode in pending:
        for eta_index, point in enumerate(points):
            jobs.append({
                **episode,
                "stage": "full_global_256",
                "representation": "P1-OrthoFlow3",
                "eta_index": eta_index,
                **point,
                "ortho_scale": scale,
                "job_id": f"full_global_256|P1-OrthoFlow3|{episode['episode_id']}|{eta_index:03d}",
            })
    if len(jobs) != 49 * 256:
        raise RuntimeError("full pending job count mismatch")
    (OUT / "jobs/P1_full_global_pending.json").write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({"state": "full_global_registered", "p1_full_cached_pilot_targets": 12 * 256, "p1_full_pending_jobs": len(jobs)})
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cached_target_rows": 3072, "pending_target_rows": len(jobs), "full_rows": 61 * 256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
