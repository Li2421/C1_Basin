"""Install completed CPU/cache audit artifacts at the required root names."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1")

MAPPING = {
    "stage_a_work/target_geometry_424.csv": "target_geometry_424.csv",
    "stage_a_work/canonical_eta_frequency.csv": "canonical_eta_frequency.csv",
    "stage_a_work/target_geometry_report.md": "target_geometry_report.md",
    "stage_bc_work/exact_feature_collisions.csv": "exact_feature_collisions.csv",
    "stage_bc_work/nearest_neighbor_target_jumps.csv": "nearest_neighbor_target_jumps.csv",
    "stage_bc_work/feature_aliasing_report.md": "feature_aliasing_report.md",
    "stage_bc_work/zero_active_probe_results.json": "zero_active_probe_results.json",
    "stage_bc_work/current_predictor_zero_active_behavior.csv": "current_predictor_zero_active_behavior.csv",
    "stage_bc_work/canonical_smoothness.csv": "canonical_smoothness.csv",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    installed = []
    for source_name, destination_name in MAPPING.items():
        source = HERE / source_name
        destination = HERE / destination_name
        if not source.exists():
            raise RuntimeError(f"missing completed source artifact: {source}")
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"refusing to replace differing artifact: {destination}")
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        installed.append(
            {
                "source": str(source),
                "source_sha256": sha(source),
                "destination": str(destination),
                "destination_sha256": sha(destination),
                "byte_identical": source.read_bytes() == destination.read_bytes(),
            }
        )
    path = HERE / "cached_output_materialization.json"
    path.write_text(json.dumps({"status": "PASS", "installed": installed}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "files": len(installed)}, indent=2))


if __name__ == "__main__":
    main()
