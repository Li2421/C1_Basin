#!/usr/bin/env python3
"""Verify frozen inputs and run the maintained regression suite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
EXPECTED = {
    "checkpoint": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl", "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd"),
    "dataset_manifest": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json", "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56"),
    "macflow_source": (ROOT / "double_bottleneck/flowbc_4a_agent.py", "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"),
    "environment": (ROOT / "double_bottleneck/environment.py", "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc"),
    "hard_projection": (ROOT / "shared_control/hard_projection.py", "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79"),
    "canonical_p0": (ROOT / "shared_control/diagnostic_corrector.py", "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    hashes = {}
    for name, (path, expected) in EXPECTED.items():
        actual = sha(path)
        hashes[name] = {"path": str(path.relative_to(ROOT)), "expected": expected, "actual": actual, "match": actual == expected}
    if not all(item["match"] for item in hashes.values()):
        raise RuntimeError("frozen input hash mismatch")
    environment = os.environ.copy()
    environment["C1_PYTHON"] = "/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python"
    process = subprocess.run(["bash", "scripts/test.sh"], cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (STUDY / "logs/regression.log").write_text(process.stdout)
    result = {
        "schema": "double_bottleneck_eta3_full_sobol_regression_v1",
        "command": "C1_PYTHON=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python bash scripts/test.sh",
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "expected_suite_counts": {"shared_and_single_integrator": 58, "toy_giveway": 2, "double_bottleneck": 19, "total": 79},
        "toy_source_modified_by_this_study": False,
        "frozen_hashes": hashes,
        "log": str((STUDY / "logs/regression.log").relative_to(ROOT)),
    }
    (STUDY / "regression_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "hashes_match": True, "returncode": process.returncode}, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
