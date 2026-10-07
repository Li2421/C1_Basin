"""Zero-rollout contract test for frozen planning, runner, and V3 merge inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from common import HERE, V3, assert_frozen_sources, read_jsonl, sha256, write_json


def main() -> None:
    hashes = assert_frozen_sources()
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = read_jsonl(HERE / "startup_state_manifest.jsonl")
    arms = json.loads((HERE / "eta_zero_arms.json").read_text())
    checks = {
        "frozen_source_hashes": bool(hashes),
        "state_count_123": len(states) == 123,
        "independent_episode_count_123": len({row["source_episode"] for row in states}) == 123,
        "steps_0_through_40_three_each": Counter(row["step"] for row in states) == Counter({step: 3 for step in range(41)}),
        "one_state_per_episode": max(Counter(row["source_episode"] for row in states).values()) == 1,
        "split_group_consistency": all(len({row["split"] for row in states if row["source_group"] == group}) == 1 for group in {row["source_group"] for row in states}),
        "eta_zero_arm_per_state": len(arms) == len(states) and {arm["state_id"] for arm in arms} == {row["state_id"] for row in states},
        "matched_64_seeds": all(arm["seeds"] == protocol["oracle_seed_default"] and len(arm["seeds"]) == 64 for arm in arms),
        "V3_samples_hash_locked": sha256(V3 / "samples.npz") == protocol["base_v3_samples_sha256"],
        "V3_state_manifest_hash_locked": sha256(V3 / "state_manifest.jsonl") == protocol["base_v3_state_manifest_sha256"],
    }
    command = [sys.executable, str(HERE / "run_oracle.py"), "--arms", "eta_zero_arms.json", "--stage", "smoke_never_created", "--device", "cpu", "--dry-run"]
    process = subprocess.run(command, cwd=HERE, text=True, capture_output=True, check=False)
    checks["oracle_runner_dry_run"] = process.returncode == 0 and "DRY_RUN_PASS" in process.stdout
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "runner_stdout": process.stdout, "runner_stderr": process.stderr, "no_rollouts_executed": True}
    write_json(HERE / "pipeline_smoke_test.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
