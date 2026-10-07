"""Build isolated Four-Way v13 broad-global splits without reusing old test.

Only v11's training trajectories are copied into the new root.  Fresh train,
development and frozen-test nominal states are sampled independently from the
declared scenario distribution.  The v11 development and test files are never
opened or copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from new_benchmark_common.dataset import collect_nominal_and_uniform
from .environment import Config
from .protocol import FourWayScenario


def _counts(rows):
    return {
        "split": {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")},
        "source": {source: sum(row["source"] == source for row in rows)
                   for source in ("nominal", "uniform_recovery", "targeted_wall_obstacle", "targeted_agent")},
    }


def build(source, output, staging, *, seed: int = 13123) -> dict:
    source, output, staging = map(Path, (source, output, staging))
    if output.exists() or staging.exists():
        raise FileExistsError("v13 output/staging must not exist")
    old = json.loads((source / "manifest.json").read_text())
    cfg = Config(**{key: value for key, value in old["scenario_config"].items() if key in Config.__dataclass_fields__})
    scenario = FourWayScenario(cfg)
    # Materialize a completely independent draw first.  The shared collector
    # creates its test nominal trajectories but never uses them as recovery
    # sources; no later code in this builder reads those test files.
    fresh_report = collect_nominal_and_uniform(
        scenario, staging, scenario_config=asdict(cfg),
        nominal_counts={"train": 120, "dev": 30, "test": 60}, uniform_anchors=8,
        seed=seed, rollout_prefix="v13_",
    )
    fresh = json.loads((staging / "manifest.json").read_text())
    if fresh["environment_fingerprint"] != old["environment_fingerprint"]:
        raise ValueError("cannot merge differing physical geometry")
    output.mkdir(parents=True)
    # Preserve previously acquired recovery coverage, but only training data.
    # This copies raw train artifacts, rather than loading dev/test arrays.
    shutil.copytree(source / "rollouts" / "train", output / "rollouts" / "train")
    for split in ("train", "dev", "test"):
        src, dst = staging / "rollouts" / split, output / "rollouts" / split
        shutil.copytree(src, dst, dirs_exist_ok=(split == "train"))
    old_train = [row for row in old["files"] if row["split"] == "train"]
    fresh_rows = list(fresh["files"])
    rows = old_train + fresh_rows
    manifest = {
        "schema": old["schema"], "complete": True, "scenario": old["scenario"],
        "environment_fingerprint": old["environment_fingerprint"], "agent_order": old["agent_order"],
        "observation_shape": old["observation_shape"], "action_shape": old["action_shape"],
        "scenario_config": old["scenario_config"], "files": rows, "counts": _counts(rows),
        "recovery_protocol": old["recovery_protocol"],
        "extra_report": {
            "v13_broad_global": {
                "source_v11": str(source), "old_train_trajectories_preserved": len(old_train),
                "old_development_reused": False, "old_test_copied": False, "old_test_opened": False,
                "fresh_independent_seed": seed, "fresh_nominal_requested": {"train": 120, "dev": 30, "test": 60},
                "fresh_uniform_anchors": 8, "fresh_collection": fresh_report,
                "new_test_generation_only": True, "new_test_opened_after_generation": False,
                "staging_manifest_sha256": hashlib.sha256((staging / "manifest.json").read_bytes()).hexdigest(),
            }
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {"output": str(output), "staging": str(staging), "counts": manifest["counts"],
              **manifest["extra_report"]["v13_broad_global"],
              "output_manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}
    (output / "v13_broad_global_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("output"); parser.add_argument("staging")
    parser.add_argument("--seed", type=int, default=13123)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.staging, seed=args.seed), indent=2))
