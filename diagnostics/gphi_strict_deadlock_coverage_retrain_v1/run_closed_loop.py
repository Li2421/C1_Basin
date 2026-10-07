"""Evaluate the frozen new checkpoint on historical-11 and fresh-200 H=8."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
FRESH = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE_DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
CHECKPOINT = HERE / "best_strict_deadlock_coverage_checkpoint.npz"

sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(WIDE))
import run_evaluation as wide_runner  # noqa: E402
from pilot_common import (  # noqa: E402
    DeterministicGphi, TrainingReference, assert_frozen_sources,
    load_environment_config, sha256, write_json,
)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def selected_checkpoint() -> dict:
    payload = json.loads((HERE / "selected_checkpoint.json").read_text())
    if payload.get("selection_frozen_before_test_or_diagnostic_evaluation") is not True:
        raise RuntimeError("checkpoint was not frozen before closed-loop diagnostics")
    if Path(payload["checkpoint"]).resolve() != CHECKPOINT.resolve():
        raise RuntimeError("selected checkpoint path mismatch")
    if sha256(CHECKPOINT) != payload["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")
    return payload


def fresh_manifest() -> dict:
    path = FRESH / "fresh_test_manifest.json"
    expected = "68eecae6d04b00b0e4db7447a8e05d2443d0c886e67d14526b2eae8952841d45"
    if sha256(path) != expected:
        raise RuntimeError("fresh regression manifest changed")
    payload = json.loads(path.read_text())
    if payload.get("frozen_before_rollout") is not True or payload.get("episode_count") != 200:
        raise RuntimeError("invalid fresh regression manifest")
    overlap = json.loads((FRESH / "overlap_audit.json").read_text())
    if overlap.get("status") != "PASS" or overlap.get("exact_initial_state_match_count") != 0:
        raise RuntimeError("fresh overlap audit invalid")
    return payload


def historical_manifest() -> tuple[dict, set[int]]:
    payload = wide_runner._load_frozen_benchmark(wide_runner.BENCHMARK_MANIFEST, True)
    cases = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())["cases"]
    indices = {int(row["episode_index"]) for row in cases if row["benchmark"] == "historical"}
    if len(indices) != 11:
        raise RuntimeError(("historical strict-deadlock count", len(indices)))
    return payload, indices


def existing_valid(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except Exception:
        return False
    if row.get("record_complete") is not True or any(row.get(key) != value for key, value in expected.items()):
        return False
    trajectory = HERE / row.get("trajectory_file", "")
    return trajectory.is_file() and sha256(trajectory) == row.get("trajectory_sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("fresh200", "historical11", "both"), default="both")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    started = datetime.now(timezone.utc).isoformat()
    selection = selected_checkpoint()
    frozen = assert_frozen_sources()
    extra = wide_runner._audit_extra_sources()

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv
    from single_integrator.evaluate import load_policy
    from single_integrator.outcomes import first_event

    resolved = {
        "projection": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "feature_builder": str(Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()),
        "retry_projector": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
    }
    model = DeterministicGphi(CHECKPOINT)
    reference = TrainingReference(DATASET / "samples.npz", model)
    environment = load_environment_config(BASE_DATASET)
    config = Config(**environment)
    cbf = CBFConfig()
    policy, provenance = load_policy(wide_runner.FLOW_CHECKPOINT)
    if not provenance or provenance.get("evaluation_environment") != environment:
        raise RuntimeError("Flow checkpoint environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])

    suites = []
    if args.suite in ("fresh200", "both"):
        manifest = fresh_manifest()
        suites.append((
            "fresh200", manifest,
            list(manifest["episodes"]),
            int(manifest["flow_randomness"]["root_seed"]),
            str(manifest["flow_randomness"]["semantics"]),
        ))
    if args.suite in ("historical11", "both"):
        manifest, indices = historical_manifest()
        suites.append((
            "historical11", manifest,
            [row for row in manifest["episodes"] if int(row["episode_index"]) in indices],
            int(manifest["flow_randomness"]["root_seed"]),
            str(manifest["flow_randomness"]["semantics"]),
        ))

    completed = skipped = 0
    for suite, manifest, episodes, root_seed, semantics in suites:
        if manifest["environment"] != environment or manifest["cbf"] != cbf.to_dict():
            raise RuntimeError((suite, "environment/projection mismatch"))
        manifest_path = (
            FRESH / "fresh_test_manifest.json" if suite == "fresh200"
            else WIDE / "frozen_benchmark_manifest.json"
        )
        manifest_sha = sha256(manifest_path)
        wide_runner.FLOW_ROOT_SEED = root_seed
        wide_runner.FLOW_KEY_SEMANTICS = semantics
        for episode_base in episodes:
            index = int(episode_base["episode_index"])
            if index % args.shard_count != args.shard_index:
                continue
            episode = dict(episode_base)
            episode["gphi_overlap_partition"] = episode.get("gphi_overlap_partition", "fresh_unseen")
            record_path = HERE / f"runs/{suite}/raw/h8/episode_{index:04d}.json"
            trajectory_path = HERE / f"runs/{suite}/trajectories/h8/episode_{index:04d}.npz"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            expected = {
                "suite": suite,
                "condition": "H=8",
                "episode_index": index,
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "source_manifest_sha256": manifest_sha,
            }
            if existing_valid(record_path, expected):
                skipped += 1
                continue
            row, arrays = wide_runner.rollout(
                controller="h8", episode=episode, policy=policy,
                sample_action=sample_action, model=model,
                training_reference=reference, config=config, cbf=cbf,
                feature_builder_cls=StartupAwareFeatureBuilder,
                project=project_velocity_with_retry, first_event=first_event,
            )
            atomic_npz(trajectory_path, **arrays)
            row.update({
                "schema": "gphi_strict_deadlock_coverage_closed_loop_v1",
                "suite": suite,
                "condition": "H=8",
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "source_manifest_sha256": manifest_sha,
                "trajectory_file": str(trajectory_path.relative_to(HERE)),
                "trajectory_sha256": sha256(trajectory_path),
                "record_complete": True,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_json(record_path, row)
            completed += 1
            print(json.dumps({"suite": suite, "episode": index, "outcome": row["outcome"], "steps": row["episode_steps"]}), flush=True)

    runtime = HERE / f"runtime_closed_loop_shard{args.shard_index}.json"
    write_json(runtime, {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "completed_tuples": completed,
        "skipped_valid_tuples": skipped,
        "device": args.device,
        "jax_devices": [str(value) for value in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": platform.platform(),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "frozen_sources": frozen,
        "extra_frozen_sources": extra,
        "resolved_sources": resolved,
    })
    print(json.dumps({"status": "PASS", "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
