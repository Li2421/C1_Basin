"""Run frozen Safety and H=8 on the pre-frozen fresh WIDE manifest."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
BASE_TRAINING = ROOT / "diagnostics/gphi_training_startup_complete_v1"
PARETO = ROOT / "diagnostics/gphi_startup_warm_pareto_v1"
MANIFEST = HERE / "fresh_test_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
CHECKPOINT = PARETO / "best_balanced_checkpoint.npz"
NORMALIZATION = BASE_TRAINING / "artifacts/normalization.json"
SAMPLES = DATASET / "samples.npz"
EXPECTED_CHECKPOINT = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
EXPECTED_MANIFEST_FILE_SHA = "68eecae6d04b00b0e4db7447a8e05d2443d0c886e67d14526b2eae8952841d45"
FLOW_ROOT_SEED = 2026092502
FLOW_SEMANTICS = (
    "episode_key=fold_in(PRNGKey(2026092502), rollout_id); "
    "step_key=fold_in(episode_key, physical_step)"
)
CONDITIONS = {"safety": "Safety", "h8": "H=8"}

sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(WIDE))

import run_evaluation as wide_runner  # noqa: E402
from pilot_common import (  # noqa: E402
    DeterministicGphi, TrainingReference, assert_frozen_sources,
    audit_startup_training_artifacts, canonical_json_hash,
    load_environment_config, sha256, write_json,
)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_manifest() -> dict[str, Any]:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_FILE_SHA:
        raise RuntimeError("fresh manifest file changed after freeze")
    payload = json.loads(MANIFEST.read_text())
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    if canonical_json_hash(content) != payload["content_sha256"]:
        raise RuntimeError("fresh manifest content hash invalid")
    if payload.get("frozen_before_rollout") is not True or payload.get("raw_rollout_records_present_at_freeze") != 0:
        raise RuntimeError("manifest was not frozen before rollout")
    if payload.get("episode_count") != 200 or len(payload.get("episodes", [])) != 200:
        raise RuntimeError("fresh cohort must contain exactly 200 episodes")
    flow = payload["flow_randomness"]
    if flow["root_seed"] != FLOW_ROOT_SEED or flow["semantics"] != FLOW_SEMANTICS:
        raise RuntimeError("fresh Flow stream mismatch")
    if payload["controller_protocol"] != {
        "controllers": ["Safety", "H=8"], "cadence_h": 8,
        "correction_one_step_only": True, "held_correction": False,
        "horizon_steps": 850, "dt_seconds": 0.05, "no_tuning": True,
    }:
        raise RuntimeError("controller protocol not frozen")
    initials = []
    for index, episode in enumerate(payload["episodes"]):
        if episode["episode_index"] != index or episode["rollout_id"] != index:
            raise RuntimeError(("noncanonical episode identity", index))
        if episode["flow_root_seed"] != FLOW_ROOT_SEED:
            raise RuntimeError(("Flow root mismatch", index))
        initial = np.asarray(episode["initial_positions"], dtype=np.float32)
        if initial.shape != (2, 2) or not np.isfinite(initial).all():
            raise RuntimeError(("invalid initial", index))
        initials.append(initial)
    if len({value.tobytes() for value in initials}) != 200:
        raise RuntimeError("fresh manifest has duplicate initial state")
    overlap = json.loads(OVERLAP.read_text())
    if overlap.get("status") != "PASS" or overlap.get("exact_initial_state_match_count") != 0 or overlap.get("new_seed_collisions"):
        raise RuntimeError("overlap audit failed")
    if sha256(OVERLAP) != payload["overlap_audit_sha256"]:
        raise RuntimeError("overlap audit changed after manifest freeze")
    return payload


def existing_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except Exception:
        return False
    if row.get("record_complete") is not True or any(row.get(k) != v for k, v in expected.items()):
        return False
    trajectory = HERE / row.get("trajectory_file", "")
    return trajectory.is_file() and sha256(trajectory) == row.get("trajectory_sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("safety", "h8", "both"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest = load_manifest()
    if not (0 <= args.start_index < args.stop_index <= 200):
        raise ValueError("invalid episode slice")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("G_phi checkpoint mismatch")

    frozen_sources = assert_frozen_sources()
    extra_sources = wide_runner._audit_extra_sources()

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
        "projection_constraints": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "retry_projector": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
        "feature_builder": str(Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()),
        "outcome_classifier": str(Path(inspect.getsourcefile(first_event)).resolve()),
    }
    audit_startup_training_artifacts(CHECKPOINT, NORMALIZATION, require_startup_complete=True)
    model = DeterministicGphi(CHECKPOINT, NORMALIZATION)
    reference = TrainingReference(SAMPLES, model)
    environment = load_environment_config(DATASET)
    if environment != manifest["environment"] or environment["max_steps"] != 850 or environment["dt"] != 0.05:
        raise RuntimeError("frozen environment mismatch")
    config = Config(**environment)
    cbf = CBFConfig()
    if cbf.to_dict() != manifest["cbf"]:
        raise RuntimeError("frozen projection config mismatch")
    policy, provenance = load_policy(wide_runner.FLOW_CHECKPOINT)
    if not provenance or provenance.get("evaluation_environment") != environment:
        raise RuntimeError("Flow checkpoint environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])

    # The shared audited rollout implementation reads these two frozen stream
    # constants at execution time.  Only the test stream identity changes.
    wide_runner.FLOW_ROOT_SEED = FLOW_ROOT_SEED
    wide_runner.FLOW_KEY_SEMANTICS = FLOW_SEMANTICS

    config_payload = {
        "schema": "gphi_h8_fresh_controller_config_v1",
        "manifest_path": str(MANIFEST), "manifest_file_sha256": sha256(MANIFEST),
        "manifest_content_sha256": manifest["content_sha256"],
        "overlap_audit_path": str(OVERLAP), "overlap_audit_sha256": sha256(OVERLAP),
        "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "normalization_path": str(NORMALIZATION), "normalization_sha256": sha256(NORMALIZATION),
        "flow_checkpoint_path": str(wide_runner.FLOW_CHECKPOINT),
        "flow_checkpoint_sha256": sha256(wide_runner.FLOW_CHECKPOINT),
        "flow_root_seed": FLOW_ROOT_SEED, "flow_key_semantics": FLOW_SEMANTICS,
        "controllers": CONDITIONS, "cadence_h8": 8,
        "correction_one_step_only": True, "held_correction": False,
        "environment": environment, "cbf": cbf.to_dict(),
        "frozen_sources": frozen_sources, "extra_frozen_sources": extra_sources,
        "resolved_sources": resolved,
        "forbidden": {"retraining": False, "gate": False, "eta": False, "online_oracle": False, "cadence_search": False},
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
    }
    config_payload["content_sha256"] = canonical_json_hash(config_payload)
    config_path = HERE / "controller_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()).get("content_sha256") != config_payload["content_sha256"]:
            raise RuntimeError("controller config changed")
    else:
        write_json(config_path, config_payload)

    controllers = ("safety", "h8") if args.controller == "both" else (args.controller,)
    completed = skipped = 0
    for slug in controllers:
        condition = CONDITIONS[slug]
        for episode_base in manifest["episodes"][args.start_index:args.stop_index]:
            episode = dict(episode_base)
            episode["gphi_overlap_partition"] = "fresh_unseen"
            index = int(episode["episode_index"])
            record_path = HERE / "runs/production/raw" / slug / f"episode_{index:04d}.json"
            trajectory_path = HERE / "runs/production/trajectories" / slug / f"episode_{index:04d}.npz"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            expected = {
                "condition": condition, "episode_index": index,
                "manifest_file_sha256": sha256(MANIFEST),
                "manifest_content_sha256": manifest["content_sha256"],
                "controller_config_sha256": config_payload["content_sha256"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT,
            }
            if existing_valid(record_path, expected):
                skipped += 1
                continue
            row, arrays = wide_runner.rollout(
                controller=slug, episode=episode, policy=policy,
                sample_action=sample_action, model=model,
                training_reference=reference, config=config, cbf=cbf,
                feature_builder_cls=StartupAwareFeatureBuilder,
                project=project_velocity_with_retry, first_event=first_event,
            )
            atomic_npz(trajectory_path, **arrays)
            row.update({
                "schema": "gphi_h8_fresh_episode_v1", "condition": condition,
                "manifest_file_sha256": sha256(MANIFEST),
                "manifest_content_sha256": manifest["content_sha256"],
                "controller_config_sha256": config_payload["content_sha256"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT,
                "trajectory_file": str(trajectory_path.relative_to(HERE)),
                "trajectory_sha256": sha256(trajectory_path),
                "rollout_started_after_manifest_frozen": started_utc > manifest["frozen_utc"],
                "process_started_utc": started_utc,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_json(record_path, row)
            completed += 1
            print(json.dumps({"condition": condition, "episode": index, "outcome": row["outcome"], "steps": row["episode_steps"]}), flush=True)

    runtime_dir = HERE / "runs/production"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_json(runtime_dir / f"runtime_{os.getpid()}.json", {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "episode_slice": [args.start_index, args.stop_index], "controller_argument": args.controller,
        "completed_tuples": completed, "skipped_valid_tuples": skipped,
        "device": args.device, "jax_devices": [str(item) for item in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "python": sys.version, "platform": platform.platform(),
        "manifest_file_sha256": sha256(MANIFEST), "controller_config_sha256": config_payload["content_sha256"],
    })
    print(json.dumps({"status": "PASS", "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
