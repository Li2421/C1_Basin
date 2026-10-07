"""Run one frozen single-segment system on a frozen full-episode manifest.

The executable is split-generic but test-locked by default.  Development
Safety outcomes are reused from the exact source collector.  For an explicitly
unlocked final test, Safety is materialized once per source and then reused by
all predeclared learned comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
HASHES = HERE / "controller_and_projection_hashes.json"
SOURCE_MANIFEST = HERE / "source_split_manifest.json"

for value in (str(SYSROOT), str(ROOT), str(PILOT), str(HERE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(HERE)]

from pilot_common import DeterministicGphi, sha256  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from state_machine import AuthoritativeStepKernel, RecoverySystem, SingleSegmentRecoveryMachine  # noqa: E402


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_semantic_hash(payload: dict, path: Path) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if not claimed or canonical_hash(body) != claimed:
        raise RuntimeError((path, "semantic hash mismatch"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


class ThresholdedFrozenHead:
    """Frozen MLP weights with an explicit manifest-frozen threshold."""

    def __init__(self, record: dict[str, Any], expected_dimension: int):
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError((path, "decision-head hash mismatch"))
        with np.load(path, allow_pickle=False) as archive:
            self.mean = np.asarray(archive["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(archive["normalization_scale"], dtype=np.float64)
            self.weights = [np.asarray(archive[f"layer_{i}_weight"], dtype=np.float64) for i in range(3)]
            self.biases = [np.asarray(archive[f"layer_{i}_bias"], dtype=np.float64) for i in range(3)]
        self.threshold = float(record["threshold_logit"])
        shapes = [item.shape for item in self.weights]
        if self.mean.shape != (expected_dimension,) or self.scale.shape != self.mean.shape:
            raise RuntimeError((path, "head normalization dimension mismatch"))
        if shapes != [(expected_dimension, 64), (64, 64), (64, 1)]:
            raise RuntimeError((path, "head weight dimension mismatch", shapes))
        if np.any(self.scale <= 0) or not all(
            np.isfinite(item).all() for item in (self.mean, self.scale, *self.weights, *self.biases)
        ) or not np.isfinite(self.threshold):
            raise RuntimeError((path, "invalid frozen head values"))

    @staticmethod
    def _silu(value: np.ndarray) -> np.ndarray:
        return value / (1.0 + np.exp(-value))

    def logit(self, feature: np.ndarray) -> float:
        value = np.asarray(feature, dtype=np.float64)
        if value.shape != self.mean.shape or not np.isfinite(value).all():
            raise ValueError(("invalid head feature", value.shape))
        value = (value - self.mean) / self.scale
        value = self._silu(value @ self.weights[0] + self.biases[0])
        value = self._silu(value @ self.weights[1] + self.biases[1])
        return float((value @ self.weights[2] + self.biases[2]).item())

    def __call__(self, feature: np.ndarray) -> bool:
        return self.logit(feature) >= self.threshold


def classify(env: GiveWayEnv, error: dict[str, Any] | None) -> str:
    if error is not None:
        return "other"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "deadlock"
    return "timeout"


def load_and_verify_assets(manifest: dict) -> tuple[dict, dict, Config, CBFConfig]:
    if manifest.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError("full-loop manifest is not frozen")
    hashes = json.loads(HASHES.read_text())
    if sha256(HASHES) != manifest["controller_hash_manifest_sha256"] or hashes.get("status") != "PASS":
        raise RuntimeError("controller/projection hash manifest mismatch")
    for name, record in hashes.items():
        if isinstance(record, dict) and "path" in record and sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((name, "frozen asset mismatch"))
    expected_implementations = {
        "state_machine": HERE / "state_machine.py",
        "full_loop_runner": HERE / "run_full_episode_system.py",
    }
    if set(manifest.get("implementation_hashes", {})) != set(expected_implementations):
        raise RuntimeError("full-loop implementation hashes are incomplete")
    for name, expected_path in expected_implementations.items():
        record = manifest["implementation_hashes"][name]
        if Path(record["path"]).resolve() != expected_path.resolve() or sha256(expected_path) != record["sha256"]:
            raise RuntimeError((name, "full-loop implementation changed after manifest freeze"))
    source = json.loads(SOURCE_MANIFEST.read_text())
    verify_semantic_hash(source, SOURCE_MANIFEST)
    if sha256(SOURCE_MANIFEST) != manifest["source_manifest_sha256"]:
        raise RuntimeError("source manifest changed after full-loop freeze")
    if manifest.get("split") == "final_test":
        gate_record = manifest.get("final_test_gate")
        if not isinstance(gate_record, dict):
            raise RuntimeError("final-test validation/calibration gate is missing")
        gate_path = Path(gate_record["path"])
        if sha256(gate_path) != gate_record["sha256"]:
            raise RuntimeError("final-test gate changed after manifest freeze")
        gate = json.loads(gate_path.read_text())
        verify_semantic_hash(gate, gate_path)
        required = {
            "schema": "single_segment_final_test_gate_v1",
            "status": "FROZEN_APPROVED_FOR_FINAL_TEST",
            "selected_policy_hash": manifest["policy_hash"],
            "validation_complete": True,
            "calibration_complete": True,
            "test_data_used": False,
            "final_test_unopened": True,
        }
        for key, expected in required.items():
            if gate.get(key) != expected:
                raise RuntimeError((gate_path, "final-test gate mismatch", key, gate.get(key), expected))
    reference = json.loads(REFERENCE.read_text())
    config = Config(**reference["environment"])
    cbf = CBFConfig(**reference["cbf"])
    if config.max_steps != 850 or config.dt != 0.05:
        raise RuntimeError("official horizon/dt mismatch")
    return hashes, source, config, cbf


def make_kernel(task: dict, env: GiveWayEnv, *, config: Config, cbf: CBFConfig,
                sample_action: Any, episode_key: Any) -> AuthoritativeStepKernel:
    return AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=episode_key,
        sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
        project=project_velocity_with_retry,
    )


def run_safety(task: dict, *, config: Config, cbf: CBFConfig,
               sample_action: Any, episode_key: Any) -> dict[str, Any]:
    env = GiveWayEnv(config)
    env.reset(np.asarray(task["initial_positions"], dtype=np.float64))
    kernel = make_kernel(task, env, config=config, cbf=cbf, sample_action=sample_action, episode_key=episode_key)
    error = None
    wall_events = agent_events = first_retry = 0
    began = time.monotonic()
    try:
        while not env.done:
            context = kernel.prepare(need_feature=False)
            first_retry += int(context.first_projection_retry)
            _, info = kernel.execute(context.u_safe)
            wall_events += int(info.get("wall_collision", False))
            agent_events += int(info.get("agent_collision", False))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    result = classify(env, error)
    return {
        "schema": "single_segment_safety_full_loop_cache_v1",
        "record_complete": error is None,
        "source_id": task["source_id"], "root_source_id": task["root_source_id"],
        "split": task["split"], "rollout_id": int(task["rollout_id"]),
        "flow_root_seed": int(task["flow_root_seed"]),
        "initial_positions": task["initial_positions"],
        "outcome": result, "success": result == "success", "deadlock": result == "deadlock",
        "timeout": result == "timeout", "collision": result == "collision",
        "terminal_step": int(env.step_count),
        "agent_collision_events": agent_events, "wall_collision_events": wall_events,
        "projection_failures": int(error is not None and "projection" in error["message"].lower()),
        "invalid_actions": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf_events": int(error is not None and "finite" in error["message"].lower()),
        "first_projection_retry_count": first_retry,
        "flow_sample_count": kernel.flow_sample_count,
        "physical_transition_count": kernel.physical_transition_count,
        "execution_error": error, "runtime_seconds": time.monotonic() - began,
    }


def _validate_safety_reference(row: dict, task: dict) -> None:
    if not row.get("record_complete") or row.get("execution_error") is not None:
        raise RuntimeError((task["source_id"], "Safety reference is incomplete"))
    exact = {
        "source_id": task["source_id"], "split": task["split"],
        "rollout_id": int(task["rollout_id"]), "flow_root_seed": int(task["flow_root_seed"]),
    }
    for key, expected in exact.items():
        if row.get(key) != expected:
            raise RuntimeError((task["source_id"], "Safety reference mismatch", key, row.get(key), expected))
    if not np.array_equal(
        np.asarray(row["initial_positions"], dtype=np.float64),
        np.asarray(task["initial_positions"], dtype=np.float64),
    ):
        raise RuntimeError((task["source_id"], "Safety initial state mismatch"))


def safety_reference(task: dict, manifest: dict, *, config: Config, cbf: CBFConfig,
                     sample_action: Any, episode_key: Any) -> tuple[dict, str, bool]:
    split, source_id = task["split"], task["source_id"]
    if split != "final_test":
        path = HERE / "runs/safety_raw" / split / f"{source_id}.json"
        if not path.is_file():
            raise RuntimeError((source_id, "frozen development Safety baseline missing"))
        row = json.loads(path.read_text())
        if row.get("source_manifest_sha256") != manifest["source_manifest_sha256"]:
            raise RuntimeError((source_id, "Safety/source manifest hash mismatch"))
        _validate_safety_reference(row, task)
        return row, "FROZEN_SOURCE_COLLECTION_REUSE", False
    if not manifest.get("final_test_explicitly_unlocked"):
        raise RuntimeError("final Safety evaluation is locked")
    path = HERE / "runs/full_loop_safety/final_test" / f"{source_id}.json"
    if path.is_file():
        row = json.loads(path.read_text())
        if row.get("source_manifest_sha256") != manifest["source_manifest_sha256"]:
            raise RuntimeError((source_id, "cached final Safety/source manifest mismatch"))
        _validate_safety_reference(row, task)
        return row, "FROZEN_FINAL_SAFETY_CACHE_REUSE", False
    # Never permit two candidate jobs to duplicate the pristine final Safety
    # rollout.  The lock is deliberately fail-closed rather than waiting and
    # hiding a scheduler/configuration error.  A later retry reuses the cache.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError((source_id, "concurrent final Safety materialization refused", str(lock))) from exc
    try:
        if path.is_file():
            row = json.loads(path.read_text())
            if row.get("source_manifest_sha256") != manifest["source_manifest_sha256"]:
                raise RuntimeError((source_id, "raced final Safety/source manifest mismatch"))
            _validate_safety_reference(row, task)
            return row, "FROZEN_FINAL_SAFETY_CACHE_REUSE", False
        row = run_safety(task, config=config, cbf=cbf, sample_action=sample_action, episode_key=episode_key)
        row["source_manifest_sha256"] = manifest["source_manifest_sha256"]
        row["final_test_manifest_sha256"] = manifest["final_test_manifest_sha256"]
        atomic_json(path, row)
        if not row["record_complete"]:
            raise RuntimeError((source_id, "final Safety baseline failed", row["execution_error"]))
        return row, "NEW_FINAL_SAFETY_CACHE", True
    finally:
        lock.unlink(missing_ok=True)


def run_learned(task: dict, manifest: dict, *, config: Config, cbf: CBFConfig,
                sample_action: Any, episode_key: Any, entry_head: ThresholdedFrozenHead,
                exit_head: ThresholdedFrozenHead, direct_model: Any | None,
                eta_model: Any | None) -> dict[str, Any]:
    env = GiveWayEnv(config)
    env.reset(np.asarray(task["initial_positions"], dtype=np.float64))
    kernel = make_kernel(task, env, config=config, cbf=cbf, sample_action=sample_action, episode_key=episode_key)
    system = RecoverySystem(manifest["system"])
    machine = SingleSegmentRecoveryMachine(
        kernel=kernel, system=system, entry_head=entry_head, exit_head=exit_head,
        direct_model=direct_model, eta_model=eta_model,
    )
    records = []
    error = None
    began = time.monotonic()
    try:
        while not env.done:
            records.append(machine.step())
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    result = classify(env, error)
    dt = float(config.dt)
    jdef = dt * float(sum(np.sum((row.u_exec - row.u_safe) ** 2) for row in records))
    wall_events = sum(int(row.monitor_info.get("wall_collision", False)) for row in records)
    agent_events = sum(int(row.monitor_info.get("agent_collision", False)) for row in records)
    entered = machine.memory.entry_step is not None
    exited = machine.memory.exit_step is not None
    return {
        "record_complete": error is None,
        "learned_outcome": result, "learned_success": result == "success",
        "learned_deadlock": result == "deadlock", "learned_timeout": result == "timeout",
        "learned_collision": result == "collision", "learned_other_failure": result == "other",
        "terminal_step": int(env.step_count), "jdef": jdef,
        "entry_step": machine.memory.entry_step, "exit_step": machine.memory.exit_step,
        "recovery_transition_count": int(machine.memory.recovery_transitions),
        "recovery_segment_count": int(entered), "returned_to_safety": bool(exited),
        "entry_query_count": int(machine.entry_query_count),
        "exit_query_count": int(machine.exit_query_count),
        "direct_query_count": int(machine.direct_query_count),
        "eta_query_count": int(machine.eta_query_count),
        "flow_sample_count": int(kernel.flow_sample_count),
        "physical_transition_count": int(kernel.physical_transition_count),
        "agent_collision": agent_events, "wall_collision": wall_events,
        "invalid_action": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf": int(error is not None and "finite" in error["message"].lower()),
        "projection_solver_failure": int(error is not None and "projection" in error["message"].lower()),
        "first_projection_retry_count": sum(int(row.first_projection_retry) for row in records),
        "second_projection_retry_count": sum(int(row.second_projection_retry) for row in records),
        "execution_error": error, "runtime_seconds": time.monotonic() - began,
    }


def combine_result(task: dict, manifest: dict, learned: dict, safety: dict,
                   safety_provenance: str) -> dict[str, Any]:
    row = {
        "schema": "single_segment_full_episode_outcome_v1",
        "policy_hash": manifest["policy_hash"], "system": manifest["system"],
        "manifest_content_sha256": manifest["content_sha256"],
        "root_source_id": task["root_source_id"], "source_id": task["source_id"],
        "split": task["split"], "episode_index": int(task["episode_index"]),
        "rollout_id": int(task["rollout_id"]), "flow_root_seed": int(task["flow_root_seed"]),
        "safety_provenance": safety_provenance,
        "safety_outcome": safety["outcome"], "safety_success": bool(safety["success"]),
        "safety_terminal_step": int(safety["terminal_step"]),
        **learned,
        "entry_threshold_probability": manifest["policy"]["entry_head"]["threshold_probability"],
        "exit_threshold_probability": manifest["policy"]["exit_head"]["threshold_probability"],
    }
    row["record_complete"] = bool(
        learned["record_complete"] and safety.get("record_complete")
        and learned.get("execution_error") is None and safety.get("execution_error") is None
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(args.manifest.read_text())
    verify_semantic_hash(manifest, args.manifest)
    if manifest.get("split") == "final_test" and (
        not args.allow_final_test or not manifest.get("final_test_explicitly_unlocked")
    ):
        raise RuntimeError("final test is doubly locked by manifest and command line")
    hashes, _, config, cbf = load_and_verify_assets(manifest)
    policy, provenance = load_policy(Path(hashes["flowbc"]["path"]))
    reference = json.loads(REFERENCE.read_text())
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC evaluation environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    system = manifest["system"]
    entry_head = ThresholdedFrozenHead(manifest["policy"]["entry_head"], 214)
    exit_head = ThresholdedFrozenHead(manifest["policy"]["exit_head"], 214 if system == "G" else 217)
    direct_model = DeterministicGphi(Path(hashes["direct_g_recovery"]["path"])) if system == "G" else None
    eta_model = FixedDEtaPredictor(Path(hashes["structured_eta_recovery"]["path"])) if system == "ETA" else None
    output_dir = HERE / "runs/full_loop" / manifest["policy_hash"] / manifest["split"]
    assigned = [task for task in manifest["tasks"] if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    new_learned = new_safety = new_steps = 0
    for task in assigned:
        output = output_dir / f"{task['source_id']}.json"
        if output.is_file():
            prior = json.loads(output.read_text())
            if (prior.get("record_complete") and prior.get("manifest_content_sha256") == manifest["content_sha256"]
                    and prior.get("execution_error") is None):
                continue
            raise RuntimeError((output, "existing full-loop result is not reusable"))
        episode_key = jax.random.fold_in(
            jax.random.PRNGKey(np.uint32(task["flow_root_seed"])), int(task["rollout_id"])
        )
        safety, safety_provenance, safety_created = safety_reference(
            task, manifest, config=config, cbf=cbf, sample_action=sample_action, episode_key=episode_key,
        )
        learned = run_learned(
            task, manifest, config=config, cbf=cbf, sample_action=sample_action,
            episode_key=episode_key, entry_head=entry_head, exit_head=exit_head,
            direct_model=direct_model, eta_model=eta_model,
        )
        row = combine_result(task, manifest, learned, safety, safety_provenance)
        atomic_json(output, row)
        if not row["record_complete"]:
            raise RuntimeError((task["source_id"], "full-loop execution failed", row["execution_error"]))
        new_learned += 1
        new_safety += int(safety_created)
        new_steps += int(learned["physical_transition_count"])
        if safety_created:
            new_steps += int(safety["physical_transition_count"])
        print(json.dumps({"source": task["source_id"], "outcome": row["learned_outcome"]}), flush=True)
    atomic_json(output_dir / f"runtime_shard{args.shard_index}.json", {
        "schema": "single_segment_full_loop_runtime_v1",
        "policy_hash": manifest["policy_hash"], "split": manifest["split"],
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_learned_continuations": new_learned, "new_safety_continuations": new_safety,
        "new_physical_steps": new_steps, "device": args.device,
        "jax_devices": [str(item) for item in jax.devices()],
        "manifest_sha256": sha256(args.manifest),
    })


if __name__ == "__main__":
    main()
