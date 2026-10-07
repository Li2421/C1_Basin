"""Execute frozen, matched ETA action branches for Stage B/C labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
FROZEN = HERE / "controller_and_projection_hashes.json"
ENVIRONMENT = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
ETA_CHECKPOINT = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
FUTURE_ROOT_SEED = 2026100201

for value in (str(SYSROOT), str(ROOT), str(PILOT), str(HERE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(HERE)]

from pilot_common import sha256  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from full_state_io import restore_full_history  # noqa: E402
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from state_machine import (  # noqa: E402
    AuthoritativeStepKernel, ControllerMemory, Mode, RecoverySystem,
    SingleSegmentRecoveryMachine,
)


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


class FrozenHead:
    """NumPy inference contract for selected 214/217->64->64->1 heads."""

    def __init__(self, path: Path, *, threshold_logit: float | None = None):
        with np.load(path, allow_pickle=False) as data:
            self.mean = np.asarray(data["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(data["normalization_scale"], dtype=np.float64)
            self.weights = [np.asarray(data[f"layer_{index}_weight"], dtype=np.float64) for index in range(3)]
            self.biases = [np.asarray(data[f"layer_{index}_bias"], dtype=np.float64) for index in range(3)]
            if "threshold_logit" not in data.files:
                raise RuntimeError((path, "selected head must freeze threshold_logit"))
            checkpoint_threshold = float(data["threshold_logit"])
            self.threshold = (
                checkpoint_threshold if threshold_logit is None else float(threshold_logit)
            )
        dimension = self.mean.size
        expected = [(dimension, 64), (64, 64), (64, 1)]
        if dimension not in (214, 217) or [item.shape for item in self.weights] != expected:
            raise RuntimeError((path, "head architecture mismatch"))
        if self.scale.shape != self.mean.shape or np.any(self.scale <= 0):
            raise RuntimeError((path, "normalization mismatch"))
        if not np.isfinite(self.threshold):
            raise RuntimeError((path, "nonfinite decision threshold"))
        if not all(np.isfinite(item).all() for item in (self.mean, self.scale, *self.weights, *self.biases)):
            raise RuntimeError((path, "nonfinite head checkpoint"))

    @staticmethod
    def silu(value: np.ndarray) -> np.ndarray:
        return value / (1.0 + np.exp(-value))

    def logit(self, feature: np.ndarray) -> float:
        feature = np.asarray(feature, dtype=np.float64)
        if feature.shape != self.mean.shape or not np.isfinite(feature).all():
            raise ValueError(("invalid head feature", feature.shape, self.mean.shape))
        value = (feature - self.mean) / self.scale
        value = self.silu(value @ self.weights[0] + self.biases[0])
        value = self.silu(value @ self.weights[1] + self.biases[1])
        return float((value @ self.weights[2] + self.biases[2]).item())

    def __call__(self, feature: np.ndarray) -> bool:
        return self.logit(feature) >= self.threshold


class ConstantHead:
    def __init__(self, value: bool):
        self.value = bool(value)

    def __call__(self, feature: np.ndarray) -> bool:
        return self.value


class FirstDecisionOverride:
    """Use the paired branch action now, then the frozen incumbent head."""

    def __init__(self, action: int, downstream: Callable[[np.ndarray], bool]):
        if action not in (0, 1):
            raise ValueError(action)
        self.action = bool(action)
        self.downstream = downstream
        self.calls = 0

    def __call__(self, feature: np.ndarray) -> bool:
        self.calls += 1
        if self.calls == 1:
            return self.action
        return bool(self.downstream(feature))


def load_policy_head(record: dict, *, expected_dimension: int) -> Callable[[np.ndarray], bool]:
    kind = record["kind"]
    if kind == "NEVER_ENTER" or kind == "NEVER_EXIT" or kind == "NOT_APPLICABLE":
        return ConstantHead(False)
    if kind != "LEARNED_HEAD":
        raise RuntimeError(("unknown policy hook", kind))
    path = Path(record["checkpoint"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError((path, "head checkpoint hash mismatch"))
    head = FrozenHead(path)
    if head.mean.size != expected_dimension:
        raise RuntimeError((path, "head input dimension mismatch", head.mean.size, expected_dimension))
    return head


def outcome(env: Any, error: dict | None) -> str:
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


def verify_frozen_assets() -> dict:
    frozen = json.loads(FROZEN.read_text())
    for name, row in frozen.items():
        if not isinstance(row, dict) or "path" not in row:
            continue
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise RuntimeError((name, "frozen asset mismatch"))
    if sha256(ETA_CHECKPOINT) != frozen["structured_eta_recovery"]["sha256"]:
        raise RuntimeError("ETA checkpoint mismatch")
    return frozen


def make_key_schedule(current_key: np.ndarray, future_episode_key: Any, decision_step: int):
    import jax

    current = np.asarray(current_key, dtype=np.uint32)
    if current.shape != (2,):
        raise ValueError(("current Flow key must be uint32[2]", current.shape))

    def key_for_step(unused_episode_key: Any, step: int):
        if int(step) == int(decision_step):
            return current
        return jax.random.fold_in(future_episode_key, int(step))

    return key_for_step


def run_task(task: dict, manifest: dict, *, policy: Any, sample_action: Callable,
             eta_model: FixedDEtaPredictor, config: Config, cbf: CBFConfig) -> dict:
    import jax

    state_path = Path(task["state_file"])
    if sha256(state_path) != task["state_sha256"]:
        raise RuntimeError((task["task_id"], "state hash mismatch"))
    env = restore_full_history(state_path, config)
    if int(env.step_count) != int(task["absolute_step"]):
        raise RuntimeError((task["task_id"], "absolute step mismatch"))
    current_key = task.get("current_flow_key_data")
    if current_key is None:
        raise RuntimeError((task["task_id"], "missing frozen current Flow key"))
    future_episode_key = jax.random.fold_in(
        jax.random.PRNGKey(np.uint32(FUTURE_ROOT_SEED)), int(task["future_rollout_id"])
    )
    kernel = AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=future_episode_key,
        sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
        project=project_velocity_with_retry,
        key_for_step=make_key_schedule(current_key, future_episode_key, int(task["absolute_step"])),
    )
    downstream = manifest["downstream_policy"]
    incumbent_entry = load_policy_head(downstream["entry"], expected_dimension=214)
    incumbent_exit = load_policy_head(downstream["exit"], expected_dimension=217)
    action = int(task["action"])
    if manifest["decision_kind"] == "entry":
        entry_head = FirstDecisionOverride(action, incumbent_entry)
        exit_head = incumbent_exit
    else:
        entry_head = incumbent_entry
        exit_head = FirstDecisionOverride(action, incumbent_exit)
    machine = SingleSegmentRecoveryMachine(
        kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
        entry_head=entry_head, exit_head=exit_head, eta_model=eta_model,
    )
    if manifest["decision_kind"] == "exit":
        machine.restore_controller_memory(ControllerMemory(
            mode=Mode.RECOVERY,
            eta_latched=np.asarray(task["eta_latched"], dtype=np.float64),
            recovery_used=True,
            entry_step=int(task["entry_step"]),
            recovery_transitions=int(task["recovery_transitions"]),
        ))
    expected_flow = np.asarray(task["current_u_flow"], dtype=np.float64)
    expected_safe = np.asarray(task["current_u_safe"], dtype=np.float64)
    expected_feature = np.asarray(task["feature"], dtype=np.float64)
    error = None
    records = []
    started = time.monotonic()
    try:
        first = machine.step()
        records.append(first)
        maxima = {
            "current_u_flow": float(np.max(np.abs(first.u_flow - expected_flow))),
            "current_u_safe": float(np.max(np.abs(first.u_safe - expected_safe))),
            "current_feature": float(np.max(np.abs(first.feature - expected_feature))),
        }
        if max(maxima.values()) > 1e-12:
            raise RuntimeError((task["task_id"], "decision input replay mismatch", maxima))
        while not env.done:
            records.append(machine.step())
    except Exception as exc:  # Failures are recorded; manifest/state mismatches still remain visible.
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    result = outcome(env, error)
    remaining_jdef = config.dt * float(sum(
        np.sum((record.u_exec - record.u_safe) ** 2) for record in records
    ))
    wall_events = sum(int(record.monitor_info.get("wall_collision", False)) for record in records)
    agent_events = sum(int(record.monitor_info.get("agent_collision", False)) for record in records)
    eta_latched_hash = (
        canonical_hash(task["eta_latched"])
        if task.get("eta_latched") is not None else "UNLATCHED_AT_DECISION"
    )
    return {
        "schema": "single_segment_paired_branch_result_v1", "record_complete": error is None,
        "pass_id": manifest["pass_id"],
        "task_id": task["task_id"], "task_index": task["task_index"],
        "decision_id": task["decision_id"], "decision_kind": manifest["decision_kind"],
        "action": action, "root_source_id": task["root_source_id"], "split": task["split"],
        "state_id": task["state_id"], "state_hash": task["state_sha256"],
        "absolute_step": int(task["absolute_step"]), "mode": (
            "SAFETY_BEFORE" if manifest["decision_kind"] == "entry" else "RECOVERY"
        ),
        "current_flow_realization_id": task["current_flow_realization_id"],
        "future_stream_id": task["future_stream_id"],
        "future_rollout_id": int(task["future_rollout_id"]),
        "future_rng_state_hash": task["future_rng_state_hash"],
        "eta_latched_hash": eta_latched_hash,
        "downstream_policy_hash": manifest["downstream_policy_hash"],
        "success": result == "success", "outcome": result,
        "deadlock": result == "deadlock", "timeout": result == "timeout",
        "collision": result == "collision", "other_failure": result == "other",
        "remaining_jdef": remaining_jdef,
        "terminal_global_step": int(env.step_count),
        "remaining_physical_steps": len(records),
        "entry_step": machine.memory.entry_step, "exit_step": machine.memory.exit_step,
        "recovery_transitions": machine.memory.recovery_transitions,
        "returned_to_safety": machine.memory.mode is Mode.SAFETY_AFTER,
        "eta_query_count": machine.eta_query_count,
        "flow_sample_count": kernel.flow_sample_count,
        "physical_transition_count": kernel.physical_transition_count,
        "first_decision_override_calls": (
            entry_head.calls if manifest["decision_kind"] == "entry" else exit_head.calls
        ),
        "first_projection_retry_count": sum(record.first_projection_retry for record in records),
        "second_projection_retry_count": sum(record.second_projection_retry for record in records),
        "agent_collision_events": agent_events, "wall_collision_events": wall_events,
        "invalid_actions": 0 if error is None or "invalid" not in error["message"].lower() else 1,
        "nan_inf_events": 0 if error is None or "finite" not in error["message"].lower() else 1,
        "projection_failures": 0 if error is None or "projection" not in error["message"].lower() else 1,
        "execution_error": error,
        "manifest_content_sha256": manifest["content_sha256"],
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(args.manifest.read_text())
    verify_semantic_hash(manifest, args.manifest)
    if manifest.get("status") != "FROZEN_READY_FOR_EXECUTION" or manifest.get("system") != "ETA":
        raise RuntimeError("branch manifest is not frozen ETA work")
    if manifest.get("pass_id") != manifest.get("stage"):
        raise RuntimeError("branch pass lineage mismatch")
    if int(manifest["matched_future_semantics"].get("future_root_seed", -1)) != FUTURE_ROOT_SEED:
        raise RuntimeError("future root seed mismatch")
    verify_frozen_assets()
    environment = json.loads(ENVIRONMENT.read_text())
    config = Config(**environment["environment"])
    cbf = CBFConfig(**environment["cbf"])
    flow_path = Path(json.loads(FROZEN.read_text())["flowbc"]["path"])
    policy, provenance = load_policy(flow_path)
    if not provenance or provenance["evaluation_environment"] != environment["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    eta_model = FixedDEtaPredictor(ETA_CHECKPOINT)
    output_dir = HERE / "runs/paired_branches" / manifest["stage"]
    assigned = [task for task in manifest["tasks"] if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    new_tasks = new_steps = 0
    for task in assigned:
        path = output_dir / f"task_{int(task['task_index']):05d}.json"
        if path.is_file():
            prior = json.loads(path.read_text())
            if (prior.get("record_complete") and prior.get("task_id") == task["task_id"]
                    and prior.get("manifest_content_sha256") == manifest["content_sha256"]):
                continue
            raise RuntimeError((path, "existing task is not reusable"))
        row = run_task(
            task, manifest, policy=policy, sample_action=sample_action,
            eta_model=eta_model, config=config, cbf=cbf,
        )
        atomic_json(path, row)
        if not row["record_complete"]:
            raise RuntimeError((task["task_id"], "branch execution failed", row["execution_error"]))
        new_tasks += 1
        new_steps += int(row["remaining_physical_steps"])
        print(json.dumps({"task": task["task_index"], "outcome": row["outcome"]}), flush=True)
    atomic_json(output_dir / f"runtime_shard{args.shard_index}.json", {
        "schema": "single_segment_branch_runtime_v1", "stage": manifest["stage"],
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_continuations": new_tasks, "new_physical_steps": new_steps,
        "device": args.device, "jax_devices": [str(item) for item in jax.devices()],
        "manifest_sha256": sha256(args.manifest),
    })


if __name__ == "__main__":
    main()
