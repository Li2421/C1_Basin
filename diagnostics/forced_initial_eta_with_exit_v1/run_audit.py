"""Matched full-episode audit of forced initial eta with the frozen learned exit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/forced_initial_eta_with_exit_v1"
SINGLE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
STARTUP = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
TRAINING = ROOT / "diagnostics/gphi_training_startup_complete_v1"
MANIFEST = HERE / "fresh_dev_manifest.json"
CONTROLLERS = HERE / "controller_hashes.json"

for value in (str(SYSROOT), str(ROOT), str(PILOT), str(WIDE), str(SINGLE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(WIDE), str(SINGLE)]

from pilot_common import DeterministicGphi, TrainingReference, canonical_json_hash, sha256  # noqa: E402

# Several frozen diagnostics contain a module named ``run_evaluation``.  Load the
# authoritative WIDE implementation by path so import-path order cannot silently
# select the closed-loop pilot runner instead.
_wide_spec = importlib.util.spec_from_file_location(
    "forced_eta_authoritative_wide_runner", WIDE / "run_evaluation.py"
)
if _wide_spec is None or _wide_spec.loader is None:
    raise RuntimeError("could not resolve authoritative WIDE runner")
wide_runner = importlib.util.module_from_spec(_wide_spec)
_wide_spec.loader.exec_module(wide_runner)
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import (  # noqa: E402
    StartupAwareFeatureBuilder, left_pad_goal_error_history,
)
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from single_integrator.outcomes import first_event  # noqa: E402
from run_full_episode_system import ThresholdedFrozenHead, classify, make_kernel, run_learned  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_semantic(payload: dict[str, Any], label: str) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if canonical_json_hash(body) != payload.get("content_sha256"):
        raise RuntimeError((label, "semantic hash mismatch"))


def load_locked() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text())
    controllers = json.loads(CONTROLLERS.read_text())
    verify_semantic(manifest, "fresh manifest")
    verify_semantic(controllers, "controller lock")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_ROLLOUT" or not manifest.get("development_only"):
        raise RuntimeError("development manifest is not frozen")
    if sha256(MANIFEST) != controllers["fresh_dev_manifest_sha256"]:
        raise RuntimeError("manifest changed after controller lock")
    for record in controllers["assets"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((record["path"], "asset hash mismatch"))
    if any(controllers["forbidden"].values()) or not controllers["no_learned_entry_in_forced_condition"]:
        raise RuntimeError("forbidden method enabled")
    return manifest, controllers


def forced_eta_exit_rollout(
    episode: dict[str, Any], *, config: Config, cbf: CBFConfig, sample_action: Any,
    eta_model: FixedDEtaPredictor, exit_head: ThresholdedFrozenHead, flow_root: int,
) -> dict[str, Any]:
    import jax

    env = GiveWayEnv(config)
    env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
    episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(flow_root)), int(episode["rollout_id"]))
    builder = StartupAwareFeatureBuilder()
    kernel = make_kernel(episode, env, config=config, cbf=cbf, sample_action=sample_action, episode_key=episode_key)
    error = None
    began = time.monotonic()
    eta = raw = clipped = None
    eta_initial = None
    corrector = None
    exit_step = None
    recovery_transitions = exit_queries = 0
    structured_jdef = total_jdef = 0.0
    first_retries = second_retries = 0
    wall_events = agent_events = 0
    feature0 = None
    feature_meta = StartupAwareFeatureBuilder.history_metadata(env)
    padded = left_pad_goal_error_history(env.distance_history)
    initial_error = np.asarray(env.distance_history[0], dtype=np.float64)
    startup_padding_exact = bool(padded.shape == (41, 2) and np.array_equal(padded, np.repeat(initial_error[None], 41, axis=0)))
    mode = "STRUCTURED"
    try:
        while not env.done:
            need_feature = mode == "STRUCTURED"
            context = kernel.prepare(need_feature=need_feature)
            first_retries += int(context.first_projection_retry)
            correction = np.zeros((2, 2), dtype=np.float64)
            if context.step == 0:
                if mode != "STRUCTURED" or context.feature is None:
                    raise AssertionError("forced structured phase did not begin at step 0")
                feature0 = np.asarray(context.feature, dtype=np.float64)
                predicted, raw_values, clipped_values = eta_model.predict(feature0[None])
                eta, raw, clipped = predicted[0].copy(), raw_values[0].copy(), clipped_values[0].copy()
                eta_initial = eta.copy()
                corrector = DiagnosticCorrector(DiagnosticPhi(*eta.tolist()))
                correction = np.asarray(corrector(context.observation, context.u_safe, config.max_speed), dtype=np.float64)
                u_exec, retry = kernel.second_projection(context, correction)
                second_retries += int(retry)
                recovery_transitions += 1
            elif mode == "STRUCTURED":
                if recovery_transitions < 1 or eta is None or corrector is None:
                    raise AssertionError("exit queried before mandatory structured transition")
                exit_queries += 1
                exit_input = np.concatenate((np.asarray(context.feature, dtype=np.float64), eta))
                if exit_head(exit_input):
                    exit_step = int(context.step)
                    mode = "SAFETY_AFTER"
                    u_exec = context.u_safe
                else:
                    if not np.array_equal(eta, eta_initial):
                        raise AssertionError("eta latch changed")
                    correction = np.asarray(corrector(context.observation, context.u_safe, config.max_speed), dtype=np.float64)
                    u_exec, retry = kernel.second_projection(context, correction)
                    second_retries += int(retry)
                    recovery_transitions += 1
            else:
                u_exec = context.u_safe
            executed = np.asarray(u_exec, dtype=np.float64) - context.u_safe
            increment = float(config.dt) * float(np.sum(executed ** 2))
            total_jdef += increment
            if mode == "STRUCTURED" or context.step == 0:
                structured_jdef += increment
            _, info = kernel.execute(u_exec)
            wall_events += int(info.get("wall_collision", False))
            agent_events += int(info.get("agent_collision", False))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    outcome = classify(env, error)
    terminal = int(env.step_count)
    if eta is None or raw is None or clipped is None or feature0 is None:
        raise RuntimeError((episode["source_id"], "step-0 eta prediction missing"))
    return {
        "schema": "forced_initial_eta_exit_episode_v1", "record_complete": error is None,
        "controller": "Forced-Eta+Exit", "outcome": outcome,
        "success": outcome == "success", "deadlock": outcome == "deadlock",
        "timeout": outcome == "timeout", "collision": outcome == "collision",
        "other_failure": outcome == "other", "terminal_step": terminal,
        "completion_time_seconds": terminal * float(config.dt) if outcome == "success" else None,
        "eta_hat": eta.tolist(), "eta_norm": float(np.linalg.norm(eta)),
        "eta_normalized_raw": raw.tolist(), "eta_normalized_clipped": clipped.tolist(),
        "eta_clipped": bool(np.any(raw < 0) or np.any(raw > 1)),
        "eta_clipped_coordinate_count": int(np.sum((raw < 0) | (raw > 1))),
        "eta_query_count": 1, "learned_entry_head_loaded": False,
        "forced_start_rule": "STRUCTURED_AT_EPISODE_STEP_0",
        "exit_step": exit_step, "exit_occurred": exit_step is not None,
        "exit_time_seconds": None if exit_step is None else exit_step * float(config.dt),
        "structured_phase_transitions": recovery_transitions,
        "structured_phase_duration_seconds": recovery_transitions * float(config.dt),
        "remaining_safety_transitions": 0 if exit_step is None else terminal - exit_step,
        "remaining_safety_duration_seconds": 0.0 if exit_step is None else (terminal - exit_step) * float(config.dt),
        "exit_query_count": exit_queries, "entry_query_count": 0,
        "jdef_structured_phase": structured_jdef, "jdef_episode": total_jdef,
        "startup_feature_dimension": int(feature0.shape[0]),
        "startup_available_history_points": feature_meta["available_history_points"],
        "startup_left_padding_points": feature_meta["left_padding_points"],
        "startup_padding_exact_41_initial_errors": startup_padding_exact,
        "flow_sample_count": int(kernel.flow_sample_count),
        "physical_transition_count": int(kernel.physical_transition_count),
        "first_projection_retry_count": first_retries,
        "second_projection_retry_count": second_retries,
        "agent_collision_events": agent_events, "wall_collision_events": wall_events,
        "invalid_actions": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf_events": int(error is not None and "finite" in error["message"].lower()),
        "projection_failures": int(error is not None and "projection" in error["message"].lower()),
        "execution_error": error, "runtime_seconds": time.monotonic() - began,
    }


def learned_entry_reference(
    episode: dict[str, Any], *, manifest: dict[str, Any], config: Config, cbf: CBFConfig,
    sample_action: Any, entry_head: ThresholdedFrozenHead, exit_head: ThresholdedFrozenHead,
    eta_model: FixedDEtaPredictor, flow_root: int,
) -> dict[str, Any]:
    import jax

    episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(flow_root)), int(episode["rollout_id"]))
    learned = run_learned(
        episode, {"system": "ETA"}, config=config, cbf=cbf, sample_action=sample_action,
        episode_key=episode_key, entry_head=entry_head, exit_head=exit_head,
        direct_model=None, eta_model=eta_model,
    )
    return {
        "schema": "frozen_learned_entry_reference_episode_v1", "record_complete": learned["record_complete"],
        "controller": "Learned-Entry+Exit reference", "outcome": learned["learned_outcome"],
        "success": learned["learned_success"], "deadlock": learned["learned_deadlock"],
        "timeout": learned["learned_timeout"], "collision": learned["learned_collision"],
        "other_failure": learned["learned_other_failure"], "terminal_step": learned["terminal_step"],
        "completion_time_seconds": learned["terminal_step"] * float(config.dt) if learned["learned_success"] else None,
        "jdef_episode": learned["jdef"], "entry_step": learned["entry_step"],
        "exit_step": learned["exit_step"], "exit_occurred": learned["exit_step"] is not None,
        "entry_query_count": learned["entry_query_count"], "exit_query_count": learned["exit_query_count"],
        "eta_query_count": learned["eta_query_count"], "execution_error": learned["execution_error"],
        "physical_transition_count": learned["physical_transition_count"],
        "agent_collision_events": learned["agent_collision"], "wall_collision_events": learned["wall_collision"],
        "invalid_actions": learned["invalid_action"], "nan_inf_events": learned["nan_inf"],
        "projection_failures": learned["projection_solver_failure"],
        "first_projection_retry_count": learned["first_projection_retry_count"],
        "second_projection_retry_count": learned["second_projection_retry_count"],
        "runtime_seconds": learned["runtime_seconds"],
    }


def normalize_wide_row(row: dict[str, Any], controller: str, dt: float) -> dict[str, Any]:
    return {
        "schema": "frozen_wide_reference_episode_v1", "record_complete": row["record_complete"],
        "controller": controller, "outcome": row["outcome"], "success": row["success"],
        "deadlock": row["deadlock"], "timeout": row["timeout"], "collision": row["collision"],
        "other_failure": row["other_failure"], "terminal_step": row["episode_steps"],
        "completion_time_seconds": row["episode_steps"] * dt if row["success"] else None,
        "jdef_episode": row["J_def"], "execution_error": row["execution_error"],
        "physical_transition_count": row["episode_steps"],
        "agent_collision_events": int(row["agent_collision"]),
        "wall_collision_events": int(row["wall_collision"]),
        "invalid_actions": row["invalid_actions"], "nan_inf_events": row["nan_inf_events"],
        "projection_failures": row["projection_failures"],
        "first_projection_retry_count": row["first_projection_retry_count"],
        "second_projection_retry_count": row["second_projection_retry_count"],
        "runtime_seconds": row["runtime_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    started = datetime.now(timezone.utc).isoformat()
    manifest, controllers = load_locked()
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    config = Config(**manifest["environment"]); cbf = CBFConfig(**manifest["cbf"])
    assets = controllers["assets"]
    policy, provenance = load_policy(Path(assets["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != manifest["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    eta_model = FixedDEtaPredictor(Path(assets["structured_eta"]["path"]))
    exit_head = ThresholdedFrozenHead(controllers["exit_head"], 217)
    entry_head = ThresholdedFrozenHead(controllers["entry_head_reference"], 214)
    direct = DeterministicGphi(Path(assets["direct_g_h8"]["path"]), TRAINING / "artifacts/normalization.json")
    training_reference = TrainingReference(STARTUP / "samples.npz", direct)
    flow_root = int(manifest["flow"]["root_seed"])
    wide_runner.FLOW_ROOT_SEED = flow_root
    wide_runner.FLOW_KEY_SEMANTICS = manifest["flow"]["semantics"]
    manifest_sha = sha256(MANIFEST); controller_sha = sha256(CONTROLLERS)
    new_records = new_steps = 0
    for episode_source in manifest["episodes"]:
        episode = dict(episode_source)
        if int(episode["episode_index"]) % args.shard_count != args.shard_index:
            continue
        episode["gphi_overlap_partition"] = "new_development_only"
        for slug in ("safety", "forced_eta_exit", "learned_entry_reference", "direct_g_h8_reference"):
            path = HERE / "runs/raw" / slug / f"episode_{int(episode['episode_index']):04d}.json"
            if path.is_file():
                prior = json.loads(path.read_text())
                if (prior.get("record_complete") and prior.get("manifest_sha256") == manifest_sha
                        and prior.get("controller_hashes_sha256") == controller_sha):
                    continue
                raise RuntimeError((path, "existing result not exactly reusable"))
            if slug in ("safety", "direct_g_h8_reference"):
                raw, _ = wide_runner.rollout(
                    controller="safety" if slug == "safety" else "h8", episode=episode,
                    policy=policy, sample_action=sample_action, model=direct,
                    training_reference=training_reference, config=config, cbf=cbf,
                    feature_builder_cls=StartupAwareFeatureBuilder,
                    project=project_velocity_with_retry, first_event=first_event,
                )
                row = normalize_wide_row(raw, "Safety" if slug == "safety" else "Direct-g H8 reference", config.dt)
            elif slug == "forced_eta_exit":
                row = forced_eta_exit_rollout(
                    episode, config=config, cbf=cbf, sample_action=sample_action,
                    eta_model=eta_model, exit_head=exit_head, flow_root=flow_root,
                )
            else:
                row = learned_entry_reference(
                    episode, manifest=manifest, config=config, cbf=cbf, sample_action=sample_action,
                    entry_head=entry_head, exit_head=exit_head, eta_model=eta_model, flow_root=flow_root,
                )
            row.update({
                "episode_index": int(episode["episode_index"]), "source_id": episode["source_id"],
                "rollout_id": int(episode["rollout_id"]), "flow_root_seed": flow_root,
                "initial_positions": episode["initial_positions"], "manifest_sha256": manifest_sha,
                "manifest_content_sha256": manifest["content_sha256"],
                "controller_hashes_sha256": controller_sha,
                "rollout_started_after_manifest_frozen": started > manifest["frozen_utc"],
            })
            atomic_json(path, row)
            if not row["record_complete"]:
                raise RuntimeError((path, row["execution_error"]))
            new_records += 1; new_steps += int(row["physical_transition_count"])
            print(json.dumps({"episode": episode["episode_index"], "controller": slug,
                              "outcome": row["outcome"]}), flush=True)
    atomic_json(HERE / "runs" / f"runtime_shard{args.shard_index}.json", {
        "schema": "forced_initial_eta_exit_runtime_shard_v1", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "shard_index": args.shard_index,
        "shard_count": args.shard_count, "new_controller_episode_records": new_records,
        "new_physical_steps": new_steps, "device": args.device,
        "jax_devices": [str(item) for item in jax.devices()],
        "manifest_sha256": manifest_sha, "controller_hashes_sha256": controller_sha,
        "python": sys.version, "platform": platform.platform(),
    })


if __name__ == "__main__":
    main()
