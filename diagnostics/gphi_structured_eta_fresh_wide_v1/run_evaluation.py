"""Evaluate Safety, frozen direct-g H8, and one-shot persistent structured eta."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
TRAINING = ROOT / "diagnostics/gphi_training_startup_complete_v1"
MANIFEST = HERE / "fresh_wide_manifest.json"
CONTROLLERS = HERE / "controller_manifest.json"
FLOW = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
DIRECT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
NORMALIZATION = TRAINING / "artifacts/normalization.json"
SAMPLES = DATASET / "samples.npz"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(WIDE))
from pilot_common import DeterministicGphi, TrainingReference, canonical_json_hash, sha256  # noqa: E402
import run_evaluation as wide_runner  # noqa: E402
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from single_integrator.outcomes import first_event  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_locked() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text())
    controllers = json.loads(CONTROLLERS.read_text())
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_json_hash(body) != manifest["content_sha256"]:
        raise RuntimeError("manifest semantic hash mismatch")
    controller_body = {key: value for key, value in controllers.items() if key != "content_sha256"}
    if canonical_json_hash(controller_body) != controllers["content_sha256"]:
        raise RuntimeError("controller semantic hash mismatch")
    if sha256(MANIFEST) != controllers["fresh_wide_manifest_sha256"]:
        raise RuntimeError("manifest changed after controller lock")
    if sha256(Path(controllers["flowbc"]["path"])) != controllers["flowbc"]["sha256"]:
        raise RuntimeError("Flow checkpoint mismatch")
    if sha256(DIRECT) != controllers["direct_g"]["sha256"] or sha256(ETA) != controllers["structured_eta"]["sha256"]:
        raise RuntimeError("learned checkpoint mismatch")
    if any(controllers["forbidden"].values()) or not controllers["locked_before_rollout"]:
        raise RuntimeError("forbidden method enabled")
    return manifest, controllers


def existing_valid(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if not row.get("record_complete") or any(row.get(key) != value for key, value in expected.items()):
        return False
    trajectory = HERE / row.get("trajectory_file", "")
    return trajectory.is_file() and sha256(trajectory) == row.get("trajectory_sha256")


def structured_rollout(*, episode: dict, policy, sample_action, model: FixedDEtaPredictor,
                       config: Config, cbf: CBFConfig, flow_root: int) -> tuple[dict, dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp

    env = GiveWayEnv(config)
    env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
    builder = StartupAwareFeatureBuilder()
    episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(flow_root)), int(episode["rollout_id"]))
    names = (
        "step", "flow_step_key", "positions_before", "positions_after", "u_flow", "u_safe",
        "g_hat", "raw_second_target", "u_exec", "eta_hat", "model_query_marker",
        "cadence_scheduled", "raw_correction_norm", "executed_correction_norm",
        "projection_rewrite_norm", "executed_linear_min", "executed_speed_excess",
        "first_retry", "second_retry", "event", "wall_collision", "agent_collision",
    )
    buffers = {name: [] for name in names}
    error = None
    terminal_info = None
    projection_failures = invalid_actions = nan_inf_events = 0
    eta_hat = raw_normalized = clipped_normalized = None
    corrector = None
    feature0 = None
    started = time.monotonic()
    for step in range(config.max_steps):
        try:
            observation = np.asarray(env.observation(), dtype=np.float32)
            step_key = jax.random.fold_in(episode_key, step)
            raw_action = np.asarray(sample_action(jnp.asarray(observation), step_key), dtype=np.float64)
            u_flow = bounded_nominal(raw_action, config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            u_safe, _, first_retry, _ = project_velocity_with_retry(u_flow, A, lower, config.max_speed, cbf)
            queried = False
            if step == 0:
                feature0, _ = builder.build(env, {"u_flow": u_flow, "u_safe": u_safe}, config, cbf)
                eta_values, raw_values, clipped_values = model.predict(feature0[None])
                eta_hat = eta_values[0]
                raw_normalized = raw_values[0]
                clipped_normalized = clipped_values[0]
                corrector = DiagnosticCorrector(DiagnosticPhi(*eta_hat.tolist()))
                queried = True
            if corrector is None or eta_hat is None:
                raise RuntimeError("one-shot eta was not initialized")
            g_hat = corrector(np.asarray(observation, dtype=np.float64), u_safe, config.max_speed)
            raw_target = u_safe + g_hat
            u_exec, _, second_retry, _ = project_velocity_with_retry(raw_target, A, lower, config.max_speed, cbf)
            linear_min = float(np.min(A @ u_exec.reshape(4) - lower))
            speed_excess = float(np.max(np.linalg.norm(u_exec, axis=-1) - config.max_speed))
            finite = all(np.isfinite(value).all() for value in (u_flow, u_safe, g_hat, u_exec, eta_hat))
            if not finite:
                nan_inf_events += 1
            if not finite or linear_min < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
                invalid_actions += 1
                raise RuntimeError(("invalid projected action", finite, linear_min, speed_excess))
            before = env.positions.copy()
            _, _, done, info = env.step(u_exec)
            terminal_info = info
            executed = u_exec - u_safe
            rewrite = u_exec - raw_target
            values = {
                "step": step, "flow_step_key": np.asarray(step_key, dtype=np.uint32),
                "positions_before": before, "positions_after": env.positions.copy(),
                "u_flow": u_flow, "u_safe": u_safe, "g_hat": g_hat,
                "raw_second_target": raw_target, "u_exec": u_exec, "eta_hat": eta_hat,
                "model_query_marker": queried, "cadence_scheduled": True,
                "raw_correction_norm": float(np.linalg.norm(g_hat)),
                "executed_correction_norm": float(np.linalg.norm(executed)),
                "projection_rewrite_norm": float(np.linalg.norm(rewrite)),
                "executed_linear_min": linear_min, "executed_speed_excess": speed_excess,
                "first_retry": bool(first_retry), "second_retry": bool(second_retry),
                "event": info["termination"], "wall_collision": info["wall_collision"],
                "agent_collision": info["agent_collision"],
            }
            for name in names:
                buffers[name].append(values[name])
            if done:
                break
        except Exception as exc:
            projection_failures += int("projection" in str(exc).lower() or "solver" in type(exc).__name__.lower())
            error = {"type": type(exc).__name__, "message": str(exc), "step": step}
            break
    arrays = {key: np.asarray(value) for key, value in buffers.items()}
    if feature0 is not None:
        arrays["feature0"] = np.asarray(feature0)
        arrays["eta_normalized_raw"] = np.asarray(raw_normalized)
        arrays["eta_normalized_clipped"] = np.asarray(clipped_normalized)
    outcome, failure_type = wide_runner._outcome(terminal_info, error, first_event)
    executed_norm = np.asarray(buffers["executed_correction_norm"], dtype=np.float64)
    raw_norm = np.asarray(buffers["raw_correction_norm"], dtype=np.float64)
    rewrite_norm = np.asarray(buffers["projection_rewrite_norm"], dtype=np.float64)
    jdef = config.dt * float(np.sum(executed_norm ** 2))
    row = {
        "schema": "structured_eta_final_wide_episode_v1", "record_complete": True,
        "controller": "structured_eta", "episode_index": int(episode["episode_index"]),
        "rollout_id": int(episode["rollout_id"]), "initial_positions": episode["initial_positions"],
        "flow_root_seed": flow_root,
        "flow_key_semantics": f"episode_key=fold_in(PRNGKey({flow_root}), rollout_id); step_key=fold_in(episode_key, physical_step)",
        "outcome": outcome, "failure_type": failure_type, "execution_error": error,
        "episode_steps": int(len(executed_norm)), "success": outcome == "success",
        "deadlock": outcome == "deadlock", "timeout": outcome == "timeout",
        "collision": outcome == "collision", "other_failure": outcome == "other",
        "wall_collision": bool(terminal_info and terminal_info.get("wall_collision")),
        "agent_collision": bool(terminal_info and terminal_info.get("agent_collision")),
        "J_def": jdef, "eta_hat": eta_hat.tolist() if eta_hat is not None else None,
        "eta_normalized_raw": raw_normalized.tolist() if raw_normalized is not None else None,
        "eta_clipped": bool(np.any((raw_normalized < 0) | (raw_normalized > 1))) if raw_normalized is not None else None,
        "eta_clipped_coordinate_count": int(np.sum((raw_normalized < 0) | (raw_normalized > 1))) if raw_normalized is not None else None,
        "eta_norm": float(np.linalg.norm(eta_hat)) if eta_hat is not None else None,
        "eta_near_zero_threshold": 0.05, "eta_near_zero": bool(np.linalg.norm(eta_hat) <= .05) if eta_hat is not None else None,
        "model_query_count": 1 if eta_hat is not None else 0,
        "structured_eta_frozen_for_episode": True, "dense_structured_feedback": True,
        "raw_correction_norm": wide_runner._summary(raw_norm),
        "executed_correction_norm": wide_runner._summary(executed_norm),
        "projection_rewrite_norm": wide_runner._summary(rewrite_norm),
        "projection_failures": projection_failures, "invalid_actions": invalid_actions,
        "nan_inf_events": nan_inf_events,
        "first_projection_retry_count": int(np.sum(buffers["first_retry"])),
        "second_projection_retry_count": int(np.sum(buffers["second_retry"])),
        "minimum_executed_linear_residual": float(np.min(buffers["executed_linear_min"])) if len(executed_norm) else None,
        "maximum_executed_speed_excess": float(np.max(buffers["executed_speed_excess"])) if len(executed_norm) else None,
        "runtime_seconds": time.monotonic() - started,
    }
    return row, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--stop-index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest, controllers = load_locked()
    if not 0 <= args.start_index < args.stop_index <= 200:
        raise ValueError("invalid episode slice")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)

    config = Config(**manifest["environment"])
    cbf = CBFConfig(**manifest["cbf"])
    policy, provenance = load_policy(FLOW)
    if not provenance or provenance["evaluation_environment"] != manifest["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    direct_model = DeterministicGphi(DIRECT, NORMALIZATION)
    reference = TrainingReference(SAMPLES, direct_model)
    eta_model = FixedDEtaPredictor(ETA)
    flow_root = int(manifest["flow_randomness"]["root_seed"])
    wide_runner.FLOW_ROOT_SEED = flow_root
    wide_runner.FLOW_KEY_SEMANTICS = manifest["flow_randomness"]["semantics"]
    manifest_sha = sha256(MANIFEST)
    controller_sha = sha256(CONTROLLERS)
    completed = 0
    for episode_source in manifest["episodes"][args.start_index:args.stop_index]:
        episode = dict(episode_source)
        episode["gphi_overlap_partition"] = "fresh_final_unseen"
        for slug in ("safety", "direct_g_h8", "structured_eta"):
            record_path = HERE / "runs/raw" / slug / f"episode_{episode['episode_index']:04d}.json"
            trajectory_path = HERE / "runs/trajectories" / slug / f"episode_{episode['episode_index']:04d}.npz"
            expected = {
                "condition": {"safety": "Safety", "direct_g_h8": "Direct-g H8", "structured_eta": "Structured eta"}[slug],
                "episode_index": episode["episode_index"], "fresh_wide_manifest_sha256": manifest_sha,
                "controller_manifest_sha256": controller_sha,
            }
            if existing_valid(record_path, expected):
                continue
            if slug in ("safety", "direct_g_h8"):
                row, arrays = wide_runner.rollout(
                    controller="safety" if slug == "safety" else "h8", episode=episode,
                    policy=policy, sample_action=sample_action, model=direct_model,
                    training_reference=reference, config=config, cbf=cbf,
                    feature_builder_cls=StartupAwareFeatureBuilder,
                    project=project_velocity_with_retry, first_event=first_event,
                )
            else:
                row, arrays = structured_rollout(
                    episode=episode, policy=policy, sample_action=sample_action,
                    model=eta_model, config=config, cbf=cbf, flow_root=flow_root,
                )
            atomic_npz(trajectory_path, **arrays)
            row.update({
                **expected, "source_id": episode["source_id"],
                "fresh_wide_manifest_content_sha256": manifest["content_sha256"],
                "flow_checkpoint_sha256": controllers["flowbc"]["sha256"],
                "learned_checkpoint_sha256": (
                    None if slug == "safety" else controllers["direct_g"]["sha256"] if slug == "direct_g_h8" else controllers["structured_eta"]["sha256"]
                ),
                "trajectory_file": str(trajectory_path.relative_to(HERE)),
                "trajectory_sha256": sha256(trajectory_path),
                "rollout_started_after_manifest_frozen": started_utc > manifest["frozen_utc"],
                "process_started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            atomic_json(record_path, row)
            completed += 1
            print(json.dumps({"episode": episode["episode_index"], "controller": slug, "outcome": row["outcome"], "steps": row["episode_steps"]}), flush=True)
    runtime = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "episode_slice": [args.start_index, args.stop_index], "completed_controller_episodes": completed,
        "device": args.device, "jax_devices": [str(item) for item in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
        "platform": platform.platform(), "manifest_sha256": manifest_sha, "controller_manifest_sha256": controller_sha,
    }
    atomic_json(HERE / "runs" / f"runtime_{args.start_index:03d}_{args.stop_index:03d}.json", runtime)
    print(json.dumps({"status": "PASS", **runtime}, indent=2))


if __name__ == "__main__":
    main()
