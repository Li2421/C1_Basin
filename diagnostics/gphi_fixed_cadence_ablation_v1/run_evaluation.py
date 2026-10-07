"""Resumable fixed-cadence evaluation of the frozen startup-complete G_phi.

This is a deliberately small extension of
``diagnostics/gphi_closed_loop_pilot_v1/run_evaluation.py``.  At every physical
step it computes FlowBC, the first safety projection, and the startup-aware
feature (so history is never downsampled).  G_phi is queried and its correction
is applied for exactly one physical step iff ``step % cadence_h == 0``.  No
previous correction is held between cadence events.

There is no gate, eta/basin search, or online oracle in this evaluator.
"""

from __future__ import annotations

import argparse
import inspect
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
HERE = ROOT / "diagnostics/gphi_fixed_cadence_ablation_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")

# Reuse the audited pilot implementation for inference, frozen-source checks,
# normalization, OOD reference, and durable JSON helpers.  Keeping the cadence
# delta in this file makes the causal change explicit and reviewable.
sys.path.insert(0, str(PILOT))
from pilot_common import (  # noqa: E402
    DATASET,
    FLOW_CHECKPOINT,
    TRAINING,
    DeterministicGphi,
    TrainingReference,
    assert_frozen_sources,
    audit_startup_training_artifacts,
    canonical_json_hash,
    contiguous_true_run_lengths,
    load_environment_config,
    sha256,
    write_json,
)


SEED_MANIFEST = PILOT / "evaluation_seed_manifest.json"
PILOT_CONFIG = PILOT / "controller_config.json"
CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
EXPECTED_SEED_MANIFEST_SHA256 = "17a6750acccfbf17dcb3cbfe46dbe5d89e6d95931cbd655e1bcb5122b397156c"
EXPECTED_PILOT_CONFIG_SHA256 = "416c48a2fb38bfb0b0cede54df127b0dbc06278466f7b964bde1cdf841e6c137"
EXPECTED_CHECKPOINT_SHA256 = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
PRIMARY_CADENCES = (4, 8, 16)
ALLOWED_CADENCES = (1, *PRIMARY_CADENCES)
MATCHED_EPISODES = 128


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _existing_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("record_complete") is True
        and all(row.get(key) == value for key, value in expected.items())
    )


def _outcome(
    info: dict[str, Any] | None, error: dict[str, str] | None,
) -> tuple[str, str]:
    if error is not None:
        return "other", error["type"]
    if info is None:
        raise RuntimeError("completed rollout has no terminal info")
    termination = str(info["termination"])
    if termination == "success":
        return "success", "success"
    if termination == "deadlock":
        return "deadlock", "deadlock"
    if termination == "timeout":
        return "timeout", "timeout"
    if termination == "collision":
        subtype = "agent_collision" if info["agent_collision"] else "wall_collision"
        return "collision", subtype
    raise RuntimeError(("unknown terminal event", termination))


def _summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {name: None for name in ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _period_metrics(
    *, scheduled: np.ndarray, effective: np.ndarray, raw_norm: np.ndarray,
    executed_norm: np.ndarray, rewrite_norm: np.ndarray, j_step: np.ndarray,
) -> dict[str, Any]:
    scheduled_count = int(scheduled.sum())
    return {
        "timesteps": int(len(scheduled)),
        "scheduled_correction_timesteps": scheduled_count,
        "scheduled_correction_fraction": float(scheduled.mean()) if len(scheduled) else 0.0,
        "effective_correction_timesteps": int(effective.sum()),
        "effective_correction_fraction": float(effective.mean()) if len(effective) else 0.0,
        "J_def": float(j_step.sum()),
        "raw_correction_norm_when_scheduled": _summary(raw_norm[scheduled]),
        "executed_correction_norm_when_scheduled": _summary(executed_norm[scheduled]),
        "projection_rewrite_norm_when_scheduled": _summary(rewrite_norm[scheduled]),
    }


def rollout(
    *, cadence_h: int, episode: dict[str, Any], policy: Any, sample_action: Any,
    model: DeterministicGphi, training_reference: TrainingReference,
    config: Any, cbf: Any, feature_builder_cls: Any, project: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one fixed-cadence episode without holding corrections."""
    if cadence_h not in ALLOWED_CADENCES:
        raise ValueError(("unsupported cadence", cadence_h, ALLOWED_CADENCES))

    import jax
    import jax.numpy as jnp
    from single_integrator.cbf import barrier_constraints
    from single_integrator.environment import GiveWayEnv, bounded_nominal

    env = GiveWayEnv(config)
    env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
    builder = feature_builder_cls()
    episode_key = jax.random.PRNGKey(np.uint32(episode["flow_seed"]))
    buffers: dict[str, list[Any]] = {name: [] for name in (
        "step", "positions_before", "positions_after", "u_flow", "u_safe", "g_hat",
        "raw_second_target", "u_exec", "cadence_scheduled",
        "raw_correction_norm", "executed_correction_norm", "projection_rewrite_norm",
        "first_linear_min", "executed_linear_min", "executed_speed_excess",
        "first_retry", "second_retry", "feature", "real_history_length",
        "candidate_since", "stuck_timer", "max_stuck_timer", "event",
        "wall_collision", "agent_collision",
    )}
    projection_failures = 0
    invalid_actions = 0
    model_query_count = 0
    error: dict[str, str] | None = None
    terminal_info: dict[str, Any] | None = None
    started = time.monotonic()
    for step in range(config.max_steps):
        try:
            observation = np.asarray(env.observation(), dtype=np.float32)
            raw = np.asarray(
                sample_action(jnp.asarray(observation), jax.random.fold_in(episode_key, step)),
                dtype=np.float64,
            )
            u_flow = bounded_nominal(raw, config.max_speed)
            snapshot = env.snapshot()
            A, lower, _ = barrier_constraints(snapshot, cbf)
            u_safe, first_status, first_retry, _ = project(
                u_flow, A, lower, config.max_speed, cbf
            )

            # This call is unconditional: the 41-step history and all feature
            # monitor state advance on every physical timestep at every cadence.
            feature, _ = builder.build(
                env, {"u_flow": u_flow, "u_safe": u_safe}, config, cbf
            )
            cadence_scheduled = step % cadence_h == 0
            if cadence_scheduled:
                # The sole cadence-controlled operation is this one-step G_phi
                # correction and its frozen second hard projection.
                g_hat = model(feature[None])[0].reshape(2, 2)
                model_query_count += 1
                raw_target = u_safe + g_hat
                u_exec, second_status, second_retry, _ = project(
                    raw_target, A, lower, config.max_speed, cbf
                )
            else:
                # Never hold the previous correction.  On every non-cadence
                # step the executed command is exactly the first projection.
                g_hat = np.zeros((2, 2), dtype=np.float64)
                raw_target = u_safe.copy()
                u_exec = u_safe.copy()
                second_status, second_retry = "not_scheduled", False

            linear_min = float(np.min(A @ u_exec.reshape(4) - lower))
            speed_excess = float(np.max(np.linalg.norm(u_exec, axis=-1) - config.max_speed))
            if (
                not np.isfinite(u_exec).all()
                or linear_min < -cbf.feasibility_tol
                or speed_excess > cbf.speed_tol
            ):
                invalid_actions += 1
                raise RuntimeError(
                    f"invalid projected action: linear_min={linear_min}, "
                    f"speed_excess={speed_excess}"
                )
            before = env.positions.copy()
            _, _, done, info = env.step(u_exec)
            terminal_info = info
            executed_correction = u_exec - u_safe
            rewrite = u_exec - raw_target
            values = {
                "step": step,
                "positions_before": before,
                "positions_after": env.positions.copy(),
                "u_flow": u_flow,
                "u_safe": u_safe,
                "g_hat": g_hat,
                "raw_second_target": raw_target,
                "u_exec": u_exec,
                "cadence_scheduled": cadence_scheduled,
                "raw_correction_norm": float(np.linalg.norm(g_hat)),
                "executed_correction_norm": float(np.linalg.norm(executed_correction)),
                "projection_rewrite_norm": float(np.linalg.norm(rewrite)),
                "first_linear_min": float(np.min(A @ u_safe.reshape(4) - lower)),
                "executed_linear_min": linear_min,
                "executed_speed_excess": speed_excess,
                "first_retry": bool(first_retry),
                "second_retry": bool(second_retry),
                "feature": feature,
                "real_history_length": min(step + 1, 41),
                "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
                "stuck_timer": env.stuck_timer,
                "max_stuck_timer": env.max_stuck_timer,
                "event": info["termination"],
                "wall_collision": info["wall_collision"],
                "agent_collision": info["agent_collision"],
            }
            for name in buffers:
                buffers[name].append(values[name])
            if done:
                break
        except Exception as exc:  # Numeric/solver errors remain experiment errors.
            projection_failures += int(
                "projection" in str(exc).lower()
                or "solver" in type(exc).__name__.lower()
            )
            error = {"type": type(exc).__name__, "message": str(exc), "step": str(step)}
            break

    arrays = {name: np.asarray(values) for name, values in buffers.items()}
    outcome, failure_type = _outcome(terminal_info, error)
    correction_norm = np.asarray(buffers["executed_correction_norm"], dtype=np.float64)
    raw_norm = np.asarray(buffers["raw_correction_norm"], dtype=np.float64)
    rewrite_norm = np.asarray(buffers["projection_rewrite_norm"], dtype=np.float64)
    scheduled = np.asarray(buffers["cadence_scheduled"], dtype=bool)
    effective = correction_norm > cbf.intervention_tol
    run_lengths = contiguous_true_run_lengths(effective)
    j_step = config.dt * correction_norm**2
    features = np.asarray(buffers["feature"], dtype=np.float64)

    # Preserve the frozen pilot OOD metric over every visited state, including
    # states at which G_phi is not queried.
    if len(features):
        normalized = model.normalize(features)
        ood_distance, nearest = training_reference.distances(normalized)
        arrays["ood_distance"] = ood_distance
        arrays["nearest_train_state_index"] = nearest
    else:
        ood_distance = np.asarray([], dtype=np.float64)
        nearest = np.asarray([], dtype=np.int64)
    top_indices = (
        np.argsort(ood_distance)[-5:][::-1]
        if len(ood_distance) else np.asarray([], dtype=int)
    )
    top_ood = [
        {
            "step": int(index),
            "distance": float(ood_distance[index]),
            "nearest_train_state_id": str(training_reference.state_ids[nearest[index]]),
            "positions": np.asarray(buffers["positions_before"][index]).tolist(),
            "feature": np.asarray(buffers["feature"][index]).tolist(),
        }
        for index in top_indices
    ]
    late_start = int(np.floor(0.75 * len(effective)))
    startup_slice = slice(0, min(41, len(scheduled)))
    warm_slice = slice(min(41, len(scheduled)), len(scheduled))
    row = {
        "schema": "gphi_fixed_cadence_episode_v1",
        "record_complete": True,
        "controller": f"h{cadence_h}",
        "cadence_h": cadence_h,
        "cadence_seconds": float(cadence_h * config.dt),
        "cadence_rule": "query/apply iff physical_step % cadence_h == 0; never hold",
        "episode_index": int(episode["episode_index"]),
        "ic_seed": int(episode["ic_seed"]),
        "flow_seed": int(episode["flow_seed"]),
        "initial_positions": episode["initial_positions"],
        "outcome": outcome,
        "failure_type": failure_type,
        "execution_error": error,
        "episode_steps": int(len(correction_norm)),
        "success": outcome == "success",
        "deadlock": outcome == "deadlock",
        "timeout": outcome == "timeout",
        "collision": outcome == "collision",
        "wall_collision": bool(terminal_info and terminal_info.get("wall_collision")),
        "agent_collision": bool(terminal_info and terminal_info.get("agent_collision")),
        "J_def": float(j_step.sum()),
        "J_def_startup_0_40": float(j_step[:41].sum()),
        "J_def_post_startup": float(j_step[41:].sum()),
        "scheduled_correction_timesteps": int(scheduled.sum()),
        "scheduled_correction_fraction": float(scheduled.mean()) if len(scheduled) else 0.0,
        "model_query_count": model_query_count,
        "raw_correction_norm": _summary(raw_norm),
        "executed_correction_norm": _summary(correction_norm),
        "projection_rewrite_norm": _summary(rewrite_norm),
        "raw_correction_norm_when_scheduled": _summary(raw_norm[scheduled]),
        "executed_correction_norm_when_scheduled": _summary(correction_norm[scheduled]),
        "projection_rewrite_norm_when_scheduled": _summary(rewrite_norm[scheduled]),
        "substantial_rewrite_fraction_when_scheduled": (
            float(np.mean(rewrite_norm[scheduled] > cbf.intervention_tol))
            if scheduled.any() else 0.0
        ),
        "corrected_timestep_fraction": float(effective.mean()) if len(effective) else 0.0,
        "corrected_timesteps": int(effective.sum()),
        "correction_run_lengths": run_lengths,
        "max_correction_run_length": max(run_lengths, default=0),
        "late_quartile_scheduled_fraction": (
            float(np.mean(scheduled[late_start:])) if len(scheduled) else 0.0
        ),
        "late_quartile_corrected_fraction": (
            float(np.mean(effective[late_start:])) if len(effective) else 0.0
        ),
        "startup_0_40": _period_metrics(
            scheduled=scheduled[startup_slice], effective=effective[startup_slice],
            raw_norm=raw_norm[startup_slice], executed_norm=correction_norm[startup_slice],
            rewrite_norm=rewrite_norm[startup_slice], j_step=j_step[startup_slice],
        ),
        "warm_41_plus": _period_metrics(
            scheduled=scheduled[warm_slice], effective=effective[warm_slice],
            raw_norm=raw_norm[warm_slice], executed_norm=correction_norm[warm_slice],
            rewrite_norm=rewrite_norm[warm_slice], j_step=j_step[warm_slice],
        ),
        "first_projection_retry_count": int(np.sum(buffers["first_retry"])),
        "second_projection_retry_count": int(np.sum(buffers["second_retry"])),
        "projection_failures": projection_failures,
        "invalid_actions": invalid_actions,
        "minimum_executed_linear_residual": (
            float(np.min(buffers["executed_linear_min"])) if len(correction_norm) else None
        ),
        "maximum_executed_speed_excess": (
            float(np.max(buffers["executed_speed_excess"])) if len(correction_norm) else None
        ),
        "ood": (
            {
                **_summary(ood_distance),
                "fraction_above_train_reference_p95": float(
                    np.mean(ood_distance > training_reference.reference_p95)
                ),
                "top_states": top_ood,
            }
            if len(ood_distance) else None
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    return row, arrays


def _assert_pilot_provenance(seed_manifest: Path, checkpoint: Path) -> dict[str, Any]:
    required = (SEED_MANIFEST, PILOT_CONFIG, CHECKPOINT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    observed = {
        "seed_manifest_sha256": sha256(SEED_MANIFEST),
        "pilot_controller_config_sha256": sha256(PILOT_CONFIG),
        "checkpoint_sha256": sha256(CHECKPOINT),
    }
    expected = {
        "seed_manifest_sha256": EXPECTED_SEED_MANIFEST_SHA256,
        "pilot_controller_config_sha256": EXPECTED_PILOT_CONFIG_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    if observed != expected:
        raise RuntimeError(("frozen pilot provenance mismatch", observed, expected))
    if seed_manifest.resolve() != SEED_MANIFEST.resolve():
        raise RuntimeError(("must use exact pilot seed manifest", seed_manifest, SEED_MANIFEST))
    if checkpoint.resolve() != CHECKPOINT.resolve():
        raise RuntimeError(("must use exact frozen checkpoint", checkpoint, CHECKPOINT))
    pilot_config = json.loads(PILOT_CONFIG.read_text())
    if pilot_config["checkpoint_audit"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("pilot config checkpoint hash disagrees with frozen checkpoint")
    return {
        "status": "PASS",
        **observed,
        "pilot_controller_config_content_sha256": pilot_config["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadence", type=int, required=True, choices=ALLOWED_CADENCES)
    parser.add_argument("--seed-manifest", type=Path, default=SEED_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--normalization", type=Path, default=TRAINING / "artifacts/normalization.json")
    parser.add_argument("--samples", type=Path, default=DATASET / "samples.npz")
    parser.add_argument("--namespace", default="production")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=MATCHED_EPISODES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    provenance = _assert_pilot_provenance(args.seed_manifest, args.checkpoint)
    frozen = assert_frozen_sources()

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv
    from single_integrator.evaluate import load_policy

    resolved = {
        "projection": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "retry": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
        "feature_builder": str(Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()),
    }
    expected_resolved = {
        "projection": str((SYSROOT / "single_integrator/cbf.py").resolve()),
        "environment": str((SYSROOT / "single_integrator/environment.py").resolve()),
        "retry": str((ROOT / "diagnostics/success_basin_multimodality/exact_projector.py").resolve()),
        "feature_builder": str((DATASET / "startup_feature_builder.py").resolve()),
    }
    if resolved != expected_resolved:
        raise RuntimeError(("frozen module path mismatch", resolved, expected_resolved))

    manifest = json.loads(args.seed_manifest.read_text())
    if manifest["training_overlap_audit"]["status"] != "PASS":
        raise RuntimeError("seed manifest overlap audit did not pass")
    if manifest.get("purpose") != "production":
        raise RuntimeError("exact pilot production seed manifest is required")
    if manifest.get("episode_count", 0) < MATCHED_EPISODES:
        raise RuntimeError("pilot seed manifest contains fewer than 128 episodes")
    if not (0 <= args.start_index < MATCHED_EPISODES):
        raise ValueError("start-index must address the frozen first-128 cohort")
    if not (args.start_index < args.stop_index <= MATCHED_EPISODES):
        raise ValueError("stop-index must be within the frozen first-128 cohort")
    if args.namespace == "production" and args.cadence == 1:
        raise ValueError("production H=1 must be reused from the pilot; H=1 is smoke-only here")
    if args.samples.resolve() != (DATASET / "samples.npz").resolve():
        raise RuntimeError("must use startup-complete samples.npz")

    episodes = list(manifest["episodes"])[args.start_index:args.stop_index]
    if args.limit is not None:
        episodes = episodes[:args.limit]
    if not episodes:
        raise ValueError("empty episode slice")

    training_audit = audit_startup_training_artifacts(
        args.checkpoint, args.normalization, require_startup_complete=True
    )
    model = DeterministicGphi(args.checkpoint, args.normalization)
    reference = TrainingReference(args.samples, model)
    environment_config = load_environment_config(args.samples.parent)
    pilot_config = json.loads(PILOT_CONFIG.read_text())
    if environment_config != pilot_config["environment"]:
        raise RuntimeError("environment differs from prior full-episode pilot")
    config = Config(**environment_config)
    cbf = CBFConfig()
    if cbf.to_dict() != pilot_config["cbf"]:
        raise RuntimeError("CBF/projection configuration differs from prior pilot")
    policy, policy_provenance = load_policy(FLOW_CHECKPOINT)
    if policy_provenance and policy_provenance.get("evaluation_environment") != environment_config:
        raise RuntimeError("Flow checkpoint and frozen environment disagree")
    sample_action = jax.jit(
        lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]
    )

    namespace_root = HERE / "runs" / args.namespace
    seed_hash = sha256(args.seed_manifest)
    config_payload = {
        "schema": "gphi_fixed_cadence_controller_config_v1",
        "namespace": args.namespace,
        "matched_episode_count": MATCHED_EPISODES,
        "primary_cadences": list(PRIMARY_CADENCES),
        "h1_policy": "reuse exact prior pilot; executable only for parity smoke",
        "dt_seconds": float(config.dt),
        "method": "At every physical step compute FlowBC, first Pi_safe, and startup-aware feature; iff t mod H == 0 query G_phi and execute second Pi_safe(u_safe+g_hat), otherwise execute u_safe exactly; never hold g_hat",
        "history_update": "StartupAwareFeatureBuilder.build called at every physical timestep",
        "forbidden_online_components": {
            "eta_search": False,
            "success_basin_search": False,
            "oracle_rollout": False,
            "learned_or_rule_gate": False,
            "held_correction": False,
        },
        "checkpoint_audit": {**model.audit(), "startup_training_artifacts": training_audit},
        "ood_reference": reference.audit(),
        "seed_manifest": str(args.seed_manifest.resolve()),
        "seed_manifest_sha256": seed_hash,
        "prior_pilot_config": str(PILOT_CONFIG.resolve()),
        "prior_pilot_provenance": provenance,
        "environment": environment_config,
        "cbf": cbf.to_dict(),
        "frozen_source_audit": frozen,
        "resolved_module_paths": resolved,
    }
    config_payload["content_sha256"] = canonical_json_hash(config_payload)
    config_path = (
        HERE / "cadence_config.json"
        if args.namespace == "production"
        else namespace_root / "cadence_config.json"
    )
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("content_sha256") != config_payload["content_sha256"]:
            raise RuntimeError(("cadence config changed within namespace", config_path))
    else:
        write_json(config_path, config_payload)

    completed = skipped = 0
    controller = f"h{args.cadence}"
    record_dir = namespace_root / "raw" / controller
    trajectory_dir = namespace_root / "trajectories" / controller
    record_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        index = int(episode["episode_index"])
        record_path = record_dir / f"episode_{index:04d}.json"
        expected_record = {
            "controller": controller,
            "cadence_h": args.cadence,
            "episode_index": index,
            "ic_seed": int(episode["ic_seed"]),
            "flow_seed": int(episode["flow_seed"]),
            "seed_manifest_sha256": seed_hash,
            "checkpoint_sha256": model.sha256,
            "cadence_config_sha256": config_payload["content_sha256"],
        }
        if _existing_valid(record_path, expected_record):
            skipped += 1
            continue
        row, arrays = rollout(
            cadence_h=args.cadence,
            episode=episode,
            policy=policy,
            sample_action=sample_action,
            model=model,
            training_reference=reference,
            config=config,
            cbf=cbf,
            feature_builder_cls=StartupAwareFeatureBuilder,
            project=project_velocity_with_retry,
        )
        trajectory_path = trajectory_dir / f"episode_{index:04d}.npz"
        _atomic_npz(trajectory_path, **arrays)
        row.update(
            seed_manifest_sha256=seed_hash,
            checkpoint_sha256=model.sha256,
            trajectory_file=str(trajectory_path.relative_to(HERE)),
            trajectory_sha256=sha256(trajectory_path),
            cadence_config_sha256=config_payload["content_sha256"],
            started_utc=started_utc,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(record_path, row)
        completed += 1
        print(json.dumps({
            "controller": controller,
            "episode": index,
            "outcome": row["outcome"],
            "steps": row["episode_steps"],
            "scheduled": row["scheduled_correction_timesteps"],
            "J_def": row["J_def"],
            "completed_this_process": completed,
            "skipped": skipped,
        }), flush=True)

    runtime_path = namespace_root / f"runtime_{os.getpid()}.json"
    write_json(runtime_path, {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "completed_tuples": completed,
        "skipped_valid_tuples": skipped,
        "cadence_h": args.cadence,
        "episode_slice": [args.start_index, args.stop_index],
        "limit": args.limit,
        "device": args.device,
        "jax_devices": [str(item) for item in jax.devices()],
        "python": sys.version,
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    print(json.dumps({
        "status": "PASS",
        "completed": completed,
        "skipped": skipped,
        "namespace": args.namespace,
        "cadence_h": args.cadence,
    }, indent=2))


if __name__ == "__main__":
    main()
