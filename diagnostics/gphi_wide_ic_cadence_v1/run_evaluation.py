"""Resumable fixed-cadence evaluation on the frozen historical WIDE suite.

Controllers are Safety and G_phi H=1/4/8/16.  G_phi is never gated and a
correction is never held: on inactive physical steps ``u_exec == u_safe``
exactly.  Flow randomness reproduces the historical evaluator contract:
``fold_in(fold_in(PRNGKey(42), rollout_id), physical_step)``.
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
HERE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
BENCHMARK_MANIFEST = HERE / "frozen_benchmark_manifest.json"
CHECKPOINT = (
    ROOT
    / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
)
CONTROLLER_CADENCE: dict[str, int | None] = {
    "safety": None,
    "h1": 1,
    "h4": 4,
    "h8": 8,
    "h16": 16,
}
FLOW_ROOT_SEED = 42
MATCHED_EPISODES = 200
FLOW_KEY_SEMANTICS = (
    "episode_key=fold_in(PRNGKey(42), rollout_id); "
    "step_key=fold_in(episode_key, physical_step)"
)

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

from prepare_frozen_benchmark import (  # noqa: E402
    EXPECTED_FLOW_CHECKPOINT_SHA256,
    EXPECTED_WIDE_SHA256,
    WIDE_SUITE,
    build_payload as build_benchmark_payload,
)


EXTRA_FROZEN_SOURCES = {
    SYSROOT / "single_integrator/evaluate.py": (
        "294d233ff6f7a3137730100fbfbaf0a5e2902d31311642cb98a84515821de4ae"
    ),
    SYSROOT / "single_integrator/outcomes.py": (
        "26b41cdaafccd0eda0e00de2fef9e551be753873716c4e96b7e9008658e19e82"
    ),
    SYSROOT / "flowbc/giveway_flowbc_agent.py": (
        "8b94fb64b7448cdcfb3c7bc882d10266427339a22a155defa6d71aee449f5adc"
    ),
    Path(
        "/home/zhihan/research/01_MACFlow_Baseline_Reproduction/"
        "MACFlow_Official/utils/networks.py"
    ): "e60a4975964adc4c67a813c01d52ce17edd4ec3311b273b10697e15c3341b8a1",
    Path(
        "/home/zhihan/research/01_MACFlow_Baseline_Reproduction/"
        "MACFlow_Official/utils/flax_utils.py"
    ): "8a1e0a19d7fff0293bcbf1fb0c91dcf1d91ff87b52d6fab1324a70d090d260b3",
}


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _audit_extra_sources() -> dict[str, Any]:
    observed = {}
    for path, expected in EXTRA_FROZEN_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        observed[str(path.resolve())] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen source mismatch: {path}; expected {expected}, got {actual}"
            )
    return {"status": "PASS", "observed_sha256": observed}


def _load_frozen_benchmark(path: Path, production: bool) -> dict[str, Any]:
    if production and path.resolve() != BENCHMARK_MANIFEST.resolve():
        raise RuntimeError("production requires the exact frozen benchmark manifest")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; run prepare_frozen_benchmark.py before any rollout"
        )
    manifest = json.loads(path.read_text())
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    observed_content_hash = canonical_json_hash(content)
    if manifest.get("content_sha256") != observed_content_hash:
        raise RuntimeError("frozen benchmark manifest content hash is invalid")
    expected = build_benchmark_payload()
    if manifest.get("content_sha256") != expected["content_sha256"]:
        raise RuntimeError("frozen benchmark manifest no longer matches source assets")
    if manifest.get("purpose") != "production":
        raise RuntimeError("benchmark manifest is not production-frozen")
    if manifest.get("episode_count") != MATCHED_EPISODES:
        raise RuntimeError("benchmark must contain exactly 200 episodes")
    if len(manifest.get("episodes", [])) != MATCHED_EPISODES:
        raise RuntimeError("benchmark episode list must contain exactly 200 rows")
    flow = manifest.get("flow_randomness", {})
    if (
        flow.get("root_seed") != FLOW_ROOT_SEED
        or flow.get("semantics") != FLOW_KEY_SEMANTICS
        or flow.get("direct_per_episode_prngkey_forbidden") is not True
    ):
        raise RuntimeError(
            "production rejects direct per-episode PRNGKey semantics; expected "
            "root seed 42 folded by rollout_id and then physical step"
        )
    for index, episode in enumerate(manifest["episodes"]):
        if set(episode) != {
            "episode_index", "rollout_id", "initial_positions",
            "gphi_overlap_partition",
        }:
            raise RuntimeError(("unexpected benchmark episode fields", index))
        if episode["episode_index"] != index or episode["rollout_id"] != index:
            raise RuntimeError(("noncanonical rollout id", index, episode))
        if episode["gphi_overlap_partition"] not in {
            "train", "validation", "test", "unseen"
        }:
            raise RuntimeError(("invalid G_phi overlap partition", index))
    return manifest


def _existing_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if row.get("record_complete") is not True:
        return False
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    trajectory_raw = row.get("trajectory_file")
    trajectory_hash = row.get("trajectory_sha256")
    if not isinstance(trajectory_raw, str) or not isinstance(trajectory_hash, str):
        return False
    trajectory = HERE / trajectory_raw
    return trajectory.is_file() and sha256(trajectory) == trajectory_hash


def _outcome(
    info: dict[str, Any] | None,
    error: dict[str, Any] | None,
    first_event: Any,
) -> tuple[str, str]:
    if error is not None:
        return "other", str(error["type"])
    if info is None:
        raise RuntimeError("completed rollout has no terminal info")
    historical = str(first_event(info))
    mapping = {
        "success": ("success", "success"),
        "safe_deadlock": ("deadlock", "deadlock"),
        "other_timeout": ("timeout", "timeout"),
        "agent_collision": ("collision", "agent_collision"),
        "wall_collision": ("collision", "wall_collision"),
    }
    if historical not in mapping:
        raise RuntimeError(("unknown first-event outcome", historical))
    return mapping[historical]


def _summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            name: None for name in ("mean", "median", "std", "p95", "max")
        }
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _period_metrics(
    *,
    scheduled: np.ndarray,
    effective: np.ndarray,
    raw_norm: np.ndarray,
    executed_norm: np.ndarray,
    rewrite_norm: np.ndarray,
    j_step: np.ndarray,
) -> dict[str, Any]:
    return {
        "timesteps": int(len(scheduled)),
        "scheduled_correction_timesteps": int(scheduled.sum()),
        "scheduled_correction_fraction": (
            float(scheduled.mean()) if len(scheduled) else 0.0
        ),
        "effective_correction_timesteps": int(effective.sum()),
        "effective_correction_fraction": (
            float(effective.mean()) if len(effective) else 0.0
        ),
        "J_def": float(j_step.sum()),
        "raw_correction_norm_when_scheduled": _summary(raw_norm[scheduled]),
        "executed_correction_norm_when_scheduled": _summary(
            executed_norm[scheduled]
        ),
        "projection_rewrite_norm_when_scheduled": _summary(
            rewrite_norm[scheduled]
        ),
    }


def rollout(
    *,
    controller: str,
    episode: dict[str, Any],
    policy: Any,
    sample_action: Any,
    model: DeterministicGphi,
    training_reference: TrainingReference,
    config: Any,
    cbf: Any,
    feature_builder_cls: Any,
    project: Any,
    first_event: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one controller/episode tuple under the historical Flow key stream."""
    cadence_h = CONTROLLER_CADENCE[controller]

    import jax
    import jax.numpy as jnp
    from single_integrator.cbf import barrier_constraints
    from single_integrator.environment import GiveWayEnv, bounded_nominal

    env = GiveWayEnv(config)
    env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
    builder = feature_builder_cls()
    # Do not replace this with PRNGKey(rollout_id) or a per-episode seed.
    root_key = jax.random.PRNGKey(np.uint32(FLOW_ROOT_SEED))
    episode_key = jax.random.fold_in(root_key, int(episode["rollout_id"]))
    buffer_names = (
        "step", "flow_step_key", "positions_before", "positions_after",
        "u_flow", "u_safe", "g_hat", "raw_second_target", "u_exec",
        "cadence_scheduled", "raw_correction_norm",
        "executed_correction_norm", "projection_rewrite_norm",
        "first_linear_min", "executed_linear_min", "executed_speed_excess",
        "first_status", "second_status", "first_retry", "second_retry",
        "feature", "real_history_length", "candidate_since", "stuck_timer",
        "max_stuck_timer", "event", "wall_collision", "agent_collision",
    )
    buffers: dict[str, list[Any]] = {name: [] for name in buffer_names}
    projection_failures = 0
    invalid_actions = 0
    nan_inf_events = 0
    model_query_count = 0
    error: dict[str, Any] | None = None
    terminal_info: dict[str, Any] | None = None
    started = time.monotonic()

    for step in range(config.max_steps):
        try:
            observation = np.asarray(env.observation(), dtype=np.float32)
            step_key = jax.random.fold_in(episode_key, step)
            raw = np.asarray(
                sample_action(jnp.asarray(observation), step_key),
                dtype=np.float64,
            )
            u_flow = bounded_nominal(raw, config.max_speed)
            snapshot = env.snapshot()
            A, lower, _ = barrier_constraints(snapshot, cbf)
            u_safe, first_status, first_retry, _ = project(
                u_flow, A, lower, config.max_speed, cbf
            )

            # Unconditional: history/monitor state advances every physical step
            # for Safety and every cadence condition alike.
            feature, _ = builder.build(
                env, {"u_flow": u_flow, "u_safe": u_safe}, config, cbf
            )
            scheduled = cadence_h is not None and step % cadence_h == 0
            if scheduled:
                g_hat = model(feature[None])[0].reshape(2, 2)
                model_query_count += 1
                raw_target = u_safe + g_hat
                u_exec, second_status, second_retry, _ = project(
                    raw_target, A, lower, config.max_speed, cbf
                )
            else:
                # Exact one-step correction semantics: never hold g_hat.
                g_hat = np.zeros((2, 2), dtype=np.float64)
                raw_target = u_safe.copy()
                u_exec = u_safe.copy()
                second_status = (
                    "not_applicable_safety"
                    if controller == "safety"
                    else "not_scheduled"
                )
                second_retry = False

            linear_min = float(np.min(A @ u_exec.reshape(4) - lower))
            speed_excess = float(
                np.max(np.linalg.norm(u_exec, axis=-1) - config.max_speed)
            )
            finite = bool(
                np.isfinite(u_flow).all()
                and np.isfinite(u_safe).all()
                and np.isfinite(g_hat).all()
                and np.isfinite(u_exec).all()
            )
            if not finite:
                nan_inf_events += 1
            if (
                not finite
                or linear_min < -cbf.feasibility_tol
                or speed_excess > cbf.speed_tol
            ):
                invalid_actions += 1
                raise RuntimeError(
                    f"invalid projected action: finite={finite}, "
                    f"linear_min={linear_min}, speed_excess={speed_excess}"
                )

            before = env.positions.copy()
            _, _, done, info = env.step(u_exec)
            terminal_info = info
            executed_correction = u_exec - u_safe
            rewrite = u_exec - raw_target
            values = {
                "step": step,
                "flow_step_key": np.asarray(step_key, dtype=np.uint32),
                "positions_before": before,
                "positions_after": env.positions.copy(),
                "u_flow": u_flow,
                "u_safe": u_safe,
                "g_hat": g_hat,
                "raw_second_target": raw_target,
                "u_exec": u_exec,
                "cadence_scheduled": scheduled,
                "raw_correction_norm": float(np.linalg.norm(g_hat)),
                "executed_correction_norm": float(
                    np.linalg.norm(executed_correction)
                ),
                "projection_rewrite_norm": float(np.linalg.norm(rewrite)),
                "first_linear_min": float(
                    np.min(A @ u_safe.reshape(4) - lower)
                ),
                "executed_linear_min": linear_min,
                "executed_speed_excess": speed_excess,
                "first_status": str(first_status),
                "second_status": str(second_status),
                "first_retry": bool(first_retry),
                "second_retry": bool(second_retry),
                "feature": feature,
                "real_history_length": min(step + 1, 41),
                "candidate_since": (
                    -1 if env.candidate_since is None else env.candidate_since
                ),
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
        except Exception as exc:  # Solver/numeric errors are experiment errors.
            projection_failures += int(
                "projection" in str(exc).lower()
                or "solver" in type(exc).__name__.lower()
            )
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "step": int(step),
            }
            break

    arrays = {name: np.asarray(values) for name, values in buffers.items()}
    outcome, failure_type = _outcome(terminal_info, error, first_event)
    executed_norm = np.asarray(
        buffers["executed_correction_norm"], dtype=np.float64
    )
    raw_norm = np.asarray(buffers["raw_correction_norm"], dtype=np.float64)
    rewrite_norm = np.asarray(
        buffers["projection_rewrite_norm"], dtype=np.float64
    )
    scheduled = np.asarray(buffers["cadence_scheduled"], dtype=bool)
    effective = executed_norm > cbf.intervention_tol
    run_lengths = contiguous_true_run_lengths(effective)
    j_step = config.dt * executed_norm**2
    features = np.asarray(buffers["feature"], dtype=np.float64)

    if controller != "safety" and len(features):
        normalized = model.normalize(features)
        ood_distance, nearest = training_reference.distances(normalized)
        arrays["ood_distance"] = ood_distance
        arrays["nearest_train_state_index"] = nearest
    else:
        ood_distance = np.asarray([], dtype=np.float64)
        nearest = np.asarray([], dtype=np.int64)
    top_indices = (
        np.argsort(ood_distance)[-5:][::-1]
        if len(ood_distance)
        else np.asarray([], dtype=int)
    )
    top_ood = [
        {
            "step": int(index),
            "distance": float(ood_distance[index]),
            "nearest_train_state_id": str(
                training_reference.state_ids[nearest[index]]
            ),
            "positions": np.asarray(
                buffers["positions_before"][index]
            ).tolist(),
        }
        for index in top_indices
    ]
    startup = slice(0, min(41, len(scheduled)))
    warm = slice(min(41, len(scheduled)), len(scheduled))
    late_start = int(np.floor(0.75 * len(effective)))
    row = {
        "schema": "gphi_wide_ic_cadence_episode_v1",
        "record_complete": True,
        "controller": controller,
        "cadence_h": cadence_h,
        "cadence_seconds": (
            None if cadence_h is None else float(cadence_h * config.dt)
        ),
        "cadence_rule": (
            "Safety: no G_phi"
            if cadence_h is None
            else "query/apply iff physical_step % cadence_h == 0; never hold"
        ),
        "episode_index": int(episode["episode_index"]),
        "rollout_id": int(episode["rollout_id"]),
        "flow_base_seed": FLOW_ROOT_SEED,
        "flow_root_seed": FLOW_ROOT_SEED,
        "flow_key_semantics": FLOW_KEY_SEMANTICS,
        "initial_positions": episode["initial_positions"],
        "gphi_overlap_partition": episode["gphi_overlap_partition"],
        "outcome": outcome,
        "failure_type": failure_type,
        "execution_error": error,
        "episode_steps": int(len(executed_norm)),
        "success": outcome == "success",
        "deadlock": outcome == "deadlock",
        "timeout": outcome == "timeout",
        "collision": outcome == "collision",
        "other_failure": outcome == "other",
        "wall_collision": bool(
            terminal_info and terminal_info.get("wall_collision")
        ),
        "agent_collision": bool(
            terminal_info and terminal_info.get("agent_collision")
        ),
        "J_def": float(j_step.sum()),
        "J_def_startup_0_40": float(j_step[:41].sum()),
        "J_def_post_startup": float(j_step[41:].sum()),
        "scheduled_correction_timesteps": int(scheduled.sum()),
        "scheduled_correction_fraction": (
            float(scheduled.mean()) if len(scheduled) else 0.0
        ),
        "model_query_count": model_query_count,
        "raw_correction_norm": _summary(raw_norm),
        "executed_correction_norm": _summary(executed_norm),
        "projection_rewrite_norm": _summary(rewrite_norm),
        "raw_correction_norm_when_scheduled": _summary(raw_norm[scheduled]),
        "executed_correction_norm_when_scheduled": _summary(
            executed_norm[scheduled]
        ),
        "projection_rewrite_norm_when_scheduled": _summary(
            rewrite_norm[scheduled]
        ),
        "substantial_rewrite_fraction_when_scheduled": (
            float(np.mean(rewrite_norm[scheduled] > cbf.intervention_tol))
            if scheduled.any()
            else 0.0
        ),
        "corrected_timestep_fraction": (
            float(effective.mean()) if len(effective) else 0.0
        ),
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
            scheduled=scheduled[startup],
            effective=effective[startup],
            raw_norm=raw_norm[startup],
            executed_norm=executed_norm[startup],
            rewrite_norm=rewrite_norm[startup],
            j_step=j_step[startup],
        ),
        "warm_41_plus": _period_metrics(
            scheduled=scheduled[warm],
            effective=effective[warm],
            raw_norm=raw_norm[warm],
            executed_norm=executed_norm[warm],
            rewrite_norm=rewrite_norm[warm],
            j_step=j_step[warm],
        ),
        "first_projection_retry_count": int(
            np.sum(buffers["first_retry"])
        ),
        "second_projection_retry_count": int(
            np.sum(buffers["second_retry"])
        ),
        "projection_failures": projection_failures,
        "invalid_actions": invalid_actions,
        "nan_inf_events": nan_inf_events,
        "minimum_executed_linear_residual": (
            float(np.min(buffers["executed_linear_min"]))
            if len(executed_norm)
            else None
        ),
        "maximum_executed_speed_excess": (
            float(np.max(buffers["executed_speed_excess"]))
            if len(executed_norm)
            else None
        ),
        "ood": (
            {
                **_summary(ood_distance),
                "fraction_above_train_reference_p95": float(
                    np.mean(
                        ood_distance > training_reference.reference_p95
                    )
                ),
                "top_states": top_ood,
            }
            if len(ood_distance)
            else None
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    return row, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=(*CONTROLLER_CADENCE, "all"),
        required=True,
    )
    parser.add_argument(
        "--benchmark-manifest", type=Path, default=BENCHMARK_MANIFEST
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=TRAINING / "artifacts/normalization.json",
    )
    parser.add_argument(
        "--samples", type=Path, default=DATASET / "samples.npz"
    )
    parser.add_argument("--namespace", default="production")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=MATCHED_EPISODES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    production = args.namespace == "production"
    started_utc = datetime.now(timezone.utc).isoformat()
    frozen = assert_frozen_sources()
    extra_frozen = _audit_extra_sources()
    benchmark = _load_frozen_benchmark(args.benchmark_manifest, production)

    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen G_phi checkpoint hash mismatch")
    if sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FlowBC checkpoint hash mismatch")
    if sha256(WIDE_SUITE) != EXPECTED_WIDE_SHA256:
        raise RuntimeError("frozen WIDE suite hash mismatch")
    if production and args.samples.resolve() != (DATASET / "samples.npz").resolve():
        raise RuntimeError("production must use startup-complete samples.npz")
    if not (0 <= args.start_index < MATCHED_EPISODES):
        raise ValueError("start-index must be in the frozen 200-case cohort")
    if not (args.start_index < args.stop_index <= MATCHED_EPISODES):
        raise ValueError("stop-index must be in the frozen 200-case cohort")

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
        "projection_constraints": str(
            Path(inspect.getsourcefile(barrier_constraints)).resolve()
        ),
        "environment_and_events": str(
            Path(inspect.getsourcefile(GiveWayEnv)).resolve()
        ),
        "retry_projector": str(
            Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()
        ),
        "feature_builder": str(
            Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()
        ),
        "flow_loader": str(Path(inspect.getsourcefile(load_policy)).resolve()),
        "outcome_classifier": str(
            Path(inspect.getsourcefile(first_event)).resolve()
        ),
    }
    expected_resolved = {
        "projection_constraints": str(
            (SYSROOT / "single_integrator/cbf.py").resolve()
        ),
        "environment_and_events": str(
            (SYSROOT / "single_integrator/environment.py").resolve()
        ),
        "retry_projector": str(
            (
                ROOT
                / "diagnostics/success_basin_multimodality/exact_projector.py"
            ).resolve()
        ),
        "feature_builder": str(
            (DATASET / "startup_feature_builder.py").resolve()
        ),
        "flow_loader": str(
            (SYSROOT / "single_integrator/evaluate.py").resolve()
        ),
        "outcome_classifier": str(
            (SYSROOT / "single_integrator/outcomes.py").resolve()
        ),
    }
    if resolved != expected_resolved:
        raise RuntimeError(
            ("authoritative implementation path mismatch", resolved, expected_resolved)
        )

    training_audit = audit_startup_training_artifacts(
        CHECKPOINT,
        args.normalization,
        require_startup_complete=True,
    )
    model = DeterministicGphi(CHECKPOINT, args.normalization)
    reference = TrainingReference(args.samples, model)
    environment_config = load_environment_config(args.samples.parent)
    if environment_config != benchmark["environment"]:
        raise RuntimeError("G_phi and historical WIDE environment mismatch")
    config = Config(**environment_config)
    cbf = CBFConfig()
    if cbf.to_dict() != benchmark["cbf"]:
        raise RuntimeError("hard-projection configuration mismatch")
    policy, policy_provenance = load_policy(FLOW_CHECKPOINT)
    if (
        not policy_provenance
        or policy_provenance.get("evaluation_environment") != environment_config
    ):
        raise RuntimeError("FlowBC checkpoint environment mismatch")
    sample_action = jax.jit(
        lambda observation, key: policy.sample_actions(
            observation[None], seed=key
        )[0]
    )

    benchmark_file_hash = sha256(args.benchmark_manifest)
    namespace_root = HERE / "runs" / args.namespace
    config_payload = {
        "schema": "gphi_wide_ic_cadence_controller_config_v1",
        "namespace": args.namespace,
        "controllers": CONTROLLER_CADENCE,
        "matched_episode_count": MATCHED_EPISODES,
        "method": {
            "safety": "FlowBC -> first Pi_safe -> environment",
            "gphi": (
                "Every step: FlowBC -> first Pi_safe -> startup-aware feature; "
                "iff t mod H == 0 query G_phi and execute second "
                "Pi_safe(u_safe+g_hat), otherwise execute u_safe exactly; "
                "never hold g_hat"
            ),
        },
        "flow_key_semantics": FLOW_KEY_SEMANTICS,
        "direct_per_episode_prngkey_forbidden": True,
        "history_update": (
            "StartupAwareFeatureBuilder.build called at every physical timestep"
        ),
        "forbidden_online_components": {
            "eta_search": False,
            "success_basin_search": False,
            "oracle_rollout": False,
            "learned_or_rule_gate": False,
            "held_correction": False,
            "retraining": False,
        },
        "checkpoint_audit": {
            **model.audit(),
            "startup_training_artifacts": training_audit,
        },
        "ood_reference": reference.audit(),
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "benchmark_manifest_sha256": benchmark_file_hash,
        "benchmark_content_sha256": benchmark["content_sha256"],
        "wide_suite": str(WIDE_SUITE.resolve()),
        "wide_suite_sha256": EXPECTED_WIDE_SHA256,
        "flow_checkpoint": str(FLOW_CHECKPOINT.resolve()),
        "flow_checkpoint_sha256": EXPECTED_FLOW_CHECKPOINT_SHA256,
        "environment": environment_config,
        "cbf": cbf.to_dict(),
        "frozen_source_audit": frozen,
        "extra_frozen_source_audit": extra_frozen,
        "resolved_module_paths": resolved,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    config_payload["content_sha256"] = canonical_json_hash(config_payload)
    config_path = (
        HERE / "wide_cadence_config.json"
        if production
        else namespace_root / "wide_cadence_config.json"
    )
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("content_sha256") != config_payload["content_sha256"]:
            raise RuntimeError(("controller config changed", config_path))
    else:
        write_json(config_path, config_payload)

    episodes = benchmark["episodes"][args.start_index:args.stop_index]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        episodes = episodes[:args.limit]
    if not episodes:
        raise ValueError("empty episode slice")
    controllers = (
        tuple(CONTROLLER_CADENCE)
        if args.controller == "all"
        else (args.controller,)
    )

    completed = skipped = 0
    for controller in controllers:
        cadence_h = CONTROLLER_CADENCE[controller]
        record_dir = namespace_root / "raw" / controller
        trajectory_dir = namespace_root / "trajectories" / controller
        record_dir.mkdir(parents=True, exist_ok=True)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            index = int(episode["episode_index"])
            record_path = record_dir / f"episode_{index:04d}.json"
            expected_record = {
                "schema": "gphi_wide_ic_cadence_episode_v1",
                "controller": controller,
                "cadence_h": cadence_h,
                "episode_index": index,
                "rollout_id": int(episode["rollout_id"]),
                "flow_base_seed": FLOW_ROOT_SEED,
                "flow_root_seed": FLOW_ROOT_SEED,
                "flow_key_semantics": FLOW_KEY_SEMANTICS,
                "gphi_overlap_partition": episode["gphi_overlap_partition"],
                "benchmark_manifest_sha256": benchmark_file_hash,
                "benchmark_content_sha256": benchmark["content_sha256"],
                "checkpoint_sha256": model.sha256,
                "controller_config_sha256": config_payload["content_sha256"],
            }
            if _existing_valid(record_path, expected_record):
                skipped += 1
                continue
            row, arrays = rollout(
                controller=controller,
                episode=episode,
                policy=policy,
                sample_action=sample_action,
                model=model,
                training_reference=reference,
                config=config,
                cbf=cbf,
                feature_builder_cls=StartupAwareFeatureBuilder,
                project=project_velocity_with_retry,
                first_event=first_event,
            )
            trajectory_path = trajectory_dir / f"episode_{index:04d}.npz"
            _atomic_npz(trajectory_path, **arrays)
            row.update(
                benchmark_manifest_sha256=benchmark_file_hash,
                benchmark_content_sha256=benchmark["content_sha256"],
                checkpoint_sha256=model.sha256,
                trajectory_file=str(trajectory_path.relative_to(HERE)),
                trajectory_sha256=sha256(trajectory_path),
                controller_config_sha256=config_payload["content_sha256"],
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
        "controller_argument": args.controller,
        "episode_slice": [args.start_index, args.stop_index],
        "limit": args.limit,
        "device": args.device,
        "jax_devices": [str(item) for item in jax.devices()],
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "benchmark_manifest_sha256": benchmark_file_hash,
        "controller_config_sha256": config_payload["content_sha256"],
    })
    print(json.dumps({
        "status": "PASS",
        "completed": completed,
        "skipped": skipped,
        "namespace": args.namespace,
        "controller": args.controller,
    }, indent=2))


if __name__ == "__main__":
    main()
