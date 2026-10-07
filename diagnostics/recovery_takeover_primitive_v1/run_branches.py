"""Run permanent recovery-controller takeovers from frozen augmented states."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_takeover_primitive_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
MANIFEST = HERE / "development_manifest.json"
TASKS = HERE / "branch_task_manifest.json"

# The authoritative Toy Give-Way tree must win over a same-named workspace
# package; StartupAwareFeatureBuilder intentionally fails closed otherwise.
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT)]
from pilot_common import DeterministicGphi, canonical_json_hash, sha256  # noqa: E402
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


class FiniteHistoryView:
    def __init__(self, env):
        self._env = env
        values = np.asarray(env.distance_history, dtype=np.float64)
        self.distance_history = values[np.isfinite(values).all(axis=1)]
        if not len(self.distance_history):
            raise RuntimeError("restored environment contains no finite history")

    def __getattr__(self, name):
        return getattr(self._env, name)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def outcome(env, error: dict | None) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(MANIFEST.read_text())
    tasks_manifest = json.loads(TASKS.read_text())
    for payload, label in ((manifest, "development"), (tasks_manifest, "tasks")):
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if canonical_json_hash(body) != payload["content_sha256"]:
            raise RuntimeError((label, "semantic hash mismatch"))
    if tasks_manifest["development_manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("task/development manifest mismatch")
    direct_path = Path(manifest["controllers"]["dense_direct_g"]["path"])
    eta_path = Path(manifest["controllers"]["persistent_structured_eta"]["path"])
    flow_path = Path(manifest["controllers"]["flowbc"]["path"])
    if sha256(direct_path) != manifest["controllers"]["dense_direct_g"]["sha256"]:
        raise RuntimeError("direct-g checkpoint changed")
    if sha256(eta_path) != manifest["controllers"]["persistent_structured_eta"]["sha256"]:
        raise RuntimeError("eta checkpoint changed")
    if sha256(flow_path) != manifest["controllers"]["flowbc"]["sha256"]:
        raise RuntimeError("FlowBC checkpoint changed")
    config = Config(**manifest["environment"])
    cbf = CBFConfig(**manifest["cbf"])
    direct_model = DeterministicGphi(direct_path)
    eta_model = FixedDEtaPredictor(eta_path)
    policy, provenance = load_policy(flow_path)
    if not provenance or provenance["evaluation_environment"] != manifest["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    flow_root = int(manifest["flow_randomness"]["root_seed"])
    task_manifest_sha = sha256(TASKS)
    all_tasks = tasks_manifest["tasks"]
    assigned = [(index, task) for index, task in enumerate(all_tasks) if index % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    counters = {"tasks": 0, "steps": 0}
    for task_index, task in assigned:
        output_path = HERE / "runs/branches" / f"task_{task_index:05d}.json"
        if output_path.is_file():
            continue
        state_path = Path(task["state_file"])
        if sha256(state_path) != task["state_sha256"]:
            raise RuntimeError((task["task_id"], "state hash mismatch"))
        env = restore_full(state_path, config)
        takeover_step = int(task["takeover_step"])
        if env.step_count != takeover_step or env.done:
            raise RuntimeError((task["task_id"], "invalid takeover state", env.step_count, env.done))
        builder = StartupAwareFeatureBuilder()
        episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(flow_root)), int(task["flow_rollout_id"]))
        eta_hat = eta_raw_normalized = None
        eta_controller = None
        direct_queries = eta_queries = 0
        jdef = 0.0
        error = None
        first_retries = second_retries = projection_failures = invalid_actions = nan_inf_events = 0
        wall_events = agent_events = 0
        correction_norms: list[float] = []
        begin = time.monotonic()
        while not env.done:
            try:
                step = int(env.step_count)
                observation = np.asarray(env.observation(), dtype=np.float32)
                step_key = jax.random.fold_in(episode_key, step)
                raw_flow = np.asarray(sample(jnp.asarray(observation), step_key), dtype=np.float64)
                u_flow = bounded_nominal(raw_flow, config.max_speed)
                A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                u_safe, _, retry1, _ = project_velocity_with_retry(u_flow, A, lower, config.max_speed, cbf)
                first_retries += int(retry1)
                if task["controller"] == "continue_safety":
                    u_exec = u_safe
                elif task["controller"] == "dense_direct_g":
                    feature, _ = builder.build(FiniteHistoryView(env), {"u_flow": u_flow, "u_safe": u_safe}, config, cbf)
                    g = direct_model(feature[None])[0].reshape(2, 2)
                    u_exec, _, retry2, _ = project_velocity_with_retry(u_safe + g, A, lower, config.max_speed, cbf)
                    second_retries += int(retry2)
                    direct_queries += 1
                elif task["controller"] == "persistent_structured_eta":
                    if eta_controller is None:
                        feature, _ = builder.build(FiniteHistoryView(env), {"u_flow": u_flow, "u_safe": u_safe}, config, cbf)
                        predicted, raw_normalized, _ = eta_model.predict(feature[None])
                        eta_hat = predicted[0]
                        eta_raw_normalized = raw_normalized[0]
                        eta_controller = DiagnosticCorrector(DiagnosticPhi(*eta_hat.tolist()))
                        eta_queries += 1
                    g = eta_controller(np.asarray(observation, dtype=np.float64), u_safe, config.max_speed)
                    u_exec, _, retry2, _ = project_velocity_with_retry(u_safe + g, A, lower, config.max_speed, cbf)
                    second_retries += int(retry2)
                else:
                    raise RuntimeError(("unknown controller", task["controller"]))
                linear_min = float(np.min(A @ u_exec.reshape(4) - lower))
                speed_excess = float(np.max(np.linalg.norm(u_exec, axis=-1) - config.max_speed))
                finite = np.isfinite(u_exec).all()
                if not finite:
                    nan_inf_events += 1
                if not finite or linear_min < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
                    invalid_actions += 1
                    raise RuntimeError(("invalid projected action", finite, linear_min, speed_excess))
                delta = u_exec - u_safe
                correction_norms.append(float(np.linalg.norm(delta)))
                jdef += config.dt * float(np.sum(delta ** 2))
                _, _, _, info = env.step(u_exec)
                wall_events += int(info["wall_collision"])
                agent_events += int(info["agent_collision"])
                counters["steps"] += 1
            except Exception as exc:
                projection_failures += int("projection" in str(exc).lower() or "solver" in type(exc).__name__.lower())
                error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
                break
        result = outcome(env, error)
        correction_array = np.asarray(correction_norms, dtype=np.float64)
        row = {
            "schema": "recovery_takeover_branch_result_v1", "record_complete": True,
            "task_index": task_index, "task_id": task["task_id"], "cohort": task["cohort"],
            "controller": task["controller"], "state_id": task["state_id"],
            "failure_episode_index": task["failure_episode_index"], "failure_type": task["failure_type"],
            "offset_steps": task["offset_steps"], "offset_seconds": task["offset_seconds"],
            "takeover_step": takeover_step, "flow_rollout_id": task["flow_rollout_id"],
            "control_episode_index": task.get("control_episode_index"),
            "outcome": result, "success": result == "success", "deadlock": result == "deadlock",
            "timeout": result == "timeout", "collision": result == "collision", "other_failure": result == "other",
            "terminal_global_step": int(env.step_count), "continuation_steps": int(env.step_count - takeover_step),
            "takeover_to_terminal_sec": (env.step_count - takeover_step) * config.dt,
            "takeover_to_success_sec": (env.step_count - takeover_step) * config.dt if result == "success" else None,
            "total_completion_time_sec": env.step_count * config.dt if result == "success" else None,
            "J_def_post_takeover": jdef, "mean_executed_correction_norm": float(correction_array.mean()) if len(correction_array) else 0.0,
            "eta_hat": eta_hat.tolist() if eta_hat is not None else None,
            "eta_raw_normalized": eta_raw_normalized.tolist() if eta_raw_normalized is not None else None,
            "eta_clipped": bool(np.any((eta_raw_normalized < 0) | (eta_raw_normalized > 1))) if eta_raw_normalized is not None else None,
            "direct_query_count": direct_queries, "eta_query_count": eta_queries,
            "persistent_eta_frozen": task["controller"] != "persistent_structured_eta" or eta_queries == 1,
            "dense_after_takeover": True, "returned_to_safety": False,
            "wall_collision_events": wall_events, "agent_collision_events": agent_events,
            "first_projection_retry_count": first_retries, "second_projection_retry_count": second_retries,
            "projection_failures": projection_failures, "invalid_actions": invalid_actions, "nan_inf_events": nan_inf_events,
            "execution_error": error, "state_sha256": task["state_sha256"],
            "development_manifest_sha256": sha256(MANIFEST), "task_manifest_sha256": task_manifest_sha,
            "flow_root_seed": flow_root, "flow_semantics": manifest["flow_randomness"]["semantics"],
            "runtime_seconds": time.monotonic() - begin,
        }
        atomic_json(output_path, row)
        counters["tasks"] += 1
        print(json.dumps({"task": task_index, "cohort": task["cohort"], "controller": task["controller"], "outcome": result}), flush=True)
    atomic_json(HERE / "runs" / f"branch_runtime_shard{args.shard_index}.json", {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        **counters, "device": args.device, "jax_devices": [str(device) for device in jax.devices()],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "task_manifest_sha256": task_manifest_sha,
    })


if __name__ == "__main__":
    main()
