"""Run original frozen G_phi under H8-triggered bursts on 17 strict-deadlock states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
NORMALIZATION = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts/normalization.json"
EXPECTED_CHECKPOINT = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
sys.path.insert(0, str(PILOT))

from pilot_common import DeterministicGphi, assert_frozen_sources, sha256  # noqa: E402

assert_frozen_sources()

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.gphi_training_dataset_v1.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


FLOW_CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_FLOW = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
CONDITIONS = (("L1", 1), ("L2", 2), ("L4", 4), ("L6", 6), ("L8", 8), ("H1", 0))
ROBUST_SEEDS = tuple(range(95310001, 95310065))


class FiniteHistoryView:
    """Expose the faithfully saved finite recent history without NaN placeholders."""

    def __init__(self, env) -> None:
        self._env = env
        history = np.asarray(env.distance_history, dtype=np.float64)
        finite = history[np.isfinite(history).all(axis=1)]
        if not len(finite):
            raise RuntimeError("saved state contains no finite distance history")
        self.distance_history = finite

    def __getattr__(self, name):
        return getattr(self._env, name)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def outcome(env, error: dict | None) -> str:
    if error is not None:
        return "execution_error"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "deadlock"
    return "timeout"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def concat_logs(logs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = logs[0].keys() if logs else []
    return {
        key: np.concatenate([entry[key] for entry in logs], axis=0)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    integrity = json.loads((HERE / "integrity_audit.json").read_text())
    if integrity["status"] != "PREPARED":
        raise RuntimeError("preparation gate not passed")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT or sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW:
        raise RuntimeError("checkpoint hash mismatch")
    source = json.loads((HERE / "source_manifest.json").read_text())
    capacity_manifest = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity_manifest["environment"])
    cbf = CBFConfig(**capacity_manifest["cbf"])
    model = DeterministicGphi(CHECKPOINT, NORMALIZATION)
    policy, provenance = load_policy(FLOW_CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != capacity_manifest["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    eta_by_state = {
        row["state_id"]: tuple(float(value) for value in row["eta"])
        for row in source["etas_for_diagnostic_only"]
    }
    all_states = source["states"]
    states = [row for index, row in enumerate(all_states) if index % args.shard_count == args.shard_index]
    started = time.monotonic()
    new_rollouts = 0
    physical_steps = 0
    active_log_rows = 0

    def run_tasks(state: dict, tasks: list[dict]) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
        nonlocal new_rollouts, physical_steps, active_log_rows
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        builders = [StartupAwareFeatureBuilder() for _ in tasks]
        teachers = [DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]])) for _ in tasks]
        errors: list[dict | None] = [None] * len(tasks)
        burst_end = np.full(len(tasks), int(state["query_step"]), dtype=np.int64)
        trigger_step = np.full(len(tasks), -1, dtype=np.int64)
        burst_count = np.zeros(len(tasks), dtype=np.int64)
        active_count = np.zeros(len(tasks), dtype=np.int64)
        jdef = np.zeros(len(tasks), dtype=np.float64)
        raw_sum = np.zeros(len(tasks), dtype=np.float64)
        executed_sum = np.zeros(len(tasks), dtype=np.float64)
        rewrite_sum = np.zeros(len(tasks), dtype=np.float64)
        first_retries = np.zeros(len(tasks), dtype=np.int64)
        second_retries = np.zeros(len(tasks), dtype=np.int64)
        teacher_retries = np.zeros(len(tasks), dtype=np.int64)
        active_logs: list[dict[str, list]] = [{key: [] for key in (
            "global_step", "burst_ordinal", "burst_step", "gphi_raw", "gphi_executed",
            "eta_raw", "eta_executed", "raw_l2", "executed_l2", "cosine",
            "norm_ratio", "gphi_rewrite", "eta_rewrite", "positions", "goal_errors",
            "inter_agent_distance", "candidate_since", "stuck_timer", "max_stuck_timer",
        )} for _ in tasks]
        traces: list[dict[str, list]] = []
        for task in tasks:
            traces.append({key: [] for key in (
                "global_step", "positions_before", "positions_after", "goal_errors_before",
                "inter_agent_distance_before", "u_flow", "u_safe", "gphi_raw",
                "gphi_executed", "eta_raw", "eta_executed", "u_exec", "active",
                "burst_ordinal", "burst_step", "executed_l2", "cosine", "norm_ratio",
                "candidate_since", "stuck_timer", "max_stuck_timer", "event",
            )} if task["trace"] else {})
        episode_keys = []
        for task in tasks:
            if task["flow_mode"] == "exact":
                key = jax.random.fold_in(
                    jax.random.PRNGKey(state["exact_flow_root_seed"]),
                    state["exact_flow_rollout_id"],
                )
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(task["seed"]), state["rng_namespace"])
            episode_keys.append(np.asarray(key, dtype=np.uint32))
        padded_keys = list(episode_keys)
        while len(padded_keys) < args.batch:
            padded_keys.append(padded_keys[-1])
        key0 = jnp.asarray(np.asarray(padded_keys))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        absolute_steps = np.zeros(args.batch, dtype=np.uint32)
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            for index, env in enumerate(envs):
                if not env.done and errors[index] is None:
                    observations[index] = env.observation()
                    absolute_steps[index] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(absolute_steps))))
            pending: dict[int, dict] = {}
            feature_indices: list[int] = []
            features: list[np.ndarray] = []
            for index, (env, builder, task) in enumerate(zip(envs, builders, tasks)):
                if env.done or errors[index] is not None:
                    continue
                try:
                    step = int(env.step_count)
                    observation = np.asarray(observations[index], dtype=np.float64)
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    if task["condition"] == "H1":
                        active = True
                        local = step - int(state["query_step"])
                        burst_ordinal = local // 8
                        burst_step = local % 8 + 1
                        if local % 8 == 0:
                            burst_count[index] += 1
                    else:
                        if step % 8 == 0:
                            burst_end[index] = step + int(task["burst_length"])
                            trigger_step[index] = step
                            burst_count[index] += 1
                        active = step < int(burst_end[index])
                        burst_ordinal = int(burst_count[index] - 1) if active else -1
                        burst_step = step - int(trigger_step[index]) + 1 if active else 0
                    pending[index] = {
                        "step": step, "observation": observation, "flow": flow,
                        "A": A, "lower": lower, "safe": safe, "retry1": retry1,
                        "active": active, "burst_ordinal": burst_ordinal, "burst_step": burst_step,
                    }
                    if active:
                        feature, _ = builder.build(FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf)
                        feature_indices.append(index)
                        features.append(feature)
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
            predictions = model(np.asarray(features, dtype=np.float64)) if features else np.empty((0, 4), dtype=np.float64)
            prediction_by_index = {index: predictions[position].reshape(2, 2) for position, index in enumerate(feature_indices)}
            for index, (env, teacher, task) in enumerate(zip(envs, teachers, tasks)):
                if index not in pending or errors[index] is not None:
                    continue
                item = pending[index]
                try:
                    safe = item["safe"]
                    if item["active"]:
                        gphi_raw = prediction_by_index[index]
                        gphi_exec_action, _, retry2, _ = project_velocity_with_retry(
                            safe + gphi_raw, item["A"], item["lower"], config.max_speed, cbf
                        )
                        gphi_executed = gphi_exec_action - safe
                        eta_raw = teacher(item["observation"], safe, config.max_speed)
                        eta_exec_action, _, retry_eta, _ = project_velocity_with_retry(
                            safe + eta_raw, item["A"], item["lower"], config.max_speed, cbf
                        )
                        eta_executed = eta_exec_action - safe
                        executed = gphi_exec_action
                        raw_l2 = float(np.linalg.norm(gphi_raw - eta_raw))
                        executed_l2 = float(np.linalg.norm(gphi_executed - eta_executed))
                        cos = cosine(gphi_executed.reshape(-1), eta_executed.reshape(-1))
                        eta_norm = float(np.linalg.norm(eta_executed))
                        ratio = float(np.linalg.norm(gphi_executed) / eta_norm) if eta_norm > 1e-12 else float("nan")
                        gphi_rewrite = float(np.linalg.norm(gphi_exec_action - (safe + gphi_raw)))
                        eta_rewrite = float(np.linalg.norm(eta_exec_action - (safe + eta_raw)))
                        active_count[index] += 1
                        raw_sum[index] += float(np.linalg.norm(gphi_raw))
                        executed_sum[index] += float(np.linalg.norm(gphi_executed))
                        rewrite_sum[index] += gphi_rewrite
                        second_retries[index] += int(retry2)
                        teacher_retries[index] += int(retry_eta)
                        log = active_logs[index]
                        values = {
                            "global_step": item["step"], "burst_ordinal": item["burst_ordinal"],
                            "burst_step": item["burst_step"], "gphi_raw": gphi_raw.reshape(4),
                            "gphi_executed": gphi_executed.reshape(4), "eta_raw": eta_raw.reshape(4),
                            "eta_executed": eta_executed.reshape(4), "raw_l2": raw_l2,
                            "executed_l2": executed_l2, "cosine": cos, "norm_ratio": ratio,
                            "gphi_rewrite": gphi_rewrite, "eta_rewrite": eta_rewrite,
                            "positions": env.positions.copy(),
                            "goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1),
                            "inter_agent_distance": float(np.linalg.norm(env.positions[0] - env.positions[1])),
                            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
                            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
                        }
                        for name, value in values.items():
                            log[name].append(value)
                    else:
                        gphi_raw = np.zeros((2, 2), dtype=np.float64)
                        gphi_executed = np.zeros((2, 2), dtype=np.float64)
                        eta_raw = np.zeros((2, 2), dtype=np.float64)
                        eta_executed = np.zeros((2, 2), dtype=np.float64)
                        executed = safe
                        raw_l2 = executed_l2 = cos = ratio = float("nan")
                    first_retries[index] += int(item["retry1"])
                    before = env.positions.copy()
                    if task["trace"]:
                        trace = traces[index]
                        trace_values = {
                            "global_step": item["step"], "positions_before": before,
                            "goal_errors_before": np.linalg.norm(env.goals - before, axis=-1),
                            "inter_agent_distance_before": float(np.linalg.norm(before[0] - before[1])),
                            "u_flow": item["flow"], "u_safe": safe, "gphi_raw": gphi_raw,
                            "gphi_executed": gphi_executed, "eta_raw": eta_raw,
                            "eta_executed": eta_executed, "u_exec": executed,
                            "active": item["active"], "burst_ordinal": item["burst_ordinal"],
                            "burst_step": item["burst_step"], "executed_l2": executed_l2,
                            "cosine": cos, "norm_ratio": ratio,
                            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
                            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
                        }
                        for name, value in trace_values.items():
                            trace[name].append(value)
                    _, _, _, info = env.step(executed)
                    if task["trace"]:
                        traces[index]["positions_after"].append(env.positions.copy())
                        traces[index]["event"].append(info["termination"])
                    jdef[index] += config.dt * float(np.sum((executed - safe) ** 2))
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        results: list[dict] = []
        packed_logs: list[dict[str, np.ndarray]] = []
        for index, task in enumerate(tasks):
            summary = envs[index].summary()
            steps = int(envs[index].step_count - state["query_step"])
            count = int(active_count[index])
            log = active_logs[index]
            executed_errors = [float(value) for value in log["executed_l2"]]
            cosines = [float(value) for value in log["cosine"] if np.isfinite(value)]
            first_mask = [ordinal == 0 for ordinal in log["burst_ordinal"]]
            first_errors = [value for value, keep in zip(executed_errors, first_mask) if keep]
            later_errors = [value for value, keep in zip(executed_errors, first_mask) if not keep]
            row = {
                "case_id": state["case_id"], "benchmark": state["benchmark"],
                "state_id": state["state_id"], "query_step": state["query_step"],
                "query_phase_mod8": int(state["query_step"]) % 8,
                "condition": task["condition"], "burst_length": int(task["burst_length"]),
                "flow_mode": task["flow_mode"], "seed": task.get("seed"),
                "outcome": outcome(envs[index], errors[index]), "execution_error": errors[index],
                "continuation_steps": steps, "terminal_global_step": int(envs[index].step_count),
                "completion_time_seconds": float(envs[index].step_count * config.dt),
                "J_def": float(jdef[index]), "burst_count": int(burst_count[index]),
                "active_steps": count, "active_fraction": count / steps if steps else 0.0,
                "active_raw_norm_mean": float(raw_sum[index] / count) if count else None,
                "active_executed_norm_mean": float(executed_sum[index] / count) if count else None,
                "active_rewrite_norm_mean": float(rewrite_sum[index] / count) if count else None,
                "teacher_executed_l2_mean": float(np.mean(executed_errors)) if executed_errors else None,
                "teacher_executed_l2_first": executed_errors[0] if executed_errors else None,
                "teacher_executed_l2_last": executed_errors[-1] if executed_errors else None,
                "teacher_executed_l2_slope_per_active_step": slope(executed_errors),
                "teacher_executed_l2_first_burst_mean": float(np.mean(first_errors)) if first_errors else None,
                "teacher_executed_l2_later_bursts_mean": float(np.mean(later_errors)) if later_errors else None,
                "teacher_cosine_mean": float(np.mean(cosines)) if cosines else None,
                "first_projection_retries": int(first_retries[index]),
                "second_projection_retries": int(second_retries[index]),
                "teacher_diagnostic_projection_retries": int(teacher_retries[index]),
                "first_success_step": summary["first_success_step"],
                "first_deadlock_step": summary["first_deadlock_step"],
                "state_complete": True,
            }
            packed = {name: np.asarray(values) for name, values in log.items()}
            packed["seed"] = np.full(len(log["global_step"]), -1 if task.get("seed") is None else int(task["seed"]), dtype=np.int64)
            packed_logs.append(packed)
            if task["trace"]:
                trace_path = HERE / "traces" / f"{state['state_id']}__{task['condition']}.npz"
                atomic_npz(trace_path, **{name: np.asarray(values) for name, values in traces[index].items()})
                row["trace_file"] = str(trace_path.relative_to(HERE))
                row["trace_sha256"] = file_hash(trace_path)
            results.append(row)
            new_rollouts += 1
            physical_steps += steps
            active_log_rows += len(log["global_step"])
        return results, packed_logs

    for position, state in enumerate(states):
        output = HERE / "raw" / f"{state['state_id']}.jsonl"
        expected_logs = [HERE / "active_logs" / f"{state['state_id']}__{condition}__robust.npz" for condition, _ in CONDITIONS]
        expected_traces = [HERE / "traces" / f"{state['state_id']}__{condition}.npz" for condition, _ in CONDITIONS]
        if output.is_file():
            previous = [json.loads(line) for line in output.read_text().splitlines() if line]
            if len(previous) == 390 and all(row.get("state_complete") for row in previous) and all(path.is_file() for path in (*expected_logs, *expected_traces)):
                print(json.dumps({"skip": state["state_id"], "rows": 390}), flush=True)
                continue
        # For the 16 phase-zero states, L8 is active at every physical step
        # and is exactly the dense H1 controller.  Reuse that identical tuple
        # rather than performing 1040 redundant rollouts.  The off-phase
        # old_r106 state still runs H1 explicitly.
        conditions_to_run = CONDITIONS if int(state["query_step"]) % 8 else CONDITIONS[:-1]
        exact_tasks = [
            {"condition": condition, "burst_length": length, "flow_mode": "exact", "trace": True}
            for condition, length in conditions_to_run
        ]
        output_rows, _ = run_tasks(state, exact_tasks)
        if conditions_to_run != CONDITIONS:
            source_trace = HERE / "traces" / f"{state['state_id']}__L8.npz"
            target_trace = HERE / "traces" / f"{state['state_id']}__H1.npz"
            atomic_copy(source_trace, target_trace)
            h1_exact = dict(next(row for row in output_rows if row["condition"] == "L8"))
            h1_exact.update({
                "condition": "H1", "burst_length": 0, "burst_count": 1,
                "trace_file": str(target_trace.relative_to(HERE)),
                "trace_sha256": file_hash(target_trace),
                "semantic_reuse_from": "L8 phase-zero identity",
            })
            output_rows.append(h1_exact)
        for condition, length in conditions_to_run:
            condition_rows: list[dict] = []
            condition_logs: list[dict[str, np.ndarray]] = []
            tasks = [
                {"condition": condition, "burst_length": length, "flow_mode": "robust", "seed": seed, "trace": False}
                for seed in ROBUST_SEEDS
            ]
            for start in range(0, len(tasks), args.batch):
                rows, logs = run_tasks(state, tasks[start:start + args.batch])
                condition_rows.extend(rows)
                condition_logs.extend(logs)
            output_rows.extend(condition_rows)
            active_path = HERE / "active_logs" / f"{state['state_id']}__{condition}__robust.npz"
            atomic_npz(active_path, **concat_logs(condition_logs))
        if conditions_to_run != CONDITIONS:
            source_active = HERE / "active_logs" / f"{state['state_id']}__L8__robust.npz"
            target_active = HERE / "active_logs" / f"{state['state_id']}__H1__robust.npz"
            atomic_copy(source_active, target_active)
            l8_rows = [row for row in output_rows if row["condition"] == "L8" and row["flow_mode"] == "robust"]
            for source_row in l8_rows:
                h1_row = dict(source_row)
                h1_row.update({
                    "condition": "H1", "burst_length": 0, "burst_count": 1,
                    "semantic_reuse_from": "L8 phase-zero identity",
                })
                output_rows.append(h1_row)
        if len(output_rows) != 390:
            raise RuntimeError((state["state_id"], len(output_rows)))
        atomic_jsonl(output, output_rows)
        counts = {
            condition: sum(
                row["condition"] == condition and row["flow_mode"] == "robust" and row["outcome"] == "success"
                for row in output_rows
            ) for condition, _ in CONDITIONS
        }
        print(json.dumps({
            "completed": state["state_id"], "position": f"{position + 1}/{len(states)}",
            "exact": [(row["condition"], row["outcome"]) for row in output_rows[:6]],
            "robust_successes": counts, "elapsed_s": round(time.monotonic() - started, 1),
        }), flush=True)

    runtime = {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "states": len(states), "new_rollouts": new_rollouts, "physical_steps": physical_steps,
        "active_timestep_log_rows": active_log_rows,
        "elapsed_seconds": time.monotonic() - started,
        "device": [str(device) for device in jax.devices()], "batch": args.batch,
        "checkpoint_sha256": model.sha256,
        "phase_zero_H1_reused_from_L8": sum(int(state["query_step"]) % 8 == 0 for state in states) * 65,
    }
    (HERE / f"runtime_shard{args.shard_index}.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
