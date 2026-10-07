"""Run exact H1/H8 traces and matched robust64 H8 fixed-eta continuations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
SOURCE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_training_dataset_v1.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_CHECKPOINT = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
ROBUST_SEEDS = list(range(95310001, 95310065))


def sha256(path: Path) -> str:
    import hashlib
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


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


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
    if integrity["status"] != "PREPARED_REUSE_GATE_PASS":
        raise RuntimeError("preparation gate failed")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Flow checkpoint mismatch")
    source_manifest = json.loads((SOURCE / "strict_deadlock_manifest.json").read_text())
    config = Config(**source_manifest["environment"])
    cbf = CBFConfig(**source_manifest["cbf"])
    policy, provenance = load_policy(CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != source_manifest["environment"]:
        raise RuntimeError("Flow checkpoint environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    states = json.loads((HERE / "source_state_manifest.json").read_text())["states"]
    eta_by_state = {
        row["state_id"]: tuple(float(value) for value in row["eta"])
        for row in json.loads((HERE / "eta_manifest.json").read_text())["etas"]
    }
    states = [row for index, row in enumerate(states) if index % args.shard_count == args.shard_index]
    started = time.monotonic()
    new_rollouts = physical_steps = 0

    def run_tasks(state: dict, tasks: list[dict]) -> list[dict]:
        nonlocal new_rollouts, physical_steps
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        correctors = [DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]])) for _ in tasks]
        errors: list[dict | None] = [None] * len(tasks)
        jdef = np.zeros(len(tasks), dtype=np.float64)
        active_raw = np.zeros(len(tasks), dtype=np.float64)
        active_exec = np.zeros(len(tasks), dtype=np.float64)
        active_rewrite = np.zeros(len(tasks), dtype=np.float64)
        active_count = np.zeros(len(tasks), dtype=np.int64)
        omitted_raw = np.zeros(len(tasks), dtype=np.float64)
        omitted_exec = np.zeros(len(tasks), dtype=np.float64)
        omitted_rewrite = np.zeros(len(tasks), dtype=np.float64)
        omitted_count = np.zeros(len(tasks), dtype=np.int64)
        omitted_exec_values: list[list[float]] = [[] for _ in tasks]
        first_retries = np.zeros(len(tasks), dtype=np.int64)
        second_retries = np.zeros(len(tasks), dtype=np.int64)
        traces: list[dict[str, list]] = []
        for task in tasks:
            traces.append({key: [] for key in (
                "global_step", "positions_before", "positions_after", "goal_errors_before",
                "inter_agent_distance_before", "u_flow", "u_safe", "dense_raw_correction",
                "dense_executed_correction", "u_exec", "scheduled", "candidate_since",
                "stuck_timer", "max_stuck_timer", "event",
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
        while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
            for i, env in enumerate(envs):
                if not env.done and errors[i] is None:
                    observations[i] = env.observation()
                    absolute_steps[i] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(absolute_steps))))
            for i, (env, corrector, task) in enumerate(zip(envs, correctors, tasks)):
                if env.done or errors[i] is not None:
                    continue
                try:
                    step = int(env.step_count)
                    observation = np.asarray(observations[i], dtype=np.float64)
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    dense_raw = corrector(observation, safe, config.max_speed)
                    dense_exec, _, retry2, _ = project_velocity_with_retry(
                        safe + dense_raw, A, lower, config.max_speed, cbf
                    )
                    dense_correction = dense_exec - safe
                    scheduled = task["cadence"] == 1 or step % 8 == 0
                    executed = dense_exec if scheduled else safe
                    if scheduled:
                        active_count[i] += 1
                        active_raw[i] += float(np.linalg.norm(dense_raw))
                        active_exec[i] += float(np.linalg.norm(dense_correction))
                        active_rewrite[i] += float(np.linalg.norm(dense_exec - (safe + dense_raw)))
                    else:
                        omitted_count[i] += 1
                        omitted_raw[i] += float(np.linalg.norm(dense_raw))
                        value = float(np.linalg.norm(dense_correction))
                        omitted_exec[i] += value
                        omitted_exec_values[i].append(value)
                        omitted_rewrite[i] += float(np.linalg.norm(dense_exec - (safe + dense_raw)))
                    before = env.positions.copy()
                    if task["trace"]:
                        trace = traces[i]
                        trace["global_step"].append(step)
                        trace["positions_before"].append(before)
                        trace["goal_errors_before"].append(np.linalg.norm(env.goals - before, axis=-1))
                        trace["inter_agent_distance_before"].append(float(np.linalg.norm(before[0] - before[1])))
                        trace["u_flow"].append(flow)
                        trace["u_safe"].append(safe)
                        trace["dense_raw_correction"].append(dense_raw)
                        trace["dense_executed_correction"].append(dense_correction)
                        trace["u_exec"].append(executed)
                        trace["scheduled"].append(scheduled)
                        trace["candidate_since"].append(-1 if env.candidate_since is None else env.candidate_since)
                        trace["stuck_timer"].append(env.stuck_timer)
                        trace["max_stuck_timer"].append(env.max_stuck_timer)
                    _, _, _, info = env.step(executed)
                    if task["trace"]:
                        traces[i]["positions_after"].append(env.positions.copy())
                        traces[i]["event"].append(info["termination"])
                    jdef[i] += config.dt * float(np.sum((executed - safe) ** 2))
                    first_retries[i] += int(retry1)
                    second_retries[i] += int(retry2)
                except Exception as exc:
                    errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        result = []
        for i, task in enumerate(tasks):
            summary = envs[i].summary()
            count_a = int(active_count[i]); count_o = int(omitted_count[i])
            values_o = np.asarray(omitted_exec_values[i], dtype=np.float64)
            row = {
                "case_id": state["case_id"], "benchmark": state["benchmark"],
                "state_id": state["state_id"], "query_step": state["query_step"],
                "eta": list(eta_by_state[state["state_id"]]),
                "condition": task["condition"], "cadence": task["cadence"],
                "flow_mode": task["flow_mode"], "seed": task.get("seed"),
                "outcome": outcome(envs[i], errors[i]), "execution_error": errors[i],
                "continuation_steps": int(envs[i].step_count - state["query_step"]),
                "terminal_global_step": int(envs[i].step_count),
                "completion_time_seconds": float(envs[i].step_count * config.dt),
                "J_def": float(jdef[i]),
                "active_steps": count_a, "off_steps": count_o,
                "active_raw_norm_mean": float(active_raw[i] / count_a) if count_a else None,
                "active_executed_norm_mean": float(active_exec[i] / count_a) if count_a else None,
                "active_rewrite_norm_mean": float(active_rewrite[i] / count_a) if count_a else None,
                "omitted_dense_raw_norm_mean": float(omitted_raw[i] / count_o) if count_o else None,
                "omitted_dense_executed_norm_mean": float(omitted_exec[i] / count_o) if count_o else None,
                "omitted_dense_executed_norm_p95": float(np.quantile(values_o, .95)) if count_o else None,
                "omitted_dense_executed_norm_max": float(values_o.max()) if count_o else None,
                "omitted_dense_executed_nontrivial_fraction_1e6": float(np.mean(values_o > 1e-6)) if count_o else None,
                "omitted_dense_rewrite_norm_mean": float(omitted_rewrite[i] / count_o) if count_o else None,
                "first_projection_retries": int(first_retries[i]),
                "second_projection_retries": int(second_retries[i]),
                "first_success_step": summary["first_success_step"],
                "first_deadlock_step": summary["first_deadlock_step"],
                "state_complete": True,
            }
            if task["trace"]:
                trace_path = HERE / "traces" / f"{state['state_id']}__{task['condition']}.npz"
                atomic_npz(trace_path, **{key: np.asarray(value) for key, value in traces[i].items()})
                row["trace_file"] = str(trace_path.relative_to(HERE))
                row["trace_sha256"] = sha256(trace_path)
            result.append(row)
            new_rollouts += 1
            physical_steps += row["continuation_steps"]
        return result

    for position, state in enumerate(states):
        output = HERE / "raw" / f"{state['state_id']}.jsonl"
        if output.is_file():
            old = [json.loads(line) for line in output.read_text().splitlines() if line]
            if len(old) == 66 and all(row.get("state_complete") for row in old):
                print(json.dumps({"skip": state["state_id"], "rows": 66}), flush=True)
                continue
        rows = run_tasks(state, [
            {"condition": "H1", "cadence": 1, "flow_mode": "exact", "trace": True},
            {"condition": "H8", "cadence": 8, "flow_mode": "exact", "trace": True},
        ])
        robust_tasks = [
            {"condition": "H8", "cadence": 8, "flow_mode": "robust", "seed": seed, "trace": False}
            for seed in ROBUST_SEEDS
        ]
        for start in range(0, len(robust_tasks), args.batch):
            rows.extend(run_tasks(state, robust_tasks[start:start + args.batch]))
        if len(rows) != 66:
            raise RuntimeError((state["state_id"], len(rows)))
        atomic_jsonl(output, rows)
        print(json.dumps({
            "completed": state["state_id"], "position": f"{position + 1}/{len(states)}",
            "exact": [(row["condition"], row["outcome"]) for row in rows[:2]],
            "H8_robust_success": sum(row["outcome"] == "success" for row in rows[2:]),
            "elapsed_s": round(time.monotonic() - started, 1),
        }), flush=True)

    runtime = {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "states": len(states), "new_rollouts": new_rollouts, "physical_steps": physical_steps,
        "elapsed_seconds": time.monotonic() - started, "device": [str(value) for value in jax.devices()],
        "batch": args.batch, "H1_robust_reused_not_rerun": 1088,
    }
    (HERE / f"runtime_shard{args.shard_index}.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
