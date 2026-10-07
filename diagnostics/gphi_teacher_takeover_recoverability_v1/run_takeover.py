"""Run dense-learned prefixes followed by permanent frozen-eta takeover."""

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
HERE = ROOT / "diagnostics/gphi_teacher_takeover_recoverability_v1"
BASE = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
FLOW_CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_FLOW = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
CHECKPOINTS = {
    "coverage": (
        ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz",
        "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700",
    ),
    "k1": (
        ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1/best_dagger_k1_checkpoint.npz",
        "83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a",
    ),
}
KS = (0, 1, 2, 4, 8, 16, 32)
ROBUST_SEEDS = tuple(range(95310001, 95310065))

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PILOT))
import run_bursts as base  # noqa: E402
from pilot_common import DeterministicGphi, sha256  # noqa: E402
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.gphi_training_dataset_v1.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    gate = json.loads((HERE / "learner_checkpoint_manifest.json").read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError("integrity gate not passed")
    if sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW:
        raise RuntimeError("Flow checkpoint hash mismatch")
    for learner, (path, expected) in CHECKPOINTS.items():
        if sha256(path) != expected:
            raise RuntimeError((learner, "checkpoint hash mismatch"))
    source = json.loads((HERE / "source_manifest.json").read_text())
    capacity = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity["environment"])
    cbf = CBFConfig(**capacity["cbf"])
    models = {name: DeterministicGphi(path) for name, (path, _) in CHECKPOINTS.items()}
    policy, provenance = load_policy(FLOW_CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != capacity["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    eta_by_state = {row["state_id"]: tuple(float(x) for x in row["eta"]) for row in source["etas_for_diagnostic_only"]}
    all_states = source["states"]
    states = [row for index, row in enumerate(all_states) if index % args.shard_count == args.shard_index]
    started = time.monotonic()
    counters = {"rollouts": 0, "steps": 0, "reference_steps": 0}

    def keys_for(state: dict, seeds: list[int]) -> jnp.ndarray:
        keys = [np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed), state["rng_namespace"]), dtype=np.uint32) for seed in seeds]
        while len(keys) < args.batch:
            keys.append(keys[-1])
        return jnp.asarray(np.asarray(keys))

    def flow_actions(envs: list, keys0: jnp.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        steps = np.zeros(args.batch, dtype=np.uint32)
        for index, env in enumerate(envs):
            if not env.done:
                observations[index] = env.observation()
                steps[index] = env.step_count
        actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
        return observations, steps, actions

    def ensure_reference(state: dict) -> dict[tuple[int, int], dict]:
        path = HERE / "reference" / f"{state['state_id']}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as values:
                if (list(values["seeds"]) == list(ROBUST_SEEDS) and list(values["ks"]) == list(KS)
                        and values["positions"].shape == (64, 7, 2, 2)):
                    return {
                        (int(seed), int(k)): {
                            "positions": values["positions"][si, ki],
                            "goal_errors": values["goal_errors"][si, ki],
                        }
                        for si, seed in enumerate(values["seeds"])
                        for ki, k in enumerate(values["ks"])
                    }
        positions = np.full((64, 7, 2, 2), np.nan, dtype=np.float64)
        goal_errors = np.full((64, 7, 2), np.nan, dtype=np.float64)
        teacher = DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]]))
        for start in range(0, 64, args.batch):
            seeds = list(ROBUST_SEEDS[start:start + args.batch])
            envs = [restore_full(Path(state["state_file"]), config) for _ in seeds]
            keys0 = keys_for(state, seeds)
            max_k = max(KS)
            while any((not env.done) and int(env.step_count) - int(state["query_step"]) <= max_k for env in envs):
                observations, _, actions = flow_actions(envs, keys0)
                for index, env in enumerate(envs):
                    if env.done:
                        continue
                    local = int(env.step_count) - int(state["query_step"])
                    if local > max_k:
                        continue
                    if local in KS:
                        ki = KS.index(local)
                        positions[start + index, ki] = env.positions
                        goal_errors[start + index, ki] = np.linalg.norm(env.goals - env.positions, axis=-1)
                    obs = np.asarray(observations[index], dtype=np.float64)
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    eta_raw = teacher(obs, safe, config.max_speed)
                    executed, _, _, _ = project_velocity_with_retry(safe + eta_raw, A, lower, config.max_speed, cbf)
                    env.step(executed)
                    counters["reference_steps"] += 1
        if not np.isfinite(positions).all() or not np.isfinite(goal_errors).all():
            raise RuntimeError((state["state_id"], "teacher reference terminated before k=32"))
        base.atomic_npz(path, seeds=np.asarray(ROBUST_SEEDS), ks=np.asarray(KS), positions=positions, goal_errors=goal_errors)
        return {
            (seed, k): {"positions": positions[si, ki], "goal_errors": goal_errors[si, ki]}
            for si, seed in enumerate(ROBUST_SEEDS) for ki, k in enumerate(KS)
        }

    def run_batch(state: dict, learner: str, tasks: list[dict], reference: dict) -> list[dict]:
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        builders = [StartupAwareFeatureBuilder() for _ in tasks]
        teachers = [DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]])) for _ in tasks]
        errors = [None] * len(tasks)
        takeover_metrics: list[dict | None] = [None] * len(tasks)
        first_retries = np.zeros(len(tasks), dtype=np.int64)
        second_retries = np.zeros(len(tasks), dtype=np.int64)
        teacher_retries = np.zeros(len(tasks), dtype=np.int64)
        keys0 = keys_for(state, [task["seed"] for task in tasks])
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            observations, _, actions = flow_actions(envs, keys0)
            pending: dict[int, dict] = {}
            feature_indices: list[int] = []
            features: list[np.ndarray] = []
            for index, (env, builder, task) in enumerate(zip(envs, builders, tasks)):
                if env.done or errors[index] is not None:
                    continue
                try:
                    step = int(env.step_count)
                    local = step - int(state["query_step"])
                    obs = np.asarray(observations[index], dtype=np.float64)
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    pending[index] = {"step": step, "local": local, "obs": obs, "flow": flow,
                                      "A": A, "lower": lower, "safe": safe, "retry1": retry1}
                    if local <= task["k"]:
                        feature, _ = builder.build(base.FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf)
                        feature_indices.append(index)
                        features.append(feature)
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
            predicted = models[learner](np.asarray(features, dtype=np.float64)) if features else np.empty((0, 4), dtype=np.float64)
            predictions = {index: predicted[pos].reshape(2, 2) for pos, index in enumerate(feature_indices)}
            for index, (env, teacher, task) in enumerate(zip(envs, teachers, tasks)):
                if index not in pending or errors[index] is not None:
                    continue
                item = pending[index]
                try:
                    safe = item["safe"]
                    eta_raw = teacher(item["obs"], safe, config.max_speed)
                    eta_action, _, eta_retry, _ = project_velocity_with_retry(safe + eta_raw, item["A"], item["lower"], config.max_speed, cbf)
                    eta_exec = eta_action - safe
                    teacher_retries[index] += int(eta_retry)
                    if item["local"] < task["k"]:
                        g_raw = predictions[index]
                        g_action, _, retry2, _ = project_velocity_with_retry(safe + g_raw, item["A"], item["lower"], config.max_speed, cbf)
                        executed = g_action
                        second_retries[index] += int(retry2)
                    else:
                        executed = eta_action
                        if item["local"] == task["k"] and takeover_metrics[index] is None:
                            g_raw = predictions[index]
                            g_action, _, retry2, _ = project_velocity_with_retry(safe + g_raw, item["A"], item["lower"], config.max_speed, cbf)
                            g_exec = g_action - safe
                            ref = reference[(task["seed"], task["k"])]
                            eta_norm = float(np.linalg.norm(eta_exec))
                            takeover_metrics[index] = {
                                "takeover_global_step": item["step"],
                                "raw_action_L2": float(np.linalg.norm(g_raw - eta_raw)),
                                "executed_action_L2": float(np.linalg.norm(g_exec - eta_exec)),
                                "cosine_similarity": safe_float(base.cosine(g_exec.reshape(-1), eta_exec.reshape(-1))),
                                "norm_ratio": safe_float(np.linalg.norm(g_exec) / eta_norm) if eta_norm > 1e-12 else None,
                                "learner_projection_rewrite": float(np.linalg.norm(g_action - (safe + g_raw))),
                                "teacher_projection_rewrite": float(np.linalg.norm(eta_action - (safe + eta_raw))),
                                "learner_executed_norm": float(np.linalg.norm(g_exec)),
                                "teacher_executed_norm": eta_norm,
                                "position_deviation_L2": float(np.linalg.norm(env.positions - ref["positions"])),
                                "position_deviation_max_agent": float(np.max(np.linalg.norm(env.positions - ref["positions"], axis=1))),
                                "goal_error_deviation_L2": float(np.linalg.norm(np.linalg.norm(env.goals - env.positions, axis=-1) - ref["goal_errors"])),
                            }
                            second_retries[index] += int(retry2)
                    first_retries[index] += int(item["retry1"])
                    env.step(executed)
                    counters["steps"] += 1
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        rows = []
        for index, task in enumerate(tasks):
            metric = takeover_metrics[index] or {
                "takeover_global_step": None, "raw_action_L2": None, "executed_action_L2": None,
                "cosine_similarity": None, "norm_ratio": None, "learner_projection_rewrite": None,
                "teacher_projection_rewrite": None, "learner_executed_norm": None,
                "teacher_executed_norm": None, "position_deviation_L2": None,
                "position_deviation_max_agent": None, "goal_error_deviation_L2": None,
            }
            rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state["state_id"],
                "query_step": state["query_step"], "learner": learner, "seed": task["seed"], "k": task["k"],
                "outcome": base.outcome(envs[index], errors[index]), "execution_error": errors[index],
                "terminal_global_step": int(envs[index].step_count),
                "continuation_steps": int(envs[index].step_count - state["query_step"]),
                "terminated_before_takeover": takeover_metrics[index] is None,
                "first_projection_retries": int(first_retries[index]),
                "learner_projection_retries": int(second_retries[index]),
                "teacher_projection_retries": int(teacher_retries[index]),
                "state_complete": True, **metric,
            })
            counters["rollouts"] += 1
        return rows

    for position, state in enumerate(states):
        reference = ensure_reference(state)
        for learner in CHECKPOINTS:
            output = HERE / "raw" / f"{state['state_id']}__{learner}.jsonl"
            if output.is_file():
                old = [json.loads(line) for line in output.read_text().splitlines() if line]
                if (len(old) == 448 and all(row.get("state_complete") for row in old)
                        and {(row["seed"], row["k"]) for row in old} == {(seed, k) for seed in ROBUST_SEEDS for k in KS}):
                    print(json.dumps({"skip": state["state_id"], "learner": learner}), flush=True)
                    continue
            tasks = [{"seed": seed, "k": k} for k in KS for seed in ROBUST_SEEDS]
            rows = []
            for start in range(0, len(tasks), args.batch):
                rows.extend(run_batch(state, learner, tasks[start:start + args.batch], reference))
            if len(rows) != 448:
                raise RuntimeError((state["state_id"], learner, len(rows)))
            base.atomic_jsonl(output, rows)
            counts = {k: sum(row["k"] == k and row["outcome"] == "success" for row in rows) for k in KS}
            print(json.dumps({"completed": state["state_id"], "learner": learner,
                              "position": f"{position + 1}/{len(states)}", "successes": counts,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    runtime = {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "assigned_states": len(states), "new_takeover_rollouts": counters["rollouts"],
        "takeover_physical_steps": counters["steps"], "reference_prefix_steps": counters["reference_steps"],
        "elapsed_seconds": time.monotonic() - started, "device": [str(x) for x in jax.devices()],
        "batch": args.batch, "checkpoint_sha256": {name: model.sha256 for name, model in models.items()},
        "gpu_memory_fraction_process_cap": float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "nan")),
    }
    (HERE / f"runtime_shard{args.shard_index}.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
