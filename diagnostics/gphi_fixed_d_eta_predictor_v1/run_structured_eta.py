"""Run one-shot fixed-D eta prediction with persistent structured feedback."""

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
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
CHECKPOINT = HERE / "best_fixed_d_eta_checkpoint.npz"
FLOW_CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_FLOW = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
ROBUST_SEEDS = tuple(range(95310001, 95310065))
TEMPORAL_STEPS = (0, 1, 2, 4, 8, 16, 32)
TAKEOVER_STEPS = (8, 16, 32)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))
sys.path.insert(0, str(PILOT))
from pilot_common import assert_frozen_sources, sha256  # noqa: E402
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.gphi_training_dataset_v1.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


class FiniteHistoryView:
    def __init__(self, env):
        self._env = env
        values = np.asarray(env.distance_history, dtype=np.float64)
        finite = values[np.isfinite(values).all(axis=1)]
        if not len(finite):
            raise RuntimeError("no finite history")
        self.distance_history = finite

    def __getattr__(self, name):
        return getattr(self._env, name)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
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


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else None


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
    assert_frozen_sources()
    integrity = json.loads((HERE / "integrity_audit.json").read_text())
    if integrity["status"] != "PASS" or sha256(CHECKPOINT) != integrity["checkpoint_sha256"]:
        raise RuntimeError("integrity gate failed")
    if sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW:
        raise RuntimeError("Flow checkpoint hash mismatch")
    source = json.loads((HERE / "source_manifest.json").read_text())
    capacity = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity["environment"])
    cbf = CBFConfig(**capacity["cbf"])
    model = FixedDEtaPredictor(CHECKPOINT)
    policy, provenance = load_policy(FLOW_CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != capacity["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    eta_by_state = {row["state_id"]: tuple(float(x) for x in row["eta"]) for row in source["etas_for_diagnostic_only"]}
    all_states = source["states"]
    states = [row for index, row in enumerate(all_states) if index % args.shard_count == args.shard_index]
    started = time.monotonic()
    counters = {"rollouts": 0, "steps": 0, "temporal_rows": 0}

    def run_tasks(state: dict, tasks: list[dict]) -> tuple[list[dict], list[dict]]:
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        builders = [StartupAwareFeatureBuilder() for _ in tasks]
        true_teachers = [DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]])) for _ in tasks]
        predicted_teachers: list[DiagnosticCorrector | None] = [None] * len(tasks)
        eta_hats: list[np.ndarray | None] = [None] * len(tasks)
        eta_raw_normalized: list[np.ndarray | None] = [None] * len(tasks)
        errors: list[dict | None] = [None] * len(tasks)
        jdef = np.zeros(len(tasks), dtype=np.float64)
        first_retries = np.zeros(len(tasks), dtype=np.int64)
        second_retries = np.zeros(len(tasks), dtype=np.int64)
        raw_sum = np.zeros(len(tasks), dtype=np.float64)
        exec_sum = np.zeros(len(tasks), dtype=np.float64)
        rewrite_sum = np.zeros(len(tasks), dtype=np.float64)
        episode_keys = []
        for task in tasks:
            if task["flow_mode"] == "exact":
                key = jax.random.fold_in(jax.random.PRNGKey(state["exact_flow_root_seed"]), state["exact_flow_rollout_id"])
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(task["seed"]), state["rng_namespace"])
            episode_keys.append(np.asarray(key, dtype=np.uint32))
        padded = list(episode_keys)
        while len(padded) < args.batch:
            padded.append(padded[-1])
        key0 = jnp.asarray(np.asarray(padded))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        absolute_steps = np.zeros(args.batch, dtype=np.uint32)
        temporal_rows: list[dict] = []
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            for index, env in enumerate(envs):
                if not env.done and errors[index] is None:
                    observations[index] = env.observation()
                    absolute_steps[index] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(absolute_steps))))
            pending = {}
            onset_indices = []
            onset_features = []
            for index, (env, builder) in enumerate(zip(envs, builders)):
                if env.done or errors[index] is not None:
                    continue
                try:
                    step = int(env.step_count)
                    local = step - int(state["query_step"])
                    obs = np.asarray(observations[index], dtype=np.float64)
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    pending[index] = {"step": step, "local": local, "obs": obs, "safe": safe,
                                      "A": A, "lower": lower, "retry1": retry1}
                    if predicted_teachers[index] is None:
                        feature, _ = builder.build(FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf)
                        onset_indices.append(index)
                        onset_features.append(feature)
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
            if onset_features:
                eta, raw_norm, _ = model.predict(np.asarray(onset_features))
                for position, index in enumerate(onset_indices):
                    eta_hats[index] = eta[position]
                    eta_raw_normalized[index] = raw_norm[position]
                    predicted_teachers[index] = DiagnosticCorrector(DiagnosticPhi(*eta[position].tolist()))
            for index, (env, task, true_teacher) in enumerate(zip(envs, tasks, true_teachers)):
                if index not in pending or errors[index] is not None:
                    continue
                item = pending[index]
                try:
                    predicted_teacher = predicted_teachers[index]
                    if predicted_teacher is None:
                        raise RuntimeError("eta prediction missing")
                    predicted_raw = predicted_teacher(item["obs"], item["safe"], config.max_speed)
                    predicted_action, _, retry2, _ = project_velocity_with_retry(
                        item["safe"] + predicted_raw, item["A"], item["lower"], config.max_speed, cbf
                    )
                    true_raw = true_teacher(item["obs"], item["safe"], config.max_speed)
                    true_action, _, _, _ = project_velocity_with_retry(
                        item["safe"] + true_raw, item["A"], item["lower"], config.max_speed, cbf
                    )
                    predicted_exec = predicted_action - item["safe"]
                    true_exec = true_action - item["safe"]
                    takeover = task.get("takeover_k")
                    use_true = takeover is not None and item["local"] >= takeover
                    executed = true_action if use_true else predicted_action
                    if (task["condition"] == "structured" and task["flow_mode"] == "robust"
                            and item["local"] in TEMPORAL_STEPS):
                        true_norm = float(np.linalg.norm(true_exec))
                        temporal_rows.append({
                            "case_id": state["case_id"], "benchmark": state["benchmark"],
                            "state_id": state["state_id"], "seed": task["seed"], "k": item["local"],
                            "predicted_eta": eta_hats[index].tolist(), "oracle_eta": list(eta_by_state[state["state_id"]]),
                            "raw_correction_l2": float(np.linalg.norm(predicted_raw - true_raw)),
                            "executed_action_l2": float(np.linalg.norm(predicted_exec - true_exec)),
                            "cosine_similarity": cosine(predicted_exec.reshape(-1), true_exec.reshape(-1)),
                            "norm_ratio": float(np.linalg.norm(predicted_exec) / true_norm) if true_norm > 1e-12 else None,
                            "predicted_projection_rewrite": float(np.linalg.norm(predicted_action - (item["safe"] + predicted_raw))),
                            "oracle_projection_rewrite": float(np.linalg.norm(true_action - (item["safe"] + true_raw))),
                        })
                    delta = executed - item["safe"]
                    jdef[index] += config.dt * float(np.sum(delta ** 2))
                    raw_sum[index] += float(np.linalg.norm(predicted_raw))
                    exec_sum[index] += float(np.linalg.norm(predicted_exec))
                    rewrite_sum[index] += float(np.linalg.norm(predicted_action - (item["safe"] + predicted_raw)))
                    first_retries[index] += int(item["retry1"])
                    second_retries[index] += int(retry2)
                    env.step(executed)
                    counters["steps"] += 1
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        rows = []
        for index, task in enumerate(tasks):
            summary = envs[index].summary()
            steps = int(envs[index].step_count - state["query_step"])
            raw_norm = eta_raw_normalized[index]
            eta_hat = eta_hats[index]
            rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state["state_id"],
                "query_step": state["query_step"], "condition": task["condition"],
                "takeover_k": task.get("takeover_k"), "flow_mode": task["flow_mode"], "seed": task.get("seed"),
                "predicted_eta": eta_hat.tolist() if eta_hat is not None else None,
                "oracle_eta": list(eta_by_state[state["state_id"]]),
                "normalized_eta_raw": raw_norm.tolist() if raw_norm is not None else None,
                "eta_prediction_clipped": bool(np.any((raw_norm < 0) | (raw_norm > 1))) if raw_norm is not None else None,
                "outcome": outcome(envs[index], errors[index]), "execution_error": errors[index],
                "continuation_steps": steps, "terminal_global_step": int(envs[index].step_count),
                "completion_time_seconds": float(envs[index].step_count * config.dt),
                "J_def": float(jdef[index]),
                "raw_correction_norm_mean": float(raw_sum[index] / steps) if steps else None,
                "executed_correction_norm_mean": float(exec_sum[index] / steps) if steps else None,
                "projection_rewrite_mean": float(rewrite_sum[index] / steps) if steps else None,
                "first_projection_retries": int(first_retries[index]),
                "second_projection_retries": int(second_retries[index]),
                "wall_collision": bool(summary["wall_collision"]),
                "agent_collision": bool(summary["agent_collision"]),
                "first_success_step": summary["first_success_step"],
                "first_deadlock_step": summary["first_deadlock_step"],
                "state_complete": True,
            })
            counters["rollouts"] += 1
        counters["temporal_rows"] += len(temporal_rows)
        return rows, temporal_rows

    for position, state in enumerate(states):
        output = HERE / "raw" / f"{state['state_id']}.jsonl"
        temporal_output = HERE / "temporal_raw" / f"{state['state_id']}.jsonl"
        if output.is_file() and temporal_output.is_file():
            old = [json.loads(line) for line in output.read_text().splitlines() if line]
            if len(old) == 257 and all(row.get("state_complete") for row in old):
                print(json.dumps({"skip": state["state_id"], "rows": 257}), flush=True)
                continue
        tasks = [{"condition": "structured", "flow_mode": "exact"}]
        for seed in ROBUST_SEEDS:
            tasks.append({"condition": "structured", "flow_mode": "robust", "seed": seed})
            for k in TAKEOVER_STEPS:
                tasks.append({"condition": f"takeover{k}", "takeover_k": k, "flow_mode": "robust", "seed": seed})
        rows, temporal = [], []
        for start in range(0, len(tasks), args.batch):
            batch_rows, batch_temporal = run_tasks(state, tasks[start:start + args.batch])
            rows.extend(batch_rows)
            temporal.extend(batch_temporal)
        if len(rows) != 257:
            raise RuntimeError((state["state_id"], len(rows)))
        atomic_jsonl(output, rows)
        atomic_jsonl(temporal_output, temporal)
        counts = {condition: sum(row["condition"] == condition and row["outcome"] == "success" for row in rows)
                  for condition in ("structured", "takeover8", "takeover16", "takeover32")}
        print(json.dumps({"completed": state["state_id"], "position": f"{position + 1}/{len(states)}",
                          "successes": counts, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    runtime = {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "assigned_states": len(states), "new_rollouts": counters["rollouts"],
        "physical_steps": counters["steps"], "temporal_rows": counters["temporal_rows"],
        "elapsed_seconds": time.monotonic() - started, "device": [str(x) for x in jax.devices()],
        "batch": args.batch, "checkpoint_sha256": integrity["checkpoint_sha256"],
        "gpu_memory_fraction_process_cap": float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "nan")),
    }
    (HERE / f"runtime_shard{args.shard_index}.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
