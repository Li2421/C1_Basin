"""Run frozen coverage-retrained G_phi under dense H1 and H8+L8."""

from __future__ import annotations

import argparse
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
HERE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1"
BASE_DIR = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
CHECKPOINT = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz"
EXPECTED = "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
FLOW_CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_FLOW = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
ROBUST_SEEDS = tuple(range(95310001, 95310065))

sys.path.insert(0, str(BASE_DIR))
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


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


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
    gate = json.loads((HERE / "checkpoint_integrity.json").read_text())
    if gate["status"] not in {"PREPARED", "PASS"}:
        raise RuntimeError("preparation gate not passed")
    if sha256(CHECKPOINT) != EXPECTED or sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW:
        raise RuntimeError("checkpoint hash mismatch")
    source = json.loads((HERE / "source_manifest.json").read_text())
    capacity = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity["environment"])
    cbf = CBFConfig(**capacity["cbf"])
    model = DeterministicGphi(CHECKPOINT)
    policy, provenance = load_policy(FLOW_CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != capacity["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    eta_by_state = {row["state_id"]: tuple(float(x) for x in row["eta"]) for row in source["etas_for_diagnostic_only"]}
    all_states = source["states"]
    states = [row for i, row in enumerate(all_states) if i % args.shard_count == args.shard_index]
    started = time.monotonic()
    counters = {"rollouts": 0, "steps": 0, "active_rows": 0}

    def run_tasks(state: dict, tasks: list[dict]):
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        builders = [StartupAwareFeatureBuilder() for _ in tasks]
        teachers = [DiagnosticCorrector(DiagnosticPhi(*eta_by_state[state["state_id"]])) for _ in tasks]
        errors = [None] * len(tasks)
        burst_end = np.full(len(tasks), int(state["query_step"]), dtype=np.int64)
        trigger_step = np.full(len(tasks), -1, dtype=np.int64)
        active_count = np.zeros(len(tasks), dtype=np.int64)
        jdef = np.zeros(len(tasks), dtype=np.float64)
        first_retries = np.zeros(len(tasks), dtype=np.int64)
        second_retries = np.zeros(len(tasks), dtype=np.int64)
        teacher_retries = np.zeros(len(tasks), dtype=np.int64)
        names = (
            "global_step", "local_step", "burst_step", "gphi_raw", "gphi_executed",
            "eta_raw", "eta_executed", "raw_l2", "executed_l2", "cosine", "norm_ratio",
            "gphi_rewrite", "eta_rewrite", "positions", "goal_errors", "inter_agent_distance",
            "candidate_since", "stuck_timer", "max_stuck_timer",
        )
        active_logs = [{name: [] for name in names} for _ in tasks]
        trace_names = (
            "global_step", "positions_before", "positions_after", "goal_errors_before",
            "inter_agent_distance_before", "u_flow", "u_safe", "gphi_raw", "gphi_executed",
            "eta_raw", "eta_executed", "u_exec", "active", "burst_step", "executed_l2",
            "cosine", "norm_ratio", "gphi_rewrite", "eta_rewrite", "candidate_since",
            "stuck_timer", "max_stuck_timer", "event",
        )
        traces = [{name: [] for name in trace_names} if task["trace"] else {} for task in tasks]
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
        keys0 = jnp.asarray(np.asarray(padded))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        absolute_steps = np.zeros(args.batch, dtype=np.uint32)
        while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
            for i, env in enumerate(envs):
                if not env.done and errors[i] is None:
                    observations[i] = env.observation()
                    absolute_steps[i] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(absolute_steps))))
            pending, feature_indices, features = {}, [], []
            for i, (env, builder, task) in enumerate(zip(envs, builders, tasks)):
                if env.done or errors[i] is not None:
                    continue
                try:
                    step = int(env.step_count)
                    obs = np.asarray(observations[i], dtype=np.float64)
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    if task["condition"] == "H1":
                        active = True
                        burst_step = (step - int(state["query_step"])) % 8 + 1
                    else:
                        if step % 8 == 0:
                            burst_end[i] = step + 8
                            trigger_step[i] = step
                        active = step < int(burst_end[i])
                        burst_step = step - int(trigger_step[i]) + 1 if active else 0
                    pending[i] = {"step": step, "obs": obs, "flow": flow, "A": A, "lower": lower,
                                  "safe": safe, "retry1": retry1, "active": active, "burst_step": burst_step}
                    if active:
                        feature, _ = builder.build(base.FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf)
                        features.append(feature)
                        feature_indices.append(i)
                except Exception as exc:
                    errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
            predictions = model(np.asarray(features, dtype=np.float64)) if features else np.empty((0, 4), dtype=np.float64)
            predictions = {i: predictions[j].reshape(2, 2) for j, i in enumerate(feature_indices)}
            for i, (env, teacher, task) in enumerate(zip(envs, teachers, tasks)):
                if i not in pending or errors[i] is not None:
                    continue
                item = pending[i]
                try:
                    safe = item["safe"]
                    if item["active"]:
                        g_raw = predictions[i]
                        g_action, _, retry2, _ = project_velocity_with_retry(safe + g_raw, item["A"], item["lower"], config.max_speed, cbf)
                        g_exec = g_action - safe
                        e_raw = teacher(item["obs"], safe, config.max_speed)
                        e_action, _, retry_eta, _ = project_velocity_with_retry(safe + e_raw, item["A"], item["lower"], config.max_speed, cbf)
                        e_exec = e_action - safe
                        executed = g_action
                        raw_l2 = float(np.linalg.norm(g_raw - e_raw))
                        exec_l2 = float(np.linalg.norm(g_exec - e_exec))
                        cos = base.cosine(g_exec.reshape(-1), e_exec.reshape(-1))
                        en = float(np.linalg.norm(e_exec))
                        ratio = float(np.linalg.norm(g_exec) / en) if en > 1e-12 else float("nan")
                        g_rewrite = float(np.linalg.norm(g_action - (safe + g_raw)))
                        e_rewrite = float(np.linalg.norm(e_action - (safe + e_raw)))
                        active_count[i] += 1
                        second_retries[i] += int(retry2)
                        teacher_retries[i] += int(retry_eta)
                        vals = {
                            "global_step": item["step"], "local_step": item["step"] - int(state["query_step"]),
                            "burst_step": item["burst_step"], "gphi_raw": g_raw.reshape(4),
                            "gphi_executed": g_exec.reshape(4), "eta_raw": e_raw.reshape(4),
                            "eta_executed": e_exec.reshape(4), "raw_l2": raw_l2, "executed_l2": exec_l2,
                            "cosine": cos, "norm_ratio": ratio, "gphi_rewrite": g_rewrite,
                            "eta_rewrite": e_rewrite, "positions": env.positions.copy(),
                            "goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1),
                            "inter_agent_distance": float(np.linalg.norm(env.positions[0] - env.positions[1])),
                            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
                            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
                        }
                        for name, value in vals.items():
                            active_logs[i][name].append(value)
                    else:
                        g_raw = g_exec = e_raw = e_exec = np.zeros((2, 2), dtype=np.float64)
                        executed = safe
                        raw_l2 = exec_l2 = cos = ratio = g_rewrite = e_rewrite = float("nan")
                    first_retries[i] += int(item["retry1"])
                    before = env.positions.copy()
                    if task["trace"]:
                        vals = {
                            "global_step": item["step"], "positions_before": before,
                            "goal_errors_before": np.linalg.norm(env.goals - before, axis=-1),
                            "inter_agent_distance_before": float(np.linalg.norm(before[0] - before[1])),
                            "u_flow": item["flow"], "u_safe": safe, "gphi_raw": g_raw,
                            "gphi_executed": g_exec, "eta_raw": e_raw, "eta_executed": e_exec,
                            "u_exec": executed, "active": item["active"], "burst_step": item["burst_step"],
                            "executed_l2": exec_l2, "cosine": cos, "norm_ratio": ratio,
                            "gphi_rewrite": g_rewrite, "eta_rewrite": e_rewrite,
                            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
                            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
                        }
                        for name, value in vals.items():
                            traces[i][name].append(value)
                    _, _, _, info = env.step(executed)
                    if task["trace"]:
                        traces[i]["positions_after"].append(env.positions.copy())
                        traces[i]["event"].append(info["termination"])
                    jdef[i] += config.dt * float(np.sum((executed - safe) ** 2))
                except Exception as exc:
                    errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        results, packed = [], []
        for i, task in enumerate(tasks):
            steps = int(envs[i].step_count - state["query_step"])
            log = {name: np.asarray(values) for name, values in active_logs[i].items()}
            log["seed"] = np.full(len(log["global_step"]), -1 if task.get("seed") is None else task["seed"], dtype=np.int64)
            row = {
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state["state_id"],
                "query_step": state["query_step"], "query_phase_mod8": int(state["query_step"]) % 8,
                "condition": task["condition"], "flow_mode": task["flow_mode"], "seed": task.get("seed"),
                "outcome": base.outcome(envs[i], errors[i]), "execution_error": errors[i],
                "continuation_steps": steps, "terminal_global_step": int(envs[i].step_count),
                "completion_time_seconds": float(envs[i].step_count * config.dt), "J_def": float(jdef[i]),
                "active_steps": int(active_count[i]), "active_fraction": active_count[i] / steps if steps else 0.0,
                "first_projection_retries": int(first_retries[i]), "second_projection_retries": int(second_retries[i]),
                "teacher_projection_retries": int(teacher_retries[i]), "state_complete": True,
            }
            if task["trace"]:
                path = HERE / "traces" / f"{state['state_id']}__{task['condition']}.npz"
                base.atomic_npz(path, **{name: np.asarray(values) for name, values in traces[i].items()})
                row.update({"trace_file": str(path.relative_to(HERE)), "trace_sha256": base.file_hash(path)})
            results.append(row)
            packed.append(log)
            counters["rollouts"] += 1
            counters["steps"] += steps
            counters["active_rows"] += len(log["global_step"])
        return results, packed

    for position, state in enumerate(states):
        output = HERE / "raw" / f"{state['state_id']}.jsonl"
        expected = [HERE / "active_logs" / f"{state['state_id']}__{c}__robust.npz" for c in ("H1", "L8")]
        expected += [HERE / "traces" / f"{state['state_id']}__{c}.npz" for c in ("H1", "L8")]
        if output.is_file():
            old = [json.loads(line) for line in output.read_text().splitlines() if line]
            if len(old) == 130 and all(row.get("state_complete") for row in old) and all(path.is_file() for path in expected):
                print(json.dumps({"skip": state["state_id"]}), flush=True)
                continue
        conditions = ("H1", "L8") if int(state["query_step"]) % 8 else ("H1",)
        rows = []
        for condition in conditions:
            exact, _ = run_tasks(state, [{"condition": condition, "flow_mode": "exact", "trace": True}])
            rows.extend(exact)
            robust_rows, robust_logs = [], []
            tasks = [{"condition": condition, "flow_mode": "robust", "seed": seed, "trace": False} for seed in ROBUST_SEEDS]
            for start in range(0, 64, args.batch):
                batch_rows, batch_logs = run_tasks(state, tasks[start:start + args.batch])
                robust_rows.extend(batch_rows); robust_logs.extend(batch_logs)
            rows.extend(robust_rows)
            base.atomic_npz(HERE / "active_logs" / f"{state['state_id']}__{condition}__robust.npz", **base.concat_logs(robust_logs))
        if conditions == ("H1",):
            for kind in ("traces", "active_logs"):
                suffix = ".npz" if kind == "traces" else "__robust.npz"
                atomic_copy(HERE / kind / f"{state['state_id']}__H1{suffix}", HERE / kind / f"{state['state_id']}__L8{suffix}")
            clones = []
            for row in rows:
                clone = dict(row); clone["condition"] = "L8"; clone["semantic_reuse_from"] = "H1 phase-zero identity"
                if row["flow_mode"] == "exact":
                    path = HERE / "traces" / f"{state['state_id']}__L8.npz"
                    clone.update({"trace_file": str(path.relative_to(HERE)), "trace_sha256": base.file_hash(path)})
                clones.append(clone)
            rows.extend(clones)
        if len(rows) != 130:
            raise RuntimeError((state["state_id"], len(rows)))
        base.atomic_jsonl(output, rows)
        counts = {c: sum(r["condition"] == c and r["flow_mode"] == "robust" and r["outcome"] == "success" for r in rows) for c in ("H1", "L8")}
        print(json.dumps({"completed": state["state_id"], "position": f"{position+1}/{len(states)}", "successes": counts,
                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    runtime = {
        "shard_index": args.shard_index, "shard_count": args.shard_count, "states": len(states),
        "new_rollouts": counters["rollouts"], "physical_steps": counters["steps"],
        "active_timestep_log_rows": counters["active_rows"], "elapsed_seconds": time.monotonic() - started,
        "device": [str(x) for x in jax.devices()], "batch": args.batch, "checkpoint_sha256": model.sha256,
        "phase_zero_semantic_reuse_rollouts": sum(int(s["query_step"]) % 8 == 0 for s in states) * 65,
    }
    (HERE / f"runtime_shard{args.shard_index}.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
