"""Adaptive batched full-horizon oracle rollout with exact two-failure stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--reuse-jsonl")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    protocol = json.loads((HERE / "protocol.json").read_text())
    protocol_hash = sha(HERE / "protocol.json")
    arms = json.loads((HERE / args.arms).read_text())
    state_rows = {row["state_id"]: row for row in (json.loads(line) for line in (HERE / "state_manifest.jsonl").read_text().splitlines())}
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    reused = {}
    if args.reuse_jsonl:
        for line in Path(args.reuse_jsonl).read_text().splitlines():
            row = json.loads(line); reused[(row["state_id"], tuple(row["eta"]), int(row["seed"]))] = row
    output = HERE / "raw" / args.stage
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    print(json.dumps({
        "stage": args.stage, "maximum_rollout_count": 64 * len(arms),
        "maximum_physical_steps": sum((config.max_steps - state_rows[arm["state_id"]]["step"]) * len(arm["seeds"]) for arm in arms),
        "CPU_GPU": args.device, "purpose": "63/64 oracle feasibility; exact stop at second physical failure",
    }), flush=True)

    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single)); fold = jax.jit(jax.vmap(jax.random.fold_in))
    runtime = {arm["arm_id"]: {"records": [], "failures": 0, "stopped": False} for arm in arms}
    all_records = []
    reused_count = 0; new_count = 0; physical_steps = 0
    started = time.monotonic(); last = started

    def consume(arm, record):
        nonlocal reused_count, new_count, physical_steps
        runtime[arm["arm_id"]]["records"].append(record)
        if record["outcome"] != "success":
            runtime[arm["arm_id"]]["failures"] += 1
        if runtime[arm["arm_id"]]["failures"] >= 2:
            runtime[arm["arm_id"]]["stopped"] = True
        if record.get("reused", False): reused_count += 1
        else:
            new_count += 1; physical_steps += int(record["steps"]); all_records.append(record)

    for seed_index in range(64):
        tasks = []
        for arm in arms:
            status = runtime[arm["arm_id"]]
            if status["stopped"] or len(status["records"]) >= 64:
                continue
            seed = int(arm["seeds"][seed_index])
            key = (arm["state_id"], tuple(arm["eta"]), seed)
            if key in reused:
                record = {**reused[key], "arm_id": arm["arm_id"], "reused": True}
                consume(arm, record)
            else:
                tasks.append({**arm, "seed": seed, "seed_index": seed_index})
        for chunk_start in range(0, len(tasks), args.batch):
            chunk = tasks[chunk_start:chunk_start + args.batch]
            envs, correctors, key0, errors = [], [], [], [None] * len(chunk)
            first = [None] * len(chunk); sums = np.zeros(len(chunk), dtype=np.float64)
            for task in chunk:
                state = state_rows[task["state_id"]]
                envs.append(restore_full(HERE / state["state_file"], config))
                correctors.append(DiagnosticCorrector(DiagnosticPhi(*task["eta"])))
                key0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), int(state["rng_namespace"]))))
            while len(key0) < args.batch:
                key0.append(key0[-1])
            keys0 = jnp.asarray(np.asarray(key0)); observations = np.zeros((args.batch, 2, 10), dtype=np.float32); steps = np.zeros(args.batch, dtype=np.uint32)
            while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
                for i, env in enumerate(envs):
                    observations[i] = env.observation(); steps[i] = env.step_count
                actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
                for i, env in enumerate(envs):
                    if env.done or errors[i] is not None: continue
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    try:
                        safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        raw = correctors[i](observations[i], safe, config.max_speed)
                        w = safe + raw
                        executed, second_status, second_retry, _ = project_velocity_with_retry(w, A, lower, config.max_speed, cbf)
                    except Exception as exc:
                        errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": env.step_count}; continue
                    if first[i] is None:
                        first[i] = {
                            "u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(),
                            "g_raw": raw.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(),
                            "first_projection_status": str(first_status), "second_projection_status": str(second_status),
                            "first_retry": bool(first_retry), "second_retry": bool(second_retry),
                            "first_min_linear_residual": float(np.min(A @ safe.reshape(4) - lower)),
                            "second_min_linear_residual": float(np.min(A @ executed.reshape(4) - lower)),
                        }
                    sums[i] += float(np.sum((executed - safe) ** 2, dtype=np.float64))
                    env.step(executed)
            for i, task in enumerate(chunk):
                if errors[i] is None:
                    summary = envs[i].summary()
                    outcome = "collision" if summary["wall_collision"] or summary["agent_collision"] else "success" if summary["success"] else "deadlock" if summary["deadlock"] else "timeout"
                else:
                    outcome = "execution_error"
                record = {
                    "arm_id": task["arm_id"], "state_id": task["state_id"], "eta": task["eta"],
                    "seed": task["seed"], "seed_index": task["seed_index"], "outcome": outcome,
                    "execution_error": errors[i], "steps": int(envs[i].step_count - state_rows[task["state_id"]]["step"]),
                    "terminal_step": int(envs[i].step_count), "J_def": float(config.dt * sums[i]),
                    "first_step": first[i], "reused": False,
                }
                consume(task, record)
        # New records only; durable, isolated cache after each matched-seed round.
        records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_records))
        if time.monotonic() - last > 20 or seed_index == 63:
            print(json.dumps({
                "seed_round_completed": seed_index + 1, "new_rollouts": new_count, "reused": reused_count,
                "active_arms": sum(not value["stopped"] and len(value["records"]) < 64 for value in runtime.values()),
                "outcomes": dict(Counter(row["outcome"] for row in all_records)), "elapsed_s": round(time.monotonic() - started, 1),
            }), flush=True); last = time.monotonic()

    arm_results = []
    for arm in arms:
        rows = runtime[arm["arm_id"]]["records"]; counts = Counter(row["outcome"] for row in rows)
        arm_results.append({
            "arm_id": arm["arm_id"], "state_id": arm["state_id"], "eta": arm["eta"],
            "evaluated": len(rows), "success": counts["success"], "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"], "execution_error": counts["execution_error"],
            "status": "complete64" if len(rows) == 64 else "early_stop_two_failures",
            "B_63_member": len(rows) == 64 and counts["success"] >= 63,
            "new_rollouts": sum(not row.get("reused", False) for row in rows),
            "reused_rollouts": sum(row.get("reused", False) for row in rows),
        })
    manifest = {
        "stage": args.stage, "device": [str(device) for device in jax.devices()], "batch_size": args.batch,
        "new_rollouts": new_count, "reused_rollouts": reused_count, "physical_steps": physical_steps,
        "elapsed_s": time.monotonic() - started, "protocol_sha256": protocol_hash,
        "arms_sha256": sha(HERE / args.arms), "records_sha256": sha(records_path), "arm_results": arm_results,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if sha(HERE / "protocol.json") != protocol_hash:
        raise RuntimeError("protocol changed during stage")
    print(json.dumps({key: value for key, value in manifest.items() if key != "arm_results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
