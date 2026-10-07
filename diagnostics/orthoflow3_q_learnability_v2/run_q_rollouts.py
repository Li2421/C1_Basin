#!/usr/bin/env python3
"""Run fixed-current-Flow, future-randomized OrthoFlow3 Q continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/orthoflow3_q_learnability_v2"
for value in (str(SYSROOT), str(ROOT), str(ROOT / "diagnostics/gphi_training_dataset_v2")):
    if value not in sys.path:
        sys.path.insert(0, value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(env, error):
    if error is not None:
        return "other_numerical"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "safe_deadlock"
    return "timeout"


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="base_rollout_plan.json")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="resume-safe smoke-test limit")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "gpu")

    from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from shared_control.basis_families import get_basis_family
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, bounded_nominal
    from single_integrator.evaluate import load_policy

    plan_path = HERE / args.plan
    plan = json.loads(plan_path.read_text())
    if plan["orthoflow3_sha256"] != sha(ROOT / "diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py"):
        raise RuntimeError("OrthoFlow3 hash mismatch")
    state_manifest = json.loads((HERE / "eligible_state_manifest.json").read_text())
    states = {row["state_id"]: row for row in state_manifest["selected_states"]}
    archived_h = np.load(HERE / "conditioning_features.npz")["features"]
    integrity = json.loads((ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text())
    config, cbf = Config(**integrity["environment"]), CBFConfig(**integrity["cbf"])
    policy, provenance = load_policy(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    if provenance["evaluation_environment"] != integrity["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]))
    basis = get_basis_family("orthoflow3")

    tasks = [row for index, row in enumerate(plan["tasks"]) if index % args.shards == args.shard]
    raw_dir = HERE / "raw" / Path(args.plan).stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_path = raw_dir / f"shard{args.shard}.jsonl"
    rows = [] if not record_path.exists() else [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    # Global durable lookup makes a deliberate shard-count restart resume-safe.
    # This is needed only after stopping the initial two-shard timing batch; no
    # writers from a different allocation may be live at the same time.
    global_seen = set()
    for prior_path in raw_dir.glob("shard*.jsonl"):
        if not prior_path.stem.removeprefix("shard").isdigit():
            continue
        global_seen.update(json.loads(line)["task_id"] for line in prior_path.read_text().splitlines() if line.strip())
    pending = [row for row in tasks if row["task_id"] not in global_seen]
    if args.limit is not None:
        pending = pending[:args.limit]
    started = time.monotonic(); new_steps = 0; max_h_error = 0.0
    for begin in range(0, len(pending), args.batch):
        chunk = pending[begin:begin + args.batch]
        envs, etas, current_keys, future_keys, errors = [], [], [], [], [None] * len(chunk)
        first_data = [None] * len(chunk)
        for task in chunk:
            state = states[task["state_id"]]
            state_path = Path(state["state_file"])
            if sha(state_path) != state["state_sha256"]:
                raise RuntimeError((task["task_id"], "state hash"))
            env = restore_full(state_path, config)
            if env.done or int(env.step_count) != state["absolute_step"]:
                raise RuntimeError((task["task_id"], "bad restored state"))
            base = jax.random.fold_in(jax.random.PRNGKey(state["flow_seed"]), state["rng_namespace"])
            current_keys.append(np.asarray(jax.random.fold_in(base, state["absolute_step"])))
            state_token = int(hashlib.sha256(task["state_id"].encode()).hexdigest()[:8], 16)
            future = jax.random.fold_in(jax.random.PRNGKey(plan["future_root_seed"]), state_token)
            future_keys.append(np.asarray(jax.random.fold_in(future, int(task["future_index"]))))
            envs.append(env); etas.append(np.asarray(task["eta"], dtype=np.float64))
        jdef = np.zeros(len(chunk), dtype=np.float64)
        while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
            active = [i for i, env in enumerate(envs) if not env.done and errors[i] is None]
            observations = np.stack([np.asarray(envs[i].observation(), dtype=np.float32) for i in active])
            keys = []
            for i in active:
                state = states[chunk[i]["state_id"]]
                step = int(envs[i].step_count)
                keys.append(current_keys[i] if step == state["absolute_step"] else np.asarray(jax.random.fold_in(jnp.asarray(future_keys[i]), step)))
            actions = np.asarray(sample(jnp.asarray(observations), jnp.asarray(np.stack(keys))))
            for local, i in enumerate(active):
                env = envs[i]; state = states[chunk[i]["state_id"]]
                try:
                    flow = bounded_nominal(actions[local], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, status1, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    fields = basis.compute(env.positions, env.goals, safe, config.max_speed)
                    correction = fields.correction(etas[i])
                    executed, status2, retry2, _ = project_velocity_with_retry(safe + correction, A, lower, config.max_speed, cbf)
                    if first_data[i] is None:
                        feature, _ = StartupAwareFeatureBuilder().build(FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf)
                        expected = archived_h[state["feature_index"]]
                        h_error = float(np.max(np.abs(feature - expected))); max_h_error = max(max_h_error, h_error)
                        if h_error > 1e-10:
                            raise RuntimeError((chunk[i]["task_id"], "conditioning feature replay", h_error))
                        first_data[i] = {
                            "feature_sha256": hashlib.sha256(np.asarray(feature, dtype=np.float64).tobytes()).hexdigest(),
                            "u_flow": np.asarray(flow).reshape(-1).tolist(), "u_safe": np.asarray(safe).reshape(-1).tolist(),
                            "current_flow_key_data": np.asarray(current_keys[i], dtype=np.uint32).tolist(),
                            "first_projection_status": str(status1), "second_projection_status": str(status2),
                            "first_projection_retry": bool(retry1), "second_projection_retry": bool(retry2),
                        }
                    jdef[i] += config.dt * float(np.sum((executed - safe) ** 2))
                    env.step(executed); new_steps += 1
                except Exception as exc:
                    errors[i] = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
        for i, (task, env) in enumerate(zip(chunk, envs, strict=True)):
            label = outcome(env, errors[i])
            rows.append({
                **task, "source_group": states[task["state_id"]]["source_group"],
                "h_conditioning_identifier": states[task["state_id"]]["h_conditioning_identifier"],
                "outcome": label, "success": label == "success",
                "continuation_steps": int(env.step_count - states[task["state_id"]]["absolute_step"]),
                "terminal_step": int(env.step_count), "J_def": float(jdef[i]),
                "execution_error": errors[i], "first_step": first_data[i],
                "plan_sha256": sha(plan_path),
            })
        atomic_jsonl(record_path, rows)
        if begin == 0 or begin + args.batch >= len(pending) or (begin // args.batch + 1) % 25 == 0:
            print(json.dumps({"shard": args.shard, "done": min(begin + args.batch, len(pending)), "pending": len(pending), "steps": new_steps}), flush=True)
    runtime = {
        "schema": "orthoflow3_q_v2_rollout_runtime", "plan": args.plan,
        "shard": args.shard, "shards": args.shards, "new_rollouts": len(pending),
        "new_physical_steps": new_steps, "wall_seconds": time.monotonic() - started,
        "max_feature_replay_error": max_h_error, "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "devices": [str(device) for device in jax.devices()], "host": platform.node(),
        "records_sha256": sha(record_path), "plan_sha256": sha(plan_path),
    }
    (raw_dir / f"shard{args.shard}_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
