"""Resume-safe fixed-OrthoFlow3 continuation runner for continuity plans."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_local_basin_continuity_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path[:0] = [str(SYSROOT), str(ROOT)]

from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from shared_control.basis_families import get_basis_family  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_plan(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    observed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if observed != value.get("content_sha256"):
        raise RuntimeError("plan semantic hash mismatch")


def atomic_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    os.replace(temporary, path)


def outcome(env, error) -> str:
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
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "gpu")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    if plan.get("basis_family") != "orthoflow3":
        raise RuntimeError("only frozen orthoflow3 plans are valid")
    basis = get_basis_family("orthoflow3")
    integrity = json.loads((ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text())
    config, cbf = Config(**integrity["environment"]), CBFConfig(**integrity["cbf"])
    policy, provenance = load_policy(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    if provenance["evaluation_environment"] != integrity["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))

    tasks = [(arm, int(seed)) for arm in plan["arms"] for seed in arm["seeds"]]
    tasks = [item for index, item in enumerate(tasks) if index % args.shards == args.shard]
    output = HERE / "raw" / args.stage
    record_path = output / f"shard{args.shard}.jsonl"
    existing = [] if not record_path.exists() else [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    seen = {(row["arm_id"], int(row["seed"])) for row in existing}
    pending = [(arm, seed) for arm, seed in tasks if (arm["arm_id"], seed) not in seen]
    rows = list(existing)
    started = time.monotonic(); new_steps = 0; new_rollouts = 0
    for begin in range(0, len(pending), args.batch):
        chunk = pending[begin:begin + args.batch]
        envs, keys, etas, errors = [], [], [], [None] * len(chunk)
        for arm, seed in chunk:
            state_path = Path(arm["state_file"])
            if sha(state_path) != arm["state_sha256"]:
                raise RuntimeError((arm["arm_id"], "state hash mismatch"))
            env = restore_full(state_path, config)
            if env.done or env.step_count != int(arm["absolute_step"]):
                raise RuntimeError((arm["arm_id"], "bad restored state"))
            envs.append(env); etas.append(np.asarray(arm["eta"], dtype=np.float64))
            keys.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed), int(arm["rng_namespace"]))))
        padded = list(keys)
        while len(padded) < args.batch:
            padded.append(padded[-1])
        key0 = jnp.asarray(np.asarray(padded))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        step_array = np.zeros(args.batch, dtype=np.uint32)
        jdef = np.zeros(len(chunk), dtype=np.float64)
        first_retry = np.zeros(len(chunk), dtype=np.int64)
        second_retry = np.zeros(len(chunk), dtype=np.int64)
        first_step = [None] * len(chunk)
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            for index, env in enumerate(envs):
                if not env.done and errors[index] is None:
                    observations[index] = env.observation(); step_array[index] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(step_array))))
            for index, env in enumerate(envs):
                if env.done or errors[index] is not None:
                    continue
                try:
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, status1, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    fields = basis.compute(env.positions, env.goals, safe, config.max_speed)
                    correction = fields.correction(etas[index])
                    executed, status2, retry2, _ = project_velocity_with_retry(safe + correction, A, lower, config.max_speed, cbf)
                    if first_step[index] is None:
                        first_step[index] = {
                            "u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(),
                            "g_raw": correction.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(),
                            "first_projection_status": str(status1), "second_projection_status": str(status2),
                        }
                    jdef[index] += config.dt * float(np.sum((executed - safe) ** 2))
                    first_retry[index] += int(retry1); second_retry[index] += int(retry2)
                    env.step(executed); new_steps += 1
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
        for index, ((arm, seed), env) in enumerate(zip(chunk, envs, strict=True)):
            label = outcome(env, errors[index])
            rows.append({
                "schema": "orthoflow3_local_basin_rollout_v1", "stage": args.stage,
                "arm_id": arm["arm_id"], "state_id": arm["state_id"], "role": arm["role"],
                    # Screen-plan anchors predate the explicit offset field;
                    # their semantic offset is unambiguously zero.  Neighbor
                    # and promotion plans carry their actual signed offset.
                    "anchor_rank": arm["anchor_rank"], "offset_steps": arm.get("offset_steps", 0),
                "probe_id": arm.get("probe_id"), "eta": arm["eta"], "seed": seed,
                "rng_namespace": arm["rng_namespace"], "outcome": label, "success": label == "success",
                "continuation_steps": int(env.step_count - int(arm["absolute_step"])), "terminal_step": int(env.step_count),
                "J_def": float(jdef[index]), "execution_error": errors[index], "first_step": first_step[index],
                "first_projection_retries": int(first_retry[index]), "second_projection_retries": int(second_retry[index]),
                "plan_sha256": sha(plan_path),
            })
            new_rollouts += 1
        atomic_rows(record_path, rows)
        if begin == 0 or begin + args.batch >= len(pending) or (begin // args.batch + 1) % 20 == 0:
            print(json.dumps({"stage": args.stage, "shard": args.shard, "done": min(begin + args.batch, len(pending)), "pending": len(pending), "new_steps": new_steps}), flush=True)
    runtime = {
        "schema": "orthoflow3_local_basin_runtime_v1", "stage": args.stage, "shard": args.shard, "shards": args.shards,
        "device": [str(item) for item in jax.devices()], "new_rollouts": new_rollouts, "new_physical_steps": new_steps,
        "wall_seconds": time.monotonic() - started, "records": len(rows), "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "records_sha256": sha(record_path), "plan_sha256": sha(plan_path), "host": platform.node(),
    }
    (output / f"shard{args.shard}_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
