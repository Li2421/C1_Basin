"""Batched Give-Way fixed-OrthoFlow3 continuation runner for frozen plans."""

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
SYS = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
sys.path[:0] = [str(SYS), str(ROOT)]

from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from shared_control.basis_families import get_basis_family  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_content(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    observed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if observed != value.get("content_sha256"):
        raise RuntimeError("plan semantic hash mismatch")


def atomic_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    os.replace(temporary, path)


def classify(env, error) -> str:
    if error is not None:
        return "execution_error"
    result = env.summary()
    if result["wall_collision"] or result["agent_collision"]:
        return "collision"
    if result["success"]:
        return "success"
    if result["deadlock"]:
        return "deadlock"
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    verify_content(plan)
    if plan["basis_family"] != "orthoflow3":
        raise RuntimeError("new migration plans must select orthoflow3 explicitly")
    basis = get_basis_family(plan["basis_family"])
    config_source = json.loads(
        (ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text()
    )
    config = Config(**config_source["environment"])
    cbf = CBFConfig(**config_source["cbf"])
    checkpoint = SYS / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    policy, provenance = load_policy(checkpoint)
    if not provenance or provenance["evaluation_environment"] != config_source["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample = jax.jit(
        jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    )
    fold = jax.jit(jax.vmap(jax.random.fold_in))

    tasks = []
    for arm in plan["arms"]:
        if arm.get("basis_family") != plan["basis_family"]:
            raise RuntimeError("arm basis-family mismatch")
        for seed in arm["seeds"]:
            tasks.append((arm, int(seed)))
    tasks = [task for index, task in enumerate(tasks) if index % args.shards == args.shard]
    output = HERE / "raw" / args.stage
    records_path = output / f"shard{args.shard}.jsonl"
    runtime_path = output / f"shard{args.shard}_runtime.json"
    existing = [] if not records_path.exists() else [
        json.loads(line) for line in records_path.read_text().splitlines() if line.strip()
    ]
    completed = {(row["arm_id"], int(row["seed"])) for row in existing}
    pending = [(arm, seed) for arm, seed in tasks if (arm["arm_id"], seed) not in completed]
    rows = list(existing)
    begun = time.monotonic()
    new_rollouts = 0
    new_steps = 0
    for begin in range(0, len(pending), args.batch):
        chunk = pending[begin : begin + args.batch]
        n = len(chunk)
        envs = []
        etas = []
        errors = [None] * n
        keys = []
        for arm, seed in chunk:
            state_path = Path(arm["state_file"])
            if sha(state_path) != arm["state_sha256"]:
                raise RuntimeError((arm["state_id"], "state hash mismatch"))
            env = restore_full(state_path, config)
            if env.done or env.step_count != int(arm["absolute_step"]):
                raise RuntimeError((arm["state_id"], "bad restored state"))
            envs.append(env)
            etas.append(np.asarray(arm["eta"], dtype=np.float64))
            keys.append(
                np.asarray(
                    jax.random.fold_in(jax.random.PRNGKey(seed), int(arm["rng_namespace"]))
                )
            )
        padded_keys = list(keys)
        while len(padded_keys) < args.batch:
            padded_keys.append(padded_keys[-1])
        key0 = jnp.asarray(np.asarray(padded_keys))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        steps = np.zeros(args.batch, dtype=np.uint32)
        jdef = np.zeros(n, dtype=np.float64)
        first_retry = np.zeros(n, dtype=np.int64)
        second_retry = np.zeros(n, dtype=np.int64)
        first_step = [None] * n
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            for index, env in enumerate(envs):
                if not env.done and errors[index] is None:
                    observations[index] = env.observation()
                    steps[index] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(steps))))
            for index, env in enumerate(envs):
                if env.done or errors[index] is not None:
                    continue
                try:
                    flow = bounded_nominal(actions[index], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, first_status, retry1, _ = project_velocity_with_retry(
                        flow, A, lower, config.max_speed, cbf
                    )
                    fields = basis.compute(env.positions, env.goals, safe, config.max_speed)
                    correction = fields.correction(etas[index])
                    executed, second_status, retry2, _ = project_velocity_with_retry(
                        safe + correction, A, lower, config.max_speed, cbf
                    )
                    if first_step[index] is None:
                        first_step[index] = {
                            "u_safe": safe.reshape(-1).tolist(),
                            "basis_values": [value.reshape(-1).tolist() for value in fields.values],
                            "g_raw": correction.reshape(-1).tolist(),
                            "u_exec": executed.reshape(-1).tolist(),
                            "first_projection_status": str(first_status),
                            "second_projection_status": str(second_status),
                        }
                    jdef[index] += config.dt * float(np.sum((executed - safe) ** 2))
                    first_retry[index] += int(retry1)
                    second_retry[index] += int(retry2)
                    env.step(executed)
                    new_steps += 1
                except Exception as error:
                    errors[index] = {
                        "type": type(error).__name__, "message": str(error),
                        "step": int(env.step_count),
                    }
        for index, ((arm, seed), env) in enumerate(zip(chunk, envs, strict=True)):
            outcome = classify(env, errors[index])
            rows.append(
                {
                    "schema": "orthoflow3_migration_rollout_v1",
                    "stage": args.stage,
                    "basis_family": plan["basis_family"],
                    "basis_version": basis.metadata.version,
                    "arm_id": arm["arm_id"], "state_id": arm["state_id"],
                    "eta_index": arm.get("eta_index"), "parameter_id": arm.get("parameter_id"),
                    "eta": arm["eta"], "seed": seed, "rng_namespace": arm["rng_namespace"],
                    "outcome": outcome, "success": outcome == "success",
                    "continuation_steps": int(env.step_count - int(arm["absolute_step"])),
                    "terminal_step": int(env.step_count), "J_def": float(jdef[index]),
                    "execution_error": errors[index], "first_step": first_step[index],
                    "first_projection_retries": int(first_retry[index]),
                    "second_projection_retries": int(second_retry[index]),
                    "plan_sha256": sha(plan_path),
                }
            )
            new_rollouts += 1
        atomic_rows(records_path, rows)
        if begin == 0 or (begin // args.batch + 1) % 20 == 0 or begin + args.batch >= len(pending):
            print(json.dumps({"stage": args.stage, "shard": args.shard, "done": min(begin + args.batch, len(pending)), "pending": len(pending), "new_steps": new_steps}), flush=True)
    counts = Counter(row["outcome"] for row in rows)
    runtime = {
        "schema": "orthoflow3_migration_runtime_v1", "stage": args.stage,
        "basis_family": plan["basis_family"], "device": [str(value) for value in jax.devices()],
        "shard": args.shard, "shards": args.shards, "batch": args.batch,
        "new_rollouts": new_rollouts, "new_physical_steps": new_steps,
        "wall_seconds": time.monotonic() - begun, "records": len(rows),
        "outcomes": dict(counts), "records_sha256": sha(records_path),
        "plan_sha256": sha(plan_path), "host": platform.node(),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
