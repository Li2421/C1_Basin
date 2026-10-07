"""Adaptive original finite fixed-eta search on the frozen unbiased fresh32 step-0 states."""

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
SYS = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
PLAN_PATH = HERE / "fresh32_oracle_plan.json"
ASSETS_PATH = HERE / "frozen_asset_hashes.json"
sys.path[:0] = [str(SYS), str(ROOT)]

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_content(value: dict) -> None:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != value["content_sha256"]:
        raise RuntimeError("semantic hash mismatch")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    os.replace(temporary, path)


def classify(env: GiveWayEnv, error: dict | None) -> str:
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
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)

    plan = json.loads(PLAN_PATH.read_text())
    assets = json.loads(ASSETS_PATH.read_text())
    verify_content(plan)
    verify_content(assets)
    for record in assets.values():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            if sha(Path(record["path"])) != record["sha256"]:
                raise RuntimeError((record["path"], "asset hash mismatch"))
    source = json.loads((ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/fresh_wide_manifest.json").read_text())
    config = Config(**source["environment"])
    cbf = CBFConfig(**source["cbf"])
    policy, provenance = load_policy(Path(assets["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != source["environment"]:
        raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))

    output = HERE / "raw/fresh32_oracle"
    records_path = output / "records.jsonl"
    existing = [] if not records_path.exists() else [json.loads(line) for line in records_path.read_text().splitlines()]
    if any(row.get("plan_sha256") != sha(PLAN_PATH) for row in existing):
        raise RuntimeError("existing records use a different plan")
    runtime = {
        arm["arm_id"]: {"records": [], "failures": 0, "stopped": False}
        for arm in plan["arms"]
    }
    by_arm = {arm["arm_id"]: arm for arm in plan["arms"]}
    for row in existing:
        status = runtime[row["arm_id"]]
        status["records"].append(row)
        status["failures"] += row["outcome"] != "success"
    for status in runtime.values():
        status["records"].sort(key=lambda row: int(row["seed_index"]))
        status["stopped"] = status["failures"] >= 2 or len(status["records"]) >= 64

    rows = list(existing)
    existing_keys = {(row["arm_id"], int(row["seed"])) for row in existing}
    stage_cap = int(plan["global_new_rollout_cap_for_this_stage"])
    begun = time.monotonic()
    new_rollouts = 0
    new_steps = 0
    cap_reached = False
    for seed_index in range(64):
        tasks = []
        for arm in plan["arms"]:
            status = runtime[arm["arm_id"]]
            seed = int(arm["seeds"][seed_index])
            if status["stopped"] or (arm["arm_id"], seed) in existing_keys:
                continue
            tasks.append({**arm, "seed": seed, "seed_index": seed_index})
        # Fair, outcome-independent ordering: every episode receives the same
        # candidate priority before later candidates consume a final capped
        # seed round.  Existing rows count against the immutable stage cap on
        # resumptions.
        tasks.sort(key=lambda task: (int(task["candidate_priority"]), int(task["episode_index"])))
        remaining = stage_cap - len(rows)
        if remaining <= 0:
            cap_reached = True
            break
        if len(tasks) > remaining:
            tasks = tasks[:remaining]
            cap_reached = True
        for begin in range(0, len(tasks), args.batch):
            chunk = tasks[begin : begin + args.batch]
            envs = [GiveWayEnv(config) for _ in chunk]
            correctors = []
            keys = []
            errors: list[dict | None] = [None] * len(chunk)
            jdef = np.zeros(len(chunk), dtype=np.float64)
            first_retry = np.zeros(len(chunk), dtype=int)
            second_retry = np.zeros(len(chunk), dtype=int)
            for task, env in zip(chunk, envs):
                env.reset(np.asarray(task["initial_positions"], dtype=np.float64))
                correctors.append(DiagnosticCorrector(DiagnosticPhi(*[float(x) for x in task["eta"]])))
                keys.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), int(task["rng_namespace"]))))
            padded = list(keys)
            while len(padded) < args.batch:
                padded.append(padded[-1])
            key0 = jnp.asarray(np.asarray(padded))
            observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
            steps = np.zeros(args.batch, dtype=np.uint32)
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
                        safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        raw = np.asarray(correctors[index](observations[index], safe, config.max_speed), dtype=np.float64)
                        executed, _, retry2, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
                        jdef[index] += config.dt * float(np.sum((executed - safe) ** 2))
                        first_retry[index] += int(retry1)
                        second_retry[index] += int(retry2)
                        env.step(executed)
                        new_steps += 1
                    except Exception as exc:
                        errors[index] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "step": int(env.step_count),
                        }
            for index, (task, env) in enumerate(zip(chunk, envs)):
                result = classify(env, errors[index])
                row = {
                    "schema": "fresh32_original_finite_oracle_rollout_v1",
                    "arm_id": task["arm_id"],
                    "episode_index": int(task["episode_index"]),
                    "source_id": task["source_id"],
                    "candidate_priority": int(task["candidate_priority"]),
                    "eta": task["eta"],
                    "seed": int(task["seed"]),
                    "seed_index": int(task["seed_index"]),
                    "rng_namespace": int(task["rng_namespace"]),
                    "outcome": result,
                    "success": result == "success",
                    "physical_steps": int(env.step_count),
                    "terminal_step": int(env.step_count),
                    "J_def": float(jdef[index]),
                    "first_projection_retries": int(first_retry[index]),
                    "second_projection_retries": int(second_retry[index]),
                    "execution_error": errors[index],
                    "plan_sha256": sha(PLAN_PATH),
                    "assets_sha256": sha(ASSETS_PATH),
                }
                rows.append(row)
                existing_keys.add((row["arm_id"], row["seed"]))
                status = runtime[row["arm_id"]]
                status["records"].append(row)
                status["failures"] += result != "success"
                status["stopped"] = status["failures"] >= 2 or len(status["records"]) >= 64
                new_rollouts += 1
            write_jsonl(records_path, rows)
        print(
            json.dumps(
                {
                    "seed_round": seed_index + 1,
                    "new_rollouts": new_rollouts,
                    "new_steps": new_steps,
                    "active_arms": sum(
                        not status["stopped"] and len(status["records"]) < 64
                        for status in runtime.values()
                    ),
                }
            ),
            flush=True,
        )
        if cap_reached:
            break

    arm_results = []
    for arm_id, status in runtime.items():
        arm = by_arm[arm_id]
        counts = Counter(row["outcome"] for row in status["records"])
        arm_results.append(
            {
                "arm_id": arm_id,
                "episode_index": arm["episode_index"],
                "candidate_priority": arm["candidate_priority"],
                "eta": arm["eta"],
                "evaluated": len(status["records"]),
                "success": counts["success"],
                "deadlock": counts["deadlock"],
                "timeout": counts["timeout"],
                "collision": counts["collision"],
                "execution_error": counts["execution_error"],
                "status": "complete64" if len(status["records"]) == 64 else "early_stop_two_failures" if status["failures"] >= 2 else "incomplete_budget_cap",
                "B_63_member": len(status["records"]) == 64 and counts["success"] >= 63,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "fresh32_original_finite_oracle_runtime_v1",
        "status": "BUDGET_CAP_REACHED" if cap_reached else "COMPLETE",
        "device": [str(value) for value in jax.devices()],
        "batch": args.batch,
        "new_rollouts": new_rollouts,
        "new_physical_steps": new_steps,
        "wall_seconds": time.monotonic() - begun,
        "records": len(rows),
        "records_sha256": sha(records_path) if records_path.exists() else None,
        "plan_sha256": sha(PLAN_PATH),
        "assets_sha256": sha(ASSETS_PATH),
        "host": platform.node(),
        "arm_results": arm_results,
    }
    (output / "runtime.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: val for key, val in manifest.items() if key != "arm_results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
