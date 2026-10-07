"""Resumable V3-identical fixed-eta continuation oracle for startup states."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from audit_startup_states import restore_full
from common import (
    HERE, ROOT, SYSROOT, assert_frozen_sources, eta_key,
    load_effective_records, read_jsonl, sha256, tuple_key, write_json, write_jsonl,
)


def outcome_name(env, error) -> str:
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
    parser.add_argument("--arms", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "/" in args.stage or args.stage in (".", ".."):
        raise ValueError("stage must be a simple directory name")
    if Path(args.arms).name != args.arms:
        raise ValueError("arms must name a file in the startup dataset directory")
    hashes = assert_frozen_sources()
    integrity = json.loads((HERE / "startup_state_integrity_checks.json").read_text())
    if integrity["status"] != "PASS":
        raise RuntimeError("startup integrity gate did not pass")
    protocol = json.loads((HERE / "protocol.json").read_text())
    protocol_hash = sha256(HERE / "protocol.json")
    arms_path = HERE / args.arms
    arms = json.loads(arms_path.read_text())
    state_rows = read_jsonl(HERE / "startup_state_manifest.jsonl")
    states = {row["state_id"]: row for row in state_rows}
    if len(states) != len(state_rows):
        raise RuntimeError("duplicate state IDs in startup state manifest")
    allowed_etas = {eta_key((0.0, 0.0, 0.0))}
    allowed_etas.update(eta_key(eta) for eta in protocol["candidate_etas"])
    arm_ids = set()
    state_eta = set()
    frozen_seeds = [int(seed) for seed in protocol["oracle_seed_default"]]
    for arm in arms:
        arm_id = str(arm["arm_id"])
        pair = (str(arm["state_id"]), eta_key(arm["eta"]))
        seeds = [int(seed) for seed in arm["seeds"]]
        if (
            arm_id in arm_ids or pair in state_eta or pair[0] not in states
            or pair[1] not in allowed_etas or seeds != frozen_seeds
        ):
            raise RuntimeError(("invalid arm", arm.get("arm_id")))
        arm_ids.add(arm_id)
        state_eta.add(pair)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "arms": len(arms), "maximum_rollouts": 64 * len(arms), "unique_states": len({a['state_id'] for a in arms}), "arms_sha256": sha256(arms_path), "projection_sha256": hashes[str(SYSROOT / 'single_integrator/cbf.py')]}, indent=2)); return

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    sys.path.insert(0, str(SYSROOT))
    from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, bounded_nominal
    from single_integrator.evaluate import load_policy

    resolved_cbf = Path(inspect.getsourcefile(barrier_constraints)).resolve()
    if resolved_cbf != (SYSROOT / "single_integrator/cbf.py").resolve():
        raise RuntimeError(("wrong projection implementation", resolved_cbf))
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    output = HERE / "raw" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    stage_meta_path = output / "stage_definition.json"
    stage_definition = {"stage": args.stage, "arms_file": args.arms, "arms_sha256": sha256(arms_path), "protocol_sha256": protocol_hash}
    if stage_meta_path.exists() and json.loads(stage_meta_path.read_text()) != stage_definition:
        raise RuntimeError("refusing to resume stage with changed arms/protocol")
    write_json(stage_meta_path, stage_definition)

    all_effective, duplicates = load_effective_records()
    existing_stage = {tuple_key(row): row for row in read_jsonl(records_path)}
    runtime = {arm["arm_id"]: {"records": [], "failures": 0, "stopped": False} for arm in arms}
    arm_by_key = {(arm["state_id"], eta_key(arm["eta"])): arm for arm in arms}
    # Only exact startup state/eta/seed tuples are reusable.  Preserve records
    # produced by earlier isolated stages, but write each tuple only once here.
    for key, row in all_effective.items():
        state_id, eta, seed = key
        arm = arm_by_key.get((state_id, eta))
        if arm is None or seed not in arm["seeds"]:
            continue
        status = runtime[arm["arm_id"]]
        status["records"].append(row)
        status["failures"] += row["outcome"] != "success"
    for status in runtime.values():
        status["records"].sort(key=lambda row: int(row["seed"]))
        status["stopped"] = status["failures"] >= 2 or len(status["records"]) >= 64

    new_records = list(existing_stage.values())
    new_keys = set(existing_stage)
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single)); fold = jax.jit(jax.vmap(jax.random.fold_in))
    started = time.monotonic(); physical_steps = sum(int(row.get("steps", 0)) for row in new_records)

    for seed_index in range(64):
        tasks = []
        for arm in arms:
            status = runtime[arm["arm_id"]]
            seed = int(arm["seeds"][seed_index])
            if status["stopped"] or any(int(row["seed"]) == seed for row in status["records"]):
                continue
            tasks.append({**arm, "seed": seed, "seed_index": seed_index})
        for start in range(0, len(tasks), args.batch):
            chunk = tasks[start:start + args.batch]
            if not chunk:
                continue
            envs = []; correctors = []; keys = []; errors = [None] * len(chunk)
            first = [None] * len(chunk); sums = np.zeros(len(chunk), dtype=np.float64)
            for task in chunk:
                state = states[task["state_id"]]
                envs.append(restore_full(HERE / state["state_file"], config))
                correctors.append(DiagnosticCorrector(DiagnosticPhi(*task["eta"])))
                keys.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), int(state["rng_namespace"]))))
            padded = list(keys)
            while len(padded) < args.batch:
                padded.append(padded[-1])
            keys0 = jnp.asarray(np.asarray(padded))
            observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
            steps = np.zeros(args.batch, dtype=np.uint32)
            while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
                for i, env in enumerate(envs):
                    observations[i] = env.observation(); steps[i] = env.step_count
                actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
                for i, env in enumerate(envs):
                    if env.done or errors[i] is not None:
                        continue
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    try:
                        safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        raw = correctors[i](observations[i], safe, config.max_speed)
                        executed, second_status, second_retry, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
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
                state = states[task["state_id"]]
                row = {
                    "arm_id": task["arm_id"], "state_id": task["state_id"], "eta": task["eta"],
                    "seed": task["seed"], "seed_index": task["seed_index"],
                    "outcome": outcome_name(envs[i], errors[i]), "execution_error": errors[i],
                    "steps": int(envs[i].step_count - state["step"]), "terminal_step": int(envs[i].step_count),
                    "J_def": float(config.dt * sums[i]), "first_step": first[i],
                    "rng_namespace": int(state["rng_namespace"]), "state_sha256": state["state_sha256"],
                }
                key = tuple_key(row)
                if key not in new_keys:
                    new_records.append(row); new_keys.add(key); physical_steps += row["steps"]
                status = runtime[task["arm_id"]]
                status["records"].append(row)
                status["failures"] += row["outcome"] != "success"
                status["stopped"] = status["failures"] >= 2 or len(status["records"]) >= 64
            # Durable checkpoint after every batch.
            write_jsonl(records_path, sorted(new_records, key=lambda row: (row["state_id"], eta_key(row["eta"]), int(row["seed"]))))
        print(json.dumps({"seed_round": seed_index + 1, "new_records_in_stage": len(new_records), "active_arms": sum(not value["stopped"] for value in runtime.values()), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    arm_results = []
    for arm in arms:
        rows = runtime[arm["arm_id"]]["records"]
        counts = Counter(row["outcome"] for row in rows)
        arm_results.append({
            "arm_id": arm["arm_id"], "state_id": arm["state_id"], "eta": arm["eta"],
            "evaluated": len(rows), "success": counts["success"], "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"], "execution_error": counts["execution_error"],
            "status": "complete64" if len(rows) == 64 else "early_stop_two_failures",
            "B_63_member": len(rows) == 64 and counts["success"] >= 63,
        })
    manifest = {
        **stage_definition, "status": "COMPLETE", "device": [str(device) for device in jax.devices()],
        "batch_size": args.batch, "records_in_stage_cache": len(new_records), "physical_steps_in_stage_cache": physical_steps,
        "elapsed_s_this_invocation": time.monotonic() - started, "preexisting_duplicate_cache_rows": len(duplicates),
        "records_sha256": sha256(records_path), "frozen_hashes": hashes, "arm_results": arm_results,
    }
    write_json(output / "manifest.json", manifest)
    if sha256(HERE / "protocol.json") != protocol_hash:
        raise RuntimeError("protocol changed during oracle execution")
    print(json.dumps({key: value for key, value in manifest.items() if key != "arm_results"}, indent=2))


if __name__ == "__main__":
    main()
