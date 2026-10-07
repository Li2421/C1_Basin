"""Batched fixed-eta exact-flow search and 64-seed robust validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_closed_loop_pilot_v1.pilot_common import sha256, write_json
from diagnostics.gphi_training_dataset_v1.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_CHECKPOINT = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
DOMAIN_LOW = np.asarray([0.5, -0.5, 0.0], dtype=np.float64)
DOMAIN_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
COARSE = [
    tuple(float(v) for v in eta)
    for eta in product((0.5, 0.75, 1.0, 1.25), (-0.5, -0.25, 0.0, 0.25, 0.5), (0.0, 0.25, 0.5, 0.75))
]
ROBUST_SEEDS = list(range(95310001, 95310065))


def eta_key(eta: Any) -> tuple[float, float, float]:
    return tuple(float(np.round(value, 8)) for value in eta)


def query_rows() -> list[dict[str, Any]]:
    import csv
    with (HERE / "queried_states.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("episode_index", "query_step", "terminal_step", "remaining_global_steps", "rng_namespace", "exact_flow_root_seed", "exact_flow_rollout_id"):
            row[key] = int(row[key])
        row["seconds_before_deadlock"] = float(row["seconds_before_deadlock"])
    return sorted(rows, key=lambda row: row["state_id"])


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(tmp, path)


def outcome(env, error: dict[str, Any] | None) -> str:
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
    parser.add_argument("--mode", choices=("exact", "robust"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    config_payload = json.loads((HERE / "eta_search_config.json").read_text())
    manifest = json.loads((HERE / "strict_deadlock_manifest.json").read_text())
    config = Config(**manifest["environment"])
    cbf = CBFConfig(**manifest["cbf"])
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("checkpoint mismatch")
    policy, provenance = load_policy(CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != manifest["environment"]:
        raise RuntimeError("Flow provenance mismatch")
    single = lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    all_states = query_rows()
    states = [row for index, row in enumerate(all_states) if index % args.shard_count == args.shard_index]
    case_by_id = {row["case_id"]: row for row in manifest["cases"]}
    output_dir = HERE / "raw" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    new_rollouts = physical_steps = 0

    def run_batch(state: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal new_rollouts, physical_steps
        envs = [restore_full(Path(state["state_file"]), config) for _ in tasks]
        correctors = [DiagnosticCorrector(DiagnosticPhi(*task["eta"])) for task in tasks]
        errors: list[dict[str, Any] | None] = [None] * len(tasks)
        sums = np.zeros(len(tasks), dtype=np.float64)
        raw_sums = np.zeros(len(tasks), dtype=np.float64)
        exec_sums = np.zeros(len(tasks), dtype=np.float64)
        rewrite_sums = np.zeros(len(tasks), dtype=np.float64)
        retries1 = np.zeros(len(tasks), dtype=np.int64)
        retries2 = np.zeros(len(tasks), dtype=np.int64)
        steps_run = np.zeros(len(tasks), dtype=np.int64)
        first_step: list[dict[str, Any] | None] = [None] * len(tasks)
        episode_keys = []
        for task in tasks:
            if args.mode == "exact":
                key = jax.random.fold_in(jax.random.PRNGKey(state["exact_flow_root_seed"]), state["exact_flow_rollout_id"])
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(task["seed"]), state["rng_namespace"])
            episode_keys.append(np.asarray(key, dtype=np.uint32))
        padded = args.batch
        while len(episode_keys) < padded:
            episode_keys.append(episode_keys[-1])
        key0 = jnp.asarray(np.asarray(episode_keys))
        observations = np.zeros((padded, 2, 10), dtype=np.float32)
        absolute_steps = np.zeros(padded, dtype=np.uint32)
        while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
            for i, env in enumerate(envs):
                if not env.done and errors[i] is None:
                    observations[i] = env.observation()
                    absolute_steps[i] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(absolute_steps))))
            for i, (env, corrector) in enumerate(zip(envs, correctors)):
                if env.done or errors[i] is not None:
                    continue
                try:
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    raw = corrector(observations[i], safe, config.max_speed)
                    target = safe + raw
                    executed, second_status, second_retry, _ = project_velocity_with_retry(target, A, lower, config.max_speed, cbf)
                    rewrite = executed - target
                    if first_step[i] is None:
                        first_step[i] = {
                            "u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(),
                            "g_raw": raw.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(),
                            "first_status": str(first_status), "second_status": str(second_status),
                        }
                    sums[i] += float(np.sum((executed - safe) ** 2))
                    raw_sums[i] += float(np.linalg.norm(raw))
                    exec_sums[i] += float(np.linalg.norm(executed - safe))
                    rewrite_sums[i] += float(np.linalg.norm(rewrite))
                    retries1[i] += int(first_retry); retries2[i] += int(second_retry)
                    steps_run[i] += 1
                    env.step(executed)
                except Exception as exc:
                    errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
        result = []
        for i, task in enumerate(tasks):
            steps = int(steps_run[i])
            row = {
                "state_id": state["state_id"], "case_id": state["case_id"], "benchmark": state["benchmark"],
                "query_label": state["query_label"], "query_step": state["query_step"],
                "seconds_before_deadlock": state["seconds_before_deadlock"],
                "eta": list(eta_key(task["eta"])), "search_stage": task["search_stage"],
                "candidate_id": task["candidate_id"], "seed": task.get("seed"),
                "flow_mode": args.mode, "outcome": outcome(envs[i], errors[i]),
                "execution_error": errors[i], "steps": steps, "terminal_step": int(envs[i].step_count),
                "J_def": float(config.dt * sums[i]),
                "raw_correction_norm_mean": float(raw_sums[i] / steps) if steps else None,
                "executed_correction_norm_mean": float(exec_sums[i] / steps) if steps else None,
                "projection_rewrite_norm_mean": float(rewrite_sums[i] / steps) if steps else None,
                "projection_rewrite_fraction_of_raw": float(rewrite_sums[i] / raw_sums[i]) if raw_sums[i] > 0 else 0.0,
                "first_projection_retries": int(retries1[i]), "second_projection_retries": int(retries2[i]),
                "first_step": first_step[i],
            }
            result.append(row)
            new_rollouts += 1; physical_steps += steps
        return result

    def evaluate(state: dict[str, Any], candidates: list[tuple[tuple[float, float, float], str]]) -> list[dict[str, Any]]:
        tasks = [{"eta": eta, "search_stage": stage, "candidate_id": f"{stage}__{index:04d}"} for index, (eta, stage) in enumerate(candidates)]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(tasks), args.batch):
            rows.extend(run_batch(state, tasks[start:start + args.batch]))
        return rows

    def representatives(successes: list[dict[str, Any]], maximum: int = 4) -> list[tuple[float, float, float]]:
        if not successes:
            return []
        ordered = sorted(successes, key=lambda row: (float(row["J_def"]), eta_key(row["eta"])))
        pool = [eta_key(row["eta"]) for row in ordered]
        chosen: list[tuple[float, float, float]] = []
        for eta in (pool[0], min(pool, key=lambda x: (np.linalg.norm(x), x)), max(pool, key=lambda x: (np.linalg.norm(x), x))):
            if eta not in chosen:
                chosen.append(eta)
        while len(chosen) < min(maximum, len(pool)):
            candidate = max((eta for eta in pool if eta not in chosen), key=lambda eta: (min(np.linalg.norm((np.asarray(eta) - np.asarray(old)) / (DOMAIN_HIGH - DOMAIN_LOW)) for old in chosen), eta))
            chosen.append(candidate)
        return chosen

    def neighborhood(centers: list[tuple[float, float, float]], delta: float, seen: set[tuple[float, float, float]], stage: str) -> list[tuple[tuple[float, float, float], str]]:
        values = set()
        for center in centers:
            for offset in product((-delta, 0.0, delta), repeat=3):
                if offset == (0.0, 0.0, 0.0):
                    continue
                eta = np.asarray(center) + np.asarray(offset)
                if np.all(eta >= DOMAIN_LOW - 1e-12) and np.all(eta <= DOMAIN_HIGH + 1e-12):
                    key = eta_key(eta)
                    if key not in seen:
                        values.add(key)
        return [(eta, stage) for eta in sorted(values)]

    for state_number, state in enumerate(states):
        output_path = output_dir / f"{state['state_id']}.jsonl"
        if output_path.is_file():
            existing = [json.loads(line) for line in output_path.read_text().splitlines() if line]
            if existing and all(row.get("state_complete") is True for row in existing[-1:]):
                print(json.dumps({"skip_complete": state["state_id"], "rows": len(existing)}), flush=True)
                continue
        rows: list[dict[str, Any]] = []
        if args.mode == "exact":
            seen: set[tuple[float, float, float]] = set(COARSE)
            rows.extend(evaluate(state, [(eta, "stage1_coarse") for eta in COARSE]))
            successes = [row for row in rows if row["outcome"] == "success"]
            if successes:
                stage2 = neighborhood(representatives(successes), 0.125, seen, "stage2_refine_0125")
                seen.update(eta for eta, _ in stage2)
                rows.extend(evaluate(state, stage2))
                successes = [row for row in rows if row["outcome"] == "success"]
            if not successes:
                sobol = qmc.Sobol(d=3, scramble=True, seed=95300001).random_base2(8)
                candidates = []
                for eta in qmc.scale(sobol, DOMAIN_LOW, DOMAIN_HIGH):
                    key = eta_key(eta)
                    if key not in seen:
                        candidates.append((key, "stage3_sobol256")); seen.add(key)
                rows.extend(evaluate(state, candidates))
                successes = [row for row in rows if row["outcome"] == "success"]
            if successes:
                stage4 = neighborhood(representatives(successes), 0.0625, seen, "stage4_refine_00625")
                rows.extend(evaluate(state, stage4))
        else:
            exact_path = HERE / "raw/exact" / f"{state['state_id']}.jsonl"
            if not exact_path.is_file():
                raise RuntimeError(("exact search absent", state["state_id"]))
            exact = [json.loads(line) for line in exact_path.read_text().splitlines() if line and not json.loads(line).get("state_complete")]
            successes = sorted((row for row in exact if row["outcome"] == "success"), key=lambda row: (float(row["J_def"]), eta_key(row["eta"])))
            if successes:
                # Four promising/diverse candidates first; extend to eight only
                # if none has a robust 63/64 basin.
                selected = representatives(successes, maximum=8)
                for wave, candidates in enumerate((selected[:4], selected[4:8]), start=1):
                    if not candidates:
                        continue
                    tasks = []
                    for candidate_index, eta in enumerate(candidates):
                        for seed_index, seed in enumerate(ROBUST_SEEDS):
                            tasks.append({"eta": eta, "search_stage": f"robust_wave{wave}", "candidate_id": f"robust_{wave}_{candidate_index}", "seed": seed, "seed_index": seed_index})
                    for start in range(0, len(tasks), args.batch):
                        rows.extend(run_batch(state, tasks[start:start + args.batch]))
                    candidate_counts = {}
                    for eta in candidates:
                        key = eta_key(eta)
                        candidate_rows = [row for row in rows if eta_key(row["eta"]) == key]
                        candidate_counts[key] = sum(row["outcome"] == "success" for row in candidate_rows)
                    if any(count >= 63 for count in candidate_counts.values()):
                        break
        marker = {
            "state_complete": True, "state_id": state["state_id"], "mode": args.mode,
            "candidate_rollouts": len(rows), "physical_steps": sum(int(row["steps"]) for row in rows),
            "exact_success_count": sum(row["outcome"] == "success" for row in rows) if args.mode == "exact" else None,
        }
        atomic_jsonl(output_path, rows + [marker])
        print(json.dumps({"completed": state["state_id"], **marker, "shard_progress": f"{state_number + 1}/{len(states)}", "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    runtime = {
        "mode": args.mode, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "state_count": len(states), "new_rollouts": new_rollouts, "physical_steps": physical_steps,
        "elapsed_seconds": time.monotonic() - started, "device": [str(device) for device in jax.devices()],
        "batch": args.batch, "search_config_sha256": sha256(HERE / "eta_search_config.json"),
    }
    write_json(HERE / f"runtime_{args.mode}_shard{args.shard_index}.json", runtime)
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
