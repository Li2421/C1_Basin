"""Resumable matched-delay runner backed by the frozen tuple reuse index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = Path(__file__).resolve().parent
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def terminal_outcome(env, execution_error):
    if execution_error is not None:
        return "execution_error"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "deadlock"
    return "timeout"


def normalize_source(state: dict, delay: int, seed: int, record: dict, entry: dict) -> dict:
    dataset_source = "J_def" in record and "J_total" not in record
    total = float(record["J_def"] if dataset_source else record["J_total"])
    return {
        "state_id": state["state_id"], "category": state["category"],
        "source_group": state["source_group"], "y_long": state["y_long"],
        "is_hard13_anchor": state["is_hard13_anchor"], "eta_best": state["eta_best"],
        "delay_steps": int(delay), "seed": int(seed), "outcome": record["outcome"],
        "success": record["outcome"] == "success", "steps": int(record["steps"]),
        "terminal_step": int(record["terminal_step"]), "J_total": total,
        "J_delay_prefix": float(record.get("J_delay_prefix", 0.0)),
        "J_after_switch": float(record.get("J_after_switch", total)),
        "first_step": record.get("first_step"), "execution_error": record.get("execution_error"),
        "reused": True, "reuse_kind": entry["reuse_kind"],
        "reuse_source": entry["source_path"], "physical_rollout_executed": False,
    }


def load_source_records(entries: list[dict]) -> dict[tuple[str, int], dict]:
    wanted = defaultdict(dict)
    for entry in entries:
        if entry.get("status") != "REUSE":
            continue
        key = (entry["source_path"], int(entry["source_line_number"]))
        wanted[entry["source_path"]][int(entry["source_line_number"])] = entry["source_line_sha256"]
    loaded = {}
    for path_string, lines in wanted.items():
        path = Path(path_string)
        expected_file_hashes = {
            entry["source_file_sha256"] for entry in entries
            if entry.get("source_path") == path_string
        }
        if len(expected_file_hashes) != 1 or sha(path) not in expected_file_hashes:
            raise AssertionError(f"source file hash mismatch: {path}")
        remaining = set(lines)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if line_number not in remaining:
                continue
            if line_sha(line) != lines[line_number]:
                raise AssertionError(f"source line hash mismatch: {path}:{line_number}")
            loaded[(path_string, line_number)] = json.loads(line)
            remaining.remove(line_number)
            if not remaining:
                break
        if remaining:
            raise AssertionError(f"source lines missing: {path}: {sorted(remaining)[:5]}")
    return loaded


def append_records(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_existing(path: Path) -> tuple[list[dict], set[tuple[str, int, int]]]:
    if not path.exists():
        return [], set()
    rows, keys = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
        if key in keys:
            raise AssertionError(f"duplicate tuple in resumable output: {key}")
        rows.append(row)
        keys.add(key)
    return rows, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", required=True, help="comma-separated integer delays in [0,128]")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=64)
    parser.add_argument("--selection-json")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    delays = sorted({int(value) for value in args.delays.split(",")})
    if not delays or any(delay < 0 or delay > 128 for delay in delays):
        raise ValueError(delays)
    if not 0 <= args.shard < args.shards:
        raise ValueError((args.shard, args.shards))

    plan_path = HERE / "audit_plan.json"
    index_path = HERE / "tuple_index.jsonl"
    plan_hash, index_hash = sha(plan_path), sha(index_path)
    plan = json.loads(plan_path.read_text())
    states_all = plan["states"]
    selection_hash = None
    common_new_seeds = None
    state_seed_map = None
    if args.selection_json:
        selection_path = Path(args.selection_json)
        selection_hash = sha(selection_path)
        selection = json.loads(selection_path.read_text())
        selected_ids = set(selection["state_ids"])
        states_all = [state for state in states_all if state["state_id"] in selected_ids]
        if {state["state_id"] for state in states_all} != selected_ids:
            raise AssertionError("adaptive selection references unknown state")
        common_new_seeds = selection.get("new_seeds")
        state_seed_map = selection.get("state_seed_map")
    states = [state for index, state in enumerate(states_all) if index % args.shards == args.shard]
    state_by_id = {state["state_id"]: state for state in states}
    seeds_by_state = {}
    for state in states:
        state_id = state["state_id"]
        if state_seed_map is not None:
            if state_id not in state_seed_map:
                raise AssertionError(f"state_seed_map missing {state_id}")
            seeds = [int(value) for value in state_seed_map[state_id]]
        elif common_new_seeds is not None:
            seeds = [int(value) for value in common_new_seeds]
        else:
            seeds = [int(value) for value in state["seeds_initial"]]
        seeds_by_state[state_id] = seeds[args.seed_start:args.seed_stop]

    preregistered = {}
    zero_canonical = {}
    for line in index_path.read_text().splitlines():
        entry = json.loads(line)
        key = (entry["state_id"], int(entry["delay_steps"]), int(entry["seed"]))
        if entry["state_id"] not in state_by_id:
            continue
        if int(entry["delay_steps"]) in delays:
            preregistered[key] = entry
        state = state_by_id[entry["state_id"]]
        canonical_key = (entry["state_id"], int(entry["seed"]))
        if (
            state["eta_best"] == [0.0, 0.0, 0.0]
            and entry["status"] == "REUSE"
            and (canonical_key not in zero_canonical or int(entry["delay_steps"]) == 0)
        ):
            zero_canonical[canonical_key] = entry
    requested = [
        (state["state_id"], delay, seed)
        for state in states for seed in seeds_by_state[state["state_id"]] for delay in delays
    ]
    if len(requested) != len(set(requested)):
        raise AssertionError("requested tuple duplication")
    entries = []
    for key in requested:
        entry = preregistered.get(key)
        if entry is None and state_by_id[key[0]]["eta_best"] == [0.0, 0.0, 0.0]:
            canonical = zero_canonical.get((key[0], key[2]))
            if canonical is not None:
                entry = dict(canonical)
                entry.update({
                    "delay_steps": key[1],
                    "reuse_kind": "zero_eta_canonical_adaptive_delay",
                })
        if entry is None:
            entry = {
                "state_id": key[0], "delay_steps": key[1], "seed": key[2],
                "status": "NEW", "reuse_kind": None, "source_path": None,
                "source_line_number": None, "source_line_sha256": None,
                "source_file_sha256": None, "source_kind": None,
            }
        entries.append(entry)

    inventory = Counter(entry["status"] for entry in entries)
    inventory_obj = {
        "stage": args.stage, "shard": args.shard, "shards": args.shards,
        "states": len(states), "delays": delays,
        "tuples": len(entries), "reuse": inventory["REUSE"], "new": inventory["NEW"],
        "audit_plan_sha256": plan_hash, "tuple_index_sha256": index_hash,
        "selection_sha256": selection_hash,
    }
    if args.inventory_only:
        print(json.dumps(inventory_obj, indent=2, sort_keys=True))
        return

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    output = HERE / "raw" / f"{args.stage}_shard{args.shard}"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    spec_path = output / "stage_spec.json"
    spec = {**inventory_obj, "batch": args.batch, "device_requested": args.device}
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise AssertionError("resume stage spec differs from frozen invocation")
    if not spec_path.exists():
        atomic_json(spec_path, spec)
    rows_existing, completed = read_existing(records_path)
    requested_set = set(requested)
    if not completed <= requested_set:
        raise AssertionError("output contains tuples outside this stage spec")

    pending_reuse = [entry for entry in entries if entry["status"] == "REUSE" and (entry["state_id"], int(entry["delay_steps"]), int(entry["seed"])) not in completed]
    sources = load_source_records(pending_reuse)
    started = time.monotonic()
    for start in range(0, len(pending_reuse), 1024):
        chunk = pending_reuse[start:start + 1024]
        materialized = []
        for entry in chunk:
            state = state_by_id[entry["state_id"]]
            source = sources[(entry["source_path"], int(entry["source_line_number"]))]
            materialized.append(normalize_source(state, int(entry["delay_steps"]), int(entry["seed"]), source, entry))
        append_records(records_path, materialized)
        completed.update((row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in materialized)
        atomic_json(output / "progress.json", {**inventory_obj, "completed": len(completed), "elapsed_s": time.monotonic() - started})

    # Group zero-eta missing delays into one physical canonical run per
    # (state,seed); nonzero tuples remain one physical task each.
    missing = [entry for entry in entries if (entry["state_id"], int(entry["delay_steps"]), int(entry["seed"])) not in completed]
    zero_groups = defaultdict(list)
    tasks = []
    for entry in missing:
        state = state_by_id[entry["state_id"]]
        if state["eta_best"] == [0.0, 0.0, 0.0]:
            zero_groups[(state["state_id"], int(entry["seed"]))].append(int(entry["delay_steps"]))
        else:
            tasks.append({"state": state, "seed": int(entry["seed"]), "delay": int(entry["delay_steps"]), "materialize_delays": [int(entry["delay_steps"])]})
    for (state_id, seed), values in zero_groups.items():
        tasks.append({"state": state_by_id[state_id], "seed": seed, "delay": 0, "materialize_delays": sorted(values)})
    tasks.sort(key=lambda item: (item["seed"], item["state"]["state_id"], item["delay"]))

    config = Config(**plan["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(plan["checkpoint"]))
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    for start in range(0, len(tasks), args.batch):
        chunk = tasks[start:start + args.batch]
        envs, correctors, keys0, initial_steps = [], [], [], []
        errors = [None for _ in chunk]
        deformation_sum = np.zeros(len(chunk), dtype=np.float64)
        prefix_sum = np.zeros(len(chunk), dtype=np.float64)
        first_steps = [None for _ in chunk]
        for task in chunk:
            state = task["state"]
            envs.append(restore_full(Path(state["state_file"]), config))
            correctors.append(DiagnosticCorrector(DiagnosticPhi(*state["eta_best"])))
            keys0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), state["rng_namespace"])))
            initial_steps.append(int(state["step"]))
        padded = max(args.batch, len(chunk))
        while len(keys0) < padded:
            keys0.append(keys0[-1])
        keys0 = jnp.asarray(np.asarray(keys0))
        observations = np.zeros((padded, 2, 10), dtype=np.float32)
        absolute_steps = np.zeros(padded, dtype=np.uint32)
        while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
            for index, env in enumerate(envs):
                observations[index] = env.observation()
                absolute_steps[index] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(absolute_steps))))
            for index, env in enumerate(envs):
                if env.done or errors[index] is not None:
                    continue
                task = chunk[index]
                local_step = int(env.step_count - initial_steps[index])
                flow = bounded_nominal(actions[index], config.max_speed)
                A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                try:
                    safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    raw = np.zeros_like(safe) if local_step < task["delay"] else correctors[index](observations[index], safe, config.max_speed)
                    executed, second_status, second_retry, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
                except Exception as exc:
                    errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
                    continue
                deformation = float(np.sum((executed - safe) ** 2, dtype=np.float64))
                deformation_sum[index] += deformation
                if local_step < task["delay"]:
                    prefix_sum[index] += deformation
                if local_step == 0:
                    first_steps[index] = {
                        "u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(),
                        "g_raw": raw.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(),
                        "first_projection_status": str(first_status), "second_projection_status": str(second_status),
                        "first_retry": bool(first_retry), "second_retry": bool(second_retry),
                        "first_min_linear_residual": float(np.min(A @ safe.reshape(4) - lower)),
                        "second_min_linear_residual": float(np.min(A @ executed.reshape(4) - lower)),
                    }
                env.step(executed)
        materialized = []
        for index, task in enumerate(chunk):
            state = task["state"]
            result = terminal_outcome(envs[index], errors[index])
            total = float(config.dt * deformation_sum[index])
            prefix = float(config.dt * prefix_sum[index])
            base = {
                "state_id": state["state_id"], "category": state["category"],
                "source_group": state["source_group"], "y_long": state["y_long"],
                "is_hard13_anchor": state["is_hard13_anchor"], "eta_best": state["eta_best"],
                "delay_steps": task["delay"], "seed": task["seed"], "outcome": result,
                "success": result == "success", "steps": int(envs[index].step_count - state["step"]),
                "terminal_step": int(envs[index].step_count), "J_total": total,
                "J_delay_prefix": prefix, "J_after_switch": float(max(0.0, total - prefix)),
                "first_step": first_steps[index], "execution_error": errors[index],
                "reused": False, "reuse_kind": None, "reuse_source": None,
                "physical_rollout_executed": True,
            }
            for materialized_index, delay in enumerate(task["materialize_delays"]):
                row = dict(base)
                row["delay_steps"] = delay
                if materialized_index:
                    row.update({
                        "reused": True, "reuse_kind": "same_run_zero_eta_exact_materialization",
                        "reuse_source": "same canonical eta=0 continuation", "physical_rollout_executed": False,
                    })
                materialized.append(row)
        if any(row["execution_error"] is not None for row in materialized):
            append_records(output / "execution_errors.jsonl", [row for row in materialized if row["execution_error"] is not None])
            raise RuntimeError("execution error; failing before adding invalid tuples to main output")
        append_records(records_path, materialized)
        completed.update((row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in materialized)
        progress = {
            **inventory_obj, "completed": len(completed), "physical_tasks_completed": min(start + len(chunk), len(tasks)),
            "physical_tasks_total": len(tasks), "elapsed_s": time.monotonic() - started,
        }
        atomic_json(output / "progress.json", progress)
        if (start // args.batch + 1) % 8 == 0 or start + len(chunk) == len(tasks):
            print(json.dumps(progress, sort_keys=True), flush=True)

    rows, keys = read_existing(records_path)
    if keys != requested_set or len(rows) != len(requested):
        raise AssertionError((len(rows), len(requested), len(keys), len(requested_set)))
    if any(row.get("execution_error") is not None for row in rows):
        raise AssertionError("execution errors in final records")
    manifest = {
        **inventory_obj, "status": "complete", "records": len(rows),
        "new_physical_rollouts": sum(bool(row["physical_rollout_executed"]) for row in rows),
        "reused_or_materialized_records": sum(not bool(row["physical_rollout_executed"]) for row in rows),
        "reuse_kinds": dict(Counter(row.get("reuse_kind") for row in rows if row.get("reused"))),
        "physical_steps_new": sum(int(row["steps"]) for row in rows if row["physical_rollout_executed"]),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "elapsed_s_this_invocation": time.monotonic() - started,
        "records_sha256": sha(records_path), "stage_spec_sha256": sha(spec_path),
        "device": [str(device) for device in jax.devices()],
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "progress.json", {**manifest, "completed": len(rows)})
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
