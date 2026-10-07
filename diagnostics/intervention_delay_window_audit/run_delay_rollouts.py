"""Run frozen matched intervention-delay continuations with exact reuse."""

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


ROOT = Path("/home/zhihan/research/Basin_C1")
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


def outcome(env, execution_error):
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


def load_v4_reuse(states, seeds):
    wanted = {(s["state_id"], tuple(s["eta_best"]), seed) for s in states for seed in seeds}
    found = {}
    for path in (ROOT / "diagnostics/gphi_training_dataset_v4/raw").rglob("records.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (
                row.get("state_id"),
                tuple(float(v) for v in row.get("eta", [])),
                int(row.get("seed", -1)),
            )
            if key in wanted:
                if key in found and found[key][1].get("outcome") != row.get("outcome"):
                    raise AssertionError(f"inconsistent V4 reuse tuple: {key}")
                found[key] = (path, row)
    return found


def load_conceptual_reuse(states, seeds, plan):
    old = ROOT / "diagnostics/gate_conceptual_validity_audit/temporal"
    old_plan = json.loads((old / "audit_plan.json").read_text())
    if old_plan["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise AssertionError("old d1 cache checkpoint differs")
    if old_plan["environment"] != plan["environment"]:
        raise AssertionError("old d1 cache environment differs")
    wanted = {(s["state_id"], delay, seed) for s in states for seed in seeds for delay in (0, 1)}
    found = {}
    for path in sorted((old / "raw").glob("*/records.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            branch = row.get("branch")
            delay = int(row.get("delay_steps", -1))
            if (branch, delay) not in {("I", 0), ("N", 1)}:
                continue
            key = (row.get("state_id"), delay, int(row.get("seed", -1)))
            if key in wanted:
                if key in found and found[key][1].get("outcome") != row.get("outcome"):
                    raise AssertionError(f"inconsistent d1 reuse tuple: {key}")
                found[key] = (path, row)
    return found


def from_v4(state, delay, path, row):
    return {
        "state_id": state["state_id"], "category": state["category"],
        "source_group": state["source_group"], "y_long": state["y_long"],
        "is_hard13_anchor": state["is_hard13_anchor"], "eta_best": state["eta_best"],
        "delay_steps": delay, "seed": int(row["seed"]), "outcome": row["outcome"],
        "success": row["outcome"] == "success", "steps": int(row["steps"]),
        "terminal_step": int(row["terminal_step"]), "J_total": float(row["J_def"]),
        "J_delay_prefix": 0.0, "J_after_switch": float(row["J_def"]),
        "first_step": row.get("first_step"), "execution_error": row.get("execution_error"),
        "reused": True, "reuse_kind": "v4_exact_eta_best",
        "reuse_source": str(path), "physical_rollout_executed": False,
    }


def from_old_delay(state, delay, path, row):
    return {
        "state_id": state["state_id"], "category": state["category"],
        "source_group": state["source_group"], "y_long": state["y_long"],
        "is_hard13_anchor": state["is_hard13_anchor"], "eta_best": state["eta_best"],
        "delay_steps": delay, "seed": int(row["seed"]), "outcome": row["outcome"],
        "success": bool(row["success"]), "steps": int(row["steps"]),
        "terminal_step": int(row["terminal_step"]), "J_total": float(row["J_total"]),
        "J_delay_prefix": 0.0,
        "J_after_switch": float(row["J_total"] if delay == 0 else row["J_future"]),
        "first_step": row.get("first_step"), "execution_error": row.get("execution_error"),
        "reused": True, "reuse_kind": f"conceptual_audit_exact_d{delay}",
        "reuse_source": str(path), "physical_rollout_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", required=True, help="comma-separated subset of 0,1,2,4")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=64)
    parser.add_argument("--y-long", choices=("all", "0", "1"), default="all")
    parser.add_argument("--selection-json", help="frozen adaptive state/seed selection")
    args = parser.parse_args()
    delays = sorted({int(value) for value in args.delays.split(",")})
    if not delays or not set(delays) <= {0, 1, 2, 4}:
        raise ValueError(delays)
    if not 0 <= args.shard < args.shards:
        raise ValueError((args.shard, args.shards))

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    plan_path = HERE / "audit_plan.json"
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    states_all = plan["states"]
    selection_hash = None
    selected_seeds = plan["seeds_initial"]
    selection_rule = None
    if args.selection_json:
        selection_path = Path(args.selection_json)
        selection_hash = sha(selection_path)
        selection = json.loads(selection_path.read_text())
        selected_ids = set(selection["state_ids"])
        states_all = [state for state in states_all if state["state_id"] in selected_ids]
        if {state["state_id"] for state in states_all} != selected_ids:
            raise AssertionError("adaptive selection references unknown state")
        if "new_seeds" in selection:
            selected_seeds = [int(value) for value in selection["new_seeds"]]
        else:
            first, stop = selection["new_seed_range_start_stop_exclusive"]
            selected_seeds = list(range(int(first), int(stop)))
        selection_rule = selection.get("selection_rule")
    if args.y_long != "all":
        states_all = [state for state in states_all if int(state["y_long"]) == int(args.y_long)]
    states = [state for index, state in enumerate(states_all) if index % args.shards == args.shard]
    seeds = selected_seeds[args.seed_start:args.seed_stop]
    config = Config(**plan["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(plan["checkpoint"]))
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    v4_reuse = load_v4_reuse(states, seeds)
    conceptual_reuse = load_conceptual_reuse(states, seeds, plan) if set(delays) & {0, 1} else {}

    output = HERE / "raw" / f"{args.stage}_shard{args.shard}"
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    records = []
    started = time.monotonic()

    for seed_index, seed in enumerate(seeds):
        tasks = []
        for state in states:
            v4_key = (state["state_id"], tuple(state["eta_best"]), seed)
            # With eta_best=0 every tested delay has the exact same action
            # sequence.  Materialize every requested branch from one cached
            # or newly simulated canonical continuation; never rerun it.
            if state["eta_best"] == [0.0, 0.0, 0.0]:
                if v4_key in v4_reuse:
                    path, prior = v4_reuse[v4_key]
                    for delay in delays:
                        records.append(from_v4(state, delay, path, prior))
                elif (state["state_id"], 0, seed) in conceptual_reuse:
                    path, prior = conceptual_reuse[(state["state_id"], 0, seed)]
                    for delay in delays:
                        cached = from_old_delay(state, delay, path, prior)
                        cached["delay_steps"] = delay
                        cached["reuse_kind"] = "conceptual_audit_zero_eta_exact_all_delays"
                        records.append(cached)
                else:
                    tasks.append({
                        "state": state, "delay": 0, "seed": seed,
                        "materialize_delays": delays,
                    })
                continue
            for delay in delays:
                # d=0 is exactly the saved eta_best continuation.  When
                # eta_best=0, changing d cannot change any executed action.
                if v4_key in v4_reuse and (delay == 0 or state["eta_best"] == [0.0, 0.0, 0.0]):
                    path, prior = v4_reuse[v4_key]
                    records.append(from_v4(state, delay, path, prior))
                elif (state["state_id"], delay, seed) in conceptual_reuse:
                    path, prior = conceptual_reuse[(state["state_id"], delay, seed)]
                    records.append(from_old_delay(state, delay, path, prior))
                else:
                    tasks.append({
                        "state": state, "delay": delay, "seed": seed,
                        "materialize_delays": [delay],
                    })

        for start in range(0, len(tasks), args.batch):
            chunk = tasks[start:start + args.batch]
            envs, correctors, keys0, initial_steps = [], [], [], []
            errors = [None for _ in chunk]
            sums = np.zeros(len(chunk), dtype=np.float64)
            prefix_sums = np.zeros(len(chunk), dtype=np.float64)
            first = [None for _ in chunk]
            for task in chunk:
                state = task["state"]
                envs.append(restore_full(Path(state["state_file"]), config))
                correctors.append(DiagnosticCorrector(DiagnosticPhi(*state["eta_best"])))
                keys0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed), state["rng_namespace"])))
                initial_steps.append(int(state["step"]))
            padded = max(args.batch, len(chunk))
            while len(keys0) < padded:
                keys0.append(keys0[-1])
            keys0 = jnp.asarray(np.asarray(keys0))
            observations = np.zeros((padded, 2, 10), dtype=np.float32)
            steps = np.zeros(padded, dtype=np.uint32)
            while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
                for i, env in enumerate(envs):
                    observations[i] = env.observation()
                    steps[i] = env.step_count
                actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
                for i, env in enumerate(envs):
                    if env.done or errors[i] is not None:
                        continue
                    task, state = chunk[i], chunk[i]["state"]
                    local_step = int(env.step_count - initial_steps[i])
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    try:
                        safe, first_status, first_retry, _ = project_velocity_with_retry(
                            flow, A, lower, config.max_speed, cbf
                        )
                        raw = np.zeros_like(safe) if local_step < task["delay"] else correctors[i](
                            observations[i], safe, config.max_speed
                        )
                        executed, second_status, second_retry, _ = project_velocity_with_retry(
                            safe + raw, A, lower, config.max_speed, cbf
                        )
                    except Exception as exc:
                        errors[i] = {
                            "type": type(exc).__name__, "message": str(exc),
                            "absolute_step": int(env.step_count),
                        }
                        continue
                    deformation = float(np.sum((executed - safe) ** 2, dtype=np.float64))
                    sums[i] += deformation
                    if local_step < task["delay"]:
                        prefix_sums[i] += deformation
                    if local_step == 0:
                        first[i] = {
                            "u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(),
                            "g_raw": raw.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(),
                            "first_projection_status": str(first_status), "second_projection_status": str(second_status),
                            "first_retry": bool(first_retry), "second_retry": bool(second_retry),
                            "first_min_linear_residual": float(np.min(A @ safe.reshape(4) - lower)),
                            "second_min_linear_residual": float(np.min(A @ executed.reshape(4) - lower)),
                        }
                    env.step(executed)
            for i, task in enumerate(chunk):
                state, delay = task["state"], task["delay"]
                result = outcome(envs[i], errors[i])
                total = float(config.dt * sums[i])
                prefix = float(config.dt * prefix_sums[i])
                base_record = {
                    "state_id": state["state_id"], "category": state["category"],
                    "source_group": state["source_group"], "y_long": state["y_long"],
                    "is_hard13_anchor": state["is_hard13_anchor"], "eta_best": state["eta_best"],
                    "delay_steps": delay, "seed": seed, "outcome": result,
                    "success": result == "success", "steps": int(envs[i].step_count - state["step"]),
                    "terminal_step": int(envs[i].step_count), "J_total": total,
                    "J_delay_prefix": prefix, "J_after_switch": float(max(0.0, total - prefix)),
                    "first_step": first[i], "execution_error": errors[i],
                    "reused": False, "reuse_kind": None, "reuse_source": None,
                    "physical_rollout_executed": True,
                }
                for materialized_index, materialized_delay in enumerate(task["materialize_delays"]):
                    record = dict(base_record)
                    record["delay_steps"] = materialized_delay
                    if materialized_index > 0:
                        record["reused"] = True
                        record["reuse_kind"] = "same_run_zero_eta_exact_materialization"
                        record["reuse_source"] = "same canonical state/seed eta=0 continuation"
                        record["physical_rollout_executed"] = False
                    records.append(record)
        records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        if (seed_index + 1) % 8 == 0 or seed_index + 1 == len(seeds):
            print(json.dumps({
                "shard": args.shard, "seed_round": seed_index + 1,
                "records": len(records), "new": sum(not row["reused"] for row in records),
                "reused": sum(row["reused"] for row in records),
                "outcomes": dict(Counter(row["outcome"] for row in records)),
                "elapsed_s": round(time.monotonic() - started, 1),
            }), flush=True)

    expected = len(states) * len(seeds) * len(delays)
    keys = {(row["state_id"], row["delay_steps"], row["seed"]) for row in records}
    if len(records) != expected or len(keys) != len(records):
        raise AssertionError((len(records), expected, len(keys)))
    if any(row["execution_error"] is not None for row in records):
        raise AssertionError("execution errors present")
    manifest = {
        "stage": args.stage, "shard": args.shard, "shards": args.shards,
        "delays": delays, "state_ids": [row["state_id"] for row in states],
        "seed_count": len(seeds), "records": len(records),
        "new_rollouts": sum(row["physical_rollout_executed"] for row in records),
        "reused_rollouts": sum(not row["physical_rollout_executed"] for row in records),
        "reuse_kinds": dict(Counter(row["reuse_kind"] for row in records if row["reused"])),
        "physical_steps_new": sum(row["steps"] for row in records if row["physical_rollout_executed"]),
        "elapsed_s": time.monotonic() - started,
        "device": [str(device) for device in jax.devices()],
        "audit_plan_sha256": plan_hash, "records_sha256": sha(records_path),
        "adaptive_selection_sha256": selection_hash,
        "adaptive_selection_rule": selection_rule,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if sha(plan_path) != plan_hash:
        raise RuntimeError("audit plan changed during rollout")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
