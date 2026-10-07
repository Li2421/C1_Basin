"""Matched causal one-step delay audit with a frozen oracle-consistent future."""

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


def load_reuse(states, seeds):
    """Load exact Branch-I tuples previously evaluated under the V4 oracle."""
    wanted = {(row["state_id"], tuple(row["eta_best"]), seed) for row in states for seed in seeds}
    found = {}
    raw = ROOT / "diagnostics/gphi_training_dataset_v4/raw"
    for path in raw.rglob("records.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row.get("state_id"), tuple(float(v) for v in row.get("eta", [])), int(row.get("seed", -1)))
            if key in wanted:
                found[key] = (path, row)
    return found


def convert_reuse(state, branch, path, row):
    first = row["first_step"]
    u_safe = np.asarray(first["u_safe"], dtype=np.float64)
    u_exec = np.asarray(first["u_exec"], dtype=np.float64)
    current = float(0.05 * np.sum((u_exec - u_safe) ** 2))
    return {
        "state_id": state["state_id"], "set_role": state["set_role"],
        "source_group": state["source_group"], "y_long": state["y_long"],
        "eta_best": state["eta_best"], "branch": branch, "delay_steps": 0 if branch == "I" else 1,
        "seed": int(row["seed"]), "outcome": row["outcome"], "success": row["outcome"] == "success",
        "steps": int(row["steps"]), "terminal_step": int(row["terminal_step"]),
        "J_total": float(row["J_def"]), "J_current": current,
        "J_future": float(max(0.0, float(row["J_def"]) - current)),
        "first_step": first, "execution_error": row.get("execution_error"),
        "reused": True, "reuse_source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=64)
    parser.add_argument("--stage", default="initial64")
    parser.add_argument("--selection-json")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)

    plan_path = HERE / "audit_plan.json"
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    all_states = plan["states"]
    selected_seeds = plan["seeds_initial"]
    if args.selection_json:
        selection = json.loads(Path(args.selection_json).read_text())
        selected_ids = set(selection["state_ids"])
        all_states = [row for row in all_states if row["state_id"] in selected_ids]
        if len(all_states) != len(selected_ids):
            raise AssertionError("selection references unknown states")
        if "new_seeds" in selection:
            selected_seeds = selection["new_seeds"]
        else:
            first, stop = selection["new_seed_range_start_stop_exclusive"]
            selected_seeds = list(range(int(first), int(stop)))
    states = [row for index, row in enumerate(all_states) if index % args.shards == args.shard]
    seeds = selected_seeds[args.seed_start:args.seed_stop]
    config = Config(**plan["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(plan["checkpoint"]))
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    reuse = load_reuse(states, seeds)

    output = HERE / "raw" / f"{args.stage}_shard{args.shard}"
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    records = []
    started = time.monotonic()

    for seed_index, seed in enumerate(seeds):
        tasks = []
        for state in states:
            key = (state["state_id"], tuple(state["eta_best"]), seed)
            for branch in ("I", "N"):
                # eta=0 makes both branches literally identical; any existing
                # full-horizon tuple is therefore valid for both.
                can_reuse = key in reuse and (branch == "I" or state["eta_best"] == [0.0, 0.0, 0.0])
                if can_reuse:
                    path, prior = reuse[key]
                    records.append(convert_reuse(state, branch, path, prior))
                else:
                    tasks.append({"state": state, "branch": branch, "seed": seed})

        for start in range(0, len(tasks), args.batch):
            chunk = tasks[start:start + args.batch]
            envs, correctors, keys0 = [], [], []
            errors = [None for _ in chunk]
            sums = np.zeros(len(chunk), dtype=np.float64)
            current_sums = np.zeros(len(chunk), dtype=np.float64)
            first = [None for _ in chunk]
            initial_steps = []
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
                    local_step = int(env.step_count - initial_steps[i])
                    flow = bounded_nominal(actions[i], config.max_speed)
                    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    try:
                        safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        if chunk[i]["branch"] == "N" and local_step == 0:
                            raw = np.zeros_like(safe)
                        else:
                            raw = correctors[i](observations[i], safe, config.max_speed)
                        executed, second_status, second_retry, _ = project_velocity_with_retry(
                            safe + raw, A, lower, config.max_speed, cbf
                        )
                    except Exception as exc:
                        errors[i] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": int(env.step_count)}
                        continue
                    deformation = float(np.sum((executed - safe) ** 2, dtype=np.float64))
                    sums[i] += deformation
                    if local_step == 0:
                        current_sums[i] = deformation
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
                result = outcome(envs[i], errors[i])
                state = task["state"]
                total = float(config.dt * sums[i])
                current = float(config.dt * current_sums[i])
                records.append({
                    "state_id": state["state_id"], "set_role": state["set_role"],
                    "source_group": state["source_group"], "y_long": state["y_long"],
                    "eta_best": state["eta_best"], "branch": task["branch"],
                    "delay_steps": 0 if task["branch"] == "I" else 1, "seed": seed,
                    "outcome": result, "success": result == "success",
                    "steps": int(envs[i].step_count - state["step"]), "terminal_step": int(envs[i].step_count),
                    "J_total": total, "J_current": current, "J_future": float(max(0.0, total - current)),
                    "first_step": first[i], "execution_error": errors[i], "reused": False, "reuse_source": None,
                })
        records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        if (seed_index + 1) % 8 == 0 or seed_index + 1 == len(seeds):
            print(json.dumps({
                "shard": args.shard, "seed_round": seed_index + 1, "records": len(records),
                "reused": sum(row["reused"] for row in records),
                "outcomes": dict(Counter(row["outcome"] for row in records)),
                "elapsed_s": round(time.monotonic() - started, 1),
            }), flush=True)

    expected = len(states) * len(seeds) * 2
    if len(records) != expected:
        raise AssertionError((len(records), expected))
    keys = {(row["state_id"], row["branch"], row["seed"]) for row in records}
    if len(keys) != len(records):
        raise AssertionError("duplicate state/branch/seed tuples")
    manifest = {
        "stage": args.stage, "shard": args.shard, "shards": args.shards,
        "state_ids": [row["state_id"] for row in states], "seed_count": len(seeds),
        "records": len(records), "new_rollouts": sum(not row["reused"] for row in records),
        "reused_rollouts": sum(row["reused"] for row in records),
        "physical_steps_new": sum(row["steps"] for row in records if not row["reused"]),
        "elapsed_s": time.monotonic() - started, "device": [str(device) for device in jax.devices()],
        "audit_plan_sha256": plan_hash, "records_sha256": sha(records_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if sha(plan_path) != plan_hash:
        raise RuntimeError("audit plan changed during rollout")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
