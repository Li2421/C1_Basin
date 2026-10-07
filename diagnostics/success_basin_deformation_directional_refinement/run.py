"""Frozen closed-loop rollout runner with isolated post-projection J_def logging."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SBMA = ROOT / "diagnostics/success_basin_multimodality"
STATES = ROOT / "diagnostics/success_basin_geometry"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

import jax
import jax.numpy as jnp

from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    protocol = json.loads((HERE / "protocol.json").read_text())
    locked = sha(HERE / "protocol.json")
    jobs = json.loads((HERE / args.jobs).read_text())
    catalog = protocol["states"]
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    assert Path(inspect.getfile(project_velocity)).resolve() == SYSROOT / "single_integrator/cbf.py"
    assert sha(Path(inspect.getfile(project_velocity))) == protocol["source_hashes"]["projection"]
    raw = HERE / "raw" / args.stage
    raw.mkdir(parents=True, exist_ok=False)
    expected_steps = sum(config.max_steps - catalog[job["state_id"]]["start_step"] for job in jobs)
    print(json.dumps({
        "stage": args.stage, "rollout_count": len(jobs),
        "expected_max_physical_steps": expected_steps, "CPU_GPU": args.device,
        "purpose": "directional local expansion of success-constrained J_def",
    }), flush=True)
    single = lambda observation, rng: policy.sample_actions(observation[None], seed=rng)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    records, total_steps = [], 0
    started = last = time.monotonic()
    for chunk_start in range(0, len(jobs), args.batch):
        tasks = jobs[chunk_start:chunk_start + args.batch]
        envs, correctors, histories, key0 = [], [], [], []
        errors = [None] * len(tasks)
        for task in tasks:
            entry = catalog[task["state_id"]]
            with np.load(STATES / entry["state_file"]) as state:
                envs.append(restore(dict(state), config))
            correctors.append(DiagnosticCorrector(DiagnosticPhi(*task["eta"])))
            histories.append({name: [] for name in (
                "positions_before", "positions_after", "u_flow", "u_safe", "g", "w",
                "u_exec", "delta_u_squared", "first_status", "second_status",
                "first_retry", "second_retry", "timer", "event",
            )})
            key0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), entry["pair_id"])))
        while len(key0) < args.batch:
            key0.append(key0[-1])
        keys0 = jnp.asarray(np.asarray(key0))
        observations = np.zeros((args.batch, 2, 10), dtype=np.float32)
        steps = np.zeros(args.batch, dtype=np.uint32)
        while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
            for i, env in enumerate(envs):
                observations[i] = env.observation()
                steps[i] = env.step_count
            actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
            for i, env in enumerate(envs):
                if env.done or errors[i] is not None:
                    continue
                before = env.positions.copy()
                flow = bounded_nominal(actions[i], config.max_speed)
                A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                try:
                    safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                    g = correctors[i](observations[i], safe, config.max_speed)
                    w = safe + g
                    executed, second_status, second_retry, _ = project_velocity_with_retry(w, A, lower, config.max_speed, cbf)
                except Exception as exception:
                    errors[i] = {"type": type(exception).__name__, "message": str(exception), "absolute_step": env.step_count}
                    continue
                delta_squared = float(np.sum((executed - safe) ** 2, dtype=np.float64))
                _, _, _, info = env.step(executed)
                values = (
                    before, env.positions.copy(), flow, safe, g, w, executed, delta_squared,
                    first_status, second_status, first_retry, second_retry, env.stuck_timer, info["termination"],
                )
                for name, value in zip(histories[i], values):
                    histories[i][name].append(value)
        for i, task in enumerate(tasks):
            history = {name: np.asarray(values) for name, values in histories[i].items()}
            filename = f"{chunk_start + i:05d}.npz"
            j_def = float(config.dt * np.sum(history["delta_u_squared"], dtype=np.float64))
            np.savez_compressed(raw / filename, **history, eta=np.asarray(task["eta"]), seed=task["seed"], J_def=j_def)
            records.append({
                **task, "outcome": str(history["event"][-1]) if errors[i] is None else None,
                "execution_error": errors[i], "steps": len(history["event"]),
                "terminal_step": envs[i].step_count, "J_def": j_def if errors[i] is None else None,
                "partial_J_def": j_def if errors[i] is not None else None,
                "file": f"raw/{args.stage}/{filename}", "sha256": sha(raw / filename),
            })
            total_steps += len(history["event"])
        (raw / "partial.json").write_text(json.dumps(records))
        if time.monotonic() - last > 20 or len(records) == len(jobs):
            print(json.dumps({"completed": len(records), "total": len(jobs), "elapsed_s": round(time.monotonic() - started, 1), "outcomes": dict(Counter(row["outcome"] for row in records))}), flush=True)
            last = time.monotonic()
    assert sha(HERE / "protocol.json") == locked
    result = {
        "stage": args.stage, "records": records, "new_rollouts": len(records),
        "physical_steps": total_steps, "elapsed_s": time.monotonic() - started,
        "device": [str(device) for device in jax.devices()], "batch_size": args.batch,
        "protocol_sha256": locked, "jobs_sha256": sha(HERE / args.jobs),
    }
    (raw / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
