"""Batched frozen eta-zero continuations for confidence-aware gate labels."""

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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu"); parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True); jax.config.update("jax_platform_name", args.device)
    protocol_path = DATA / "protocol.json"; protocol = json.loads(protocol_path.read_text()); protocol_hash = sha(protocol_path)
    arms = json.loads((HERE / args.arms).read_text())
    states = {row["state_id"]: row for row in (json.loads(line) for line in (DATA / "state_manifest.jsonl").read_text().splitlines())}
    config = Config(**protocol["environment"]); cbf = CBFConfig(); policy, _ = load_policy(Path(protocol["checkpoint"]))
    output = HERE / "raw" / args.stage; output.mkdir(parents=True, exist_ok=False); records_path = output / "records.jsonl"
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single)); fold = jax.jit(jax.vmap(jax.random.fold_in))
    corrector = DiagnosticCorrector(DiagnosticPhi(0., 0., 0.))
    all_records = []; physical_steps = 0; started = time.monotonic(); last = started
    max_rounds = max((len(arm["seeds"]) for arm in arms), default=0)
    print(json.dumps({"stage": args.stage, "states": len(arms), "planned_rollouts": sum(len(arm["seeds"]) for arm in arms), "device": [str(x) for x in jax.devices()]}), flush=True)
    for seed_index in range(max_rounds):
        tasks = [{**arm, "seed": int(arm["seeds"][seed_index]), "seed_index": seed_index} for arm in arms if seed_index < len(arm["seeds"])]
        for chunk_start in range(0, len(tasks), args.batch):
            chunk = tasks[chunk_start:chunk_start+args.batch]
            envs = [restore_full(DATA / states[task["state_id"]]["state_file"], config) for task in chunk]
            keys = [np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), int(states[task["state_id"]]["rng_namespace"]))) for task in chunk]
            while len(keys) < args.batch:
                keys.append(keys[-1])
            keys0 = jnp.asarray(np.asarray(keys)); observations = np.zeros((args.batch, 2, 10), np.float32); steps = np.zeros(args.batch, np.uint32)
            errors = [None]*len(chunk); first = [None]*len(chunk)
            while any(not env.done and errors[index] is None for index, env in enumerate(envs)):
                for index, env in enumerate(envs):
                    observations[index] = env.observation(); steps[index] = env.step_count
                actions = np.asarray(sample(jnp.asarray(observations), fold(keys0, jnp.asarray(steps))))
                for index, env in enumerate(envs):
                    if env.done or errors[index] is not None:
                        continue
                    flow = bounded_nominal(actions[index], config.max_speed); A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                    try:
                        safe, first_status, first_retry, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        raw = corrector(observations[index], safe, config.max_speed)
                        executed, second_status, second_retry, _ = project_velocity_with_retry(safe+raw, A, lower, config.max_speed, cbf)
                    except Exception as exc:
                        errors[index] = {"type": type(exc).__name__, "message": str(exc), "absolute_step": env.step_count}; continue
                    if first[index] is None:
                        first[index] = {"u_flow": flow.reshape(-1).tolist(), "u_safe": safe.reshape(-1).tolist(), "g_raw": raw.reshape(-1).tolist(), "u_exec": executed.reshape(-1).tolist(), "first_projection_status": str(first_status), "second_projection_status": str(second_status), "first_retry": bool(first_retry), "second_retry": bool(second_retry)}
                    env.step(executed)
            for index, task in enumerate(chunk):
                summary = envs[index].summary()
                if errors[index] is not None: outcome = "execution_error"
                elif summary["wall_collision"] or summary["agent_collision"]: outcome = "collision"
                elif summary["success"]: outcome = "success"
                elif summary["deadlock"]: outcome = "deadlock"
                else: outcome = "timeout"
                record = {"state_id": task["state_id"], "eta": [0., 0., 0.], "seed": task["seed"], "seed_index": seed_index, "outcome": outcome, "execution_error": errors[index], "steps": int(envs[index].step_count-states[task["state_id"]]["step"]), "terminal_step": int(envs[index].step_count), "first_step": first[index], "confidence_stage": args.stage, "reused": False}
                all_records.append(record); physical_steps += record["steps"]
        records_path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in all_records))
        if time.monotonic()-last > 30 or seed_index == max_rounds-1:
            print(json.dumps({"round": seed_index+1, "records": len(all_records), "outcomes": dict(Counter(row["outcome"] for row in all_records)), "elapsed_s": round(time.monotonic()-started, 1)}), flush=True); last = time.monotonic()
    manifest = {"stage": args.stage, "device": [str(x) for x in jax.devices()], "batch_size": args.batch, "state_count": len(arms), "new_rollouts": len(all_records), "physical_steps": physical_steps, "outcomes": dict(Counter(row["outcome"] for row in all_records)), "elapsed_s": time.monotonic()-started, "protocol_sha256": protocol_hash, "arms_sha256": sha(HERE/args.arms), "records_sha256": sha(records_path)}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    if sha(protocol_path) != protocol_hash:
        raise RuntimeError("V4 protocol changed")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
