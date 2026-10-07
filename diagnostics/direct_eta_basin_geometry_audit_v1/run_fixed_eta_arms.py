"""Run frozen fixed-eta full-horizon arms from exact augmented state snapshots."""

from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYS = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
sys.path[:0] = [str(SYS), str(ROOT)]

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_content(value: dict) -> None:
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    observed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if observed != value.get("content_sha256"): raise RuntimeError("plan semantic hash mismatch")


def atomic_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)); os.replace(tmp, path)


def classify(env, error) -> str:
    if error is not None: return "execution_error"
    result = env.summary()
    if result["wall_collision"] or result["agent_collision"]: return "collision"
    if result["success"]: return "success"
    if result["deadlock"]: return "deadlock"
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu"); parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args(); plan_path = Path(args.plan).resolve()
    jax.config.update("jax_enable_x64", True); jax.config.update("jax_platform_name", args.device)
    plan = json.loads(plan_path.read_text()); verify_content(plan)
    assets_path = HERE / "frozen_asset_hashes.json"; assets = json.loads(assets_path.read_text()); verify_content(assets)
    for record in assets.values():
        if isinstance(record, dict) and "path" in record and "sha256" in record and sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((record["path"], "asset hash mismatch"))
    config_source = json.loads((ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text())
    config = Config(**config_source["environment"]); cbf = CBFConfig(**config_source["cbf"])
    policy, provenance = load_policy(Path(assets["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != config_source["environment"]: raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])); fold = jax.jit(jax.vmap(jax.random.fold_in))
    output = HERE / "raw" / args.stage; records_path = output / "records.jsonl"; runtime_path = output / "runtime.json"
    existing = [] if not records_path.exists() else [json.loads(line) for line in records_path.read_text().splitlines()]
    completed = {(row["arm_id"], int(row["seed"])) for row in existing}; rows = list(existing)
    begun = time.monotonic(); new_rollouts = new_steps = 0
    for arm in plan["arms"]:
        state_path = Path(arm["state_file"])
        if sha(state_path) != arm["state_sha256"]: raise RuntimeError((arm["state_id"], "state hash mismatch"))
        pending = [int(seed) for seed in arm["seeds"] if (arm["arm_id"], int(seed)) not in completed]
        for begin in range(0, len(pending), args.batch):
            seeds = pending[begin:begin + args.batch]; n = len(seeds)
            envs = [restore_full(state_path, config) for _ in seeds]
            if any(env.done or env.step_count != int(arm["absolute_step"]) for env in envs): raise RuntimeError((arm["state_id"], "bad restored state"))
            corrector = DiagnosticCorrector(DiagnosticPhi(*[float(x) for x in arm["eta"]])); errors = [None] * n
            keys = [np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed), int(arm["rng_namespace"]))) for seed in seeds]
            while len(keys) < args.batch: keys.append(keys[-1])
            key0 = jnp.asarray(np.asarray(keys)); observations = np.zeros((args.batch, 2, 10), dtype=np.float32); steps = np.zeros(args.batch, dtype=np.uint32)
            jdef = np.zeros(n); first_retry = np.zeros(n, dtype=int); second_retry = np.zeros(n, dtype=int)
            while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
                for i, env in enumerate(envs):
                    if not env.done and errors[i] is None: observations[i] = env.observation(); steps[i] = env.step_count
                actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(steps))))
                for i, env in enumerate(envs):
                    if env.done or errors[i] is not None: continue
                    try:
                        obs = np.asarray(observations[i], dtype=np.float64); flow = bounded_nominal(actions[i], config.max_speed)
                        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                        safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        correction = np.asarray(corrector(obs, safe, config.max_speed), dtype=np.float64)
                        executed, _, retry2, _ = project_velocity_with_retry(safe + correction, A, lower, config.max_speed, cbf)
                        jdef[i] += config.dt * float(np.sum((executed - safe) ** 2)); first_retry[i] += int(retry1); second_retry[i] += int(retry2)
                        env.step(executed); new_steps += 1
                    except Exception as exc: errors[i] = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
            for i, (env, seed) in enumerate(zip(envs, seeds)):
                result = classify(env, errors[i])
                rows.append({
                    "schema": "direct_eta_fixed_arm_rollout_v1", "stage": args.stage, "arm_id": arm["arm_id"],
                    "state_id": arm["state_id"], "eta": arm["eta"], "seed": seed, "rng_namespace": arm["rng_namespace"],
                    "outcome": result, "success": result == "success", "continuation_steps": int(env.step_count - int(arm["absolute_step"])),
                    "terminal_step": int(env.step_count), "J_def": float(jdef[i]), "execution_error": errors[i],
                    "first_projection_retries": int(first_retry[i]), "second_projection_retries": int(second_retry[i]),
                    "plan_sha256": sha(plan_path), "assets_sha256": sha(assets_path),
                }); new_rollouts += 1
            atomic_rows(records_path, rows)
        print(json.dumps({"stage": args.stage, "arm": arm["arm_id"], "records": len(rows), "new_steps": new_steps}), flush=True)
    counts = Counter(row["outcome"] for row in rows)
    runtime = {"schema": "direct_eta_fixed_arm_runtime_v1", "stage": args.stage, "device": [str(x) for x in jax.devices()],
               "batch": args.batch, "new_rollouts": new_rollouts, "new_physical_steps": new_steps,
               "wall_seconds": time.monotonic() - begun, "records": len(rows), "outcomes": dict(counts),
               "records_sha256": sha(records_path), "plan_sha256": sha(plan_path), "assets_sha256": sha(assets_path), "host": platform.node()}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n"); print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__": main()
