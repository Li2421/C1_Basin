"""Matched 64-stream ZERO versus frozen one-shot G_eta evaluation on fresh32."""

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
PLAN = HERE / "fresh32_zero_prediction_plan.json"
ASSETS = HERE / "frozen_asset_hashes.json"
RAW = HERE / "raw/fresh32_zero_prediction.jsonl"
sys.path[:0] = [str(SYS), str(ROOT), str(ROOT / "diagnostics/gphi_closed_loop_pilot_v1")]

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_content(value: dict) -> None:
    expected = value["content_sha256"]
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    observed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if observed != expected: raise RuntimeError("semantic hash mismatch")


def write_rows(rows: list[dict]) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW.with_name(f".{RAW.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    os.replace(tmp, RAW)


def outcome(env, error) -> str:
    if error is not None: return "execution_error"
    value = env.summary()
    if value["wall_collision"] or value["agent_collision"]: return "collision"
    if value["success"]: return "success"
    if value["deadlock"]: return "deadlock"
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu"); parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True); jax.config.update("jax_platform_name", args.device)
    plan = json.loads(PLAN.read_text()); verify_content(plan)
    assets = json.loads(ASSETS.read_text()); verify_content(assets)
    for record in assets.values():
        if isinstance(record, dict) and "path" in record and "sha256" in record and sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((record["path"], "asset hash mismatch"))
    source = json.loads((ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/fresh_wide_manifest.json").read_text())
    config = Config(**source["environment"]); cbf = CBFConfig(**source["cbf"])
    policy, provenance = load_policy(Path(assets["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != source["environment"]: raise RuntimeError("Flow environment mismatch")
    sample = jax.jit(jax.vmap(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    existing = [] if not RAW.exists() else [json.loads(line) for line in RAW.read_text().splitlines()]
    completed = {(r["episode_index"], r["condition"], r["seed"]) for r in existing}
    rows = list(existing); started = time.monotonic(); new_steps = new_rollouts = 0

    for episode in plan["selected_episodes"]:
        episode_index = int(episode["episode_index"]); namespace = 500000 + episode_index
        for condition in plan["conditions"]:
            tasks = [{"seed": int(seed)} for seed in plan["robust_seeds"] if (episode_index, condition, int(seed)) not in completed]
            for begin in range(0, len(tasks), args.batch):
                chunk = tasks[begin:begin + args.batch]; n = len(chunk)
                envs = [GiveWayEnv(config) for _ in chunk]
                for env in envs: env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
                frozen_eta = np.zeros(3, dtype=np.float64) if condition == "ZERO" else np.asarray(episode["frozen_eta_hat"], dtype=np.float64)
                correctors = [DiagnosticCorrector(DiagnosticPhi(*frozen_eta.tolist())) for _ in chunk]
                eta_hats: list[np.ndarray | None] = [frozen_eta.copy() for _ in chunk]
                raw_norm: list[np.ndarray | None] = [None] * n; errors = [None] * n
                keys = [np.asarray(jax.random.fold_in(jax.random.PRNGKey(task["seed"]), namespace)) for task in chunk]
                while len(keys) < args.batch: keys.append(keys[-1])
                key0 = jnp.asarray(np.asarray(keys)); observations = np.zeros((args.batch, 2, 10), dtype=np.float32); steps = np.zeros(args.batch, dtype=np.uint32)
                jdef = np.zeros(n); first_retry = np.zeros(n, dtype=int); second_retry = np.zeros(n, dtype=int)
                while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
                    for i, env in enumerate(envs):
                        if not env.done and errors[i] is None: observations[i] = env.observation(); steps[i] = env.step_count
                    actions = np.asarray(sample(jnp.asarray(observations), fold(key0, jnp.asarray(steps))))
                    contexts = {}
                    for i, env in enumerate(envs):
                        if env.done or errors[i] is not None: continue
                        try:
                            obs = np.asarray(observations[i], dtype=np.float64); flow = bounded_nominal(actions[i], config.max_speed)
                            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                            safe, _, retry1, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                            contexts[i] = (obs, safe, A, lower, retry1)
                        except Exception as exc: errors[i] = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
                    for i, env in enumerate(envs):
                        if i not in contexts or errors[i] is not None: continue
                        obs, safe, A, lower, retry1 = contexts[i]
                        try:
                            raw_correction = np.asarray(correctors[i](obs, safe, config.max_speed), dtype=np.float64)
                            executed, _, retry2, _ = project_velocity_with_retry(safe + raw_correction, A, lower, config.max_speed, cbf)
                            jdef[i] += config.dt * float(np.sum((executed - safe) ** 2)); first_retry[i] += int(retry1); second_retry[i] += int(retry2)
                            env.step(executed); new_steps += 1
                        except Exception as exc: errors[i] = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
                for i, (env, task) in enumerate(zip(envs, chunk)):
                    eta = eta_hats[i]; raw = raw_norm[i]
                    rows.append({
                        "schema": "fresh32_zero_prediction_rollout_v1", "episode_index": episode_index,
                        "source_id": episode["source_id"], "condition": condition, "seed": task["seed"], "rng_namespace": namespace,
                        "eta": eta.tolist(), "eta_norm": float(np.linalg.norm(eta)),
                        "eta_normalized_raw": None if raw is None else raw.tolist(),
                        "eta_clipped": False if raw is None else bool(np.any((raw < 0) | (raw > 1))),
                        "outcome": outcome(env, errors[i]), "success": outcome(env, errors[i]) == "success",
                        "terminal_step": int(env.step_count), "physical_steps": int(env.step_count), "J_def": float(jdef[i]),
                        "first_projection_retries": int(first_retry[i]), "second_projection_retries": int(second_retry[i]),
                        "execution_error": errors[i], "plan_sha256": sha(PLAN), "assets_sha256": sha(ASSETS),
                    }); new_rollouts += 1
                write_rows(rows)
            print(json.dumps({"episode": episode_index, "condition": condition, "rows": len(rows), "new_steps": new_steps}), flush=True)
    runtime = {
        "schema": "fresh32_zero_prediction_runtime_v1", "device": [str(x) for x in jax.devices()], "batch": args.batch,
        "new_rollouts": new_rollouts, "new_physical_steps": new_steps, "wall_seconds": time.monotonic() - started,
        "outcomes": dict(Counter(r["outcome"] for r in rows)), "records": len(rows), "records_sha256": sha(RAW),
        "plan_sha256": sha(PLAN), "assets_sha256": sha(ASSETS), "host": platform.node(),
    }
    path = HERE / "runtime_fresh32_zero_prediction.json"; path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
