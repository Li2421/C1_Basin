"""CPU-only restoration/projection audit; must pass before oracle execution."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import HERE, ROOT, SYSROOT, assert_frozen_sources, read_jsonl, sha256, write_json


def restore_full(path: Path, config):
    from single_integrator.environment import GiveWayEnv
    with np.load(path, allow_pickle=False) as data:
        env = GiveWayEnv(config)
        env.positions = np.asarray(data["positions"], dtype=np.float64).copy()
        env.velocities = np.asarray(data["velocities"], dtype=np.float64).copy()
        env.step_count = int(data["step"])
        env.distance_history = [np.full(2, np.nan) for _ in range(env.step_count + 1)]
        start = int(data["history_start_step"])
        for offset, value in enumerate(np.asarray(data["error_history"], dtype=np.float64)):
            env.distance_history[start + offset] = value.copy()
        candidate = int(data["candidate_since"])
        env.candidate_since = None if candidate < 0 else candidate
        for field in ("stuck_timer", "max_stuck_timer"):
            setattr(env, field, float(data[field]))
        env.ever_candidate_deadlock = bool(data["ever_candidate_deadlock"])
        for field in ("first_success_step", "first_deadlock_step", "first_wall_collision_step", "first_agent_collision_step"):
            value = int(data[field]); setattr(env, field, None if value < 0 else value)
        env.done = bool(data["done"])
    return env


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    hashes = assert_frozen_sources()
    sys.path.insert(0, str(SYSROOT))
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy

    resolved = {
        "cbf": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "projector": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
    }
    expected = {
        "cbf": str((SYSROOT / "single_integrator/cbf.py").resolve()),
        "environment": str((SYSROOT / "single_integrator/environment.py").resolve()),
        "projector": str((ROOT / "diagnostics/success_basin_multimodality/exact_projector.py").resolve()),
    }
    if resolved != expected:
        raise RuntimeError(("module path mismatch", resolved, expected))

    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    states = read_jsonl(HERE / "startup_state_manifest.jsonl")
    # Pre-registered generic audit subset: first, middle, last source plus one
    # source at each decile.  This selection predates labels.
    indices = sorted(set([0, len(states) // 2, len(states) - 1] + [round(q * (len(states) - 1) / 10) for q in range(11)]))
    maxima = {"positions": 0.0, "velocities": 0.0, "history": 0.0, "u_flow": 0.0, "u_safe": 0.0}
    checks = []
    for index in indices:
        row = states[index]
        path = HERE / row["state_file"]
        if sha256(path) != row["state_sha256"]:
            raise RuntimeError((row["state_id"], "snapshot hash mismatch"))
        restored = restore_full(path, config)
        with np.load(row["source_path"], allow_pickle=False) as source:
            replay = GiveWayEnv(config); replay.reset(np.asarray(source["initial_positions"], dtype=np.float64))
            for action in np.asarray(source["executed_velocity"][:row["step"]], dtype=np.float64):
                replay.step(action)
        maxima["positions"] = max(maxima["positions"], float(np.max(np.abs(restored.positions - replay.positions))))
        maxima["velocities"] = max(maxima["velocities"], float(np.max(np.abs(restored.velocities - replay.velocities))))
        a = np.asarray(restored.distance_history, dtype=np.float64)
        b = np.asarray(replay.distance_history, dtype=np.float64)
        maxima["history"] = max(maxima["history"], float(np.max(np.abs(a - b))))

        episode_key = jax.random.fold_in(jax.random.PRNGKey(95810001), int(row["rng_namespace"]))
        step_key = jax.random.fold_in(episode_key, int(row["step"]))
        raw_a = np.asarray(policy.sample_actions(jnp.asarray(restored.observation()[None]), seed=step_key)[0])
        raw_b = np.asarray(policy.sample_actions(jnp.asarray(replay.observation()[None]), seed=step_key)[0])
        flow_a = bounded_nominal(raw_a, config.max_speed); flow_b = bounded_nominal(raw_b, config.max_speed)
        maxima["u_flow"] = max(maxima["u_flow"], float(np.max(np.abs(flow_a - flow_b))))
        A, lower, _ = barrier_constraints(restored.snapshot(), cbf)
        safe_a, _, _, _ = project_velocity_with_retry(flow_a, A, lower, config.max_speed, cbf)
        A2, lower2, _ = barrier_constraints(replay.snapshot(), cbf)
        safe_b, _, _, _ = project_velocity_with_retry(flow_b, A2, lower2, config.max_speed, cbf)
        maxima["u_safe"] = max(maxima["u_safe"], float(np.max(np.abs(safe_a - safe_b))))
        checks.append({"state_id": row["state_id"], "step": row["step"], "real_history_length": len(restored.distance_history), "min_linear_residual": float(np.min(A @ safe_a.reshape(4) - lower))})
    tolerance = 2e-12
    failures = [{"quantity": name, "value": value} for name, value in maxima.items() if value > tolerance]
    result = {
        "status": "PASS" if not failures else "FAIL_STOP_ORACLE",
        "projection_path_locked": resolved == expected,
        "resolved_module_paths": resolved,
        "expected_module_paths": expected,
        "source_hashes": hashes,
        "representative_count": len(checks),
        "max_absolute_errors": maxima,
        "tolerance": tolerance,
        "checks": checks,
        "failures": failures,
    }
    write_json(HERE / "startup_state_integrity_checks.json", result)
    print(json.dumps({"status": result["status"], "checked": len(checks), "maxima": maxima}, indent=2))
    if failures:
        raise SystemExit("startup restoration/projection gate failed")


if __name__ == "__main__":
    main()

