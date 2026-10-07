"""Pre-bulk exact restoration, action, and branch-semantics checks."""

from __future__ import annotations

import json
import sys
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


def env_state(env):
    return {
        "positions": env.positions.copy(), "velocities": env.velocities.copy(),
        "distance_history": np.asarray(env.distance_history).copy(), "step": env.step_count,
        "candidate_since": env.candidate_since, "ever_candidate_deadlock": env.ever_candidate_deadlock,
        "done": env.done,
    }


def differences(a, b):
    def max_abs(left, right):
        left = np.asarray(left); right = np.asarray(right)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            return float("inf"), False
        finite = np.isfinite(left) & np.isfinite(right)
        value = 0.0 if not np.any(finite) else float(np.max(np.abs(left[finite] - right[finite])))
        return value, bool(np.array_equal(left, right, equal_nan=True))
    pos, pos_equal = max_abs(a["positions"], b["positions"])
    vel, vel_equal = max_abs(a["velocities"], b["velocities"])
    hist, hist_equal = max_abs(a["distance_history"], b["distance_history"])
    return {
        "positions": pos, "velocities": vel, "history": hist,
        "arrays_exact_equal_including_nan_pattern": bool(pos_equal and vel_equal and hist_equal),
        "scalars_equal": bool(all(a[key] == b[key] for key in ("step", "candidate_since", "ever_candidate_deadlock", "done"))),
    }


def one_step(state, eta, seed, force_safe, policy, config, cbf):
    env = restore_full(Path(state["state_file"]), config)
    observation = env.observation()
    key0 = jax.random.fold_in(jax.random.PRNGKey(seed), state["rng_namespace"])
    step_key = jax.random.fold_in(key0, env.step_count)
    action = np.asarray(policy.sample_actions(jnp.asarray(observation[None]), seed=step_key)[0], dtype=np.float64)
    flow = bounded_nominal(action, config.max_speed)
    A, lower, _ = barrier_constraints(env.snapshot(), cbf)
    safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
    raw = np.zeros_like(safe) if force_safe else DiagnosticCorrector(DiagnosticPhi(*eta))(observation, safe, config.max_speed)
    executed, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
    env.step(executed)
    return {"flow": flow, "safe": safe, "raw": raw, "executed": executed, "next": env_state(env)}


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    plan = json.loads((HERE / "audit_plan.json").read_text())
    states = {row["state_id"]: row for row in plan["states"]}
    config = Config(**plan["environment"]); cbf = CBFConfig()
    policy, _ = load_policy(Path(plan["checkpoint"]))
    seed = plan["seeds_initial"][0]
    nz = states["RBV_Q_pair228_m080_s95401001_p018"]
    z = states["RBV_Q_pair228_m080_s95401003_p030"]

    i1 = one_step(nz, nz["eta_best"], seed, False, policy, config, cbf)
    i2 = one_step(nz, nz["eta_best"], seed, False, policy, config, cbf)
    n = one_step(nz, nz["eta_best"], seed, True, policy, config, cbf)
    zi = one_step(z, z["eta_best"], seed, False, policy, config, cbf)
    zn = one_step(z, z["eta_best"], seed, True, policy, config, cbf)

    prior = None
    for path in (ROOT / "diagnostics/gphi_training_dataset_v4/raw").rglob("records.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("state_id") == nz["state_id"] and row.get("seed") == seed and row.get("eta") == nz["eta_best"]:
                prior = row; prior["_path"] = str(path); break
        if prior is not None: break
    if prior is None:
        raise RuntimeError("existing oracle first-step record not found")
    fields = {key: float(np.max(np.abs(i1[local] .reshape(-1) - np.asarray(prior["first_step"][saved]))))
              for key, local, saved in (
                  ("u_flow", "flow", "u_flow"), ("u_safe", "safe", "u_safe"),
                  ("g_raw", "raw", "g_raw"), ("u_exec", "executed", "u_exec"))}
    deterministic = differences(i1["next"], i2["next"])
    zero_equal = differences(zi["next"], zn["next"])
    checks = {
        "existing_oracle_record": prior["_path"],
        "nonzero_first_step_max_abs_errors": fields,
        "deterministic_one_step_replay": deterministic,
        "branch_N_exec_equals_u_safe": float(np.max(np.abs(n["executed"] - n["safe"]))),
        "zero_eta_branch_next_state_equality": zero_equal,
        "nonzero_branch_current_action_difference": float(np.linalg.norm(i1["executed"] - n["executed"])),
    }
    passed = (
        max(fields.values()) < 1e-10 and max(deterministic[k] for k in ("positions", "velocities", "history")) < 1e-12
        and deterministic["scalars_equal"] and deterministic["arrays_exact_equal_including_nan_pattern"]
        and checks["branch_N_exec_equals_u_safe"] < 1e-10
        and max(zero_equal[k] for k in ("positions", "velocities", "history")) < 1e-12
        and zero_equal["scalars_equal"] and zero_equal["arrays_exact_equal_including_nan_pattern"]
        and checks["nonzero_branch_current_action_difference"] > 1e-8
    )
    checks["passed"] = bool(passed)
    (HERE / "prebulk_smoke_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not passed:
        raise AssertionError("pre-bulk smoke failed")


if __name__ == "__main__":
    main()
