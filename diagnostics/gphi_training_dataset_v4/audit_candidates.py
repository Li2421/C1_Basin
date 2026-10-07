"""Exact V4 boundary-state restoration and matched one-step audit."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def differences(a, b) -> dict:
    return {
        "positions": float(np.max(np.abs(a.positions - b.positions))),
        "velocities": float(np.max(np.abs(a.velocities - b.velocities))),
        "observation": float(np.max(np.abs(a.observation() - b.observation()))),
        "history": float(np.max(np.abs(np.asarray(a.distance_history[-41:]) - np.asarray(b.distance_history[-41:])))),
        "stuck_timer": abs(float(a.stuck_timer) - float(b.stuck_timer)),
        "max_stuck_timer": abs(float(a.max_stuck_timer) - float(b.max_stuck_timer)),
    }


def main() -> None:
    started = time.monotonic()
    jax.config.update("jax_enable_x64", True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    candidates = read_jsonl(HERE / "candidate_state_manifest.jsonl")
    v3_states = {row["state_id"]: row for row in read_jsonl(V3 / "state_manifest.jsonl")}
    single = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    results = []
    maxima = {}
    for row in candidates:
        anchor = v3_states[row["anchor_state"]]
        replay = restore_full(V3 / anchor["state_file"], config)
        with np.load(row["source_path"]) as source:
            actions = np.asarray(source["u_exec"])
        for action in actions[: row["source_local_step"]]:
            replay.step(action)
        restored = restore_full(HERE / row["state_file"], config)
        state_diff = differences(replay, restored)
        scalar_equal = replay.step_count == restored.step_count and replay.candidate_since == restored.candidate_since and replay.ever_candidate_deadlock == restored.ever_candidate_deadlock and replay.done == restored.done
        with np.load(HERE / row["audit_reference_file"]) as ref:
            observation = restored.observation()
            episode_key = jax.random.fold_in(jax.random.PRNGKey(row["source_seed"]), row["anchor_rng_namespace"])
            step_key = jax.random.fold_in(episode_key, restored.step_count)
            flow = bounded_nominal(np.asarray(single(jnp.asarray(observation), step_key)), config.max_speed)
            A, lower, _ = barrier_constraints(restored.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            raw = DiagnosticCorrector(DiagnosticPhi(*row["source_eta"]))(observation, safe, config.max_speed)
            executed, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
            action_diff = {
                "u_flow": float(np.max(np.abs(flow - ref["u_flow"]))),
                "u_safe": float(np.max(np.abs(safe - ref["u_safe"]))),
                "g_raw": float(np.max(np.abs(raw - ref["g_raw"]))),
                "u_exec": float(np.max(np.abs(executed - ref["u_exec"]))),
            }
        expected_next = restore_full(V3 / anchor["state_file"], config)
        for action in actions[: row["source_local_step"] + 1]:
            expected_next.step(action)
        restored.step(executed)
        next_diff = differences(expected_next, restored)
        next_scalar_equal = expected_next.step_count == restored.step_count and expected_next.candidate_since == restored.candidate_since and expected_next.ever_candidate_deadlock == restored.ever_candidate_deadlock and expected_next.done == restored.done
        passed = scalar_equal and next_scalar_equal and max(state_diff.values()) <= 1e-12 and max(action_diff.values()) <= 1e-10 and max(next_diff.values()) <= 1e-12
        for prefix, values in (("state", state_diff), ("action", action_diff), ("next", next_diff)):
            for name, value in values.items():
                maxima[f"{prefix}_{name}"] = max(maxima.get(f"{prefix}_{name}", 0.0), value)
        results.append({"state_id": row["state_id"], "source_group": row["source_group"], "state_differences": state_diff, "action_differences": action_diff, "next_differences": next_diff, "scalar_equal": scalar_equal, "next_scalar_equal": next_scalar_equal, "passed": passed})
    report = {
        "status": "PASS" if all(row["passed"] for row in results) else "FAIL",
        "candidate_states": len(results), "passed_states": sum(row["passed"] for row in results),
        "rejected_states": sum(not row["passed"] for row in results),
        "matched_randomness": True,
        "checks": ["physical state", "history", "monitor/latch/timers", "u_Flow", "u_safe", "one-step transition"],
        "maximum_absolute_differences": maxima, "per_state": results,
        "elapsed_s": time.monotonic() - started,
    }
    (HERE / "restoration_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_state"}, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("restoration audit failed")


if __name__ == "__main__":
    main()
