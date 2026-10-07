"""Independently audit exact restoration and one-step matched reconstruction."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V2 = ROOT / "diagnostics/gphi_training_dataset_v2"
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


def env_differences(a, b) -> dict[str, float]:
    values = {
        "positions": float(np.max(np.abs(a.positions - b.positions))),
        "velocities": float(np.max(np.abs(a.velocities - b.velocities))),
        "observation": float(np.max(np.abs(a.observation() - b.observation()))),
        "error_history": float(np.max(np.abs(np.asarray(a.distance_history[-41:]) - np.asarray(b.distance_history[-41:])))),
        "stuck_timer": abs(float(a.stuck_timer) - float(b.stuck_timer)),
        "max_stuck_timer": abs(float(a.max_stuck_timer) - float(b.max_stuck_timer)),
    }
    return values


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    candidates = read_jsonl(HERE / "candidate_state_manifest.jsonl")
    v2_states = {row["state_id"]: row for row in read_jsonl(V2 / "state_manifest.jsonl")}
    single = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    rows = []
    maxima = {key: 0.0 for key in (
        "state_positions", "state_velocities", "state_observation", "state_error_history",
        "u_flow", "u_safe", "g_raw", "u_exec", "next_positions", "next_velocities",
        "next_observation", "next_error_history",
    )}
    started = time.monotonic()
    for row in candidates:
        anchor = v2_states[row["anchor_state"]]
        replay = restore_full(V2 / anchor["state_file"], config)
        with np.load(HERE / row["source_path"]) as source:
            actions = np.asarray(source["u_exec"])
        for action in actions[: int(row["source_local_step"])]:
            _, _, done, _ = replay.step(action)
            if done:
                raise AssertionError((row["state_id"], "source replay ended early"))
        restored = restore_full(HERE / row["state_file"], config)
        diffs = env_differences(replay, restored)
        scalar_equal = (
            replay.step_count == restored.step_count
            and replay.candidate_since == restored.candidate_since
            and replay.ever_candidate_deadlock == restored.ever_candidate_deadlock
            and replay.done == restored.done
        )
        with np.load(HERE / row["audit_reference_file"]) as ref:
            observation = restored.observation()
            episode_key = jax.random.fold_in(jax.random.PRNGKey(int(row["source_seed"])), int(row["anchor_rng_namespace"]))
            step_key = jax.random.fold_in(episode_key, restored.step_count)
            flow = bounded_nominal(np.asarray(single(jnp.asarray(observation), step_key)), config.max_speed)
            A, lower, _ = barrier_constraints(restored.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            raw = DiagnosticCorrector(DiagnosticPhi(*row["source_eta"]))(observation, safe, config.max_speed)
            executed, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
            action_diffs = {
                "u_flow": float(np.max(np.abs(flow - ref["u_flow"]))),
                "u_safe": float(np.max(np.abs(safe - ref["u_safe"]))),
                "g_raw": float(np.max(np.abs(raw - ref["g_raw"]))),
                "u_exec": float(np.max(np.abs(executed - ref["u_exec"]))),
            }
        expected_next = restore_full(V2 / anchor["state_file"], config)
        for action in actions[: int(row["source_local_step"]) + 1]:
            expected_next.step(action)
        restored.step(executed)
        next_diffs = env_differences(expected_next, restored)
        next_scalar_equal = (
            expected_next.step_count == restored.step_count
            and expected_next.candidate_since == restored.candidate_since
            and expected_next.ever_candidate_deadlock == restored.ever_candidate_deadlock
            and expected_next.done == restored.done
        )
        passed = scalar_equal and next_scalar_equal and max(diffs.values()) <= 1e-12 and max(action_diffs.values()) <= 1e-10 and max(next_diffs.values()) <= 1e-12
        for key, value in diffs.items():
            maxima[f"state_{key}"] = max(maxima.get(f"state_{key}", 0.0), value)
        for key, value in action_diffs.items():
            maxima[key] = max(maxima[key], value)
        for key, value in next_diffs.items():
            maxima[f"next_{key}"] = max(maxima.get(f"next_{key}", 0.0), value)
        rows.append({
            "state_id": row["state_id"], "source_group": row["source_group"],
            "state_differences": diffs, "action_differences": action_diffs,
            "next_state_differences": next_diffs, "scalar_state_equal": scalar_equal,
            "next_scalar_state_equal": next_scalar_equal, "passed": passed,
        })
    passed_count = sum(row["passed"] for row in rows)
    report = {
        "status": "PASS" if passed_count == len(rows) else "FAIL",
        "candidate_states": len(rows), "passed_states": passed_count,
        "rejected_states": len(rows) - passed_count, "matched_randomness": True,
        "checks": ["augmented state", "monitor/history/latch", "u_Flow", "u_safe", "u_exec", "one subsequent environment transition"],
        "maximum_absolute_differences": maxima, "per_state": rows,
        "elapsed_s": time.monotonic() - started,
    }
    (HERE / "restoration_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_state"}, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("candidate restoration audit failed")


if __name__ == "__main__":
    main()
