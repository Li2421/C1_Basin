"""Hard gate: exact augmented-state, Flow, projection, and one-step replay."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v1.build_states import anchor_source, replay_actions, restore_full, sha
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = [json.loads(line) for line in (HERE / "state_manifest.jsonl").read_text().splitlines()]
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    policy, _ = load_policy(Path(protocol["checkpoint"]))
    by_category = {category: [row for row in states if row["category"] == category] for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")}
    selected = [by_category["NORMAL"][0], by_category["NORMAL"][-1], by_category["PRE_DEADLOCK"][0], by_category["PRE_DEADLOCK"][-1], by_category["RECOVERY"][0], by_category["RECOVERY"][7], by_category["RECOVERY"][-1]]
    checks = []
    maxima = {name: 0.0 for name in (
        "positions", "velocities", "history", "u_flow", "u_safe", "positions_after",
        "stuck_timer_after", "window_progress_after", "integration",
    )}
    historical_cache_flow_errors = []

    def reconstruct_source_env(row: dict):
        source_path = Path(row["source_path"])
        with np.load(source_path) as source:
            if row["source_type"] == "baseline":
                return replay_actions(np.asarray(source["initial_positions"]), np.asarray(source["executed_velocity"]), int(row["source_local_step"]), config)
            if row["source_type"] == "qualification":
                return replay_actions(np.asarray(source["positions_before"][0]), np.asarray(source["u_exec"]), int(row["source_local_step"]), config)
            anchor_path, anchor_step, _ = anchor_source(row["anchor_state"])
            with np.load(anchor_path) as anchor:
                env = replay_actions(np.asarray(anchor["positions_before"][0]), np.asarray(anchor["u_exec"]), anchor_step, config)
            for action in np.asarray(source["u_exec"][:int(row["source_local_step"])]):
                _, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError((row["state_id"], "recovery source terminated before snapshot"))
            return env

    for row in selected:
        state_path = HERE / row["state_file"]
        ref_path = HERE / row["audit_reference_file"]
        if sha(state_path) != row["state_sha256"] or sha(ref_path) != row["audit_reference_sha256"]:
            raise RuntimeError((row["state_id"], "artifact hash mismatch"))
        env = restore_full(state_path, config)
        source_env = reconstruct_source_env(row)
        with np.load(state_path) as saved, np.load(ref_path) as ref:
            maxima["positions"] = max(maxima["positions"], float(np.max(np.abs(env.positions - source_env.positions))))
            maxima["velocities"] = max(maxima["velocities"], float(np.max(np.abs(env.velocities - source_env.velocities))))
            tail = np.asarray(env.distance_history[int(saved["history_start_step"]):], dtype=np.float64)
            maxima["history"] = max(maxima["history"], float(np.max(np.abs(tail - saved["error_history"]))))
            # Gate on a newly frozen current-deployment random sample evaluated
            # independently on the source-replayed and restored environments.
            episode_key = jax.random.fold_in(jax.random.PRNGKey(95300001), int(row["rng_namespace"]))
            step_key = jax.random.fold_in(episode_key, int(row["absolute_step"]))
            raw = np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]), seed=step_key)[0])
            source_raw = np.asarray(policy.sample_actions(jnp.asarray(source_env.observation()[None]), seed=step_key)[0])
            # Frozen current CL semantics use the explicit bounded_nominal
            # interface, regardless of optional helper methods on the agent.
            flow = bounded_nominal(raw, config.max_speed)
            source_flow = bounded_nominal(source_raw, config.max_speed)
            maxima["u_flow"] = max(maxima["u_flow"], float(np.max(np.abs(flow - source_flow))))
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            source_A, source_lower, _ = barrier_constraints(source_env.snapshot(), cbf)
            source_safe, _, _, _ = project_velocity_with_retry(source_flow, source_A, source_lower, config.max_speed, cbf)
            maxima["u_safe"] = max(maxima["u_safe"], float(np.max(np.abs(safe - source_safe))))
            before = env.positions.copy()
            _, _, done, info = env.step(safe)
            _, _, source_done, source_info = source_env.step(source_safe)
            maxima["positions_after"] = max(maxima["positions_after"], float(np.max(np.abs(env.positions - source_env.positions))))
            maxima["integration"] = max(maxima["integration"], float(np.max(np.abs(env.positions - before - config.dt * safe))))
            maxima["stuck_timer_after"] = max(maxima["stuck_timer_after"], abs(float(info["stuck_timer"]) - float(source_info["stuck_timer"])))
            if np.isfinite(info["window_progress"]).all() and np.isfinite(source_info["window_progress"]).all():
                maxima["window_progress_after"] = max(maxima["window_progress_after"], float(np.max(np.abs(info["window_progress"] - source_info["window_progress"]))))
            # Historical baseline caches were sampled in x32, while the current
            # diagnostics deploy x64. Record (but do not gate on) that provenance
            # difference; the exact state/transition audit above uses current law.
            old_key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(int(row["source_seed"])), int(row["source_rng_id"])), int(row["absolute_step"]))
            old_flow = bounded_nominal(np.asarray(policy.sample_actions(jnp.asarray(source_env.observation()[None]), seed=old_key)[0]), config.max_speed)
            historical_cache_flow_errors.append({"state_id": row["state_id"], "error": float(np.max(np.abs(old_flow - ref["u_flow"])))})
            checks.append({
                "state_id": row["state_id"], "category": row["category"],
                "positions_error": float(np.max(np.abs(env.positions - source_env.positions))),
                "u_flow_error": float(np.max(np.abs(flow - source_flow))),
                "u_safe_error": float(np.max(np.abs(safe - source_safe))),
                "event_source": str(source_info["termination"]), "event_replayed": str(info["termination"]),
                "done_source": bool(source_done), "done_replayed": bool(done),
            })
    tolerances = {"u_flow": 2e-6, "u_safe": 2e-6, "window_progress_after": 2e-12}
    failures = []
    for name, value in maxima.items():
        if value > tolerances.get(name, 2e-12):
            failures.append({"quantity": name, "error": value, "tolerance": tolerances.get(name, 2e-12)})
    for row in checks:
        if row["event_source"] != row["event_replayed"] or row["done_source"] != row["done_replayed"]:
            failures.append({"state_id": row["state_id"], "event_mismatch": [row["event_source"], row["event_replayed"]]})
    result = {
        "status": "PASS" if not failures else "FAIL_STOP_DATASET_CONSTRUCTION",
        "representative_snapshot_count": len(selected), "categories_checked": sorted(by_category),
        "max_absolute_errors": maxima, "tolerances": tolerances,
        "checks": checks, "failures": failures,
        "historical_cache_flow_provenance_errors_non_gating": historical_cache_flow_errors,
        "historical_cache_note": "Old baseline cache used x32 Flow inference; current diagnostic deployment and oracle runner use x64. Exact restoration is gated on identical current-law samples from source-replayed versus serialized/restored state.",
        "restored_fields": [
            "positions", "velocities", "step", "41-sample error-history tail and absolute start",
            "candidate_since", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock",
            "success/deadlock/wall/agent latches", "done",
        ],
        "same_randomness_protocol": "fold_in(fold_in(PRNGKey(95300001), state_rng_namespace), absolute_step), current x64 deployment",
        "source_hashes": protocol["frozen_hashes"],
    }
    (HERE / "restoration_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checked": len(selected), "max_errors": maxima}, indent=2))
    if failures:
        raise SystemExit("Exact restoration gate failed; no oracle rollout is authorized")


if __name__ == "__main__":
    main()
