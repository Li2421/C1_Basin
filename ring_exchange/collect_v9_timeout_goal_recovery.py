"""Development-only terminal/goal-near drift recovery for local-frame Ring.

This is deliberately *not* collision targeting: it re-queries only safe states
from development policy timeouts, records their rollout/time provenance, and
stores every accepted continuation as ``uniform_recovery``.  In particular it
never opens a test rollout archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.dev_closed_loop import load_dev_nominal_cases
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions

from .environment import LocalFrameConfig, RingExchangeEnv
from .local_frame import local_actions_to_world, local_observation
from .protocol_v6 import RingExchangeLocalFrameScenario


def _write(path: Path, continuation, initial_state, rollout_id: str, audit: RecoveryAudit,
           anchor_class: str) -> dict:
    states = np.asarray(continuation.states)
    observations = np.asarray(continuation.observations, dtype=np.float32)
    actions = np.asarray(continuation.actions, dtype=np.float32)
    digest = _digest(states, observations, actions)
    metadata = dict(continuation.metadata)
    metadata.update(success=True, terminal_reason=continuation.terminal_reason,
                    rollout_id=rollout_id, split="train", source="uniform_recovery",
                    trajectory_digest=digest, recovery_anchor_class=anchor_class)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, schema=np.asarray(TRAJECTORY_SCHEMA), states=states, observations=observations,
        actions=actions, initial_state_json=np.asarray(json.dumps(_jsonable(initial_state), sort_keys=True)),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
        recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__), sort_keys=True)),
    )
    return {"file": str(path.relative_to(path.parents[2])), "rollout_id": rollout_id,
            "split": "train", "source": "uniform_recovery", "trajectory_digest": digest,
            "length": int(len(actions))}


def _anchors(errors: np.ndarray, *, goal_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Dense safe final window plus bounded uniform goal-near anchors."""
    total = len(errors)
    terminal = np.arange(max(0, total - 121), total, 4, dtype=np.int64)
    goal_near = np.flatnonzero((errors.min(axis=1) <= 0.35) &
                               ~(errors <= goal_tolerance).all(axis=1))
    if len(goal_near) > 30:
        goal_near = goal_near[np.unique(np.linspace(0, len(goal_near) - 1, 30, dtype=np.int64))]
    return terminal, goal_near.astype(np.int64)


def collect(source: str | Path, output: str | Path, checkpoint: str | Path, evaluation: str | Path,
            *, seed: int = 3126, expected_timeouts: int = 12) -> dict:
    source, output, checkpoint, evaluation = (Path(value).resolve() for value in (source, output, checkpoint, evaluation))
    if output.exists():
        raise FileExistsError(output)
    manifest, cases, selection_audit = load_dev_nominal_cases(source, expected_count=30)
    evaluation_report = json.loads(evaluation.read_text())
    if evaluation_report.get("dev_selection_audit", {}).get("opened_test_archives") != 0:
        raise ValueError("refusing an evaluation without a no-test audit")
    timeout_ids = {str(row["rollout_id"]) for row in evaluation_report["rollouts"]
                   if row.get("termination") == "timeout"}
    if len(timeout_ids) != expected_timeouts:
        raise ValueError(f"expected {expected_timeouts} timeout ids, found {len(timeout_ids)}")
    config = LocalFrameConfig(**manifest["scenario_config"])
    agent, _ = load_checkpoint(checkpoint, expected_environment_fingerprint=manifest["environment_fingerprint"])
    scenario = RingExchangeLocalFrameScenario(config=config, perturb_position_std=.035, perturb_velocity_std=.025)

    # This copy preserves the independently-created nominal/dev/test split;
    # subsequent reads are confined to selected development nominal archives.
    shutil.copytree(source, output)
    base_key = jax.random.PRNGKey(seed)
    records, source_rows = [], []
    attempted = accepted = invalid = expert_failed = serial = 0
    terminal_anchor_count = goal_anchor_count = 0
    for case_index, case in enumerate(cases):
        if case.rollout_id not in timeout_ids:
            continue
        env = RingExchangeEnv(config)
        initial = case.initial_state
        env.reset(np.asarray(initial["positions"], dtype=np.float64),
                  velocities=np.asarray(initial["velocities"], dtype=np.float64),
                  goals=np.asarray(initial["goals"], dtype=np.float64))
        episode_key = jax.random.fold_in(base_key, case_index)
        snapshots, errors = [env.augmented_state()], [np.linalg.norm(env.goals - env.positions, axis=-1)]
        termination = "timeout"
        for timestep in range(config.max_steps):
            local = np.asarray(sample_bounded_actions(
                agent, local_observation(env.positions, env.velocities, env.goals, config)[None],
                jax.random.fold_in(episode_key, timestep))[0], dtype=np.float64)
            norms = np.linalg.norm(local, axis=-1, keepdims=True)
            local *= np.minimum(1.0, (config.max_speed - 1e-8) / np.maximum(norms, 1e-12))
            _, _, done, info = env.step(local_actions_to_world(local, env.positions))
            termination = str(info["termination"])
            # Timeouts have valid terminal positions.  Collision states would
            # never be admitted, even if this collector's input changed.
            if termination != "collision":
                snapshots.append(env.augmented_state())
                errors.append(np.linalg.norm(env.goals - env.positions, axis=-1))
            if done:
                break
        if termination != "timeout":
            raise RuntimeError(f"reproduced {case.rollout_id} as {termination}, not timeout")
        error_array = np.asarray(errors)
        terminal_times, goal_times = _anchors(error_array, goal_tolerance=config.goal_tolerance)
        selected: dict[int, str] = {int(time): "terminal_window" for time in terminal_times}
        # Preserve the denser terminal label where categories overlap.
        for time in goal_times:
            selected.setdefault(int(time), "goal_near_not_jointly_converged")
        made = 0
        for source_time, anchor_class in sorted(selected.items()):
            snapshot = snapshots[source_time]
            state = {"positions": np.asarray(snapshot["positions"]),
                     "velocities": np.asarray(snapshot["last_applied_velocity"]),
                     "goals": np.asarray(snapshot["goals"]), "split": "dev", "recovery": True,
                     "source_time": source_time}
            perturbation_seed = int(np.random.SeedSequence([seed, case_index, source_time]).generate_state(1)[0])
            attempted += 1
            perturbed = scenario.perturb_state(state, np.random.default_rng(perturbation_seed))
            if not scenario.valid_state(perturbed):
                invalid += 1
                continue
            continuation = scenario.expert(perturbed, np.random.default_rng(perturbation_seed))
            if not continuation.success:
                expert_failed += 1
                continue
            audit = RecoveryAudit(f"dev_policy_v9_timeout_{case_index:03d}", "dev", source_time,
                                  None, None, None, perturbation_seed, True)
            rollout_id = f"train_uniform_v9_timeout_goal_{serial:06d}"
            records.append(_write(output / "rollouts" / "train" / f"{rollout_id}.npz", continuation,
                                  perturbed, rollout_id, audit, anchor_class))
            serial += 1
            accepted += 1
            made += 1
            if anchor_class == "terminal_window":
                terminal_anchor_count += 1
            else:
                goal_anchor_count += 1
        source_rows.append({"source_rollout_id": f"dev_policy_v9_timeout_{case_index:03d}",
                            "nominal_rollout_id": case.rollout_id, "terminal": termination,
                            "valid_policy_states": len(snapshots), "terminal_window_candidates": len(terminal_times),
                            "goal_near_candidates": len(goal_times), "accepted": made})
    if len(source_rows) != expected_timeouts:
        raise AssertionError("not all audited development timeouts were collected")
    new_manifest = json.loads((output / "manifest.json").read_text())
    new_manifest["files"].extend(records)
    new_manifest["counts"]["source"]["uniform_recovery"] += accepted
    new_manifest["counts"]["split"]["train"] += accepted
    report = {"base_source": str(source), "policy_checkpoint": str(checkpoint),
              "evaluation_summary": str(evaluation), "representation": "radial_tangential_local_v7",
              "source": "development_policy_timeout_safe_terminal_and_goal_near_requery",
              "timeout_rollouts": len(source_rows), "attempted": attempted, "accepted": accepted,
              "accepted_terminal_window": terminal_anchor_count, "accepted_goal_near": goal_anchor_count,
              "invalid_perturbation": invalid, "expert_failed_skipped": expert_failed,
              "source_rollouts": source_rows, "dev_selection_audit": selection_audit, "test_opened": False}
    new_manifest.setdefault("extra_report", {})["v9_timeout_goal_uniform_recovery"] = report
    (output / "manifest.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n")
    report["output"] = str(output)
    report["output_manifest_sha256"] = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    (output / "v9_timeout_goal_uniform_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.source, args.output, args.checkpoint, args.evaluation), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
