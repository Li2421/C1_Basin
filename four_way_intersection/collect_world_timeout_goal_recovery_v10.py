"""Development-only, physical late-timeout/goal-settling recovery collection.

This is generic local-drift data, not a coordination-mode or agent-collision
targeter: only valid preterminal states from *timeout* policy rollouts are
selected.  The additional dense anchors are late rollout states and physical
goal-near/goal-overshoot states, both determined without labels.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil
from pathlib import Path
import jax
import numpy as np

from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import Config, FourWayIntersectionEnv
from .lane_local import world_to_local
from .protocol import FourWayScenario


def _initial(root, row):
    with np.load(Path(root) / row["file"], allow_pickle=False) as data:
        return json.loads(str(data["initial_state_json"].item()))


def _action(agent, observation, key, max_speed):
    action = np.asarray(sample_bounded_actions(agent, observation[None], key)[0], dtype=np.float64)
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    # Keep source rollout selection numerically identical to the development
    # evaluator; otherwise tiny radial-bound differences can compound.
    return action * np.minimum(1.0, (max_speed - 1e-10) / np.maximum(norm, 1e-12))


def _write(path, continuation, initial, audit, rollout_id, kind):
    states = np.asarray(continuation.states, dtype=np.float64)
    observations = np.asarray(continuation.observations, dtype=np.float32)
    actions = np.asarray(continuation.actions, dtype=np.float32)
    digest = _digest(states, observations, actions)
    metadata = dict(continuation.metadata)
    metadata.update(success=True, terminal_reason=continuation.terminal_reason,
                    rollout_id=rollout_id, split="train", source="uniform_recovery",
                    trajectory_digest=digest, recovery_anchor_kind=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.asarray(TRAJECTORY_SCHEMA), states=states,
                        observations=observations, actions=actions,
                        initial_state_json=np.asarray(json.dumps(_jsonable(initial), sort_keys=True)),
                        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
                        recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__), sort_keys=True)))
    return {"file": str(path.relative_to(path.parents[2])), "rollout_id": rollout_id,
            "split": "train", "source": "uniform_recovery", "trajectory_digest": digest,
            "length": len(actions)}


def collect(source, output, checkpoint, *, seed=10123, late_anchors=50, goal_anchors=24):
    source, output, checkpoint = map(Path, (source, output, checkpoint))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    cfg = Config(**{key: value for key, value in manifest["scenario_config"].items() if key in Config.__dataclass_fields__})
    agent, _ = load_checkpoint(checkpoint, expected_environment_fingerprint=cfg.fingerprint)
    scenario = FourWayScenario(cfg)
    # Frozen test contents are copied as opaque provenance only, never loaded.
    shutil.copytree(source, output)
    dev = [row for row in manifest["files"] if row["split"] == "dev" and row["source"] == "nominal"]
    additions, rollout_audit = [], []
    serial = attempted = accepted = invalid = failed = transitions = 0
    for case, row in enumerate(dev):
        initial = _initial(source, row)
        env = FourWayIntersectionEnv(cfg)
        env.reset(np.asarray(initial["positions"]), np.asarray(initial["velocities"]))
        snapshots, key = [env.augmented_state()], jax.random.PRNGKey(seed + case)
        terminal = "timeout"
        for time in range(cfg.max_steps):
            _, _, done, info = env.step(_action(agent, env.observation(), jax.random.fold_in(key, time), cfg.max_speed))
            terminal = str(info["termination"])
            if done:
                break
            snapshots.append(env.augmented_state())
        # This intentionally does not mine collision rollouts; a future agent
        # collision augmentation would require separate documentation.
        if terminal != "timeout":
            rollout_audit.append({"source_rollout_id": f"dev_policy_world_v9_{case:03d}", "terminal": terminal,
                                  "selected": 0, "reason": "not_timeout"})
            continue
        late_start = int(np.floor(.60 * (len(snapshots) - 1)))
        late_pool = np.arange(late_start, len(snapshots), dtype=int)
        late = np.unique(np.linspace(0, len(late_pool) - 1, min(late_anchors, len(late_pool)), dtype=int))
        late = late_pool[late]
        goal_pool = []
        for time, snapshot in enumerate(snapshots):
            remaining = env.goals - np.asarray(snapshot["positions"])
            near = np.linalg.norm(remaining, axis=1) <= .65
            beyond = world_to_local(remaining)[:, 0] < -cfg.goal_tolerance
            if bool(np.any(near | beyond)):
                goal_pool.append(time)
        goal_indices = (np.unique(np.linspace(0, len(goal_pool) - 1, min(goal_anchors, len(goal_pool)), dtype=int))
                        if goal_pool else np.empty(0, dtype=int))
        candidates = [(int(t), "late_timeout_uniform") for t in late]
        candidates += [(int(goal_pool[i]), "goal_settling_or_overshoot") for i in goal_indices]
        selected = {time: kind for time, kind in candidates}
        accepted_rows = []
        for time, kind in sorted(selected.items()):
            snapshot = snapshots[time]
            state = {"positions": np.asarray(snapshot["positions"], dtype=np.float64),
                     "velocities": np.asarray(snapshot["last_applied_velocity"], dtype=np.float64),
                     "split": "dev", "recovery": True, "source_time": time}
            attempted += 1
            perturbation_seed = int(np.random.SeedSequence([seed, case, time, 0 if kind == "late_timeout_uniform" else 1]).generate_state(1)[0])
            perturbed = scenario.perturb_state(state, np.random.default_rng(perturbation_seed))
            if not scenario.valid_state(perturbed):
                invalid += 1
                continue
            continuation = scenario.expert(perturbed, np.random.default_rng(perturbation_seed))
            if not continuation.success:
                failed += 1
                continue
            audit = RecoveryAudit(source_rollout_id=f"dev_policy_world_v9_{case:03d}", source_split="dev",
                                  source_time=time, collision_type=None, collision_identity=None,
                                  distance_to_collision=None, perturbation_seed=perturbation_seed, expert_success=True)
            # This collector can extend an already extended dataset.  Keep a
            # distinct ID namespace so copied provenance files are never
            # overwritten (and their digest guarantees remain meaningful).
            rollout_id = f"train_world_timeout_goal_extension_{serial:06d}"
            entry = _write(output / "rollouts" / "train" / f"{rollout_id}.npz", continuation, perturbed, audit, rollout_id, kind)
            additions.append(entry); serial += 1; accepted += 1; transitions += int(entry["length"])
            accepted_rows.append({"time": time, "kind": kind})
        rollout_audit.append({"source_rollout_id": f"dev_policy_world_v9_{case:03d}", "terminal": terminal,
                              "valid_preterminal_states": len(snapshots), "late_times": late.tolist(),
                              "goal_candidate_count": len(goal_pool), "selected": len(accepted_rows),
                              "accepted": accepted_rows})
    changed = json.loads((output / "manifest.json").read_text())
    changed["files"].extend(additions)
    changed["counts"]["source"]["uniform_recovery"] += accepted
    changed["counts"]["split"]["train"] += accepted
    report = {"policy_checkpoint": str(checkpoint), "source": "development nominal v9 policy timeout rollouts only",
              "destination_split": "train", "source_split": "dev", "late_timeout_anchors_per_rollout": late_anchors,
              "goal_settling_or_overshoot_anchors_per_rollout": goal_anchors,
              "perturbation_distribution": {"position_std": scenario.perturb_position_std, "velocity_std": scenario.perturb_velocity_std},
              "attempted": attempted, "accepted": accepted, "accepted_transitions": transitions,
              "invalid_perturbation": invalid, "expert_failed_skipped": failed, "source_rollouts": rollout_audit,
              "test_opened": False}
    changed["extra_report"]["world_timeout_goal_recovery_v10"] = report
    (output / "manifest.json").write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    final = {**report, "base_source": str(source), "output": str(output),
             "output_manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}
    (output / "world_timeout_goal_recovery_report.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("output"); parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=10123)
    args = parser.parse_args()
    print(json.dumps(collect(args.source, args.output, args.checkpoint, seed=args.seed), indent=2))
