"""Development-policy recovery acquisition for the world-frame Four-Way baseline.

The selection predicates are entirely physical: uniform valid rollout states,
plus valid states close to a goal or already beyond it along that agent's fixed
approach axis.  They are provenance/audit fields, never policy inputs or
coordination labels.  Test trajectory files are deliberately never opened.
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
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions

from .environment import Config, FourWayIntersectionEnv
from .lane_local import world_to_local
from .protocol import FourWayScenario


def _initial(root: Path, row: dict) -> dict:
    with np.load(root / row["file"], allow_pickle=False) as data:
        return json.loads(str(data["initial_state_json"].item()))


def _bounded(agent, observation: np.ndarray, key, limit: float) -> np.ndarray:
    action = np.asarray(sample_bounded_actions(agent, observation[None], key)[0], dtype=np.float64)
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    return action * np.minimum(1.0, (limit - 1e-8) / np.maximum(norm, 1e-12))


def _write(path: Path, continuation, initial: dict, audit: RecoveryAudit, rollout_id: str, kind: str) -> dict:
    states = np.asarray(continuation.states, dtype=np.float64)
    observations = np.asarray(continuation.observations, dtype=np.float32)
    actions = np.asarray(continuation.actions, dtype=np.float32)
    digest = _digest(states, observations, actions)
    metadata = dict(continuation.metadata)
    metadata.update(
        success=True,
        terminal_reason=continuation.terminal_reason,
        rollout_id=rollout_id,
        split="train",
        source="uniform_recovery",
        trajectory_digest=digest,
        recovery_anchor_kind=kind,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(TRAJECTORY_SCHEMA),
        states=states,
        observations=observations,
        actions=actions,
        initial_state_json=np.asarray(json.dumps(_jsonable(initial), sort_keys=True)),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
        recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__), sort_keys=True)),
    )
    return {
        "file": str(path.relative_to(path.parents[2])),
        "rollout_id": rollout_id,
        "split": "train",
        "source": "uniform_recovery",
        "trajectory_digest": digest,
        "length": len(actions),
    }


def collect(source, output, checkpoint, *, seed: int = 9123, uniform_anchors: int = 30, goal_anchors: int = 12) -> dict:
    source, output, checkpoint = Path(source), Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    cfg = Config(**{key: value for key, value in manifest["scenario_config"].items() if key in Config.__dataclass_fields__})
    agent, _ = load_checkpoint(checkpoint, expected_environment_fingerprint=cfg.fingerprint)
    scenario = FourWayScenario(cfg)
    # Copying preserves the frozen split's provenance without examining its files.
    shutil.copytree(source, output)
    development = [row for row in manifest["files"] if row["split"] == "dev" and row["source"] == "nominal"]
    added, provenance = [], []
    serial = attempted = accepted = invalid = failed = transitions = 0
    for case, row in enumerate(development):
        initial = _initial(source, row)
        env = FourWayIntersectionEnv(cfg)
        env.reset(np.asarray(initial["positions"]), np.asarray(initial["velocities"]))
        snapshots = [env.augmented_state()]
        terminal = "timeout"
        key = jax.random.PRNGKey(seed + case)
        for step in range(cfg.max_steps):
            action = _bounded(agent, env.observation(), jax.random.fold_in(key, step), cfg.max_speed)
            _, _, done, info = env.step(action)
            terminal = str(info["termination"])
            if done:
                break
            snapshots.append(env.augmented_state())
        uniform = np.unique(np.linspace(0, len(snapshots) - 1, min(uniform_anchors, len(snapshots)), dtype=int))
        goal_candidates = []
        for time, snapshot in enumerate(snapshots):
            positions = np.asarray(snapshot["positions"])
            remaining = env.goals - positions
            near_goal = np.linalg.norm(remaining, axis=1) <= 0.55
            beyond_goal = world_to_local(remaining)[:, 0] < -cfg.goal_tolerance
            if bool(np.any(near_goal | beyond_goal)):
                goal_candidates.append(time)
        dense_indices = (
            np.unique(np.linspace(0, len(goal_candidates) - 1, min(goal_anchors, len(goal_candidates)), dtype=int))
            if goal_candidates else np.empty(0, dtype=int)
        )
        candidates = [(int(time), "uniform_full_rollout") for time in uniform]
        candidates += [(int(goal_candidates[index]), "goal_near_or_overshoot") for index in dense_indices]
        # Prefer explicit physical goal provenance when the two selection sets overlap.
        selected = {time: kind for time, kind in candidates}
        local_accepted = []
        for time, kind in sorted(selected.items()):
            snapshot = snapshots[time]
            state = {
                "positions": np.asarray(snapshot["positions"], dtype=np.float64),
                "velocities": np.asarray(snapshot["last_applied_velocity"], dtype=np.float64),
                "split": "dev",
                "recovery": True,
                "source_time": time,
            }
            attempted += 1
            perturbation_seed = int(np.random.SeedSequence([seed, case, time, 0 if kind == "uniform_full_rollout" else 1]).generate_state(1)[0])
            perturbed = scenario.perturb_state(state, np.random.default_rng(perturbation_seed))
            if not scenario.valid_state(perturbed):
                invalid += 1
                continue
            continuation = scenario.expert(perturbed, np.random.default_rng(perturbation_seed))
            if not continuation.success:
                failed += 1
                continue
            audit = RecoveryAudit(
                source_rollout_id=f"dev_policy_world_v6_{case:03d}",
                source_split="dev",
                source_time=time,
                collision_type=None,
                collision_identity=None,
                distance_to_collision=None,
                perturbation_seed=perturbation_seed,
                expert_success=True,
            )
            rollout_id = f"train_world_v9_recovery_{serial:06d}"
            entry = _write(output / "rollouts" / "train" / f"{rollout_id}.npz", continuation, perturbed, audit, rollout_id, kind)
            added.append(entry)
            serial += 1
            accepted += 1
            transitions += int(entry["length"])
            local_accepted.append({"time": time, "kind": kind, "accepted": True})
        provenance.append({
            "source_rollout_id": f"dev_policy_world_v6_{case:03d}",
            "nominal_rollout_id": row["rollout_id"],
            "terminal": terminal,
            "valid_preterminal_states": len(snapshots),
            "uniform_times": uniform.tolist(),
            "goal_candidate_count": len(goal_candidates),
            "selected_goal_times": [int(goal_candidates[index]) for index in dense_indices],
            "accepted": local_accepted,
        })
    changed = json.loads((output / "manifest.json").read_text())
    changed["files"].extend(added)
    changed["counts"]["source"]["uniform_recovery"] += accepted
    changed["counts"]["split"]["train"] += accepted
    report = {
        "policy_checkpoint": str(checkpoint),
        "source": "development nominal world-frame policy trajectories only",
        "destination_split": "train",
        "source_split": "dev",
        "uniform_anchors_per_rollout": uniform_anchors,
        "goal_near_or_overshoot_anchors_per_rollout": goal_anchors,
        "perturbation_distribution": {"position_std": scenario.perturb_position_std, "velocity_std": scenario.perturb_velocity_std},
        "attempted": attempted,
        "accepted": accepted,
        "accepted_transitions": transitions,
        "invalid_perturbation": invalid,
        "expert_failed_skipped": failed,
        "source_rollouts": provenance,
        "test_opened": False,
    }
    changed["extra_report"]["world_goal_recovery_v9"] = report
    (output / "manifest.json").write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    final = {**report, "base_source": str(source), "output": str(output), "output_manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}
    (output / "world_goal_recovery_report.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=9123)
    args = parser.parse_args()
    print(json.dumps(collect(args.source, args.output, args.checkpoint, seed=args.seed), indent=2))
