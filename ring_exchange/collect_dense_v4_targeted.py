"""Append auditable dense dev-only obstacle recovery to frozen Ring Base-U v4.

This is acquisition, not an evaluation: it rolls the supplied Stage-I policy
only from development nominal states to locate wall/obstacle collision
precursors.  Test trajectories are never opened.  Successful expert
continuations alone are appended to a copied Base-U dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import numpy as np

from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import RingExchangeEnv
from .protocol import RingExchangeScenario


def _load_initial(dataset: Path, row: dict) -> dict:
    with np.load(dataset / row["file"], allow_pickle=False) as values:
        return json.loads(str(values["initial_state_json"].item()))


def _write(path: Path, continuation, *, initial_state: dict, rollout_id: str, audit: RecoveryAudit,
           destination_split: str) -> dict:
    states = np.asarray(continuation.states, dtype=np.float64)
    observations = np.asarray(continuation.observations, dtype=np.float32)
    actions = np.asarray(continuation.actions, dtype=np.float32)
    digest = _digest(states, observations, actions)
    metadata = dict(continuation.metadata)
    metadata.update(success=True, terminal_reason=continuation.terminal_reason, rollout_id=rollout_id,
                    split=destination_split, source="targeted_wall_obstacle", trajectory_digest=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.asarray(TRAJECTORY_SCHEMA), states=states, observations=observations,
                        actions=actions, initial_state_json=np.asarray(json.dumps(_jsonable(initial_state), sort_keys=True)),
                        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
                        recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__), sort_keys=True)))
    return {"file": str(path.relative_to(path.parents[2])), "rollout_id": rollout_id, "split": destination_split,
            "source": "targeted_wall_obstacle", "trajectory_digest": digest, "length": int(len(actions))}


def collect(source: str | Path, output: str | Path, checkpoint: str | Path, *, seed: int = 31415,
            jitter_count: int = 3, position_std: float = .025, velocity_std: float = .020,
            destination_split: str = "train", precursor_stride: int = 3, max_precursor_offset: int = 30,
            include_agent: bool = False) -> dict:
    source, output, checkpoint = Path(source), Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError(f"target must not exist: {output}")
    if destination_split not in {"train", "dev"}:
        raise ValueError("targeted recovery destination must be train or dev")
    if precursor_stride <= 0 or max_precursor_offset < 3:
        raise ValueError("invalid precursor window")
    manifest = json.loads((source / "manifest.json").read_text())
    # A later maturation round may use an existing Base-U+W copy as its base;
    # every new row still receives independent development-rollout provenance.
    # Copy before adding preserves Base-U v4 byte-for-byte under Base-U+W v4.
    shutil.copytree(source, output)
    config_fingerprint = str(manifest["environment_fingerprint"])
    agent, _ = load_checkpoint(checkpoint, expected_environment_fingerprint=config_fingerprint)
    scenario = RingExchangeScenario(perturb_position_std=position_std, perturb_velocity_std=velocity_std)
    nominal_dev = [row for row in manifest["files"] if row["split"] == "dev" and row["source"] == "nominal"]
    additions, source_rows, attempted, accepted, expert_failed, invalid = [], [], 0, 0, 0, 0
    serial = 0
    for index, row in enumerate(nominal_dev):
        initial = _load_initial(source, row)
        env = RingExchangeEnv()
        env.reset(initial["positions"], velocities=initial["velocities"], goals=initial["goals"])
        snapshots = [env.augmented_state()]
        collision_info = None
        for step in range(env.config.max_steps):
            key = jax.random.fold_in(jax.random.PRNGKey(seed + index), step)
            action = np.asarray(sample_bounded_actions(agent, env.observation()[None], key)[0], dtype=np.float64)
            # JAX returns float32: an exact 0.52 command can round to
            # 0.52000004 after host conversion, while the physical plant
            # correctly enforces a 1e-9 strict bound.  This is only a numeric
            # representation guard, not a safety/controller intervention.
            norm = np.linalg.norm(action, axis=-1, keepdims=True)
            action *= np.minimum(1.0, (env.config.max_speed - 1e-8) / np.maximum(norm, 1e-12))
            _, _, done, info = env.step(action)
            if done:
                collision_info = info
                break
            snapshots.append(env.augmented_state())
        if not collision_info:
            continue
        if collision_info.get("obstacle_collision"):
            collision_type, collision_identity, source_name = "obstacle", "central_obstacle", "targeted_wall_obstacle"
        elif collision_info.get("outer_collision"):
            collision_type, collision_identity, source_name = "wall", "outer_boundary", "targeted_wall_obstacle"
        elif include_agent and collision_info.get("agent_collision"):
            # Explicitly separate this optional second-pass source from wall/
            # obstacle augmentation in the manifest and every audit record.
            collision_type, collision_identity, source_name = "agent", "agent_pair", "targeted_agent"
        else:
            continue
        collision_step = int(collision_info["step"])
        used = 0
        for offset in range(3, max_precursor_offset + 1, precursor_stride):
            time = collision_step - offset
            if time < 0 or time >= len(snapshots):
                continue
            snapshot = snapshots[time]
            wall, _ = env.distances(np.asarray(snapshot["positions"], dtype=np.float64))
            # Valid source state, before perturbation and before collision.
            source_state = {"positions": np.asarray(snapshot["positions"], dtype=np.float64),
                            "velocities": np.asarray(snapshot["last_applied_velocity"], dtype=np.float64),
                            "goals": np.asarray(snapshot["goals"], dtype=np.float64), "split": "dev",
                            "recovery": True, "source_time": time}
            for copy in range(jitter_count):
                attempted += 1
                perturbation_seed = int(np.random.SeedSequence([seed, index, time, copy]).generate_state(1)[0])
                perturbed = scenario.perturb_state(source_state, np.random.default_rng(perturbation_seed))
                if not scenario.valid_state(perturbed):
                    invalid += 1
                    continue
                continuation = scenario.expert(perturbed, np.random.default_rng(perturbation_seed))
                if not continuation.success:
                    expert_failed += 1
                    continue
                audit = RecoveryAudit(source_rollout_id=f"dev_policy_v4_{index:03d}", source_split="dev",
                                      source_time=time, collision_type=collision_type,
                                      collision_identity=collision_identity,
                                      distance_to_collision=float(wall.min()), perturbation_seed=perturbation_seed,
                                      expert_success=True)
                roll_id = f"{destination_split}_{source_name}_v4_{serial:06d}"
                additions.append(_write(output / "rollouts" / destination_split / f"{roll_id}.npz", continuation,
                                        initial_state=perturbed, rollout_id=roll_id, audit=audit,
                                        destination_split=destination_split))
                # _write uses the standard targeted-wall source.  Rewrite the
                # small NPZ metadata for separately authorized agent data.
                if source_name == "targeted_agent":
                    created = output / "rollouts" / destination_split / f"{roll_id}.npz"
                    with np.load(created, allow_pickle=False) as values:
                        payload = {name: values[name] for name in values.files}
                    metadata = json.loads(str(payload["metadata_json"].item())); metadata["source"] = source_name
                    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
                    np.savez_compressed(created, **payload)
                    additions[-1]["source"] = source_name
                serial += 1; accepted += 1; used += 1
        source_rows.append({"source_rollout_id": f"dev_policy_v4_{index:03d}", "nominal_rollout_id": row["rollout_id"],
                            "collision_step": collision_step, "collision_type": collision_type,
                            "collision_identity": collision_identity, "accepted": used})
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"].extend(additions)
    manifest["counts"]["source"]["targeted_wall_obstacle"] += sum(r["source"] == "targeted_wall_obstacle" for r in additions)
    manifest["counts"]["source"]["targeted_agent"] += sum(r["source"] == "targeted_agent" for r in additions)
    manifest["counts"]["split"][destination_split] += accepted
    manifest["extra_report"]["dense_v4_targeted"] = {
        "policy_checkpoint": str(checkpoint), "source": "development_nominal_policy_rollouts_only",
        "precursor_offsets_steps": list(range(3, max_precursor_offset + 1, precursor_stride)), "spacing_steps": precursor_stride,
        "jitter_count_per_precursor": jitter_count, "position_std": position_std, "velocity_std": velocity_std,
        "destination_split": destination_split, "attempted": attempted, "accepted": accepted, "invalid_perturbation": invalid,
        "expert_failed_skipped": expert_failed, "source_rollouts": source_rows, "include_agent": include_agent, "test_opened": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = dict(manifest["extra_report"]["dense_v4_targeted"], output=str(output), base_source=str(source),
                  output_manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest())
    (output / "dense_v4_targeted_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("output"); parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=31415); parser.add_argument("--jitter-count", type=int, default=3)
    parser.add_argument("--destination-split", choices=("train", "dev"), default="train")
    parser.add_argument("--precursor-stride", type=int, default=3); parser.add_argument("--max-precursor-offset", type=int, default=30)
    parser.add_argument("--include-agent", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.source, args.output, args.checkpoint, seed=args.seed, jitter_count=args.jitter_count,
                             destination_split=args.destination_split, precursor_stride=args.precursor_stride,
                             max_precursor_offset=args.max_precursor_offset, include_agent=args.include_agent), indent=2))
