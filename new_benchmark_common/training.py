"""Executable, scenario-neutral trainer for :class:`JointMACFlowAgent`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import JointTransitionDataset, fit_train_normalization
from .macflow import JointMACFlowAgent, get_config, parameter_count, save_checkpoint


def _mean_loss(agent, dataset, batch_size: int, batches: int, seed: int) -> float:
    key = jax.random.PRNGKey(seed)
    losses = []
    for index in range(batches):
        loss, _ = agent.total_loss(dataset.sample(batch_size), agent.network.params,
                                   rng=jax.random.fold_in(key, index))
        losses.append(float(loss))
    return float(np.mean(losses))


def train_stage1(dataset_root: str | Path, output: str | Path, *, seed: int = 0, steps: int = 20_000,
                 batch_size: int = 256, log_interval: int = 250, validation_batches: int = 8,
                 early_transition_fraction: float = 0.0, early_steps: int = 1,
                 early_nominal_only: bool = False, actor_hidden_dims=None,
                 source_balanced_sampling: bool = False, allow_non_four_agents: bool = False,
                 snapshot_interval: int | None = None, near_goal_fraction: float = 0.0,
                 near_goal_distance: float = 1.0):
    """Train only conventional Stage-I CFM; test is deliberately never opened."""
    if min(steps, batch_size, log_interval, validation_batches) <= 0:
        raise ValueError("training counts must be positive")
    if not 0.0 <= early_transition_fraction < 1.0 or early_steps <= 0:
        raise ValueError("invalid early-transition sampling parameters")
    if snapshot_interval is not None and snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    if not 0 <= near_goal_fraction < 1 or early_transition_fraction + near_goal_fraction >= 1:
        raise ValueError("invalid near-goal fraction")
    if near_goal_distance <= 0:
        raise ValueError("near_goal_distance must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train = JointTransitionDataset(dataset_root, "train", seed=seed)
    dev = JointTransitionDataset(dataset_root, "dev", seed=seed + 1)
    if train.environment_fingerprint != dev.environment_fingerprint:
        raise ValueError("train/dev geometry fingerprints differ")
    if (train.observation_shape[0] != train.action_shape[0] or
            (train.observation_shape[0] != 4 and not allow_non_four_agents)):
        raise ValueError("agent count mismatch or non-four training not explicitly enabled")
    config = get_config(
        agent_order=train.agent_order, num_agents=train.observation_shape[0],
        obs_dim=train.observation_shape[1], act_dim=train.action_shape[1],
        environment_fingerprint=train.environment_fingerprint,
        max_speed=float(train.manifest.get("scenario_config", {}).get("max_speed", 1.0)),
    )
    if actor_hidden_dims is not None:
        config["actor_hidden_dims"] = tuple(int(x) for x in actor_hidden_dims)
    config.update(fit_train_normalization(train))
    example = train.sample(batch_size)
    agent = JointMACFlowAgent.create(seed, jnp.asarray(example["observations"]),
                                     jnp.asarray(example["actions"]), config)
    # Long expert episodes make initial crossing/circulation decisions rare
    # under purely transition-uniform sampling.  This optional, documented
    # data-coverage intervention changes neither the CFM objective nor model:
    # it merely draws a bounded fraction from the first physical transitions
    # of independently sampled expert/recovery continuations.
    early_trajectories = tuple(
        t for t in train.trajectories if not early_nominal_only or t.source == "nominal"
    )
    if not early_trajectories:
        raise ValueError("no early transitions selected")
    early_obs = np.concatenate([t.observations[: min(early_steps, t.length)] for t in early_trajectories])
    early_act = np.concatenate([t.actions[: min(early_steps, t.length)] for t in early_trajectories])
    if near_goal_fraction:
        if train.observation_shape[1] < 6:
            raise ValueError("near-goal sampler requires relative-goal features at columns 4:6")
        goal_distance = np.linalg.norm(train.observations[:, :, 4:6], axis=2)
        goal_tolerance = float(train.manifest.get("scenario_config", {}).get("goal_tolerance", 0.08))
        near_goal_indices = np.flatnonzero(np.any(
            (goal_distance > goal_tolerance) & (goal_distance < near_goal_distance), axis=1))
        if not len(near_goal_indices):
            raise ValueError("near-goal sampler found no eligible transitions")
    source_indices = {
        source: np.flatnonzero(train.sources == source)
        for source in sorted(set(train.sources.tolist()))
    }
    early_sources = np.concatenate([
        np.repeat(t.source, min(early_steps, t.length)) for t in early_trajectories
    ])
    early_source_indices = {
        source: np.flatnonzero(early_sources == source)
        for source in sorted(set(early_sources.tolist()))
    }
    if source_balanced_sampling and not source_indices:
        raise ValueError("source-balanced sampling requires training transitions")
    sample_rng = np.random.default_rng(seed + 8181)
    def sample_source_balanced(indices_by_source, observations, actions, count):
        if count == 0:
            return observations[:0], actions[:0]
        names = tuple(indices_by_source)
        selected_names = sample_rng.choice(names, size=count)
        index = np.empty(count, dtype=np.int64)
        for source in names:
            mask = selected_names == source
            index[mask] = sample_rng.choice(indices_by_source[source], size=int(mask.sum()))
        return observations[index], actions[index]
    def sample_train():
        if early_transition_fraction == 0.0 and near_goal_fraction == 0.0:
            if not source_balanced_sampling:
                return train.sample(batch_size)
            obs, act = sample_source_balanced(source_indices, train.observations, train.actions, batch_size)
            return {"observations": obs, "actions": act}
        count = int(round(batch_size * early_transition_fraction))
        near_count = int(round(batch_size * near_goal_fraction))
        if source_balanced_sampling:
            ordinary_obs, ordinary_act = sample_source_balanced(
                source_indices, train.observations, train.actions, batch_size - count - near_count)
            early_sample_obs, early_sample_act = sample_source_balanced(
                early_source_indices, early_obs, early_act, count)
            parts_obs=[ordinary_obs,early_sample_obs]
            parts_act=[ordinary_act,early_sample_act]
        else:
            ordinary = train.sample(batch_size - count - near_count)
            index = sample_rng.integers(len(early_obs), size=count)
            parts_obs=[ordinary["observations"],early_obs[index]]
            parts_act=[ordinary["actions"],early_act[index]]
        if near_count:
            chosen=sample_rng.choice(near_goal_indices,size=near_count)
            parts_obs.append(train.observations[chosen])
            parts_act.append(train.actions[chosen])
        return {"observations": np.concatenate(parts_obs),
                "actions": np.concatenate(parts_act)}
    fixed_train = tuple(sample_train() for _ in range(validation_batches))
    fixed_dev = tuple(dev.sample(batch_size) for _ in range(validation_batches))
    def fixed_loss(batches, salt):
        return float(np.mean([float(agent.total_loss(batch, agent.network.params, rng=jax.random.fold_in(jax.random.PRNGKey(salt), i))[0])
                              for i, batch in enumerate(batches)]))
    initial_train, initial_dev = fixed_loss(fixed_train, 1001), fixed_loss(fixed_dev, 1002)
    run = {"schema": "new_benchmark_joint_stage1_train_v1", "dataset_manifest_sha256": train.manifest_sha256,
           "environment_fingerprint": train.environment_fingerprint, "seed": seed, "steps": steps,
           "batch_size": batch_size, "implementation": "official_ActorVectorField_ModuleDict_TrainState_Adam_CFM",
           "flow_steps": 10, "agent_order": list(train.agent_order), "parameter_count": parameter_count(agent),
           "source_transitions": train.source_counts(), "test_opened": False,
           "early_transition_fraction": early_transition_fraction, "early_steps": early_steps,
           "early_nominal_only": early_nominal_only,
           "source_balanced_sampling": source_balanced_sampling,
           "allow_non_four_agents": allow_non_four_agents,
           "snapshot_interval": snapshot_interval}
    run.update(near_goal_fraction=near_goal_fraction, near_goal_distance=near_goal_distance,
               near_goal_transition_count=int(len(near_goal_indices)) if near_goal_fraction else 0)
    (output / "config.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    best, best_loss, best_step = agent, initial_dev, 0
    started = time.perf_counter()
    with (output / "metrics.jsonl").open("w") as log:
        for step in range(steps + 1):
            if step:
                agent, info = agent.update(sample_train(), step)
                if not np.isfinite(np.asarray([float(value) for value in info.values()])).all():
                    raise FloatingPointError(f"non-finite update at step {step}")
            if snapshot_interval is not None and step > 0 and (step % snapshot_interval == 0 or step == steps):
                save_checkpoint(output / f"snapshot_{step:07d}.pkl", agent,
                                {"step": step, "selection_metric": "pending_closed_loop_dev"})
            if step % log_interval == 0 or step == steps:
                train_loss, dev_loss = fixed_loss(fixed_train, 1001), fixed_loss(fixed_dev, 1002)
                row = {"step": step, "fixed_train_loss": train_loss, "fixed_dev_loss": dev_loss,
                       "elapsed_seconds": time.perf_counter() - started}
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                if dev_loss < best_loss:
                    best, best_loss, best_step = agent, dev_loss, step
                    save_checkpoint(output / "best.pkl", best, {"step": step, "fixed_dev_loss": dev_loss})
        if best_step == 0:
            save_checkpoint(output / "best.pkl", best, {"step": 0, "fixed_dev_loss": best_loss})
    save_checkpoint(output / "final.pkl", agent, {"step": steps})
    result = {"initial_fixed_train_loss": initial_train, "initial_fixed_dev_loss": initial_dev,
              "best_fixed_dev_loss": best_loss, "best_step": best_step,
              "final_fixed_train_loss": fixed_loss(fixed_train, 1001),
              "final_fixed_dev_loss": fixed_loss(fixed_dev, 1002), "training_seconds": time.perf_counter() - started}
    (output / "training_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    print(json.dumps(train_stage1(args.dataset, args.output, seed=args.seed, steps=args.steps,
                                  batch_size=args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
