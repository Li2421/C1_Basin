"""Executable, scenario-neutral trainer for :class:`JointMACFlowAgent`."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import flax.core
import jax
import jax.numpy as jnp
import numpy as np

from .dataset import JointTransitionDataset, fit_train_normalization
from .macflow import JointMACFlowAgent, get_config, load_checkpoint, parameter_count, save_checkpoint


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
                 onset_recovery_fraction: float = 0.0, onset_recovery_steps: int = 50,
                 terminal_recovery_fraction: float = 0.0, terminal_recovery_steps: int = 150,
                 gate_recovery_fraction: float = 0.0, gate_recovery_steps: int = 300,
                 late_postgate_recovery_fraction: float = 0.0,
                 late_postgate_recovery_steps: int = 300,
                 postgate_fraction: float = 0.0, postgate_min_goal_distance: float = 1.0,
                 initial_nominal_fraction: float = 0.0, initial_nominal_steps: int = 5,
                 near_goal_distance: float = 1.0, motion_loss_weight: float = 1.0,
                 endpoint_action_loss_weight: float = 0.0,
                 endpoint_motion_weight: float = 1.0,
                 permutation_augmentation: bool = False,
                 shared_agent_normalization: bool = False,
                 active_action_normalization: bool = False,
                 architecture: str = "flat_mlp", set_width: int = 128,
                 set_layers: int = 2, init_checkpoint: str | Path | None = None,
                 allow_motion_loss_change: bool = False,
                 transfer_set_checkpoint: str | Path | None = None):
    """Train only conventional Stage-I CFM; test is deliberately never opened."""
    if min(steps, batch_size, log_interval, validation_batches) <= 0:
        raise ValueError("training counts must be positive")
    if not 0.0 <= early_transition_fraction < 1.0 or early_steps <= 0:
        raise ValueError("invalid early-transition sampling parameters")
    if snapshot_interval is not None and snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    if (not 0 <= near_goal_fraction < 1 or not 0 <= onset_recovery_fraction < 1
            or not 0 <= terminal_recovery_fraction < 1
            or not 0 <= gate_recovery_fraction < 1
            or not 0 <= late_postgate_recovery_fraction < 1
            or not 0 <= postgate_fraction < 1
            or not 0 <= initial_nominal_fraction < 1
            or onset_recovery_steps <= 0 or terminal_recovery_steps <= 0
            or gate_recovery_steps <= 0
            or late_postgate_recovery_steps <= 0
            or initial_nominal_steps <= 0
            or early_transition_fraction + near_goal_fraction + onset_recovery_fraction
            + terminal_recovery_fraction + gate_recovery_fraction
            + late_postgate_recovery_fraction + postgate_fraction
            + initial_nominal_fraction >= 1):
        raise ValueError("invalid phase-sampling fractions")
    if near_goal_distance <= 0:
        raise ValueError("near_goal_distance must be positive")
    if postgate_min_goal_distance <= 0:
        raise ValueError("postgate_min_goal_distance must be positive")
    if not np.isfinite(motion_loss_weight) or motion_loss_weight < 1:
        raise ValueError("motion_loss_weight must be finite and at least one")
    if not np.isfinite(endpoint_action_loss_weight) or endpoint_action_loss_weight < 0:
        raise ValueError("endpoint_action_loss_weight must be finite and nonnegative")
    if not np.isfinite(endpoint_motion_weight) or endpoint_motion_weight < 1:
        raise ValueError("endpoint_motion_weight must be finite and at least one")
    if permutation_augmentation and not shared_agent_normalization:
        raise ValueError("online row permutation requires shared-agent normalization")
    if active_action_normalization and not shared_agent_normalization:
        raise ValueError("active-action normalization requires shared-agent normalization")
    if architecture not in ("flat_mlp", "set_attention") or set_width <= 0 or set_layers <= 0:
        raise ValueError("invalid Flow architecture configuration")
    if init_checkpoint is not None and transfer_set_checkpoint is not None:
        raise ValueError("resume and cross-N transfer are mutually exclusive")
    if allow_motion_loss_change and init_checkpoint is None:
        raise ValueError("changing the loss weight requires an initial checkpoint")
    if transfer_set_checkpoint is not None and architecture != "set_attention":
        raise ValueError("cross-N parameter transfer requires the set-attention actor")
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
        motion_loss_weight=float(motion_loss_weight),
        endpoint_action_loss_weight=float(endpoint_action_loss_weight),
        endpoint_motion_weight=float(endpoint_motion_weight),
        architecture=architecture, set_width=int(set_width), set_layers=int(set_layers),
    )
    if actor_hidden_dims is not None:
        config["actor_hidden_dims"] = tuple(int(x) for x in actor_hidden_dims)
    config.update(fit_train_normalization(train, shared_agents=shared_agent_normalization,
                                          active_action_scale=active_action_normalization))
    config["active_action_normalization"] = bool(active_action_normalization)
    example = train.sample(batch_size)
    if init_checkpoint is None:
        agent = JointMACFlowAgent.create(seed, jnp.asarray(example["observations"]),
                                         jnp.asarray(example["actions"]), config)
        init_sha = None
    else:
        init_checkpoint = Path(init_checkpoint)
        init_sha = hashlib.sha256(init_checkpoint.read_bytes()).hexdigest()
        agent, _ = load_checkpoint(init_checkpoint,
                                   expected_environment_fingerprint=train.environment_fingerprint)
        original_motion_loss_weight = float(agent.config["motion_loss_weight"])
        if (agent.config["num_agents"] != train.observation_shape[0]
                or agent.config["obs_dim"] != train.observation_shape[1]
                or agent.config["act_dim"] != train.action_shape[1]
                or agent.config["architecture"] != architecture
                or (architecture == "set_attention" and
                    (agent.config["set_width"] != set_width or
                     agent.config["set_layers"] != set_layers))
                or (not allow_motion_loss_change and
                    agent.config["motion_loss_weight"] != motion_loss_weight)):
            raise ValueError("initial checkpoint architecture/shape/loss mismatch")
        if bool(agent.config.get("active_action_normalization", False)) != active_action_normalization:
            raise ValueError("initial checkpoint action normalization mode mismatch")
        # A requested loss-only continuation keeps weights/normalization
        # exactly; neither change alters the Flow sampler or safety chain.
        agent = agent.replace(config=flax.core.FrozenDict({
            **dict(agent.config),
            "motion_loss_weight": float(motion_loss_weight),
            "endpoint_action_loss_weight": float(endpoint_action_loss_weight),
            "endpoint_motion_weight": float(endpoint_motion_weight)}))
    transfer_sha = None
    if transfer_set_checkpoint is not None:
        import pickle
        transfer_set_checkpoint = Path(transfer_set_checkpoint)
        transfer_sha = hashlib.sha256(transfer_set_checkpoint.read_bytes()).hexdigest()
        with transfer_set_checkpoint.open("rb") as handle:
            source_payload = pickle.load(handle)
        source_config = source_payload["config"]
        source_agent, _ = load_checkpoint(transfer_set_checkpoint,
            expected_environment_fingerprint=source_config["environment_fingerprint"])
        if (source_agent.config["architecture"] != "set_attention"
                or source_agent.config["obs_dim"] != train.observation_shape[1]
                or source_agent.config["act_dim"] != train.action_shape[1]
                or source_agent.config["set_width"] != set_width
                or source_agent.config["set_layers"] != set_layers):
            raise ValueError("cross-N source architecture or feature shape mismatch")
        target_leaves = jax.tree_util.tree_leaves(agent.network.params)
        source_leaves = jax.tree_util.tree_leaves(source_agent.network.params)
        if len(target_leaves) != len(source_leaves) or any(
                a.shape != b.shape for a,b in zip(target_leaves,source_leaves)):
            raise ValueError("cross-N set-actor parameter shapes differ")
        agent = agent.replace(network=agent.network.replace(params=source_agent.network.params))
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
    if initial_nominal_fraction:
        initial_trajectories = tuple(t for t in train.trajectories if t.source == "nominal")
        if not initial_trajectories:
            raise ValueError("initial nominal sampler found no nominal trajectories")
        initial_obs = np.concatenate([t.observations[: min(initial_nominal_steps, t.length)]
                                      for t in initial_trajectories])
        initial_act = np.concatenate([t.actions[: min(initial_nominal_steps, t.length)]
                                      for t in initial_trajectories])
    if onset_recovery_fraction:
        onset_trajectories = tuple(t for t in train.trajectories
                                   if t.source == "early_queue_recovery")
        if not onset_trajectories:
            raise ValueError("onset recovery sampler found no early_queue_recovery trajectories")
        onset_obs = np.concatenate([t.observations[: min(onset_recovery_steps, t.length)]
                                    for t in onset_trajectories])
        onset_act = np.concatenate([t.actions[: min(onset_recovery_steps, t.length)]
                                    for t in onset_trajectories])
    if terminal_recovery_fraction:
        terminal_trajectories = tuple(t for t in train.trajectories
                                      if t.source == "terminal_multi_recovery")
        if not terminal_trajectories:
            raise ValueError("terminal recovery sampler found no terminal_multi_recovery trajectories")
        terminal_obs = np.concatenate([t.observations[: min(terminal_recovery_steps, t.length)]
                                       for t in terminal_trajectories])
        terminal_act = np.concatenate([t.actions[: min(terminal_recovery_steps, t.length)]
                                       for t in terminal_trajectories])
    if gate_recovery_fraction:
        gate_trajectories = tuple(t for t in train.trajectories
                                  if t.source == "gate_local_recovery")
        if not gate_trajectories:
            raise ValueError("gate recovery sampler found no gate_local_recovery trajectories")
        gate_obs = np.concatenate([t.observations[: min(gate_recovery_steps, t.length)]
                                   for t in gate_trajectories])
        gate_act = np.concatenate([t.actions[: min(gate_recovery_steps, t.length)]
                                   for t in gate_trajectories])
    if late_postgate_recovery_fraction:
        late_trajectories = tuple(t for t in train.trajectories
                                  if t.source == "late_postgate_recovery")
        if not late_trajectories:
            raise ValueError("late postgate sampler found no recovery trajectories")
        late_obs = np.concatenate([t.observations[: min(late_postgate_recovery_steps, t.length)]
                                   for t in late_trajectories])
        late_act = np.concatenate([t.actions[: min(late_postgate_recovery_steps, t.length)]
                                   for t in late_trajectories])
    if postgate_fraction:
        if train.observation_shape[1] < 6:
            raise ValueError("postgate sampler requires planar position and relative-goal features")
        position_x = train.observations[:, :, 0]
        goal_x = position_x + train.observations[:, :, 4]
        goal_distance = np.linalg.norm(train.observations[:, :, 4:6], axis=2)
        moving = np.linalg.norm(train.actions, axis=2) > 0.1
        # A joint state qualifies only when an actually moving agent is on
        # the goal side of the wall but still far from its final goal.
        postgate_indices = np.flatnonzero(np.any(
            (position_x * goal_x > 0) & (np.abs(position_x) > 0.5)
            & (goal_distance > postgate_min_goal_distance) & moving, axis=1))
        if not len(postgate_indices):
            raise ValueError("postgate sampler found no eligible transitions")
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
    def maybe_permute(batch):
        if not permutation_augmentation:
            return batch
        observations, actions = batch["observations"], batch["actions"]
        order = np.argsort(sample_rng.random((len(observations), train.observation_shape[0])), axis=1)
        return {"observations":np.take_along_axis(observations, order[:, :, None], axis=1),
                "actions":np.take_along_axis(actions, order[:, :, None], axis=1)}
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
        if (early_transition_fraction == 0.0 and near_goal_fraction == 0.0
                and onset_recovery_fraction == 0.0 and terminal_recovery_fraction == 0.0
                and gate_recovery_fraction == 0.0
                and late_postgate_recovery_fraction == 0.0
                and postgate_fraction == 0.0
                and initial_nominal_fraction == 0.0):
            if not source_balanced_sampling:
                return maybe_permute(train.sample(batch_size))
            obs, act = sample_source_balanced(source_indices, train.observations, train.actions, batch_size)
            return maybe_permute({"observations": obs, "actions": act})
        count = int(round(batch_size * early_transition_fraction))
        near_count = int(round(batch_size * near_goal_fraction))
        onset_count = int(round(batch_size * onset_recovery_fraction))
        terminal_count = int(round(batch_size * terminal_recovery_fraction))
        gate_count = int(round(batch_size * gate_recovery_fraction))
        late_count = int(round(batch_size * late_postgate_recovery_fraction))
        postgate_count = int(round(batch_size * postgate_fraction))
        initial_count = int(round(batch_size * initial_nominal_fraction))
        ordinary_count = (batch_size - count - near_count - onset_count
                          - terminal_count - gate_count - late_count
                          - postgate_count - initial_count)
        if ordinary_count < 1:
            raise ValueError("rounded phase-sampling counts leave no ordinary samples")
        if source_balanced_sampling:
            ordinary_obs, ordinary_act = sample_source_balanced(
                source_indices, train.observations, train.actions, ordinary_count)
            early_sample_obs, early_sample_act = sample_source_balanced(
                early_source_indices, early_obs, early_act, count)
            parts_obs=[ordinary_obs,early_sample_obs]
            parts_act=[ordinary_act,early_sample_act]
        else:
            ordinary = train.sample(ordinary_count)
            index = sample_rng.integers(len(early_obs), size=count)
            parts_obs=[ordinary["observations"],early_obs[index]]
            parts_act=[ordinary["actions"],early_act[index]]
        if near_count:
            chosen=sample_rng.choice(near_goal_indices,size=near_count)
            parts_obs.append(train.observations[chosen])
            parts_act.append(train.actions[chosen])
        if onset_count:
            chosen=sample_rng.integers(len(onset_obs),size=onset_count)
            parts_obs.append(onset_obs[chosen])
            parts_act.append(onset_act[chosen])
        if terminal_count:
            chosen=sample_rng.integers(len(terminal_obs),size=terminal_count)
            parts_obs.append(terminal_obs[chosen])
            parts_act.append(terminal_act[chosen])
        if gate_count:
            chosen=sample_rng.integers(len(gate_obs),size=gate_count)
            parts_obs.append(gate_obs[chosen])
            parts_act.append(gate_act[chosen])
        if late_count:
            chosen=sample_rng.integers(len(late_obs),size=late_count)
            parts_obs.append(late_obs[chosen])
            parts_act.append(late_act[chosen])
        if postgate_count:
            chosen=sample_rng.choice(postgate_indices,size=postgate_count)
            parts_obs.append(train.observations[chosen])
            parts_act.append(train.actions[chosen])
        if initial_count:
            chosen=sample_rng.integers(len(initial_obs),size=initial_count)
            parts_obs.append(initial_obs[chosen])
            parts_act.append(initial_act[chosen])
        return maybe_permute({"observations": np.concatenate(parts_obs),
                              "actions": np.concatenate(parts_act)})
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
           "onset_recovery_fraction": onset_recovery_fraction,
           "onset_recovery_steps": onset_recovery_steps,
           "onset_recovery_transition_count": int(len(onset_obs)) if onset_recovery_fraction else 0,
           "terminal_recovery_fraction": terminal_recovery_fraction,
           "terminal_recovery_steps": terminal_recovery_steps,
           "terminal_recovery_transition_count": int(len(terminal_obs)) if terminal_recovery_fraction else 0,
           "gate_recovery_fraction": gate_recovery_fraction,
           "gate_recovery_steps": gate_recovery_steps,
           "gate_recovery_transition_count": int(len(gate_obs)) if gate_recovery_fraction else 0,
           "late_postgate_recovery_fraction": late_postgate_recovery_fraction,
           "late_postgate_recovery_steps": late_postgate_recovery_steps,
           "late_postgate_recovery_transition_count": (
               int(len(late_obs)) if late_postgate_recovery_fraction else 0),
           "postgate_fraction": postgate_fraction,
           "postgate_min_goal_distance": postgate_min_goal_distance,
           "postgate_transition_count": int(len(postgate_indices)) if postgate_fraction else 0,
           "initial_nominal_fraction": initial_nominal_fraction,
           "initial_nominal_steps": initial_nominal_steps,
           "initial_nominal_transition_count": int(len(initial_obs)) if initial_nominal_fraction else 0,
           "source_balanced_sampling": source_balanced_sampling,
           "allow_non_four_agents": allow_non_four_agents,
           "snapshot_interval": snapshot_interval}
    run.update(near_goal_fraction=near_goal_fraction, near_goal_distance=near_goal_distance,
               near_goal_transition_count=int(len(near_goal_indices)) if near_goal_fraction else 0,
               motion_loss_weight=motion_loss_weight,
               endpoint_action_loss_weight=endpoint_action_loss_weight,
               endpoint_motion_weight=endpoint_motion_weight,
               permutation_augmentation=permutation_augmentation,
               shared_agent_normalization=shared_agent_normalization,
               active_action_normalization=active_action_normalization,
               architecture=architecture, set_width=set_width, set_layers=set_layers)
    run.update(init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
               init_checkpoint_sha256=init_sha,
               allow_motion_loss_change=bool(allow_motion_loss_change),
               init_motion_loss_weight=(original_motion_loss_weight
                                        if init_checkpoint else None),
               normalization_from_init_checkpoint=init_checkpoint is not None,
               transfer_set_checkpoint=str(transfer_set_checkpoint) if transfer_set_checkpoint else None,
               transfer_set_checkpoint_sha256=transfer_sha,
               normalization_from_target_train=init_checkpoint is None)
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
