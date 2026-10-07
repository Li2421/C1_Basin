#!/usr/bin/env python3
"""Train the preregistered initial-state coverage scaling arms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np

from double_bottleneck.flowbc_4a_agent import (
    DoubleBottleneckFlowBCAgent,
    get_config,
    parameter_count,
    save_checkpoint,
)
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


VARIANTS = ("S-Small", "S-Medium", "S-Large")


@dataclass
class ArrayDataset:
    observations: np.ndarray
    actions: np.ndarray
    seed: int

    def __post_init__(self):
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.observations.shape[0] != self.actions.shape[0]:
            raise ValueError("observation/action count mismatch")
        if self.observations.shape[1:] != (4, 18) or self.actions.shape[1:] != (4, 2):
            raise ValueError("wrong joint state/action shape")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError("non-finite data")
        self.rng = np.random.default_rng(self.seed)

    def sample(self, count: int):
        index = self.rng.integers(0, len(self.observations), size=count)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def __len__(self):
        return len(self.observations)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_recovery(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["observations"].astype(np.float32),
            archive["actions"].astype(np.float32),
            archive["source_family_id"].astype(str),
        )


def _episode_arrays(episodes):
    return (
        np.concatenate([episode.observations[:-1] for episode in episodes]).astype(np.float32),
        np.concatenate([episode.actions for episode in episodes]).astype(np.float32),
    )


def _assemble_train(root: Path, variant: str):
    data_root = root / "diagnostics/double_bottleneck_initial_state_coverage/data"
    selection_path = data_root / "variant_manifests" / f"{variant.lower().replace('-', '_')}.json"
    selection = json.loads(selection_path.read_text())
    selected = set(selection["family_ids"])
    nominal = FlowBC4ADataset(data_root / "train_pool", "train", seed=0)
    episodes = [episode for episode in nominal.episodes if episode.family_id in selected]
    if len(episodes) != selection["expert_trajectories"]:
        raise ValueError("selected nominal trajectory count mismatch")
    nominal_obs, nominal_act = _episode_arrays(episodes)
    recovery_obs, recovery_act, recovery_family = _load_recovery(
        data_root / "recovery_train.npz"
    )
    mask = np.isin(recovery_family, list(selected))
    if int(np.sum(mask)) != selection["recovery_transitions"]:
        raise ValueError("selected recovery count mismatch")
    train = ArrayDataset(
        np.concatenate((nominal_obs, recovery_obs[mask]), axis=0),
        np.concatenate((nominal_act, recovery_act[mask]), axis=0),
        seed=0,
    )
    return train, selection, selection_path


def _assemble_validation(root: Path):
    data_root = root / "diagnostics/double_bottleneck_initial_state_coverage/data"
    nominal = FlowBC4ADataset(data_root / "validation_pool", "val", seed=1)
    nominal_obs, nominal_act = nominal.all_transitions()
    recovery_obs, recovery_act, _ = _load_recovery(data_root / "recovery_validation.npz")
    return ArrayDataset(
        np.concatenate((nominal_obs, recovery_obs), axis=0),
        np.concatenate((nominal_act, recovery_act), axis=0),
        seed=1,
    )


def _normalization(dataset: ArrayDataset):
    result = {}
    for name, values in (("obs", dataset.observations), ("act", dataset.actions)):
        flat = values.reshape((len(values), -1)).astype(np.float64)
        result[f"{name}_mean"] = tuple(flat.mean(axis=0))
        result[f"{name}_scale"] = tuple(np.maximum(flat.std(axis=0), 0.01))
    return result


def _fixed_batches(dataset, count=16, size=256):
    return tuple(dataset.sample(size) for _ in range(count))


def _loss(agent, batches, seed):
    root = jax.random.PRNGKey(seed)
    values = []
    for index, batch in enumerate(batches):
        value, _ = agent.total_loss(
            batch, agent.network.params, rng=jax.random.fold_in(root, index)
        )
        values.append(float(value))
    return float(np.mean(values))


def _updates(rows: int) -> int:
    return int(math.ceil((64.0 * rows / 256.0) / 1000.0) * 1000)


def _train_one(root: Path, output: Path, variant: str, validation: ArrayDataset):
    train, selection, selection_path = _assemble_train(root, variant)
    steps = _updates(len(train))
    config = get_config()
    config.update(_normalization(train))
    config["normalize"] = True
    nominal = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_initial_state_coverage/data/train_pool",
        "train",
        seed=0,
    )
    config["environment_fingerprint"] = nominal.environment_fingerprint
    config["max_speed"] = float(nominal.config["max_speed"])
    example = train.sample(256)
    agent = DoubleBottleneckFlowBCAgent.create(
        0, jnp.asarray(example["observations"]), jnp.asarray(example["actions"]), config
    )
    output.mkdir(parents=True, exist_ok=False)
    fixed_train = _fixed_batches(train)
    fixed_val = _fixed_batches(validation)
    prereg = root / "diagnostics/double_bottleneck_initial_state_coverage/PREREGISTRATION.json"
    data_manifest = root / "diagnostics/double_bottleneck_initial_state_coverage/data/manifest.json"
    metadata = {
        "schema": "double_bottleneck_initial_state_coverage_training_v1",
        "variant": variant,
        "independent_initial_states": selection["initial_states"],
        "expert_trajectories": selection["expert_trajectories"],
        "nominal_transitions": selection["nominal_transitions"],
        "recovery_transitions": selection["recovery_transitions"],
        "training_transitions": len(train),
        "validation_transitions": len(validation),
        "steps": steps,
        "seed": 0,
        "batch_size": 256,
        "expected_sample_exposures": steps * 256 / len(train),
        "exposure_rule": "ceil((64 * N / 256) / 1000) * 1000",
        "sampling": "uniform over concatenated nominal+recovery transitions with replacement",
        "architecture": "unchanged official Toy-sourced ActorVectorField 72D->8D, 256x3",
        "parameter_count": parameter_count(agent),
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "flow_steps": 10,
        "selection": "fixed final update; no validation/test checkpoint selection",
        "preregistration_sha256": _sha(prereg),
        "data_manifest_sha256": _sha(data_manifest),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": _sha(selection_path),
        "canonical_agent_sha256": _sha(root / "double_bottleneck/flowbc_4a_agent.py"),
        "prohibited_features": {
            "explicit_mode_input": False,
            "failure_triggered_data": False,
            "region_specific_sampling": False,
            "persistent_latent": False,
            "trajectory_or_horizon_model": False,
            "recurrent_state": False,
            "g_phi": False,
            "eta": False,
        },
        "model": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dict(agent.config).items()
        },
    }
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    initial_train = _loss(agent, fixed_train, 1000)
    initial_val = _loss(agent, fixed_val, 2000)
    started = time.perf_counter()
    metrics_path = output / "metrics.jsonl"
    final_batch_loss = float("nan")
    log_interval = 5000
    for step in range(1, steps + 1):
        agent, info = agent.update(train.sample(256), step)
        final_batch_loss = float(info["bc_flow_loss"])
        if not np.isfinite(final_batch_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step % log_interval == 0 or step == steps:
            row = {
                "step": step,
                "batch_loss": final_batch_loss,
                "fixed_train_loss": _loss(agent, fixed_train, 1000),
                "fixed_val_loss": _loss(agent, fixed_val, 2000),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"variant": variant, **row}, sort_keys=True), flush=True)
    checkpoint = output / "ckpt_final.pkl"
    save_checkpoint(checkpoint, agent, metadata)
    summary = {
        **{key: value for key, value in metadata.items() if key != "model"},
        "initial_fixed_train_loss": initial_train,
        "initial_fixed_val_loss": initial_val,
        "final_batch_loss": final_batch_loss,
        "final_fixed_train_loss": _loss(agent, fixed_train, 1000),
        "final_fixed_val_loss": _loss(agent, fixed_val, 2000),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    del train, agent
    gc.collect()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_initial_state_coverage/models"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    validation = _assemble_validation(root)
    summaries = []
    for variant in VARIANTS:
        summaries.append(
            _train_one(
                root, output / variant.lower().replace("-", "_"), variant, validation
            )
        )
    (output / "training_comparison.json").write_text(
        json.dumps({"variants": summaries}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
