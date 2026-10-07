#!/usr/bin/env python3
"""Train the three preregistered uniform-recovery density arms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
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


VARIANTS = ("U-Low", "U-Mid", "U-High")


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
            raise ValueError("wrong 4-agent state/action contract")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError("non-finite training data")
        self.rng = np.random.default_rng(self.seed)

    def sample(self, count: int) -> dict[str, np.ndarray]:
        index = self.rng.integers(0, len(self.observations), size=count)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def __len__(self) -> int:
        return len(self.observations)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["observations"].astype(np.float32),
            archive["actions"].astype(np.float32),
        )


def _assemble(root: Path, variant: str, split: str, seed: int) -> tuple[ArrayDataset, dict]:
    nominal = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", split, seed=seed
    )
    nominal_obs, nominal_act = nominal.all_transitions()
    tag = variant.lower().replace("-", "_")
    uniform_path = (
        root / "diagnostics/double_bottleneck_uniform_recovery/data" / f"{tag}_{split}.npz"
    )
    uniform_obs, uniform_act = _load_npz(uniform_path)
    return (
        ArrayDataset(
            np.concatenate((nominal_obs, uniform_obs), axis=0),
            np.concatenate((nominal_act, uniform_act), axis=0),
            seed,
        ),
        {
            "nominal": len(nominal_obs),
            "uniform_recovery": len(uniform_obs),
            "uniform_path": str(uniform_path.resolve()),
            "uniform_sha256": _sha(uniform_path),
        },
    )


def _normalization(dataset: ArrayDataset) -> dict[str, tuple[float, ...]]:
    result = {}
    for name, values in (("obs", dataset.observations), ("act", dataset.actions)):
        flat = values.reshape((len(values), -1)).astype(np.float64)
        result[f"{name}_mean"] = tuple(flat.mean(axis=0))
        result[f"{name}_scale"] = tuple(np.maximum(flat.std(axis=0), 0.01))
    return result


def _fixed_batches(dataset: ArrayDataset, count: int = 16, size: int = 256):
    return tuple(dataset.sample(size) for _ in range(count))


def _loss(agent, batches, seed: int) -> float:
    root = jax.random.PRNGKey(seed)
    values = []
    for index, batch in enumerate(batches):
        loss, _ = agent.total_loss(
            batch, agent.network.params, rng=jax.random.fold_in(root, index)
        )
        values.append(float(loss))
    return float(np.mean(values))


def _train_one(root: Path, output: Path, variant: str, steps: int) -> dict:
    train, train_parts = _assemble(root, variant, "train", seed=0)
    validation, val_parts = _assemble(root, variant, "val", seed=1)
    nominal = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "train", seed=0
    )
    config = get_config()
    config.update(_normalization(train))
    config["normalize"] = True
    config["environment_fingerprint"] = nominal.environment_fingerprint
    config["max_speed"] = float(nominal.config["max_speed"])
    example = train.sample(256)
    agent = DoubleBottleneckFlowBCAgent.create(
        0, jnp.asarray(example["observations"]), jnp.asarray(example["actions"]), config
    )
    output.mkdir(parents=True, exist_ok=False)
    fixed_train = _fixed_batches(train)
    fixed_val = _fixed_batches(validation)
    prereg = root / "diagnostics/double_bottleneck_uniform_recovery/PREREGISTRATION.json"
    manifest = root / "diagnostics/double_bottleneck_uniform_recovery/data/manifest.json"
    metadata = {
        "schema": "double_bottleneck_uniform_recovery_training_v1",
        "variant": variant,
        "training_transitions": len(train),
        "validation_transitions": len(validation),
        "training_components": train_parts,
        "validation_components": val_parts,
        "steps": steps,
        "seed": 0,
        "batch_size": 256,
        "expected_training_passes": steps * 256 / len(train),
        "sampling": "uniform over concatenated nominal+uniform-recovery transitions with replacement",
        "architecture": "unchanged official Toy-sourced ActorVectorField 72D->8D, 256x3",
        "parameter_count": parameter_count(agent),
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "flow_steps": 10,
        "selection": "fixed final update; no rollout-informed selection",
        "preregistration_sha256": _sha(prereg),
        "data_manifest_sha256": _sha(manifest),
        "canonical_agent_sha256": _sha(root / "double_bottleneck/flowbc_4a_agent.py"),
        "prohibited_features": {
            "explicit_mode_input": False,
            "persistent_latent": False,
            "trajectory_model": False,
            "action_chunks": False,
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    initial_train = _loss(agent, fixed_train, 1000)
    initial_val = _loss(agent, fixed_val, 2000)
    started = time.perf_counter()
    metrics_path = output / "metrics.jsonl"
    final_batch_loss = float("nan")
    for step in range(1, steps + 1):
        agent, info = agent.update(train.sample(256), step)
        final_batch_loss = float(info["bc_flow_loss"])
        if not np.isfinite(final_batch_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step % 1000 == 0 or step == steps:
            row = {
                "step": step,
                "batch_loss": final_batch_loss,
                "fixed_train_loss": _loss(agent, fixed_train, 1000),
                "fixed_val_loss": _loss(agent, fixed_val, 2000),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_uniform_recovery/models"),
    )
    parser.add_argument("--steps", type=int, default=25000)
    args = parser.parse_args()
    if args.steps != 25000:
        raise ValueError("the preregistered comparison requires exactly 25,000 updates")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for variant in VARIANTS:
        summaries.append(
            _train_one(root, output / variant.lower().replace("-", "_"), variant, args.steps)
        )
    (output / "training_comparison.json").write_text(
        json.dumps({"variants": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
