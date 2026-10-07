#!/usr/bin/env python3
"""Train fixed-setting Stage-I MACFlow on Recovery-V2 coverage variants."""

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


VARIANTS = ("D2-T", "D2-G", "D2-V", "D2")


@dataclass
class ArrayDataset:
    observations: np.ndarray
    actions: np.ndarray
    seed: int

    def __post_init__(self):
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.observations.shape[0] != self.actions.shape[0]:
            raise ValueError("observation/action size mismatch")
        if self.observations.shape[1:] != (4, 18) or self.actions.shape[1:] != (4, 2):
            raise ValueError("wrong joint tensor contract")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError("non-finite dataset")
        self.rng = np.random.default_rng(self.seed)

    def sample(self, count: int) -> dict[str, np.ndarray]:
        index = self.rng.integers(0, len(self.observations), size=count)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def __len__(self) -> int:
        return len(self.observations)


def _load_npz(path: Path, category: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions"]
        if category is not None:
            mask = data["category"] == category
            observations = observations[mask]
            actions = actions[mask]
    return observations.astype(np.float32), actions.astype(np.float32)


def _component_arrays(root: Path, split: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    nominal = FlowBC4ADataset(root / "diagnostics/double_bottleneck_expert_dataset_8mode", split)
    v1 = _load_npz(
        root / f"diagnostics/double_bottleneck_toy_transfer_audit/recovery_{split}.npz"
    )
    targeted_path = root / f"diagnostics/double_bottleneck_recovery_v2/data/targeted_{split}.npz"
    result = {
        "nominal": nominal.all_transitions(),
        "v1": v1,
        "transition": _load_npz(targeted_path, "transition"),
        "goal": _load_npz(targeted_path, "goal"),
    }
    if split == "train":
        velocity = FlowBC4ADataset(
            root / "diagnostics/double_bottleneck_recovery_v2/data/velocity_trajectories",
            "train",
        )
        result["velocity"] = velocity.all_transitions()
    return result


def _assemble(
    components: dict[str, tuple[np.ndarray, np.ndarray]], names: tuple[str, ...], seed: int
) -> ArrayDataset:
    return ArrayDataset(
        np.concatenate([components[name][0] for name in names], axis=0),
        np.concatenate([components[name][1] for name in names], axis=0),
        seed,
    )


def _normalization(dataset: ArrayDataset) -> dict[str, tuple[float, ...]]:
    result = {}
    for name, value in (("obs", dataset.observations), ("act", dataset.actions)):
        flat = value.reshape((len(value), -1)).astype(np.float64)
        result[f"{name}_mean"] = tuple(flat.mean(axis=0))
        result[f"{name}_scale"] = tuple(np.maximum(flat.std(axis=0), 0.01))
    return result


def _fixed_batches(dataset: ArrayDataset, count: int = 16, batch: int = 256):
    return tuple(dataset.sample(batch) for _ in range(count))


def _loss(agent, batches, seed: int) -> float:
    values = []
    root = jax.random.PRNGKey(seed)
    for index, batch in enumerate(batches):
        value, _ = agent.total_loss(
            batch, agent.network.params, rng=jax.random.fold_in(root, index)
        )
        values.append(float(value))
    return float(np.mean(values))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_one(
    root: Path,
    output: Path,
    variant: str,
    train_components,
    val_components,
    steps: int,
) -> dict:
    component_names = {
        "D2-T": ("nominal", "v1", "transition"),
        "D2-G": ("nominal", "v1", "goal"),
        "D2-V": ("nominal", "v1", "velocity"),
        "D2": ("nominal", "v1", "transition", "goal", "velocity"),
    }[variant]
    val_names = tuple(name for name in component_names if name in val_components)
    train = _assemble(train_components, component_names, seed=0)
    validation = _assemble(val_components, val_names, seed=1)
    config = get_config()
    config.update(_normalization(train))
    config["normalize"] = True
    nominal_dataset = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "train"
    )
    config["environment_fingerprint"] = nominal_dataset.environment_fingerprint
    config["max_speed"] = float(nominal_dataset.config["max_speed"])
    example = train.sample(256)
    agent = DoubleBottleneckFlowBCAgent.create(
        0, jnp.asarray(example["observations"]), jnp.asarray(example["actions"]), config
    )
    output.mkdir(parents=True, exist_ok=False)
    fixed_train = _fixed_batches(train)
    fixed_val = _fixed_batches(validation)
    per_component_batches = {
        name: _fixed_batches(ArrayDataset(*arrays, seed=100 + index), count=8)
        for index, (name, arrays) in enumerate(val_components.items())
    }
    metadata = {
        "schema": "double_bottleneck_recovery_v2_training_v1",
        "variant": variant,
        "components": list(component_names),
        "validation_components": list(val_names),
        "training_transitions": len(train),
        "validation_transitions": len(validation),
        "component_counts": {
            name: len(train_components[name][0]) for name in component_names
        },
        "steps": steps,
        "seed": 0,
        "batch_size": 256,
        "sampling": "uniform over concatenated transitions with replacement",
        "architecture": "unchanged official Toy-sourced ActorVectorField 72D->8D, 256x3",
        "parameter_count": parameter_count(agent),
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "flow_steps": 10,
        "selection": "fixed final update; no rollout-informed checkpoint selection",
        "data_manifest": str(
            (root / "diagnostics/double_bottleneck_recovery_v2/data/manifest.json").resolve()
        ),
        "data_manifest_sha256": _sha(
            root / "diagnostics/double_bottleneck_recovery_v2/data/manifest.json"
        ),
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
        "model": {key: list(value) if isinstance(value, tuple) else value for key, value in dict(agent.config).items()},
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    initial_train = _loss(agent, fixed_train, 1000)
    initial_val = _loss(agent, fixed_val, 2000)
    started = time.perf_counter()
    metrics = output / "metrics.jsonl"
    last = initial_train
    for step in range(1, steps + 1):
        agent, info = agent.update(train.sample(256), step)
        last = float(info["bc_flow_loss"])
        if not np.isfinite(last):
            raise FloatingPointError(f"non-finite loss at {step}")
        if step % 1000 == 0 or step == steps:
            row = {
                "step": step,
                "batch_loss": last,
                "fixed_train_loss": _loss(agent, fixed_train, 1000),
                "fixed_val_loss": _loss(agent, fixed_val, 2000),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"variant": variant, **row}), flush=True)
    final_train = _loss(agent, fixed_train, 1000)
    final_val = _loss(agent, fixed_val, 2000)
    component_loss = {
        name: _loss(agent, batches, 3000 + index)
        for index, (name, batches) in enumerate(per_component_batches.items())
    }
    checkpoint = output / "ckpt_final.pkl"
    save_checkpoint(checkpoint, agent, metadata)
    summary = {
        **{key: value for key, value in metadata.items() if key != "model"},
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "initial_fixed_train_loss": initial_train,
        "initial_fixed_val_loss": initial_val,
        "final_fixed_train_loss": final_train,
        "final_fixed_val_loss": final_val,
        "final_component_val_losses": component_loss,
        "training_seconds": time.perf_counter() - started,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_recovery_v2/models"),
    )
    parser.add_argument("--steps", type=int, default=25000)
    args = parser.parse_args()
    if args.steps != 25000:
        raise ValueError("Recovery-V2 causal comparison is frozen at 25,000 updates")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train_components = _component_arrays(root, "train")
    val_components = _component_arrays(root, "val")
    summaries = []
    for variant in VARIANTS:
        summaries.append(
            train_one(
                root,
                output / variant.lower().replace("-", "_"),
                variant,
                train_components,
                val_components,
                args.steps,
            )
        )
    (output / "training_comparison.json").write_text(
        json.dumps({"variants": summaries}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
