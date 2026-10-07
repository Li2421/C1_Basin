#!/usr/bin/env python3
"""Train S-XL under the preregistered exposure and extension rules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
            raise ValueError("non-finite training data")
        self.rng = np.random.default_rng(self.seed)

    def sample(self, count):
        index = self.rng.integers(0, len(self.observations), size=count)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def __len__(self):
        return len(self.observations)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalization(dataset):
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


def _budget(exposures, rows):
    return int(math.ceil((float(exposures) * rows / 256.0) / 1000.0) * 1000)


def _milestone(fraction, primary):
    return int(round((fraction * primary) / 1000.0) * 1000)


def _assemble(root):
    study = root / "diagnostics/double_bottleneck_sxl_baseline_maturation"
    nominal = FlowBC4ADataset(study / "data/train_pool", "train", seed=0)
    with np.load(study / "data/recovery_train.npz", allow_pickle=False) as archive:
        recovery_obs = archive["observations"].astype(np.float32)
        recovery_act = archive["actions"].astype(np.float32)
    train = ArrayDataset(
        np.concatenate((nominal.observations, recovery_obs), axis=0),
        np.concatenate((nominal.actions, recovery_act), axis=0),
        seed=0,
    )
    parent = root / "diagnostics/double_bottleneck_initial_state_coverage/data"
    validation_nominal = FlowBC4ADataset(parent / "validation_pool", "val", seed=1)
    with np.load(parent / "recovery_validation.npz", allow_pickle=False) as archive:
        validation_recovery_obs = archive["observations"].astype(np.float32)
        validation_recovery_act = archive["actions"].astype(np.float32)
    validation = ArrayDataset(
        np.concatenate(
            (validation_nominal.observations, validation_recovery_obs), axis=0
        ),
        np.concatenate((validation_nominal.actions, validation_recovery_act), axis=0),
        seed=1,
    )
    return train, validation, nominal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_sxl_baseline_maturation/model"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    study = root / "diagnostics/double_bottleneck_sxl_baseline_maturation"
    prereg = study / "PREREGISTRATION.json"
    data_manifest_path = study / "data/manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    train, validation, nominal = _assemble(root)
    if len(train) != data_manifest["variant"]["total_transitions"]:
        raise ValueError("assembled transition count differs from frozen manifest")

    primary_steps = _budget(64, len(train))
    extension_steps = _budget(96, len(train))
    milestone_steps = {
        "primary_75pct": _milestone(0.75, primary_steps),
        "primary_87_5pct": _milestone(0.875, primary_steps),
        "primary_final": primary_steps,
    }
    config = get_config()
    config.update(_normalization(train))
    config["normalize"] = True
    config["environment_fingerprint"] = nominal.environment_fingerprint
    config["max_speed"] = float(nominal.config["max_speed"])
    example = train.sample(256)
    agent = DoubleBottleneckFlowBCAgent.create(
        0, jnp.asarray(example["observations"]), jnp.asarray(example["actions"]), config
    )
    fixed_train = _fixed_batches(train)
    fixed_val = _fixed_batches(validation)
    metadata = {
        "schema": "double_bottleneck_sxl_training_v1",
        "variant": "S-XL",
        "independent_initial_states": 288,
        "expert_trajectories": 2304,
        "nominal_transitions": data_manifest["variant"]["nominal_transitions"],
        "recovery_transitions": data_manifest["variant"]["recovery_transitions"],
        "training_transitions": len(train),
        "validation_transitions": len(validation),
        "primary_steps": primary_steps,
        "extension_target_steps": extension_steps,
        "primary_expected_exposures": primary_steps * 256 / len(train),
        "extension_expected_exposures": extension_steps * 256 / len(train),
        "seed": 0,
        "batch_size": 256,
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "flow_steps": 10,
        "sampling": "uniform transitions with replacement from nominal+recovery",
        "architecture": "unchanged official Toy-sourced ActorVectorField 72D->8D, 256x3",
        "parameter_count": parameter_count(agent),
        "selection": "pre-registered final primary or final triggered extension; no rollout/test selection",
        "milestone_steps": milestone_steps,
        "undertraining_extension_trigger": {
            "validation_reduction_from_75pct_at_least": 0.05,
            "train_reduction_from_75pct_at_least": 0.03,
            "final_validation_train_relative_gap_at_most": 0.25,
        },
        "capacity_check_trigger": {
            "requires_no_extension": True,
            "fixed_validation_loss_over_s_large_ratio": 1.10,
            "validation_train_relative_gap_at_most": 0.15,
            "s_large_fixed_validation_loss": 0.05335135851055384,
        },
        "preregistration_sha256": _sha(prereg),
        "data_manifest_sha256": _sha(data_manifest_path),
        "canonical_agent_sha256": _sha(root / "double_bottleneck/flowbc_4a_agent.py"),
        "prohibited_features": {
            "explicit_mode_input": False,
            "failure_triggered_data": False,
            "region_specific_sampling": False,
            "persistent_latent": False,
            "trajectory_or_horizon_model": False,
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    initial_train = _loss(agent, fixed_train, 1000)
    initial_val = _loss(agent, fixed_val, 2000)
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    metric_rows = {}
    saved_checkpoints = {}
    final_batch_loss = float("nan")

    def evaluate(step, label=None):
        row = {
            "step": step,
            "batch_loss": final_batch_loss,
            "fixed_train_loss": _loss(agent, fixed_train, 1000),
            "fixed_val_loss": _loss(agent, fixed_val, 2000),
            "sample_exposures": step * 256 / len(train),
            "elapsed_seconds": time.perf_counter() - started,
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        metric_rows[step] = row
        if label is not None:
            checkpoint = output / f"ckpt_{label}.pkl"
            save_checkpoint(checkpoint, agent, {**metadata, "checkpoint_step": step})
            saved_checkpoints[label] = {
                "path": str(checkpoint.resolve()),
                "sha256": _sha(checkpoint),
                "step": step,
            }
        print(json.dumps({"variant": "S-XL", **row, "label": label}, sort_keys=True), flush=True)
        return row

    checkpoint_by_step = {step: label for label, step in milestone_steps.items()}
    for step in range(1, primary_steps + 1):
        agent, info = agent.update(train.sample(256), step)
        final_batch_loss = float(info["bc_flow_loss"])
        if not np.isfinite(final_batch_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step % 10000 == 0 or step in checkpoint_by_step:
            evaluate(step, checkpoint_by_step.get(step))

    at_75 = metric_rows[milestone_steps["primary_75pct"]]
    at_final = metric_rows[primary_steps]
    validation_reduction = (
        at_75["fixed_val_loss"] - at_final["fixed_val_loss"]
    ) / at_75["fixed_val_loss"]
    train_reduction = (
        at_75["fixed_train_loss"] - at_final["fixed_train_loss"]
    ) / at_75["fixed_train_loss"]
    relative_gap = abs(
        at_final["fixed_val_loss"] - at_final["fixed_train_loss"]
    ) / at_final["fixed_train_loss"]
    extension_triggered = bool(
        validation_reduction >= 0.05
        and train_reduction >= 0.03
        and relative_gap <= 0.25
    )
    extension_diagnostics = {
        "validation_reduction_from_75pct": validation_reduction,
        "train_reduction_from_75pct": train_reduction,
        "primary_final_validation_train_relative_gap": relative_gap,
        "triggered": extension_triggered,
    }
    if extension_triggered:
        for step in range(primary_steps + 1, extension_steps + 1):
            agent, info = agent.update(train.sample(256), step)
            final_batch_loss = float(info["bc_flow_loss"])
            if not np.isfinite(final_batch_loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % 10000 == 0 or step == extension_steps:
                evaluate(
                    step,
                    "extension_final" if step == extension_steps else None,
                )
    selected_step = extension_steps if extension_triggered else primary_steps
    selected_checkpoint = output / "ckpt_selected.pkl"
    selected_metadata = {
        **metadata,
        "checkpoint_step": selected_step,
        "extension_triggered": extension_triggered,
    }
    save_checkpoint(selected_checkpoint, agent, selected_metadata)
    final_train = _loss(agent, fixed_train, 1000)
    final_val = _loss(agent, fixed_val, 2000)
    final_gap = abs(final_val - final_train) / final_train
    capacity_triggered = bool(
        not extension_triggered
        and final_val > 1.10 * 0.05335135851055384
        and final_gap <= 0.15
    )
    summary = {
        **{key: value for key, value in metadata.items() if key != "model"},
        "initial_fixed_train_loss": initial_train,
        "initial_fixed_val_loss": initial_val,
        "extension_diagnostics": extension_diagnostics,
        "extension_triggered": extension_triggered,
        "selected_step": selected_step,
        "selected_sample_exposures": selected_step * 256 / len(train),
        "final_batch_loss": final_batch_loss,
        "final_fixed_train_loss": final_train,
        "final_fixed_val_loss": final_val,
        "final_validation_train_relative_gap": final_gap,
        "capacity_check_triggered": capacity_triggered,
        "capacity_check_status": (
            "required" if capacity_triggered else "skipped_by_preregistered_trigger"
        ),
        "saved_checkpoints": saved_checkpoints,
        "selected_checkpoint": str(selected_checkpoint.resolve()),
        "selected_checkpoint_sha256": _sha(selected_checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "untouched_test_loaded_or_used": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "primary_steps": primary_steps,
                "extension_triggered": extension_triggered,
                "selected_step": selected_step,
                "selected_exposures": summary["selected_sample_exposures"],
                "final_fixed_train_loss": final_train,
                "final_fixed_val_loss": final_val,
                "capacity_check_triggered": capacity_triggered,
                "selected_checkpoint_sha256": summary["selected_checkpoint_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
