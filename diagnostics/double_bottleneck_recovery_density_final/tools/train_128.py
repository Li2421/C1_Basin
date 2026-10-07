#!/usr/bin/env python3
"""Train the frozen canonical MACFlow on S-XL-128 recovery data."""

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
    DoubleBottleneckFlowBCAgent, get_config, parameter_count, save_checkpoint,
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
        if self.observations.shape[1:] != (4, 18) or self.actions.shape[1:] != (4, 2):
            raise ValueError("wrong joint state/action shape")
        if len(self.observations) != len(self.actions) or not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError("invalid training array")
        self.rng = np.random.default_rng(self.seed)

    def sample(self, count):
        index = self.rng.integers(0, len(self.observations), size=count)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def __len__(self):
        return len(self.observations)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalization(dataset):
    output = {}
    for name, values in (("obs", dataset.observations), ("act", dataset.actions)):
        flat = values.reshape((len(values), -1)).astype(np.float64)
        output[f"{name}_mean"] = tuple(flat.mean(axis=0))
        output[f"{name}_scale"] = tuple(np.maximum(flat.std(axis=0), 0.01))
    return output


def fixed_batches(dataset, count=16, size=256):
    return tuple(dataset.sample(size) for _ in range(count))


def loss(agent, batches, seed):
    root = jax.random.PRNGKey(seed)
    return float(np.mean([
        float(agent.total_loss(batch, agent.network.params, rng=jax.random.fold_in(root, index))[0])
        for index, batch in enumerate(batches)
    ]))


def budget(exposures, rows):
    return int(math.ceil((exposures * rows / 256.0) / 1000.0) * 1000)


def assemble(root):
    reference = root / "diagnostics/double_bottleneck_sxl_baseline_maturation"
    study = root / "diagnostics/double_bottleneck_recovery_density_final"
    nominal = FlowBC4ADataset(reference / "data/train_pool", "train", seed=0)
    with np.load(study / "data/recovery_128_train.npz", allow_pickle=False) as archive:
        recovery_obs, recovery_act = archive["observations"].astype(np.float32), archive["actions"].astype(np.float32)
    train = ArrayDataset(np.concatenate((nominal.observations, recovery_obs)), np.concatenate((nominal.actions, recovery_act)), 0)
    parent = root / "diagnostics/double_bottleneck_initial_state_coverage/data"
    val_nominal = FlowBC4ADataset(parent / "validation_pool", "val", seed=1)
    with np.load(parent / "recovery_validation.npz", allow_pickle=False) as archive:
        val_recovery_obs, val_recovery_act = archive["observations"].astype(np.float32), archive["actions"].astype(np.float32)
    validation = ArrayDataset(np.concatenate((val_nominal.observations, val_recovery_obs)), np.concatenate((val_nominal.actions, val_recovery_act)), 1)
    return train, validation, nominal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("diagnostics/double_bottleneck_recovery_density_final/model"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    study = root / "diagnostics/double_bottleneck_recovery_density_final"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((study / "data/manifest.json").read_text())
    train, validation, nominal = assemble(root)
    if len(train) != manifest["total_transitions"]:
        raise ValueError("data manifest/training count mismatch")
    primary, extension = budget(64, len(train)), budget(96, len(train))
    milestone = {"primary_75pct": int(round(0.75 * primary / 1000) * 1000), "primary_87_5pct": int(round(0.875 * primary / 1000) * 1000), "primary_final": primary}
    config = get_config()
    config.update(normalization(train))
    config.update(normalize=True, environment_fingerprint=nominal.environment_fingerprint, max_speed=float(nominal.config["max_speed"]))
    example = train.sample(256)
    agent = DoubleBottleneckFlowBCAgent.create(0, jnp.asarray(example["observations"]), jnp.asarray(example["actions"]), config)
    train_fixed, val_fixed = fixed_batches(train), fixed_batches(validation)
    metadata = {
        "schema": "double_bottleneck_recovery_density_final_training_v1", "variant": "S-XL-128",
        "preregistration_sha256": sha(study / "PREREGISTRATION.json"), "data_manifest_sha256": sha(study / "data/manifest.json"),
        "reference_sxl_training_summary_sha256": sha(root / "diagnostics/double_bottleneck_sxl_baseline_maturation/model/training_summary.json"),
        "canonical_agent_sha256": sha(root / "double_bottleneck/flowbc_4a_agent.py"),
        "independent_initial_states": 288, "expert_trajectories": 2304,
        "nominal_transitions": manifest["nominal_transitions"], "recovery_transitions": manifest["combined_recovery_128"]["rows"],
        "training_transitions": len(train), "validation_transitions": len(validation),
        "primary_steps": primary, "extension_target_steps": extension, "primary_expected_exposures": primary * 256 / len(train), "extension_expected_exposures": extension * 256 / len(train),
        "seed": 0, "batch_size": 256, "optimizer": "Adam", "learning_rate": 3e-4, "flow_steps": 10,
        "sampling": "uniform transitions with replacement from fixed S-XL nominal+128-anchor recovery",
        "architecture": "unchanged official Toy-sourced ActorVectorField 72D->8D, 256x3", "parameter_count": parameter_count(agent),
        "milestone_steps": milestone,
        "extension_rule": "same preregistered S-XL 75%-to-final criterion", "selection": "pre-registered final primary or triggered extension; no rollout/test selection",
        "prohibited_features": {"explicit_mode_input": False, "failure_triggered_data": False, "region_specific_sampling": False, "persistent_latent": False, "trajectory_or_horizon_model": False, "action_chunks": False, "recurrent_state": False, "g_phi": False, "eta": False},
        "untouched_test_loaded_or_used": False,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    initial_train, initial_val = loss(agent, train_fixed, 1000), loss(agent, val_fixed, 2000)
    metrics_path = output / "metrics.jsonl"; rows = {}; saved = {}; started = time.perf_counter(); final_batch = float("nan")
    def measure(step, label=None):
        row = {"step": step, "batch_loss": final_batch, "fixed_train_loss": loss(agent, train_fixed, 1000), "fixed_val_loss": loss(agent, val_fixed, 2000), "sample_exposures": step * 256 / len(train), "elapsed_seconds": time.perf_counter() - started}
        with metrics_path.open("a") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
        rows[step] = row
        if label:
            path = output / f"ckpt_{label}.pkl"; save_checkpoint(path, agent, {**metadata, "checkpoint_step": step}); saved[label] = {"path": str(path.resolve()), "sha256": sha(path), "step": step}
        print(json.dumps({"variant": "S-XL-128", **row, "label": label}, sort_keys=True), flush=True)
    labels = {value: key for key, value in milestone.items()}
    for step in range(1, primary + 1):
        agent, info = agent.update(train.sample(256), step); final_batch = float(info["bc_flow_loss"])
        if not np.isfinite(final_batch): raise FloatingPointError(f"non-finite loss at {step}")
        if step % 10000 == 0 or step in labels: measure(step, labels.get(step))
    at75, atfinal = rows[milestone["primary_75pct"]], rows[primary]
    val_drop = (at75["fixed_val_loss"] - atfinal["fixed_val_loss"]) / at75["fixed_val_loss"]
    train_drop = (at75["fixed_train_loss"] - atfinal["fixed_train_loss"]) / at75["fixed_train_loss"]
    gap = abs(atfinal["fixed_val_loss"] - atfinal["fixed_train_loss"]) / atfinal["fixed_train_loss"]
    extension_triggered = bool(val_drop >= .05 and train_drop >= .03 and gap <= .25)
    if extension_triggered:
        for step in range(primary + 1, extension + 1):
            agent, info = agent.update(train.sample(256), step); final_batch = float(info["bc_flow_loss"])
            if not np.isfinite(final_batch): raise FloatingPointError(f"non-finite loss at {step}")
            if step % 10000 == 0 or step == extension: measure(step, "extension_final" if step == extension else None)
    selected_step = extension if extension_triggered else primary
    selected = output / "ckpt_selected.pkl"; save_checkpoint(selected, agent, {**metadata, "checkpoint_step": selected_step, "extension_triggered": extension_triggered})
    final_train, final_val = loss(agent, train_fixed, 1000), loss(agent, val_fixed, 2000)
    summary = {**metadata, "initial_fixed_train_loss": initial_train, "initial_fixed_val_loss": initial_val, "extension_diagnostics": {"validation_reduction_from_75pct": val_drop, "train_reduction_from_75pct": train_drop, "primary_final_validation_train_relative_gap": gap, "triggered": extension_triggered}, "extension_triggered": extension_triggered, "selected_step": selected_step, "selected_sample_exposures": selected_step * 256 / len(train), "final_batch_loss": final_batch, "final_fixed_train_loss": final_train, "final_fixed_val_loss": final_val, "final_validation_train_relative_gap": abs(final_val-final_train)/final_train, "saved_checkpoints": saved, "selected_checkpoint": str(selected.resolve()), "selected_checkpoint_sha256": sha(selected), "elapsed_seconds": time.perf_counter()-started}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"primary_steps": primary, "extension_triggered": extension_triggered, "selected_step": selected_step, "exposures": summary["selected_sample_exposures"], "train_loss": final_train, "val_loss": final_val, "checkpoint": summary["selected_checkpoint_sha256"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
