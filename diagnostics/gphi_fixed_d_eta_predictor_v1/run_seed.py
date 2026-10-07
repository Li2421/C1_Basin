"""Train one fixed-D eta predictor trajectory from scratch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
EXPECTED_SHA = "79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9"
V3_COUNT = 18816
BASE_COUNT = 26432

sys.path.insert(0, str(ROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_metrics(prediction: np.ndarray, target: np.ndarray,
                   state_id: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    value = core.subset_metrics(prediction[mask], target[mask], state_id[mask])
    return {"mean_l2": value["state_grouped_mean_l2"], "mse": value["state_grouped_mse"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(17, 23, 41))
    parser.add_argument("--epochs", type=int, default=1200)
    args = parser.parse_args()
    output = HERE / f"seed{args.seed}"
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()

    samples_path = DATASET / "samples.npz"
    if sha256(samples_path) != EXPECTED_SHA:
        raise RuntimeError("dataset hash mismatch")
    with np.load(samples_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    with np.load(HERE / "eta_targets.npz", allow_pickle=False) as source:
        targets = {key: np.asarray(source[key]).copy() for key in source.files}
    if not np.array_equal(arrays["sample_id"], targets["sample_id"]):
        raise RuntimeError("eta target alignment mismatch")
    if set(arrays["split"][BASE_COUNT:].tolist()) != {"train"}:
        raise RuntimeError("historical augmentation not training-only")

    schema = json.loads((BASE / "feature_schema.json").read_text())
    split = arrays["split"]
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    mean, scale, binary = core.fit_normalization(arrays["features"][train], schema)
    x = core.normalize(arrays["features"], mean, scale).astype(np.float32)
    y = targets["eta_normalized"].astype(np.float32)
    state_id = arrays["state_id"]
    startup_val = (validation >= V3_COUNT) & (validation < BASE_COUNT)
    warm_val = validation < V3_COUNT
    if not np.all(startup_val | warm_val):
        raise RuntimeError("training-only augmentation leaked into validation")

    import jax
    import jax.numpy as jnp
    import optax

    config = json.loads((HERE / "training_config.json").read_text())
    model_config = {
        "name": f"fixed_d_eta_seed{args.seed}",
        "seed": args.seed,
        "architecture": [214, 128, 128, 3],
        "activation": "SiLU",
        "output": "normalized_eta_linear",
        "eta_low": targets["eta_low"].tolist(),
        "eta_high": targets["eta_high"].tolist(),
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "max_epochs": args.epochs,
    }
    params = core.init_mlp(jax, [214, 128, 128, 3], args.seed)
    optimizer = optax.adamw(config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(candidate, state, xb, yb):
        def objective(value):
            prediction = core.mlp_apply(jax, value, xb)
            return jnp.mean((prediction - yb) ** 2)
        loss, gradient = jax.value_and_grad(objective)(candidate)
        updates, state = optimizer.update(gradient, state, candidate)
        return optax.apply_updates(candidate, updates), state, loss

    @jax.jit
    def predict(candidate, xb):
        return core.mlp_apply(jax, candidate, xb)

    x_train = jnp.asarray(x[train])
    y_train = jnp.asarray(y[train])
    x_val = jnp.asarray(x[validation])
    rng = np.random.default_rng(args.seed)
    rows = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        ordering = rng.permutation(len(train))
        losses = []
        for begin in range(0, len(ordering), config["batch_size"]):
            batch = ordering[begin:begin + config["batch_size"]]
            params, optimizer_state, loss = update(
                params, optimizer_state, x_train[batch], y_train[batch]
            )
            losses.append(float(loss))
        prediction = np.asarray(predict(params, x_val))
        startup = cohort_metrics(prediction, y[validation], state_id[validation], startup_val)
        warm = cohort_metrics(prediction, y[validation], state_id[validation], warm_val)
        aggregate = cohort_metrics(
            prediction, y[validation], state_id[validation],
            np.ones(len(validation), dtype=bool),
        )
        rows.append({
            "seed": args.seed,
            "epoch": epoch,
            "training_minibatch_normalized_eta_mse": float(np.mean(losses)),
            "startup_validation_state_grouped_mean_l2": startup["mean_l2"],
            "startup_validation_state_grouped_mse": startup["mse"],
            "warm_validation_state_grouped_mean_l2": warm["mean_l2"],
            "warm_validation_state_grouped_mse": warm["mse"],
            "all_validation_state_grouped_mean_l2": aggregate["mean_l2"],
            "all_validation_state_grouped_mse": aggregate["mse"],
            "elapsed_s": time.monotonic() - started,
        })
        core.save_checkpoint(
            checkpoints / f"epoch_{epoch:04d}.npz", core.tree_to_numpy(jax, params),
            model_config, mean, scale, binary,
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with (output / "epoch_metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output / "progress.json").write_text(json.dumps({
                "status": "IN_PROGRESS" if epoch < args.epochs else "COMPLETED",
                "seed": args.seed,
                "epoch": epoch,
                "epochs": args.epochs,
                "elapsed_s": time.monotonic() - started,
                "device": str(jax.devices()[0]),
                "pid": os.getpid(),
            }, indent=2) + "\n")
    np.asarray(predict(params, x_val)).copy()
    (output / "runtime.json").write_text(json.dumps({
        "seed": args.seed,
        "epochs": args.epochs,
        "wall_seconds": time.monotonic() - started,
        "device": str(jax.devices()[0]),
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "startup_validation_states": int(len(set(state_id[validation][startup_val].tolist()))),
        "warm_validation_states": int(len(set(state_id[validation][warm_val].tolist()))),
        "parameter_count": 44419,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
