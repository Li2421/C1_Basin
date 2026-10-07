"""Validation-only training grid for the 214->64->64->3 semantic mode head."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage2_recovery_entry"
sys.path[:0] = [str(BASE), str(HERE)]

from decision_learning import fit_feature_normalization, normalize_features  # noqa: E402
from mode_common import ACTION_NAMES, content_hash, sha256, verify_content_hash  # noqa: E402


SEEDS = (17, 23, 41)
LAMBDA_BREAK = (0.0, 1.0, 3.0)
MAX_EPOCHS = 1200


def load_dataset(manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text())
    verify_content_hash(manifest, manifest_path)
    if manifest.get("schema") != "semantic_mode_finalized_dataset_v1" or manifest.get("status") != "COMPLETE_FROZEN_READY_FOR_TRAINING":
        raise RuntimeError("mode dataset is not complete/frozen")
    snapshot = Path(manifest["dataset_snapshot"])
    if sha256(snapshot) != manifest["dataset_snapshot_sha256"]:
        raise RuntimeError("mode dataset snapshot hash mismatch")
    with np.load(snapshot, allow_pickle=False) as values:
        data = {name: np.asarray(values[name]) for name in values.files}
    if data["features"].ndim != 2 or data["features"].shape[1] != 214:
        raise RuntimeError("mode dataset feature mismatch")
    if data["targets"].shape != (len(data["features"]),) or data["break_vs_safety"].shape != (len(data["features"]), 3):
        raise RuntimeError("mode dataset target mismatch")
    split_values = set(data["splits"].tolist())
    if not split_values <= {"train", "validation"} or split_values != {"train", "validation"}:
        raise RuntimeError(("invalid mode dataset splits", split_values))
    for split in ("train", "validation"):
        mask = (data["splits"] == split) & (data["targets"] >= 0)
        if not np.any(mask):
            raise RuntimeError((split, "has no conservatively resolved three-way targets"))
    return manifest, data


def _lazy_jax():
    import jax
    import jax.numpy as jnp
    import optax
    return jax, jnp, optax


def init_params(seed: int):
    jax, jnp, _ = _lazy_jax()
    dimensions = (214, 64, 64, 3)
    params = []
    for key, fan_in, fan_out in zip(jax.random.split(jax.random.PRNGKey(seed), 3), dimensions[:-1], dimensions[1:]):
        limit = np.sqrt(6.0/(fan_in+fan_out))
        params.append({
            "w": jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=jnp.float32),
            "b": jnp.zeros((fan_out,), dtype=jnp.float32),
        })
    return params


def logits(params: Any, features: Any, jnp: Any) -> Any:
    value = features
    for layer in params[:-1]:
        value = value @ layer["w"] + layer["b"]
        value = value / (1.0+jnp.exp(-value))
    return value @ params[-1]["w"] + params[-1]["b"]


def train_one(data: dict[str, np.ndarray], *, seed: int, lambda_break: float,
              epochs: int = MAX_EPOCHS) -> tuple[Any, dict[str, np.ndarray], list[dict[str, float]], dict[str, Any]]:
    train_mask = (data["splits"] == "train") & (data["targets"] >= 0)
    val_mask = (data["splits"] == "validation") & (data["targets"] >= 0)
    x_train, x_val = data["features"][train_mask].astype(np.float32), data["features"][val_mask].astype(np.float32)
    y_train, y_val = data["targets"][train_mask].astype(np.int32), data["targets"][val_mask].astype(np.int32)
    b_train, b_val = data["break_vs_safety"][train_mask].astype(np.float32), data["break_vs_safety"][val_mask].astype(np.float32)
    mean, scale = fit_feature_normalization(x_train)
    x_train, x_val = normalize_features(x_train, mean, scale), normalize_features(x_val, mean, scale)
    jax, jnp, optax = _lazy_jax()
    params = init_params(seed)
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    state = optimizer.init(params)
    xt, xv = jnp.asarray(x_train), jnp.asarray(x_val)
    yt, yv = jnp.asarray(y_train), jnp.asarray(y_val)
    bt, bv = jnp.asarray(b_train), jnp.asarray(b_val)

    def loss_terms(candidate: Any, features: Any, targets: Any, breaks: Any):
        scores = logits(candidate, features, jnp)
        log_prob = jax.nn.log_softmax(scores, axis=-1)
        ce = -jnp.mean(log_prob[jnp.arange(targets.size), targets])
        probability = jax.nn.softmax(scores, axis=-1)
        break_term = jnp.mean(jnp.sum(probability*breaks, axis=-1))
        return ce + lambda_break*break_term, ce, break_term

    @jax.jit
    def step(candidate: Any, opt_state: Any):
        def objective(value: Any):
            total, ce, break_term = loss_terms(value, xt, yt, bt)
            return total, (ce, break_term)
        (total, (ce, break_term)), gradients = jax.value_and_grad(objective, has_aux=True)(candidate)
        updates, opt_state = optimizer.update(gradients, opt_state, candidate)
        return optax.apply_updates(candidate, updates), opt_state, total, ce, break_term

    best, best_epoch, best_val = None, 0, float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs+1):
        params, state, train_total, train_ce, train_break = step(params, state)
        val_total, val_ce, val_break = loss_terms(params, xv, yv, bv)
        row = {
            "epoch": float(epoch), "train_total_loss": float(train_total),
            "train_cross_entropy": float(train_ce), "train_break_penalty": float(train_break),
            "validation_total_loss": float(val_total), "validation_cross_entropy": float(val_ce),
            "validation_break_penalty": float(val_break),
        }
        history.append(row)
        if row["validation_total_loss"] < best_val:
            best_val, best_epoch = row["validation_total_loss"], epoch
            best = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
    assert best is not None
    prediction = np.argmax(np.asarray(logits(best, xv, jnp)), axis=1)
    metadata = {
        "architecture": [214, 64, 64, 3], "activation": "SiLU", "decision_rule": "argmax",
        "seed": seed, "lambda_break": lambda_break, "best_epoch": best_epoch,
        "validation_total_loss": best_val, "validation_accuracy_resolved_only": float(np.mean(prediction == y_val)),
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-5,
        "normalization_fit_split": "train", "resolved_train_count": int(len(y_train)),
        "resolved_validation_count": int(len(y_val)),
        "parameter_count": int(sum(value.size for layer in best for value in layer.values())),
        "scores_are_not_task_success_probabilities": True,
    }
    return best, {"mean": mean, "scale": scale}, history, metadata


def save_checkpoint(path: Path, params: Any, normalization: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    payload = {
        "normalization_mean": np.asarray(normalization["mean"], dtype=np.float32),
        "normalization_scale": np.asarray(normalization["scale"], dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, layer in enumerate(params):
        payload[f"layer_{index}_weight"] = np.asarray(layer["w"], dtype=np.float32)
        payload[f"layer_{index}_bias"] = np.asarray(layer["b"], dtype=np.float32)
    np.savez(path, **payload)


def run_grid(manifest_path: Path, output: Path, *, epochs: int = MAX_EPOCHS,
             test_configuration: bool = False) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError((output, "refusing overwrite"))
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError("invalid epoch count")
    if epochs != MAX_EPOCHS and not test_configuration:
        raise ValueError("production grid requires 1200 epochs")
    manifest, data = load_dataset(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    candidates, history_rows = [], []
    for lambda_break in LAMBDA_BREAK:
        for seed in SEEDS:
            params, normalization, history, metadata = train_one(
                data, seed=seed, lambda_break=lambda_break, epochs=epochs,
            )
            run_id = f"mode_lambda{lambda_break:g}_seed{seed}"
            checkpoint = output / f"{run_id}.npz"
            checkpoint_metadata = {
                **metadata, "run_id": run_id, "dataset_content_sha256": manifest["content_sha256"],
                "selection_split": "validation", "test_data_used": False,
            }
            save_checkpoint(checkpoint, params, normalization, checkpoint_metadata)
            history_rows.extend({"run_id": run_id, **row} for row in history)
            candidate = {
                **checkpoint_metadata, "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
            }
            candidate["policy_sha256"] = content_hash(candidate)
            candidates.append(candidate)
    provisional = min(candidates, key=lambda row: (
        row["validation_total_loss"], -row["validation_accuracy_resolved_only"],
        row["lambda_break"], row["seed"], row["best_epoch"],
    ))
    with (output / "training_history.csv").open("w", newline="") as handle:
        fields = list(history_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(history_rows)
    result = {
        "schema": "semantic_mode_training_grid_v1", "dataset_manifest": str(manifest_path.resolve()),
        "dataset_content_sha256": manifest["content_sha256"], "candidate_count": len(candidates),
        "seeds": list(SEEDS), "lambda_break_family": list(LAMBDA_BREAK), "epochs": epochs,
        "candidates": candidates, "provisional_validation_choice": provisional,
        "requires_full_episode_validation": True, "selection_data": "validation only",
        "calibration_or_test_used": False,
    }
    (output / "checkpoint_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        manifest, data = load_dataset(args.dataset_manifest)
        print(json.dumps({
            "status": "PREFLIGHT_ONLY", "decision_count": int(len(data["features"])),
            "resolved_train": manifest["resolved_train"],
            "resolved_validation": manifest["resolved_validation"],
            "would_train_candidates": len(SEEDS)*len(LAMBDA_BREAK),
        }, indent=2))
        return
    result = run_grid(args.dataset_manifest, args.output_directory)
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
