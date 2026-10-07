"""Run one preregistered (H, model, LOGO fold, seed) gate job.

The script consumes only TRAINING_READY.json/fold_manifest.json and refuses
to train if any frozen hash changed.  It never reads Stage-1 ambiguous states
into normalization, BCE, validation thresholding, or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import jax
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4")
CV = Path("/home/zhihan/research/Basin_C1/diagnostics/hard_stable_boundary_crossval")
sys.path.insert(0, str(CV))
import run_crossval as cv  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, params, mean, scale, binary, metadata: dict) -> None:
    arrays = {
        "normalization_mean": np.asarray(mean), "normalization_scale": np.asarray(scale),
        "normalization_binary_mask": np.asarray(binary),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, layer in enumerate(params):
        arrays[f"layer_{index}_weight"] = np.asarray(layer["w"])
        arrays[f"layer_{index}_bias"] = np.asarray(layer["b"])
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--model", choices=("LINEAR", "MLP_64x64"), required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, choices=(17,), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)

    ready_path = HERE / "TRAINING_READY.json"
    ready = json.loads(ready_path.read_text())
    folds_path = HERE / "fold_manifest.json"
    targets_path = HERE / "window_targets.csv"
    if ready["fold_manifest_sha256"] != sha(folds_path) or ready["window_targets_sha256"] != sha(targets_path):
        raise AssertionError("Stage-2 frozen hash changed")
    if ready["candidate_windows_sha256"] != sha(ROOT / "stage1_window" / "candidate_windows.json"):
        raise AssertionError("candidate horizon hash changed after training-ready freeze")
    if ready["seed_preregistration_sha256"] != sha(HERE / "seed_preregistration.json"):
        raise AssertionError("seed preregistration changed")
    manifest = json.loads(folds_path.read_text())
    if manifest["samples_npz_sha256"] != sha(DATA / "samples.npz"):
        raise AssertionError("frozen samples.npz hash changed")
    if manifest["feature_schema_sha256"] != sha(DATA / "feature_schema.json"):
        raise AssertionError("frozen feature schema hash changed")
    fold = next((row for row in manifest["folds"] if int(row["H"]) == args.horizon and row["fold_id"] == args.fold_id), None)
    if fold is None:
        raise ValueError("requested preregistered fold not found")
    hidden = [] if args.model == "LINEAR" else [64, 64]
    output = HERE / "runs" / f"H{args.horizon}" / args.model / args.fold_id / f"seed_{args.seed}"
    if output.exists():
        if args.resume and (output / "manifest.json").exists():
            old = json.loads((output / "manifest.json").read_text())
            if old.get("training_ready_sha256") == sha(ready_path):
                print(json.dumps({"resumed": True, "output": str(output)}, indent=2))
                return
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()

    target_rows = [row for row in read_csv(targets_path) if int(row["H"]) == args.horizon]
    label = {row["state_id"]: int(row["y_H"]) for row in target_rows}
    meta = {row["state_id"]: row for row in target_rows}
    with np.load(DATA / "samples.npz", allow_pickle=False) as loaded:
        features = np.asarray(loaded["features"])
        state_id = loaded["state_id"].astype(str)
    if features.shape[1] != 214:
        raise AssertionError(features.shape)
    schema = json.loads((DATA / "feature_schema.json").read_text())
    train_ids = set(fold["train_state_ids"])
    val_ids = set(fold["validation_state_ids"])
    test_ids = set(fold["test_state_ids"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("state leakage")
    available_ids = set(state_id.tolist())
    missing_sample_ids = (train_ids | val_ids | test_ids) - available_ids
    if missing_sample_ids:
        raise AssertionError(f"manifest states missing samples: {sorted(missing_sample_ids)}")
    train_idx = np.flatnonzero(np.isin(state_id, sorted(train_ids)))
    val_idx = np.flatnonzero(np.isin(state_id, sorted(val_ids)))
    test_idx = np.flatnonzero(np.isin(state_id, sorted(test_ids)))
    if any(sid not in label for sid in train_ids | val_ids | test_ids):
        raise AssertionError("ambiguous/unregistered state entered fold")
    mean, scale, binary = cv.normalization(features[train_idx], schema)
    normalized = ((features - mean) / scale).astype(np.float32)
    y_train = np.asarray([label[sid] for sid in state_id[train_idx]], np.int8)
    params, best_epoch, val_bce = cv.train_model(
        hidden, args.seed, normalized[train_idx], y_train, state_id[train_idx],
        normalized[val_idx], state_id[val_idx], label,
    )
    val_ids_out, val_p, val_y = cv.aggregate_state(cv.predict(params, normalized[val_idx]), state_id[val_idx], label)
    test_ids_out, test_p, test_y = cv.aggregate_state(cv.predict(params, normalized[test_idx]), state_id[test_idx], label)
    threshold = cv.select_threshold(val_y, val_p)
    metrics = cv.metrics(test_y, test_p, threshold)
    val_rows = [{"split": "validation", "H": args.horizon, "model": args.model, "fold_id": args.fold_id, "seed": args.seed, "state_id": sid, "source_group": meta[sid]["source_group"], "category": meta[sid]["category"], "is_hard13_anchor": meta[sid]["is_hard13_anchor"], "y_long_diagnostic_only": meta[sid]["y_long_diagnostic_only"], "y_H": int(y), "p_gate": float(p), "threshold": threshold, "predicted": int(p >= threshold), "correct": int((p >= threshold) == y)} for sid, p, y in zip(val_ids_out, val_p, val_y)]
    test_rows = [{"split": "test", "H": args.horizon, "model": args.model, "fold_id": args.fold_id, "seed": args.seed, "state_id": sid, "source_group": meta[sid]["source_group"], "category": meta[sid]["category"], "is_hard13_anchor": meta[sid]["is_hard13_anchor"], "y_long_diagnostic_only": meta[sid]["y_long_diagnostic_only"], "y_H": int(y), "p_gate": float(p), "threshold": threshold, "predicted": int(p >= threshold), "correct": int((p >= threshold) == y)} for sid, p, y in zip(test_ids_out, test_p, test_y)]
    write_csv(output / "predictions.csv", val_rows + test_rows)
    metadata = {"H": args.horizon, "model": args.model, "architecture": [214, *hidden, 1], "activation": "SiLU" if hidden else "linear", "loss": "ordinary BCE", "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)", "seed": args.seed, "fold_id": args.fold_id, "heldout_source_group": fold["outer_test_source_group"], "best_epoch": best_epoch, "validation_state_BCE": val_bce, "validation_selected_threshold": threshold, "training_ready_sha256": sha(ready_path)}
    save_checkpoint(output / "checkpoint.npz", params, mean, scale, binary, metadata)
    result = {**metadata, **metrics, "wall_seconds": time.monotonic() - started, "device": [str(x) for x in jax.devices()], "finite": bool(np.isfinite(val_p).all() and np.isfinite(test_p).all())}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    run_manifest = {"training_ready_sha256": sha(ready_path), "fold_manifest_sha256": sha(folds_path), "window_targets_sha256": sha(targets_path), "predictions_sha256": sha(output / "predictions.csv"), "checkpoint_sha256": sha(output / "checkpoint.npz"), "result_sha256": sha(output / "result.json")}
    (output / "manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
