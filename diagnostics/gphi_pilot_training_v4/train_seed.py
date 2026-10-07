"""One frozen V4 MLP training seed; launched in parallel across two shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
core.DATA = DATA; core.HERE = HERE


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); args = parser.parse_args()
    arrays, audit, _, schema, _ = core.dataset_audit()
    if not audit["passed"]:
        raise RuntimeError(audit["failures"])
    train_indices = np.flatnonzero(arrays["split"] == "train")
    val_indices = np.flatnonzero(arrays["split"] == "validation")
    mean, scale, binary = core.fit_normalization(arrays["features"][train_indices], schema)
    x_train = core.normalize(arrays["features"][train_indices], mean, scale).astype(np.float32)
    x_val = core.normalize(arrays["features"][val_indices], mean, scale).astype(np.float32)
    y = arrays["targets"].astype(np.float32)
    config = {
        "name": f"FINAL_128x128_s{args.seed}", "hidden": [128, 128],
        "learning_rate": 1e-3, "weight_decay": 1e-5, "seed": args.seed,
        "batch_size": 256, "max_epochs": 1200, "eval_interval": 5,
        "patience_evaluations": 35, "min_delta": 1e-9,
    }
    params, history, summary = core.train_mlp(
        x_train, y[train_indices], arrays["state_id"][train_indices],
        x_val, y[val_indices], arrays["state_id"][val_indices], config)
    train_pred = core.predict_mlp(params, x_train); val_pred = core.predict_mlp(params, x_val)
    train_metrics = core.subset_metrics(train_pred, y[train_indices], arrays["state_id"][train_indices])
    val_metrics = core.subset_metrics(val_pred, y[val_indices], arrays["state_id"][val_indices])
    workers = HERE / "workers"; workers.mkdir(parents=True, exist_ok=True)
    core.save_checkpoint(workers / f"seed{args.seed}.npz", params, config, mean, scale, binary)
    (workers / f"seed{args.seed}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    output = {"seed": args.seed, "config": config, "summary": summary, "train_metrics": train_metrics, "validation_metrics": val_metrics}
    (workers / f"seed{args.seed}_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "best_epoch": summary["best_epoch"], "validation_state_grouped_mse": val_metrics["state_grouped_mse"], "validation_mean_l2": val_metrics["state_grouped_mean_l2"], "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
