"""Small leakage-safe loader for the deterministic G_phi pilot dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_split(split: str, directory: str | Path = HERE) -> dict[str, np.ndarray]:
    """Return deployment inputs/targets for one source-state-grouped split.

    Oracle eta, J_def, and future outcomes are intentionally absent.  Use the
    JSONL metadata separately for audits, never as model input.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    directory = Path(directory)
    with np.load(directory / "samples.npz", allow_pickle=False) as data:
        mask = data["split"] == split
        return {
            "features": np.asarray(data["features"][mask]),
            "targets": np.asarray(data["targets"][mask]),
            "state_id": np.asarray(data["state_id"][mask]),
            "flow_seed": np.asarray(data["flow_seed"][mask]),
            "sample_id": np.asarray(data["sample_id"][mask]),
        }


def feature_schema(directory: str | Path = HERE) -> dict:
    return json.loads((Path(directory) / "feature_schema.json").read_text())


if __name__ == "__main__":
    for name in ("train", "validation", "test"):
        split = load_split(name)
        print(name, split["features"].shape, split["targets"].shape)
