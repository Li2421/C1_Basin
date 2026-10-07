"""Frozen constants and small inference helpers for semantic Stage 2.

Stage 2 contains no periodic clock.  A NORMAL decision is one of Safety (S),
one local Direct-g correction (L), or one-way entry into persistent structured
eta recovery (R).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SEMANTIC_ROOT = ROOT / "diagnostics/semantic_unified_controller_v1"
HERE = SEMANTIC_ROOT / "stage2_recovery_entry"
PILOT_ROOT = ROOT / "diagnostics/single_segment_recovery_training_v1"
LOCAL_STAGE = SEMANTIC_ROOT / "stage1_state_driven_local"

DIRECT_CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
DIRECT_SHA256 = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
ETA_CHECKPOINT = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
ETA_SHA256 = "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"
LOCAL_HEAD = LOCAL_STAGE / "training/local_final/entry_g_lambda0_seed17.npz"
LOCAL_THRESHOLD_PROBABILITY = 0.75

HORIZON = 850
FUTURES = 32
FUTURE_ROOT_SEED = 2026102201
MAX_BRANCH_CONTINUATIONS = 20_000
MAX_PHYSICAL_STEPS = 10_000_000
ACTION_NAMES = {0: "SAFETY_NOW", 1: "LOCAL_NOW", 2: "ENTER_RECOVERY_NOW"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_content_hash(value: dict[str, Any], path: Path) -> None:
    claimed = value.get("content_sha256")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if not claimed or content_hash(body) != claimed:
        raise RuntimeError((str(path), "semantic content hash mismatch"))


def logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError(probability)
    return float(np.log(probability) - np.log1p(-probability))


class FrozenModeHead:
    """NumPy inference for a frozen 214->64->64->3 SiLU mode selector."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as values:
            self.mean = np.asarray(values["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(values["normalization_scale"], dtype=np.float64)
            self.weights = [np.asarray(values[f"layer_{i}_weight"], dtype=np.float64) for i in range(3)]
            self.biases = [np.asarray(values[f"layer_{i}_bias"], dtype=np.float64) for i in range(3)]
        if self.mean.shape != (214,) or self.scale.shape != (214,):
            raise RuntimeError("mode-head normalization mismatch")
        if [value.shape for value in self.weights] != [(214, 64), (64, 64), (64, 3)]:
            raise RuntimeError("mode-head architecture mismatch")
        if np.any(self.scale <= 0):
            raise RuntimeError("invalid mode-head scale")

    def logits(self, feature: np.ndarray) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float64)
        if value.shape != (214,) or not np.isfinite(value).all():
            raise ValueError(("invalid mode feature", value.shape))
        value = (value - self.mean) / self.scale
        for index in range(2):
            value = value @ self.weights[index] + self.biases[index]
            value = value / (1.0 + np.exp(-value))
        return np.asarray(value @ self.weights[2] + self.biases[2], dtype=np.float64)

    def __call__(self, feature: np.ndarray) -> int:
        return int(np.argmax(self.logits(feature)))


__all__ = [
    "ACTION_NAMES", "DIRECT_CHECKPOINT", "DIRECT_SHA256", "ETA_CHECKPOINT",
    "ETA_SHA256", "FUTURES", "FUTURE_ROOT_SEED", "HERE", "HORIZON",
    "LOCAL_HEAD", "LOCAL_STAGE", "LOCAL_THRESHOLD_PROBABILITY",
    "MAX_BRANCH_CONTINUATIONS", "MAX_PHYSICAL_STEPS", "PILOT_ROOT", "ROOT",
    "SEMANTIC_ROOT", "FrozenModeHead", "content_hash", "logit", "sha256",
    "verify_content_hash",
]
