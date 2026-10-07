"""NumPy inference for the frozen 214->128->128->3 eta predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FixedDEtaPredictor:
    def __init__(self, checkpoint: Path):
        with np.load(checkpoint, allow_pickle=False) as payload:
            self.mean = np.asarray(payload["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(payload["normalization_scale"], dtype=np.float64)
            self.weights = [np.asarray(payload[f"layer_{i}_weight"], dtype=np.float64) for i in range(3)]
            self.biases = [np.asarray(payload[f"layer_{i}_bias"], dtype=np.float64) for i in range(3)]
            self.config = json.loads(str(payload["architecture_json"].item()))
        if [tuple(value.shape) for value in self.weights] != [(214, 128), (128, 128), (128, 3)]:
            raise RuntimeError([value.shape for value in self.weights])
        self.low = np.asarray(self.config["eta_low"], dtype=np.float64)
        self.high = np.asarray(self.config["eta_high"], dtype=np.float64)

    @staticmethod
    def silu(value: np.ndarray) -> np.ndarray:
        return value / (1.0 + np.exp(-value))

    def predict_normalized_raw(self, features: np.ndarray) -> np.ndarray:
        value = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        value = self.silu(value @ self.weights[0] + self.biases[0])
        value = self.silu(value @ self.weights[1] + self.biases[1])
        return value @ self.weights[2] + self.biases[2]

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self.predict_normalized_raw(features)
        clipped = np.clip(raw, 0.0, 1.0)
        eta = self.low + clipped * (self.high - self.low)
        return eta, raw, clipped
