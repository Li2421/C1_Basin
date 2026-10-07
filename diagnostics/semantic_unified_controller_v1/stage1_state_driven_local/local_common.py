"""Shared Stage-1 state-driven Safety-vs-local-correction semantics.

This module contains no H/L clock, cooldown, or recovery mode.  At every
nonterminal transition a 214-D state feature selects Safety (S) or one frozen
Direct-g correction (L); the next transition makes a fresh state decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage1_state_driven_local"
DIRECT_CHECKPOINT = (
    ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
)
DIRECT_SHA256 = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
HORIZON = 850
FUTURES = 32
MAX_CONTINUATIONS = 20_000
MAX_PHYSICAL_STEPS = 10_000_000
FUTURE_ROOT_SEED = 2026101201


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


class ConstantLocalHead:
    def __init__(self, local: bool):
        self.local = bool(local)

    def __call__(self, feature: np.ndarray) -> bool:
        return self.local


class FrozenLocalHead:
    """Inference for the frozen 214->64->64->1 local-decision head."""

    def __init__(self, path: Path, *, threshold_logit: float | None = None):
        with np.load(path, allow_pickle=False) as values:
            self.mean = np.asarray(values["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(values["normalization_scale"], dtype=np.float64)
            self.weights = [np.asarray(values[f"layer_{i}_weight"], dtype=np.float64) for i in range(3)]
            self.biases = [np.asarray(values[f"layer_{i}_bias"], dtype=np.float64) for i in range(3)]
            stored = float(values["threshold_logit"])
        self.threshold = stored if threshold_logit is None else float(threshold_logit)
        if self.mean.shape != (214,) or self.scale.shape != (214,):
            raise RuntimeError((path, "normalization dimension mismatch"))
        if [item.shape for item in self.weights] != [(214, 64), (64, 64), (64, 1)]:
            raise RuntimeError((path, "head architecture mismatch"))
        if [item.shape for item in self.biases] != [(64,), (64,), (1,)]:
            raise RuntimeError((path, "head bias mismatch"))
        if np.any(self.scale <= 0) or not np.isfinite(self.threshold):
            raise RuntimeError((path, "invalid head normalization/threshold"))

    def logit(self, feature: np.ndarray) -> float:
        value = np.asarray(feature, dtype=np.float64)
        if value.shape != (214,) or not np.isfinite(value).all():
            raise ValueError(("invalid local-decision feature", value.shape))
        value = (value - self.mean) / self.scale
        for index in range(2):
            value = value @ self.weights[index] + self.biases[index]
            value = value / (1.0 + np.exp(-value))
        return float((value @ self.weights[2] + self.biases[2]).item())

    def __call__(self, feature: np.ndarray) -> bool:
        return self.logit(feature) >= self.threshold


class FirstLocalOverride:
    """Execute the paired S/L action once, then use the same incumbent."""

    def __init__(self, action: int, downstream: Callable[[np.ndarray], bool]):
        if action not in (0, 1):
            raise ValueError(action)
        self.action = bool(action)
        self.downstream = downstream
        self.calls = 0

    def __call__(self, feature: np.ndarray) -> bool:
        self.calls += 1
        return self.action if self.calls == 1 else bool(self.downstream(feature))


@dataclass(frozen=True)
class LocalStepRecord:
    step: int
    decision: str
    feature: np.ndarray
    u_flow: np.ndarray
    u_safe: np.ndarray
    correction: np.ndarray
    u_exec: np.ndarray
    first_projection_retry: bool
    second_projection_retry: bool
    done: bool
    monitor_info: dict[str, Any]


class StateDrivenLocalMachine:
    """Stateless semantic normal-mode decision repeated every physical step."""

    def __init__(self, *, kernel: Any, local_head: Callable[[np.ndarray], bool], direct_model: Any):
        self.kernel = kernel
        self.local_head = local_head
        self.direct_model = direct_model
        self.head_query_count = 0
        self.direct_query_count = 0
        self.records: list[LocalStepRecord] = []

    def step(self) -> LocalStepRecord:
        if bool(self.kernel.env.done):
            raise RuntimeError("terminal state is absorbing")
        context = self.kernel.prepare(need_feature=True)
        feature = np.asarray(context.feature, dtype=np.float64)
        self.head_query_count += 1
        use_local = bool(self.local_head(feature))
        correction = np.zeros((2, 2), dtype=np.float64)
        second_retry = False
        if use_local:
            predicted = np.asarray(self.direct_model(feature[None]), dtype=np.float64)
            if predicted.shape != (1, 4) or not np.isfinite(predicted).all():
                raise RuntimeError(("invalid frozen Direct-g prediction", predicted.shape))
            correction = predicted[0].reshape(2, 2)
            u_exec, second_retry = self.kernel.second_projection(context, correction)
            self.direct_query_count += 1
            decision = "LOCAL"
        else:
            u_exec = context.u_safe
            decision = "SAFETY"
        done, info = self.kernel.execute(u_exec)
        row = LocalStepRecord(
            step=int(context.step), decision=decision, feature=feature.copy(),
            u_flow=context.u_flow.copy(), u_safe=context.u_safe.copy(),
            correction=correction.copy(), u_exec=np.asarray(u_exec).copy(),
            first_projection_retry=bool(context.first_projection_retry),
            second_projection_retry=bool(second_retry), done=bool(done),
            monitor_info=dict(info),
        )
        self.records.append(row)
        return row

    def run_to_terminal(self) -> tuple[LocalStepRecord, ...]:
        while not bool(self.kernel.env.done):
            self.step()
        return tuple(self.records)


__all__ = [
    "BASE", "DIRECT_CHECKPOINT", "DIRECT_SHA256", "FUTURES", "FUTURE_ROOT_SEED",
    "HERE", "HORIZON", "MAX_CONTINUATIONS", "MAX_PHYSICAL_STEPS",
    "ConstantLocalHead", "FirstLocalOverride", "FrozenLocalHead",
    "LocalStepRecord", "StateDrivenLocalMachine", "content_hash", "sha256",
    "verify_content_hash",
]
