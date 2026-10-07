"""Causal startup adapter for the frozen V3 214-D feature builder.

Only the goal-error history presented to the authoritative builder is adapted:
when fewer than 41 observations exist, the earliest available error is repeated
on the left.  Every other field and all post-warmup behavior are delegated to
the frozen V3 implementation.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
AUTHORITATIVE_PROJECTION = SYSROOT / "single_integrator/cbf.py"
AUTHORITATIVE_RETRY = ROOT / "diagnostics/success_basin_multimodality/exact_projector.py"
AUTHORITATIVE_FEATURE_BUILDER = ROOT / "diagnostics/gphi_training_dataset_v2/finalize_dataset.py"

EXPECTED_SHA256 = {
    "projection": "841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48",
    "projection_retry": "e29d510dc1752f138bdfcc491f8f5008dbcd215282301396852bdbfd754ac544",
    "feature_builder": "5ec2ba5ab447a707022f693719731595a76d92e6352b3f66c19caedf2baf2341",
}
HISTORY_LENGTH = 41
FEATURE_DIMENSION = 214


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_module_file(module: Any) -> Path:
    source = getattr(module, "__file__", None)
    if source is None:
        raise RuntimeError(("module has no source file", module.__name__))
    return Path(source).resolve()


def assert_authoritative_sources() -> dict[str, Any]:
    """Fail closed if imports or bytes differ from the V3 frozen sources."""
    expected_paths = {
        "projection": AUTHORITATIVE_PROJECTION,
        "projection_retry": AUTHORITATIVE_RETRY,
        "feature_builder": AUTHORITATIVE_FEATURE_BUILDER,
    }
    actual_hashes: dict[str, str] = {}
    for name, path in expected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hashes[name] = sha256(path)
        if actual_hashes[name] != EXPECTED_SHA256[name]:
            raise RuntimeError(
                ("frozen source hash mismatch", name, str(path), EXPECTED_SHA256[name], actual_hashes[name])
            )

    # SYSROOT must win module resolution.  Refuse an already-loaded alternate
    # module instead of silently replacing it.
    loaded = sys.modules.get("single_integrator.cbf")
    if loaded is not None and _resolved_module_file(loaded) != AUTHORITATIVE_PROJECTION.resolve():
        raise RuntimeError(
            ("alternate projection already imported", str(_resolved_module_file(loaded)), str(AUTHORITATIVE_PROJECTION))
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SYSROOT) in sys.path:
        sys.path.remove(str(SYSROOT))
    sys.path.insert(0, str(SYSROOT))
    cbf_module = importlib.import_module("single_integrator.cbf")
    imported_projection = _resolved_module_file(cbf_module)
    if imported_projection != AUTHORITATIVE_PROJECTION.resolve():
        raise RuntimeError(("projection import path mismatch", str(imported_projection)))

    return {
        "status": "PASS",
        "sysroot": str(SYSROOT.resolve()),
        "authoritative_paths": {name: str(path.resolve()) for name, path in expected_paths.items()},
        "expected_sha256": dict(EXPECTED_SHA256),
        "actual_sha256": actual_hashes,
        "imported_projection_path": str(imported_projection),
    }


# Assert before importing the authoritative implementation and its transitive
# projection dependency.
SOURCE_AUDIT = assert_authoritative_sources()

from diagnostics.gphi_training_dataset_v2.finalize_dataset import FeatureBuilder  # noqa: E402


def left_pad_goal_error_history(history: Any, length: int = HISTORY_LENGTH) -> np.ndarray:
    """Return a finite ``[length, 2]`` tail with causal earliest-value padding."""
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or len(values) == 0:
        raise ValueError(("expected nonempty goal-error history [n,2]", values.shape))
    if not np.isfinite(values).all():
        raise ValueError("goal-error history contains NaN/Inf")
    tail = values[-length:]
    if len(tail) < length:
        prefix = np.repeat(tail[:1], length - len(tail), axis=0)
        tail = np.concatenate((prefix, tail), axis=0)
    if tail.shape != (length, 2):
        raise AssertionError(("padded history shape", tail.shape))
    return tail


class _HistoryOverride:
    """Read-through environment view overriding only ``distance_history``."""

    def __init__(self, env: Any, distance_history: np.ndarray) -> None:
        self._env = env
        self.distance_history = np.asarray(distance_history, dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class StartupAwareFeatureBuilder:
    """Composition wrapper preserving the frozen FeatureBuilder implementation."""

    def __init__(self) -> None:
        assert_authoritative_sources()
        self._delegate = FeatureBuilder()

    @property
    def schema(self) -> list[dict]:
        return self._delegate.schema

    def build(self, env: Any, first: dict, config: Any, cbf: Any) -> tuple[np.ndarray, dict]:
        padded = left_pad_goal_error_history(env.distance_history)
        feature_env = _HistoryOverride(env, padded)
        vector, structured = self._delegate.build(feature_env, first, config, cbf)
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
            raise AssertionError(("invalid startup-aware feature", vector.shape, bool(np.isfinite(vector).all())))
        # Expose provenance without changing the authoritative structured keys.
        return vector, structured

    @staticmethod
    def history_metadata(env: Any) -> dict[str, Any]:
        available = int(len(env.distance_history))
        return {
            "available_history_points": available,
            "required_history_points": HISTORY_LENGTH,
            "left_padding_points": max(0, HISTORY_LENGTH - min(available, HISTORY_LENGTH)),
            "startup_padding_active": available < HISTORY_LENGTH,
            "padding_value": np.asarray(env.distance_history[0], dtype=np.float64).tolist(),
        }


__all__ = [
    "AUTHORITATIVE_FEATURE_BUILDER",
    "AUTHORITATIVE_PROJECTION",
    "AUTHORITATIVE_RETRY",
    "EXPECTED_SHA256",
    "FEATURE_DIMENSION",
    "HISTORY_LENGTH",
    "SOURCE_AUDIT",
    "StartupAwareFeatureBuilder",
    "assert_authoritative_sources",
    "left_pad_goal_error_history",
]
