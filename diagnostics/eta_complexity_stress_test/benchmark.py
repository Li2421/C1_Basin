"""Frozen, reproducible initial-condition set for the eta stress test."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .environment import CoupledDualIntersectionEnv, StressConfig, default_positions


BENCHMARK_VERSION = "coupled_dual_intersection_ic_v5"
DEFAULT_SEED = 20260924
DEFAULT_COUNT = 96


@dataclass(frozen=True)
class BenchmarkIC:
    identifier: str
    family: str
    positions: np.ndarray

    def to_dict(self) -> dict:
        return {"id": self.identifier, "family": self.family, "positions": self.positions.tolist()}


def _candidate(rng: np.random.Generator, family: str) -> np.ndarray:
    """Sample a perturbation without embedding an ordering/schedule decision."""
    p = default_positions().copy()
    approach = rng.uniform(-0.18, 0.18, size=8)
    lateral = rng.uniform(-0.026, 0.026, size=8)
    if family == "j1_vertical_pressure":
        approach[4:6] = rng.uniform(-0.04, 0.18, size=2)
        approach[:4] = rng.uniform(-0.18, -0.02, size=4)
    elif family == "j2_vertical_pressure":
        approach[6:8] = rng.uniform(-0.04, 0.18, size=2)
        approach[:4] = rng.uniform(-0.18, -0.02, size=4)
    elif family == "left_right_bridge_pressure":
        approach[:4] = rng.uniform(0.02, 0.20, size=4)
    elif family != "balanced":
        raise ValueError(f"unknown family {family}")
    # Positive approach means closer to the relevant junction.
    p[0:2, 0] += approach[0:2]
    p[2:4, 0] -= approach[2:4]
    p[4, 1] -= approach[4]
    p[5, 1] += approach[5]
    p[6, 1] -= approach[6]
    p[7, 1] += approach[7]
    p[0:4, 1] += lateral[0:4]
    p[4:6, 0] += lateral[4:6]
    p[6:8, 0] += lateral[6:8]
    return p


def generate_benchmark(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED, config: StressConfig | None = None) -> list[BenchmarkIC]:
    if count <= 0:
        raise ValueError("count must be positive")
    env = CoupledDualIntersectionEnv(config)
    rng = np.random.default_rng(seed)
    families = ("balanced", "j1_vertical_pressure", "j2_vertical_pressure", "left_right_bridge_pressure")
    result: list[BenchmarkIC] = []
    for index in range(count):
        family = families[index % len(families)]
        for _ in range(1000):
            positions = _candidate(rng, family)
            try:
                env.reset(positions)
            except ValueError:
                continue
            result.append(BenchmarkIC(f"IC{index:03d}", family, positions))
            break
        else:
            raise RuntimeError(f"could not produce safe {family} initial condition")
    return result


def benchmark_digest(items: list[BenchmarkIC]) -> str:
    payload = json.dumps([item.to_dict() for item in items], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_benchmark(path: str | Path, items: list[BenchmarkIC], config: StressConfig) -> Path:
    """Write once; refusing a changed set makes the IC corpus an artifact."""
    destination = Path(path)
    payload = {
        "schema": BENCHMARK_VERSION,
        "seed": DEFAULT_SEED,
        "environment_fingerprint": config.fingerprint,
        "count": len(items),
        "digest": benchmark_digest(items),
        "items": [item.to_dict() for item in items],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace frozen IC artifact: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    return destination


def load_benchmark(path: str | Path, config: StressConfig) -> list[BenchmarkIC]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BENCHMARK_VERSION:
        raise ValueError("unexpected benchmark schema")
    if payload.get("environment_fingerprint") != config.fingerprint:
        raise ValueError("benchmark/environment fingerprint mismatch")
    result = [BenchmarkIC(str(row["id"]), str(row["family"]), np.asarray(row["positions"], dtype=np.float64)) for row in payload["items"]]
    if benchmark_digest(result) != payload.get("digest"):
        raise ValueError("benchmark digest mismatch")
    return result
