"""Reproducible direct search and black-box solver probes over fixed 3-D eta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .controller import RolloutTrace, rollout
from .environment import StressConfig


@dataclass(frozen=True)
class EtaBounds:
    """Declared experimental search domain, not a new eta law or frozen bound.

    The repository has no frozen eta bounds. Each current basis row has norm
    at most ``v_max``; this finite, reproducible domain caps the triangle-bound
    on the unprojected shared correction at
    ``0.5 * (1.5 + 1 + 1) = 1.75 m/s`` before the existing second projection.
    Results are statements about this declared domain, never all real-valued
    eta.
    """

    lower: tuple[float, float, float] = (0.0, -1.0, -1.0)
    upper: tuple[float, float, float] = (1.5, 1.0, 1.0)

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("eta lower bound must be strictly smaller than upper")

    @property
    def width(self) -> np.ndarray:
        return np.asarray(self.upper) - np.asarray(self.lower)

    def scale(self, unit: np.ndarray) -> np.ndarray:
        return np.asarray(self.lower) + np.asarray(unit, dtype=np.float64) * self.width

    def normalize(self, eta: np.ndarray) -> np.ndarray:
        return (np.asarray(eta, dtype=np.float64) - np.asarray(self.lower)) / self.width

    def clip(self, eta: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(eta, dtype=np.float64), self.lower, self.upper)

    def to_dict(self) -> dict:
        return {"lower": list(self.lower), "upper": list(self.upper)}


@dataclass(frozen=True)
class SearchSamples:
    strategy: str
    points: np.ndarray
    success: np.ndarray
    steps: np.ndarray
    objective: np.ndarray
    terminations: tuple[str, ...]
    failure_modes: tuple[str, ...]
    coordination_modes: tuple[str, ...]

    @property
    def success_fraction(self) -> float:
        return float(np.mean(self.success)) if len(self.success) else float("nan")

    @property
    def first_success_evaluation(self) -> int | None:
        indices = np.flatnonzero(self.success)
        return None if not len(indices) else int(indices[0] + 1)

    def summary(self, bounds: EtaBounds) -> dict:
        values, counts = np.unique(self.terminations, return_counts=True)
        return {
            "strategy": self.strategy, "evaluations": int(len(self.points)),
            "successes": int(self.success.sum()), "success_fraction": self.success_fraction,
            "first_success_evaluation": self.first_success_evaluation,
            "termination_counts": {str(name): int(count) for name, count in zip(values, counts)},
            "bounds": bounds.to_dict(), "components": basin_components(self.points, self.success, bounds),
        }


def rollout_objective(trace: RolloutTrace) -> float:
    """A deterministic ranking signal for CEM, never an optimization over actions."""
    if trace.success:
        return -10_000.0 + trace.episode_steps / 10_000.0
    final_error = trace.summary["final_sum_goal_error"]
    penalty = {
        "projection_failure": 100.0, "agent_collision": 30.0,
        "wall_collision": 30.0, "agent_and_wall_collision": 35.0,
        "strict_deadlock": 15.0, "bridge_blockage": 12.0,
        "junction_cyclic_waiting": 11.0, "stalled_congestion": 10.0,
    }.get(trace.failure_mode, 20.0)
    return float(penalty + final_error + 0.002 * trace.episode_steps)


def _evaluate(
    points: np.ndarray,
    initial_positions: np.ndarray,
    config: StressConfig,
    strategy: str,
) -> SearchSamples:
    traces = [rollout(initial_positions, eta=eta, variant="eta", config=config) for eta in points]
    return SearchSamples(
        strategy=strategy, points=np.asarray(points, dtype=np.float64),
        success=np.asarray([trace.success for trace in traces], dtype=bool),
        steps=np.asarray([trace.episode_steps for trace in traces], dtype=np.int64),
        objective=np.asarray([rollout_objective(trace) for trace in traces], dtype=np.float64),
        terminations=tuple(trace.termination for trace in traces),
        failure_modes=tuple(trace.failure_mode for trace in traces),
        coordination_modes=tuple(trace.summary["coordination_signature"] for trace in traces),
    )


def uniform_search(initial_positions: np.ndarray, count: int, seed: int, config: StressConfig, bounds: EtaBounds | None = None) -> SearchSamples:
    bounds = bounds or EtaBounds()
    if count <= 0:
        raise ValueError("count must be positive")
    points = np.random.default_rng(seed).uniform(bounds.lower, bounds.upper, size=(count, 3))
    return _evaluate(points, initial_positions, config, "uniform")


def sobol_search(initial_positions: np.ndarray, count: int, seed: int, config: StressConfig, bounds: EtaBounds | None = None) -> SearchSamples:
    bounds = bounds or EtaBounds()
    if count <= 0:
        raise ValueError("count must be positive")
    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    if count > 0 and count & (count - 1) == 0:
        unit = engine.random_base2(int(np.log2(count)))
    else:
        unit = engine.random(count)
    return _evaluate(bounds.scale(unit), initial_positions, config, "sobol")


def basin_components(points: np.ndarray, success: np.ndarray, bounds: EtaBounds, radius: float | None = None) -> dict:
    """Resolution-aware components in normalized eta space.

    This is a sample-topology diagnostic, not a claim about the exact continuum
    topology of the unknown success set.
    """
    points, success = np.asarray(points), np.asarray(success, dtype=bool)
    good = bounds.normalize(points[success])
    if not len(good):
        return {"success_points": 0, "components": 0, "largest_component_fraction": 0.0, "radius": None}
    radius = float(radius if radius is not None else 1.75 * (np.log(max(len(points), 2)) / len(points)) ** (1 / 3))
    parent = np.arange(len(good))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left, right in cKDTree(good).query_pairs(radius):
        union(int(left), int(right))
    sizes: dict[int, int] = {}
    for index in range(len(good)):
        sizes[find(index)] = sizes.get(find(index), 0) + 1
    largest = max(sizes.values())
    return {
        "success_points": int(len(good)), "components": int(len(sizes)),
        "largest_component_fraction": float(largest / len(good)), "radius": radius,
        "component_sizes": sorted(sizes.values(), reverse=True),
    }


def local_robustness(
    initial_positions: np.ndarray,
    centers: np.ndarray,
    seed: int,
    config: StressConfig,
    bounds: EtaBounds | None = None,
    samples_per_radius: int = 96,
    radii: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
) -> list[dict]:
    bounds = bounds or EtaBounds()
    rng = np.random.default_rng(seed)
    rows = []
    for center_index, center in enumerate(np.asarray(centers, dtype=np.float64)):
        for radius in radii:
            directions = rng.normal(size=(samples_per_radius, 3))
            directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-12)
            magnitudes = radius * rng.random(samples_per_radius) ** (1 / 3)
            points = bounds.clip(center + directions * magnitudes[:, None] * bounds.width)
            result = _evaluate(points, initial_positions, config, f"local_r{radius:g}")
            rows.append({
                "center_index": int(center_index), "center": center.tolist(), "normalized_radius": float(radius),
                "samples": int(samples_per_radius), "success_fraction": result.success_fraction,
                "first_success_evaluation": result.first_success_evaluation,
            })
    return rows


def cem_search(
    initial_positions: np.ndarray,
    budget: int,
    seed: int,
    config: StressConfig,
    bounds: EtaBounds | None = None,
    population: int = 32,
    elite_fraction: float = 0.25,
) -> SearchSamples:
    """Diagonal cross-entropy method: a generic derivative-free eta solver."""
    bounds = bounds or EtaBounds()
    if budget <= 0 or population <= 1:
        raise ValueError("positive budget and population > 1 required")
    rng = np.random.default_rng(seed)
    mean, std = (np.asarray(bounds.lower) + np.asarray(bounds.upper)) / 2, bounds.width / 2
    batches = []
    remaining = budget
    elite_count = max(1, int(np.ceil(population * elite_fraction)))
    while remaining:
        count = min(population, remaining)
        points = bounds.clip(rng.normal(mean, std, size=(count, 3)))
        batch = _evaluate(points, initial_positions, config, "cem")
        batches.append(batch)
        if batch.success.any():
            break
        elite = batch.points[np.argsort(batch.objective)[:min(elite_count, count)]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0) * 1.4, bounds.width * 0.015)
        remaining -= count
    return SearchSamples(
        strategy="cem", points=np.concatenate([batch.points for batch in batches]),
        success=np.concatenate([batch.success for batch in batches]), steps=np.concatenate([batch.steps for batch in batches]),
        objective=np.concatenate([batch.objective for batch in batches]),
        terminations=tuple(value for batch in batches for value in batch.terminations),
        failure_modes=tuple(value for batch in batches for value in batch.failure_modes),
        coordination_modes=tuple(value for batch in batches for value in batch.coordination_modes),
    )


def solver_benchmark(initial_positions: np.ndarray, config: StressConfig, bounds: EtaBounds | None = None, budget: int = 512, repeats: int = 12, seed: int = 3701) -> list[dict]:
    bounds = bounds or EtaBounds()
    rows = []
    solvers: tuple[tuple[str, Callable[..., SearchSamples]], ...] = (
        ("uniform", uniform_search), ("sobol", sobol_search), ("cem", cem_search),
    )
    for solver_index, (name, function) in enumerate(solvers):
        for repeat in range(repeats):
            result = function(initial_positions, count=budget, seed=seed + 10_000 * solver_index + repeat, config=config, bounds=bounds) if name != "cem" else function(initial_positions, budget=budget, seed=seed + 10_000 * solver_index + repeat, config=config, bounds=bounds)
            rows.append({
                "solver": name, "repeat": repeat, "budget": budget,
                "success": bool(result.success.any()), "evaluations": int(len(result.points)),
                "evaluations_to_first_success": result.first_success_evaluation,
            })
    return rows
