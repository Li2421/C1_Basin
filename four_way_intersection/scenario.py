"""Geometry and independent initial-state distribution for Four-Way.

The entire central square is free space.  There are deliberately no gates,
lanes, traffic lights, priority variables, or order rules in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

AGENT_NAMES = ("A", "B", "C", "D")
# Canonical ordering: North->South, East->West, South->North, West->East.
HEADINGS = np.asarray(((0., -1.), (-1., 0.), (0., 1.), (1., 0.)))
RIGHTS = np.asarray(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))
SPLITS = ("train", "development", "test")


@dataclass(frozen=True)
class Layout:
    walls: np.ndarray
    wall_names: tuple[str, ...]
    goals: np.ndarray


def build_layout(config) -> Layout:
    h = float(config.world_half_extent)
    vertices = np.asarray(((-h, -h), (h, -h), (h, h), (-h, h)), dtype=np.float64)
    walls = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    # Opposite starts/goals use separated parallel tracks.  Their small offset
    # prevents terminal disks from occupying a future agent's start location.
    q = float(config.nominal_lateral_offset)
    g = float(config.nominal_goal_distance)
    goals = np.asarray(((q, -g), (-g, q), (-q, g), (g, -q)), dtype=np.float64)
    return Layout(walls, ("south", "east", "north", "west"), goals)


def nominal_positions(config) -> np.ndarray:
    q, g = float(config.nominal_lateral_offset), float(config.nominal_start_distance)
    return np.asarray(((q, g), (g, q), (-q, -g), (-g, -q)), dtype=np.float64)


def split_seed(split: str, index: int, base_seed: int = 20260930) -> int:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected {SPLITS}")
    # Stable disjoint streams, not a failure-derived test distribution.
    return int(base_seed + {"train": 0, "development": 10_000_000, "test": 20_000_000}[split] + index)


def draw_initial_state(config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw a globally varied, valid start state and reported prior velocity."""
    starts = nominal_positions(config).copy()
    # Longitudinal distance and lateral offset change independently per agent.
    long = rng.uniform(-config.longitudinal_jitter, config.longitudinal_jitter, size=4)
    lateral = rng.uniform(-config.lateral_jitter, config.lateral_jitter, size=4)
    # Small globally correlated asymmetry avoids a dataset of exact symmetries.
    global_shift = rng.uniform(-config.global_lateral_jitter, config.global_lateral_jitter)
    starts += HEADINGS * (-long[:, None]) + RIGHTS * (lateral[:, None] + global_shift)
    velocities = rng.uniform(-config.initial_speed_jitter, config.initial_speed_jitter, size=(4, 2))
    return starts, velocities, {
        "longitudinal_jitter": long.tolist(), "lateral_jitter": lateral.tolist(),
        "global_lateral_shift": float(global_shift),
    }


def sample_initial_state(config, split: str = "train", seed: int = 0, base_seed: int = 20260930):
    """Independent named-split sample; ``seed`` is an index, not a failure ID."""
    return draw_initial_state(config, np.random.default_rng(split_seed(split, seed, base_seed)))


def initial_state_for_split(config, split: str, index: int, base_seed: int = 20260930):
    return sample_initial_state(config, split, index, base_seed)
