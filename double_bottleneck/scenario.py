"""Scenario geometry and deterministic initial-state regimes.

This module contains no controller logic.  In particular, it assigns neither
right-of-way nor a scripted crossing order to any agent.
"""
from dataclasses import dataclass

import numpy as np


AGENT_NAMES = ("A1", "A2", "B1", "B2")
LEFT_TO_RIGHT = (0, 1)
RIGHT_TO_LEFT = (2, 3)
INITIAL_REGIMES = ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric")


@dataclass(frozen=True)
class Layout:
    """Closed, piecewise-linear boundary of the union-of-rectangles workspace."""

    walls: np.ndarray
    wall_names: tuple[str, ...]
    goals: np.ndarray


def build_layout(config) -> Layout:
    """Build two narrow passages separated by a wider middle chamber.

    The free workspace is the union of five axis-aligned rectangles: left
    staging area, left bottleneck, middle chamber, right bottleneck, and right
    staging area.  The returned segments are exactly the exposed boundary of
    that union.
    """
    length = float(config.world_half_length)
    outer = float(config.outer_half_height)
    narrow = float(config.bottleneck_width) / 2.0
    chamber = float(config.chamber_half_height)
    inner = float(config.chamber_half_length)
    outer_gate = inner + float(config.bottleneck_length)

    # Traverse the lower boundary left-to-right, close the right end, then
    # traverse the upper boundary right-to-left and close the left end.
    vertices = np.array(
        [
            [-length, -outer],
            [-outer_gate, -outer],
            [-outer_gate, -narrow],
            [-inner, -narrow],
            [-inner, -chamber],
            [inner, -chamber],
            [inner, -narrow],
            [outer_gate, -narrow],
            [outer_gate, -outer],
            [length, -outer],
            [length, outer],
            [outer_gate, outer],
            [outer_gate, narrow],
            [inner, narrow],
            [inner, chamber],
            [-inner, chamber],
            [-inner, narrow],
            [-outer_gate, narrow],
            [-outer_gate, outer],
            [-length, outer],
        ],
        dtype=np.float64,
    )
    walls = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    wall_names = (
        "left_lower_outer",
        "left_lower_entry",
        "left_bottleneck_floor",
        "left_lower_chamber_step",
        "middle_floor",
        "right_lower_chamber_step",
        "right_bottleneck_floor",
        "right_lower_exit",
        "right_lower_outer",
        "right_end",
        "right_upper_outer",
        "right_upper_exit",
        "right_bottleneck_ceiling",
        "right_upper_chamber_step",
        "middle_ceiling",
        "left_upper_chamber_step",
        "left_bottleneck_ceiling",
        "left_upper_entry",
        "left_upper_outer",
        "left_end",
    )

    # The four goal disks are separated in the staging regions.  Maintaining
    # the initial vertical lanes is a geometric choice, not an assigned order.
    goals = np.array(
        [
            [length - 0.45, -0.38],
            [length - 0.78, 0.38],
            [-length + 0.45, 0.38],
            [-length + 0.78, -0.38],
        ],
        dtype=np.float64,
    )
    return Layout(walls=walls, wall_names=wall_names, goals=goals)


def initial_positions(config, regime: str) -> np.ndarray:
    """Return one of three predeclared, collision-free four-agent starts."""
    if regime not in INITIAL_REGIMES:
        raise ValueError(f"Unknown initial regime {regime!r}; expected one of {INITIAL_REGIMES}")
    length = float(config.world_half_length)

    if regime == "clearly_asymmetric":
        # A1 reaches the first bottleneck noticeably before the other agents.
        positions = [
            [-length + 1.42, -0.33],
            [-length + 0.64, 0.35],
            [length - 0.96, 0.34],
            [length - 0.40, -0.36],
        ]
    elif regime == "weakly_asymmetric":
        positions = [
            [-length + 1.14, -0.33],
            [-length + 0.70, 0.34],
            [length - 1.04, 0.31],
            [length - 0.61, -0.35],
        ]
    else:
        # Deliberately close to, but not exactly on, a reflection symmetry.
        positions = [
            [-length + 1.00, -0.33],
            [-length + 0.55, 0.33],
            [length - 1.02, 0.325],
            [length - 0.54, -0.335],
        ]
    return np.asarray(positions, dtype=np.float64)

