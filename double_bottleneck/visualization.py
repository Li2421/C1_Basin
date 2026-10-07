"""Dependency-free trajectory visualization for Double-Bottleneck.

The engineering smoke test intentionally writes SVG rather than depending on
matplotlib.  This module observes saved rollout data only; it has no controller
or environment side effects.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

from .scenario import AGENT_NAMES


_COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00")


def _points(values: np.ndarray, project) -> str:
    return " ".join(f"{project(x, y)[0]:.2f},{project(x, y)[1]:.2f}" for x, y in values)


def save_trajectory_svg(
    env,
    positions: np.ndarray,
    eta: Iterable[float],
    terminal_reason: str,
    output_path: str | Path,
    *,
    title: str = "Double-Bottleneck rollout",
) -> Path:
    """Render walls, starts, goals, and four trajectories to a standalone SVG.

    Args:
        env: A configured :class:`DoubleBottleneckEnv`.  It is read only.
        positions: Array with shape ``[T, 4, 2]``, including the initial state.
        eta: Fixed three-dimensional diagnostic rollout parameter.
        terminal_reason: Terminal label shown in the figure.
        output_path: Destination ``.svg`` path.
    """
    trajectory = np.asarray(positions, dtype=np.float64)
    eta_array = np.asarray(tuple(eta), dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 2) or trajectory.shape[0] < 1:
        raise ValueError("positions must have shape [T,4,2] with T >= 1")
    if eta_array.shape != (3,) or not np.isfinite(eta_array).all():
        raise ValueError("eta must contain three finite values")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains NaN/Inf")

    width, height, margin = 1000.0, 360.0, 42.0
    xlim = float(env.config.world_half_length) + 0.18
    ylim = max(float(env.config.outer_half_height), float(env.config.chamber_half_height)) + 0.18

    def project(x, y):
        sx = margin + (float(x) + xlim) / (2 * xlim) * (width - 2 * margin)
        sy = height - margin - (float(y) + ylim) / (2 * ylim) * (height - 2 * margin)
        return sx, sy

    def rect(x0, y0, x1, y1, color):
        sx0, sy1 = project(x0, y0)
        sx1, sy0 = project(x1, y1)
        return (
            f'<rect x="{sx0:.2f}" y="{sy0:.2f}" width="{sx1-sx0:.2f}" '
            f'height="{sy1-sy0:.2f}" fill="{color}" stroke="none"/>'
        )

    cfg = env.config
    inner = float(cfg.chamber_half_length)
    outer_gate = inner + float(cfg.bottleneck_length)
    length = float(cfg.world_half_length)
    narrow = float(cfg.bottleneck_width) / 2
    chamber = float(cfg.chamber_half_height)
    outer = float(cfg.outer_half_height)

    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        # Light fills make both narrow passages and the wider chamber explicit.
        rect(-length, -outer, -outer_gate, outer, "#F2F2F2"),
        rect(-outer_gate, -narrow, -inner, narrow, "#DCEAF7"),
        rect(-inner, -chamber, inner, chamber, "#EEF6DD"),
        rect(inner, -narrow, outer_gate, narrow, "#DCEAF7"),
        rect(outer_gate, -outer, length, outer, "#F2F2F2"),
    ]

    for segment in np.asarray(env.walls):
        (x1, y1), (x2, y2) = segment
        sx1, sy1 = project(x1, y1)
        sx2, sy2 = project(x2, y2)
        body.append(
            f'<line x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}" '
            'stroke="#222" stroke-width="3"/>'
        )

    radius_px = max(4.0, float(cfg.agent_radius) / (2 * ylim) * (height - 2 * margin))
    for index, (name, color) in enumerate(zip(AGENT_NAMES, _COLORS)):
        path = trajectory[:, index]
        body.append(
            f'<polyline points="{_points(path, project)}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        start_x, start_y = project(*path[0])
        end_x, end_y = project(*path[-1])
        goal_x, goal_y = project(*np.asarray(env.goals)[index])
        body.extend(
            (
                f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="{radius_px:.2f}" '
                f'fill="white" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="4" fill="{color}"/>',
                f'<circle cx="{goal_x:.2f}" cy="{goal_y:.2f}" r="{radius_px:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"/>',
                f'<text x="{start_x+6:.2f}" y="{start_y-6:.2f}" font-size="12" '
                f'fill="{color}">{escape(name)}</text>',
            )
        )

    eta_text = ", ".join(f"{value:g}" for value in eta_array)
    body.extend(
        (
            f'<text x="{margin:.1f}" y="22" font-family="sans-serif" font-size="16" '
            f'font-weight="bold">{escape(title)}</text>',
            f'<text x="{margin:.1f}" y="{height-10:.1f}" font-family="monospace" font-size="13">'
            f'fixed eta=({escape(eta_text)}) | terminal={escape(str(terminal_reason))}</text>',
        )
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">' + "".join(body) + "</svg>\n"
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


__all__ = ("save_trajectory_svg",)
