"""Standalone SVG visualizations for Ring Exchange trajectories and failures."""
from __future__ import annotations
from html import escape
from pathlib import Path
import numpy as np
from .environment import AGENT_NAMES
from .expert import circulation_signature

_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def save_trajectory_svg(env, positions: np.ndarray, output_path: str | Path, *, title: str = "Ring Exchange rollout",
                        terminal_reason: str = "unknown") -> Path:
    """Draw the wide annulus, central forbidden disk, starts/goals, and paths."""
    traj = np.asarray(positions, dtype=float)
    if traj.ndim != 3 or traj.shape[1:] != (4, 2):
        raise ValueError("positions must have shape [T,4,2]")
    scale, margin, size = env.config.outer_radius + .25, 35., 700.
    def xy(point):
        return (size / 2 + point[0] / (2 * scale) * (size - 2 * margin),
                size / 2 - point[1] / (2 * scale) * (size - 2 * margin))
    factor = (size - 2 * margin) / (2 * scale)
    def path(points):
        return " ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in points)
    centre = xy(np.zeros(2)); c = env.config
    body = ['<rect width="100%" height="100%" fill="white"/>',
            f'<circle cx="{centre[0]:.2f}" cy="{centre[1]:.2f}" r="{c.outer_radius*factor:.2f}" fill="#F4F4F4" stroke="#333" stroke-width="2"/>',
            f'<circle cx="{centre[0]:.2f}" cy="{centre[1]:.2f}" r="{c.obstacle_radius*factor:.2f}" fill="#555"/>']
    radius = c.agent_radius * factor
    for i, color in enumerate(_COLORS):
        sx, sy = xy(traj[0, i]); gx, gy = xy(env.goals[i]); ex, ey = xy(traj[-1, i])
        body += [f'<polyline points="{path(traj[:, i])}" fill="none" stroke="{color}" stroke-width="3"/>',
                 f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{radius:.2f}" fill="white" stroke="{color}" stroke-width="2"/>',
                 f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="{radius:.2f}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"/>',
                 f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="4" fill="{color}"/>',
                 f'<text x="{sx+6:.2f}" y="{sy-6:.2f}" font-size="14" fill="{color}">{AGENT_NAMES[i]}</text>']
    body += [f'<text x="{margin}" y="22" font-family="sans-serif" font-size="16" font-weight="bold">{escape(title)}</text>',
             f'<text x="{margin}" y="{size-12}" font-family="monospace" font-size="13">mode={circulation_signature(traj)} | terminal={escape(str(terminal_reason))}</text>']
    result = Path(output_path); result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">' + ''.join(body) + '</svg>\n', encoding="utf-8")
    return result


def save_failure_points_svg(env, points: np.ndarray, output_path: str | Path, *, title: str = "Ring collision precursors") -> Path:
    """Plot valid collision-predecessor locations for augmentation audits."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    fake = np.repeat(np.asarray(env.positions)[None], 2, axis=0)
    # Reuse geometry renderer with compact point glyphs appended to its SVG.
    output = save_trajectory_svg(env, fake, output_path, title=title, terminal_reason="analysis")
    data = output.read_text(encoding="utf-8").replace("</svg>\n", "")
    scale, margin, size = env.config.outer_radius + .25, 35., 700.
    factor = (size - 2 * margin) / (2 * scale)
    dots = ''.join(f'<circle cx="{size/2+x*factor:.2f}" cy="{size/2-y*factor:.2f}" r="3" fill="#E41A1C" fill-opacity=".65"/>' for x,y in p)
    output.write_text(data + dots + "</svg>\n", encoding="utf-8")
    return output


__all__ = ("save_trajectory_svg", "save_failure_points_svg")
