"""Static scientific figures for the isolated eta stress-test results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .controller import RolloutTrace
from .environment import CoupledDualIntersectionEnv, StressConfig
from .search import EtaBounds, SearchSamples


COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#7F7F7F", "#000000")


def _save(figure, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_trajectory(trace: RolloutTrace, output: str | Path, config: StressConfig | None = None, title: str | None = None) -> Path:
    env = CoupledDualIntersectionEnv(config)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for wall in env.walls:
        axis.plot(wall[:, 0], wall[:, 1], color="0.15", linewidth=1.5, zorder=1)
    for index, color in enumerate(COLORS):
        path = trace.positions[:, index]
        axis.plot(path[:, 0], path[:, 1], color=color, linewidth=1.4, label=f"A{index + 1}", zorder=2)
        axis.scatter(*path[0], s=20, color=color, marker="o", zorder=3)
        axis.scatter(*env.goals[index], s=32, facecolors="none", edgecolors=color, marker="o", zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title or f"{trace.variant}: {trace.termination} / {trace.failure_mode}")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(figure, output)


def plot_oracle_trajectory(positions: np.ndarray, termination: str, mode: str, output: str | Path, config: StressConfig | None = None) -> Path:
    """Render the explicitly scheduled oracle separately from eta traces."""
    env = CoupledDualIntersectionEnv(config)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for wall in env.walls:
        axis.plot(wall[:, 0], wall[:, 1], color="0.15", linewidth=1.5, zorder=1)
    for index, color in enumerate(COLORS):
        axis.plot(positions[:, index, 0], positions[:, index, 1], color=color, linewidth=1.4, label=f"A{index + 1}")
        axis.scatter(*positions[0, index], s=20, color=color, zorder=3)
        axis.scatter(*env.goals[index], s=32, facecolors="none", edgecolors=color, zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)")
    axis.set_title(f"centralized oracle {mode}: {termination}")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(figure, output)


def plot_eta_scatter(samples: SearchSamples, bounds: EtaBounds, output: str | Path, title: str) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    pairs = ((0, 1), (0, 2), (1, 2))
    names = ("eta_goal", "eta_safe", "eta_relative")
    for axis, (left, right) in zip(axes, pairs):
        failed = ~samples.success
        axis.scatter(samples.points[failed, left], samples.points[failed, right], s=8, color="0.75", alpha=0.45, label="failure")
        axis.scatter(samples.points[samples.success, left], samples.points[samples.success, right], s=13, color="#009E73", alpha=0.85, label="success")
        axis.set_xlim(bounds.lower[left], bounds.upper[left]); axis.set_ylim(bounds.lower[right], bounds.upper[right])
        axis.set_xlabel(names[left]); axis.set_ylabel(names[right])
    axes[0].legend(loc="best")
    figure.suptitle(title)
    return _save(figure, output)


def plot_success_heatmaps(samples: SearchSamples, bounds: EtaBounds, output: str | Path, title: str, bins: int = 30) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    pairs = ((0, 1), (0, 2), (1, 2))
    names = ("eta_goal", "eta_safe", "eta_relative")
    image = None
    for axis, (left, right) in zip(axes, pairs):
        total, xedges, yedges = np.histogram2d(samples.points[:, left], samples.points[:, right], bins=bins, range=((bounds.lower[left], bounds.upper[left]), (bounds.lower[right], bounds.upper[right])))
        good, _, _ = np.histogram2d(samples.points[samples.success, left], samples.points[samples.success, right], bins=(xedges, yedges))
        fraction = np.divide(good, total, out=np.full_like(good, np.nan), where=total > 0)
        image = axis.pcolormesh(xedges, yedges, fraction.T, vmin=0, vmax=1, cmap="viridis", shading="auto")
        axis.set_xlabel(names[left]); axis.set_ylabel(names[right])
    figure.colorbar(image, ax=axes, label="sampled success fraction per 2-D bin")
    figure.suptitle(title)
    return _save(figure, output)


def plot_solver_difficulty(rows: list[dict], output: str | Path, title: str) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    names = ("uniform", "sobol", "cem")
    success_rates, medians = [], []
    for name in names:
        selected = [row for row in rows if row["solver"] == name]
        success = [row["success"] for row in selected]
        evaluations = [row["evaluations_to_first_success"] for row in selected if row["evaluations_to_first_success"] is not None]
        success_rates.append(np.mean(success) if success else np.nan)
        medians.append(np.median(evaluations) if evaluations else np.nan)
    axes[0].bar(names, success_rates, color=("#0072B2", "#D55E00", "#009E73"))
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("probability of success within budget")
    axes[1].bar(names, medians, color=("#0072B2", "#D55E00", "#009E73"))
    axes[1].set_ylabel("median evaluations to first success")
    figure.suptitle(title)
    return _save(figure, output)
