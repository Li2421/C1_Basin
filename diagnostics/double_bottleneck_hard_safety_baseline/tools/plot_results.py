#!/usr/bin/env python3
"""Static scientific figures for the frozen hard-safety comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from double_bottleneck.environment import DoubleBottleneckEnv


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
FIGURES = STUDY / "figures"
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def load(name):
    return json.loads((STUDY / name).read_text())


def outcomes_and_conversion():
    no = load("no_safety_summary.json")
    safe = load("hard_safety_summary.json")
    conversion = load("matched_episode_conversion_matrix.json")["combined"]["counts"]
    labels = ("Success", "Wall collision", "Agent collision", "Strict deadlock", "Timeout")
    keys = ("success", "wall_collision", "agent_collision", "runtime_strict_deadlock", "timeout")
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, [no["combined"][key] for key in keys], width, label="No safety", color="#666666")
    axes[0].bar(x + width / 2, [safe["combined"][key] for key in keys], width, label="Hard safety", color="#0072B2")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Episodes (combined n=192)")
    axes[0].set_title("Full-task terminal outcomes")
    axes[0].legend(frameon=False)

    raw = ("success", "wall_collision", "agent_collision", "timeout")
    projected = ("success", "timeout")
    matrix = np.zeros((len(raw), len(projected)), dtype=int)
    for i, source in enumerate(raw):
        for j, target in enumerate(projected):
            matrix[i, j] = conversion.get(f"{source} -> {target}", 0)
    image = axes[1].imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(j, i, str(value), ha="center", va="center", color="black")
    axes[1].set_xticks(range(len(projected)), ("Success", "Timeout"))
    axes[1].set_yticks(range(len(raw)), ("Success", "Wall collision", "Agent collision", "Timeout"))
    axes[1].set_xlabel("Hard-safety outcome")
    axes[1].set_ylabel("No-safety outcome")
    axes[1].set_title("Matched outcome conversion")
    fig.colorbar(image, ax=axes[1], label="Episodes", fraction=0.045)
    fig.savefig(FIGURES / "outcomes_and_conversion.svg")
    plt.close(fig)


def representative_trajectories():
    representatives = load("representative_trajectory_analysis.json")["representatives"]
    desired = (("success", "timeout"), ("agent_collision", "timeout"), ("timeout", "success"))
    selected = []
    for source, target in desired:
        selected.append(next(row for row in representatives if row["no_safety_outcome"] == source and row["safety_outcome"] == target))
    env = DoubleBottleneckEnv()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.7), sharex=True, sharey=True, constrained_layout=True)
    for axis, row in zip(axes, selected, strict=True):
        set_name, rollout_id = row["set"], row["rollout_id"]
        arrays = {}
        for controller in ("no_safety", "hard_safety"):
            path = STUDY / "trajectories" / controller / set_name / f"rollout_{rollout_id:03d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays[controller] = archive["positions"].astype(np.float64)
        for segment in env.walls:
            axis.plot(segment[:, 0], segment[:, 1], color="black", linewidth=2.0, solid_capstyle="round")
        for agent in range(4):
            axis.plot(arrays["no_safety"][:, agent, 0], arrays["no_safety"][:, agent, 1], color=COLORS[agent], linestyle="--", linewidth=1.1, alpha=0.7)
            axis.plot(arrays["hard_safety"][:, agent, 0], arrays["hard_safety"][:, agent, 1], color=COLORS[agent], linewidth=1.5)
            axis.scatter(arrays["hard_safety"][-1, agent, 0], arrays["hard_safety"][-1, agent, 1], color=COLORS[agent], s=18, zorder=4)
        axis.set_title(f"{row['no_safety_outcome'].replace('_', ' ')} → {row['safety_outcome']}\n{set_name.replace('_', ' ')}, id {rollout_id}", fontsize=9)
        axis.set_xlim(-4.1, 4.1)
        axis.set_ylim(-0.95, 0.95)
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.text(0.5, -0.015, "Dashed: no safety; solid: hard safety; endpoint markers show hard-safety terminal state", ha="center", fontsize=9)
    fig.savefig(FIGURES / "representative_trajectories.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=False)
    outcomes_and_conversion()
    representative_trajectories()


if __name__ == "__main__":
    main()
