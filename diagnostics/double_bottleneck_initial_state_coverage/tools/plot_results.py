#!/usr/bin/env python3
"""Create static figures from the frozen initial-state coverage results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
SCALES = ("S-Small", "S-Medium", "S-Large")
COLORS = ("#4C78A8", "#F58518", "#54A24B")


def _read(path):
    return json.loads(path.read_text())


def _save(fig, name):
    output = STUDY / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / name, bbox_inches="tight")
    plt.close(fig)


def coverage_scaling(posthoc):
    rows = [posthoc["coverage_scaling"][name] for name in SCALES]
    x = np.asarray([row["independent_initial_states"] for row in rows])
    metrics = (
        ("timestep0_mean_distance", "Mean timestep-0 support distance"),
        ("whole_rollout_ood_fraction", "Whole-rollout OOD fraction"),
        ("success_rate", "Untouched-test success rate"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for axis, (key, label) in zip(axes, metrics):
        y = np.asarray([row[key] for row in rows])
        axis.plot(x, y, marker="o", color="#4C78A8", linewidth=2)
        for x_value, y_value in zip(x, y):
            axis.annotate(f"{y_value:.3f}", (x_value, y_value), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=8)
        axis.set_xlabel("Independent training initial states")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1].set_ylim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    fig.suptitle("Broad initial-state coverage scaling (frozen untouched test)")
    fig.tight_layout()
    _save(fig, "coverage_scaling.svg")


def closed_loop_outcomes(comparison):
    names = ("D0", "D1", "U-High", "D2-T", *SCALES)
    results = comparison["sets"]["untouched_test"]
    categories = (
        ("success", "Success", "#54A24B"),
        ("collision", "Collision terminal", "#E45756"),
        ("timeout", "Timeout", "#9D755D"),
    )
    fig, axis = plt.subplots(figsize=(9.5, 4.0))
    bottom = np.zeros(len(names))
    for key, label, color in categories:
        values = np.asarray(
            [
                sum(
                    row["termination"] == key
                    for row in results[name]["rollouts"]
                )
                for name in names
            ]
        )
        axis.bar(names, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.axvline(3.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axis.text(1.5, 99, "Historical references", ha="center", va="bottom", fontsize=8)
    axis.text(5.0, 99, "Coverage scaling", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("Rollouts (24 states × 4 seeds)")
    axis.set_ylim(0, 108)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "closed_loop_outcomes.svg")


def kstep(comparison):
    horizons = (1, 5, 10, 25, 50, 100)
    results = comparison["sets"]["untouched_test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for name, color in zip(SCALES, COLORS):
        summary = results[name]["k_step"]["summary"]
        axes[0].plot(
            horizons,
            [summary[str(value)]["position_rmse"] for value in horizons],
            marker="o",
            label=name,
            color=color,
        )
        axes[1].plot(
            horizons,
            [summary[str(value)]["collision_rate"] for value in horizons],
            marker="o",
            label=name,
            color=color,
        )
    axes[0].set_ylabel("Joint-position RMSE (m)")
    axes[1].set_ylabel("Collision rate")
    axes[1].set_ylim(-0.005, 0.05)
    for axis in axes:
        axis.set_xlabel("Closed-loop continuation horizon K")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Matched expert-state K-step continuations")
    fig.tight_layout()
    _save(fig, "kstep_divergence.svg")


def main():
    posthoc = _read(STUDY / "posthoc_analysis.json")
    comparison = _read(STUDY / "evaluation/comparison.json")
    coverage_scaling(posthoc)
    closed_loop_outcomes(comparison)
    kstep(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
