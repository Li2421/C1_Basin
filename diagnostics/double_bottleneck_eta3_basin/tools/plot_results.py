#!/usr/bin/env python3
"""Create static scientific figures from frozen eta-basin result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
FIG = STUDY / "figures"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    episodes = json.loads((STUDY / "per_episode_basin.json").read_text())["episodes"]
    coverage = json.loads((STUDY / "per_eta_cross_state_coverage.json").read_text())["etas"]
    rows = json.loads((STUDY / "all_rollout_outcomes.json").read_text())["rollouts"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].hist([row["stage_a_basin_fraction"] for row in episodes], bins=np.linspace(0, 1, 12), color="#3478b8", edgecolor="white")
    axes[0].set(xlabel="Stage-A basin fraction", ylabel="Timeout episodes", title="Coarse success-basin volume")
    type_counts = {}
    for row in episodes:
        key = row["basin_type"].split(" — ")[0]
        type_counts[key] = type_counts.get(key, 0) + 1
    axes[1].bar(list(type_counts), list(type_counts.values()), color="#4c956c")
    axes[1].set(xlabel="Registered basin class", ylabel="Episodes", title="Basin existence / geometry")
    save(fig, "basin_fraction_and_types.png")

    rescue = np.asarray([row["timeout_targets_rescued"] for row in coverage])
    preserve = np.asarray([row["baseline_success_controls_preserved"] for row in coverage])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].hist(rescue, bins=np.arange(rescue.min(), rescue.max() + 2) - 0.5, color="#f28e2b", edgecolor="white")
    axes[0].set(xlabel="Timeout episodes rescued / 61", ylabel="Shared eta count", title="Cross-state eta coverage")
    scatter = axes[1].scatter(rescue, preserve, c=[row["eta"][1] for row in coverage], cmap="coolwarm", s=38)
    axes[1].set(xlabel="Timeouts rescued / 61", ylabel="Controls preserved / 24", title="Rescue–preservation tradeoff")
    fig.colorbar(scatter, ax=axes[1], label=r"$\eta_2$ (safe-action feedback)")
    save(fig, "cross_state_coverage.png")

    representatives = sorted(
        [row for row in episodes if row["basin_exists"]],
        key=lambda row: (-row["stage_a_basin_fraction"], row["episode_id"]),
    )
    if representatives:
        choices = []
        for row in representatives:
            if row["basin_type"] not in {item["basin_type"] for item in choices}:
                choices.append(row)
            if len(choices) == 4:
                break
        while len(choices) < min(4, len(representatives)):
            candidate = representatives[len(choices)]
            if candidate not in choices:
                choices.append(candidate)
            else:
                break
        fig = plt.figure(figsize=(4.4 * len(choices), 4.1))
        for index, item in enumerate(choices, 1):
            eid = item["episode_id"]
            selected = [
                row for row in rows
                if f"{row['set']}|{row['rollout_id']:03d}" == eid and row["eta_id"] != "ZERO"
            ]
            ax = fig.add_subplot(1, len(choices), index, projection="3d")
            failed = np.asarray([row["eta"] for row in selected if not row["success"]])
            passed = np.asarray([row["eta"] for row in selected if row["success"]])
            if len(failed):
                ax.scatter(failed[:, 0], failed[:, 1], failed[:, 2], c="#bdbdbd", s=13, alpha=0.55, label="fail")
            if len(passed):
                ax.scatter(passed[:, 0], passed[:, 1], passed[:, 2], c="#2ca25f", s=26, label="success")
            ax.set(xlabel=r"$\eta_1$", ylabel=r"$\eta_2$", zlabel=r"$\eta_3$", title=f"{eid}\n{item['basin_type'].split(' — ')[0]}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2)
        save(fig, "representative_eta3_scatter.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    stage_rows = [row for row in rows if row["eta_id"].startswith("A") and row["population"] == "safe_timeout_target"]
    for axis, pair, labels in zip(axes, ((0, 1), (0, 2), (1, 2)), ((r"$\eta_1$", r"$\eta_2$"), (r"$\eta_1$", r"$\eta_3$"), (r"$\eta_2$", r"$\eta_3$")), strict=True):
        counts = {}
        for row in stage_rows:
            counts.setdefault(row["eta_id"], {"eta": row["eta"], "success": 0})
            counts[row["eta_id"]]["success"] += int(row["success"])
        points = np.asarray([value["eta"] for value in counts.values()])
        colors = np.asarray([value["success"] for value in counts.values()])
        image = axis.scatter(points[:, pair[0]], points[:, pair[1]], c=colors, cmap="viridis", s=45)
        axis.set(xlabel=labels[0], ylabel=labels[1])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Timeout episodes rescued / 61", shrink=0.85)
    fig.subplots_adjust(wspace=0.3, right=0.88)
    fig.savefig(FIG / "eta_2d_projections.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
