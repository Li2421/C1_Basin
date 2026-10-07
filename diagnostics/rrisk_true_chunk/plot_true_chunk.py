"""Static scientific figures for true persistent-correction risk."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def q_vs_correction(arms, out):
    states = sorted({row["state_id"] for row in arms})
    horizons = sorted({row["horizon_steps"] for row in arms})
    fig, axes = plt.subplots(len(states), len(horizons),
                             figsize=(4.0 * len(horizons), 3.0 * len(states)),
                             sharey=True, squeeze=False)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for row_index, state in enumerate(states):
        for col_index, horizon in enumerate(horizons):
            ax = axes[row_index, col_index]
            rows = [row for row in arms if row["state_id"] == state and
                    row["horizon_steps"] == horizon]
            baseline = next(row for row in rows if row["arm_id"] == "baseline")
            ax.errorbar([0], [baseline["Q_D"]],
                        yerr=[[baseline["Q_D"]-baseline["Q_D_ci95"][0]],
                              [baseline["Q_D_ci95"][1]-baseline["Q_D"]]],
                        fmt="ko", ms=4, capsize=2, label="baseline")
            for axis in range(4):
                selected = sorted(
                    (row for row in rows if row["axis"] == axis),
                    key=lambda row: row["sign"] * row["alpha_norm"])
                x = np.asarray([row["sign"] * row["alpha_norm"]
                                for row in selected])
                y = np.asarray([row["Q_D"] for row in selected])
                lo = y - np.asarray([row["Q_D_ci95"][0] for row in selected])
                hi = np.asarray([row["Q_D_ci95"][1] for row in selected]) - y
                ax.errorbar(x, y, yerr=np.stack((lo, hi)), marker="o", ms=3,
                            lw=.9, capsize=1.5, color=colors[axis],
                            label=f"axis {axis}")
            ax.axhline(baseline["Q_D"], color="0.7", lw=.7, ls="--")
            ax.set_ylim(-.08, 1.08)
            ax.set_title(f"{state} / L={horizon} ({.05*horizon:.2f}s)",
                         fontsize=9)
            ax.set_xlabel("signed coordinate correction norm (m/s)")
            if col_index == 0:
                ax.set_ylabel("true continuation $Q_D$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", frameon=False)
    fig.suptitle("True $Q_D$ versus constant chunk correction")
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.savefig(out / "qd_vs_correction.png", dpi=180)
    plt.close(fig)


def q_vs_horizon(arms, cells, out):
    states = sorted({row["state_id"] for row in cells})
    ncols = min(3, len(states))
    nrows = math.ceil(len(states) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
        sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    for ax, state in zip(flat_axes, states):
        rows = sorted((row for row in cells if row["state_id"] == state),
                      key=lambda row: row["horizon_steps"])
        horizons = [row["horizon_steps"] for row in rows]
        baseline = []
        for horizon in horizons:
            baseline.append(next(
                row["Q_D"] for row in arms if row["state_id"] == state and
                row["horizon_steps"] == horizon and row["arm_id"] == "baseline"))
        ax.plot(horizons, baseline, "ko--", label="baseline")
        ax.plot(horizons, [row["best_Q_D"] for row in rows], "o-",
                color="tab:green", label="best observed")
        ax.plot(horizons, [row["worst_Q_D"] for row in rows], "o-",
                color="tab:red", label="worst observed")
        ax.set_xticks(horizons)
        ax.set_ylim(-.05, 1.05)
        ax.set_xlabel("intervention horizon L (steps)")
        ax.set_title(state, fontsize=9)
    for index, ax in enumerate(flat_axes):
        if index >= len(states):
            ax.set_visible(False)
        elif index % ncols == 0:
            ax.set_ylabel("true continuation $Q_D$")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("True-risk envelope versus persistent-correction horizon")
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(out / "qd_vs_horizon.png", dpi=180)
    plt.close(fig)


def temporal_map(cells, out):
    states = sorted({row["state_id"] for row in cells}, key=lambda state:
                    -next(row["time_before_reference_event_seconds"]
                          for row in cells if row["state_id"] == state))
    horizons = sorted({row["horizon_steps"] for row in cells})
    lookup = {(row["state_id"], row["horizon_steps"]): row for row in cells}
    best = np.full((len(states), len(horizons)), np.nan)
    worst = np.full_like(best, np.nan)
    for i, state in enumerate(states):
        for j, horizon in enumerate(horizons):
            row = lookup.get((state, horizon))
            if row:
                best[i, j], worst[i, j] = row["best_Q_D"], row["worst_Q_D"]
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, max(3.5, .52*len(states)+1.5)),
        sharey=True, constrained_layout=True)
    for ax, values, title in zip(
            axes, (best, worst), ("best observed $Q_D$", "worst observed $Q_D$")):
        image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(horizons)), horizons)
        ax.set_yticks(range(len(states)), states)
        ax.set_xlabel("horizon L (steps)")
        ax.set_title(title)
        for i in range(len(states)):
            for j in range(len(horizons)):
                if np.isfinite(values[i, j]):
                    color = "white" if values[i, j] > .55 else "black"
                    ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                            color=color, fontsize=8)
    axes[0].set_ylabel("predeclared state (earliest to latest)")
    fig.colorbar(image, ax=axes, shrink=.85, pad=.03,
                 label="true continuation $Q_D$")
    fig.suptitle("Temporal controllability map")
    fig.savefig(out / "temporal_controllability_map.png", dpi=180)
    plt.close(fig)


def clipping_plot(arms, out):
    rows = [row for row in arms if row["alpha_norm"] > 0]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {"early": "tab:blue", "emerging": "tab:orange", "late": "tab:red"}
    for stratum, color in colors.items():
        selected = [row for row in rows if row["stratum"] == stratum]
        x = np.asarray([row["horizon_steps"] for row in selected], float)
        x += np.asarray([-.16, -.05, .05, .16])[
            np.asarray([row["axis"] for row in selected], int)]
        ax.scatter(x, [row["median_removed_fraction"] for row in selected],
                   s=18, alpha=.58, color=color, label=stratum)
    ax.axhline(.8, color="0.25", ls="--", lw=1,
               label="predeclared mostly-removed threshold")
    ax.set_xticks(sorted({row["horizon_steps"] for row in rows}))
    ax.set_xlabel("intervention horizon L (steps)")
    ax.set_ylabel("median fraction of requested correction removed by Pi_2")
    ax.set_title("Projection clipping versus persistent-correction horizon")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "projection_clipping_vs_horizon.png", dpi=180)
    plt.close(fig)


def gradient_plot(path, out):
    if not path or not Path(path).exists():
        return
    payload = load(path)
    rows = payload["arms"] if isinstance(payload, dict) else payload
    labels = [row.get("arm", row.get("arm_id")) for row in rows]
    q = np.asarray([row["Q_D"] for row in rows])
    lo = q - np.asarray([row["Q_D_ci95"][0] for row in rows])
    hi = np.asarray([row["Q_D_ci95"][1] for row in rows]) - q
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.errorbar(range(len(rows)), q, yerr=np.stack((lo, hi)), fmt="o",
                capsize=3, color="tab:purple")
    ax.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel("fresh-validation true $Q_D$")
    ax.set_title("Independent true-gradient intervention comparison")
    fig.tight_layout()
    fig.savefig(out / "true_gradient_intervention_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--gradient", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analysis, arms = load(args.analysis), load(args.arms)
    args.out.mkdir(parents=True, exist_ok=True)
    q_vs_correction(arms, args.out)
    q_vs_horizon(arms, analysis["cells"], args.out)
    temporal_map(analysis["cells"], args.out)
    clipping_plot(arms, args.out)
    gradient_plot(args.gradient, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
