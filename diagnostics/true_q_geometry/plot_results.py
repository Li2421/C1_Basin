"""Create lightweight static figures from finalized JSON diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    qmap = json.loads((HERE / "full_horizon_q_map.json").read_text())
    direction = json.loads((HERE / "directional_geometry.json").read_text())
    features = json.loads((HERE / "feature_analysis.json").read_text())
    projection = json.loads((HERE / "projection_role.json").read_text())
    temporal = json.loads((HERE / "temporal_control_window.json").read_text())
    escape = json.loads((HERE / "raw/escape_delay_summary.json").read_text())

    # Q_D versus the goal coordinate at both predeclared delta scales.
    goal_names = ["goal_m2", "goal_m1", "p0", "goal_p1", "goal_p2"]
    x = np.array([-.25, -.125, 0, .125, .25])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for state in qmap["states"]:
        if not state["state_id"].startswith("D"):
            continue
        lookup = {p["probe_name"]: p["Q_D"] for p in state["policies"]}
        ax.plot(x, [lookup[n] for n in goal_names], marker="o", label=state["state_id"])
    ax.set(xlabel=r"$\phi_g$", ylabel=r"full-horizon $Q_D$", ylim=(-.05, 1.05),
           title="True deadlock risk: goal-coordinate slices")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.25)
    save(fig, "q_d_vs_phi_coordinates.png")

    # Fresh directional validation.
    states = direction["fresh_direction_validation"]["states"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xx = np.arange(len(states))
    width = .24
    for j, (name, label) in enumerate((("phi_minus", r"$\phi_-$"), ("phi_0", r"$\phi_0$"), ("phi_plus", r"$\phi_+$"))):
        ax.bar(xx + (j - 1) * width, [s[name]["Q_D"] for s in states], width, label=label)
    ax.set_xticks(xx, [s["state_id"] for s in states])
    ax.set(ylabel=r"fresh full-horizon $Q_D$", ylim=(0, 1.08), title="True-Q finite-difference direction validation")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    save(fig, "q_d_directional_slices.png")

    # Sparse local basin map averaged over states.
    points = {}
    for state in qmap["states"]:
        for policy in state["policies"]:
            phi = tuple(policy["phi"])
            points.setdefault(phi, []).append((policy["Q_D"], policy["Q_success"], policy["Q_timeout"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharex=True, sharey=True)
    for ax, index, title in ((axes[0], 0, "mean $Q_D$"), (axes[1], 2, "mean $Q_{timeout}$")):
        for phi, vals in points.items():
            value = np.mean([v[index] for v in vals])
            ax.scatter(phi[0], phi[2], c=[value], vmin=0, vmax=1, cmap="viridis", s=230, edgecolor="black")
            ax.text(phi[0], phi[2], f"{value:.2f}", ha="center", va="center", color="white" if value > .48 else "black", fontsize=8)
        ax.set(xlabel=r"$\phi_g$", ylabel=r"$\phi_r$", title=title)
        ax.grid(alpha=.2)
    save(fig, "success_deadlock_basin.png")

    # Feature versus Q_D.
    rows = features["rows_for_plotting"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    axes[0].scatter([r["progress_100"] for r in rows], [r["Q_D"] for r in rows], alpha=.7, s=24)
    axes[0].set(xlabel="100-step task progress", ylabel=r"$Q_D$", title="Progress is not sufficient")
    axes[1].scatter([r["state_displacement_100"] for r in rows], [r["Q_D"] for r in rows], alpha=.7, s=24)
    axes[1].set(xlabel="100-step state displacement", ylabel=r"$Q_D$", title="Strong association, overlapping basins")
    for ax in axes:
        ax.grid(alpha=.25)
    save(fig, "feature_vs_q_d.png")

    # Escape/delay summary.
    counts = escape["classification_counts_for_baseline_deadlock_pairs"]
    labels = ["TRUE_ESCAPE", "DELAYED_DEADLOCK", "TIMEOUT", "NO_EFFECT"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(labels, [counts.get(x, 0) for x in labels], color=["#2a9d8f", "#e9c46a", "#e76f51", "#888888"])
    ax.set(ylabel="same-state/seed policy comparisons", title="Deadlock-baseline outcome changes")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=.25)
    save(fig, "escape_vs_delayed_deadlock.png")

    # Time-to-deadlock geometry.
    rows = sorted(temporal["deadlock_source_states"], key=lambda r: r["offset_seconds"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    offsets = [r["offset_seconds"] for r in rows]
    axes[0].plot(offsets, [r["baseline_Q_D"] for r in rows], "o-", label="baseline")
    axes[0].plot(offsets, [r["minimum_Q_D"] for r in rows], "o-", label="minimum probe")
    axes[0].set(xlabel="source time to deadlock (s)", ylabel=r"$Q_D$", title="Policy sensitivity vs lead time", ylim=(-.05, 1.05))
    axes[0].legend()
    axes[1].plot(offsets, [r["g_TRUE_norm"] for r in rows], "o-", color="#7b2cbf")
    axes[1].set(xlabel="source time to deadlock (s)", ylabel=r"$\|g_{TRUE}\|$", title="Finite-difference jump strength")
    for ax in axes:
        ax.grid(alpha=.25)
    save(fig, "q_d_geometry_vs_time_to_deadlock.png")

    # Projection role.
    pairs = projection["policy_pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    dq = [abs(r["Q_D_b_minus_a"]) for r in pairs]
    axes[0].scatter([r["active_set_disagreement_fraction"] for r in pairs], dq, s=32, alpha=.75)
    axes[0].set(xlabel="active-set disagreement fraction", ylabel=r"$|\Delta Q_D|$", title="Switching accompanies risk changes")
    axes[1].scatter([r["pre_projection_g_rms"] for r in pairs], [r["executed_action_rms"] for r in pairs], s=32, alpha=.75)
    axes[1].set(xlabel="pre-projection residual RMS", ylabel="executed-action RMS", title="Differences survive hard projection")
    for ax in axes:
        ax.grid(alpha=.25)
    save(fig, "projection_active_set_vs_q_d_change.png")


if __name__ == "__main__":
    main()
