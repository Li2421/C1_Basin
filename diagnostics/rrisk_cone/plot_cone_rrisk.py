"""Static figures for the frozen projection-cone pilot (no simulation)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data_v1"
FIGURES = HERE / "figures"


def load(name):
    return json.loads((DATA / name).read_text())


def plane_basis(vector, projection, generators):
    candidates = [np.asarray(projection), np.asarray(vector)]
    candidates.extend(np.asarray(generators))
    basis = []
    for candidate in candidates:
        residual = candidate.copy()
        for unit in basis:
            residual -= np.dot(residual, unit) * unit
        norm = np.linalg.norm(residual)
        if norm > 1e-10:
            basis.append(residual / norm)
        if len(basis) == 2:
            break
    for axis in np.eye(len(vector)):
        if len(basis) == 2:
            break
        residual = axis.copy()
        for unit in basis:
            residual -= np.dot(residual, unit) * unit
        norm = np.linalg.norm(residual)
        if norm > 1e-10:
            basis.append(residual / norm)
    return np.asarray(basis)


def representative_geometry(q_rows):
    representatives = [
        max(q_rows, key=lambda row: row["R_cone_pi2"]),
        min(q_rows, key=lambda row: row["R_cone_pi2"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, row in zip(axes, representatives):
        with np.load(row["trace"], allow_pickle=False) as trace:
            A = trace["A"][100]
        active = row["pi2"]["kkt"]["active_linear"]
        generators = -A[active]
        vector = np.asarray(row["candidate"])
        executed = np.asarray(row["executed"])
        projection = np.asarray(row["pi2"]["cone"]["projection"])
        basis = plane_basis(vector, projection, generators)
        rays = generators @ basis.T if len(generators) else np.empty((0, 2))
        scale = max(np.linalg.norm(vector), np.linalg.norm(executed),
                    np.linalg.norm(rays, axis=1).max(initial=0.), .05)
        for index, ray in enumerate(rays):
            norm = np.linalg.norm(ray)
            if norm:
                endpoint = .85 * scale * ray / norm
                ax.arrow(0, 0, *endpoint, width=.003 * scale,
                         head_width=.045 * scale, color="0.55", alpha=.75,
                         length_includes_head=True)
                ax.text(*(1.08 * endpoint), f"g{active[index]}", fontsize=8,
                        color="0.35")
        items = [(vector, "drive $y_2$", "tab:red"),
                 (executed, "executed $p_2$", "tab:blue"),
                 (projection, "$\\Pi_C(y_2)$", "tab:green")]
        for item, label, color in items:
            point = item @ basis.T
            ax.arrow(0, 0, *point, width=.005 * scale,
                     head_width=.06 * scale, color=color,
                     length_includes_head=True, label=label)
        ax.axhline(0, color="0.85", lw=.8)
        ax.axvline(0, color="0.85", lw=.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.15 * scale, 1.15 * scale)
        ax.set_ylim(-1.15 * scale, 1.15 * scale)
        ax.set_xlabel("orthogonal-view coordinate 1 (m/s)")
        ax.set_ylabel("orthogonal-view coordinate 2 (m/s)")
        ax.set_title(
            f"rid {row['rid']} / {row['arm']}\n"
            f"rank {row['pi2']['cone']['rank']}, R={row['R_cone_pi2']:.4f}")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Representative joint-R$^4$ cone rays (2-D orthogonal view only)")
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(FIGURES / "representative_cone_geometry.png", dpi=180)
    plt.close(fig)


def prediction_figure(p_rows):
    rng = np.random.default_rng(2026092103)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for deadlock, color, label in [(False, "tab:blue", "future non-deadlock"),
                                   (True, "tab:red", "future deadlock")]:
        rows = [row for row in p_rows if row["deadlock"] == deadlock]
        x = np.asarray([row["early_step"] * .05 for row in rows])
        x += rng.uniform(-.08, .08, len(x))
        y = [row["R_cone_pi2"] for row in rows]
        ax.scatter(x, y, s=27, alpha=.78, color=color, label=label)
    ax.axhline(0, color="0.25", lw=.8, ls="--")
    ax.set_xticks([0, 2.5, 5, 7.5])
    ax.set_xlabel("conditioning time (s)")
    ax.set_ylabel("primary $R_{cone}=-M_{cone}$ (m/s)")
    ax.set_title("TEST P: instantaneous cone risk vs frozen future event")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_p_cone_vs_outcome.png", dpi=180)
    plt.close(fig)


def q_figures(q_rows, pairs):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {0: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for rid in sorted(colors):
        rows = [row for row in q_rows if row["rid"] == rid]
        ax.scatter([row["R_cone_pi2"] for row in rows],
                   [row["Q_D"] for row in rows], s=45, color=colors[rid],
                   label=f"rid {rid}")
    ax.set_ylim(.94, 1.06)
    ax.set_xlabel("primary $R_{cone}$ (m/s)")
    ax.set_ylabel("empirical continuation $Q_D$ (8 seeds)")
    ax.set_title("TEST Q: score variation without true-risk variation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_q_rcone_vs_qd.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for rid in sorted(colors):
        rows = [row for row in pairs if row["rid"] == rid]
        ax.scatter([row["delta_R"] for row in rows],
                   [row["delta_Q"] for row in rows], s=25,
                   alpha=.75, color=colors[rid], label=f"rid {rid}")
    ax.axhline(0, color="0.2", lw=.8)
    ax.axvline(0, color="0.7", lw=.8)
    ax.set_xlabel("pairwise $\\Delta R_{cone}$ (m/s)")
    ax.set_ylabel("pairwise $\\Delta Q_D$")
    ax.set_title("108 action pairs: every true-risk difference is zero")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_q_action_pair_ranking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    switch = np.asarray([row["active_set_switch"] for row in pairs])
    pre = np.asarray([row["pre_projection_difference"] for row in pairs])
    post = np.asarray([row["executed_action_difference"] for row in pairs])
    for value, color, label in [(False, "tab:blue", "same active set"),
                                (True, "tab:red", "active-set switch")]:
        mask = switch == value
        ax.scatter(pre[mask], post[mask], s=30, alpha=.72, color=color,
                   label=f"{label} ({mask.sum()})")
    limit = max(pre.max(initial=0), post.max(initial=0)) * 1.05
    ax.plot([0, limit], [0, limit], color="0.45", ls="--", lw=1,
            label="no attenuation")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("pre-projection action difference (m/s)")
    ax.set_ylabel("executed-action difference (m/s)")
    ax.set_title("Projection check: no mass aliasing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "active_set_changes_and_projection.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    p_rows = load("raw/p_state_scores.json")
    q_rows = load("raw/q_action_scores.json")
    pairs = load("raw/q_pair_scores.json")
    representative_geometry(q_rows)
    prediction_figure(p_rows)
    q_figures(q_rows, pairs)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
