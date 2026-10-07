"""Plot Ring Exchange four-formulation basins from the cached 27-point study."""
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2] / ".." / "Basin_C1_flow_field_poc_20261004" / "family_study_v1" / "main_analysis.json"
FORMULATIONS = ("TT", "TF", "FT", "FF")
COLORS = {"TT": "#7048a8", "TF": "#e07a16", "FT": "#168a80", "FF": "#d33f49"}


def main():
    data = json.loads(ANALYSIS.resolve().read_text())
    ring_states = [s for s in data["states"] if s["scenario"] == "ring_exchange" and s["complete"]]
    candidates = {s["state_uid"]: [] for s in ring_states}
    for row in data["candidates"]:
        if row["state_uid"] in candidates:
            candidates[row["state_uid"]].append(row)

    scored = []
    for state in ring_states:
        rows = sorted(candidates[state["state_uid"]], key=lambda r: r["eta_index"])
        assert len(rows) == 27 and all(all(r["cells"][f]["B15"] is not None for f in FORMULATIONS) for r in rows)
        patterns = [[bool(r["cells"][f]["B15"]) for f in FORMULATIONS] for r in rows]
        pairwise = sum(sum(patterns[j][a] != patterns[j][b] for j in range(27))
                       for a, b in itertools.combinations(range(4), 2))
        counts = {f: sum(r["cells"][f]["B15"] is True for r in rows) for f in FORMULATIONS}
        # Deterministic, illustration-only ranking: maximize total pairwise Hamming
        # differences, then total robust samples, then state UID ascending.
        scored.append((pairwise, sum(counts.values()), state["state_uid"], rows, counts))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    selected = scored[0]
    _, _, state_uid, rows, counts = selected

    with (HERE / "ring_dense_cache_state_candidates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "state_uid", "pairwise_membership_hamming_total", "TT_robust", "TF_robust", "FT_robust", "FF_robust", "total_robust"])
        for rank, (pairwise, total, uid, _, cc) in enumerate(scored, 1):
            w.writerow([rank, uid, pairwise, *(cc[k] for k in FORMULATIONS), total])

    with (HERE / "ring_dense_cache_selected_membership.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "state_uid", "eta_index", "eta1", "eta2", "eta3", "TT_B15", "TF_B15", "FT_B15", "FF_B15"])
        for r in rows:
            w.writerow([r["scenario"], state_uid, r["eta_index"], *r["eta"], *(int(r["cells"][k]["B15"]) for k in FORMULATIONS)])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "axes.titleweight": "semibold"})
    fig = plt.figure(figsize=(12.4, 9.3), facecolor="white")
    fig.suptitle("Ring Exchange · formulation-specific robust basins", fontsize=18, y=.975, fontweight="semibold")
    fig.text(.5, .936, f"Illustrative state: {state_uid}   ·   shared 27-point 3×3×3 η grid   ·   B15 ≥ 15/16", ha="center", fontsize=11, color="#4c5661")
    eta = np.asarray([r["eta"] for r in rows], dtype=float)
    for p, formulation in enumerate(FORMULATIONS):
        ax = fig.add_subplot(2, 2, p + 1, projection="3d")
        robust = np.asarray([r["cells"][formulation]["B15"] is True for r in rows])
        ax.scatter(*eta[~robust].T, s=34, c="#cbd0d5", alpha=.48, marker="o", edgecolors="white", linewidths=.45, depthshade=False, label="Non-robust")
        ax.scatter(*eta[robust].T, s=145, c=COLORS[formulation], alpha=.97, marker="o", edgecolors="white", linewidths=1.0, depthshade=False, label="B15 robust")
        ax.set_title(f"{formulation}   ·   {int(robust.sum())}/27 robust", loc="left", pad=7, fontsize=14)
        ax.set(xlim=(.46, 1.29), ylim=(-.54, .54), zlim=(-.035, .785),
               xlabel=r"$\eta_1$", ylabel=r"$\eta_2$", zlabel=r"$\eta_3$")
        ax.set_xticks([.5, .875, 1.25]); ax.set_yticks([-.5, 0, .5]); ax.set_zticks([0, .375, .75])
        ax.tick_params(labelsize=9, pad=1)
        ax.view_init(elev=23, azim=-56)
        ax.set_box_aspect((1.1, 1, .85))
        ax.xaxis.pane.set_facecolor((.96, .97, .98, .65))
        ax.yaxis.pane.set_facecolor((.96, .97, .98, .65))
        ax.zaxis.pane.set_facecolor((.96, .97, .98, .65))
        ax.grid(True, alpha=.18)
    legend = [Line2D([], [], marker="o", linestyle="", markersize=6, color="#cbd0d5", alpha=.8, label="Non-robust η sample"),
              Line2D([], [], marker="o", linestyle="", markersize=10, color="#555d66", label="B15 robust η sample")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .06), fontsize=11)
    fig.text(.5, .025, "Exact cached η coordinates; empirical samples only. No interpolated membership or hull boundary is shown.", ha="center", fontsize=9.5, color="#4c5661")
    fig.subplots_adjust(left=.025, right=.98, top=.90, bottom=.12, wspace=.0, hspace=.09)
    fig.savefig(HERE / "our_formulation_basin_ring_exchange.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "our_formulation_basin_ring_exchange.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Three orthogonal 2D projections of the exact same 27 empirical points.
    projection_specs = ((0, 1, "eta1_eta2"), (0, 2, "eta1_eta3"), (1, 2, "eta2_eta3"))
    for xdim, ydim, suffix in projection_specs:
        fig, axs = plt.subplots(2, 2, figsize=(10.6, 8.2), facecolor="white", constrained_layout=True)
        fig.suptitle(f"Ring Exchange · {suffix.replace('_', '–')} basin projection\n{state_uid} · shared 27-point grid · B15 ≥ 15/16",
                     fontsize=15, fontweight="semibold")
        for ax, formulation in zip(axs.flat, FORMULATIONS):
            robust = np.asarray([r["cells"][formulation]["B15"] is True for r in rows])
            ax.scatter(eta[~robust, xdim], eta[~robust, ydim], s=45, c="#cbd0d5", alpha=.52,
                       marker="o", edgecolors="white", linewidths=.5, zorder=1)
            ax.scatter(eta[robust, xdim], eta[robust, ydim], s=135, c=COLORS[formulation], alpha=.98,
                       marker="o", edgecolors="white", linewidths=1.0, zorder=2)
            ax.set_title(f"{formulation} · {int(robust.sum())}/27 robust", loc="left", fontsize=13)
            ax.set_xlabel(rf"$\eta_{xdim + 1}$", fontsize=12)
            ax.set_ylabel(rf"$\eta_{ydim + 1}$", fontsize=12)
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
            lims = ((.46, 1.29), (-.54, .54), (-.04, .79))
            ax.set_xlim(lims[xdim])
            ax.set_ylim(lims[ydim])
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.015), fontsize=10)
        fig.savefig(HERE / f"our_formulation_basin_ring_exchange_projection_{suffix}.png", dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(HERE / f"our_formulation_basin_ring_exchange_projection_{suffix}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    (HERE / "ring_dense_cache_figure_caption.txt").write_text(
        "Ring Exchange formulation-specific empirical basins for illustrative state " + state_uid + ". "
        "The state was selected from the four complete Ring states by a deterministic ranking: total pairwise Hamming distance across TT/TF/FT/FF B15 membership over the same cached 27-point 3×3×3 η grid, then total robust count, then ascending state ID. "
        "Robust membership uses the established B15 criterion (at least 15 successes among 16 fixed continuation seeds). "
        "Gray points are sampled η that are non-robust for the panel formulation; colored points are robust η. "
        "This is an illustrative visualization, not a population-level statistical claim. No rollouts or training were performed for this figure; all values come from the existing family_study_v1 cache.\n"
    )
    print(json.dumps({"state": state_uid, "counts": counts, "pairwise_hamming": selected[0], "n_eta": len(rows),
                      "png": str(HERE / "our_formulation_basin_ring_exchange.png"),
                      "projections": [f"our_formulation_basin_ring_exchange_projection_{suffix}.png" for _, _, suffix in projection_specs]}, indent=2))


if __name__ == "__main__":
    main()
