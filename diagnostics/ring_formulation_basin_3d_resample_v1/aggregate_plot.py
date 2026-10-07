"""Aggregate the frozen Ring augmentation and draw empirical 3D hull envelopes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull, QhullError

HERE = Path(__file__).resolve().parent
POC = HERE.parents[1].parent / "Basin_C1_flow_field_poc_20261004"
CHAINS = ("TT", "TF", "FT", "FF")
COLORS = {"TT": "#7048a8", "TF": "#e07a16", "FT": "#168a80", "FF": "#d33f49"}


def load_new(protocol):
    points = []
    for i, eta in enumerate(protocol["eta_points"]):
        chains = {}
        for chain in CHAINS:
            results = []
            for seed in protocol["seeds"]:
                p = HERE / "raw" / f"eta_{i:02d}" / f"seed_{seed:02d}_{chain}.json"
                if not p.exists():
                    raise RuntimeError(f"incomplete run: missing {p.relative_to(HERE)}")
                results.append(json.loads(p.read_text()))
            successes = sum(r["success"] for r in results)
            unknown = sum(r["error"] is not None for r in results)
            b15 = True if successes >= 15 else False if successes + unknown < 15 else None
            chains[chain] = {"B15": b15, "successes": successes, "numerical_unknown": unknown}
        points.append({"eta_index": f"sobol_{i:02d}", "eta": eta, "cells": chains, "source": "new_64_sobol"})
    return points


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    old = json.loads((POC / "family_study_v1/main_analysis.json").read_text())
    old_uid = "family_v1_ring_dev_20261019"
    old_points = []
    for r in old["candidates"]:
        if r["state_uid"] != old_uid:
            continue
        old_points.append({"eta_index": f"grid_{r['eta_index']:02d}", "eta": r["eta"],
                           "cells": r["cells"], "source": "cached_27_grid"})
    old_points.sort(key=lambda r: int(r["eta_index"].split("_")[1]))
    new_points = load_new(protocol)
    points = old_points + new_points
    if len(old_points) != 27 or len(new_points) != 64:
        raise RuntimeError(f"expected 27 cached and 64 new points; got {len(old_points)} and {len(new_points)}")
    coords = np.asarray([p["eta"] for p in points], dtype=float)
    if len({tuple(np.round(e, 12)) for e in coords}) != len(coords):
        raise RuntimeError("duplicate η coordinates found; refusing to count any point twice")
    with (HERE / "membership_all_91.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "state_uid", "eta_index", "source", "eta1", "eta2", "eta3"] +
                   [x for c in CHAINS for x in (f"{c}_B15", f"{c}_successes", f"{c}_numerical_unknown")])
        for p in points:
            row = ["ring_exchange", protocol["state"]["uid"], p["eta_index"], p["source"], *p["eta"]]
            for c in CHAINS:
                cell = p["cells"][c]
                numerical_unknown = cell.get("numerical_unknown", cell.get("numerical", 0))
                row.extend(["unknown" if cell["B15"] is None else int(cell["B15"]), cell["successes"], numerical_unknown])
            w.writerow(row)

    counts = {c: sum(p["cells"][c]["B15"] is True for p in points) for c in CHAINS}
    hull_records = {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.titleweight": "bold", "axes.labelpad": 7})
    fig = plt.figure(figsize=(13.0, 10.1), facecolor="white")
    fig.suptitle("Ring Exchange · empirical 3D robust basin envelopes", fontsize=19, y=.978, fontweight="bold")
    fig.text(.5, .938, f"Illustrative state {old_uid}  ·  91 exact η samples  ·  B15 ≥ 15/16", ha="center", fontsize=11.5, color="#4c5661")
    for k, chain in enumerate(CHAINS):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        robust = np.asarray([p["cells"][chain]["B15"] is True for p in points])
        unknown = np.asarray([p["cells"][chain]["B15"] is None for p in points])
        cloud = coords[robust]
        ax.scatter(*coords[~robust & ~unknown].T, s=21, c="#cbd0d5", alpha=.34, marker="o", edgecolors="none", depthshade=False, zorder=1)
        if unknown.any():
            ax.scatter(*coords[unknown].T, s=50, facecolors="none", edgecolors="#b87900", linewidths=1.3,
                       marker="D", depthshade=False, zorder=6)
        if len(cloud):
            ax.scatter(*cloud.T, s=76, c=COLORS[chain], alpha=.98, marker="o", edgecolors="white", linewidths=.65, depthshade=False, zorder=5)
        hull_rec = {"robust_points": len(cloud), "hull": False, "reason": None, "affine_rank": 0}
        if len(cloud) >= 4:
            centered = cloud - cloud.mean(axis=0)
            singular = np.linalg.svd(centered, compute_uv=False)
            rank = int(np.sum(singular > max(singular[0], 1e-12) * 1e-7))
            hull_rec["affine_rank"] = rank
            if rank == 3:
                try:
                    hull = ConvexHull(cloud)
                    faces = cloud[hull.simplices]
                    surface = Poly3DCollection(faces, facecolor=COLORS[chain], edgecolor=COLORS[chain],
                                               linewidth=.42, alpha=.19, zorder=2)
                    ax.add_collection3d(surface)
                    hull_rec.update(hull=True, vertices=len(hull.vertices), facets=len(hull.simplices))
                except QhullError as exc:
                    hull_rec["reason"] = f"QhullError: {str(exc).splitlines()[0][:180]}"
            else:
                hull_rec["reason"] = "robust η cloud is affine-degenerate"
        else:
            hull_rec["reason"] = "fewer than four robust η points"
        hull_records[chain] = hull_rec
        ax.set_title(f"{chain}  ·  {counts[chain]}/91 robust", loc="left", pad=8, fontsize=15)
        ax.set(xlim=(.46, 1.29), ylim=(-.54, .54), zlim=(-.035, .785),
               xlabel=r"$\eta_1$", ylabel=r"$\eta_2$", zlabel=r"$\eta_3$")
        ax.set_xticks([.5, .875, 1.25]); ax.set_yticks([-.5, 0, .5]); ax.set_zticks([0, .375, .75])
        ax.tick_params(labelsize=9, pad=1)
        ax.view_init(elev=23, azim=-56)
        ax.set_box_aspect((1.1, 1, .85))
        ax.xaxis.pane.set_facecolor((.96, .97, .98, .45))
        ax.yaxis.pane.set_facecolor((.96, .97, .98, .45))
        ax.zaxis.pane.set_facecolor((.96, .97, .98, .45))
        ax.grid(True, alpha=.16)

    legend = [Line2D([], [], marker="o", linestyle="", markersize=5, color="#cbd0d5", alpha=.6, label="Non-robust sampled η"),
              Line2D([], [], marker="o", linestyle="", markersize=8, color="#555d66", label="B15 robust sampled η"),
              Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor="#aab2bb", markeredgecolor="#707983", alpha=.28, label="Convex hull visual envelope")]
    if any(p["cells"][c]["B15"] is None for p in points for c in CHAINS):
        legend.append(Line2D([], [], marker="D", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor="#b87900", label="Unresolved B15 η"))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .063), fontsize=10.5)
    fig.text(.5, .027, "Hulls are visual envelopes of sampled robust interventions, not certified continuous basin boundaries.", ha="center", fontsize=9.5, color="#4c5661")
    fig.subplots_adjust(left=.015, right=.985, top=.91, bottom=.11, wspace=.0, hspace=.09)
    fig.savefig(HERE / "ring_four_formulation_basin_3d_hulls.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "ring_four_formulation_basin_3d_hulls.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (HERE / "hull_status.json").write_text(json.dumps(hull_records, indent=2, sort_keys=True) + "\n")
    caption = (
        f"Ring Exchange, illustrative state {old_uid}, selected from the previously completed Ring study by a deterministic ranking over its four complete candidate states. "
        "The figure combines 27 η points from the existing 3×3×3 grid with 64 newly evaluated, preregistered scrambled Sobol η points in the same domain. "
        "At every point, TT/TF/FT/FF use the same physical state and the same 16 continuation seeds; robust membership is B15 (at least 15 scientific successes out of 16). "
        "Gray markers denote non-robust sampled η and vivid markers denote robust sampled η for that panel. "
        "Where stable full-rank point clouds permit, translucent surfaces show their 3D convex hulls. "
        "Hulls are visual envelopes of sampled robust interventions, not certified continuous basin boundaries. They do not represent exact volume, boundary, or topology. "
        "No interpolation or synthetic membership labels are used. No model training was performed.\n"
    )
    (HERE / "figure_caption.txt").write_text(caption)
    (HERE / "summary.json").write_text(json.dumps({"scene": "ring_exchange", "state_uid": old_uid, "sample_count": len(points),
                                                       "source_counts": {"existing_27_grid": len(old_points), "new_fixed_sobol": len(new_points)},
                                                       "robust_counts": counts, "hulls": hull_records,
                                                       "png": "ring_four_formulation_basin_3d_hulls.png",
                                                       "pdf": "ring_four_formulation_basin_3d_hulls.pdf"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state_uid": old_uid, "robust_counts": counts, "hulls": hull_records,
                      "png": str(HERE / "ring_four_formulation_basin_3d_hulls.png")}, indent=2))


if __name__ == "__main__":
    main()
