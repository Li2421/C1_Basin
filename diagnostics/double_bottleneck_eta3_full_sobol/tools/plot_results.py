#!/usr/bin/env python3
"""Dependency-free SVG summaries for the common-256 P0 study."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
FIGURES = STUDY / "figures"


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">{escape(title)}</text>']


def basin_bars(rows):
    width, height, left, top, bottom = 1000, 440, 72, 55, 82
    plot_w, plot_h = width-left-25, height-top-bottom
    ordered = sorted(rows, key=lambda row: (row["rho_B"], row["episode_id"]), reverse=True)
    parts = svg_start(width, height, "P0-3D basin fractions on 61 safe-timeout episodes")
    for tick in range(6):
        value = max((row["rho_B"] for row in rows), default=0) * tick / 5
        y = top + plot_h * (1-tick/5)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>', f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{value:.3f}</text>']
    bw = plot_w / len(ordered)
    for index, row in enumerate(ordered):
        h = plot_h * row["rho_B"] / max(ordered[0]["rho_B"], 1e-12)
        color = "#2b8cbe" if row["basin_exists"] else "#d9d9d9"
        parts.append(f'<rect x="{left+index*bw:.2f}" y="{top+plot_h-h:.2f}" width="{max(.5,bw*.82):.2f}" height="{h:.2f}" fill="{color}"/>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-32}" text-anchor="middle" font-family="sans-serif" font-size="12">Episodes sorted by sampled basin fraction</text>', f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">successful eta / 256</text>', '</svg>']
    (FIGURES / "basin_fraction_distribution.svg").write_text("\n".join(parts)+"\n")


def tradeoff(etas):
    width, height, left, top, bottom = 760, 510, 72, 55, 65
    plot_w, plot_h = width-left-28, height-top-bottom
    parts = svg_start(width, height, "Shared eta coverage vs baseline-success preservation")
    for x in range(0, 62, 10):
        px = left + plot_w*x/61
        parts += [f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+plot_h}" stroke="#eee"/>', f'<text x="{px:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="sans-serif" font-size="10">{x}</text>']
    for y in range(0, 25, 4):
        py = top + plot_h*(1-y/24)
        parts += [f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_w}" y2="{py:.1f}" stroke="#eee"/>', f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{y}</text>']
    for row in etas:
        x = left + plot_w*row["timeout_coverage"]/61
        y = top + plot_h*(1-row["controls_preserved"]/24)
        color = "#d7301f" if row["pareto_nondominated"] else "#6baed6"
        radius = 4.0 if row["pareto_nondominated"] else 2.3
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="0.72"/>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="12">timeout episodes rescued (of 61)</text>', f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">success controls preserved (of 24)</text>', '<circle cx="570" cy="45" r="4" fill="#d7301f"/><text x="580" y="49" font-family="sans-serif" font-size="10">Pareto nondominated</text>', '</svg>']
    (FIGURES / "eta_rescue_preservation_tradeoff.svg").write_text("\n".join(parts)+"\n")


def overlap_heatmap(overlap):
    matrix = np.asarray(overlap["jaccard_matrix"], dtype=float)
    n = len(matrix)
    width = height = 760
    left, top, size = 85, 60, 620
    cell = size/max(n, 1)
    parts = svg_start(width, height, "Cross-state basin overlap (Jaccard)")
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            r = int(245 - 190*value); g = int(248 - 115*value); b = int(252 - 30*value)
            parts.append(f'<rect x="{left+j*cell:.2f}" y="{top+i*cell:.2f}" width="{cell+.2:.2f}" height="{cell+.2:.2f}" fill="rgb({r},{g},{b})"/>')
    parts += [f'<text x="{left+size/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif" font-size="11">basin-positive episode index</text>', f'<text x="20" y="{top+size/2}" transform="rotate(-90 20 {top+size/2})" text-anchor="middle" font-family="sans-serif" font-size="11">basin-positive episode index</text>', '</svg>']
    (FIGURES / "cross_state_overlap.svg").write_text("\n".join(parts)+"\n")


def geometry(matrix_data, geometry):
    eta = np.asarray(matrix_data["eta"], dtype=float)
    membership = np.asarray(matrix_data["timeout_membership"], dtype=bool)
    ids = [str(x) for x in matrix_data["timeout_episode_ids"]]
    low, high = np.asarray((.5,-.5,0.)), np.asarray((1.25,.5,.75))
    unit = (eta-low)/(high-low)
    roles = ("high", "median", "low", "none")
    width, height = 1040, 800
    parts = svg_start(width, height, "Representative P0-3D basin projections (common 256 points)")
    projections = ((0,1,"eta1 goal","eta2 safe"),(0,2,"eta1 goal","eta3 relation"),(1,2,"eta2 safe","eta3 relation"))
    for row_index, role in enumerate(roles):
        episode_id = geometry.get(role)
        if episode_id is None: continue
        mask = membership[ids.index(episode_id)]
        parts.append(f'<text x="20" y="{105+row_index*170}" font-family="sans-serif" font-size="11" font-weight="600">{role}: {escape(episode_id)}</text>')
        for col, (a,b,xlabel,ylabel) in enumerate(projections):
            x0, y0, w, h = 280+col*245, 62+row_index*170, 205, 135
            parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#bbb"/>')
            for point, success in zip(unit, mask):
                x=x0+point[a]*w; y=y0+(1-point[b])*h
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.0 if success else 1.55}" fill="{"#d7301f" if success else "#9ecae1"}" fill-opacity="{.9 if success else .55}"/>')
            parts += [f'<text x="{x0+w/2}" y="{y0+h+14}" text-anchor="middle" font-family="sans-serif" font-size="9">{xlabel}</text>', f'<text x="{x0+5}" y="{y0+12}" font-family="sans-serif" font-size="9">{ylabel}</text>']
    parts += ['<circle cx="820" cy="32" r="3" fill="#d7301f"/><text x="828" y="36" font-family="sans-serif" font-size="10">success</text>', '<circle cx="900" cy="32" r="2" fill="#9ecae1"/><text x="908" y="36" font-family="sans-serif" font-size="10">failure</text>', '</svg>']
    (FIGURES / "representative_eta_projections.svg").write_text("\n".join(parts)+"\n")

    width, height = 1000, 650
    parts = svg_start(width, height, "Representative P0-3D success/failure scatter (isometric view)")
    for index, role in enumerate(roles):
        episode_id = geometry.get(role)
        if episode_id is None:
            continue
        mask = membership[ids.index(episode_id)]
        col, row = index % 2, index // 2
        x0, y0, w, h = 65 + col * 485, 65 + row * 285, 405, 220
        parts += [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#bbb"/>', f'<text x="{x0+8}" y="{y0+17}" font-family="sans-serif" font-size="10" font-weight="600">{role}: {escape(episode_id)}</text>']
        # Painter's order only aids visibility; coordinates are a fixed isometric projection.
        order = np.argsort(unit[:, 2])
        for point_index in order:
            a, b, c = unit[point_index]
            x = x0 + 42 + (0.72*a + 0.28*c) * (w-84)
            y = y0 + 34 + (1.0 - (0.72*b + 0.28*c)) * (h-65)
            success = bool(mask[point_index])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.4 if success else 1.65}" fill="{"#d7301f" if success else "#9ecae1"}" fill-opacity="{.9 if success else .48}"/>')
        parts += [f'<text x="{x0+w/2}" y="{y0+h-6}" text-anchor="middle" font-family="sans-serif" font-size="9">eta1 / eta2 plane; eta3 shifts points up-right</text>']
    parts += ['<circle cx="790" cy="35" r="3" fill="#d7301f"/><text x="798" y="39" font-family="sans-serif" font-size="10">success</text>', '<circle cx="870" cy="35" r="2" fill="#9ecae1"/><text x="878" y="39" font-family="sans-serif" font-size="10">failure</text>', '</svg>']
    (FIGURES / "representative_eta3_scatter.svg").write_text("\n".join(parts)+"\n")


def robustness(local, stochastic):
    labels = ("broad", "narrow", "isolated", "seed_robust", "seed_sensitive", "lucky")
    lc = local["summary"]["center_classifications"]; sc = stochastic["summary"]["center_classifications"]
    values = (lc.get("broad",0),lc.get("narrow",0),lc.get("isolated",0),sc.get("seed_robust",0),sc.get("seed_sensitive",0),sc.get("lucky",0))
    width,height,left,top,bottom=820,450,70,55,95; pw=width-left-25; ph=height-top-bottom; ymax=max(values)+(max(values)*.1 or 1)
    parts=svg_start(width,height,"Local-eta and MACFlow-seed robustness of selected centers")
    bw=pw/len(values)
    colors=("#238b45","#fdae6b","#cb181d","#2171b5","#9ecae1","#756bb1")
    for i,(label,value,color) in enumerate(zip(labels,values,colors)):
        h=ph*value/ymax; x=left+i*bw+bw*.15; y=top+ph-h
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*.7:.1f}" height="{h:.1f}" fill="{color}"/>', f'<text x="{x+bw*.35:.1f}" y="{max(top+12,y-5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value}</text>', f'<text x="{x+bw*.35:.1f}" y="{top+ph+22}" text-anchor="middle" font-family="sans-serif" font-size="10">{label}</text>']
    parts += ['</svg>']
    (FIGURES/"robustness_classifications.svg").write_text("\n".join(parts)+"\n")


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    membership = json.loads((STUDY/"timeout_basin_membership.json").read_text())
    etas = json.loads((STUDY/"per_eta_coverage.json").read_text())["eta"]
    overlap = json.loads((STUDY/"cross_state_overlap.json").read_text())
    geometry_roles = json.loads((STUDY/"geometry_representatives.json").read_text())
    local = json.loads((STUDY/"local_robustness_results.json").read_text())
    stochastic = json.loads((STUDY/"stochastic_seed_robustness_results.json").read_text())
    basin_bars(membership["episodes"]); tradeoff(etas); overlap_heatmap(overlap)
    with np.load(STUDY/"basin_matrices.npz") as data:
        geometry(data, geometry_roles)
    robustness(local, stochastic)
    print(json.dumps({"figures": len(list(FIGURES.glob('*.svg')))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
