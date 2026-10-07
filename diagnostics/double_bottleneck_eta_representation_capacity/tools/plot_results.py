#!/usr/bin/env python3
"""Dependency-free SVG figures for the H0-stopped representation audit."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
FIGURES = STUDY / "figures"
NAMES = ("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6")
LABELS = ("P0 3D", "P1 Agent6", "P2 Pair8", "P3 Temporal6")


def bar_svg(path: Path, title: str, ylabel: str, labels, series, ymax: float, note: str = ""):
    width, height = 820, 460
    left, right, top, bottom = 92, 25, 58, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    groups, nseries = len(labels), len(series)
    group_w = plot_w / groups
    bar_w = min(66, group_w * 0.72 / nseries)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">{escape(title)}</text>')
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    for group, label in enumerate(labels):
        center = left + group_w * (group + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+26}" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(label)}</text>')
        for index, item in enumerate(series):
            value = float(item["values"][group])
            x = center + (index - (nseries - 1) / 2) * bar_w - bar_w * 0.43
            h = plot_h * value / ymax
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.86:.1f}" height="{h:.1f}" fill="{item["color"]}"/>')
            parts.append(f'<text x="{x+bar_w*0.43:.1f}" y="{max(top+11,y-5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:g}</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(ylabel)}</text>')
    if nseries > 1:
        x0 = left + 8
        for index, item in enumerate(series):
            x = x0 + index * 180
            parts.append(f'<rect x="{x}" y="{height-29}" width="13" height="13" fill="{item["color"]}"/>')
            parts.append(f'<text x="{x+19}" y="{height-18}" font-family="sans-serif" font-size="11">{escape(item["label"])}</text>')
    if note:
        parts.append(f'<text x="{width-25}" y="{height-48}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#555">{escape(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    h0 = json.loads((STUDY / "H0_STOP_AUDIT.json").read_text())
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    offline = json.loads((STUDY / "offline_expert_fit_and_projection_audit.json").read_text())
    bar_svg(
        FIGURES / "h0_full_population_coverage.svg",
        "Dense P0 cross-state reuse triggers the H0 stop",
        "Safe-timeout episodes rescued (of 61)",
        ("Old A004", "Six new P0", "Observed union"),
        ({"label": "coverage", "values": (h0["old_A004"]["target_success"], h0["new_dense_P0_points"]["target_success"], h0["combined_observed_P0"]["target_success"]), "color": "#3182bd"},),
        61,
        "Union 37/61; 22 episodes newly covered beyond A004",
    )
    existence = [pilot["representations"][name]["target_basin_exists"] for name in NAMES]
    bar_svg(FIGURES / "pilot_basin_existence.svg", "Pilot basin existence after exact P0 inheritance", "Targets with observed basin (of 12)", LABELS, ({"label": "existence", "values": existence, "color": "#f58518"},), 12, "Agent6 passed pilot promotion; full attribution stopped by H0")
    residual = [offline["representations"][name]["best_residual"]["median"] for name in NAMES]
    bar_svg(FIGURES / "offline_expert_fit.svg", "Oracle local executable-action fit", "Median residual (m/s); 7 valid expert states", LABELS, ({"label": "best residual", "values": residual, "color": "#4c78a8"},), 0.6, "47/54 sampled states had no validated expert continuation")
    raw_rank = [offline["representations"][name]["rank_at_best_raw"]["median"] for name in NAMES]
    exec_rank = [offline["representations"][name]["rank_at_best_executable"]["median"] for name in NAMES]
    bar_svg(FIGURES / "projection_effective_rank.svg", "Projection-coupling audit at oracle fits", "Median numerical effective rank", LABELS, ({"label": "raw", "values": raw_rank, "color": "#9ecae9"}, {"label": "after second projection", "values": exec_rank, "color": "#e6550d"}), 8)
    raw = [pilot["representations"][name]["successful_correction"]["raw_norm"]["median"] for name in NAMES]
    executed = [pilot["representations"][name]["successful_correction"]["executable_norm"]["median"] for name in NAMES]
    bar_svg(FIGURES / "pilot_projection_coupling.svg", "Successful pilot rollouts remain projection-dominated", "Median joint correction norm (m/s)", LABELS, ({"label": "raw", "values": raw, "color": "#9ecae9"}, {"label": "executed", "values": executed, "color": "#e6550d"}), 0.7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
