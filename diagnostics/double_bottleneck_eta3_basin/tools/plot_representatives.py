#!/usr/bin/env python3
"""Plot paired eta=0 and representative successful trace diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"


def main() -> int:
    metadata = json.loads((STUDY / "representatives/metadata.json").read_text())["episodes"]
    output = STUDY / "figures"
    output.mkdir(exist_ok=True)
    for item in metadata:
        zero = np.load(ROOT / item["traces"]["eta0"]["npz"])
        success = np.load(ROOT / item["traces"]["eta_success"]["npz"])
        dt = float(success["dt"])
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.8), sharex=False)
        for label, data, color in (("eta=0", zero, "#777777"), ("successful eta", success, "#2ca25f")):
            time = np.arange(len(data["total_goal_error"])) * float(data["dt"])
            axes[0].plot(time, data["total_goal_error"], color=color, label=label, linewidth=1.8)
        axes[0].set(ylabel="Sum goal distance (m)", title=f"{item['episode_id']} — timeout recovery mechanism")
        axes[0].legend()
        time = np.arange(len(success["g_norm"])) * dt
        axes[1].plot(time, success["g_norm"], label=r"$||g^\eta||$", color="#3478b8")
        axes[1].plot(time, success["second_projection_norm"], label="second-projection correction", color="#e15759", alpha=0.85)
        axes[1].set(xlabel="Time (s)", ylabel="Joint norm", title="Correction and projection interaction")
        axes[1].legend()
        fig.tight_layout()
        stem = item["episode_id"].replace("|", "_")
        fig.savefig(output / f"{stem}_mechanism.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
