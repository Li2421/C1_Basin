#!/usr/bin/env python3
"""Create compact tables and figures for the frozen uniform-recovery study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = ("D0", "D1", "D2-T", "U-Low", "U-Mid", "U-High")
UNIFORM_ORDER = ("U-Low", "U-Mid", "U-High")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    study = root / "diagnostics/double_bottleneck_uniform_recovery"
    tables = study / "tables"
    plots = study / "plots"
    tables.mkdir(exist_ok=False)
    plots.mkdir(exist_ok=False)
    manifest = json.loads((study / "data/manifest.json").read_text())
    comparison = json.loads((study / "evaluation/comparison.json").read_text())
    training = json.loads((study / "models/training_comparison.json").read_text())
    failures = json.loads((study / "failure_analysis.json").read_text())

    dataset_rows = [
        {
            "variant": "D0",
            "nominal_rows": 52873,
            "recovery_rows": 0,
            "total_training_rows": 52873,
            "anchors_per_trajectory": 0,
        }
    ]
    for variant in UNIFORM_ORDER:
        item = manifest["splits"]["train"][variant]
        dataset_rows.append(
            {
                "variant": variant,
                "nominal_rows": 52873,
                "recovery_rows": item["rows"],
                "total_training_rows": 52873 + item["rows"],
                "anchors_per_trajectory": item["anchors_per_trajectory"],
            }
        )
    dataset_rows.append(
        {
            "variant": "D1/full-per-transition reference",
            "nominal_rows": 52873,
            "recovery_rows": 52873,
            "total_training_rows": 105746,
            "anchors_per_trajectory": "all",
        }
    )
    _write_csv(tables / "datasets.csv", dataset_rows)

    phase_rows = []
    for variant in UNIFORM_ORDER:
        item = manifest["splits"]["train"][variant]
        for phase, count in item["source_phase_counts"].items():
            phase_rows.append(
                {
                    "variant": variant,
                    "phase": phase,
                    "count": count,
                    "fraction": count / item["rows"],
                }
            )
    _write_csv(tables / "natural_phase_distribution.csv", phase_rows)

    training_rows = []
    for row in training["variants"]:
        training_rows.append(
            {
                "variant": row["variant"],
                "training_transitions": row["training_transitions"],
                "validation_transitions": row["validation_transitions"],
                "updates": row["steps"],
                "expected_passes": row["expected_training_passes"],
                "final_train_loss": row["final_fixed_train_loss"],
                "final_validation_loss": row["final_fixed_val_loss"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )
    _write_csv(tables / "training.csv", training_rows)

    closed_rows, teacher_rows, k_rows = [], [], []
    for set_name in ("development", "untouched_test"):
        for variant in VARIANT_ORDER:
            result = comparison["sets"][set_name][variant]
            aggregate = result["aggregate"]
            successful_directions = sorted(
                {
                    row["coordination_mode"].get("first_direction")
                    for row in result["rollouts"]
                    if row["success"]
                }
            )
            closed_rows.append(
                {
                    "set": set_name,
                    "variant": variant,
                    "rollouts": aggregate["rollouts"],
                    "successes": aggregate["successes"],
                    "wall_collisions": aggregate["wall_collisions"],
                    "agent_collisions": aggregate["agent_collisions"],
                    "timeouts": aggregate["timeouts"],
                    "median_steps": aggregate["median_episode_steps"],
                    "mean_steps": aggregate["mean_episode_steps"],
                    "minimum_wall_clearance": aggregate["minimum_wall_clearance"],
                    "minimum_pair_clearance": aggregate["minimum_pair_clearance"],
                    "ood_fraction": aggregate["support"]["outside_fraction"],
                    "successful_first_directions": "|".join(successful_directions),
                }
            )
            teacher = result["teacher_forced"]
            teacher_rows.append(
                {
                    "set": set_name,
                    "variant": variant,
                    "action_rmse": teacher["overall"]["joint_action_rmse"],
                    "wall_directed_error": teacher["overall"]["wall_directed_absolute_error"],
                    "waiting_transition_rmse": teacher["waiting_transition"]["joint_action_rmse"],
                    "goal_region_rmse": teacher["goal_region"]["joint_action_rmse"],
                }
            )
            for horizon, values in result["k_step"]["summary"].items():
                k_rows.append(
                    {
                        "set": set_name,
                        "variant": variant,
                        "horizon": int(horizon),
                        "position_rmse": values["position_rmse"],
                        "action_rmse": values["action_rmse"],
                        "collision_rate": values["collision_rate"],
                        "continuations": values["continuations"],
                    }
                )
    _write_csv(tables / "closed_loop.csv", closed_rows)
    _write_csv(tables / "teacher_forced.csv", teacher_rows)
    _write_csv(tables / "k_step.csv", k_rows)

    failure_rows = []
    for variant in UNIFORM_ORDER:
        item = failures["variants"][variant]
        failure_rows.append(
            {
                "variant": variant,
                "failures": item["failures"],
                "clear_divergences": item["failures_with_clear_divergence"],
                "clear_divergences_inside_support": item[
                    "clear_divergence_inside_support"
                ],
                "phase_counts_json": json.dumps(
                    item["clear_divergence_phase_counts"], sort_keys=True
                ),
            }
        )
    _write_csv(tables / "failure_summary.csv", failure_rows)

    test = comparison["sets"]["untouched_test"]
    colors = ["#777777", "#4C78A8", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    x = np.arange(len(VARIANT_ORDER))
    success = [test[v]["aggregate"]["successes"] for v in VARIANT_ORDER]
    wall = [test[v]["aggregate"]["wall_collisions"] for v in VARIANT_ORDER]
    agent = [test[v]["aggregate"]["agent_collisions"] for v in VARIANT_ORDER]
    timeout = [test[v]["aggregate"]["timeouts"] for v in VARIANT_ORDER]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(x, success, label="success", color="#54A24B")
    ax.bar(x, wall, bottom=success, label="wall collision", color="#E45756")
    ax.bar(x, agent, bottom=np.asarray(success) + wall, label="agent collision", color="#F2CF5B")
    ax.bar(
        x,
        timeout,
        bottom=np.asarray(success) + wall + agent,
        label="timeout",
        color="#B279A2",
    )
    ax.set_xticks(x, VARIANT_ORDER)
    ax.set_ylabel("Untouched-test rollouts (n=24)")
    ax.set_title("Uniform recovery does not reach reliable closed-loop success")
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(plots / "untouched_test_outcomes.svg")
    plt.close(fig)

    density = [0, 64, 256, 640]
    density_variants = ["D0", *UNIFORM_ORDER]
    success_fraction = [test[v]["aggregate"]["successes"] / 24 for v in density_variants]
    k100 = [test[v]["k_step"]["summary"]["100"]["collision_rate"] for v in density_variants]
    ood = [test[v]["aggregate"]["support"]["outside_fraction"] for v in density_variants]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True)
    for ax, values, label in zip(
        axes,
        (success_fraction, k100, ood),
        ("success fraction", "K=100 collision rate", "closed-loop OOD fraction"),
        strict=True,
    ):
        ax.plot(density, values, marker="o", color="#4C78A8")
        ax.set_xlabel("uniform anchors / trajectory")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.suptitle("Density effects are not monotonic in the primary untouched test")
    fig.tight_layout()
    fig.savefig(plots / "uniform_density_effect.svg")
    plt.close(fig)

    summary = {
        "schema": "double_bottleneck_uniform_recovery_summary_v1",
        "tables": sorted(path.name for path in tables.iterdir()),
        "plots": sorted(path.name for path in plots.iterdir()),
    }
    (study / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
