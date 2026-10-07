#!/usr/bin/env python3
"""Create compact Recovery-V2 tables and static figures from retained JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("D0", "D1", "D2-T", "D2-G", "D2-V", "D2")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evaluation = {
        name: json.loads((root / "evaluation" / f"{name.lower()}.json").read_text())
        for name in ORDER
    }
    training = {
        row["variant"]: row
        for row in json.loads((root / "models/training_comparison.json").read_text())[
            "variants"
        ]
    }
    baseline_training = json.loads(
        (root.parent / "double_bottleneck_toy_transfer_audit/training_comparison.json").read_text()
    )["variants"]
    for row in baseline_training:
        if row["variant"] == "nominal_25k":
            training["D0"] = row
        elif row["variant"] == "recovery_union_25k":
            training["D1"] = row

    closed = []
    teacher = []
    k_rows = []
    for name in ORDER:
        value = evaluation[name]
        aggregate = value["aggregate"]
        closed.append(
            {
                "variant": name,
                "training_rows": aggregate["support"]["training_rows"],
                "successes": aggregate["successes"],
                "wall_collisions": aggregate["wall_collisions"],
                "agent_collisions": aggregate["agent_collisions"],
                "timeouts": aggregate["timeouts"],
                "median_steps": aggregate["median_episode_steps"],
                "mean_steps": aggregate["mean_episode_steps"],
                "minimum_wall_clearance": aggregate["minimum_wall_clearance"],
                "minimum_pair_clearance": aggregate["minimum_pair_clearance"],
                "fixed_support_ood_fraction": aggregate["support"]["outside_fraction"],
                "k100_collision_rate": value["k_step"]["summary"]["100"]["collision_rate"],
                "successful_mode_signatures": len(aggregate["successful_mode_signatures"]),
                "training_validation_loss": training[name].get(
                    "final_fixed_val_loss",
                    training[name].get("final_selection_val_loss"),
                ),
            }
        )
        teacher_value = value["teacher_forced"]
        teacher.append(
            {
                "variant": name,
                "overall_action_rmse": teacher_value["overall"]["joint_action_rmse"],
                "waiting_transition_rmse": teacher_value["waiting_transition"][
                    "joint_action_rmse"
                ],
                "goal_region_rmse": teacher_value["goal_region"]["joint_action_rmse"],
                "wall_directed_absolute_error": teacher_value["overall"][
                    "wall_directed_absolute_error"
                ],
            }
        )
        for horizon, row in value["k_step"]["summary"].items():
            k_rows.append(
                {
                    "variant": name,
                    "horizon": int(horizon),
                    **row,
                }
            )
    write_csv(root / "tables/closed_loop.csv", closed)
    write_csv(root / "tables/teacher_forced.csv", teacher)
    write_csv(root / "tables/k_step.csv", k_rows)

    failures = json.loads((root / "failure_replay.json").read_text())
    failure_rows = []
    for variant, rows in failures["variants"].items():
        for row in rows:
            divergence = row["first_clear_divergence"]
            failure_rows.append(
                {
                    "variant": variant,
                    "family_id": row["family_id"],
                    "seed": row["seed"],
                    "termination": row["termination"],
                    "episode_steps": row["episode_steps"],
                    "first_clear_divergence_step": row["first_clear_divergence_step"],
                    "divergence_phase": None if divergence is None else divergence["phase"],
                    "divergence_support_distance": None
                    if divergence is None
                    else divergence["nearest_D2_support"]["distance"],
                    "divergence_support_component": None
                    if divergence is None
                    else divergence["nearest_D2_support"]["component"],
                    "inside_fixed_support": None
                    if divergence is None
                    else divergence["inside_fixed_support"],
                    "local_recoverable_at_divergence": None
                    if divergence is None
                    else divergence["local_hypothesis_recovery"]["success"],
                    "local_recoverable_pre_collision": row["last_safe_pre_collision"][
                        "local_hypothesis_recovery"
                    ]["success"],
                }
            )
    write_csv(root / "tables/failure_replay.csv", failure_rows)

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    x = np.arange(len(ORDER))
    success = np.asarray([row["successes"] for row in closed])
    wall = np.asarray([row["wall_collisions"] for row in closed])
    agent = np.asarray([row["agent_collisions"] for row in closed])
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(x - 0.25, success, 0.25, label="success")
    ax.bar(x, wall, 0.25, label="wall collision")
    ax.bar(x + 0.25, agent, 0.25, label="agent collision")
    ax.set_xticks(x, ORDER)
    ax.set_ylabel("rollouts (n=12)")
    ax.set_title("Matched closed-loop outcomes")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "closed_loop_outcomes.svg")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.2, 4.5))
    ood = 100 * np.asarray([row["fixed_support_ood_fraction"] for row in closed])
    k100 = 100 * np.asarray([row["k100_collision_rate"] for row in closed])
    ax1.plot(x, ood, "o-", label="closed-loop OOD", color="#1f77b4")
    ax1.plot(x, k100, "s-", label="K=100 collision", color="#d62728")
    ax1.set_xticks(x, ORDER)
    ax1.set_ylabel("percent")
    ax1.set_ylim(-2, 102)
    ax1.set_title("Support departure and 100-step collision")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "support_and_k100.svg")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
