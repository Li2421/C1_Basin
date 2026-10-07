"""Summarize Phase 1 and freeze one outcome-adaptive refinement stage."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product

import numpy as np
from scipy.stats import beta

from .setup import HERE, SBMA, key, sha, write


def phase2_points():
    points = {}
    # D1: the new Phase-1 winner is on all three edges of its sparse local box.
    d1 = {(g, -0.25, -0.125) for g in (0.5, 0.5625, 0.59375, 0.625, 0.65625, 0.6875)}
    d1 |= {(0.625, s, -0.125) for s in (-0.375, -0.3125, -0.25, -0.1875, -0.125)}
    d1 |= {(0.625, -0.25, r) for r in (-0.25, -0.1875, -0.125, -0.0625, 0.0)}
    d1 |= set(product((0.5625, 0.59375, 0.625), (-0.3125, -0.25, -0.1875), (-0.1875, -0.125, -0.0625)))
    points["D1_pair231"] = {key(point) for point in d1}

    # D2: refine the apparent interior safe-gain minimum at (0.5,-0.625,0).
    d2 = set(product((0.4375, 0.5, 0.5625), (-0.6875, -0.625, -0.5625), (-0.0625, 0.0, 0.0625)))
    d2 |= {(0.5, s, 0.0) for s in (-0.75, -0.6875, -0.625, -0.5625, -0.5)}
    points["D2_pair228"] = {key(point) for point in d2}

    # D4: resolve the goal-gain transition between eta1=.25 and .375.
    d4 = {(g, -0.5, 0.0) for g in (0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375)}
    d4 |= set(product((0.34375, 0.375, 0.40625), (-0.5625, -0.5, -0.4375), (-0.0625, 0.0, 0.0625)))
    points["D4_pair227"] = {key(point) for point in d4}
    return points


def one_sided(success: int, n: int):
    return [
        float(beta.ppf(0.05, success, n - success + 1)) if success else 0.0,
        float(beta.ppf(0.95, success + 1, n - success)) if success < n else 1.0,
    ]


def classify(success: int, n: int):
    lo, hi = one_sided(success, n)
    if n < 16:
        return "UNKNOWN_CELL", [lo, hi]
    if lo >= 0.8:
        return "SUCCESS_CELL", [lo, hi]
    if hi <= 0.2:
        return "FAILURE_CELL", [lo, hi]
    return "UNKNOWN_CELL", [lo, hi]


def load_phase1():
    protocol = json.loads((HERE / "protocol.json").read_text())
    groups = defaultdict(dict)
    for line in (HERE.parent / "success_basin_deformation" / "per_rollout_j_def.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cohort"] == "phase_a" and row["J_def"] is not None:
            groups[(row["state_id"], key(row["eta"]))][row["seed"]] = {
                "seed": row["seed"], "outcome": row["terminal_outcome"],
                "J_def": row["J_def"], "steps": row["episode_length_steps"],
                "source": "cached_phase_a",
            }
    for manifest_path in sorted((HERE / "raw").glob("phase1_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["records"]:
            if row["outcome"] is None:
                continue
            groups[(row["state_id"], key(row["eta"]))][row["seed"]] = {
                "seed": row["seed"], "outcome": row["outcome"],
                "J_def": row["J_def"], "steps": row["steps"],
                "source": manifest["stage"],
            }
    cells = []
    for state_id, points in protocol["phase1_points"].items():
        for eta_values in points:
            eta = key(eta_values)
            rows = list(groups[(state_id, eta)].values())
            if len(rows) != 16:
                raise AssertionError((state_id, eta, len(rows)))
            counts = Counter(row["outcome"] for row in rows)
            label, interval = classify(counts["success"], len(rows))
            cells.append({
                "state_id": state_id, "eta": list(eta), "n": len(rows),
                "counts": {name: counts[name] for name in ("success", "deadlock", "timeout", "collision")},
                "success_rate": counts["success"] / len(rows),
                "Q_S_one_sided_95": interval, "classification": label,
                "mean_J_def": float(np.mean([row["J_def"] for row in rows])),
                "std_J_def": float(np.std([row["J_def"] for row in rows], ddof=1)),
                "mean_episode_length_steps": float(np.mean([row["steps"] for row in rows])),
            })
    return protocol, groups, cells


def main():
    protocol, groups, cells = load_phase1()
    winners = {}
    for state_id in protocol["states"]:
        successful = sorted(
            (cell for cell in cells if cell["state_id"] == state_id and cell["classification"] == "SUCCESS_CELL"),
            key=lambda cell: (cell["mean_J_def"], cell["eta"]),
        )
        winners[state_id] = successful[0]
    analysis = {
        "phase": "phase1_local_boundary_expansion",
        "cells": cells,
        "winners": winners,
        "interpretation": {
            "D1_pair231": "new lower successful point is on the sparse local box boundary; expand once",
            "D2_pair228": "small decrease near an apparent interior safe-gain minimum; refine",
            "D4_pair227": "new lower successful point neighbors the goal-gain failure/unknown transition; refine boundary",
        },
    }
    write("phase1_analysis.json", analysis)

    points = phase2_points()
    already = {
        state_id: {key(point) for point in protocol["phase1_points"][state_id]}
        for state_id in protocol["states"]
    }
    seeds = protocol["seeds"]
    jobs = []
    for state_id, state_points in points.items():
        for eta in sorted(state_points - already[state_id]):
            for seed in seeds:
                jobs.append({
                    "state_id": state_id, "eta": list(eta), "seed": seed,
                    "cell_id": "phase2_" + "_".join(f"{value:+.5f}" for value in eta),
                })
    plan = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": analysis["interpretation"],
        "centers": {
            "D1_pair231": [0.625, -0.25, -0.125],
            "D2_pair228": [0.5, -0.625, 0.0],
            "D4_pair227": [0.375, -0.5, 0.0],
        },
        "points": {state: [list(point) for point in sorted(values)] for state, values in points.items()},
        "new_points": {
            state: [list(point) for point in sorted(values - already[state])]
            for state, values in points.items()
        },
        "new_rollouts": len(jobs),
        "seeds": seeds,
        "common_random_numbers": True,
        "phase1_analysis_sha256": sha(HERE / "phase1_analysis.json"),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "stopping_rule": "This is the single permitted expansion/refinement stage; stop after classifying its local cells.",
    }
    write("phase2_plan.json", plan)
    write("phase2_jobs.json", jobs)
    for state_id in points:
        write(f"phase2_{state_id}_jobs.json", [job for job in jobs if job["state_id"] == state_id])
    print(json.dumps({
        "phase1_winners": {state: cell["eta"] for state, cell in winners.items()},
        "phase2_points": {state: len(values) for state, values in points.items()},
        "phase2_new_points": {state: len(values - already[state]) for state, values in points.items()},
        "phase2_new_rollouts": len(jobs),
    }, indent=2))


if __name__ == "__main__":
    main()
