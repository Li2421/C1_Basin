"""Read-only extraction of archived Toy actions and a Ring reference-action check.

This script does not call env.step, sample a controller, or solve a QP.  It
rebuilds only the Ring CBF rows from the frozen environment and checks a
specified controller-independent reference at one ideal configuration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ring_exchange.environment import RingExchangeEnv
from ring_exchange.safety import polygonal_obstacle_snapshot
from shared_control.hard_projection import HardProjectionConfig, barrier_constraints


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "diagnostics/tt_ff_correction_realization_20261017_v1/tt_ff_action_validation.csv"
RING_STATES = ROOT / "diagnostics/crosstype1c_two_eta_schedule_v1/NEW_EVAL_STATES.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_metrics() -> dict:
    rows = list(csv.DictReader(SOURCE.open()))
    fields = [
        "source_row", "state_id", "base_uid", "physical_step", "seed", "noise_key",
        "controller", "placement", "eta", "parameter", "x1", "y1", "x2", "y2",
        "K_entry", "K_pass", "raw_yield_delta2y", "raw_pass_delta_gap",
        "actual_yield_u2y", "actual_pass_gap", "min_wall_h", "pair_h",
        "min_wall_cbf_slack", "pair_cbf_slack", "min_speed_ball_slack",
        "numerical_failure", "episode_outcome", "r_m", "d_c", "eta_dagger_raw_yield",
        "eta_dagger_raw_pass", "eta_dagger_action",
        "classification",
    ]
    counts = {"rows": len(rows), "distinct_state_key": len({(r["state_id"], r["noise_key"]) for r in rows})}
    eligible = 0
    with (OUT / "EXPLOITABILITY_METRICS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            (x1, y1), (x2, y2) = json.loads(row["positions"])
            basis = np.asarray(json.loads(row["basis_columns"]), dtype=np.float64)
            anchor = np.asarray(json.loads(row["safe_anchor"]), dtype=np.float64)
            goal = basis[:, :, 0]
            goal_norm2 = np.sum(goal * goal, axis=1)
            along = np.zeros_like(anchor)
            nonzero = goal_norm2 > 0
            along[nonzero] = (np.sum(anchor[nonzero] * goal[nonzero], axis=1) / goal_norm2[nonzero])[:, None] * goal[nonzero]
            d_c = float(np.linalg.norm(along))
            dagger_raw = np.einsum("ijk,k->ij", basis, np.asarray([.5, -1 / 3.303687238760696, .55]))
            entry = -.65 <= x1 <= -.35 and -.025 <= x2 <= .025 and -.02 <= y1 <= 0 and -.02 <= y2 <= .02
            passing = -.025 <= x2 <= .025 and -.65 <= x1 <= x2 + .34 and -.02 <= y1 <= 0 and .38 <= y2 <= .46
            eligible += int(entry or passing)
            writer.writerow({
                "source_row": idx, "state_id": row["state_id"], "base_uid": row["base_uid"],
                "physical_step": row["physical_step"], "seed": row["seed"], "noise_key": row["noise_key"],
                "controller": "Toy frozen Flow BC checkpoint", "placement": row["chain"],
                "eta": row["eta"], "parameter": row["parameter"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "K_entry": int(entry), "K_pass": int(passing),
                "raw_yield_delta2y": row["raw_delta2y"], "raw_pass_delta_gap": row["raw_delta1x_minus_delta2x"],
                "actual_yield_u2y": row["u2y"], "actual_pass_gap": row["u1x_minus_u2x"],
                "min_wall_h": row["min_wall_h"], "pair_h": row["pair_h"],
                "min_wall_cbf_slack": row["min_wall_cbf_slack"], "pair_cbf_slack": row["pair_cbf_slack"],
                "min_speed_ball_slack": row["min_speed_ball_slack"],
                "numerical_failure": row["numerical_failure"],
                "d_c": d_c, "eta_dagger_raw_yield": dagger_raw[1, 1],
                "eta_dagger_raw_pass": dagger_raw[0, 0] - dagger_raw[1, 0],
                "classification": "outside_prespecified_stage_domain" if not (entry or passing) else "stage_data_incomplete",
            })
    counts["eligible_rows"] = eligible
    counts["TT_rows"] = sum(r["chain"] == "TT" for r in rows)
    counts["TT_witness_rows"] = sum(r["chain"] == "TT" and r["parameter"] == "witness" for r in rows)
    counts["TT_witness_both_positive"] = sum(
        r["chain"] == "TT" and r["parameter"] == "witness" and
        float(r["u2y"]) > 1e-8 and float(r["u1x_minus_u2x"]) > 1e-8 for r in rows
    )
    return counts


def ring_reference() -> dict:
    env = RingExchangeEnv()
    theta = np.arange(4) * math.pi / 2
    p = 2.5 * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    env.reset(p, goals=-p)
    snapshot = polygonal_obstacle_snapshot(env, sides=48)
    config = HardProjectionConfig()
    matrix, lower, geometry = barrier_constraints(snapshot, config)
    omega = 0.1  # angular speed in radians/second; v=0.25 m/s at r=2.5
    u = omega * np.stack((-p[:, 1], p[:, 0]), axis=1)
    slack = matrix @ u.reshape(-1) - lower
    n = 628
    multiplier = (1 + 1j * omega * env.config.dt) ** n
    terminal = p * multiplier.real + np.stack((-p[:, 1], p[:, 0]), axis=1) * multiplier.imag
    goal_error = np.linalg.norm(terminal + p, axis=1)
    result = {
        "scenario": "ring_exchange", "constructed_reference_only": True,
        "CBF_rows": len(lower), "pair_rows": 6, "wall_rows": len(lower) - 6,
        "initial_min_pair_slack": float(slack[:6].min()),
        "initial_min_wall_slack": float(slack[6:].min()),
        "initial_min_wall_h": float(geometry["wall_h"].min()),
        "initial_max_speed": float(np.linalg.norm(u, axis=1).max()),
        "initial_max_speed_limit": env.config.max_speed,
        "reference_steps": n, "terminal_max_goal_error": float(goal_error.max()),
        "goal_tolerance": env.config.goal_tolerance,
        "polygon_sides_each_boundary": 48, "angular_rate_rad_per_s": omega,
        "source_hashes": {
            str((ROOT / path).relative_to(ROOT)): sha(ROOT / path)
            for path in ["ring_exchange/environment.py", "ring_exchange/safety.py", "shared_control/hard_projection.py"]
        },
    }
    (OUT / "RING_REFERENCE_CERTIFICATE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def ring_initial_coverage() -> dict:
    """Check necessary, phase-independent conditions for the declared open class."""
    states = json.loads(RING_STATES.read_text())
    rows = []
    for state in states:
        p = np.asarray(state["physical"]["positions"], dtype=np.float64)
        g = np.asarray(state["physical"]["goals"], dtype=np.float64)
        radial = float(np.max(np.abs(np.linalg.norm(p, axis=1) - 2.5)))
        antipodal = float(np.max(np.linalg.norm(p + g, axis=1)))
        # If |p-p0|<.01 and |g+p0|<.05, triangle inequality requires |p+g|<.06.
        rows.append({"state_uid": state["state_uid"], "max_radial_deviation": radial,
                     "max_antipodal_error": antipodal, "passes_necessary_radial": radial < .01,
                     "passes_necessary_antipodal": antipodal < .06})
    result = {
        "source": str(RING_STATES.relative_to(ROOT)), "source_sha256": sha(RING_STATES),
        "state_count": len(rows), "necessary_radial_pass_count": sum(r["passes_necessary_radial"] for r in rows),
        "necessary_antipodal_pass_count": sum(r["passes_necessary_antipodal"] for r in rows),
        "joint_possible_count": sum(r["passes_necessary_radial"] and r["passes_necessary_antipodal"] for r in rows),
        "min_max_radial_deviation": min(r["max_radial_deviation"] for r in rows),
        "min_max_antipodal_error": min(r["max_antipodal_error"] for r in rows),
        "rows": rows,
        "note": "Necessary-condition exclusion only; no phase fitting or endpoint labels used.",
    }
    (OUT / "RING_INITIAL_COVERAGE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {k: v for k, v in result.items() if k != "rows"}


if __name__ == "__main__":
    print(json.dumps({"toy": toy_metrics(), "ring": ring_reference(), "ring_initial_coverage": ring_initial_coverage()}, indent=2))
