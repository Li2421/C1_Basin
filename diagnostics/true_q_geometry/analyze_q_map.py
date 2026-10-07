"""Summarize the predeclared full-horizon Q map and select direction checks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from diagnostics.true_q_geometry.run_q_map import HERE, locate_assets
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


RAW = HERE / "raw/q_map"
OUTCOMES = ("deadlock", "success", "timeout", "collision")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def paired_test(a: list[int], b: list[int]) -> dict:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    diff = b - a
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    mean = float(np.mean(diff))
    return {
        "n": int(len(a)),
        "q_deadlock_a": float(np.mean(a)),
        "q_deadlock_b": float(np.mean(b)),
        "difference_b_minus_a": mean,
        "normal_95_ci_difference": [max(-1.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)],
        "a_deadlock_b_not": a_only,
        "a_not_b_deadlock": b_only,
        "discordant": discordant,
        "two_sided_exact_mcnemar_p": p,
    }


def active_switch_fraction(active: np.ndarray) -> float:
    if len(active) < 2:
        return 0.0
    return float(np.mean(np.any(active[1:] != active[:-1], axis=1)))


def horizon_features(data, goals: np.ndarray, horizon: int) -> dict:
    length = len(data["event"])
    count = min(horizon, length)
    initial = np.asarray(data["positions_before"][0])
    final = np.asarray(data["positions_after"][count - 1])
    e0 = np.linalg.norm(goals - initial, axis=-1)
    e1 = np.linalg.norm(goals - final, axis=-1)
    order0 = initial[0, 0] - initial[1, 0]
    order1 = final[0, 0] - final[1, 0]
    return {
        "requested_steps": horizon,
        "observed_steps": count,
        "task_progress_sum": float(np.sum(e0 - e1)),
        "task_progress_min_agent": float(np.min(e0 - e1)),
        "longitudinal_order_change": float(order1 - order0),
        "state_displacement": float(np.linalg.norm(final - initial)),
        "executed_speed_mean": float(np.mean(np.linalg.norm(data["u_exec"][:count], axis=-1))),
    }


def trace_summary(path: Path, record: dict, goals: np.ndarray, dt: float) -> dict:
    with np.load(path) as data:
        outcome = str(data["outcome"])
        steps = len(data["event"])
        gnorm = np.linalg.norm(data["g"], axis=-1)
        first_residual = np.linalg.norm(data["u_flow"] - data["u_safe"], axis=(1, 2))
        second_residual = np.linalg.norm(data["w"] - data["u_exec"], axis=(1, 2))
        second_active = np.asarray(data["second_active"], dtype=bool)
        summary = {
            "trace_id": record["trace_id"],
            "flow_seed": record["flow_seed"],
            "origin": record["origin"],
            "outcome": outcome,
            "episode_length_steps": steps,
            "episode_length_seconds": steps * dt,
            "time_to_deadlock_seconds": steps * dt if outcome == "deadlock" else None,
            "time_to_success_seconds": steps * dt if outcome == "success" else None,
            "cumulative_correction_norm": float(np.sum(gnorm)),
            "integrated_correction_norm": float(np.sum(gnorm) * dt),
            "mean_first_projection_residual": float(np.mean(first_residual)),
            "mean_second_projection_residual": float(np.mean(second_residual)),
            "second_projection_clipping_fraction": float(np.mean(second_residual > 1e-8)),
            "second_active_switch_fraction": active_switch_fraction(second_active),
            "mean_executed_action_norm": float(np.mean(np.linalg.norm(data["u_exec"], axis=-1))),
            "terminal_positions": np.asarray(data["positions_after"][-1]).tolist(),
            "horizon_20": horizon_features(data, goals, 20),
            "horizon_100": horizon_features(data, goals, 100),
        }
    return summary


def main():
    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    manifest = json.loads((RAW / "manifest.json").read_text())
    catalog = json.loads((RAW / "state_catalog.json").read_text())
    assets = locate_assets(None)
    _, provenance = load_policy(assets / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    config = Config(**provenance["evaluation_environment"])
    goals = GiveWayEnv(config).goals
    by_state_probe = defaultdict(lambda: defaultdict(list))
    record_lookup = {}
    for record in manifest["records"]:
        summary = trace_summary(RAW / record["relative_path"], record, goals, config.dt)
        by_state_probe[record["state_id"]][record["probe_name"]].append(summary)
        record_lookup[(record["state_id"], record["probe_name"], record["flow_seed"])] = summary

    probes = protocol["phi_probes"]
    state_results = []
    deadlock_vectors = {}
    for state in catalog:
        state_id = state["state_id"]
        policies = []
        deadlock_vectors[state_id] = {}
        for probe_name, phi in probes.items():
            traces = sorted(by_state_probe[state_id][probe_name], key=lambda x: x["flow_seed"])
            n = len(traces)
            counts = {name: sum(x["outcome"] == name for x in traces) for name in OUTCOMES}
            deadlock_vectors[state_id][probe_name] = [int(x["outcome"] == "deadlock") for x in traces]
            policies.append({
                "probe_name": probe_name,
                "phi": phi,
                "n": n,
                "counts": counts,
                "Q_D": counts["deadlock"] / n,
                "Q_success": counts["success"] / n,
                "Q_timeout": counts["timeout"] / n,
                "Q_collision": counts["collision"] / n,
                "uncertainty": {name: wilson(counts[name], n) for name in OUTCOMES},
                "mean_time_to_deadlock_seconds": (
                    float(np.mean([x["time_to_deadlock_seconds"] for x in traces if x["outcome"] == "deadlock"]))
                    if counts["deadlock"] else None
                ),
                "mean_time_to_success_seconds": (
                    float(np.mean([x["time_to_success_seconds"] for x in traces if x["outcome"] == "success"]))
                    if counts["success"] else None
                ),
                "mean_episode_length_steps": float(np.mean([x["episode_length_steps"] for x in traces])),
                "mean_cumulative_correction_norm": float(np.mean([x["cumulative_correction_norm"] for x in traces])),
                "mean_projection_clipping_fraction": float(np.mean([x["second_projection_clipping_fraction"] for x in traces])),
                "mean_projection_residual": float(np.mean([x["mean_second_projection_residual"] for x in traces])),
                "mean_active_switch_fraction": float(np.mean([x["second_active_switch_fraction"] for x in traces])),
                "mean_executed_action_norm": float(np.mean([x["mean_executed_action_norm"] for x in traces])),
                "trajectory_summaries": traces,
            })
        state_results.append({
            "state_id": state_id,
            "selection": {k: state[k] for k in (
                "pair_id", "source_phi", "source_outcome", "start_step", "nominal_offset_seconds"
            )},
            "initial_features": state["features"],
            "policies": policies,
        })

    qmap = {
        "schema": "c1_true_q_map_v1",
        "definition": "P_phi(strict deadlock before success/timeout | augmented starting state)",
        "outcome_taxonomy": list(OUTCOMES),
        "dt": config.dt,
        "samples_per_state_phi": 4,
        "uncertainty_rule": "Wilson 95% binomial intervals; estimates are empirical, not formal continuous-domain guarantees",
        "new_continuations": manifest["new_continuations"],
        "cached_continuations_reused": manifest["cached_continuations_reused"],
        "actual_new_physical_steps": manifest["actual_new_physical_steps"],
        "states": state_results,
    }
    (HERE / "full_horizon_q_map.json").write_text(json.dumps(qmap, indent=2) + "\n")

    coordinate_slices = {
        "goal": ("goal_m1", "p0", "goal_p1"),
        "safe": ("safe_m1", "p0", "safe_p1"),
        "relative": ("rel_m1", "p0", "rel_p1"),
    }
    qlookup = {
        (s["state_id"], p["probe_name"]): p["Q_D"]
        for s in state_results for p in s["policies"]
    }
    state_geometry = []
    for state in catalog:
        sid = state["state_id"]
        q = {name: qlookup[(sid, name)] for name in probes}
        qrange = max(q.values()) - min(q.values())
        fd = {}
        monotone = []
        for coord, (minus, center, plus) in coordinate_slices.items():
            paired_diffs = np.asarray(deadlock_vectors[sid][plus]) - np.asarray(deadlock_vectors[sid][minus])
            estimate = (q[plus] - q[minus]) / (2 * protocol["delta_primary"])
            se = float(np.std(paired_diffs, ddof=1) / math.sqrt(len(paired_diffs)) / (2 * protocol["delta_primary"]))
            fd[coord] = {
                "central_fd": estimate,
                "normal_95_ci": [estimate - 1.96 * se, estimate + 1.96 * se],
                "q_minus": q[minus], "q_center": q[center], "q_plus": q[plus],
                "paired_test_minus_vs_plus": paired_test(deadlock_vectors[sid][minus], deadlock_vectors[sid][plus]),
            }
            if ((q[minus] <= q[center] <= q[plus]) or (q[minus] >= q[center] >= q[plus])) and abs(q[plus] - q[minus]) >= 0.5:
                monotone.append(coord)
        outer_goal = (q["goal_p2"] - q["goal_m2"]) / (2 * protocol["delta_secondary"])
        inner_goal = fd["goal"]["central_fd"]
        goal_sign_stable = inner_goal == 0 or outer_goal == 0 or np.sign(inner_goal) == np.sign(outer_goal)
        largest_adjacent = max(
            abs(q[a] - q[b]) for a, b in (
                ("goal_m2", "goal_m1"), ("goal_m1", "p0"), ("p0", "goal_p1"), ("goal_p1", "goal_p2"),
                ("safe_m1", "p0"), ("p0", "safe_p1"), ("rel_m1", "p0"), ("p0", "rel_p1"),
            )
        )
        if len(set(q.values())) == 1 and q["p0"] in (0.0, 1.0):
            cls = "SATURATED"
        elif largest_adjacent >= 0.75:
            cls = "DISCONTINUOUS / SWITCHING"
        elif monotone and goal_sign_stable:
            cls = "DIRECTIONAL"
        elif qrange <= 0.25:
            cls = "FLAT"
        else:
            cls = "UNCERTAIN"
        vector = [fd["goal"]["central_fd"], fd["safe"]["central_fd"], fd["relative"]["central_fd"]]
        state_geometry.append({
            "state_id": sid,
            "Q_D_by_probe": q,
            "Q_D_range": qrange,
            "classification": cls,
            "monotone_primary_slices": monotone,
            "largest_adjacent_Q_D_jump": largest_adjacent,
            "finite_differences": fd,
            "goal_outer_central_fd": outer_goal,
            "goal_delta_sign_stable": bool(goal_sign_stable),
            "g_TRUE": vector,
            "g_TRUE_norm": float(np.linalg.norm(vector)),
        })

    pooled = {}
    for coord, (minus, _, plus) in coordinate_slices.items():
        avec, bvec = [], []
        for state in catalog:
            sid = state["state_id"]
            avec += deadlock_vectors[sid][minus]
            bvec += deadlock_vectors[sid][plus]
        pooled[coord] = paired_test(avec, bvec)

    state_pair_tests = []
    for state in catalog:
        sid = state["state_id"]
        for a, b in itertools.combinations(probes, 2):
            result = paired_test(deadlock_vectors[sid][a], deadlock_vectors[sid][b])
            if result["discordant"]:
                state_pair_tests.append({"state_id": sid, "probe_a": a, "probe_b": b, **result})
    significant_005 = sum(x["two_sided_exact_mcnemar_p"] < 0.05 for x in state_pair_tests)
    significant_010 = sum(x["two_sided_exact_mcnemar_p"] < 0.10 for x in state_pair_tests)
    nonflat_gate = any(x["Q_D_range"] >= 0.5 for x in state_geometry) or any(
        x["two_sided_exact_mcnemar_p"] < 0.05 for x in pooled.values()
    )
    candidates = sorted(
        [x for x in state_geometry if x["Q_D_range"] >= 0.5],
        key=lambda x: (-x["g_TRUE_norm"], x["state_id"]),
    )[:3]
    selection = []
    eta = protocol["stage_2_direction_validation"]["eta"]
    for item in candidates:
        g = np.asarray(item["g_TRUE"], dtype=float)
        unit = g / np.linalg.norm(g)
        selection.append({
            "state_id": item["state_id"],
            "g_TRUE": g.tolist(),
            "g_TRUE_norm": float(np.linalg.norm(g)),
            "phi_minus": (-eta * unit).tolist(),
            "phi_0": [0.0, 0.0, 0.0],
            "phi_plus": (eta * unit).tolist(),
        })
    direction = {
        "schema": "c1_true_q_directional_geometry_v1",
        "delta_primary": protocol["delta_primary"],
        "delta_secondary_goal": protocol["delta_secondary"],
        "finite_difference_uncertainty": "paired Bernoulli-difference normal 95% interval; n=4 is coarse",
        "nonflat_gate_passed": bool(nonflat_gate),
        "state_geometry": state_geometry,
        "pooled_symmetric_pair_tests": pooled,
        "state_pair_tests_with_discordance": state_pair_tests,
        "statistically_distinguishable_state_pairs_p_lt_0_05": significant_005,
        "statistically_distinguishable_state_pairs_p_lt_0_10": significant_010,
        "direction_validation_selection": selection,
    }
    (HERE / "directional_geometry.json").write_text(json.dumps(direction, indent=2) + "\n")
    selection_payload = {
        "selection_rule": protocol["stage_2_direction_validation"]["selection"],
        "fresh_flow_seeds": protocol["stage_2_direction_validation"]["fresh_flow_seeds"],
        "eta": eta,
        "states": selection,
    }
    (HERE / "raw/directional_selection.json").write_text(json.dumps(selection_payload, indent=2) + "\n")
    print(json.dumps({
        "nonflat_gate_passed": nonflat_gate,
        "classifications": {x["state_id"]: x["classification"] for x in state_geometry},
        "selected": [x["state_id"] for x in selection],
        "pooled_tests": pooled,
        "full_horizon_q_map_sha256": digest(HERE / "full_horizon_q_map.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
