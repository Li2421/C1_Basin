"""Produce the remaining descriptive true-Q geometry diagnostics."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from diagnostics.true_q_geometry.analyze_q_map import paired_test, wilson
from diagnostics.true_q_geometry.run_q_map import HERE


QRAW = HERE / "raw/q_map"
DRAW = HERE / "raw/direction_validation"


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return {"rho": None, "p": None, "n": int(np.sum(mask))}
    result = spearmanr(x[mask], y[mask])
    return {"rho": float(result.statistic), "p": float(result.pvalue), "n": int(np.sum(mask))}


def load_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def pair_trajectory_metrics(a: dict, b: dict, dt: float) -> dict:
    n = min(len(a["event"]), len(b["event"]))
    gdiff = np.linalg.norm(a["g"][:n] - b["g"][:n], axis=-1)
    wdiff = np.linalg.norm(a["w"][:n] - b["w"][:n], axis=-1)
    udiff = np.linalg.norm(a["u_exec"][:n] - b["u_exec"][:n], axis=-1)
    zdiff = np.linalg.norm(a["positions_after"][:n] - b["positions_after"][:n], axis=-1)
    clip_a = np.linalg.norm(a["w"][:n] - a["u_exec"][:n], axis=(1, 2))
    clip_b = np.linalg.norm(b["w"][:n] - b["u_exec"][:n], axis=(1, 2))
    active = np.any(a["second_active"][:n] != b["second_active"][:n], axis=1)
    action_scalar = np.linalg.norm(a["u_exec"][:n] - b["u_exec"][:n], axis=(1, 2))
    state_scalar = np.linalg.norm(a["positions_after"][:n] - b["positions_after"][:n], axis=(1, 2))
    def first(mask):
        indices = np.flatnonzero(mask)
        return None if not len(indices) else float((indices[0] + 1) * dt)
    return {
        "overlap_steps": n,
        "pre_projection_g_rms": float(np.sqrt(np.mean(gdiff ** 2))),
        "pre_projection_w_rms": float(np.sqrt(np.mean(wdiff ** 2))),
        "executed_action_rms": float(np.sqrt(np.mean(udiff ** 2))),
        "state_trajectory_rms": float(np.sqrt(np.mean(zdiff ** 2))),
        "active_set_disagreement_fraction": float(np.mean(active)),
        "clipping_fraction_a": float(np.mean(clip_a > 1e-8)),
        "clipping_fraction_b": float(np.mean(clip_b > 1e-8)),
        "first_executed_action_difference_gt_1e-3_seconds": first(action_scalar > 1e-3),
        "first_state_difference_gt_1e-3_seconds": first(state_scalar > 1e-3),
        "executable_collapse": bool(np.sqrt(np.mean(udiff ** 2)) < 1e-8),
    }


def mean_dict(rows, fields):
    return {field: float(np.mean([row[field] for row in rows])) for field in fields}


def main():
    qmap = json.loads((HERE / "full_horizon_q_map.json").read_text())
    direction = json.loads((HERE / "directional_geometry.json").read_text())
    qmanifest = json.loads((QRAW / "manifest.json").read_text())
    dmanifest = json.loads((DRAW / "manifest.json").read_text())
    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    dt = qmap["dt"]

    # Fresh directional validation.
    dgroup = defaultdict(list)
    for record in dmanifest["records"]:
        dgroup[(record["state_id"], record["policy_name"])].append(record)
    validation = []
    for selected in direction["direction_validation_selection"]:
        sid = selected["state_id"]
        entry = {"state_id": sid, "g_TRUE": selected["g_TRUE"], "g_TRUE_norm": selected["g_TRUE_norm"]}
        vectors = {}
        for name in ("phi_minus", "phi_0", "phi_plus"):
            records = sorted(dgroup[(sid, name)], key=lambda x: x["flow_seed"])
            counts = Counter(x["outcome"] for x in records)
            dvec = [int(x["outcome"] == "deadlock") for x in records]
            vectors[name] = dvec
            entry[name] = {
                "phi": selected[name], "n": len(records), "counts": dict(counts),
                "Q_D": float(np.mean(dvec)), "Q_D_wilson_95": wilson(sum(dvec), len(dvec)),
                "Q_success": counts["success"] / len(records),
                "Q_timeout": counts["timeout"] / len(records),
                "Q_collision": counts["collision"] / len(records),
            }
        entry["minus_vs_plus"] = paired_test(vectors["phi_minus"], vectors["phi_plus"])
        entry["minus_vs_baseline"] = paired_test(vectors["phi_minus"], vectors["phi_0"])
        entry["predeclared_success_rule_met"] = bool(
            entry["phi_minus"]["Q_D"] < entry["phi_plus"]["Q_D"]
            and entry["minus_vs_plus"]["two_sided_exact_mcnemar_p"] < 0.10
            and entry["phi_minus"]["Q_D"] <= entry["phi_0"]["Q_D"] <= entry["phi_plus"]["Q_D"]
        )
        validation.append(entry)
    direction["fresh_direction_validation"] = {
        "seeds": protocol["stage_2_direction_validation"]["fresh_flow_seeds"],
        "new_continuations": dmanifest["new_continuations"],
        "actual_new_physical_steps": dmanifest["actual_new_physical_steps"],
        "states": validation,
        "states_passing_predeclared_rule": sum(x["predeclared_success_rule_met"] for x in validation),
        "interpretation": "The true-Q direction replicated, but risk reduction was timeout substitution rather than task success.",
    }
    (HERE / "directional_geometry.json").write_text(json.dumps(direction, indent=2) + "\n")

    qrecords = {(x["state_id"], x["probe_name"], x["flow_seed"]): x for x in qmanifest["records"]}
    qarrays = {}
    def qa(key):
        if key not in qarrays:
            record = qrecords[key]
            qarrays[key] = load_npz(QRAW / record["relative_path"])
        return qarrays[key]
    drecords = {(x["state_id"], x["policy_name"], x["flow_seed"]): x for x in dmanifest["records"]}
    darrays = {}
    def da(key):
        if key not in darrays:
            record = drecords[key]
            darrays[key] = load_npz(DRAW / record["relative_path"])
        return darrays[key]

    qlookup = {(s["state_id"], p["probe_name"]): p["Q_D"] for s in qmap["states"] for p in s["policies"]}
    projection_pairs = []
    symmetric = (("goal_m1", "goal_p1"), ("safe_m1", "safe_p1"), ("rel_m1", "rel_p1"), ("goal_m2", "goal_p2"))
    seeds = [protocol["stage_1"]["cached_common_seed"]] + protocol["stage_1"]["new_independent_flow_seeds"]
    for state in qmap["states"]:
        sid = state["state_id"]
        for a, b in symmetric:
            qdiff = qlookup[(sid, b)] - qlookup[(sid, a)]
            if abs(qdiff) < 0.5:
                continue
            metrics = [pair_trajectory_metrics(qa((sid, a, seed)), qa((sid, b, seed)), dt) for seed in seeds]
            projection_pairs.append({
                "source": "local_probe", "state_id": sid, "policy_a": a, "policy_b": b,
                "Q_D_b_minus_a": qdiff,
                **mean_dict(metrics, (
                    "pre_projection_g_rms", "pre_projection_w_rms", "executed_action_rms",
                    "state_trajectory_rms", "active_set_disagreement_fraction",
                    "clipping_fraction_a", "clipping_fraction_b",
                )),
                "collapsed_seed_count": sum(x["executable_collapse"] for x in metrics),
                "seed_count": len(metrics),
            })
    for selected in validation:
        sid = selected["state_id"]
        metrics = [pair_trajectory_metrics(da((sid, "phi_minus", seed)), da((sid, "phi_plus", seed)), dt)
                   for seed in protocol["stage_2_direction_validation"]["fresh_flow_seeds"]]
        projection_pairs.append({
            "source": "fresh_direction_validation", "state_id": sid,
            "policy_a": "phi_minus", "policy_b": "phi_plus",
            "Q_D_b_minus_a": selected["phi_plus"]["Q_D"] - selected["phi_minus"]["Q_D"],
            **mean_dict(metrics, (
                "pre_projection_g_rms", "pre_projection_w_rms", "executed_action_rms",
                "state_trajectory_rms", "active_set_disagreement_fraction",
                "clipping_fraction_a", "clipping_fraction_b",
            )),
            "collapsed_seed_count": sum(x["executable_collapse"] for x in metrics),
            "seed_count": len(metrics),
        })
    total_pairs = sum(x["seed_count"] for x in projection_pairs)
    collapsed = sum(x["collapsed_seed_count"] for x in projection_pairs)
    projection = {
        "schema": "c1_true_q_projection_role_v1",
        "comparison_scope": "all symmetric local probes with |Delta Q_D|>=0.5 plus every fresh direction-validation pair",
        "policy_pairs": projection_pairs,
        "paired_trajectory_count": total_pairs,
        "executable_collapse_count": collapsed,
        "executable_collapse_fraction": collapsed / total_pairs if total_pairs else None,
        "role": "PRESERVED_BY_PROJECTION" if collapsed == 0 else "PARTIALLY_DESTROYED_BY_PROJECTION",
        "interpretation": "Outcome-relevant policy distinctions survive the second hard projection; active-set switching accompanies the basin changes.",
    }
    (HERE / "projection_role.json").write_text(json.dumps(projection, indent=2) + "\n")

    # Trace-level and state-policy-level feature analysis.
    initial_by_state = {s["state_id"]: s["initial_features"] for s in qmap["states"]}
    trace_rows = []
    policy_rows = []
    for state in qmap["states"]:
        sid = state["state_id"]
        initial = initial_by_state[sid]
        for policy in state["policies"]:
            local_rows = []
            for trace in policy["trajectory_summaries"]:
                row = {
                    "state_id": sid, "probe_name": policy["probe_name"], "outcome": trace["outcome"],
                    "deadlock": int(trace["outcome"] == "deadlock"),
                    "initial_sum_task_error": initial["sum_task_error"],
                    "initial_recent_progress_sum": initial["recent_progress_sum"],
                    "initial_relative_order": initial["relative_longitudinal_order"],
                    "initial_inter_agent_distance": initial["inter_agent_distance"],
                    "initial_joint_speed_max": initial["joint_speed_max"],
                    "initial_strict_timer_age": initial["strict_timer_age"],
                    "time_to_go_seconds": initial["time_to_go_seconds"],
                    "progress_20": trace["horizon_20"]["task_progress_sum"],
                    "progress_100": trace["horizon_100"]["task_progress_sum"],
                    "order_change_20": trace["horizon_20"]["longitudinal_order_change"],
                    "order_change_100": trace["horizon_100"]["longitudinal_order_change"],
                    "state_displacement_100": trace["horizon_100"]["state_displacement"],
                    "projection_residual": trace["mean_second_projection_residual"],
                    "projection_clipping_fraction": trace["second_projection_clipping_fraction"],
                    "active_switch_fraction": trace["second_active_switch_fraction"],
                    "cumulative_correction_norm": trace["cumulative_correction_norm"],
                }
                trace_rows.append(row)
                local_rows.append(row)
            numeric = [k for k in local_rows[0] if k not in ("state_id", "probe_name", "outcome", "deadlock")]
            policy_rows.append({
                "state_id": sid, "probe_name": policy["probe_name"], "Q_D": policy["Q_D"],
                **{name: float(np.mean([r[name] for r in local_rows])) for name in numeric},
            })
    features = [k for k in trace_rows[0] if k not in ("state_id", "probe_name", "outcome", "deadlock")]
    correlations = {name: safe_spearman([r[name] for r in trace_rows], [r["deadlock"] for r in trace_rows]) for name in features}
    policy_correlations = {name: safe_spearman([r[name] for r in policy_rows], [r["Q_D"] for r in policy_rows]) for name in features}
    conditional = {}
    for outcome in ("deadlock", "success", "timeout", "collision"):
        rows = [r for r in trace_rows if r["outcome"] == outcome]
        conditional[outcome] = {
            "n": len(rows),
            "means": {name: (float(np.mean([r[name] for r in rows])) if rows else None) for name in features},
        }

    # Coarse, non-fitted sufficiency checks.
    progress = np.asarray([r["progress_100"] for r in policy_rows])
    qvals = np.asarray([r["Q_D"] for r in policy_rows])
    def quartile_bins(values):
        values = np.asarray(values, dtype=float)
        edges = np.unique(np.quantile(values, [0, .25, .5, .75, 1]))
        bins = []
        for i in range(len(edges) - 1):
            mask = (values >= edges[i]) & (values <= edges[i + 1] if i == len(edges) - 2 else values < edges[i + 1])
            if np.any(mask):
                bins.append({"range": [float(edges[i]), float(edges[i + 1])], "n": int(np.sum(mask)),
                             "mean_Q_D": float(np.mean(qvals[mask])), "Q_D_range": float(np.ptp(qvals[mask]))})
        return bins
    bins_1d = quartile_bins(progress)
    displacement = np.asarray([r["state_displacement_100"] for r in policy_rows])
    displacement_by_state = {}
    for sid in sorted({r["state_id"] for r in policy_rows}):
        rows = [r for r in policy_rows if r["state_id"] == sid]
        displacement_by_state[sid] = safe_spearman(
            [r["state_displacement_100"] for r in rows], [r["Q_D"] for r in rows]
        )
    med_progress = float(np.median(progress))
    order = np.asarray([r["order_change_100"] for r in policy_rows])
    residual = np.asarray([r["projection_residual"] for r in policy_rows])
    def median_cells(second, second_name):
        m2 = float(np.median(second))
        cells = []
        for hi1 in (False, True):
            for hi2 in (False, True):
                mask = ((progress >= med_progress) == hi1) & ((second >= m2) == hi2)
                cells.append({
                    "progress_high": hi1, f"{second_name}_high": hi2, "n": int(np.sum(mask)),
                    "mean_Q_D": float(np.mean(qvals[mask])) if np.any(mask) else None,
                    "Q_D_range": float(np.ptp(qvals[mask])) if np.any(mask) else None,
                })
        return {"progress_median": med_progress, f"{second_name}_median": m2, "cells": cells}
    feature_analysis = {
        "schema": "c1_true_q_feature_analysis_v1",
        "feature_policy": "small predeclared physical set; descriptive statistics only; no fitted predictor",
        "trace_count": len(trace_rows), "state_policy_count": len(policy_rows),
        "trace_level_spearman_with_deadlock": correlations,
        "state_policy_spearman_with_Q_D": policy_correlations,
        "conditional_means_by_outcome": conditional,
        "low_dimensional_checks": {
            "progress_100_quartiles": bins_1d,
            "state_displacement_100_quartiles": quartile_bins(displacement),
            "state_displacement_within_state_rank": displacement_by_state,
            "progress_plus_order": median_cells(order, "order_change_100"),
            "progress_plus_projection": median_cells(residual, "projection_residual"),
        },
        "summary": "100-step persistent displacement strongly orders strict-deadlock Q_D at the extremes, but middle bins overlap and progress has state-dependent sign because low-Q timeout trajectories can make poor task progress.",
        "candidate_for_future_R_risk": {
            "variable": "100-step persistent state displacement",
            "status": "PROMISING_ORDERING_COMPONENT_NOT_SUFFICIENT",
            "caveat": "It orders deadlock versus non-deadlock, including timeout; it does not establish true escape or task success.",
        },
        "optional_diagnostic_fit_run": False,
        "rows_for_plotting": policy_rows,
    }
    (HERE / "feature_analysis.json").write_text(json.dumps(feature_analysis, indent=2) + "\n")

    # Escape, delay, and timeout substitution relative to the same-state/seed p0 continuation.
    class_counts = Counter()
    adverse = Counter()
    delayed_examples = []
    timeout_examples = []
    for state in qmap["states"]:
        sid = state["state_id"]
        for seed in seeds:
            base = qa((sid, "p0", seed))
            base_out = str(base["outcome"])
            for probe in protocol["phi_probes"]:
                if probe == "p0":
                    continue
                other = qa((sid, probe, seed))
                out = str(other["outcome"])
                metrics = pair_trajectory_metrics(base, other, dt)
                item = {
                    "state_id": sid, "flow_seed": seed, "policy": probe,
                    "baseline_outcome": base_out, "policy_outcome": out,
                    "baseline_steps": len(base["event"]), "policy_steps": len(other["event"]),
                    **metrics,
                }
                if base_out == "deadlock":
                    if out == "success":
                        label = "TRUE_ESCAPE"
                    elif out == "timeout":
                        label = "TIMEOUT"
                        timeout_examples.append(item)
                    elif out == "deadlock" and len(other["event"]) > len(base["event"]):
                        label = "DELAYED_DEADLOCK"
                        delayed_examples.append(item)
                    else:
                        label = "NO_EFFECT"
                    class_counts[label] += 1
                elif out != base_out:
                    adverse[f"{base_out}_TO_{out}"] += 1
    delayed_examples.sort(key=lambda x: x["policy_steps"] - x["baseline_steps"], reverse=True)
    timeout_examples.sort(key=lambda x: x["state_trajectory_rms"])
    escape = {
        "classification_counts_for_baseline_deadlock_pairs": dict(class_counts),
        "adverse_or_other_changes_from_non_deadlock_baselines": dict(adverse),
        "true_escape_observed": class_counts["TRUE_ESCAPE"] > 0,
        "representative_delayed_deadlock": delayed_examples[:5],
        "representative_timeout_substitution_with_smallest_overlap_state_rms": timeout_examples[:5],
        "finding": "No deadlock-starting p0 continuation became success. Low-Q changes were timeout substitution; other policies sometimes delayed strict deadlock without changing its eventual occurrence.",
    }
    (HERE / "raw/escape_delay_summary.json").write_text(json.dumps(escape, indent=2) + "\n")

    # Smoothness and temporal control window.
    smooth_states = []
    for item in direction["state_geometry"]:
        inner = item["finite_differences"]["goal"]["central_fd"]
        outer = item["goal_outer_central_fd"]
        relative_change = None if inner == 0 else abs(outer - inner) / abs(inner)
        if item["classification"] == "SATURATED":
            region = "FLAT/SATURATED"
        elif item["largest_adjacent_Q_D_jump"] >= 0.75:
            region = "DISCONTINUOUS/BASIN_BOUNDARY"
        elif relative_change is not None and relative_change <= 0.5:
            region = "LOCALLY_SMOOTH"
        else:
            region = "PIECEWISE_SMOOTH"
        smooth_states.append({
            "state_id": item["state_id"], "region": region,
            "primary_goal_fd": inner, "secondary_goal_fd": outer,
            "relative_delta_change": relative_change,
            "sign_stable": item["goal_delta_sign_stable"],
            "largest_adjacent_Q_D_jump": item["largest_adjacent_Q_D_jump"],
        })
    smoothness = {
        "schema": "c1_true_q_smoothness_audit_v1",
        "states": smooth_states,
        "counts": dict(Counter(x["region"] for x in smooth_states)),
        "active_set_evidence": {
            "mean_disagreement_on_outcome_relevant_pairs": float(np.mean([x["active_set_disagreement_fraction"] for x in projection_pairs])),
            "executable_collapse_fraction": projection["executable_collapse_fraction"],
        },
        "conclusion": "Useful directions exist, but plateaus and adjacent Q_D jumps dominate; a globally smooth scalar-gradient geometry is unsupported.",
    }
    (HERE / "smoothness_audit.json").write_text(json.dumps(smoothness, indent=2) + "\n")

    offsets = []
    for state in qmap["states"]:
        if not state["state_id"].startswith("D"):
            continue
        geom = next(x for x in direction["state_geometry"] if x["state_id"] == state["state_id"])
        policies = {p["probe_name"]: p for p in state["policies"]}
        low = min(policies.values(), key=lambda p: p["Q_D"])
        offsets.append({
            "state_id": state["state_id"], "offset_seconds": state["selection"]["nominal_offset_seconds"],
            "baseline_Q_D": policies["p0"]["Q_D"], "minimum_Q_D": low["Q_D"],
            "minimum_Q_policy": low["probe_name"], "minimum_Q_success": low["Q_success"],
            "minimum_Q_timeout": low["Q_timeout"], "Q_D_range": geom["Q_D_range"],
            "g_TRUE_norm": geom["g_TRUE_norm"], "classification": geom["classification"],
        })
    offsets.sort(key=lambda x: x["offset_seconds"], reverse=True)
    temporal = {
        "schema": "c1_true_q_temporal_control_window_v1",
        "deadlock_source_states": offsets,
        "earliest_tested_policy_sensitive_offset_seconds": max(x["offset_seconds"] for x in offsets if x["Q_D_range"] >= .5),
        "latest_tested_policy_sensitive_offset_seconds": min(x["offset_seconds"] for x in offsets if x["Q_D_range"] >= .5),
        "direction_strength_trend": "central-FD norm increases toward the latch in this selected sequence",
        "true_escape_window": None,
        "interpretation": "Strict-deadlock probability remains policy-sensitive through 1 s before the source deadlock, but no tested deadlock source achieved success; the apparent window is a deadlock-versus-timeout boundary.",
    }
    (HERE / "temporal_control_window.json").write_text(json.dumps(temporal, indent=2) + "\n")

    summary = {
        "decision": "CASE B — CONTROL GEOMETRY EXISTS BUT IS NONSMOOTH / MULTIMODAL",
        "new_inference_continuations": qmanifest["new_continuations"] + dmanifest["new_continuations"],
        "new_inference_physical_steps": qmanifest["actual_new_physical_steps"] + dmanifest["actual_new_physical_steps"],
        "direction_states_passing": direction["fresh_direction_validation"]["states_passing_predeclared_rule"],
        "projection_role": projection["role"],
        "escape": escape,
        "smoothness_counts": smoothness["counts"],
        "temporal": temporal,
    }
    (HERE / "raw/final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "decision": summary["decision"],
        "direction_states_passing": summary["direction_states_passing"],
        "projection_role": summary["projection_role"],
        "class_counts": dict(class_counts),
        "smoothness_counts": smoothness["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
