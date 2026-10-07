"""Assemble required quantitative outputs after all frozen rollout stages finish."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
LOW = np.asarray([0.0, -0.53125, -0.125], dtype=np.float64)
HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
SPAN = HIGH - LOW


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eta_key(value) -> tuple[float, float, float]:
    return tuple(round(float(x), 12) for x in value)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["status"]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    os.replace(temporary, path)


def counts_result(rows: list[dict], expected: int | None = None) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    evaluated = len(rows)
    success = counts["success"]
    failures = evaluated - success
    complete = expected is not None and evaluated == expected
    if complete:
        status = "CONFIRMED_B63" if success >= 63 else "REJECTED_B63"
    elif failures >= 2:
        status = "REJECTED_B63_TWO_FAILURES"
    elif evaluated >= 16 and failures == 0:
        status = "SCREEN_PASS_UNCONFIRMED"
    else:
        status = "UNRESOLVED"
    return {
        "evaluated": evaluated,
        "success": success,
        "deadlock": counts["deadlock"],
        "timeout": counts["timeout"],
        "collision": counts["collision"],
        "execution_error": counts["execution_error"],
        "B_63_member": complete and success >= 63,
        "membership_status": status,
        "mean_J_def_success": float(np.mean([row["J_def"] for row in rows if row["outcome"] == "success"])) if success else math.nan,
    }


def load_cached_cells(selected: set[str] | None = None) -> dict[tuple[str, tuple], dict]:
    cache = {}
    for path in (
        ROOT / "diagnostics/gphi_training_dataset_v3/oracle_search_results.jsonl",
        ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1/startup_oracle_search_results.jsonl",
    ):
        for row in read_jsonl(path):
            if selected is not None and row["state_id"] not in selected:
                continue
            for cell in row["cells"]:
                success = int(cell.get("success", cell.get("counts", {}).get("success", 0)))
                evaluated = int(cell["evaluated"])
                failures = evaluated - success
                status = "CONFIRMED_B63" if bool(cell["B_63_member"]) else "REJECTED_B63_TWO_FAILURES" if failures >= 2 else "UNRESOLVED"
                cache[(row["state_id"], eta_key(cell["eta"]))] = {
                    "state_id": row["state_id"], "eta": list(cell["eta"]), "evaluated": evaluated,
                    "success": success, "deadlock": int(cell.get("deadlock", cell.get("counts", {}).get("deadlock", 0))),
                    "timeout": int(cell.get("timeout", cell.get("counts", {}).get("timeout", 0))),
                    "collision": int(cell.get("collision", cell.get("counts", {}).get("collision", 0))),
                    "execution_error": int(cell.get("execution_error", 0)), "B_63_member": bool(cell["B_63_member"]),
                    "membership_status": status, "mean_J_def_success": cell.get("mean_J_def_success"),
                    "evidence_source": str(path), "evidence_sha256": sha(path),
                }
    if selected:
        strict_dir = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/raw/robust"
        for state_id in selected:
            path = strict_dir / f"{state_id}.jsonl"
            if not path.exists():
                continue
            grouped = defaultdict(list)
            for row in read_jsonl(path):
                if "eta" in row:
                    grouped[eta_key(row["eta"])].append(row)
            for eta, rows in grouped.items():
                result = counts_result(rows, 64)
                cache[(state_id, eta)] = {"state_id": state_id, "eta": list(eta), **result,
                    "evidence_source": str(path), "evidence_sha256": sha(path)}
    return cache


def result_index(path: Path, expected: int | None) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["arm_id"]].append(row)
    return {arm: counts_result(rows, expected) for arm, rows in grouped.items()}


def make_cross_eta() -> dict:
    pairs = {int(row["priority_rank"]): row for row in csv.DictReader((HERE / "stage_bc_work/cross_eta_candidate_pairs.csv").open())}
    plan = json.loads((HERE / "cross_eta_incremental_plan_v4.json").read_text())
    raw_path = HERE / "raw/cross_eta_incremental/records.jsonl"
    simulated = result_index(raw_path, 64)
    requested = {row["requested_arm_id"]: simulated[row["simulation_arm_id"]] for row in plan["requested_new_assignments"]}
    reused = {row["arm_id"]: row for row in plan["reused_arms"]}
    def lookup(arm_id: str) -> dict:
        return reused.get(arm_id, requested[arm_id] if arm_id in requested else {})
    output = []
    interpolation = []
    for rank, pair in sorted(pairs.items()):
        for side in ("I", "J"):
            cross = lookup(f"PAIR{rank:02d}__{side}__CROSS")
            midpoint = lookup(f"PAIR{rank:02d}__{side}__MIDPOINT")
            row = {
                "pair_rank": rank, "state_side": side.lower(), "state_id": pair[f"state_{side.lower()}"],
                "normalized_h_l2": pair["normalized_h_l2"], "eta_target_jump": pair["eta_coordinate_normalized_l2"],
                "zero_active_crossing": pair["zero_active_crossing"],
                "cross_evaluated": cross.get("evaluated"), "cross_success": cross.get("success"),
                "cross_membership_status": cross.get("membership_status", "CONFIRMED_B63" if cross.get("B_63_member") else "REJECTED_B63_TWO_FAILURES"),
                "cross_B63": bool(cross.get("B_63_member", False)),
                "midpoint_evaluated": midpoint.get("evaluated"), "midpoint_success": midpoint.get("success"),
                "midpoint_membership_status": midpoint.get("membership_status"),
                "midpoint_B63": bool(midpoint.get("B_63_member", False)),
                "self_canonical_B63": True,
            }
            output.append(row)
            interpolation.append({"stage": "D_close_pair_midpoint", **row})
    atomic_csv(HERE / "cross_eta_transfer.csv", output)
    return {"rows": output, "interpolation": interpolation, "raw_sha256": sha(raw_path)}


def make_basin() -> dict:
    subset = json.loads((HERE / "basin_subset_manifest.json").read_text())
    state_rows = {row["state_id"]: row for row in subset["selected_states"]}
    selected = set(state_rows)
    cache = load_cached_cells(selected)
    eta_plan = json.loads((HERE / "basin_eta_hat_plan.json").read_text())
    eta_results = result_index(HERE / "raw/basin_eta_hat/records.jsonl", 64)
    interp_plan = json.loads((HERE / "basin_interpolation_screen_plan.json").read_text())
    screen_grouped = defaultdict(list)
    for row in read_jsonl(HERE / "raw/basin_interp_screen/records.jsonl"):
        screen_grouped[row["arm_id"]].append(row)
    promotion_path = HERE / "basin_interpolation_promotion_plan.json"
    promotion_map = {}
    promotion_grouped = defaultdict(list)
    if promotion_path.exists():
        promotion_plan = json.loads(promotion_path.read_text())
        promotion_map = {
            (row["state_id"], row["kind"], row["alpha"]): row["promotion_arm_id"]
            for row in promotion_plan["assignments"]
        }
        promotion_records = HERE / "raw/basin_interp_promotion/records.jsonl"
        if promotion_records.exists():
            for row in read_jsonl(promotion_records):
                promotion_grouped[row["arm_id"]].append(row)
    interp_by_request = []
    for request in interp_plan["requested_new_assignments"]:
        rows = list(screen_grouped[request["simulation_arm_id"]])
        promotion_arm = promotion_map.get((request["state_id"], request["kind"], request["alpha"]))
        if promotion_arm is not None:
            rows.extend(promotion_grouped[promotion_arm])
        interp_by_request.append(({**request, "promotion_arm_id": promotion_arm}, counts_result(rows, 64)))
    cloud = []
    membership = []
    robust_points = defaultdict(set)
    for (_, _), cell in cache.items():
        row = {"candidate_kind": "original_oracle_cache", **cell}
        cloud.append(row); membership.append(row)
        if cell["B_63_member"]:
            robust_points[cell["state_id"]].add(eta_key(cell["eta"]))
    eta_hat_by_state = {}
    for arm in eta_plan["arms"]:
        result = eta_results[arm["arm_id"]]
        eta_hat_by_state[arm["state_id"]] = (arm, result)
        row = {"candidate_kind": "fixed_current_G_eta", "state_id": arm["state_id"], "eta": arm["eta"], **result,
               "evidence_source": str(HERE / "raw/basin_eta_hat/records.jsonl"), "evidence_sha256": sha(HERE / "raw/basin_eta_hat/records.jsonl")}
        cloud.append(row); membership.append(row)
        if result["B_63_member"]:
            robust_points[arm["state_id"]].add(eta_key(arm["eta"]))
    interpolation_rows = []
    for request, result in interp_by_request:
        arm = next(value for value in interp_plan["arms"] if value["arm_id"] == request["simulation_arm_id"])
        row = {"candidate_kind": request["kind"], "state_id": request["state_id"], "eta": arm["eta"], "alpha": request["alpha"], **result,
               "evidence_source": str(HERE / "raw/basin_interp_screen/records.jsonl"), "evidence_sha256": sha(HERE / "raw/basin_interp_screen/records.jsonl"),
               "promotion_arm_id": request.get("promotion_arm_id"),
               "promotion_records_sha256": sha(HERE / "raw/basin_interp_promotion/records.jsonl") if request.get("promotion_arm_id") and (HERE / "raw/basin_interp_promotion/records.jsonl").exists() else None}
        cloud.append(row); membership.append(row); interpolation_rows.append({"stage": "E_uniform_subset_farthest_pair", **row})
    def flatten(row):
        value = dict(row); eta = value.pop("eta", [math.nan] * 3)
        value.update({"eta1": eta[0], "eta2": eta[1], "eta3": eta[2]})
        return value
    atomic_csv(HERE / "basin_candidate_cloud.csv", [flatten(row) for row in cloud])
    atomic_csv(HERE / "robust_basin_membership.csv", [flatten(row) for row in membership])
    components = []
    mse = []
    for state_id, state in state_rows.items():
        points = sorted(robust_points[state_id])
        arm, prediction = eta_hat_by_state[state_id]
        canonical = np.asarray(state["canonical_eta"], dtype=np.float64)
        learned = np.asarray(arm["eta"], dtype=np.float64)
        distances = [float(np.linalg.norm((learned - np.asarray(point)) / SPAN)) for point in points]
        screens = [result for request, result in interp_by_request if request["state_id"] == state_id and request["kind"] == "interpolation"]
        screen_success = sum(result["success"] for result in screens)
        screen_total = sum(result["evaluated"] for result in screens)
        confirmed_interpolants = sum(result["B_63_member"] for result in screens)
        confirmed_farthest_edge = len(screens) == 3 and confirmed_interpolants == 3
        graph_components = max(1, len(points) - int(confirmed_farthest_edge)) if points else 0
        components.append({
            "state_id": state_id, "category": state["category"], "canonical_eta_in_basin": eta_key(canonical) in set(points),
            "zero_in_basin": (0.0, 0.0, 0.0) in set(points), "learned_eta_in_basin": prediction["B_63_member"],
            "confirmed_robust_point_count": len(points), "observed_graph_component_count_under_confirmed_edges": graph_components,
            "interpolation_screen_success": screen_success, "interpolation_screen_evaluated": screen_total,
            "confirmed_interpolation_points": confirmed_interpolants,
            "farthest_endpoint_edge_confirmed": confirmed_farthest_edge,
            "topology_resolution": "PARTIAL_ONE_FARTHEST_EDGE_CONFIRMED" if confirmed_farthest_edge else "UNRESOLVED_SPARSE_NO_CONFIRMED_INTERPOLATION_EDGE",
            "empirical_topology_class": "UNRESOLVED_SPARSE",
            "sobol_omitted_due_to_15000_rollout_cap": True,
        })
        mse.append({
            "state_id": state_id, "category": state["category"],
            "eta_hat_to_canonical_physical_L2": float(np.linalg.norm(learned - canonical)),
            "eta_hat_to_canonical_normalized_L2": float(np.linalg.norm((learned - canonical) / SPAN)),
            "eta_hat_success": prediction["success"], "eta_hat_Q": prediction["success"] / 64.0,
            "eta_hat_B63": prediction["B_63_member"],
            "nearest_confirmed_basin_distance_normalized": min(distances) if distances else math.nan,
        })
    atomic_csv(HERE / "empirical_basin_components.csv", components)
    atomic_csv(HERE / "mse_surrogate_audit.csv", mse)
    l2 = np.asarray([row["eta_hat_to_canonical_normalized_L2"] for row in mse])
    q = np.asarray([row["eta_hat_Q"] for row in mse])
    nearest = np.asarray([row["nearest_confirmed_basin_distance_normalized"] for row in mse])
    return {
        "interpolation": interpolation_rows, "components": components, "mse": mse,
        "eta_l2_vs_q_spearman": float(spearmanr(l2, q).statistic),
        "nearest_basin_distance_vs_q_spearman": float(spearmanr(nearest, q, nan_policy="omit").statistic),
    }


def make_fresh() -> dict:
    zero_rows = list(csv.DictReader((HERE / "fresh32_zero_results.csv").open()))
    pred_rows = list(csv.DictReader((HERE / "fresh32_prediction_results.csv").open()))
    zero = {int(row["episode_index"]): row for row in zero_rows}
    pred = {int(row["episode_index"]): row for row in pred_rows}
    plan = json.loads((HERE / "fresh32_oracle_plan.json").read_text())
    oracle_raw = read_jsonl(HERE / "raw/fresh32_oracle/records.jsonl") if plan["arms"] else []
    grouped = defaultdict(list)
    for row in oracle_raw:
        grouped[row["arm_id"]].append(row)
    arm_results = {arm["arm_id"]: counts_result(grouped.get(arm["arm_id"], []), 64) for arm in plan["arms"]}
    arms_by_episode = defaultdict(list)
    for arm in plan["arms"]:
        arms_by_episode[int(arm["episode_index"])].append((arm, arm_results[arm["arm_id"]]))
    oracle_summary = []
    decomposition = []
    taxonomy = []
    for episode in sorted(zero):
        z = zero[episode]; p = pred[episode]
        zero_b63 = z["B_63_member"].lower() == "true"
        pred_b63 = p["B_63_member"].lower() == "true"
        active = [(arm, result) for arm, result in arms_by_episode[episode] if result["B_63_member"]]
        if zero_b63:
            category = "ZERO_SUFFICIENT"; oracle_eta = [0.0, 0.0, 0.0]; oracle_success = int(z["success"]); oracle_q = float(z["success_rate"]); search_complete = True
        elif active:
            best = min(active, key=lambda item: (item[1]["mean_J_def_success"], tuple(item[0]["eta"])))
            category = "ACTIVE_ORACLE_NEEDED"; oracle_eta = best[0]["eta"]; oracle_success = best[1]["success"]; oracle_q = best[1]["success"] / 64.0; search_complete = True
        else:
            category = "NO_ROBUST_ETA_FOUND"; oracle_eta = [math.nan] * 3; oracle_success = 0; oracle_q = 0.0
            search_complete = all(
                result["membership_status"] in {
                    "CONFIRMED_B63", "REJECTED_B63", "REJECTED_B63_TWO_FAILURES"
                }
                for _, result in arms_by_episode[episode]
            )
        oracle_summary.append({
            "episode_index": episode, "root_cause_class": category, "zero_success": int(z["success"]),
            "active_candidate_arms": len(arms_by_episode[episode]), "active_B63_candidates": len(active),
            "oracle_eta1": oracle_eta[0], "oracle_eta2": oracle_eta[1], "oracle_eta3": oracle_eta[2],
            "oracle_success": oracle_success, "oracle_Q": oracle_q, "search_complete_within_frozen_domain": search_complete,
        })
        eta = np.asarray([float(p["eta1"]), float(p["eta2"]), float(p["eta3"])])
        zero_distance = float(np.linalg.norm(eta / SPAN))
        if category == "ZERO_SUFFICIENT":
            error = "ZERO_BUT_HARMLESS_ACTIVE" if pred_b63 else "ZERO_FALSE_ACTIVATION"
        elif category == "ACTIVE_ORACLE_NEEDED":
            if pred_b63:
                error = "ACTIVE_NONCANONICAL_BUT_SUCCESSFUL"
            elif zero_distance <= 0.1:
                error = "ACTIVE_MISS_TO_NEAR_ZERO"
            else:
                error = "ACTIVE_WRONG_BASIN_OR_OUTSIDE_BASIN"
        else:
            error = "OTHER_UNRESOLVED"
        if p["eta_clipped"].lower() == "true" and not pred_b63:
            error = "CLIPPING_ASSOCIATED_FAILURE"
        decomposition.append({
            "episode_index": episode, "oracle_root_cause_class": category, "zero_B63": zero_b63,
            "prediction_B63": pred_b63, "prediction_success": int(p["success"]), "prediction_Q": float(p["success_rate"]),
            "prediction_behavior": "prediction_robustly_succeeds" if pred_b63 else "prediction_fails_robustness",
            "error_taxonomy": error,
        })
        taxonomy.append({"scope": "fresh32_step0", "episode_index": episode, "taxonomy": error, "count": 1})
    atomic_csv(HERE / "fresh32_oracle_search.csv", oracle_summary)
    atomic_csv(HERE / "fresh32_failure_decomposition.csv", decomposition)
    atomic_csv(HERE / "current_g_eta_error_taxonomy.csv", taxonomy)
    return {"oracle": oracle_summary, "decomposition": decomposition, "taxonomy_counts": dict(Counter(row["taxonomy"] for row in taxonomy)),
            "oracle_raw_sha256": sha(HERE / "raw/fresh32_oracle/records.jsonl") if oracle_raw else None}


def main() -> None:
    cross = make_cross_eta()
    basin = make_basin()
    all_interpolation = cross["interpolation"] + basin["interpolation"]
    atomic_csv(HERE / "interpolation_tests.csv", all_interpolation)
    midpoint = [row for row in all_interpolation if row["stage"] == "D_close_pair_midpoint"]
    average_failure = []
    for row in midpoint:
        average_failure.append({
            "stage": "D_close_pair_cross_transfer",
            "pair_rank": row["pair_rank"], "state_id": row["state_id"], "successful_endpoint_pair": row["cross_B63"],
            "midpoint_evaluated": row["midpoint_evaluated"], "midpoint_success": row["midpoint_success"],
            "midpoint_membership_status": row["midpoint_membership_status"],
            "confirmed_failing_midpoint": bool(row["cross_B63"]) and str(row["midpoint_membership_status"]).startswith("REJECTED"),
        })
    for row in basin["interpolation"]:
        if row.get("alpha") != 0.5:
            continue
        average_failure.append({
            "stage": "E_uniform_basin_farthest_confirmed_endpoints",
            "pair_rank": None, "state_id": row["state_id"], "successful_endpoint_pair": True,
            "midpoint_evaluated": row["evaluated"], "midpoint_success": row["success"],
            "midpoint_membership_status": row["membership_status"],
            "confirmed_failing_midpoint": str(row["membership_status"]).startswith("REJECTED"),
        })
    atomic_csv(HERE / "interpolation_average_failure.csv", average_failure)
    fresh = make_fresh()
    summary = {
        "schema": "direct_eta_quantitative_finalize_v1",
        "cross_eta_assignments": len(cross["rows"]),
        "cross_eta_B63": sum(row["cross_B63"] for row in cross["rows"]),
        "cross_eta_rejected": sum(str(row["cross_membership_status"]).startswith("REJECTED") for row in cross["rows"]),
        "midpoint_rejected": sum(row["confirmed_failing_midpoint"] for row in average_failure),
        "midpoint_screen_pass_unconfirmed": sum(row["midpoint_membership_status"] == "SCREEN_PASS_UNCONFIRMED" for row in average_failure),
        "basin_eta_hat_B63": sum(row["eta_hat_B63"] for row in basin["mse"]),
        "eta_l2_vs_Q_spearman": basin["eta_l2_vs_q_spearman"],
        "nearest_basin_distance_vs_Q_spearman": basin["nearest_basin_distance_vs_q_spearman"],
        "fresh_root_cause_counts": dict(Counter(row["root_cause_class"] for row in fresh["oracle"])),
        "fresh_prediction_B63": sum(row["prediction_B63"] for row in fresh["decomposition"]),
        "fresh_error_taxonomy": fresh["taxonomy_counts"],
    }
    (HERE / "quantitative_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
