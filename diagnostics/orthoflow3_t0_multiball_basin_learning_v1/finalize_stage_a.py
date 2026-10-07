#!/usr/bin/env python3
"""Aggregate the frozen fixed-8 pilot, apply gates, and stop before training if needed."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_t0_multiball_basin_learning_v1"
STAGE = HERE / "stage_a"
AFF = np.array([0.875, 0.0, 0.375], dtype=float)
SCALE = np.array([0.75, 1.0, 0.75], dtype=float)
BASIS = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py"
BASIS_SHA = "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38"


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nt(eta: list[float] | np.ndarray) -> np.ndarray:
    return (np.asarray(eta, dtype=float) - AFF) / SCALE


def feasible_intersection(balls: list[dict]) -> tuple[bool, np.ndarray, float]:
    centers = np.asarray([b["center"] for b in balls], dtype=float)
    radii = np.asarray([b["radius"] for b in balls], dtype=float)
    if len(balls) == 1:
        return True, centers[0], 0.0
    # Necessary pairwise test is exact and cheaply removes almost every tuple here.
    for i, j in itertools.combinations(range(len(balls)), 2):
        if np.linalg.norm(centers[i] - centers[j]) > radii[i] + radii[j] + 1e-10:
            return False, centers.mean(axis=0), math.inf
    def objective(x: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(x - centers, axis=1) / radii))
    starts = [centers.mean(axis=0), *centers]
    best_value, best_point = math.inf, centers.mean(axis=0)
    for start in starts:
        result = minimize(objective, start, method="Powell",
                          options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 10000})
        if result.success and float(result.fun) < best_value:
            best_value, best_point = float(result.fun), np.asarray(result.x)
    return best_value <= 1.0 + 1e-8, best_point, best_value


def union_overlap_audit(per_state: dict[str, list[dict]], gamma: float = 0.8) -> dict:
    states = list(per_state)
    tested = 0
    pairwise_pruned = 0
    for size in range(len(states), 0, -1):
        for subset in itertools.combinations(states, size):
            choices = []
            for sid in subset:
                choices.append([
                    {"state_id": sid, "component": int(b["component"]),
                     "center": nt([float(b["c1"]), float(b["c2"]), float(b["c3"])]),
                     "radius": gamma * float(b["r_ball"])}
                    for b in per_state[sid]
                ])
            for combo in itertools.product(*choices):
                tested += 1
                # Record pruning separately while sharing the same deterministic solver.
                ok_pairs = all(
                    np.linalg.norm(combo[i]["center"] - combo[j]["center"])
                    <= combo[i]["radius"] + combo[j]["radius"] + 1e-10
                    for i, j in itertools.combinations(range(size), 2)
                )
                if not ok_pairs:
                    pairwise_pruned += 1
                    continue
                ok, point, ratio = feasible_intersection(list(combo))
                if ok:
                    return {
                        "retained_gamma": gamma,
                        "all_state_intersection": size == len(states),
                        "maximum_common_coverage_count": size,
                        "maximum_common_coverage_fraction": size / len(states),
                        "covered_states": list(subset),
                        "selected_components": [x["component"] for x in combo],
                        "common_eta_normalized": point.tolist(),
                        "common_eta_raw": (AFF + SCALE * point).tolist(),
                        "minimax_normalized_ratio": ratio,
                        "candidate_tuples_tested": tested,
                        "pairwise_pruned": pairwise_pruned,
                        "method": "descending exact state-subset/component enumeration; pairwise pruning; deterministic minimax ball-intersection solve",
                    }
    raise RuntimeError("at least one ball should be feasible")


def main() -> None:
    if sha(BASIS) != BASIS_SHA:
        raise RuntimeError("authoritative OrthoFlow3 hash changed")
    manifest = json.load(open(HERE / "fixed8_manifest.json"))
    states = [x["state_id"] for x in manifest["states"]]
    summaries = []
    per_state: dict[str, list[dict]] = {}
    aggregate = {
        "outer_center_promotions.csv": [],
        "component_balls.csv": [],
        "union_inside_validation.csv": [],
        "independent_coverage_audit.csv": [],
    }
    for sid in states:
        directory = STAGE / sid
        summaries.append(json.load(open(directory / "state_summary.json")))
        per_state[sid] = read_csv(directory / "component_balls.csv")
        for name in aggregate:
            aggregate[name].extend(read_csv(directory / name))
    for name, rows in aggregate.items():
        write_csv(HERE / name, rows)

    overlap = union_overlap_audit(per_state, 0.8)
    dump(HERE / "universal_solution_audit.json", overlap)

    usable = sum(bool(x["training_usable"]) for x in summaries)
    false_inclusions = sum(int(x["union_false_inclusions"]) for x in summaries)
    med_largest = statistics.median(float(x["largest_radius"]) for x in summaries)
    med_diameter = statistics.median(float(x["union_diameter"]) for x in summaries)
    coverages = [float(x["robust_coverage"]) for x in summaries if x["robust_coverage"] is not None]
    med_coverage = statistics.median(coverages)
    criteria = {
        "training_usable_at_least_6_of_8": {"observed": usable, "threshold": 6, "pass": usable >= 6},
        "zero_confirmed_union_false_inclusions": {"observed": false_inclusions, "threshold": 0, "pass": false_inclusions == 0},
        "median_largest_component_radius_at_least_0.15": {"observed": med_largest, "threshold": 0.15, "pass": med_largest >= 0.15},
        "median_union_diameter_at_least_0.30": {"observed": med_diameter, "threshold": 0.30, "pass": med_diameter >= 0.30},
        "median_independent_robust_coverage_at_least_0.50": {"observed": med_coverage, "threshold": 0.50, "pass": med_coverage >= 0.50},
        "no_eta_in_more_than_75pct_retained_sets": {
            "observed": overlap["maximum_common_coverage_fraction"], "threshold": 0.75,
            "pass": overlap["maximum_common_coverage_fraction"] <= 0.75,
        },
    }
    gate_pass = all(x["pass"] for x in criteria.values())
    gate = {
        "stage": "FIXED8_GEOMETRY_GATE",
        "pass": gate_pass,
        "criteria": criteria,
        "state_summaries": summaries,
        "decision": "PROCEED_DATASET" if gate_pass else "STOP_WITHOUT_TRAINING",
        "failed_criteria": [k for k, v in criteria.items() if not v["pass"]],
    }
    dump(HERE / "geometry_gate.json", gate)

    # Gate-stopped dataset/training artifacts explicitly preserve non-execution.
    dataset_rows = []
    rejected = []
    for x in summaries:
        dataset_rows.append({
            "state_id": x["state_id"], "split": "FIXED8_PILOT", "K": x["K"],
            "largest_radius": x["largest_radius"], "union_diameter": x["union_diameter"],
            "robust_coverage": x["robust_coverage"], "training_usable": x["training_usable"],
        })
        if not x["training_usable"]:
            rejected.append({"state_id": x["state_id"], "reason": "FIXED8_PER_STATE_TRAINING_ELIGIBILITY_FAILED"})
    write_csv(HERE / "verified_multiball_dataset.csv", dataset_rows)
    write_csv(HERE / "rejected_labels.csv", rejected, ["state_id", "reason"])
    dump(HERE / "t0_source_split.json", {
        "status": "NOT_BUILT_GEOMETRY_GATE_FAILED", "pilot_source_groups": manifest["source_groups"],
        "train": 0, "val": 0, "test": 0,
    })
    dataset_gate = {
        "pass": False, "status": "NOT_REACHED_GEOMETRY_GATE_FAILED",
        "training_authorized": False, "training_executed": False,
    }
    dump(HERE / "dataset_gate.json", dataset_gate)

    for family in ("g_center", "g_set"):
        directory = HERE / family
        directory.mkdir(exist_ok=True)
        dump(directory / "selected_checkpoint.json", {"status": "NOT_TRAINED_GEOMETRY_GATE_FAILED"})
        (directory / "checkpoint_sha256.txt").write_text("NOT_TRAINED_GEOMETRY_GATE_FAILED\n")
        write_csv(directory / "training_history.csv", [{"status": "NOT_TRAINED_GEOMETRY_GATE_FAILED"}])
        dump(directory / "seed_results.json", {"status": "NOT_TRAINED_GEOMETRY_GATE_FAILED"})
    dump(HERE / "common_model_config.json", {"status": "NOT_CREATED_GEOMETRY_GATE_FAILED"})
    dump(HERE / "g_set/collapse_diagnostic.json", {
        "status": "TRAINING_NOT_RUN", "pilot_geometric_max_common_coverage": overlap["maximum_common_coverage_fraction"],
    })
    for name in ("val_closedloop.csv", "test_predictions.csv", "test_64seed.csv",
                 "set_distance_vs_q.csv", "perturbation_robustness.csv", "fresh_wide_results.csv",
                 "fresh_wide_pairwise.csv", "robustness_deformation.csv"):
        write_csv(HERE / name, [{"status": "NOT_EXECUTED_GEOMETRY_GATE_FAILED"}])
    dump(HERE / "frozen_lowj_reference.json", {"status": "NOT_EVALUATED_GEOMETRY_GATE_FAILED"})
    dump(HERE / "fresh_wide_manifest.json", {"status": "NOT_CREATED_GEOMETRY_GATE_FAILED"})

    total_cont = sum(int(x["new_continuations"]) for x in summaries)
    total_steps = sum(int(x["new_physical_steps"]) for x in summaries)
    max_state_wall = max(float(x["wall_seconds"]) for x in summaries)
    sum_gpu_wall = sum(float(x["wall_seconds"]) for x in summaries)
    runtime = {
        "reused_prior_continuations": 0,
        "note_on_reuse": "Prior geometry labels were reused, but continuation-level prior-cache hits were not separately instrumented; current raw resumption rows are counted once as new.",
        "new_continuations": total_cont,
        "physical_steps": total_steps,
        "sum_state_wall_seconds": sum_gpu_wall,
        "critical_path_rollout_seconds": max_state_wall,
        "max_gpu_shards": 6,
        "operational_batch_sizes": [32, 64],
        "cpu_threads_per_shard": 1,
        "peak_gpu_memory": "not instrumented; observed approximately 3.7 GiB aggregate during monitoring",
        "peak_ram": "not instrumented",
    }
    dump(HERE / "runtime_statistics.json", runtime)

    final = {
        "classification": "T0_MULTIBALL_GEOMETRY_SUPPORTED" if gate_pass else "T0_MULTIBALL_GEOMETRY_INADEQUATE",
        "geometry_gate_pass": gate_pass,
        "dataset_gate_pass": False,
        "training_executed": False,
        "fixed8_training_usable": usable,
        "fixed8_false_inclusions": false_inclusions,
        "median_largest_component_radius": med_largest,
        "median_union_diameter": med_diameter,
        "median_independent_robust_coverage": med_coverage,
        "retained_set_max_common_coverage": overlap["maximum_common_coverage_fraction"],
        "failed_geometry_criteria": gate["failed_criteria"],
        "answer_multiball_target": "No: reliable but insufficiently large/covering under the preregistered gate.",
        "answer_set_vs_center": "Not tested because the geometry gate prohibited training.",
        "answer_robustness_deformation": "Not evaluated because no models were trained.",
        "next_scientific_step": "Determine why verified local spheres cover so little of independently discovered robust eta space, using a geometry representation that preserves disconnected/nonconvex success regions without filling unverified gaps.",
    }
    dump(HERE / "final_decision.json", final)

    rows = []
    for x in summaries:
        rows.append(
            f"| {x['state_id']} | {x['K']} | {float(x['largest_radius']):.5f} | "
            f"{float(x['union_diameter']):.4f} | {int(x['union_false_inclusions'])} | "
            f"{int(x['coverage_inside_union'])}/{int(x['coverage_B63_discovered'])} "
            f"({100*float(x['robust_coverage']):.1f}%) | {'yes' if x['training_usable'] else 'no'} |"
        )
    report = f"""# OrthoFlow3 true-t0 multiball basin-learning audit

## Outcome

**T0_MULTIBALL_GEOMETRY_INADEQUATE.** The fixed-eight geometry gate failed, so the mandated hard stop was applied before dataset construction or neural training.

| State | K | largest r | union diameter | false inclusion | independent robust coverage | usable |
|---|---:|---:|---:|---:|---:|---:|
{os.linesep.join(rows)}

All eight states reached four verified components. Independent union validation produced **{false_inclusions} confirmed false inclusions**. The sets were therefore reliable where claimed, but not sufficiently representative of independently discovered robust success geometry.

## Frozen gate

- Training-usable states: **{usable}/8** (required at least 6/8).
- Median largest-component radius: **{med_largest:.5f}** (required at least 0.15).
- Median union diameter: **{med_diameter:.4f}** (required at least 0.30).
- Median independent robust coverage: **{med_coverage:.3f}** (required at least 0.50).
- Retained-set maximum common coverage: **{overlap['maximum_common_coverage_count']}/8 = {overlap['maximum_common_coverage_fraction']:.3f}** (must not exceed 0.75).
- Confirmed union false inclusions: **{false_inclusions}** (required zero).

Failed criteria: `{', '.join(gate['failed_criteria'])}`.

The failure is not a reliability failure: every accepted component remained conservative under the frozen validation protocol. It is a size/coverage failure. With only four local spheres, the union captured a median of 0% of independently discovered B63 cloud points, despite large center-to-center union diameters.

## Learning decision

The verified multi-ball representation did **not** provide a sufficiently large learning target under the preregistered criteria. Consequently no source-diverse dataset was built, neither `G_CENTER` nor `G_SET` was trained, and no held-out or fresh-WIDE controller comparison was run. Set supervision therefore cannot be compared with center regression in this experiment.

## Runtime

- New continuations: **{total_cont:,}**
- Physical steps: **{total_steps:,}**
- Longest per-state rollout wall time: **{max_state_wall/3600:.2f} h**
- Maximum GPU shards: **6**

## Smallest justified next step

Audit why sparse independently robust eta points lie outside the verified four-ball union, then test a conservative representation capable of retaining disconnected/nonconvex regions without asserting success in unverified gaps. Do not begin learning until that representation passes an independent coverage gate.
"""
    (HERE / "final_report.md").write_text(report)

    artifact_names = [
        "protocol.md", "fixed8_manifest.json", "outward_sobol_cloud.csv", "outer_center_promotions.csv",
        "component_balls.csv", "union_inside_validation.csv", "independent_coverage_audit.csv",
        "geometry_gate.json", "t0_source_split.json", "verified_multiball_dataset.csv", "rejected_labels.csv",
        "universal_solution_audit.json", "dataset_gate.json", "final_decision.json",
        "runtime_statistics.json", "final_report.md",
    ]
    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "orthoflow3_path": str(BASIS), "orthoflow3_sha256": sha(BASIS),
        "fixed8_manifest_sha256": sha(HERE / "fixed8_manifest.json"),
        "training_executed": False,
        "artifacts": {name: sha(HERE / name) for name in artifact_names},
    }
    dump(HERE / "manifest.json", output_manifest)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
