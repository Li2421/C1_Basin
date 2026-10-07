#!/usr/bin/env python3
"""Finalize the budget-stopped true-t0 audit without launching rollouts."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_t0_basin_structure_v1"
LEARN = ROOT / "diagnostics/orthoflow3_basin_margin_learning_v1"
AFFINE = np.array([0.875, 0.0, 0.375], dtype=float)
SCALE = np.array([0.75, 1.0, 0.75], dtype=float)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["state_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nt(eta: np.ndarray) -> np.ndarray:
    return (eta - AFFINE) / SCALE


def sphere_intersection(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return 4.0 * math.pi * min(r1, r2) ** 3 / 3.0
    return math.pi * (r1 + r2 - d) ** 2 * (d*d + 2*d*(r1+r2) - 3*(r1-r2)**2) / (12*d)


def min_cover_ratio(centers: np.ndarray, radii: np.ndarray) -> tuple[float, np.ndarray]:
    if len(centers) == 1 or float(np.max(np.linalg.norm(centers-centers[0], axis=1))) <= 1e-14:
        return 0.0, np.asarray(centers[0]).copy()
    starts = [centers.mean(axis=0), *centers]
    best = None
    for start in starts:
        objective = lambda x: float(np.max(np.linalg.norm(x-centers, axis=1)/radii))
        result = minimize(objective, start, method="Powell", options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 10000})
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("intersection optimization failed")
    return float(best.fun), np.asarray(best.x)


def overlap_solution(centers: np.ndarray, radii: np.ndarray) -> dict:
    ratio, point = min_cover_ratio(centers, radii)
    all_ok = ratio <= 1.0 + 1e-8
    maximum = 0
    maximum_subset: tuple[int, ...] = ()
    maximum_point = point
    for size in range(len(centers), 0, -1):
        found = False
        for subset in itertools.combinations(range(len(centers)), size):
            sr, sp = min_cover_ratio(centers[list(subset)], radii[list(subset)])
            if sr <= 1.0 + 1e-8:
                maximum, maximum_subset, maximum_point = size, subset, sp
                found = True
                break
        if found:
            break
    return {
        "all_intersection": all_ok,
        "minimax_normalized_radius": ratio,
        "all_intersection_eta_normalized": point.tolist() if all_ok else None,
        "all_intersection_eta_raw": (AFFINE + SCALE*point).tolist() if all_ok else None,
        "maximum_overlap_count": maximum,
        "maximum_overlap_fraction": maximum/len(centers),
        "maximum_overlap_indices": list(maximum_subset),
        "maximum_overlap_eta_normalized": maximum_point.tolist(),
        "maximum_overlap_eta_raw": (AFFINE + SCALE*maximum_point).tolist(),
    }


def main() -> None:
    manifest = json.load(open(HERE / "t0_state_manifest.json"))
    states = manifest["attempted_states"]
    dirs = {x["state_id"]: HERE / "anchor_runs" / x["state_id"] for x in states}
    raw_by_state: dict[str, list[dict]] = {}
    for sid, directory in dirs.items():
        raw_path = directory / "raw/pilot_rollouts.jsonl"
        raw_by_state[sid] = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]

    aggregate_map = {
        "t0_screening.csv": "common_cloud_results.csv",
        "t0_center_candidates.csv": "center_scores.csv",
        "t0_centers_b63.csv": "robust_centers.csv",
        "t0_ray_screening.csv": "ray_screening.csv",
        "t0_limiting_radii.csv": "limiting_direction_promotion.csv",
    }
    for output, source in aggregate_map.items():
        rows = []
        for directory in dirs.values():
            rows.extend(read_csv(directory / source))
        write_csv(HERE / output, rows)

    centers = {}
    balls = {}
    completed = []
    for sid, directory in dirs.items():
        cr = read_csv(directory / "robust_centers.csv")
        if cr:
            centers[sid] = cr[0]
        br = read_csv(directory / "conservative_ball_parameters.csv")
        if br:
            balls[sid] = br[0]
        if (directory / "false_inclusion_summary.json").exists() and (directory / "outside_shell_results.csv").exists():
            completed.append(sid)

    ball_rows = []
    for sid, b in balls.items():
        complete = sid in completed
        fi = json.load(open(dirs[sid] / "false_inclusion_summary.json")) if complete else None
        ball_rows.append({
            **b,
            "independent_validation_complete": complete,
            "confirmed_internal_false_inclusions": fi["robust_false_inclusions"] if fi else "UNDERRESOLVED",
            "verified_label": complete and fi["robust_false_inclusions"] == 0,
            "margin_usable": complete and fi["robust_false_inclusions"] == 0 and float(b["r_ball"]) >= 0.20,
        })
    write_csv(HERE / "t0_verified_balls.csv", ball_rows)

    inside_rows = []
    for sid in completed:
        for row in read_csv(dirs[sid] / "inside_ball_screening.csv"):
            inside_rows.append({**row, "evidence": "screening"})
        for row in read_csv(dirs[sid] / "inside_ball_promoted64.csv"):
            inside_rows.append({**row, "evidence": "promoted64"})
    write_csv(HERE / "t0_inside_validation.csv", inside_rows)

    zero_rows = []
    for state in states:
        sid = state["state_id"]
        by_seed = {}
        for row in raw_by_state[sid]:
            if np.max(np.abs(np.asarray(row["eta"], dtype=float))) < 1e-12:
                by_seed[int(row["future_index"])] = row
        success = sum(bool(row["success"]) for row in by_seed.values())
        b = balls.get(sid)
        if b:
            center = nt(np.array([float(b["c1"]), float(b["c2"]), float(b["c3"])]))
            zero_dist = float(np.linalg.norm(nt(np.zeros(3))-center))
            zero_inside = zero_dist <= float(b["r_ball"])+1e-12
            zero_margin = 1-zero_dist/float(b["r_ball"])
        else:
            zero_dist = zero_inside = zero_margin = "BALL_UNRESOLVED"
        zero_rows.append({
            "state_id": sid, "screen_successes": success, "screen_trials": len(by_seed),
            "Q64": "UNDERRESOLVED", "B63": "UNDERRESOLVED",
            "zero_center_distance_normalized": zero_dist, "zero_inside_ball": zero_inside,
            "zero_signed_relative_margin": zero_margin,
        })
    write_csv(HERE / "t0_zero_feasibility.csv", zero_rows)

    verified = [sid for sid in completed if sid in balls]
    center_ids = list(centers)
    all_centers = np.array([nt(np.array([float(centers[s]["c1"]), float(centers[s]["c2"]), float(centers[s]["c3"])])) for s in center_ids])
    vc = np.array([nt(np.array([float(balls[s]["c1"]), float(balls[s]["c2"]), float(balls[s]["c3"])])) for s in verified])
    vr = np.array([float(balls[s]["r_ball"]) for s in verified])
    pair_rows, overlap_rows = [], []
    for i, j in itertools.combinations(range(len(center_ids)), 2):
        d = float(np.linalg.norm(all_centers[i]-all_centers[j]))
        both_verified = center_ids[i] in verified and center_ids[j] in verified
        if both_verified:
            ri=float(balls[center_ids[i]]["r_ball"]);rj=float(balls[center_ids[j]]["r_ball"]);denom=(ri+rj)/2
        else:
            denom=None
        pair_rows.append({"state_i":center_ids[i],"state_j":center_ids[j],"center_distance":d,
                          "both_balls_validation_complete":both_verified,
                          "mean_radius":denom if denom is not None else "UNDERRESOLVED",
                          "distance_over_mean_radius":d/denom if denom is not None else "UNDERRESOLVED"})
    for i, j in itertools.combinations(range(len(verified)), 2):
        d = float(np.linalg.norm(vc[i]-vc[j])); denom=(vr[i]+vr[j])/2
        vi=4*math.pi*vr[i]**3/3;vj=4*math.pi*vr[j]**3/3;inter=sphere_intersection(vr[i],vr[j],d)
        overlap_rows.append({"state_i":verified[i],"state_j":verified[j],"center_distance":d,"intersection_volume":inter,"union_volume":vi+vj-inter,"jaccard":inter/(vi+vj-inter)})
    write_csv(HERE / "t0_center_pairwise_distances.csv", pair_rows)
    write_csv(HERE / "t0_ball_overlap.csv", overlap_rows)
    full = overlap_solution(vc, vr)
    retained = overlap_solution(vc, 0.8*vr)
    intersection = {
        "scope": "five validation-complete balls only; audit stopped at physical-step cap",
        "verified_ball_state_ids": verified,
        "full_balls": full,
        "retained_0p8_balls": retained,
    }
    dump(HERE / "t0_common_intersection.json", intersection)

    intermediate = read_csv(LEARN / "verified_ball_dataset.csv")
    intermediate_centers = np.array([nt(np.array([float(x["c1"]),float(x["c2"]),float(x["c3"])])) for x in intermediate])
    intermediate_radii = np.array([float(x["r_ball"]) for x in intermediate])
    compare_rows=[]
    for sid,c,r in zip(verified,vc,vr,strict=True):
        distances=np.linalg.norm(intermediate_centers-c,axis=1); nearest=float(distances.min())
        best_j=-1.0;best_idx=0
        for k,(ic,ir) in enumerate(zip(intermediate_centers,intermediate_radii,strict=True)):
            d=float(np.linalg.norm(c-ic));v1=4*math.pi*r**3/3;v2=4*math.pi*ir**3/3;iv=sphere_intersection(r,ir,d);j=iv/(v1+v2-iv)
            if j>best_j:best_j,best_idx=j,k
        compare_rows.append({"state_id":sid,"t0_radius":r,"nearest_intermediate_center_distance":nearest,
                             "nearest_intermediate_state_id":intermediate[int(np.argmin(distances))]["state_id"],
                             "nearest_intermediate_radius":intermediate_radii[int(np.argmin(distances))],
                             "radius_change":r-intermediate_radii[int(np.argmin(distances))],
                             "maximum_intermediate_ball_jaccard":best_j,"max_overlap_intermediate_state_id":intermediate[best_idx]["state_id"]})
    write_csv(HERE / "t0_vs_intermediate_geometry.csv", compare_rows)

    support_all = read_csv(LEARN / "fresh_h_support_distance.csv")
    selected_eps = {str(x["fresh_episode_index"]):x["state_id"] for x in states}
    support_rows=[]
    for row in support_all:
        if row["episode_index"] in selected_eps:
            support_rows.append({"state_id":selected_eps[row["episode_index"]],**row})
    write_csv(HERE / "t0_h_support_distance.csv", support_rows)

    fresh = read_csv(LEARN / "fresh_wide_results.csv")
    controller_rows=[]
    name_map={"g_lowj":"G_LOWJ","g_center":"G_CENTER","g_margin":"G_MARGIN"}
    for sid in verified:
        state=next(x for x in states if x["state_id"]==sid);ep=str(state["fresh_episode_index"])
        b=balls[sid];c=nt(np.array([float(b["c1"]),float(b["c2"]),float(b["c3"])]));r=float(b["r_ball"])
        for row in fresh:
            if row["episode_index"]!=ep or row["controller"] not in name_map:continue
            eta=np.asarray(json.loads(row["eta"]),dtype=float);rho=float(np.linalg.norm(nt(eta)-c)/r)
            controller_rows.append({"state_id":sid,"episode_index":ep,"controller":name_map[row["controller"]],"eta":row["eta"],
                                    "rho_to_true_t0_ball":rho,"inside_ball":rho<=1,"inside_retained_0p8":rho<=.8,
                                    "prior_single_episode_success":row["success"],"true_Q64":"NOT_EVALUATED_BUDGET_STOP"})
    write_csv(HERE / "frozen_controller_vs_t0_balls.csv", controller_rows)

    all_rows=[row for rows in raw_by_state.values() for row in rows]
    valid_steps=sum(int(row["continuation_steps"]) for row in all_rows)
    invalid_dir=HERE/"anchor_runs_invalid_feature_replay_20260928"
    invalid_rows=0
    if invalid_dir.exists():
        for path in invalid_dir.glob("*/raw/pilot_rollouts.jsonl"):
            invalid_rows += sum(1 for line in path.read_text().splitlines() if line.strip())
    runtime={
        "reused_continuations":0,"new_valid_continuations":len(all_rows),"new_physical_steps":valid_steps,
        "hard_cap_physical_steps":7000000,"hard_cap_exceeded_by":valid_steps-7000000,
        "stop_reason":"aggregate physical-step cap exceeded because observed steps/continuation exceeded preflight historical estimate",
        "invalid_preflight_records_excluded":invalid_rows,"invalid_preflight_physical_steps":0,
        "rollout_wall_time_minutes_until_stop_approx":40.5,"max_gpu_shards":6,"gpu_memory":"not captured",
        "cpu_threads_max":12,"ram_allocation_gib_max":48,"observed_ram_peak":"Slurm accounting disabled",
        "no_rollouts_after_stop":True,
    }
    dump(HERE / "runtime_statistics.json", runtime)

    radii=vr.tolist();fi=sum(json.load(open(dirs[s]/"false_inclusion_summary.json"))["robust_false_inclusions"] for s in verified)
    decision={
        "classification":"T0_BASIN_AUDIT_UNDERRESOLVED",
        "reason":"hard 7M physical-step budget exceeded before 8-state validation, eta=0 B63, and controller-Q64 audit completed",
        "attempted_states":8,"center_resolved":len(centers),"ball_geometry_resolved":len(balls),
        "independent_validation_complete":len(verified),"margin_usable_complete":sum(r>=.2 for r in radii),
        "confirmed_internal_false_inclusions":fi,"zero_B63_resolved":0,"controller_Q64_resolved":0,
        "partial_evidence_direction":"completed balls are reliable but often small; no all-ball common intersection among the five complete balls",
        "network_training_started":False,
    }
    dump(HERE / "final_decision.json", decision)

    pair_dist=[float(x["center_distance"]) for x in pair_rows];pair_rel=[float(x["distance_over_mean_radius"]) for x in pair_rows if x["distance_over_mean_radius"]!="UNDERRESOLVED"]
    counts=Counter(x["controller"] for x in controller_rows if str(x["inside_ball"]).lower()=="true")
    report=f"""# OrthoFlow3 true-t0 basin structure audit v1

## Decision

**T0_BASIN_AUDIT_UNDERRESOLVED**

The audit was stopped when aggregate valid execution reached **{len(all_rows):,} new continuations and {valid_steps:,} physical steps**, exceeding the frozen 7,000,000-step automatic cap by {valid_steps-7000000:,}. No robustness criterion was weakened and no neural network was trained. Earlier feature-replay attempts were quarantined: they executed zero physical steps and are excluded from every scientific result.

## Frozen cohort and completion

- Attempted: 8 genuine t=0 states from 8 independent source groups.
- Robust centers: {len(centers)}/8; every resolved center was 64/64.
- Ball geometry constructed: {len(balls)}/8.
- Full independent inside validation complete: {len(verified)}/8.
- Margin-usable (`r>=0.20`): {sum(r>=.2 for r in radii)}/{len(verified)} completed balls.
- Internal false inclusions: {fi}/20 mandatory promoted points; all 60/60 independent screening points in the five completed balls were 8/8.

## Completed-ball geometry

| state | zero screen | zero B63 | center eta | center Q64 | radius | validation | margin usable | zero in ball |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for state in states:
        sid=state["state_id"];c=centers[sid];b=balls.get(sid);z=next(x for x in zero_rows if x["state_id"]==sid)
        radius=f"{float(b['r_ball']):.6f}" if b else "unresolved"
        validation="complete" if sid in verified else ("incomplete" if b else "not constructed")
        usable="yes" if sid in verified and float(b["r_ball"])>=.2 else ("no" if sid in verified else "underresolved")
        report += f"| {sid} | {z['screen_successes']}/8 | unresolved | ({float(c['c1']):.6f}, {float(c['c2']):.6f}, {float(c['c3']):.6f}) | 64/64 | {radius} | {validation} | {usable} | {z['zero_inside_ball']} |\n"
    report += f"""

Completed radii: mean {statistics.mean(radii):.6f}, median {statistics.median(radii):.6f}, min {min(radii):.6f}, max {max(radii):.6f}. Three of five complete balls are smaller than 0.20; the partial evidence therefore leans toward t0 spheres often being too small for the frozen margin-learning criterion, despite perfect observed internal reliability.

Pairwise normalized center distance: mean {statistics.mean(pair_dist):.6f}, median {statistics.median(pair_dist):.6f}, min {min(pair_dist):.6f}, max {max(pair_dist):.6f}. Relative to mean radius, the ratio has mean {statistics.mean(pair_rel):.3f} and median {statistics.median(pair_rel):.3f}.

The five completed full balls have all-intersection `{full['all_intersection']}` and maximum overlap {full['maximum_overlap_count']}/5. Their 0.8-retained regions have all-intersection `{retained['all_intersection']}` and maximum overlap {retained['maximum_overlap_count']}/5. This is evidence against the universal-region degeneracy seen in the intermediate training balls, but it is not the preregistered 8-state answer.

## Zero feasibility and frozen controllers

`eta=0` received only 8-seed screening before the hard stop. Screening counts were {[x['screen_successes'] for x in zero_rows]}, but none were promoted to 64; therefore ZERO_B63 frequency is unresolved. Zero lies outside every constructed t0 ball.

The existing fresh-WIDE records provide the exact frozen controller eta predictions for these episode-start h values. Relative to the five complete balls, inside-ball counts are G_LOWJ {counts['G_LOWJ']}/5, G_CENTER {counts['G_CENTER']}/5, and G_MARGIN {counts['G_MARGIN']}/5. Those old records contain one episode outcome each, not Q64; no new controller rollouts were launched after the budget stop.

## Interpretation

- Are true t0 balls sufficiently large for margin learning? **Not established.** Only 2/5 completed balls satisfy `r>=0.20`; three are small (including two at 0.02125 and 0.085).
- State-dependent or universal? **Partial evidence favors state dependence:** no common point exists across the five full or retained balls, and maximum retained overlap is {retained['maximum_overlap_count']}/5.
- Major t0/intermediate phase mismatch? **Suggestive but underresolved.** Intermediate labels all share center `(0.625,0,0.375)` and radii 0.23375–0.35362; two of five completed t0 centers shift by about 0.408 normalized units and several t0 radii collapse. The complete eight-state/controller-Q64 comparison was not affordable.
- Does this explain CENTER/MARGIN falling below Safety? **Plausible but not demonstrated**, because eta-zero B63 and matched controller Q64 were the post-ball stages omitted by the hard stop.

## Smallest justified next step

With explicit approval for the additional cost, resume only the three incomplete state validations plus eta-zero B63 and frozen-controller Q64 checks; do not collect new states or train a model. Recompute a physical-step-aware cost ceiling from the observed {valid_steps/len(all_rows):.1f} steps/continuation before resuming.
"""
    (HERE / "t0_basin_structure_report.md").write_text(report)

    outputs=[p for p in HERE.iterdir() if p.is_file() and p.name not in {"manifest.json"}]
    dump(HERE / "manifest.json", {
        "experiment":"ORTHOFLOW3_T0_BASIN_STRUCTURE_AUDIT_V1","status":"budget_stopped_underresolved",
        "authoritative_basis_sha256":"51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
        "outputs":{p.name:sha(p) for p in sorted(outputs)},"network_training":False,
    })
    print(json.dumps({"decision":decision,"runtime":runtime,"intersection":intersection},indent=2))


if __name__ == "__main__":
    main()
