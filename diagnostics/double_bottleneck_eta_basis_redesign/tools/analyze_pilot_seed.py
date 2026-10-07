#!/usr/bin/env python3
"""Freeze pilot robust centers, promotion trigger, local and control jobs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
SEED_STUDY = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=float)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=float)
WIDTH = HIGH - LOW


def rows_from(directory: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    selection = json.loads((OUT / "candidate_selection.json").read_text())["representations"]
    global_summary = json.loads((OUT / "pilot_global_summary.json").read_text())["representations"]
    pilot = json.loads((OUT / "pilot_catalog.json").read_text())["episodes"]
    targets = [row for row in pilot if row["population"] == "safe_timeout_target"]
    controls = [row for row in pilot if row["population"] == "baseline_success_control"]
    episode_meta = {row["episode_id"]: row for row in targets}
    points = json.loads((ETA3 / "eta_points.json").read_text())["points"]
    scale = json.loads((OUT / "P1_SCALE.json").read_text())["scale"]
    p0_rows = rows_from(OUT / "raw", "P0_pilot_candidate_seed16_cached.jsonl")
    p1_rows = rows_from(OUT / "raw", "P1_pilot_seed16_shard*.jsonl")
    expected = {
        "P0-3D": sum(row["candidate_count"] for row in selection["P0-3D"]) * 16,
        "P1-OrthoFlow3": sum(row["candidate_count"] for row in selection["P1-OrthoFlow3"]) * 16,
    }
    if len(p0_rows) != expected["P0-3D"] or len(p1_rows) != expected["P1-OrthoFlow3"]:
        raise RuntimeError("pilot seed evaluation incomplete")
    rows_by_rep = {"P0-3D": p0_rows, "P1-OrthoFlow3": p1_rows}
    outputs = {}
    for name, rows in rows_by_rep.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["episode_id"], int(row["eta_index"]))].append(row)
        episode_output = []
        candidate_meta = {(row["episode_id"], candidate["eta_index"]): candidate for row in selection[name] for candidate in row["candidates"]}
        for episode in targets:
            candidates = []
            for selected in selection[name]:
                if selected["episode_id"] != episode["episode_id"]:
                    continue
                for candidate in selected["candidates"]:
                    group = grouped[(episode["episode_id"], candidate["eta_index"])]
                    if len(group) != 16 or {row["seed"] for row in group} != set(range(2001, 2017)):
                        raise RuntimeError("candidate seed coverage mismatch")
                    successes = sum(row["success"] for row in group)
                    candidates.append({**candidate, "successes": successes, "trials": 16, "Q_seed": successes / 16.0, "outcomes": dict(sorted(Counter(row["outcome"] for row in group).items()))})
            if candidates:
                best = min(candidates, key=lambda row: (-row["Q_seed"], -row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
                center = {key: best[key] for key in ("eta_index", "parameter_id", "theta", "successful_neighbors_among_8", "normalized_distance_to_eta_zero")}
                q_max = best["Q_seed"]
            else:
                center, q_max = None, 0.0
            episode_output.append({**episode, "candidates": candidates, "eta_robust": center, "Q_max": q_max})
        q_values = [row["Q_max"] for row in episode_output]
        outputs[name] = {
            "episodes": episode_output,
            "median_Q_max": float(np.median(q_values)),
            "Q_ge_0_25": sum(value >= 0.25 for value in q_values),
            "Q_ge_0_50": sum(value >= 0.50 for value in q_values),
            "Q_ge_0_75": sum(value >= 0.75 for value in q_values),
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        }

    p0_by_id = {row["episode_id"]: row for row in outputs["P0-3D"]["episodes"]}
    p1_by_id = {row["episode_id"]: row for row in outputs["P1-OrthoFlow3"]["episodes"]}
    condition_1 = global_summary["P1-OrthoFlow3"]["basin_existence"] >= global_summary["P0-3D"]["basin_existence"] + 3
    condition_2 = outputs["P1-OrthoFlow3"]["median_Q_max"] >= outputs["P0-3D"]["median_Q_max"] + 0.15
    condition_3_count = sum(p1_by_id[eid]["Q_max"] >= 0.50 and p0_by_id[eid]["Q_max"] < 0.50 for eid in p0_by_id)
    condition_3 = condition_3_count >= 3
    trigger = condition_1 or condition_2 or condition_3
    trigger_result = {
        "condition_1_existence_plus_3": condition_1,
        "existence_P0": global_summary["P0-3D"]["basin_existence"],
        "existence_P1": global_summary["P1-OrthoFlow3"]["basin_existence"],
        "condition_2_median_Q_plus_0_15": condition_2,
        "median_Q_P0": outputs["P0-3D"]["median_Q_max"],
        "median_Q_P1": outputs["P1-OrthoFlow3"]["median_Q_max"],
        "condition_3_three_new_Q_ge_0_50": condition_3,
        "new_Q_ge_0_50_cases": condition_3_count,
        "full_P1_triggered": trigger,
    }

    offsets = (2.0 * qmc.Sobol(3, scramble=True, seed=2026092703).random_base2(4) - 1.0) * 0.05
    local_jobs = []
    for name in ("P0-3D", "P1-OrthoFlow3"):
        for episode in outputs[name]["episodes"]:
            if episode["Q_max"] < 0.50:
                continue
            center = np.asarray(episode["eta_robust"]["theta"], dtype=float)
            center_unit = (center - LOW) / WIDTH
            for local_index, offset in enumerate(offsets):
                local_theta = LOW + np.clip(center_unit + offset, 0.0, 1.0) * WIDTH
                for seed in range(3001, 3009):
                    local_jobs.append({
                        **episode_meta[episode["episode_id"]],
                        "stage": "pilot_local_seed",
                        "representation": name,
                        "center_eta_index": episode["eta_robust"]["eta_index"],
                        "eta_index": episode["eta_robust"]["eta_index"],
                        "local_index": local_index,
                        "parameter_id": f"L{local_index:02d}",
                        "sample_type": "fixed_sobol_local_radius_0.05",
                        "theta": local_theta.tolist(),
                        "ortho_scale": scale,
                        "seed": seed,
                        "job_id": f"pilot_local|{name}|{episode['episode_id']}|{local_index:02d}|{seed}",
                    })

    # Control preservation: reuse exact P0 center/control/seed rows; run P1 only.
    p0_centers = {int(row["eta_robust"]["eta_index"]) for row in outputs["P0-3D"]["episodes"] if row["eta_robust"] is not None}
    p0_control_cached = []
    for path in sorted((SEED_STUDY / "raw").glob("control_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["eta_index"]) in p0_centers and row["episode_id"] in {episode["episode_id"] for episode in controls}:
                p0_control_cached.append(row)
    expected_p0_controls = len(p0_centers) * 12 * 4
    if len(p0_control_cached) != expected_p0_controls:
        raise RuntimeError(f"P0 control cache incomplete {len(p0_control_cached)} != {expected_p0_controls}")
    p0_control_cached.sort(key=lambda row: (int(row["eta_index"]), row["episode_id"], int(row["seed"])))
    (OUT / "raw/P0_pilot_control_seed4_cached.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in p0_control_cached))
    p1_centers = {int(row["eta_robust"]["eta_index"]): row["eta_robust"] for row in outputs["P1-OrthoFlow3"]["episodes"] if row["eta_robust"] is not None}
    control_jobs = []
    for eta_index, center in sorted(p1_centers.items()):
        for control in controls:
            for seed in range(4001, 4005):
                control_jobs.append({
                    **control,
                    "stage": "pilot_robust_control_seed4",
                    "representation": "P1-OrthoFlow3",
                    "eta_index": eta_index,
                    "parameter_id": center["parameter_id"],
                    "sample_type": "pilot_robust_center_control",
                    "theta": center["theta"],
                    "ortho_scale": scale,
                    "seed": seed,
                    "job_id": f"pilot_control|P1-OrthoFlow3|{eta_index:03d}|{control['episode_id']}|{seed}",
                })

    (OUT / "pilot_seed_robustness.json").write_text(json.dumps({"schema": "eta_basis_redesign_pilot_seed_v1", "representations": outputs, "full_trigger": trigger_result}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/pilot_local_seed.json").write_text(json.dumps({"jobs": local_jobs}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/P1_pilot_control_seed4.json").write_text(json.dumps({"jobs": control_jobs}, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({
        "state": "pilot_followups_registered",
        "p1_pilot_seed_completed": len(p1_rows),
        "pilot_local_jobs": len(local_jobs),
        "p0_pilot_control_cached": len(p0_control_cached),
        "p1_pilot_control_jobs": len(control_jobs),
        "full_p1_trigger": trigger_result,
    })
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"P0": {key: outputs["P0-3D"][key] for key in ("median_Q_max", "Q_ge_0_25", "Q_ge_0_50", "Q_ge_0_75")}, "P1": {key: outputs["P1-OrthoFlow3"][key] for key in ("median_Q_max", "Q_ge_0_25", "Q_ge_0_50", "Q_ge_0_75")}, "trigger": trigger_result, "local_jobs": len(local_jobs), "control_jobs": len(control_jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
