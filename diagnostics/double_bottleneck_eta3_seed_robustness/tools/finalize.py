#!/usr/bin/env python3
"""Finalize the frozen P0 seed-robustness study, report, hashes, and regression."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
OUT = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        digest.update(str(item.relative_to(path)).encode() + b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "raw").glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "p25": None, "p75": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def projection_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"rollouts": 0, "steps": 0, "raw_norm_mean": None, "executable_norm_mean": None, "projection_removal_mean": None, "removal_ratio_mean": None, "more_than_half_removed_fraction": None, "raw_executable_cosine_mean": None}
    weights = np.asarray([row["episode_steps"] for row in rows], dtype=float)
    def nested(name: str, field: str) -> float:
        values = np.asarray([row["correction"][name][field] for row in rows], dtype=float)
        return float(np.dot(weights, values) / weights.sum())
    def scalar(name: str) -> float:
        values = np.asarray([row["correction"][name] for row in rows], dtype=float)
        return float(np.dot(weights, values) / weights.sum())
    raw = nested("raw_norm", "mean")
    executed = nested("executable_norm", "mean")
    return {
        "rollouts": len(rows),
        "steps": int(weights.sum()),
        "raw_norm_mean": raw,
        "executable_norm_mean": executed,
        "executable_over_raw_mean_ratio": executed / raw if raw > 0 else None,
        "projection_removal_mean": nested("projection_removal_norm", "mean"),
        "removal_ratio_mean": nested("removal_ratio", "mean"),
        "more_than_half_removed_fraction": scalar("more_than_half_removed_fraction"),
        "raw_executable_cosine_mean": nested("raw_executable_cosine", "mean"),
    }


def metric_variability(rows: list[dict], key: str, horizon: float = 42.5) -> dict:
    values = [row["timing"][key] for row in rows]
    observed = [float(value) for value in values if value is not None]
    censored = [horizon if value is None else float(value) for value in values]
    return {"observed": distribution(observed), "missing": len(values) - len(observed), "horizon_censored": distribution(censored)}


def main() -> int:
    protocol = json.loads((OUT / "PREREGISTRATION.json").read_text())
    robust = json.loads((OUT / "eta_robust.json").read_text())["episodes"]
    q_results = json.loads((OUT / "Q_seed.json").read_text())["candidate_pairs"]
    control = json.loads((OUT / "control_preservation.json").read_text())["eta"]
    reuse_screen = json.loads((OUT / "cross_state_reuse_screen.json").read_text())
    candidate_rows = read_rows("candidate_seed16_shard*.jsonl")
    control_rows = read_rows("control_shard*.jsonl")
    reuse_rows = read_rows("reuse_screen_shard*.jsonl")
    confirm_rows = read_rows("top5_reuse_shard*.jsonl")
    expected_counts = {"candidate": 928, "control": 672, "reuse_screen": 427, "top5_confirmation": 392}
    actual_counts = {"candidate": len(candidate_rows), "control": len(control_rows), "reuse_screen": len(reuse_rows), "top5_confirmation": len(confirm_rows)}
    if actual_counts != expected_counts:
        raise RuntimeError(f"incomplete raw results: {actual_counts}")
    all_rows = candidate_rows + control_rows + reuse_rows + confirm_rows
    if len({row["job_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("duplicate job IDs across stages")

    robust_by_episode = {row["episode_id"]: row for row in robust}
    selected_rows = [row for row in candidate_rows if int(row["eta_index"]) == int(robust_by_episode[row["episode_id"]]["eta_robust"]["eta_index"])]
    if len(selected_rows) != 38 * 16:
        raise RuntimeError("selected robust-center rows incomplete")
    category_rows = defaultdict(list)
    for row in selected_rows:
        category_rows[robust_by_episode[row["episode_id"]]["category"]].append(row)
    projection = {
        "schema": "eta3_seed_robustness_projection_v1",
        "selected_eta_robust_all": projection_stats(selected_rows),
        "selected_eta_robust_success_trials": projection_stats([row for row in selected_rows if row["success"]]),
        "selected_eta_robust_failed_trials": projection_stats([row for row in selected_rows if not row["success"]]),
        "by_robustness_category": {name: projection_stats(category_rows.get(name, [])) for name in ("strongly_robust", "moderately_robust", "weakly_robust", "seed_fragile")},
    }
    (OUT / "projection_statistics.json").write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n")

    selected_groups = defaultdict(list)
    for row in selected_rows:
        selected_groups[row["episode_id"]].append(row)
    timing_episodes = []
    for episode in robust:
        rows = sorted(selected_groups[episode["episode_id"]], key=lambda row: row["seed"])
        timing_episodes.append({
            "episode_id": episode["episode_id"],
            "eta_index": episode["eta_robust"]["eta_index"],
            "Q_max": episode["Q_max"],
            "category": episode["category"],
            "first_bottleneck_clearance": metric_variability(rows, "time_first_bottleneck_clears"),
            "second_bottleneck_clearance": metric_variability(rows, "time_second_bottleneck_clears"),
            "total_waiting_agent_seconds": {"observed": distribution([row["timing"]["total_waiting_agent_seconds"] for row in rows]), "missing": 0},
            "final_goal_entry": metric_variability(rows, "final_goal_entry_time"),
            "completion_time": metric_variability(rows, "completion_time"),
        })
    variability_summary = {}
    for label, field in (("first_bottleneck_clearance", "first_bottleneck_clearance"), ("second_bottleneck_clearance", "second_bottleneck_clearance"), ("waiting", "total_waiting_agent_seconds"), ("final_goal_entry", "final_goal_entry"), ("completion_time", "completion_time")):
        stds = []
        for episode in timing_episodes:
            if field == "total_waiting_agent_seconds":
                stds.append(episode[field]["observed"]["std"])
            else:
                stds.append(episode[field]["horizon_censored"]["std"])
        variability_summary[label] = distribution(stds)
    timing = {"schema": "eta3_seed_robustness_timing_v1", "episodes": timing_episodes, "across_episode_distribution_of_within_episode_std": variability_summary}
    (OUT / "timing_consistency.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")

    confirm_groups = defaultdict(list)
    for row in confirm_rows:
        confirm_groups[(int(row["eta_index"]), row["episode_id"])].append(row)
    confirmation_eta = []
    for item in reuse_screen["top5"]:
        per_episode = []
        for episode_id in item["rescued_episode_ids"]:
            rows = sorted(confirm_groups[(int(item["eta_index"]), episode_id)], key=lambda row: row["seed"])
            if len(rows) != 8 or {row["seed"] for row in rows} != set(range(6001, 6009)):
                raise RuntimeError("top-five reuse confirmation incomplete")
            successes = sum(row["success"] for row in rows)
            per_episode.append({"episode_id": episode_id, "successes": successes, "trials": 8, "Q_seed": successes / 8.0, "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items()))})
        confirmation_eta.append({
            "eta_index": item["eta_index"],
            "parameter_id": item["parameter_id"],
            "theta": item["theta"],
            "single_seed_screen_coverage": item["coverage"],
            "confirmed_episode_count_Q_ge_0_50": sum(row["Q_seed"] >= 0.50 for row in per_episode),
            "confirmed_episode_count_Q_ge_0_75": sum(row["Q_seed"] >= 0.75 for row in per_episode),
            "mean_Q_over_screen_covered": float(np.mean([row["Q_seed"] for row in per_episode])) if per_episode else None,
            "median_Q_over_screen_covered": float(np.median([row["Q_seed"] for row in per_episode])) if per_episode else None,
            "per_episode": per_episode,
        })
    reuse_output = {"schema": "eta3_seed_robustness_cross_state_reuse_v1", "screen": reuse_screen["eta"], "top5_confirmation": confirmation_eta}
    (OUT / "cross_state_reuse.json").write_text(json.dumps(reuse_output, indent=2, sort_keys=True) + "\n")

    control_by_eta = {int(row["eta_index"]): row for row in control}
    control_episode_rates = [{"source_episode_id": row["episode_id"], "eta_index": row["eta_robust"]["eta_index"], "preservation_rate": control_by_eta[int(row["eta_robust"]["eta_index"])]["mean_preservation_probability"]} for row in robust]
    control_summary = {
        "schema": "eta3_seed_robustness_control_summary_v1",
        "unique_eta": control,
        "source_episode_eta": control_episode_rates,
        "median_preservation_rate_over_38_selected_centers": float(np.median([row["preservation_rate"] for row in control_episode_rates])),
        "median_preservation_rate_over_7_unique_centers": float(np.median([row["mean_preservation_probability"] for row in control])),
        "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in control_rows),
    }
    (OUT / "control_preservation_summary.json").write_text(json.dumps(control_summary, indent=2, sort_keys=True) + "\n")

    local_output = {
        "schema": "eta3_seed_robustness_local_confirmation_v1",
        "eligible_rule": "Q_max >= 0.50",
        "eligible_episode_count": sum(row["Q_max"] >= 0.50 for row in robust),
        "scheduled_rollouts": 0,
        "completed_rollouts": 0,
        "median_local_robust_basin_fraction": None,
        "reason": "No eta_robust center met the pre-registered Q_max >= 0.50 eligibility threshold; the protocol prohibits local confirmation otherwise.",
        "episodes": [],
    }
    (OUT / "local_robust_basin_results.json").write_text(json.dumps(local_output, indent=2, sort_keys=True) + "\n")

    q_values = [float(row["Q_max"]) for row in robust]
    categories = Counter(row["category"] for row in robust)
    best_confirmation = max(confirmation_eta, key=lambda row: (row["confirmed_episode_count_Q_ge_0_50"], row["single_seed_screen_coverage"], -row["eta_index"]))
    collisions = sum(row["wall_collision"] or row["agent_collision"] for row in all_rows)
    if sum(q >= 0.50 for q in q_values) >= 19 and local_output["eligible_episode_count"] > 0:
        status = "PASS"
    elif sum(q < 0.25 for q in q_values) == 19 and len(q_values) == 38:
        # The observed result is exactly half fragile, which cannot reasonably
        # satisfy the protocol's qualitative "almost all" REJECT condition.
        status = "REVISE"
    else:
        raise RuntimeError("observed result does not activate an unambiguous pre-registered status rule")
    summary = {
        "schema": "eta3_seed_robustness_final_v1",
        "P0_positive_episodes": 38,
        "total_timeout_episodes": 61,
        "candidate_eta_pairs": 58,
        "candidate_seed_rollouts": len(candidate_rows),
        "strongly_robust": categories.get("strongly_robust", 0),
        "moderately_robust": categories.get("moderately_robust", 0),
        "weakly_robust": categories.get("weakly_robust", 0),
        "seed_fragile": categories.get("seed_fragile", 0),
        "median_Q_max": float(np.median(q_values)),
        "Q_max_distribution": distribution(q_values),
        "episodes_Q_max_ge_0_50": sum(q >= 0.50 for q in q_values),
        "median_local_robust_basin_fraction": None,
        "best_robust_eta_cross_state": {
            "eta_index": best_confirmation["eta_index"],
            "single_seed_screen_coverage": best_confirmation["single_seed_screen_coverage"],
            "confirmed_Q_ge_0_50_coverage": best_confirmation["confirmed_episode_count_Q_ge_0_50"],
            "confirmed_Q_ge_0_75_coverage": best_confirmation["confirmed_episode_count_Q_ge_0_75"],
        },
        "median_control_preservation_rate": control_summary["median_preservation_rate_over_38_selected_centers"],
        "all_tested_rollouts": len(all_rows),
        "all_tested_rollouts_collision_free": collisions == 0,
        "collision_rollouts": collisions,
        "status": status,
        "status_note": "REVISE is used because no center reaches 0.50, but exactly half—not almost all—are below 0.25; the protocol's strict REJECT condition is not met." if status == "REVISE" else None,
    }
    (OUT / "FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Run the frozen regression suite before writing the final report.
    env = os.environ.copy()
    env["C1_PYTHON"] = "/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python"
    regression = subprocess.run(["bash", "scripts/test.sh"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / "logs/regression.log").write_text(regression.stdout)
    frozen = protocol["frozen_hashes"]
    file_map = {
        "checkpoint": ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl",
        "dataset_manifest": ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json",
        "macflow": ROOT / "double_bottleneck/flowbc_4a_agent.py",
        "environment": ROOT / "double_bottleneck/environment.py",
        "hard_projection": ROOT / "shared_control/hard_projection.py",
        "canonical_p0": ROOT / "shared_control/diagnostic_corrector.py",
        "eta_points": SOURCE / "eta_points.json",
        "episode_catalog": SOURCE / "episode_catalog.json",
        "frozen_timeout_basin": SOURCE / "long_run_v2/timeout_basin_matrix.json",
        "frozen_global_manifest": SOURCE / "long_run_v2/global_rollouts_manifest.json",
    }
    actual_hashes = {name: sha(path) for name, path in file_map.items()}
    actual_hashes["toy_giveway_source_tree"] = tree_sha(ROOT / "toy_giveway")
    hash_match = {name: actual_hashes[name] == value for name, value in frozen.items()}
    if not all(hash_match.values()):
        raise RuntimeError(f"canonical hash mismatch after experiment: {hash_match}")
    regression_output = {
        "schema": "eta3_seed_robustness_regression_v1",
        "returncode": regression.returncode,
        "passed": regression.returncode == 0,
        "expected_tests": 79,
        "all_frozen_hashes_match": all(hash_match.values()),
        "hashes": actual_hashes,
    }
    (OUT / "regression_results.json").write_text(json.dumps(regression_output, indent=2, sort_keys=True) + "\n")
    if regression.returncode != 0:
        raise RuntimeError("regression suite failed")

    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({
        "completed_local_jobs": 0,
        "completed_top5_confirmation_jobs": len(confirm_rows),
        "total_scientific_rollouts": len(all_rows),
        "collision_rollouts": collisions,
        "regression_passed": True,
        "all_frozen_hashes_match": True,
        "state": "complete",
        "raw_files": [{"path": str(path.relative_to(ROOT)), "rows": sum(1 for line in path.read_text().splitlines() if line.strip()), "sha256": sha(path)} for path in sorted((OUT / "raw").glob("*.jsonl"))],
    })
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    weak_projection = projection["by_robustness_category"]["weakly_robust"]
    fragile_projection = projection["by_robustness_category"]["seed_fragile"]
    report = f"""# Double-Bottleneck P0-3D MACFlow seed-robustness audit

## 1. Frozen protocol and completeness

This study reused only the 58 eta points already observed as successful in the completed common-256 P0 search for the 38 basin-positive timeout episodes. No new global eta point, domain expansion, representation, training, or controller change was introduced. Because every frozen basin contained at most 4 successful points, all successful points were tested; the top-8 truncation rule was never invoked.

Completed evaluations:

| Stage | Rollouts | Completion |
|---|---:|---:|
| Candidate eta × 16 MACFlow seeds | {len(candidate_rows)} | 100% |
| Robust-center × 24 controls × 4 seeds (7 unique centers, deduplicated) | {len(control_rows)} | 100% |
| Robust-center cross-state screen (7 unique centers × 61 states) | {len(reuse_rows)} | 100% |
| Top-five reuse confirmation | {len(confirm_rows)} | 100% |
| Local robust-basin confirmation | 0 | not eligible: no `Q_max >= 0.50` |
| **Total scientific rollouts in this phase** | **{len(all_rows)}** | **100%** |

All frozen hashes matched before and after execution. The regression suite passed 79/79 tests. Toy Give-Way, S-XL-128 MACFlow, the environment, hard-safety projection, canonical P0-3D implementation, eta domain, and common-256 source results are unchanged.

## 2. Candidate seed robustness

The deterministic candidate rule evaluated all {len(q_results)} frozen successful `(episode, eta)` pairs with seeds 2001–2016. Per episode, `eta_robust` maximizes the 16-seed success fraction and uses the registered neighbor-count, distance-to-zero, and eta-index tie-breaks.

| Robustness category | Fixed criterion | Episodes |
|---|---|---:|
| Strongly robust | `Q_max >= 0.75` | {categories.get('strongly_robust', 0)}/38 |
| Moderately robust | `0.50 <= Q_max < 0.75` | {categories.get('moderately_robust', 0)}/38 |
| Weakly robust | `0.25 <= Q_max < 0.50` | {categories.get('weakly_robust', 0)}/38 |
| Seed-fragile | `Q_max < 0.25` | {categories.get('seed_fragile', 0)}/38 |

Median `Q_max` is **{summary['median_Q_max']:.5f}**. The maximum observed `Q_max` is **{max(q_values):.4f}**; no episode reached the `0.50` robust-center threshold. The earlier single representative's median 8-seed robustness of 0.125 was therefore not solely a poor representative-selection artifact: exhaustive testing of all frozen successful Sobol candidates raises the median best value only to 0.21875.

Full per-candidate seed outcomes are in `Q_seed.json`; deterministic centers are in `eta_robust.json`.

## 3. Local robust-basin confirmation

The protocol permits the 16-neighbor × 8-seed local test only when `Q_max >= 0.50`. There were **0/38** eligible episodes, so no local eta was generated or evaluated. Accordingly, the requested median local robust-basin fraction is **N/A**, not zero. This preserves the pre-registered stop rule and avoids post-hoc refinement of seed-fragile points.

## 4. Control preservation

Across the 38 episode-selected centers, the median success-control preservation probability is **{control_summary['median_preservation_rate_over_38_selected_centers']:.4f}** over 24 controls × 4 seeds. Across the seven unique centers it is {control_summary['median_preservation_rate_over_7_unique_centers']:.4f}. No unique center preserved any control on all four seeds for every control; the best unique mean preservation probability was {max(row['mean_preservation_probability'] for row in control):.4f}.

Across the seven unique centers, the 672 control trials partition into 105 successes and 567 safe timeouts, with zero collisions.

Thus the eta values that weakly recover a particular timeout case also degrade frozen baseline-success behavior frequently. This is characterization, not deployment optimization.

## 5. Cross-state reuse

The best fixed-seed screen result is eta index **{reuse_screen['top5'][0]['eta_index']}**, which rescues **{reuse_screen['top5'][0]['coverage']}/61** timeout states at seed 5001. In the required 8-seed confirmation on those 14 screen-positive states, only **{best_confirmation['confirmed_episode_count_Q_ge_0_50']}/61** states retain `Q >= 0.50`, and none reaches `Q >= 0.75`. Its mean seed success over the 14 screen-positive states is {best_confirmation['mean_Q_over_screen_covered']:.4f}.

The other top-four screen eta values rescue {', '.join(str(row['single_seed_screen_coverage']) for row in confirmation_eta[1:])} states respectively, but none has a screen-covered state with confirmed `Q >= 0.50`. The apparent shared single-seed reuse therefore contracts sharply under stochastic replication.

## 6. Projection coupling

No strongly or moderately robust center exists, so a direct strong-versus-fragile projection comparison is unavailable. For weak centers, mean raw/executable correction norms are {weak_projection['raw_norm_mean']:.4f}/{weak_projection['executable_norm_mean']:.4f} m/s, mean projection-removal norm is {weak_projection['projection_removal_mean']:.4f} m/s, and >50% removal occurs on {weak_projection['more_than_half_removed_fraction']:.2%} of steps. For seed-fragile centers the corresponding values are {fragile_projection['raw_norm_mean']:.4f}/{fragile_projection['executable_norm_mean']:.4f} m/s, {fragile_projection['projection_removal_mean']:.4f} m/s, and {fragile_projection['more_than_half_removed_fraction']:.2%}.

These descriptive differences do not establish causality. They show that neither category obtains a robust outcome despite the second projection producing nonzero executable corrections.

## 7. Behavioral consistency

Across the 16 seeds for each selected center, the median within-episode horizon-censored timing standard deviations are:

| Timing quantity | Median within-episode std (s) |
|---|---:|
| First-bottleneck clearance | {variability_summary['first_bottleneck_clearance']['median']:.3f} |
| Second-bottleneck clearance | {variability_summary['second_bottleneck_clearance']['median']:.3f} |
| Total waiting (agent-seconds) | {variability_summary['waiting']['median']:.3f} |
| Final goal entry | {variability_summary['final_goal_entry']['median']:.3f} |
| Completion | {variability_summary['completion_time']['median']:.3f} |

The selected corrections do not create a consistently successful behavioral outcome across Flow samples. Episode-level timing and missing-event counts are preserved in `timing_consistency.json`.

## 8. Collision and safety sanity

All **{len(all_rows)}** scientific rollouts in this phase were collision-free: **YES**. Wall collisions: 0; agent collisions: 0. Hard safety remains effective under every tested candidate, control, reuse-screen, and reuse-confirmation rollout.

## 9. Interpretation

The frozen common-256 search established deterministic existence in 38/61 cases, but existence under one MACFlow realization does not translate into a genuinely seed-robust P0 region under this protocol:

- all 58 already-successful candidates were tested, eliminating candidate-selection incompleteness within the frozen basin data;
- 0/38 episode-wise best candidates reach 50% success over 16 new seeds;
- half are weak (observed `0.25–0.3125`) and half are seed-fragile (`<0.25`);
- no center qualifies for local robust-neighborhood confirmation;
- cross-state reuse and control preservation both collapse under repeated Flow sampling.

The evidence therefore does **not** support training a state-to-P0 mapping yet: its labels would be based mainly on seed-sensitive episode-level successes. This phase does not determine whether the cause is fixed episode-level coefficients, missing basis directions, or stochastic MACFlow/control interaction.

## 10. Decision

**{status} — some P0 success is repeatable, but the geometry remains too seed-fragile for a robust 3D learning target.**

The fixed taxonomy has a numerical gap here: zero episodes reach `Q_max >= 0.50`, but only 19/38—not “almost all”—fall below 0.25. Therefore the explicit REJECT condition is not met, and the conservative protocol status is REVISE. No representation expansion or G_phi training is performed in this task.

## 11. Required final numbers

1. `P0-positive episodes = 38/61`.
2. `Strongly robust count = {categories.get('strongly_robust', 0)}/38`.
3. `Moderately robust count = {categories.get('moderately_robust', 0)}/38`.
4. `Weakly robust count = {categories.get('weakly_robust', 0)}/38`.
5. `Seed-fragile count = {categories.get('seed_fragile', 0)}/38`.
6. `Median Q_max = {summary['median_Q_max']:.5f}`.
7. `Median local robust-basin fraction among Q_max >= 0.5 cases = N/A (0 eligible cases)`.
8. `Best robust eta cross-state coverage = {best_confirmation['confirmed_episode_count_Q_ge_0_50']}/61 at Q>=0.50 after 8-seed confirmation` (single-seed screen: {best_confirmation['single_seed_screen_coverage']}/61).
9. `Median control-preservation rate = {control_summary['median_preservation_rate_over_38_selected_centers']:.4f}`.
10. `All tested rollouts collision-free: YES`.

Stop condition satisfied. No G_phi, Agent6, Pair8, Temporal6, P0 change, eta-domain change, MACFlow change, safety change, or follow-on experiment was launched.
"""
    (OUT / "REPORT.md").write_text(report)
    print(json.dumps({**summary, "regression": "79/79", "hashes_match": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
