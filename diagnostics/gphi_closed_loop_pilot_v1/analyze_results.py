"""Aggregate completed production tuples without running any controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import binomtest, norm

from pilot_common import HERE, canonical_json_hash, sha256, write_json


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_conservative_ci(rescues: int, breaks: int, total: int) -> list[float]:
    """Bonferroni 95% CI for (rescues-breaks)/n.

    A 97.5% Wilson interval bounds the discordance rate and a 97.5%
    Clopper-Pearson interval bounds the rescue share among discordances.  The
    extrema of r*(2p-1) give a transparent, nondegenerate paired interval even
    when no discordant episode is observed.
    """
    discordant = rescues + breaks
    z = float(norm.ppf(1.0 - 0.025 / 2.0))
    r_low, r_high = wilson(discordant, total, z=z)
    if discordant == 0:
        p_low, p_high = 0.0, 1.0
    else:
        tail = 0.025 / 2.0
        p_low = 0.0 if rescues == 0 else float(beta_distribution.ppf(
            tail, rescues, discordant - rescues + 1))
        p_high = 1.0 if rescues == discordant else float(beta_distribution.ppf(
            1.0 - tail, rescues + 1, discordant - rescues))
    candidates = [r * (2.0 * p - 1.0)
                  for r in (r_low, r_high) for p in (p_low, p_high)]
    return [float(min(candidates)), float(max(candidates))]


def distribution(values: list[float]) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {name: None for name in ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(data.mean()), "median": float(np.median(data)),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "p95": float(np.quantile(data, 0.95)), "max": float(data.max()),
    }


def correlation(x: list[float], y: list[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64); y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]; y_array = y_array[valid]
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="production")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()
    run_root = HERE / "runs" / args.namespace
    output_root = HERE if args.namespace == "production" else run_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = HERE / "controller_config.json" if args.namespace == "production" else run_root / "controller_config.json"
    controller_config = json.loads(config_path.read_text())
    seed_manifest = json.loads(Path(controller_config["seed_manifest"]).read_text())
    expected_seed_hash = sha256(Path(controller_config["seed_manifest"]))
    expected_config_hash = controller_config["content_sha256"]
    expected_checkpoint_hash = controller_config["checkpoint_audit"]["checkpoint_sha256"]
    selected_ids = [int(row["episode_index"]) for row in seed_manifest["episodes"][:args.episodes]]
    episode_spec = {int(row["episode_index"]): row for row in seed_manifest["episodes"][:args.episodes]}
    by_controller: dict[str, dict[int, dict[str, Any]]] = {}
    missing = []
    for controller in ("safety", "learned"):
        values = {}
        for episode_index in selected_ids:
            path = run_root / "raw" / controller / f"episode_{episode_index:04d}.json"
            if not path.exists():
                missing.append(str(path)); continue
            row = json.loads(path.read_text())
            if not row.get("record_complete"):
                missing.append(str(path)); continue
            spec = episode_spec[episode_index]
            expected = {
                "controller": controller,
                "episode_index": episode_index,
                "ic_seed": int(spec["ic_seed"]),
                "flow_seed": int(spec["flow_seed"]),
                "seed_manifest_sha256": expected_seed_hash,
                "checkpoint_sha256": expected_checkpoint_hash,
                "controller_config_sha256": expected_config_hash,
            }
            mismatch = {key: [row.get(key), value] for key, value in expected.items()
                        if row.get(key) != value}
            if mismatch:
                raise RuntimeError(("raw record provenance mismatch", str(path), mismatch))
            values[episode_index] = row
        by_controller[controller] = values
    if missing:
        raise RuntimeError(("production tuples incomplete", len(missing), missing[:10]))

    flat_rows = []
    for controller, mapping in by_controller.items():
        for index in selected_ids:
            row = mapping[index]
            flat_rows.append({
                "controller": controller, "episode_index": index, "ic_seed": row["ic_seed"],
                "flow_seed": row["flow_seed"], "outcome": row["outcome"],
                "failure_type": row["failure_type"], "episode_steps": row["episode_steps"],
                "J_def": row["J_def"], "J_def_startup_0_40": row["J_def_startup_0_40"],
                "J_def_post_startup": row["J_def_post_startup"],
                "corrected_timestep_fraction": row["corrected_timestep_fraction"],
                "projection_rewrite_mean": row["projection_rewrite_norm"]["mean"],
                "projection_rewrite_p95": row["projection_rewrite_norm"]["p95"],
                "ood_mean": None if row["ood"] is None else row["ood"]["mean"],
                "ood_p95": None if row["ood"] is None else row["ood"]["p95"],
                "runtime_seconds": row["runtime_seconds"],
            })
    write_csv(output_root / "per_episode_results.csv", flat_rows)

    success_metrics: dict[str, Any] = {}
    for controller, mapping in by_controller.items():
        rows = [mapping[index] for index in selected_ids]
        counts = Counter(row["outcome"] for row in rows)
        successes = counts["success"]
        success_metrics[controller] = {
            "episodes": len(rows), "success": successes, "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"],
            "other": counts["other"], "success_rate": successes / len(rows),
            "success_wilson_95_ci": wilson(successes, len(rows)),
            "wall_collision": sum(row["wall_collision"] for row in rows),
            "agent_collision": sum(row["agent_collision"] for row in rows),
            "solver_or_projection_failures": sum(row["projection_failures"] for row in rows),
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "numerical_or_execution_errors": sum(row.get("execution_error") is not None for row in rows),
        }
    write_json(output_root / "success_metrics.json", success_metrics)

    safety_success = np.asarray([by_controller["safety"][index]["success"] for index in selected_ids], dtype=np.float64)
    learned_success = np.asarray([by_controller["learned"][index]["success"] for index in selected_ids], dtype=np.float64)
    differences = learned_success - safety_success
    rescues = int(np.sum((learned_success == 1) & (safety_success == 0)))
    breaks = int(np.sum((safety_success == 1) & (learned_success == 0)))
    rng = np.random.default_rng(20260925)
    bootstrap_replicates = max(args.bootstrap, 100_000)
    draws = rng.integers(0, len(differences), size=(bootstrap_replicates, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    discordant = rescues + breaks
    mcnemar_two_sided = 1.0 if discordant == 0 else float(
        binomtest(rescues, discordant, 0.5, alternative="two-sided").pvalue)
    mcnemar_improvement = 1.0 if discordant == 0 else float(
        binomtest(rescues, discordant, 0.5, alternative="greater").pvalue)
    paired_rows = []
    rescue_by_type = Counter()
    for index in selected_ids:
        safety = by_controller["safety"][index]
        learned = by_controller["learned"][index]
        if safety["success"] and learned["success"]:
            cell = "both_success"
        elif not safety["success"] and learned["success"]:
            cell = "safety_fail_gphi_success"
            rescue_by_type[safety["outcome"]] += 1
        elif safety["success"] and not learned["success"]:
            cell = "safety_success_gphi_fail"
        else:
            cell = "both_fail"
        paired_rows.append({
            "episode_index": index, "ic_seed": safety["ic_seed"],
            "flow_seed": safety["flow_seed"], "paired_cell": cell,
            "safety_outcome": safety["outcome"], "learned_outcome": learned["outcome"],
            "safety_failure_type": safety["failure_type"],
            "learned_failure_type": learned["failure_type"],
        })
    write_csv(output_root / "paired_outcomes.csv", paired_rows)
    paired = {
        "episodes": len(differences), "learned_minus_safety_success_rate": float(differences.mean()),
        "paired_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "paired_conservative_95_ci": paired_conservative_ci(rescues, breaks, len(differences)),
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 20260925,
        "learned_success_safety_failure": rescues,
        "safety_success_learned_failure": breaks,
        "both_success": int(np.sum((safety_success == 1) & (learned_success == 1))),
        "both_failure": int(np.sum((safety_success == 0) & (learned_success == 0))),
        "discordant_pairs": discordant,
        "net_rescues": rescues - breaks,
        "rescues_by_safety_failure_type": dict(rescue_by_type),
        "exact_mcnemar_two_sided_p": mcnemar_two_sided,
        "exact_mcnemar_one_sided_improvement_p": mcnemar_improvement,
    }
    paired["statistically_clear_improvement"] = paired["paired_conservative_95_ci"][0] > 0
    paired["comparison_inconclusive"] = paired["paired_conservative_95_ci"][0] <= 0 <= paired["paired_conservative_95_ci"][1]
    write_json(output_root / "paired_baseline_comparison.json", paired)
    write_json(output_root / "paired_success_analysis.json", paired)

    learned_rows = [by_controller["learned"][index] for index in selected_ids]
    j_values = [row["J_def"] for row in learned_rows]
    deformation = {
        "all_episodes": distribution(j_values),
        "successful_episodes": distribution([row["J_def"] for row in learned_rows if row["success"]]),
        "failed_episodes": distribution([row["J_def"] for row in learned_rows if not row["success"]]),
        "definition": "dt * sum_t ||u_exec,t-u_safe,t||_2^2",
    }
    write_json(output_root / "deformation_metrics.json", deformation)
    startup_def = [row["J_def_startup_0_40"] for row in learned_rows]
    post_def = [row["J_def_post_startup"] for row in learned_rows]
    write_json(output_root / "startup_deformation_metrics.json", {
        "startup_steps_inclusive": [0, 40], "startup": distribution(startup_def),
        "post_startup": distribution(post_def),
        "mean_startup_fraction_of_total": float(np.mean([
            row["J_def_startup_0_40"] / row["J_def"] if row["J_def"] > 0 else 0.0
            for row in learned_rows
        ])),
    })

    raw_correction = []; executed_correction = []; rewrite = []
    startup_raw = []; startup_executed = []; startup_rewrite = []
    post_raw = []; post_executed = []; post_rewrite = []
    diagnostic_features = []; diagnostic_positions = []; diagnostic_episode = []
    diagnostic_step = []; diagnostic_kind = []; diagnostic_ood = []
    jdef_half_steps = []; jdef_ninety_steps = []
    for row in learned_rows:
        trajectory_path = HERE / row["trajectory_file"]
        if sha256(trajectory_path) != row["trajectory_sha256"]:
            raise RuntimeError(("trajectory hash mismatch", trajectory_path))
        with np.load(trajectory_path, allow_pickle=False) as values:
            raw_correction.extend(np.asarray(values["raw_correction_norm"], dtype=float).tolist())
            executed_correction.extend(np.asarray(values["executed_correction_norm"], dtype=float).tolist())
            rewrite.extend(np.asarray(values["projection_rewrite_norm"], dtype=float).tolist())
            raw_values = np.asarray(values["raw_correction_norm"], dtype=float)
            executed_values = np.asarray(values["executed_correction_norm"], dtype=float)
            rewrite_values = np.asarray(values["projection_rewrite_norm"], dtype=float)
            startup_raw.extend(raw_values[:41].tolist())
            startup_executed.extend(executed_values[:41].tolist())
            startup_rewrite.extend(rewrite_values[:41].tolist())
            post_raw.extend(raw_values[41:].tolist())
            post_executed.extend(executed_values[41:].tolist())
            post_rewrite.extend(rewrite_values[41:].tolist())
            per_step_jdef = float(controller_config["environment"]["dt"]) * executed_values**2
            cumulative = np.cumsum(per_step_jdef)
            if len(cumulative) and cumulative[-1] > 0:
                jdef_half_steps.append(int(np.searchsorted(cumulative, 0.5 * cumulative[-1])))
                jdef_ninety_steps.append(int(np.searchsorted(cumulative, 0.9 * cumulative[-1])))
            features = np.asarray(values["feature"], dtype=np.float64)
            positions = np.asarray(values["positions_before"], dtype=np.float64)
            ood = (np.asarray(values["ood_distance"], dtype=np.float64)
                   if "ood_distance" in values.files else np.asarray([], dtype=np.float64))
            chosen: dict[int, str] = {int(index): "high_ood" for index in np.argsort(ood)[-3:]}
            if not row["success"]:
                for index in range(max(0, len(features) - 5), len(features)):
                    chosen[index] = "failure_preceding" if index not in chosen else "high_ood_and_failure_preceding"
            for index, kind in chosen.items():
                diagnostic_features.append(features[index]); diagnostic_positions.append(positions[index])
                diagnostic_episode.append(row["episode_index"]); diagnostic_step.append(index)
                diagnostic_kind.append(kind); diagnostic_ood.append(ood[index])
    projection = {
        "raw_correction_norm": distribution(raw_correction),
        "executed_correction_norm": distribution(executed_correction),
        "projection_rewrite_norm": distribution(rewrite),
        "substantial_rewrite_threshold": controller_config["cbf"]["intervention_tol"],
        "substantial_rewrite_fraction": float(np.mean(np.asarray(rewrite) > controller_config["cbf"]["intervention_tol"])),
        "second_projection_retry_count": sum(row["second_projection_retry_count"] for row in learned_rows),
        "solver_or_projection_failures": sum(row["projection_failures"] for row in learned_rows),
        "invalid_actions": sum(row["invalid_actions"] for row in learned_rows),
    }
    write_json(output_root / "projection_metrics.json", projection)
    startup_diagnostics = {
        "startup_steps_inclusive": [0, 40],
        "post_startup_steps": ">=41",
        "startup": {
            "raw_correction_norm": distribution(startup_raw),
            "executed_correction_norm": distribution(startup_executed),
            "projection_rewrite_norm": distribution(startup_rewrite),
            "J_def": distribution(startup_def),
        },
        "post_startup": {
            "raw_correction_norm": distribution(post_raw),
            "executed_correction_norm": distribution(post_executed),
            "projection_rewrite_norm": distribution(post_rewrite),
            "J_def": distribution(post_def),
        },
        "mean_startup_fraction_of_total_J_def": float(np.mean([
            row["J_def_startup_0_40"] / row["J_def"] if row["J_def"] > 0 else 0.0
            for row in learned_rows
        ])),
    }
    write_json(output_root / "startup_diagnostics.json", startup_diagnostics)
    _arrays = {
        "feature": np.asarray(diagnostic_features, dtype=np.float64).reshape(-1, 214),
        "positions": np.asarray(diagnostic_positions, dtype=np.float64).reshape(-1, 2, 2),
        "episode_index": np.asarray(diagnostic_episode, dtype=np.int64),
        "step": np.asarray(diagnostic_step, dtype=np.int64),
        "kind": np.asarray(diagnostic_kind), "ood_distance": np.asarray(diagnostic_ood, dtype=np.float64),
    }
    np.savez_compressed(output_root / "diagnostic_states.npz", **_arrays)

    ood_mean = [float("nan") if row["ood"] is None else float(row["ood"]["mean"]) for row in learned_rows]
    ood_max = [float("nan") if row["ood"] is None else float(row["ood"]["max"]) for row in learned_rows]
    failures = [float(not row["success"]) for row in learned_rows]
    success_ood = [float(row["ood"]["mean"]) for row in learned_rows if row["success"] and row["ood"] is not None]
    failure_ood = [float(row["ood"]["mean"]) for row in learned_rows if not row["success"] and row["ood"] is not None]
    ood_metrics = {
        "metric": controller_config["ood_reference"],
        "episode_mean_distance": distribution(ood_mean), "episode_max_distance": distribution(ood_max),
        "successful_episode_mean_distance": distribution(success_ood),
        "failed_episode_mean_distance": distribution(failure_ood),
        "correlation_episode_mean_ood_with_failure": correlation(ood_mean, failures),
        "correlation_episode_max_ood_with_failure": correlation(ood_max, failures),
        "diagnostic_states_file": "diagnostic_states.npz",
    }
    paired_cell_by_episode = {int(row["episode_index"]): row["paired_cell"] for row in paired_rows}
    ood_metrics["episode_mean_distance_by_paired_outcome"] = {
        cell: distribution([
            float(row["ood"]["mean"]) for row in learned_rows
            if row["ood"] is not None and paired_cell_by_episode[row["episode_index"]] == cell
        ])
        for cell in sorted(set(paired_cell_by_episode.values()))
    }
    write_json(output_root / "ood_metrics.json", ood_metrics)

    all_runs = [length for row in learned_rows for length in row["correction_run_lengths"]]
    timing = {
        "active_threshold": controller_config["cbf"]["intervention_tol"],
        "corrected_timestep_fraction_by_episode": distribution([row["corrected_timestep_fraction"] for row in learned_rows]),
        "correction_run_lengths": distribution(all_runs),
        "maximum_run_length": max(all_runs, default=0),
        "late_quartile_corrected_fraction_by_episode": distribution([row["late_quartile_corrected_fraction"] for row in learned_rows]),
        "J_def_half_accumulation_step": distribution(jdef_half_steps),
        "J_def_ninety_percent_accumulation_step": distribution(jdef_ninety_steps),
        "mean_startup_fraction_of_total_J_def": startup_diagnostics["mean_startup_fraction_of_total_J_def"],
        "interpretation_limit": "diagnostic only; no threshold gate was used for control",
    }
    write_json(output_root / "timing_diagnostics.json", timing)
    failure_rows = []
    divergence_threshold = 0.05
    for index in selected_ids:
        safety = by_controller["safety"][index]
        learned = by_controller["learned"][index]
        if safety["success"] == learned["success"]:
            continue
        safety_path = HERE / safety["trajectory_file"]
        learned_path = HERE / learned["trajectory_file"]
        if sha256(safety_path) != safety["trajectory_sha256"] or sha256(learned_path) != learned["trajectory_sha256"]:
            raise RuntimeError(("discordant trajectory hash mismatch", index))
        with np.load(safety_path, allow_pickle=False) as safe_values, np.load(learned_path, allow_pickle=False) as learned_values:
            length = min(len(safe_values["positions_after"]), len(learned_values["positions_after"]))
            separation = np.linalg.norm(
                np.asarray(safe_values["positions_after"][:length], dtype=float)
                - np.asarray(learned_values["positions_after"][:length], dtype=float), axis=(1, 2))
            candidates = np.flatnonzero(separation > divergence_threshold)
            divergence = int(candidates[0]) if len(candidates) else (length - 1 if length else -1)
            learned_correction = np.asarray(learned_values["executed_correction_norm"], dtype=float)
            learned_rewrite = np.asarray(learned_values["projection_rewrite_norm"], dtype=float)
            learned_ood = np.asarray(learned_values["ood_distance"], dtype=float)
            prefix_stop = max(0, min(divergence + 1, len(learned_correction)))
            failure_rows.append({
                "episode_index": index, "ic_seed": safety["ic_seed"], "flow_seed": safety["flow_seed"],
                "safety_outcome": safety["outcome"], "learned_outcome": learned["outcome"],
                "paired_cell": paired_cell_by_episode[index],
                "safety_steps": safety["episode_steps"], "learned_steps": learned["episode_steps"],
                "first_position_divergence_step": divergence,
                "position_divergence_threshold_m": divergence_threshold,
                "position_separation_at_divergence": float(separation[divergence]) if divergence >= 0 else None,
                "max_executed_correction_through_divergence": float(np.max(learned_correction[:prefix_stop])) if prefix_stop else None,
                "max_projection_rewrite_through_divergence": float(np.max(learned_rewrite[:prefix_stop])) if prefix_stop else None,
                "ood_at_divergence": float(learned_ood[min(divergence, len(learned_ood)-1)]) if divergence >= 0 and len(learned_ood) else None,
                "safety_trajectory_file": safety["trajectory_file"],
                "learned_trajectory_file": learned["trajectory_file"],
            })
    write_csv(output_root / "failure_cases.csv", failure_rows)

    hard_safety = (
        success_metrics["learned"]["collision"] == 0
        and success_metrics["learned"]["invalid_actions"] == 0
        and success_metrics["learned"]["solver_or_projection_failures"] == 0
        and success_metrics["learned"]["numerical_or_execution_errors"] == 0
    )
    delta = paired["learned_minus_safety_success_rate"]
    paired["hard_safety_passed"] = hard_safety
    paired["extend_to_256"] = bool(
        args.episodes == 128 and hard_safety and paired["comparison_inconclusive"])
    paired["extension_rule"] = (
        "extend only after 128 when hard safety passes and the conservative paired 95% CI contains zero"
    )
    write_json(output_root / "paired_baseline_comparison.json", paired)
    write_json(output_root / "paired_success_analysis.json", paired)
    if not hard_safety or delta < 0:
        classification = "GPHI_CLOSED_LOOP_FAIL"
    elif success_metrics["learned"]["success_rate"] >= 0.95 and (
            paired["statistically_clear_improvement"] or delta == 0):
        classification = "GPHI_CLOSED_LOOP_PILOT_PASS"
    else:
        classification = "GPHI_IMPROVES_BUT_NEEDS_REFINEMENT"
    ood_corr = ood_metrics["correlation_episode_mean_ood_with_failure"]
    if failure_ood and success_ood and np.mean(failure_ood) > np.mean(success_ood) and (ood_corr or 0) > 0:
        dominant = "distribution shift"
    elif projection["substantial_rewrite_fraction"] > 0.25:
        dominant = "projection interaction"
    elif timing["corrected_timestep_fraction_by_episode"]["mean"] > 0.5:
        dominant = "intervention timing/window"
    else:
        dominant = "approximation error"

    sanity = {
        "status": "PASS", "episodes_per_controller": args.episodes,
        "matched_episode_ids": sorted(by_controller["safety"]) == sorted(by_controller["learned"]) == selected_ids,
        "matched_ic_and_flow_seeds": all(
            by_controller["safety"][index]["ic_seed"] == by_controller["learned"][index]["ic_seed"]
            and by_controller["safety"][index]["flow_seed"] == by_controller["learned"][index]["flow_seed"]
            for index in selected_ids
        ),
        "training_overlap_audit": seed_manifest["training_overlap_audit"]["status"],
        "online_eta_or_oracle_calls": 0,
        "checkpoint_hash_consistent": len({row["checkpoint_sha256"] for mapping in by_controller.values() for row in mapping.values()}) == 1,
        "hard_safety_intact": hard_safety,
    }
    if not all((sanity["matched_episode_ids"], sanity["matched_ic_and_flow_seeds"], sanity["checkpoint_hash_consistent"])):
        sanity["status"] = "FAIL"
    write_json(output_root / "sanity_checks.json", sanity)
    write_json(output_root / "checkpoint_audit.json", controller_config["checkpoint_audit"])
    write_json(output_root / "integrity_audit.json", {
        "status": sanity["status"],
        "checkpoint": controller_config["checkpoint_audit"],
        "frozen_sources": controller_config["frozen_source_audit"],
        "resolved_module_paths": controller_config["resolved_module_paths"],
        "training_overlap_audit": seed_manifest["training_overlap_audit"],
        "seed_manifest_sha256": expected_seed_hash,
        "controller_config_sha256": expected_config_hash,
        "raw_record_provenance_verified": True,
        "trajectory_hashes_verified_for_all_learned_and_discordant_safety_episodes": True,
        "online_eta_oracle_basin_gate_calls": 0,
    })
    runtimes = [json.loads(path.read_text()) for path in run_root.glob("runtime_*.json")]
    runtime = {
        "analysis_finished_utc": datetime.now(timezone.utc).isoformat(),
        "runner_processes": runtimes,
        "episode_runtime_seconds_sum": float(sum(row["runtime_seconds"] for row in flat_rows)),
        "gpu_shards_observed": sorted({value for row in runtimes for value in [row.get("slurm_job_gpus")] if value}),
    }
    write_json(output_root / "runtime_statistics.json", runtime)
    report = f"""# Startup-complete learned G_phi closed-loop pilot

The evaluation used {args.episodes} fresh, matched full episodes per controller.  It made no online eta, basin, oracle, or gate call.

| controller | success | deadlock | timeout | collision | other | Q (Wilson 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Safety | {success_metrics['safety']['success']} | {success_metrics['safety']['deadlock']} | {success_metrics['safety']['timeout']} | {success_metrics['safety']['collision']} | {success_metrics['safety']['other']} | {success_metrics['safety']['success_rate']:.6f} [{success_metrics['safety']['success_wilson_95_ci'][0]:.6f}, {success_metrics['safety']['success_wilson_95_ci'][1]:.6f}] |
| Learned G_phi | {success_metrics['learned']['success']} | {success_metrics['learned']['deadlock']} | {success_metrics['learned']['timeout']} | {success_metrics['learned']['collision']} | {success_metrics['learned']['other']} | {success_metrics['learned']['success_rate']:.6f} [{success_metrics['learned']['success_wilson_95_ci'][0]:.6f}, {success_metrics['learned']['success_wilson_95_ci'][1]:.6f}] |

Paired success improvement is {delta:+.6f}, with conservative paired 95% CI [{paired['paired_conservative_95_ci'][0]:+.6f}, {paired['paired_conservative_95_ci'][1]:+.6f}]. Rescues/breaks are {paired['learned_success_safety_failure']} / {paired['safety_success_learned_failure']}; exact two-sided McNemar p={paired['exact_mcnemar_two_sided_p']:.6g}. Learned deformation mean/median/P95 is {deformation['all_episodes']['mean']:.6g} / {deformation['all_episodes']['median']:.6g} / {deformation['all_episodes']['p95']:.6g}. Hard-safety integrity: **{hard_safety}**.

Classification: **{classification}**.  If refinement is required, the single descriptively dominant logged axis is **{dominant}**.  Timing remains diagnostic only and did not alter control.
"""
    (output_root / "pilot_report.md").write_text(report)
    products = [
        "pilot_report.md", "evaluation_seed_manifest.json", "controller_config.json",
        "checkpoint_audit.json", "integrity_audit.json", "per_episode_results.csv", "success_metrics.json",
        "paired_outcomes.csv", "paired_baseline_comparison.json", "paired_success_analysis.json",
        "deformation_metrics.json", "startup_deformation_metrics.json", "startup_diagnostics.json",
        "projection_metrics.json", "ood_metrics.json",
        "timing_diagnostics.json", "failure_cases.csv", "sanity_checks.json",
        "runtime_statistics.json", "diagnostic_states.npz",
    ]
    manifest = {
        "study": "startup_complete_gphi_closed_loop_pilot_v1", "namespace": args.namespace,
        "episodes_per_controller": args.episodes, "classification": classification,
        "dominant_remaining_issue": None if classification == "GPHI_CLOSED_LOOP_PILOT_PASS" else dominant,
        "artifacts": {name: sha256(output_root / name) for name in products if (output_root / name).exists()},
    }
    manifest["content_sha256"] = canonical_json_hash(manifest)
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps({
        "status": sanity["status"], "classification": classification,
        "safety_success": success_metrics["safety"]["success"],
        "learned_success": success_metrics["learned"]["success"],
        "paired_delta": delta, "paired_ci": paired["paired_bootstrap_95_ci"],
    }, indent=2))


if __name__ == "__main__":
    main()
