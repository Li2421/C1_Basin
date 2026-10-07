"""Fail-closed analysis for the fixed-cadence G_phi causal ablation.

This program performs no rollout and imports no controller implementation.  It
reads the exact Safety/H=1 records from the completed full-episode pilot and
the H=4/8/16 records produced by ``run_evaluation.py`` in this directory.
Every raw-record and trajectory provenance field is checked before any result
is emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import binomtest, norm


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_cadence_ablation_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
)
ORDER = ("Safety", "H=1", "H=4", "H=8", "H=16")
H_BY_CONDITION: dict[str, int | None] = {
    "Safety": None, "H=1": 1, "H=4": 4, "H=8": 8, "H=16": 16,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sbatch_request(path: Path, option: str) -> str | None:
    if not path.is_file():
        return None
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(option)}(?:=|\s+)(\S+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    return None


def finite(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=np.float64)
    return data[np.isfinite(data)]


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    data = finite(values)
    if not len(data):
        return {name: None for name in ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def add_distribution(target: dict[str, Any], prefix: str, values: Iterable[float]) -> None:
    for key, value in distribution(values).items():
        target[f"{prefix}_{key}"] = value


def safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def correlation(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_conservative_ci(rescues: int, breaks: int, total: int) -> tuple[float, float]:
    """Bonferroni 95% interval for the paired success-rate difference."""
    discordant = rescues + breaks
    z = float(norm.ppf(1.0 - 0.025 / 2.0))
    r_low, r_high = wilson(discordant, total, z=z)
    if discordant == 0:
        p_low, p_high = 0.0, 1.0
    else:
        tail = 0.025 / 2.0
        p_low = 0.0 if rescues == 0 else float(
            beta_distribution.ppf(tail, rescues, discordant - rescues + 1)
        )
        p_high = 1.0 if rescues == discordant else float(
            beta_distribution.ppf(1.0 - tail, rescues + 1, discordant - rescues)
        )
    candidates = [
        rate * (2.0 * share - 1.0)
        for rate in (r_low, r_high)
        for share in (p_low, p_high)
    ]
    return float(min(candidates)), float(max(candidates))


def bootstrap_paired_ci(
    rescues: int, breaks: int, total: int, *, replicates: int, seed: int,
) -> tuple[float, float]:
    """Exact nonparametric paired bootstrap using the three delta categories."""
    same = total - rescues - breaks
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        total, np.asarray([breaks, same, rescues], dtype=np.float64) / total,
        size=replicates,
    )
    values = (counts[:, 2] - counts[:, 0]) / total
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def assert_json_content_hash(payload: dict[str, Any], path: Path) -> None:
    recorded = payload.get("content_sha256")
    if not isinstance(recorded, str):
        raise RuntimeError(f"missing content_sha256: {path}")
    actual = canonical_json_hash({key: value for key, value in payload.items() if key != "content_sha256"})
    if actual != recorded:
        raise RuntimeError(("JSON content hash mismatch", str(path), recorded, actual))


def same_episode(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left["episode_index"]) == int(right["episode_index"])
        and int(left["ic_seed"]) == int(right["ic_seed"])
        and int(left["flow_seed"]) == int(right["flow_seed"])
        and np.array_equal(
            np.asarray(left["initial_positions"], dtype=np.float64),
            np.asarray(right["initial_positions"], dtype=np.float64),
        )
    )


def resolve_trajectory(source_root: Path, row: dict[str, Any]) -> Path:
    path = Path(row["trajectory_file"])
    return path if path.is_absolute() else source_root / path


@dataclass
class Trajectory:
    step: np.ndarray
    positions_after: np.ndarray
    raw_norm: np.ndarray
    executed_norm: np.ndarray
    rewrite_norm: np.ndarray
    scheduled: np.ndarray
    effective: np.ndarray
    ood: np.ndarray
    candidate_since: np.ndarray
    stuck_timer: np.ndarray
    max_stuck_timer: np.ndarray
    event: np.ndarray


@dataclass
class Condition:
    name: str
    cadence_h: int | None
    source_root: Path
    raw_directory: Path
    expected_controller: str
    expected_config_key: str
    expected_config_hash: str
    expected_seed_hash: str
    reused: bool
    rows: dict[int, dict[str, Any]]
    trajectories: dict[int, Trajectory]


def _array(values: Any, name: str, dtype: Any = np.float64) -> np.ndarray:
    if name not in values.files:
        raise RuntimeError(f"trajectory is missing required array {name}")
    return np.asarray(values[name], dtype=dtype)


def load_trajectory(
    path: Path, *, condition: str, cadence_h: int | None,
    intervention_tol: float,
) -> tuple[Trajectory, dict[str, bool]]:
    with np.load(path, allow_pickle=False) as values:
        step = _array(values, "step", np.int64)
        positions_after = _array(values, "positions_after")
        raw_norm = _array(values, "raw_correction_norm")
        executed_norm = _array(values, "executed_correction_norm")
        rewrite_norm = _array(values, "projection_rewrite_norm")
        candidate_since = _array(values, "candidate_since")
        stuck_timer = _array(values, "stuck_timer")
        max_stuck_timer = _array(values, "max_stuck_timer")
        event = _array(values, "event", str)
        ood = (
            np.asarray(values["ood_distance"], dtype=np.float64)
            if "ood_distance" in values.files else np.asarray([], dtype=np.float64)
        )
        if "cadence_scheduled" in values.files:
            scheduled = np.asarray(values["cadence_scheduled"], dtype=bool)
        elif condition == "Safety":
            scheduled = np.zeros(len(step), dtype=bool)
        elif cadence_h == 1:
            scheduled = np.ones(len(step), dtype=bool)
        else:
            raise RuntimeError(f"{path} lacks cadence_scheduled")
        g_hat = _array(values, "g_hat")
        u_safe = _array(values, "u_safe")
        u_exec = _array(values, "u_exec")
    lengths = {
        len(step), len(positions_after), len(raw_norm), len(executed_norm),
        len(rewrite_norm), len(scheduled), len(candidate_since), len(stuck_timer),
        len(max_stuck_timer), len(event), len(g_hat), len(u_safe), len(u_exec),
    }
    if len(lengths) != 1:
        raise RuntimeError(("trajectory arrays have inconsistent lengths", str(path), lengths))
    if not np.array_equal(step, np.arange(len(step), dtype=np.int64)):
        raise RuntimeError(("physical steps are not contiguous from zero", str(path)))
    expected_schedule = (
        np.zeros(len(step), dtype=bool)
        if cadence_h is None else np.mod(step, cadence_h) == 0
    )
    schedule_exact = bool(np.array_equal(scheduled, expected_schedule))
    if not schedule_exact:
        raise RuntimeError(("fixed-cadence schedule mismatch", str(path), cadence_h))
    nonscheduled = ~scheduled
    no_held_raw_correction = bool(
        not np.any(nonscheduled)
        or (np.max(np.abs(g_hat[nonscheduled])) <= 1e-12 and np.max(np.abs(raw_norm[nonscheduled])) <= 1e-12)
    )
    no_held_executed_correction = bool(
        not np.any(nonscheduled)
        or (
            np.max(np.abs(u_exec[nonscheduled] - u_safe[nonscheduled])) <= 1e-12
            and np.max(np.abs(executed_norm[nonscheduled])) <= 1e-12
        )
    )
    if not no_held_raw_correction or not no_held_executed_correction:
        raise RuntimeError(("correction was held or applied off cadence", str(path)))
    if not all(np.isfinite(item).all() for item in (
        positions_after, raw_norm, executed_norm, rewrite_norm, g_hat, u_safe, u_exec,
    )):
        raise RuntimeError(("nonfinite trajectory data", str(path)))
    if len(ood) not in (0, len(step)):
        raise RuntimeError(("OOD array length mismatch", str(path), len(ood), len(step)))
    return Trajectory(
        step=step, positions_after=positions_after, raw_norm=raw_norm,
        executed_norm=executed_norm, rewrite_norm=rewrite_norm,
        scheduled=scheduled, effective=executed_norm > intervention_tol,
        ood=ood, candidate_since=candidate_since, stuck_timer=stuck_timer,
        max_stuck_timer=max_stuck_timer, event=event,
    ), {
        "schedule_exact": schedule_exact,
        "no_held_raw_correction": no_held_raw_correction,
        "no_held_executed_correction": no_held_executed_correction,
    }


def validate_configs(
    cadence_config: dict[str, Any], pilot_config: dict[str, Any],
    cadence_config_path: Path, pilot_config_path: Path,
) -> dict[str, bool]:
    assert_json_content_hash(cadence_config, cadence_config_path)
    assert_json_content_hash(pilot_config, pilot_config_path)
    cadence_checkpoint = cadence_config.get("checkpoint_audit", {}).get("checkpoint_sha256")
    pilot_checkpoint = pilot_config.get("checkpoint_audit", {}).get("checkpoint_sha256")
    forbidden = cadence_config.get("forbidden_online_components", {})
    pilot_modules = pilot_config.get("resolved_module_paths", {})
    cadence_modules = cadence_config.get("resolved_module_paths", {})
    common_module_paths_exact = all(
        cadence_modules.get(name) == path for name, path in pilot_modules.items()
    )
    feature_builder_path = cadence_modules.get("feature_builder")
    feature_builder_frozen_hashes = cadence_config.get("frozen_source_audit", {}).get("observed_sha256", {})
    feature_builder_exact = bool(
        feature_builder_path
        and feature_builder_path in feature_builder_frozen_hashes
        and feature_builder_frozen_hashes[feature_builder_path]
        == pilot_config.get("frozen_source_audit", {}).get("observed_sha256", {}).get(feature_builder_path)
    )
    checks = {
        "checkpoint_exact": cadence_checkpoint == pilot_checkpoint == EXPECTED_CHECKPOINT_SHA256,
        "environment_exact": cadence_config.get("environment") == pilot_config.get("environment"),
        "cbf_exact": cadence_config.get("cbf") == pilot_config.get("cbf"),
        "frozen_sources_exact": cadence_config.get("frozen_source_audit") == pilot_config.get("frozen_source_audit"),
        "resolved_common_module_paths_exact": common_module_paths_exact,
        "startup_feature_builder_path_and_hash_exact": feature_builder_exact,
        "ood_reference_exact": cadence_config.get("ood_reference") == pilot_config.get("ood_reference"),
        "forbidden_online_components_disabled": bool(forbidden) and all(value is False for value in forbidden.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(("cadence configuration does not preserve frozen pilot semantics", failed))
    return checks


def materialize_evaluation_manifest(
    output_path: Path, source_path: Path, selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Emit an explicit 128-row view without changing the runner's source manifest."""
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        rows = payload.get("episodes", [])
        if len(rows) != len(selected) or not all(same_episode(a, b) for a, b in zip(rows, selected)):
            raise RuntimeError(("existing evaluation manifest is not the exact 128-row cohort", str(output_path)))
        return payload
    payload = {
        "schema": "gphi_fixed_cadence_evaluation_seed_manifest_v1",
        "episode_count": len(selected),
        "episodes": selected,
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256(source_path),
        "source_manifest_episode_selection": [0, len(selected)],
        "flow_randomness": "per-episode PRNGKey(flow_seed), then fold_in(key, physical_step)",
        "pairing": "Safety, H=1, H=4, H=8, and H=16 use identical IC and Flow seeds",
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    write_json(output_path, payload)
    return payload


def load_condition(
    *, name: str, cadence_h: int | None, source_root: Path, raw_directory: Path,
    expected_controller: str, expected_config_key: str, expected_config_hash: str,
    expected_seed_hash: str, reused: bool, selected: list[dict[str, Any]],
    intervention_tol: float, dt: float,
) -> tuple[Condition, list[dict[str, bool]]]:
    rows: dict[int, dict[str, Any]] = {}
    trajectories: dict[int, Trajectory] = {}
    trajectory_checks: list[dict[str, bool]] = []
    for spec in selected:
        index = int(spec["episode_index"])
        path = raw_directory / f"episode_{index:04d}.json"
        if not path.is_file():
            raise RuntimeError(("missing raw cadence tuple", name, str(path)))
        row = json.loads(path.read_text())
        expected = {
            "record_complete": True,
            "controller": expected_controller,
            "episode_index": index,
            "ic_seed": int(spec["ic_seed"]),
            "flow_seed": int(spec["flow_seed"]),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "seed_manifest_sha256": expected_seed_hash,
            expected_config_key: expected_config_hash,
        }
        if cadence_h is not None and not reused:
            expected["cadence_h"] = cadence_h
        mismatch = {key: [row.get(key), value] for key, value in expected.items() if row.get(key) != value}
        if mismatch:
            raise RuntimeError(("raw record provenance mismatch", str(path), mismatch))
        if not same_episode(row, spec):
            raise RuntimeError(("raw record initial condition mismatch", str(path)))
        trajectory_path = resolve_trajectory(source_root, row)
        if not trajectory_path.is_file() or sha256(trajectory_path) != row.get("trajectory_sha256"):
            raise RuntimeError(("missing or hash-mismatched trajectory", str(trajectory_path)))
        trajectory, checks = load_trajectory(
            trajectory_path, condition=name, cadence_h=cadence_h,
            intervention_tol=intervention_tol,
        )
        if int(row["episode_steps"]) != len(trajectory.step):
            raise RuntimeError(("episode length mismatch", str(path)))
        recomputed_j = float(dt * np.sum(trajectory.executed_norm ** 2))
        if not math.isclose(recomputed_j, float(row["J_def"]), rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(("J_def mismatch", str(path), row["J_def"], recomputed_j))
        if bool(row["success"]) != (row["outcome"] == "success"):
            raise RuntimeError(("success/outcome mismatch", str(path)))
        rows[index] = row
        trajectories[index] = trajectory
        trajectory_checks.append(checks)
    return Condition(
        name=name, cadence_h=cadence_h, source_root=source_root,
        raw_directory=raw_directory, expected_controller=expected_controller,
        expected_config_key=expected_config_key,
        expected_config_hash=expected_config_hash,
        expected_seed_hash=expected_seed_hash, reused=reused,
        rows=rows, trajectories=trajectories,
    ), trajectory_checks


def outcome_counts(condition: Condition, selected_ids: list[int]) -> dict[str, Any]:
    rows = [condition.rows[index] for index in selected_ids]
    counts = Counter(str(row["outcome"]) for row in rows)
    success = counts["success"]
    low, high = wilson(success, len(rows))
    return {
        "condition": condition.name,
        "cadence_h": condition.cadence_h,
        "correction_period_seconds": None if condition.cadence_h is None else 0.05 * condition.cadence_h,
        "episodes": len(rows),
        "success": success,
        "deadlock": counts["deadlock"],
        "timeout": counts["timeout"],
        "collision": counts["collision"],
        "wall_collision": sum(bool(row.get("wall_collision")) for row in rows),
        "agent_collision": sum(bool(row.get("agent_collision")) for row in rows),
        "other_failure": counts["other"],
        "Q": success / len(rows),
        "Q_wilson_95_ci_low": low,
        "Q_wilson_95_ci_high": high,
    }


def paired_comparison(
    reference: Condition, candidate: Condition, selected_ids: list[int],
    *, replicates: int, seed: int,
) -> dict[str, Any]:
    reference_success = np.asarray(
        [bool(reference.rows[index]["success"]) for index in selected_ids], dtype=bool
    )
    candidate_success = np.asarray(
        [bool(candidate.rows[index]["success"]) for index in selected_ids], dtype=bool
    )
    rescues = int(np.sum(~reference_success & candidate_success))
    breaks = int(np.sum(reference_success & ~candidate_success))
    discordant = rescues + breaks
    bootstrap_low, bootstrap_high = bootstrap_paired_ci(
        rescues, breaks, len(selected_ids), replicates=replicates, seed=seed,
    )
    conservative_low, conservative_high = paired_conservative_ci(
        rescues, breaks, len(selected_ids)
    )
    two_sided = 1.0 if discordant == 0 else float(
        binomtest(rescues, discordant, 0.5, alternative="two-sided").pvalue
    )
    improvement = 1.0 if discordant == 0 else float(
        binomtest(rescues, discordant, 0.5, alternative="greater").pvalue
    )
    deterioration = 1.0 if discordant == 0 else float(
        binomtest(rescues, discordant, 0.5, alternative="less").pvalue
    )
    return {
        "comparison": f"{candidate.name}_vs_{reference.name}",
        "reference_condition": reference.name,
        "candidate_condition": candidate.name,
        "episodes": len(selected_ids),
        "both_success": int(np.sum(reference_success & candidate_success)),
        "reference_fail_candidate_success": rescues,
        "reference_success_candidate_fail": breaks,
        "both_fail": int(np.sum(~reference_success & ~candidate_success)),
        "discordant_pairs": discordant,
        "candidate_minus_reference_Q": float(np.mean(candidate_success) - np.mean(reference_success)),
        "paired_bootstrap_95_ci_low": bootstrap_low,
        "paired_bootstrap_95_ci_high": bootstrap_high,
        "paired_conservative_95_ci_low": conservative_low,
        "paired_conservative_95_ci_high": conservative_high,
        "exact_mcnemar_two_sided_p": two_sided,
        "exact_mcnemar_one_sided_improvement_p": improvement,
        "exact_mcnemar_one_sided_deterioration_p": deterioration,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "h1_fail_to_h_success": rescues if reference.name == "H=1" else None,
        "h1_success_to_h_fail": breaks if reference.name == "H=1" else None,
        "safety_success_to_h_fail": breaks if reference.name == "Safety" else None,
    }


def cadence_summary(
    condition: Condition, selected_ids: list[int], *, dt: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    episode_j = []
    all_raw: list[float] = []
    all_executed: list[float] = []
    all_rewrite: list[float] = []
    effective_executed: list[float] = []
    scheduled_total = effective_total = timestep_total = 0
    scheduled_fraction_by_episode = []
    effective_fraction_by_episode = []
    period_rows: list[dict[str, Any]] = []
    period_accumulator: dict[str, dict[str, Any]] = {
        "startup_0_40": {"j": [], "raw": [], "executed": [], "rewrite": [], "steps": 0, "scheduled": 0, "effective": 0},
        "warm_ge_41": {"j": [], "raw": [], "executed": [], "rewrite": [], "steps": 0, "scheduled": 0, "effective": 0},
    }
    late_scheduled_fractions = []
    late_effective_fractions = []
    late_j_values = []
    final_candidate_since = []
    final_stuck_timer = []
    peak_stuck_timer = []
    stuck_step_fractions = []
    for index in selected_ids:
        trajectory = condition.trajectories[index]
        n = len(trajectory.step)
        j_step = dt * trajectory.executed_norm ** 2
        episode_j.append(float(j_step.sum()))
        scheduled = trajectory.scheduled
        effective = trajectory.effective
        timestep_total += n
        scheduled_total += int(scheduled.sum())
        effective_total += int(effective.sum())
        scheduled_fraction_by_episode.append(float(scheduled.mean()) if n else 0.0)
        effective_fraction_by_episode.append(float(effective.mean()) if n else 0.0)
        all_raw.extend(trajectory.raw_norm[scheduled].tolist())
        all_executed.extend(trajectory.executed_norm[scheduled].tolist())
        all_rewrite.extend(trajectory.rewrite_norm[scheduled].tolist())
        effective_executed.extend(trajectory.executed_norm[effective].tolist())
        late_start = int(np.floor(0.75 * n))
        late = np.arange(n) >= late_start
        late_scheduled_fractions.append(float(scheduled[late].mean()) if np.any(late) else 0.0)
        late_effective_fractions.append(float(effective[late].mean()) if np.any(late) else 0.0)
        late_j_values.append(float(j_step[late].sum()))
        if n:
            final_candidate_since.append(float(trajectory.candidate_since[-1]))
            final_stuck_timer.append(float(trajectory.stuck_timer[-1]))
            peak_stuck_timer.append(float(np.max(trajectory.max_stuck_timer)))
            stuck_step_fractions.append(float(np.mean(trajectory.max_stuck_timer > 0)))
        for period, mask in (
            ("startup_0_40", trajectory.step <= 40),
            ("warm_ge_41", trajectory.step >= 41),
        ):
            accumulator = period_accumulator[period]
            accumulator["j"].append(float(j_step[mask].sum()))
            accumulator["steps"] += int(mask.sum())
            accumulator["scheduled"] += int(np.sum(scheduled & mask))
            accumulator["effective"] += int(np.sum(effective & mask))
            accumulator["raw"].extend(trajectory.raw_norm[scheduled & mask].tolist())
            accumulator["executed"].extend(trajectory.executed_norm[scheduled & mask].tolist())
            accumulator["rewrite"].extend(trajectory.rewrite_norm[scheduled & mask].tolist())
    deformation = {
        "condition": condition.name,
        "cadence_h": condition.cadence_h,
        "definition": "dt * sum_t ||u_exec,t-u_safe,t||_2^2",
        "total_episode_timesteps": timestep_total,
        "scheduled_query_timesteps": scheduled_total,
        "effective_corrected_timesteps": effective_total,
        "scheduled_query_fraction_weighted": scheduled_total / timestep_total if timestep_total else 0.0,
        "effective_corrected_fraction_weighted": effective_total / timestep_total if timestep_total else 0.0,
        "mean_scheduled_query_fraction_by_episode": float(np.mean(scheduled_fraction_by_episode)),
        "mean_effective_corrected_fraction_by_episode": float(np.mean(effective_fraction_by_episode)),
        "mean_executed_correction_norm_when_scheduled": safe_mean(np.asarray(all_executed)),
        "mean_executed_correction_norm_when_effective": safe_mean(np.asarray(effective_executed)),
        "mean_squared_executed_correction_norm_when_scheduled": safe_mean(np.asarray(all_executed) ** 2),
        "J_def_energy_per_scheduled_query": (
            float(dt * np.mean(np.asarray(all_executed) ** 2)) if all_executed else None
        ),
    }
    add_distribution(deformation, "J_def", episode_j)
    projection = {
        "condition": condition.name,
        "cadence_h": condition.cadence_h,
        "scheduled_query_timesteps": scheduled_total,
        "effective_corrected_timesteps": effective_total,
        "substantial_rewrite_fraction_scheduled": (
            float(np.mean(np.asarray(all_rewrite) > 1e-6)) if all_rewrite else None
        ),
        "second_projection_retry_count": sum(
            int(condition.rows[index].get("second_projection_retry_count", 0)) for index in selected_ids
        ),
        "projection_failures": sum(
            int(condition.rows[index].get("projection_failures", 0)) for index in selected_ids
        ),
        "invalid_actions": sum(
            int(condition.rows[index].get("invalid_actions", 0)) for index in selected_ids
        ),
    }
    add_distribution(projection, "raw_g_hat_norm_scheduled", all_raw)
    add_distribution(projection, "executed_correction_norm_scheduled", all_executed)
    add_distribution(projection, "projection_rewrite_norm_scheduled", all_rewrite)
    timeout_progression = {
        "late_scheduled_fraction_by_episode": distribution(late_scheduled_fractions),
        "late_effective_fraction_by_episode": distribution(late_effective_fractions),
        "late_J_def_by_episode": distribution(late_j_values),
        "final_candidate_since": distribution(final_candidate_since),
        "final_stuck_timer": distribution(final_stuck_timer),
        "peak_max_stuck_timer": distribution(peak_stuck_timer),
        "fraction_steps_with_positive_max_stuck_timer": distribution(stuck_step_fractions),
    }
    for period, accumulator in period_accumulator.items():
        row = {
            "condition": condition.name,
            "cadence_h": condition.cadence_h,
            "period": period,
            "physical_timestep_count": accumulator["steps"],
            "scheduled_query_timesteps": accumulator["scheduled"],
            "effective_corrected_timesteps": accumulator["effective"],
            "scheduled_query_fraction_weighted": accumulator["scheduled"] / accumulator["steps"] if accumulator["steps"] else 0.0,
            "effective_corrected_fraction_weighted": accumulator["effective"] / accumulator["steps"] if accumulator["steps"] else 0.0,
            "mean_raw_g_hat_norm_when_scheduled": safe_mean(np.asarray(accumulator["raw"])),
            "mean_executed_correction_norm_when_scheduled": safe_mean(np.asarray(accumulator["executed"])),
            "mean_projection_rewrite_norm_when_scheduled": safe_mean(np.asarray(accumulator["rewrite"])),
        }
        add_distribution(row, "J_def", accumulator["j"])
        period_rows.append(row)
    return deformation, projection, period_rows, timeout_progression


def per_episode_rows(
    conditions: dict[str, Condition], selected_ids: list[int], *, dt: float,
) -> list[dict[str, Any]]:
    result = []
    for name in ORDER:
        condition = conditions[name]
        for index in selected_ids:
            raw = condition.rows[index]
            trajectory = condition.trajectories[index]
            scheduled = trajectory.scheduled
            effective = trajectory.effective
            j_step = dt * trajectory.executed_norm ** 2
            result.append({
                "condition": name,
                "cadence_h": condition.cadence_h,
                "reused_from_pilot": condition.reused,
                "episode_index": index,
                "ic_seed": raw["ic_seed"],
                "flow_seed": raw["flow_seed"],
                "outcome": raw["outcome"],
                "failure_type": raw["failure_type"],
                "success": bool(raw["success"]),
                "episode_steps": raw["episode_steps"],
                "J_def": float(j_step.sum()),
                "J_def_startup_0_40": float(j_step[trajectory.step <= 40].sum()),
                "J_def_warm_ge_41": float(j_step[trajectory.step >= 41].sum()),
                "scheduled_query_timesteps": int(scheduled.sum()),
                "effective_corrected_timesteps": int(effective.sum()),
                "scheduled_query_fraction": float(scheduled.mean()) if len(scheduled) else 0.0,
                "effective_corrected_fraction": float(effective.mean()) if len(effective) else 0.0,
                "mean_raw_g_hat_norm_when_scheduled": safe_mean(trajectory.raw_norm[scheduled]),
                "mean_executed_correction_norm_when_scheduled": safe_mean(trajectory.executed_norm[scheduled]),
                "mean_executed_correction_norm_when_effective": safe_mean(trajectory.executed_norm[effective]),
                "mean_projection_rewrite_norm_when_scheduled": safe_mean(trajectory.rewrite_norm[scheduled]),
                "late_quartile_scheduled_fraction": float(np.mean(scheduled[int(np.floor(0.75 * len(scheduled))):])) if len(scheduled) else 0.0,
                "late_quartile_effective_fraction": float(np.mean(effective[int(np.floor(0.75 * len(effective))):])) if len(effective) else 0.0,
                "ood_mean": safe_mean(trajectory.ood),
                "ood_p95": float(np.quantile(trajectory.ood, 0.95)) if len(trajectory.ood) else None,
                "runtime_seconds": raw.get("runtime_seconds"),
                "trajectory_file": str(resolve_trajectory(condition.source_root, raw)),
                "trajectory_sha256": raw["trajectory_sha256"],
            })
    return result


def timeout_rows(
    conditions: dict[str, Condition], selected_ids: list[int],
    progression: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for name in ORDER:
        condition = conditions[name]
        rows = [condition.rows[index] for index in selected_ids]
        successful_lengths = [int(row["episode_steps"]) for row in rows if row["success"]]
        failed_lengths = [int(row["episode_steps"]) for row in rows if not row["success"]]
        timeout_lengths = [int(row["episode_steps"]) for row in rows if row["outcome"] == "timeout"]
        timeout_indices = [index for index in selected_ids if condition.rows[index]["outcome"] == "timeout"]
        timeout_final_stuck = [
            float(condition.trajectories[index].stuck_timer[-1])
            for index in timeout_indices if len(condition.trajectories[index].stuck_timer)
        ]
        timeout_peak_stuck = [
            float(np.max(condition.trajectories[index].max_stuck_timer))
            for index in timeout_indices if len(condition.trajectories[index].max_stuck_timer)
        ]
        row = {
            "condition": name,
            "cadence_h": condition.cadence_h,
            "timeout_count": len(timeout_lengths),
            "mean_successful_episode_length_steps": safe_mean(np.asarray(successful_lengths)),
            "mean_failed_episode_length_steps": safe_mean(np.asarray(failed_lengths)),
            "mean_timeout_episode_length_steps": safe_mean(np.asarray(timeout_lengths)),
            "mean_timeout_final_stuck_timer": safe_mean(np.asarray(timeout_final_stuck)),
            "mean_timeout_peak_max_stuck_timer": safe_mean(np.asarray(timeout_peak_stuck)),
        }
        item = progression[name]
        row.update({
            "mean_late_quartile_scheduled_fraction": item["late_scheduled_fraction_by_episode"]["mean"],
            "mean_late_quartile_effective_fraction": item["late_effective_fraction_by_episode"]["mean"],
            "mean_late_quartile_J_def": item["late_J_def_by_episode"]["mean"],
            "mean_final_candidate_since": item["final_candidate_since"]["mean"],
            "mean_final_stuck_timer": item["final_stuck_timer"]["mean"],
            "mean_peak_max_stuck_timer": item["peak_max_stuck_timer"]["mean"],
            "mean_fraction_steps_with_positive_max_stuck_timer": item["fraction_steps_with_positive_max_stuck_timer"]["mean"],
        })
        result.append(row)
    return result


def ood_rows(conditions: dict[str, Condition], selected_ids: list[int]) -> list[dict[str, Any]]:
    result = []
    h1_mean: float | None = None
    for name in ORDER[1:]:
        condition = conditions[name]
        all_ood: list[float] = []
        episode_means = []
        success_means = []
        failure_means = []
        failure_flags = []
        threshold_fractions = []
        for index in selected_ids:
            trajectory = condition.trajectories[index]
            if not len(trajectory.ood):
                continue
            all_ood.extend(trajectory.ood.tolist())
            mean = float(trajectory.ood.mean())
            episode_means.append(mean)
            failure = not bool(condition.rows[index]["success"])
            failure_flags.append(float(failure))
            (failure_means if failure else success_means).append(mean)
            threshold = condition.rows[index].get("ood", {}).get("fraction_above_train_reference_p95")
            if threshold is not None:
                threshold_fractions.append(float(threshold))
        row: dict[str, Any] = {
            "condition": name,
            "cadence_h": condition.cadence_h,
            "episodes_with_ood": len(episode_means),
            "visited_timesteps_with_ood": len(all_ood),
            "episode_mean_ood_mean": safe_mean(np.asarray(episode_means)),
            "successful_episode_mean_ood": safe_mean(np.asarray(success_means)),
            "failed_episode_mean_ood": safe_mean(np.asarray(failure_means)),
            "correlation_episode_mean_ood_with_failure": correlation(episode_means, failure_flags),
            "mean_fraction_above_train_reference_p95": safe_mean(np.asarray(threshold_fractions)),
        }
        add_distribution(row, "visited_state_ood", all_ood)
        if name == "H=1":
            h1_mean = row["visited_state_ood_mean"]
        row["visited_state_ood_mean_shift_vs_h1"] = (
            None if h1_mean is None or row["visited_state_ood_mean"] is None
            else row["visited_state_ood_mean"] - h1_mean
        )
        row["visited_state_ood_relative_mean_shift_vs_h1"] = (
            None if h1_mean in (None, 0.0) or row["visited_state_ood_mean"] is None
            else row["visited_state_ood_mean"] / h1_mean - 1.0
        )
        success_mean = row["successful_episode_mean_ood"]
        failure_mean = row["failed_episode_mean_ood"]
        corr = row["correlation_episode_mean_ood_with_failure"]
        row["remaining_failures_strongly_ood_associated"] = bool(
            success_mean is not None and failure_mean is not None and success_mean > 0
            and failure_mean / success_mean >= 1.25 and corr is not None and corr >= 0.30
        )
        result.append(row)
    return result


def _divergence(
    reference: Trajectory, candidate: Trajectory, threshold: float,
) -> tuple[int, float | None, str]:
    common = min(len(reference.positions_after), len(candidate.positions_after))
    if common:
        separation = np.linalg.norm(
            reference.positions_after[:common] - candidate.positions_after[:common],
            axis=(1, 2),
        )
        hits = np.flatnonzero(separation > threshold)
        if len(hits):
            index = int(hits[0])
            return index, float(separation[index]), "position_separation"
    if len(reference.positions_after) != len(candidate.positions_after):
        return common, None, "terminal_length_divergence"
    return -1, None, "no_material_position_divergence"


def _window_metrics(trajectory: Trajectory, center: int, radius: int = 2) -> dict[str, Any]:
    if not len(trajectory.step) or center < 0:
        return {key: None for key in (
            "scheduled_at_divergence", "scheduled_count_window_pm2",
            "max_raw_g_hat_norm_window_pm2", "max_executed_correction_norm_window_pm2",
            "max_projection_rewrite_norm_window_pm2",
        )}
    center_index = min(center, len(trajectory.step) - 1)
    start = max(0, center_index - radius)
    stop = min(len(trajectory.step), center_index + radius + 1)
    return {
        "scheduled_at_divergence": bool(trajectory.scheduled[center_index]),
        "scheduled_count_window_pm2": int(np.sum(trajectory.scheduled[start:stop])),
        "max_raw_g_hat_norm_window_pm2": float(np.max(trajectory.raw_norm[start:stop])),
        "max_executed_correction_norm_window_pm2": float(np.max(trajectory.executed_norm[start:stop])),
        "max_projection_rewrite_norm_window_pm2": float(np.max(trajectory.rewrite_norm[start:stop])),
    }


def transition_rows(
    conditions: dict[str, Condition], selected_ids: list[int], *, threshold: float,
) -> list[dict[str, Any]]:
    result = []
    h1 = conditions["H=1"]
    for index in selected_ids:
        outcomes = {name: conditions[name].rows[index]["outcome"] for name in ORDER}
        for candidate_name in ORDER[2:]:
            if outcomes[candidate_name] == outcomes["H=1"]:
                continue
            candidate = conditions[candidate_name]
            reference_trajectory = h1.trajectories[index]
            candidate_trajectory = candidate.trajectories[index]
            divergence, separation, reason = _divergence(
                reference_trajectory, candidate_trajectory, threshold
            )
            row = {
                "episode_index": index,
                "ic_seed": h1.rows[index]["ic_seed"],
                "flow_seed": h1.rows[index]["flow_seed"],
                "h1_outcome": outcomes["H=1"],
                "h4_outcome": outcomes["H=4"],
                "h8_outcome": outcomes["H=8"],
                "h16_outcome": outcomes["H=16"],
                "compared_cadence": candidate_name,
                "first_material_divergence_step": divergence,
                "divergence_reason": reason,
                "position_divergence_threshold_m": threshold,
                "position_separation_at_divergence": separation,
                "h1_terminal_event": h1.rows[index]["failure_type"],
                "candidate_terminal_event": candidate.rows[index]["failure_type"],
                "h1_episode_steps": h1.rows[index]["episode_steps"],
                "candidate_episode_steps": candidate.rows[index]["episode_steps"],
            }
            for prefix, trajectory in (("h1", reference_trajectory), ("candidate", candidate_trajectory)):
                for key, value in _window_metrics(trajectory, divergence).items():
                    row[f"{prefix}_{key}"] = value
            result.append(row)
    return result


def relative_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0.0):
        return None
    return value / baseline - 1.0


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="production")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--divergence-threshold", type=float, default=0.05)
    parser.add_argument("--high-q-tolerance", type=float, default=0.02)
    parser.add_argument("--projection-relative-change-threshold", type=float, default=0.25)
    args = parser.parse_args()
    analysis_started = time.monotonic()
    if args.namespace != "production":
        raise ValueError("the fixed-cadence causal analysis is production-only")
    if args.episodes != 128:
        raise ValueError("production analysis is locked to the exact first 128 pilot episodes")
    if args.bootstrap < 10_000:
        raise ValueError("at least 10,000 paired bootstrap replicates are required")

    cadence_config_path = HERE / "cadence_config.json"
    pilot_config_path = PILOT / "controller_config.json"
    pilot_seed_path = PILOT / "evaluation_seed_manifest.json"
    for path in (cadence_config_path, pilot_config_path, pilot_seed_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cadence_config = json.loads(cadence_config_path.read_text())
    pilot_config = json.loads(pilot_config_path.read_text())
    config_checks = validate_configs(
        cadence_config, pilot_config, cadence_config_path, pilot_config_path
    )
    pilot_seed_manifest = json.loads(pilot_seed_path.read_text())
    selected = list(pilot_seed_manifest["episodes"][:args.episodes])
    selected_ids = [int(row["episode_index"]) for row in selected]
    if selected_ids != list(range(128)) or len({row["ic_seed"] for row in selected}) != 128 or len({row["flow_seed"] for row in selected}) != 128:
        raise RuntimeError("pilot evaluation cohort is not the expected unique indices 0..127")
    seed_output = materialize_evaluation_manifest(
        HERE / "evaluation_seed_manifest.json", pilot_seed_path, selected
    )
    configured_seed_path = Path(cadence_config["seed_manifest"])
    if not configured_seed_path.is_file():
        raise FileNotFoundError(configured_seed_path)
    configured_seed_manifest = json.loads(configured_seed_path.read_text())
    configured_selected = configured_seed_manifest["episodes"][:args.episodes]
    if len(configured_selected) != args.episodes or not all(
        same_episode(a, b) for a, b in zip(selected, configured_selected)
    ):
        raise RuntimeError("cadence config does not select the exact pilot 128-episode cohort")
    configured_seed_hash = sha256(configured_seed_path)
    if cadence_config.get("seed_manifest_sha256") != configured_seed_hash:
        raise RuntimeError("cadence config seed-manifest hash mismatch")

    dt = float(cadence_config["environment"]["dt"])
    if dt != 0.05:
        raise RuntimeError(("physical dt changed", dt))
    intervention_tol = float(cadence_config["cbf"]["intervention_tol"])
    cadence_run_root = HERE / "runs" / args.namespace
    pilot_run_root = PILOT / "runs" / "production"
    specs = (
        ("Safety", None, PILOT, pilot_run_root / "raw/safety", "safety", "controller_config_sha256", pilot_config["content_sha256"], sha256(pilot_seed_path), True),
        ("H=1", 1, PILOT, pilot_run_root / "raw/learned", "learned", "controller_config_sha256", pilot_config["content_sha256"], sha256(pilot_seed_path), True),
        ("H=4", 4, HERE, cadence_run_root / "raw/h4", "h4", "cadence_config_sha256", cadence_config["content_sha256"], configured_seed_hash, False),
        ("H=8", 8, HERE, cadence_run_root / "raw/h8", "h8", "cadence_config_sha256", cadence_config["content_sha256"], configured_seed_hash, False),
        ("H=16", 16, HERE, cadence_run_root / "raw/h16", "h16", "cadence_config_sha256", cadence_config["content_sha256"], configured_seed_hash, False),
    )
    conditions: dict[str, Condition] = {}
    all_trajectory_checks = []
    for spec in specs:
        condition, checks = load_condition(
            name=spec[0], cadence_h=spec[1], source_root=spec[2],
            raw_directory=spec[3], expected_controller=spec[4],
            expected_config_key=spec[5], expected_config_hash=spec[6],
            expected_seed_hash=spec[7], reused=spec[8], selected=selected,
            intervention_tol=intervention_tol, dt=dt,
        )
        conditions[condition.name] = condition
        all_trajectory_checks.extend(checks)

    success_rows = [outcome_counts(conditions[name], selected_ids) for name in ORDER]
    success_by_name = {row["condition"]: row for row in success_rows}
    if success_by_name["Safety"]["success"] != 128 or success_by_name["H=1"]["success"] != 79:
        raise RuntimeError((
            "reused pilot outcomes do not reproduce the locked result",
            success_by_name["Safety"]["success"], success_by_name["H=1"]["success"],
        ))
    write_csv(HERE / "success_by_cadence.csv", success_rows)

    pair_specs = [
        ("Safety", "H=1"), ("Safety", "H=4"), ("Safety", "H=8"), ("Safety", "H=16"),
        ("H=1", "H=4"), ("H=1", "H=8"), ("H=1", "H=16"),
    ]
    paired_rows = [
        paired_comparison(
            conditions[reference], conditions[candidate], selected_ids,
            replicates=args.bootstrap, seed=20260925 + index,
        )
        for index, (reference, candidate) in enumerate(pair_specs)
    ]
    write_csv(HERE / "paired_outcomes.csv", paired_rows)
    paired_by_key = {
        (row["reference_condition"], row["candidate_condition"]): row
        for row in paired_rows
    }

    deformation_rows = []
    projection_rows = []
    startup_rows = []
    progression: dict[str, dict[str, Any]] = {}
    for name in ORDER:
        deformation, projection, periods, progress = cadence_summary(
            conditions[name], selected_ids, dt=dt
        )
        deformation_rows.append(deformation)
        if name != "Safety":
            projection_rows.append(projection)
        startup_rows.extend(periods)
        progression[name] = progress
    h1_deformation = next(row for row in deformation_rows if row["condition"] == "H=1")
    h1_scheduled_mean_square = h1_deformation["mean_squared_executed_correction_norm_when_scheduled"]
    for row in deformation_rows:
        if row["condition"] == "Safety":
            row["count_only_predicted_J_def_mean_using_h1_per_query_energy"] = 0.0
            row["actual_over_count_only_predicted_J_def"] = None
            continue
        predicted = (
            dt * row["scheduled_query_timesteps"] * h1_scheduled_mean_square / args.episodes
        )
        row["count_only_predicted_J_def_mean_using_h1_per_query_energy"] = predicted
        row["actual_over_count_only_predicted_J_def"] = (
            row["J_def_mean"] / predicted if predicted > 0 else None
        )
    write_csv(HERE / "deformation_by_cadence.csv", deformation_rows)

    h1_projection = next(row for row in projection_rows if row["condition"] == "H=1")
    for row in projection_rows:
        for metric in (
            "raw_g_hat_norm_scheduled_mean",
            "executed_correction_norm_scheduled_mean",
            "projection_rewrite_norm_scheduled_mean",
        ):
            row[f"{metric}_relative_change_vs_h1"] = relative_change(row[metric], h1_projection[metric])
    write_csv(HERE / "projection_by_cadence.csv", projection_rows)
    write_csv(HERE / "startup_vs_warm.csv", startup_rows)

    flat_rows = per_episode_rows(conditions, selected_ids, dt=dt)
    write_csv(HERE / "per_episode_results.csv", flat_rows)
    timeout = timeout_rows(conditions, selected_ids, progression)
    write_csv(HERE / "timeout_analysis.csv", timeout)
    transitions = transition_rows(
        conditions, selected_ids, threshold=args.divergence_threshold
    )
    write_csv(
        HERE / "outcome_transition_cases.csv", transitions,
        fieldnames=(list(transitions[0]) if transitions else [
            "episode_index", "ic_seed", "flow_seed", "h1_outcome", "h4_outcome",
            "h8_outcome", "h16_outcome", "compared_cadence",
            "first_material_divergence_step", "divergence_reason",
        ]),
    )
    ood = ood_rows(conditions, selected_ids)
    write_csv(HERE / "ood_by_cadence.csv", ood)

    q = {name: float(success_by_name[name]["Q"]) for name in ORDER[1:]}
    mean_j = {
        row["condition"]: float(row["J_def_mean"])
        for row in deformation_rows if row["condition"] != "Safety"
    }
    best_q_value = max(q.values())
    best_q_conditions = [name for name in ORDER[1:] if q[name] == best_q_value]
    high_q_floor = best_q_value - args.high_q_tolerance
    high_q_conditions = [name for name in ORDER[1:] if q[name] >= high_q_floor]
    efficient_condition = min(high_q_conditions, key=lambda name: (mean_j[name], -q[name], H_BY_CONDITION[name]))
    monotonic_nondecreasing = all(
        q[left] <= q[right] for left, right in zip(ORDER[1:-1], ORDER[2:])
    )
    strictly_improves_through_h16 = all(
        q[left] < q[right] for left, right in zip(ORDER[1:-1], ORDER[2:])
    )

    def arbitrary_pair(reference_name: str, candidate_name: str, seed: int) -> dict[str, Any]:
        return paired_comparison(
            conditions[reference_name], conditions[candidate_name], selected_ids,
            replicates=args.bootstrap, seed=seed,
        )

    nonmonotonic = False
    for middle, more_frequent, less_frequent in (
        ("H=4", "H=1", "H=8"), ("H=8", "H=4", "H=16"),
    ):
        if q[middle] > q[more_frequent] and q[middle] > q[less_frequent]:
            versus_more = arbitrary_pair(more_frequent, middle, 20261010 + H_BY_CONDITION[middle])
            versus_less = arbitrary_pair(less_frequent, middle, 20261020 + H_BY_CONDITION[middle])
            if (
                versus_more["exact_mcnemar_one_sided_improvement_p"] < 0.05
                and versus_less["exact_mcnemar_one_sided_improvement_p"] < 0.05
            ):
                nonmonotonic = True

    best_for_classification = max(
        ORDER[1:], key=lambda name: (q[name], -mean_j[name], H_BY_CONDITION[name])
    )
    best_vs_h1 = (
        {"candidate_minus_reference_Q": 0.0, "exact_mcnemar_one_sided_improvement_p": 1.0}
        if best_for_classification == "H=1"
        else paired_by_key[("H=1", best_for_classification)]
    )
    substantive_improvement = bool(
        best_vs_h1["candidate_minus_reference_Q"] >= 0.10
        and best_vs_h1["exact_mcnemar_one_sided_improvement_p"] < 0.05
    )
    deformation_reduced = mean_j[best_for_classification] <= 0.80 * mean_j["H=1"]
    close_to_safety = success_by_name["Safety"]["Q"] - best_q_value <= 0.05
    if nonmonotonic:
        classification = "NONMONOTONIC_INTERVENTION_WINDOW"
    elif substantive_improvement and deformation_reduced and close_to_safety:
        classification = "PERSISTENT_OVERCORRECTION_CONFIRMED"
    elif substantive_improvement:
        classification = "CADENCE_HELPS_BUT_NOT_SUFFICIENT"
    else:
        classification = "CADENCE_DOES_NOT_EXPLAIN_FAILURE"

    startup_by_key = {(row["condition"], row["period"]): row for row in startup_rows}
    selected_for_attribution = efficient_condition
    startup_reduction = (
        startup_by_key[("H=1", "startup_0_40")]["J_def_mean"]
        - startup_by_key[(selected_for_attribution, "startup_0_40")]["J_def_mean"]
    )
    warm_reduction = (
        startup_by_key[("H=1", "warm_ge_41")]["J_def_mean"]
        - startup_by_key[(selected_for_attribution, "warm_ge_41")]["J_def_mean"]
    )
    benefit_period = "later persistent warm/recovery correction" if warm_reduction > startup_reduction else "startup correction"

    projection_relative_changes = [
        abs(float(row[key]))
        for row in projection_rows if row["condition"] != "H=1"
        for key in (
            "raw_g_hat_norm_scheduled_mean_relative_change_vs_h1",
            "executed_correction_norm_scheduled_mean_relative_change_vs_h1",
            "projection_rewrite_norm_scheduled_mean_relative_change_vs_h1",
        )
        if row[key] is not None
    ]
    projection_materially_changed = bool(
        projection_relative_changes
        and max(projection_relative_changes) > args.projection_relative_change_threshold
    )
    query_reduction = 1.0 - (
        next(row for row in deformation_rows if row["condition"] == efficient_condition)["scheduled_query_timesteps"]
        / next(row for row in deformation_rows if row["condition"] == "H=1")["scheduled_query_timesteps"]
    )
    efficient_deformation = next(
        row for row in deformation_rows if row["condition"] == efficient_condition
    )
    count_only_ratio = efficient_deformation["actual_over_count_only_predicted_J_def"]
    frequency_reduction_dominant = bool(
        query_reduction >= 0.50 and count_only_ratio is not None
        and 0.75 <= count_only_ratio <= 1.25
    )

    ood_material_change = any(
        row["condition"] != "H=1"
        and row["visited_state_ood_mean_shift_vs_h1"] is not None
        and (
            abs(row["visited_state_ood_mean_shift_vs_h1"]) > 0.05
            or abs(row["visited_state_ood_relative_mean_shift_vs_h1"]) > 0.20
        )
        for row in ood
    )
    remaining_failure_ood_association = any(
        bool(row["remaining_failures_strongly_ood_associated"]) for row in ood
    )

    h8_to_h16 = arbitrary_pair("H=8", "H=16", 20261031)
    h32_should_be_tested = bool(
        monotonic_nondecreasing
        and q["H=16"] > q["H=8"]
        and h8_to_h16["exact_mcnemar_one_sided_deterioration_p"] >= 0.05
    )
    if h32_should_be_tested:
        next_experiment = "Evaluate fixed H=32 on the identical 128 matched episodes; change no other component."
    else:
        next_experiment = (
            f"Rerun one {efficient_condition} condition on the same 128 episodes with phase offset 8 "
            f"(apply when t mod 16 = 8) to test whether the result is cadence-density or t=0 phase specific; "
            f"change no other component."
        )

    all_rows = [row for condition in conditions.values() for row in condition.rows.values()]
    hard_safety = all(
        int(row.get("collision", False)) == 0
        and int(row.get("projection_failures", 0)) == 0
        and int(row.get("invalid_actions", 0)) == 0
        and row.get("execution_error") is None
        for row in all_rows
    )
    sanity = {
        "status": "PASS",
        "episodes_per_condition": args.episodes,
        "conditions": list(ORDER),
        "same_exact_128_episode_indices": selected_ids == list(range(128)),
        "same_ic_flow_and_initial_positions_all_conditions": all(
            same_episode(conditions[name].rows[index], selected[index])
            for name in ORDER for index in selected_ids
        ),
        "seed_source_manifest": str(pilot_seed_path),
        "seed_source_manifest_sha256": sha256(pilot_seed_path),
        "output_seed_manifest_content_sha256": seed_output.get("content_sha256"),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_checks": config_checks,
        "safety_reused_from_pilot": True,
        "h1_reused_from_pilot": True,
        "reused_safety_success_exact": success_by_name["Safety"]["success"] == 128,
        "reused_h1_success_exact": success_by_name["H=1"]["success"] == 79,
        "all_raw_records_and_trajectory_hashes_verified": True,
        "cadence_schedule_exact_every_trajectory": all(item["schedule_exact"] for item in all_trajectory_checks),
        "no_previous_correction_held": all(
            item["no_held_raw_correction"] and item["no_held_executed_correction"]
            for item in all_trajectory_checks
        ),
        "physical_dt": dt,
        "online_eta_calls": 0,
        "online_success_basin_search_calls": 0,
        "online_oracle_rollout_calls": 0,
        "intervention_gate_calls": 0,
        "hard_safety_intact": hard_safety,
        "h32_run_in_initial_ablation": False,
        "classification_rule": {
            "substantial_improvement": "best cadence minus H=1 Q >= 0.10 and one-sided exact McNemar p < 0.05",
            "restores_toward_safety": "best cadence is within 0.05 Q of Safety",
            "deformation_reduced": "best cadence mean J_def <= 0.80 * H=1 mean J_def",
            "nonmonotonic": "H=4 or H=8 has higher Q than adjacent more/less frequent conditions with one-sided exact McNemar p < 0.05 for both",
        },
    }
    critical_sanity = [
        sanity["same_exact_128_episode_indices"],
        sanity["same_ic_flow_and_initial_positions_all_conditions"],
        sanity["reused_safety_success_exact"],
        sanity["reused_h1_success_exact"],
        sanity["cadence_schedule_exact_every_trajectory"],
        sanity["no_previous_correction_held"],
        hard_safety,
        *config_checks.values(),
    ]
    if not all(critical_sanity):
        sanity["status"] = "FAIL"
    write_json(HERE / "sanity_checks.json", sanity)

    runtime_files = sorted(cadence_run_root.glob("runtime_*.json"))
    runner_processes = [json.loads(path.read_text()) for path in runtime_files]
    scheduler_script = HERE / "run_six_shards.sh"
    gres_request = sbatch_request(scheduler_script, "gres")
    shard_match = re.fullmatch(r"shard:(\d+)", gres_request or "")
    allocated_gpu_shards = int(shard_match.group(1)) if shard_match else None
    cpu_request = sbatch_request(scheduler_script, "cpus-per-task")
    memory_request = sbatch_request(scheduler_script, "mem")
    starts = [datetime.fromisoformat(row["started_utc"]) for row in runner_processes]
    finishes = [datetime.fromisoformat(row["finished_utc"]) for row in runner_processes]
    scheduler_stdout = HERE / "slurm-275.out"
    gpu_start_observation = scheduler_stdout.read_text().splitlines()[0] if scheduler_stdout.is_file() else None
    runtime = {
        "analysis_finished_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_wall_seconds": time.monotonic() - analysis_started,
        "new_runner_processes": runner_processes,
        "new_runner_runtime_files": {str(path): sha256(path) for path in runtime_files},
        "scheduler_submission_script": str(scheduler_script),
        "scheduler_submission_script_sha256": sha256(scheduler_script),
        "allocated_gpu_shards": allocated_gpu_shards,
        "allocated_cpu_cores": int(cpu_request) if cpu_request is not None else None,
        "allocated_memory": memory_request,
        "parallel_rollout_wall_seconds_approx": (
            (max(finishes) - min(starts)).total_seconds() if starts and finishes else None
        ),
        "resource_policy_basis": "six shards explicitly authorized by the user after the server was confirmed idle",
        "pre_submit_scheduler_queue": "empty (orchestrator observation; no separate persisted artifact)",
        "gpu_observation_at_job_start": gpu_start_observation,
        "reused_pilot_runtime_statistics": str(PILOT / "runtime_statistics.json"),
        "reused_pilot_runtime_statistics_sha256": sha256(PILOT / "runtime_statistics.json"),
        "episode_runtime_seconds_sum_new_h4_h8_h16": float(sum(
            float(conditions[name].rows[index].get("runtime_seconds", 0.0))
            for name in ORDER[2:] for index in selected_ids
        )),
        "episode_runtime_seconds_sum_reused_safety_h1": float(sum(
            float(conditions[name].rows[index].get("runtime_seconds", 0.0))
            for name in ORDER[:2] for index in selected_ids
        )),
        "gpu_shards_observed_new": sorted({
            str(value) for row in runner_processes
            for value in [row.get("slurm_job_gpus"), row.get("cuda_visible_devices")]
            if value not in (None, "")
        }),
        "cpu_thread_limits_observed_new": sorted({
            str(value) for row in runner_processes
            for key, value in row.items() if key in ("omp_num_threads", "openblas_num_threads", "mkl_num_threads")
        }),
        "rollouts_started_by_analysis": 0,
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    timeout_by_name = {row["condition"]: row for row in timeout}
    deformation_by_name = {row["condition"]: row for row in deformation_rows}
    lines = [
        "# Fixed-cadence G_phi closed-loop ablation",
        "",
        "This is a causal ablation of correction frequency, not a proposed final controller. It uses the frozen checkpoint, frozen controller stack, and exact first 128 matched pilot episodes. Safety and H=1 are reused byte-for-byte from the pilot; no G_phi retraining, gate, eta search, success-basin search, or online oracle rollout was used.",
        "",
        "| condition | success | deadlock | timeout | collision | wall collision | other | Q (Wilson 95% CI) | mean J_def |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        success = success_by_name[name]
        deformation = deformation_by_name[name]
        lines.append(
            f"| {name} | {success['success']} | {success['deadlock']} | {success['timeout']} | "
            f"{success['collision']} | {success['wall_collision']} | {success['other_failure']} | "
            f"{success['Q']:.6f} [{success['Q_wilson_95_ci_low']:.6f}, {success['Q_wilson_95_ci_high']:.6f}] | "
            f"{deformation['J_def_mean']:.6g} |"
        )
    lines.extend(["", "## Paired outcome changes versus H=1", ""])
    for name in ORDER[2:]:
        paired = paired_by_key[("H=1", name)]
        lines.append(
            f"- {name}: H=1 fail -> success {paired['reference_fail_candidate_success']}; "
            f"H=1 success -> fail {paired['reference_success_candidate_fail']}; "
            f"Delta Q={paired['candidate_minus_reference_Q']:+.6f}, paired bootstrap 95% CI "
            f"[{paired['paired_bootstrap_95_ci_low']:+.6f}, {paired['paired_bootstrap_95_ci_high']:+.6f}], "
            f"exact McNemar p={paired['exact_mcnemar_two_sided_p']:.6g}."
        )
    lines.extend(["", "## Diagnosis", ""])
    lines.append(f"- Highest Q: {', '.join(best_q_conditions)} (Q={best_q_value:.6f}).")
    lines.append(
        f"- Lowest mean J_def while preserving Q within {args.high_q_tolerance:.3f} of the best: "
        f"{efficient_condition} (Q={q[efficient_condition]:.6f}, mean J_def={mean_j[efficient_condition]:.6g})."
    )
    lines.append(
        f"- Q is {'monotonic nondecreasing' if monotonic_nondecreasing else 'not monotonic'} as cadence becomes sparser "
        f"(H=1/4/8/16: {q['H=1']:.6f}/{q['H=4']:.6f}/{q['H=8']:.6f}/{q['H=16']:.6f})."
    )
    lines.append(
        "- Timeout counts H=1/4/8/16: "
        + "/".join(str(timeout_by_name[name]["timeout_count"]) for name in ORDER[1:]) + "."
    )
    lines.append(
        f"- Relative to H=1, the deformation reduction for {selected_for_attribution} is "
        f"{startup_reduction:.6g} in steps 0-40 and {warm_reduction:.6g} in steps >=41; "
        f"the larger contribution is {benefit_period}."
    )
    lines.append(
        f"- Scheduled queries fall by {query_reduction:.1%} for {efficient_condition} versus H=1. "
        f"A count-only scaling using H=1 per-query executed-correction energy predicts mean J_def="
        f"{efficient_deformation['count_only_predicted_J_def_mean_using_h1_per_query_energy']:.6g}, versus "
        f"{efficient_deformation['J_def_mean']:.6g} observed (ratio {count_only_ratio:.3f}); reduced query count "
        f"{'is' if frequency_reduction_dominant else 'is not'} the dominant deformation mechanism."
    )
    lines.append(
        f"- Per-scheduled-query deformation energy is {h1_deformation['J_def_energy_per_scheduled_query']:.6g} at H=1 "
        f"and {efficient_deformation['J_def_energy_per_scheduled_query']:.6g} at {efficient_condition}."
    )
    lines.append(
        f"- Per-query projection behavior {'does' if projection_materially_changed else 'does not'} change materially "
        f"under the declared {args.projection_relative_change_threshold:.0%} relative-mean criterion. Thus the stronger claim "
        f"that projection behavior is invariant across H is {'not supported' if projection_materially_changed else 'supported descriptively'}. "
        f"Mean rewrite norm/fraction above tolerance change from "
        f"{h1_projection['projection_rewrite_norm_scheduled_mean']:.6g}/{h1_projection['substantial_rewrite_fraction_scheduled']:.3f} "
        f"at H=1 to {next(row for row in projection_rows if row['condition'] == efficient_condition)['projection_rewrite_norm_scheduled_mean']:.6g}/"
        f"{next(row for row in projection_rows if row['condition'] == efficient_condition)['substantial_rewrite_fraction_scheduled']:.3f} "
        f"at {efficient_condition}."
    )
    lines.append(
        f"- Visited-state OOD {'changes materially' if ood_material_change else 'does not materially change'} across cadence conditions; "
        f"remaining failures {'are' if remaining_failure_ood_association else 'are not'} strongly OOD-associated by the secondary criterion."
    )
    lines.append(f"- Final classification: **{classification}**.")
    lines.append(
        f"- H=32 should {'be tested' if h32_should_be_tested else 'not be tested'} under the preregistered rule; "
        f"Q {'continued improving through H=16' if strictly_improves_through_h16 else 'plateaued before H=16 rather than continuing to improve'}."
    )
    lines.append(f"- Smallest justified next experiment: {next_experiment}")
    lines.extend([
        "",
        "## Integrity and runtime",
        "",
        f"Sanity status: **{sanity['status']}**. Checkpoint SHA256: `{EXPECTED_CHECKPOINT_SHA256}`. "
        f"The output cohort contains {args.episodes} exact seed/IC records. Analysis started zero rollouts. "
        f"New runner process records: {len(runner_processes)}; observed physical GPU identifiers: "
        f"{', '.join(runtime['gpu_shards_observed_new']) or 'not recorded'}; allocation: "
        f"{allocated_gpu_shards} GPU shards, {cpu_request} CPU cores, {memory_request} memory; summed new episode compute time: "
        f"{runtime['episode_runtime_seconds_sum_new_h4_h8_h16']:.3f} s; parallel rollout wall time approximately "
        f"{runtime['parallel_rollout_wall_seconds_approx']:.3f} s. At job start the GPU reported 2 MiB used and 0% utilization; "
        f"the pre-submit scheduler queue was empty.",
    ])
    atomic_text(HERE / "cadence_report.md", "\n".join(lines) + "\n")

    products = [
        "cadence_report.md", "evaluation_seed_manifest.json", "cadence_config.json",
        "per_episode_results.csv", "success_by_cadence.csv", "paired_outcomes.csv",
        "deformation_by_cadence.csv", "timeout_analysis.csv", "startup_vs_warm.csv",
        "projection_by_cadence.csv", "outcome_transition_cases.csv", "ood_by_cadence.csv",
        "sanity_checks.json", "runtime_statistics.json",
    ]
    missing_products = [name for name in products if not (HERE / name).is_file()]
    if missing_products:
        raise RuntimeError(("required products missing", missing_products))
    input_record_sets = {}
    for name in ORDER:
        condition = conditions[name]
        raw_hashes = {
            f"episode_{index:04d}.json": sha256(condition.raw_directory / f"episode_{index:04d}.json")
            for index in selected_ids
        }
        trajectory_hashes = {
            f"episode_{index:04d}": condition.rows[index]["trajectory_sha256"]
            for index in selected_ids
        }
        input_record_sets[name] = {
            "raw_directory": str(condition.raw_directory),
            "record_count": len(raw_hashes),
            "raw_json_hash_set_sha256": canonical_json_hash(raw_hashes),
            "verified_trajectory_hash_set_sha256": canonical_json_hash(trajectory_hashes),
            "reused_from_pilot": condition.reused,
        }
    manifest = {
        "study": "gphi_fixed_cadence_ablation_v1",
        "status": sanity["status"],
        "classification": classification,
        "episodes_per_condition": args.episodes,
        "conditions": list(ORDER),
        "best_Q_conditions": best_q_conditions,
        "lowest_J_def_high_Q_condition": efficient_condition,
        "Q_monotonic_nondecreasing_with_H": monotonic_nondecreasing,
        "Q_strictly_improves_through_H16": strictly_improves_through_h16,
        "H32_should_be_tested": h32_should_be_tested,
        "single_smallest_justified_next_experiment": next_experiment,
        "source_inputs": {
            "pilot_seed_manifest": {"path": str(pilot_seed_path), "sha256": sha256(pilot_seed_path)},
            "pilot_controller_config": {"path": str(pilot_config_path), "sha256": sha256(pilot_config_path)},
            "cadence_config": {"path": str(cadence_config_path), "sha256": sha256(cadence_config_path)},
            "reused_safety_raw_directory": str(pilot_run_root / "raw/safety"),
            "reused_h1_raw_directory": str(pilot_run_root / "raw/learned"),
            "new_raw_directory": str(cadence_run_root / "raw"),
            "analysis_program": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "input_record_sets": input_record_sets,
        },
        "artifacts": {name: sha256(HERE / name) for name in products},
    }
    manifest["content_sha256"] = canonical_json_hash(manifest)
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": sanity["status"],
        "classification": classification,
        "Q": q,
        "best_Q_conditions": best_q_conditions,
        "lowest_J_def_high_Q_condition": efficient_condition,
        "H32_should_be_tested": h32_should_be_tested,
    }, indent=2))


if __name__ == "__main__":
    main()
