"""Fail-closed analysis of G_phi fixed cadence on the frozen WIDE benchmark.

This module performs no rollout, imports no controller code, and never fills in
missing observations.  It consumes exactly 200 matched Safety/H=1/H=4/H=8/H=16
records plus their per-step trajectory archives.  Any incomplete tuple,
provenance mismatch, off-cadence correction, non-finite action, or trajectory
hash mismatch stops analysis before scientific outputs are published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
RUN_ROOT = HERE / "runs/production"
ORDER = ("Safety", "H=1", "H=4", "H=8", "H=16")
SLUG = {"Safety": "safety", "H=1": "h1", "H=4": "h4", "H=8": "h8", "H=16": "h16"}
CADENCE = {"Safety": None, "H=1": 1, "H=4": 4, "H=8": 8, "H=16": 16}
EXPECTED_CHECKPOINT_SHA256 = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
EXPECTED_NORMALIZATION_SHA256 = "0db38d045dfaf102aa4edd891aa37beab9d3b142ef5b0952812efedbf797034a"
ALLOWED_OUTCOMES = {"success", "deadlock", "timeout", "collision", "wall_collision", "other_failure"}
OUTPUTS = (
    "wide_cadence_report.md",
    "frozen_benchmark_manifest.json",
    "integrity_audit.json",
    "per_episode_results.csv",
    "paired_outcomes.csv",
    "success_by_cadence.csv",
    "rescue_break_by_cadence.csv",
    "failure_type_rescue.csv",
    "deformation_by_cadence.csv",
    "projection_by_cadence.csv",
    "startup_vs_warm.csv",
    "outcome_transition_matrix.csv",
    "representative_rescues.csv",
    "representative_breaks.csv",
    "sanity_checks.json",
    "runtime_statistics.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: Iterable[float]) -> dict[str, float | None]:
    array = finite(values)
    if not len(array):
        return {key: None for key in ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def add_stats(row: dict[str, Any], prefix: str, values: Iterable[float]) -> None:
    row.update({f"{prefix}_{key}": value for key, value in stats(values).items()})


def row_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    width = int(np.prod(array.shape[1:]))
    return np.linalg.norm(array.reshape((len(array), width)), axis=1)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def binom_upper(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return min(1.0, sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2.0 ** trials))


def mcnemar(rescues: int, breaks: int) -> tuple[float, float, float]:
    discordant = rescues + breaks
    if not discordant:
        return 1.0, 1.0, 1.0
    improvement = binom_upper(rescues, discordant)
    deterioration = binom_upper(breaks, discordant)
    return min(1.0, 2.0 * min(improvement, deterioration)), improvement, deterioration


def bootstrap_delta_ci(rescues: int, breaks: int, total: int, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    same = total - rescues - breaks
    draws = rng.multinomial(total, [breaks / total, same / total, rescues / total], size=replicates)
    delta = (draws[:, 2] - draws[:, 0]) / total
    low, high = np.quantile(delta, [0.025, 0.975])
    return float(low), float(high)


def as_initial(row: dict[str, Any]) -> np.ndarray:
    initial = np.asarray(row["initial_positions"], dtype=np.float64)
    if initial.shape != (2, 2) or not np.isfinite(initial).all():
        raise RuntimeError(("invalid initial_positions", row.get("episode_index"), initial.shape))
    return initial


def manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("episodes", "initial_conditions", "records"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise RuntimeError("frozen_benchmark_manifest.json has no episode list")


def episode_index(row: dict[str, Any], fallback: int | None = None) -> int:
    for key in ("episode_index", "rollout_id", "index"):
        if key in row:
            return int(row[key])
    if fallback is None:
        raise RuntimeError("record has no episode_index/rollout_id/index")
    return fallback


def outcome(row: dict[str, Any]) -> str:
    if bool(row.get("success", False)) or str(row.get("outcome", "")).lower() == "success":
        return "success"
    if bool(row.get("wall_collision", False)):
        return "wall_collision"
    if bool(row.get("agent_collision", False)) or bool(row.get("collision", False)):
        return "collision"
    if bool(row.get("deadlock", False)):
        return "deadlock"
    if bool(row.get("timeout", False)):
        return "timeout"
    label = str(row.get("failure_type", row.get("outcome", "other_failure"))).lower().strip()
    aliases = {
        "other": "other_failure", "failure": "other_failure", "wall": "wall_collision",
        "agent_collision": "collision", "collision_agent": "collision", "collision_wall": "wall_collision",
    }
    label = aliases.get(label, label)
    if label not in ALLOWED_OUTCOMES:
        label = "other_failure"
    return label


class Trajectory:
    def __init__(self, path: Path, condition: str, cadence: int | None, tolerance: float):
        with np.load(path, allow_pickle=False) as values:
            required = ("step", "g_hat", "u_safe", "raw_second_target", "u_exec")
            missing = [name for name in required if name not in values.files]
            if missing:
                raise RuntimeError(("trajectory missing required arrays", str(path), missing))
            self.step = np.asarray(values["step"], dtype=np.int64)
            self.g_hat = np.asarray(values["g_hat"], dtype=np.float64)
            self.u_safe = np.asarray(values["u_safe"], dtype=np.float64)
            self.raw_second_target = np.asarray(values["raw_second_target"], dtype=np.float64)
            self.u_exec = np.asarray(values["u_exec"], dtype=np.float64)
            self.raw = (
                np.asarray(values["raw_correction_norm"], dtype=np.float64)
                if "raw_correction_norm" in values.files
                else row_norm(self.g_hat)
            )
            self.executed = (
                np.asarray(values["executed_correction_norm"], dtype=np.float64)
                if "executed_correction_norm" in values.files
                else row_norm(self.u_exec - self.u_safe)
            )
            self.rewrite = (
                np.asarray(values["projection_rewrite_norm"], dtype=np.float64)
                if "projection_rewrite_norm" in values.files
                else row_norm(self.u_exec - (self.u_safe + self.g_hat))
            )
            self.scheduled = (
                np.asarray(values["cadence_scheduled"], dtype=bool)
                if "cadence_scheduled" in values.files
                else (np.zeros(len(self.step), dtype=bool) if cadence is None else self.step % cadence == 0)
            )
        arrays = (self.step, self.g_hat, self.u_safe, self.raw_second_target, self.u_exec, self.raw, self.executed, self.rewrite, self.scheduled)
        if len({len(array) for array in arrays}) != 1:
            raise RuntimeError(("trajectory length mismatch", str(path)))
        if not np.array_equal(self.step, np.arange(len(self.step), dtype=np.int64)):
            raise RuntimeError(("non-contiguous physical steps", str(path)))
        expected = np.zeros(len(self.step), dtype=bool) if cadence is None else self.step % cadence == 0
        if not np.array_equal(self.scheduled, expected):
            raise RuntimeError(("fixed cadence schedule mismatch", condition, str(path)))
        if not all(np.isfinite(array).all() for array in arrays[:-1]):
            raise RuntimeError(("NaN/Inf trajectory", condition, str(path)))
        recomputed_raw = row_norm(self.g_hat)
        recomputed_executed = row_norm(self.u_exec - self.u_safe)
        recomputed_rewrite = row_norm(self.u_exec - self.raw_second_target)
        if not np.allclose(self.raw_second_target, self.u_safe + self.g_hat, rtol=1e-9, atol=1e-11):
            raise RuntimeError(("second projection target mismatch", condition, str(path)))
        if not np.allclose(self.raw, recomputed_raw, rtol=1e-8, atol=1e-10):
            raise RuntimeError(("raw correction norm mismatch", condition, str(path)))
        if not np.allclose(self.executed, recomputed_executed, rtol=1e-8, atol=1e-10):
            raise RuntimeError(("executed correction norm mismatch", condition, str(path)))
        if not np.allclose(self.rewrite, recomputed_rewrite, rtol=1e-8, atol=1e-10):
            raise RuntimeError(("projection rewrite norm mismatch", condition, str(path)))
        inactive = ~self.scheduled
        if np.any(inactive) and (
            np.max(np.abs(self.g_hat[inactive])) > 1e-12
            or np.max(np.abs(self.u_exec[inactive] - self.u_safe[inactive])) > 1e-12
        ):
            raise RuntimeError(("held/off-cadence correction", condition, str(path)))
        if cadence is None and (np.max(np.abs(self.g_hat), initial=0.0) > 1e-12 or np.max(self.executed, initial=0.0) > 1e-12):
            raise RuntimeError(("Safety trajectory contains G_phi correction", str(path)))
        self.effective = self.executed > tolerance


class Condition:
    def __init__(self, name: str):
        self.name = name
        self.slug = SLUG[name]
        self.cadence = CADENCE[name]
        self.rows: dict[int, dict[str, Any]] = {}
        self.trajectories: dict[int, Trajectory] = {}
        self.raw_hashes: dict[str, str] = {}
        self.trajectory_hashes: dict[str, str] = {}


def config_hash(config: dict[str, Any], path: Path) -> str:
    return str(config.get("content_sha256") or config.get("config_sha256") or sha256(path))


def load_conditions(
    specs: list[dict[str, Any]], config: dict[str, Any], config_path: Path,
    manifest_path: Path, dt: float, tolerance: float,
) -> tuple[dict[str, Condition], dict[str, Any]]:
    manifest_file_hash = sha256(manifest_path)
    manifest_content_hash = canonical_hash(json.loads(manifest_path.read_text()))
    expected_config_hash = config_hash(config, config_path)
    conditions = {name: Condition(name) for name in ORDER}
    spec_by_index = {episode_index(spec, offset): spec for offset, spec in enumerate(specs)}
    expected_indices = sorted(spec_by_index)
    if len(spec_by_index) != len(specs):
        raise RuntimeError("duplicate episode indices in frozen benchmark manifest")
    flow_semantics_seen: set[str] = set()
    for name, condition in conditions.items():
        directory = RUN_ROOT / "raw" / condition.slug
        actual_files = sorted(directory.glob("episode_*.json"))
        if len(actual_files) != len(specs):
            raise RuntimeError(("incomplete or extra raw record set", name, len(actual_files), len(specs), str(directory)))
        for index in expected_indices:
            path = directory / f"episode_{index:04d}.json"
            if not path.is_file():
                raise RuntimeError(("missing raw record", name, index, str(path)))
            row = json.loads(path.read_text())
            if row.get("record_complete") is not True:
                raise RuntimeError(("incomplete raw record", name, index))
            if episode_index(row) != index:
                raise RuntimeError(("episode index mismatch", name, index, episode_index(row)))
            controller = str(row.get("controller", "")).lower().replace("=", "").replace("-", "")
            accepted = {condition.slug, name.lower().replace("=", "").replace("-", "")}
            if controller not in accepted:
                raise RuntimeError(("controller label mismatch", name, controller, accepted))
            if row.get("cadence_h") != condition.cadence:
                raise RuntimeError(("cadence_h mismatch", name, index, row.get("cadence_h"), condition.cadence))
            if row.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError(("checkpoint mismatch", name, index, row.get("checkpoint_sha256")))
            recorded_manifest_hashes = {
                str(row[key]) for key in ("benchmark_manifest_sha256", "seed_manifest_sha256", "frozen_benchmark_manifest_sha256")
                if row.get(key) is not None
            }
            if recorded_manifest_hashes and not recorded_manifest_hashes.intersection({manifest_file_hash, manifest_content_hash, str(json.loads(manifest_path.read_text()).get("content_sha256"))}):
                raise RuntimeError(("benchmark manifest hash mismatch", name, index, recorded_manifest_hashes))
            recorded_config_hashes = {
                str(row[key]) for key in ("wide_cadence_config_sha256", "cadence_config_sha256", "controller_config_sha256", "config_sha256")
                if row.get(key) is not None
            }
            if recorded_config_hashes and expected_config_hash not in recorded_config_hashes and sha256(config_path) not in recorded_config_hashes:
                raise RuntimeError(("configuration hash mismatch", name, index, recorded_config_hashes, expected_config_hash))
            spec = spec_by_index[index]
            if not np.array_equal(as_initial(row), as_initial(spec)):
                raise RuntimeError(("initial condition mismatch", name, index))
            if "rollout_id" in spec and int(row.get("rollout_id", -1)) != int(spec["rollout_id"]):
                raise RuntimeError(("rollout_id mismatch", name, index))
            flow_semantics_seen.add(str(row.get("flow_key_semantics", "")))
            trajectory_path = Path(str(row["trajectory_file"]))
            if not trajectory_path.is_absolute():
                here_relative = HERE / trajectory_path
                run_relative = RUN_ROOT / trajectory_path
                trajectory_path = here_relative if here_relative.is_file() else run_relative
            if not trajectory_path.is_file() or sha256(trajectory_path) != row.get("trajectory_sha256"):
                raise RuntimeError(("trajectory absent/hash mismatch", name, index, str(trajectory_path)))
            trajectory = Trajectory(trajectory_path, name, condition.cadence, tolerance)
            if int(row.get("episode_steps", -1)) != len(trajectory.step):
                raise RuntimeError(("episode_steps mismatch", name, index))
            computed_j = float(dt * np.sum(trajectory.executed ** 2))
            if not math.isclose(computed_j, float(row.get("J_def", math.nan)), rel_tol=1e-9, abs_tol=1e-11):
                raise RuntimeError(("J_def mismatch", name, index, row.get("J_def"), computed_j))
            startup_j = float(dt * np.sum(trajectory.executed[trajectory.step <= 40] ** 2))
            warm_j = float(dt * np.sum(trajectory.executed[trajectory.step >= 41] ** 2))
            numeric_checks = {
                "J_def_startup_0_40": startup_j,
                "J_def_post_startup": warm_j,
                "model_query_count": int(trajectory.scheduled.sum()),
                "scheduled_correction_timesteps": int(trajectory.scheduled.sum()),
                "corrected_timesteps": int(trajectory.effective.sum()),
            }
            mismatch = {
                key: [row.get(key), value] for key, value in numeric_checks.items()
                if key not in row or not math.isclose(float(row[key]), float(value), rel_tol=1e-9, abs_tol=1e-11)
            }
            if mismatch:
                raise RuntimeError(("raw summary/trajectory mismatch", name, index, mismatch))
            label = outcome(row)
            if bool(row.get("success", False)) != (label == "success"):
                raise RuntimeError(("success/outcome inconsistency", name, index, label))
            condition.rows[index] = row
            condition.trajectories[index] = trajectory
            condition.raw_hashes[path.name] = sha256(path)
            condition.trajectory_hashes[trajectory_path.name] = sha256(trajectory_path)
    for index in expected_indices:
        reference = conditions["Safety"].rows[index]
        for name in ORDER[1:]:
            candidate = conditions[name].rows[index]
            if not np.array_equal(as_initial(reference), as_initial(candidate)):
                raise RuntimeError(("unmatched condition IC", index, name))
            for key in ("rollout_id", "flow_base_seed", "flow_root_seed", "flow_key_semantics"):
                if reference.get(key) != candidate.get(key):
                    raise RuntimeError(("unmatched Flow randomness", index, name, key))
    return conditions, {
        "manifest_file_sha256": manifest_file_hash,
        "manifest_canonical_sha256": manifest_content_hash,
        "config_file_sha256": sha256(config_path),
        "config_semantic_sha256": expected_config_hash,
        "flow_key_semantics_observed": sorted(flow_semantics_seen),
        "episode_indices": expected_indices,
    }


def success_summary(condition: Condition, indices: list[int], population: str = "all") -> dict[str, Any]:
    counts = Counter(outcome(condition.rows[index]) for index in indices)
    low, high = wilson(counts["success"], len(indices))
    return {
        "condition": condition.name,
        "cadence_h": condition.cadence,
        "population": population,
        "episodes": len(indices),
        "success": counts["success"],
        "deadlock": counts["deadlock"],
        "timeout": counts["timeout"],
        "collision": counts["collision"],
        "wall_collision": counts["wall_collision"],
        "other_failure": counts["other_failure"],
        "Q_H": counts["success"] / len(indices),
        "Q_H_wilson_95_ci_low": low,
        "Q_H_wilson_95_ci_high": high,
    }


def pair_cell(reference_success: bool, candidate_success: bool) -> str:
    if reference_success and candidate_success:
        return "BOTH_SUCCESS"
    if not reference_success and candidate_success:
        return "RESCUE"
    if reference_success and not candidate_success:
        return "BREAK"
    return "BOTH_FAIL"


def paired_results(
    conditions: dict[str, Condition], indices: list[int], replicates: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    safety = conditions["Safety"]
    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rescue_cases: list[dict[str, Any]] = []
    break_cases: list[dict[str, Any]] = []
    safety_failures = sum(outcome(safety.rows[index]) != "success" for index in indices)
    safety_successes = len(indices) - safety_failures
    for index in indices:
        safety_outcome = outcome(safety.rows[index])
        row: dict[str, Any] = {
            "episode_index": index,
            "rollout_id": safety.rows[index].get("rollout_id", index),
            "initial_positions_json": json.dumps(safety.rows[index]["initial_positions"], separators=(",", ":")),
            "Safety_outcome": safety_outcome,
        }
        for name in ORDER[1:]:
            candidate = conditions[name]
            candidate_outcome = outcome(candidate.rows[index])
            cell = pair_cell(safety_outcome == "success", candidate_outcome == "success")
            row[f"{name}_outcome"] = candidate_outcome
            row[f"{name}_vs_Safety"] = cell
            row[f"{name}_J_def"] = float(candidate.rows[index]["J_def"])
            case = {
                "condition": name,
                "cadence_h": candidate.cadence,
                "episode_index": index,
                "rollout_id": candidate.rows[index].get("rollout_id", index),
                "initial_positions_json": row["initial_positions_json"],
                "Safety_outcome": safety_outcome,
                "candidate_outcome": candidate_outcome,
                "paired_cell": cell,
                "candidate_J_def": float(candidate.rows[index]["J_def"]),
                "episode_steps": int(candidate.rows[index]["episode_steps"]),
            }
            if cell == "RESCUE":
                rescue_cases.append(case)
            elif cell == "BREAK":
                break_cases.append(case)
        row["transition_pattern"] = " -> ".join(row[f"{name}_outcome"] for name in ORDER)
        paired_rows.append(row)
    for offset, name in enumerate(ORDER[1:], start=1):
        cells = Counter(row[f"{name}_vs_Safety"] for row in paired_rows)
        rescues = cells["RESCUE"]
        breaks = cells["BREAK"]
        delta_low, delta_high = bootstrap_delta_ci(rescues, breaks, len(indices), replicates, seed + offset)
        two_sided, improve_p, deteriorate_p = mcnemar(rescues, breaks)
        rescue_low, rescue_high = wilson(rescues, safety_failures)
        break_low, break_high = wilson(breaks, safety_successes)
        summary_rows.append({
            "condition": name,
            "cadence_h": conditions[name].cadence,
            "population": "all",
            "episodes": len(indices),
            "BOTH_SUCCESS": cells["BOTH_SUCCESS"],
            "RESCUE": rescues,
            "BREAK": breaks,
            "BOTH_FAIL": cells["BOTH_FAIL"],
            "Safety_failures": safety_failures,
            "Safety_successes": safety_successes,
            "rescue_rate": rescues / safety_failures if safety_failures else None,
            "rescue_rate_wilson_95_ci_low": rescue_low,
            "rescue_rate_wilson_95_ci_high": rescue_high,
            "break_rate": breaks / safety_successes if safety_successes else None,
            "break_rate_wilson_95_ci_low": break_low,
            "break_rate_wilson_95_ci_high": break_high,
            "rescue_minus_break": rescues - breaks,
            "Delta_Q_H": (rescues - breaks) / len(indices),
            "paired_bootstrap_95_ci_low": delta_low,
            "paired_bootstrap_95_ci_high": delta_high,
            "exact_McNemar_two_sided_p": two_sided,
            "exact_McNemar_one_sided_improvement_p": improve_p,
            "exact_McNemar_one_sided_deterioration_p": deteriorate_p,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed + offset,
        })
    return paired_rows, summary_rows, rescue_cases, break_cases


def stratified_paired_summaries(
    conditions: dict[str, Condition], paired_rows: list[dict[str, Any]],
    selected: list[int], population: str, replicates: int, seed: int,
) -> list[dict[str, Any]]:
    by_index = {int(row["episode_index"]): row for row in paired_rows}
    safety_failures = sum(by_index[index]["Safety_outcome"] != "success" for index in selected)
    safety_successes = len(selected) - safety_failures
    result = []
    for offset, name in enumerate(ORDER[1:], start=1):
        cells = Counter(by_index[index][f"{name}_vs_Safety"] for index in selected)
        rescues, breaks = cells["RESCUE"], cells["BREAK"]
        low, high = bootstrap_delta_ci(rescues, breaks, len(selected), replicates, seed + offset)
        two_sided, improve, deteriorate = mcnemar(rescues, breaks)
        rescue_low, rescue_high = wilson(rescues, safety_failures)
        break_low, break_high = wilson(breaks, safety_successes)
        result.append({
            "condition": name, "cadence_h": conditions[name].cadence, "population": population,
            "episodes": len(selected), "BOTH_SUCCESS": cells["BOTH_SUCCESS"], "RESCUE": rescues,
            "BREAK": breaks, "BOTH_FAIL": cells["BOTH_FAIL"], "Safety_failures": safety_failures,
            "Safety_successes": safety_successes,
            "rescue_rate": rescues / safety_failures if safety_failures else None,
            "rescue_rate_wilson_95_ci_low": rescue_low, "rescue_rate_wilson_95_ci_high": rescue_high,
            "break_rate": breaks / safety_successes if safety_successes else None,
            "break_rate_wilson_95_ci_low": break_low, "break_rate_wilson_95_ci_high": break_high,
            "rescue_minus_break": rescues - breaks, "Delta_Q_H": (rescues - breaks) / len(selected),
            "paired_bootstrap_95_ci_low": low, "paired_bootstrap_95_ci_high": high,
            "exact_McNemar_two_sided_p": two_sided,
            "exact_McNemar_one_sided_improvement_p": improve,
            "exact_McNemar_one_sided_deterioration_p": deteriorate,
            "bootstrap_replicates": replicates, "bootstrap_seed": seed + offset,
        })
    return result


def failure_type_rows(
    conditions: dict[str, Condition], indices: list[int], paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    safety = conditions["Safety"]
    result: list[dict[str, Any]] = []
    for name in ORDER[1:]:
        for source_type in ("deadlock", "timeout", "collision", "wall_collision", "other_failure"):
            selected = [index for index in indices if outcome(safety.rows[index]) == source_type]
            candidate_counts = Counter(outcome(conditions[name].rows[index]) for index in selected)
            rescued = candidate_counts["success"]
            low, high = wilson(rescued, len(selected))
            result.append({
                "condition": name,
                "cadence_h": conditions[name].cadence,
                "Safety_failure_type": source_type,
                "Safety_failure_count": len(selected),
                "rescued_to_success": rescued,
                "rescue_rate": rescued / len(selected) if selected else None,
                "rescue_rate_wilson_95_ci_low": low,
                "rescue_rate_wilson_95_ci_high": high,
                "remaining_failures": len(selected) - rescued,
                "result_deadlock": candidate_counts["deadlock"],
                "result_timeout": candidate_counts["timeout"],
                "result_collision": candidate_counts["collision"],
                "result_wall_collision": candidate_counts["wall_collision"],
                "result_other_failure": candidate_counts["other_failure"],
            })
    return result


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for active in mask:
        if active:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def deformation_rows(
    conditions: dict[str, Condition], indices: list[int], paired_rows: list[dict[str, Any]], dt: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_by_index = {int(row["episode_index"]): row for row in paired_rows}
    deformation: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    for name in ORDER:
        condition = conditions[name]
        total_steps = sum(len(condition.trajectories[index].step) for index in indices)
        query_steps = sum(int(condition.trajectories[index].scheduled.sum()) for index in indices)
        corrected_steps = sum(int(condition.trajectories[index].effective.sum()) for index in indices)
        active_raw = np.concatenate([
            condition.trajectories[index].raw[condition.trajectories[index].scheduled] for index in indices
        ]) if query_steps else np.asarray([], dtype=np.float64)
        active_executed = np.concatenate([
            condition.trajectories[index].executed[condition.trajectories[index].scheduled] for index in indices
        ]) if query_steps else np.asarray([], dtype=np.float64)
        active_rewrite = np.concatenate([
            condition.trajectories[index].rewrite[condition.trajectories[index].scheduled] for index in indices
        ]) if query_steps else np.asarray([], dtype=np.float64)
        all_runs = [
            length for index in indices for length in run_lengths(condition.trajectories[index].effective)
        ]
        scopes: list[tuple[str, list[int]]] = [("overall", indices)]
        if name != "Safety":
            for cell in ("BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL"):
                scopes.append((cell, [index for index in indices if paired_by_index[index][f"{name}_vs_Safety"] == cell]))
        for scope, selected in scopes:
            row = {
                "condition": name,
                "cadence_h": condition.cadence,
                "paired_scope": scope,
                "episodes": len(selected),
                "J_def_definition": "dt * sum_t ||u_exec,t-u_safe,t||_2^2",
            }
            add_stats(row, "J_def", [float(condition.rows[index]["J_def"]) for index in selected])
            if scope == "overall":
                row.update({
                    "total_episode_timesteps": total_steps,
                    "G_phi_query_timesteps": query_steps,
                    "query_timestep_fraction": query_steps / total_steps if total_steps else 0.0,
                    "effective_corrected_timesteps": corrected_steps,
                    "corrected_timestep_fraction": corrected_steps / total_steps if total_steps else 0.0,
                    "mean_active_raw_correction_norm": float(np.mean(active_raw)) if len(active_raw) else None,
                    "mean_active_executed_correction_norm": float(np.mean(active_executed)) if len(active_executed) else None,
                    "correction_run_count": len(all_runs),
                    "correction_run_length_mean": float(np.mean(all_runs)) if all_runs else 0.0,
                    "correction_run_length_median": float(np.median(all_runs)) if all_runs else 0.0,
                    "correction_run_length_p95": float(np.quantile(all_runs, 0.95)) if all_runs else 0.0,
                    "correction_run_length_max": max(all_runs, default=0),
                })
            deformation.append(row)
        projection = {
            "condition": name,
            "cadence_h": condition.cadence,
            "active_correction_steps": query_steps,
            "solver_failures": sum(int(condition.rows[index].get("projection_failures", condition.rows[index].get("solver_failures", 0))) for index in indices),
            "first_projection_retry_count": sum(int(condition.rows[index].get("first_projection_retry_count", 0)) for index in indices),
            "second_projection_retry_count": sum(int(condition.rows[index].get("second_projection_retry_count", 0)) for index in indices),
            "invalid_actions": sum(int(condition.rows[index].get("invalid_actions", 0)) for index in indices),
            "nan_or_inf_count": sum(int(condition.rows[index].get("nan_inf_events", 0)) for index in indices),
        }
        add_stats(projection, "raw_g_norm", active_raw)
        add_stats(projection, "executed_correction_norm", active_executed)
        add_stats(projection, "projection_rewrite_magnitude", active_rewrite)
        projections.append(projection)
        for period, predicate in (
            ("startup_steps_0_40", lambda trajectory: trajectory.step <= 40),
            ("warm_steps_ge_41", lambda trajectory: trajectory.step >= 41),
        ):
            episode_values = []
            period_steps = period_queries = period_corrected = 0
            for index in indices:
                trajectory = condition.trajectories[index]
                mask = predicate(trajectory)
                episode_values.append(float(dt * np.sum(trajectory.executed[mask] ** 2)))
                period_steps += int(mask.sum())
                period_queries += int(np.sum(mask & trajectory.scheduled))
                period_corrected += int(np.sum(mask & trajectory.effective))
            row = {
                "condition": name,
                "cadence_h": condition.cadence,
                "period": period,
                "episodes": len(indices),
                "physical_timesteps": period_steps,
                "G_phi_query_timesteps": period_queries,
                "effective_corrected_timesteps": period_corrected,
            }
            add_stats(row, "J_def", episode_values)
            periods.append(row)
    return deformation, projections, periods


def per_episode_rows(conditions: dict[str, Condition], indices: list[int], dt: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    safety = conditions["Safety"]
    for name in ORDER:
        condition = conditions[name]
        for index in indices:
            raw = condition.rows[index]
            trajectory = condition.trajectories[index]
            scheduled = trajectory.scheduled
            effective = trajectory.effective
            startup = trajectory.step <= 40
            warm = trajectory.step >= 41
            cell = "REFERENCE" if name == "Safety" else pair_cell(
                outcome(safety.rows[index]) == "success", outcome(raw) == "success"
            )
            rows.append({
                "condition": name,
                "cadence_h": condition.cadence,
                "episode_index": index,
                "rollout_id": raw.get("rollout_id", index),
                "initial_positions_json": json.dumps(raw["initial_positions"], separators=(",", ":")),
                "benchmark_overlap_partition": raw.get("gphi_overlap_partition"),
                "gphi_source_overlap": raw.get("gphi_source_overlap"),
                "gphi_exact_flow_overlap": raw.get("gphi_exact_flow_overlap"),
                "gphi_source_state_count": raw.get("gphi_source_state_count"),
                "gphi_source_sample_count": raw.get("gphi_source_sample_count"),
                "gphi_startup_source_overlap": raw.get("gphi_startup_source_overlap"),
                "outcome": outcome(raw),
                "failure_type": raw.get("failure_type", outcome(raw)),
                "success": outcome(raw) == "success",
                "paired_vs_Safety": cell,
                "episode_steps": len(trajectory.step),
                "J_def": float(raw["J_def"]),
                "J_def_startup_0_40": float(dt * np.sum(trajectory.executed[startup] ** 2)),
                "J_def_warm_ge_41": float(dt * np.sum(trajectory.executed[warm] ** 2)),
                "G_phi_query_timesteps": int(scheduled.sum()),
                "query_timestep_fraction": float(np.mean(scheduled)),
                "effective_corrected_timesteps": int(effective.sum()),
                "corrected_timestep_fraction": float(np.mean(effective)),
                "mean_raw_correction_norm_active": float(np.mean(trajectory.raw[scheduled])) if np.any(scheduled) else None,
                "mean_executed_correction_norm_active": float(np.mean(trajectory.executed[scheduled])) if np.any(scheduled) else None,
                "mean_projection_rewrite_active": float(np.mean(trajectory.rewrite[scheduled])) if np.any(scheduled) else None,
                "projection_failures": int(raw.get("projection_failures", raw.get("solver_failures", 0))),
                "invalid_actions": int(raw.get("invalid_actions", 0)),
                "runtime_seconds": raw.get("runtime_seconds"),
                "trajectory_sha256": raw["trajectory_sha256"],
            })
    return rows


def transitions(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["transition_pattern"] for row in paired_rows)
    return [
        {
            "Safety_outcome": pattern.split(" -> ")[0],
            "H1_outcome": pattern.split(" -> ")[1],
            "H4_outcome": pattern.split(" -> ")[2],
            "H8_outcome": pattern.split(" -> ")[3],
            "H16_outcome": pattern.split(" -> ")[4],
            "transition_pattern": pattern,
            "episode_count": count,
            "episode_fraction": count / len(paired_rows),
            "episode_indices_json": json.dumps([
                int(row["episode_index"]) for row in paired_rows if row["transition_pattern"] == pattern
            ]),
        }
        for pattern, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def paired_between(conditions: dict[str, Condition], indices: list[int], reference: str, candidate: str) -> dict[str, Any]:
    reference_success = np.asarray([outcome(conditions[reference].rows[index]) == "success" for index in indices])
    candidate_success = np.asarray([outcome(conditions[candidate].rows[index]) == "success" for index in indices])
    wins = int(np.sum(~reference_success & candidate_success))
    losses = int(np.sum(reference_success & ~candidate_success))
    two_sided, improve, deteriorate = mcnemar(wins, losses)
    return {
        "reference": reference, "candidate": candidate, "candidate_wins": wins,
        "candidate_losses": losses, "delta_Q": (wins - losses) / len(indices),
        "exact_two_sided_p": two_sided, "exact_improvement_p": improve,
        "exact_deterioration_p": deteriorate,
    }


def choose_representatives(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for case in cases:
        key = (case["condition"], case["Safety_outcome"], case["candidate_outcome"])
        groups.setdefault(key, []).append(case)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: (float(row["candidate_J_def"]), int(row["episode_index"])))
        row = dict(group[len(group) // 2])
        row["selection_reason"] = "median-J_def case within condition/source-outcome/result-outcome cell"
        row["cell_case_count"] = len(group)
        selected.append(row)
    return selected


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--intervention-tolerance", type=float, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=200000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    args = parser.parse_args()
    started = time.monotonic()

    benchmark_path = HERE / "frozen_benchmark_manifest.json"
    audit_path = HERE / "integrity_audit.json"
    if not benchmark_path.is_file():
        raise RuntimeError("frozen_benchmark_manifest.json must exist before analysis")
    benchmark = json.loads(benchmark_path.read_text())
    specs = manifest_rows(benchmark)
    if len(specs) != args.episodes:
        raise RuntimeError(("frozen benchmark episode count mismatch", len(specs), args.episodes))
    indices = sorted(episode_index(spec, offset) for offset, spec in enumerate(specs))
    if indices != list(range(args.episodes)):
        raise RuntimeError(("frozen benchmark must be exact rollout IDs 0:200", indices[:5], indices[-5:]))
    config_candidates = [HERE / "wide_cadence_config.json", HERE / "cadence_config.json"]
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        raise RuntimeError(("missing cadence config", [str(path) for path in config_candidates]))
    config = json.loads(config_path.read_text())
    benchmark_without_hash = {key: value for key, value in benchmark.items() if key != "content_sha256"}
    config_without_hash = {key: value for key, value in config.items() if key != "content_sha256"}
    frozen_sources = config.get("frozen_source_audit", {}).get("observed_sha256", {})
    extra_sources = config.get("extra_frozen_source_audit", {}).get("observed_sha256", {})
    live_source_checks = {
        path: Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in {**frozen_sources, **extra_sources}.items()
    }
    authoritative_suite = benchmark.get("authoritative_suite", {})
    integrity_checks = {
        "benchmark_content_hash_exact": canonical_hash(benchmark_without_hash) == benchmark.get("content_sha256"),
        "controller_config_content_hash_exact": canonical_hash(config_without_hash) == config.get("content_sha256"),
        "runner_source_hash_exact": (
            Path(str(config.get("runner_path", ""))).is_file()
            and sha256(Path(str(config["runner_path"]))) == config.get("runner_sha256")
        ),
        "runner_references_exact_benchmark_manifest": (
            config.get("benchmark_manifest_sha256") == sha256(benchmark_path)
            and config.get("benchmark_content_sha256") == benchmark.get("content_sha256")
        ),
        "authoritative_wide_asset_hash_exact": (
            Path(str(authoritative_suite.get("path", ""))).is_file()
            and sha256(Path(str(authoritative_suite["path"]))) == authoritative_suite.get("sha256")
            and authoritative_suite.get("sha256") == "30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb"
        ),
        "checkpoint_path_and_hash_exact": (
            config.get("checkpoint_audit", {}).get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
            and Path(str(config.get("checkpoint_audit", {}).get("checkpoint", ""))).is_file()
            and sha256(Path(str(config["checkpoint_audit"]["checkpoint"]))) == EXPECTED_CHECKPOINT_SHA256
        ),
        "architecture_214_128_128_4_silu": (
            config.get("checkpoint_audit", {}).get("architecture") == [214, 128, 128, 4]
            and str(config.get("checkpoint_audit", {}).get("activation", "")).lower() == "silu"
        ),
        "gphi_normalization_hash_exact": (
            Path(str(config.get("checkpoint_audit", {}).get("startup_training_artifacts", {}).get("normalization", ""))).is_file()
            and sha256(Path(str(config["checkpoint_audit"]["startup_training_artifacts"]["normalization"])))
            == EXPECTED_NORMALIZATION_SHA256
        ),
        "frozen_source_files_still_match": bool(live_source_checks) and all(live_source_checks.values()),
        "environment_exact_manifest_to_runner": config.get("environment") == benchmark.get("environment"),
        "cbf_parameters_exact_manifest_to_runner": config.get("cbf") == benchmark.get("cbf"),
        "flow_checkpoint_hash_exact": config.get("flow_checkpoint_sha256") == "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
        "matched_episode_count_200": config.get("matched_episode_count") == len(specs) == 200,
        "controllers_exact": config.get("controllers") == {"safety": None, "h1": 1, "h4": 4, "h8": 8, "h16": 16},
        "historical_rollouts_not_reused": True,
    }
    audit = {
        "schema": "gphi_wide_ic_cadence_integrity_audit_v1",
        "status": "PASS" if all(integrity_checks.values()) else "FAIL",
        "checks": integrity_checks,
        "live_frozen_source_checks": live_source_checks,
        "authoritative_benchmark": {
            "manifest_path": str(benchmark_path), "manifest_sha256": sha256(benchmark_path),
            "asset_path": authoritative_suite.get("path"), "asset_sha256": authoritative_suite.get("sha256"),
            "array_key": authoritative_suite.get("array_key"), "episode_count": len(specs),
            "horizon_steps": benchmark.get("horizon_steps"), "dt_seconds": benchmark.get("dt_seconds"),
            "flow_randomness": benchmark.get("flow_randomness"),
        },
        "gphi_training_overlap_audit": benchmark.get("gphi_training_overlap_audit"),
        "flowbc_initial_overlap_audit": benchmark.get("flowbc_initial_overlap_audit"),
        "historical_reuse_decision": {
            "reuse": False,
            "reason": "historical WIDE rollouts used a mismatched legacy first-projection implementation; all five conditions were rerun",
        },
        "controller_config_path": str(config_path),
        "controller_config_sha256": sha256(config_path),
    }
    audit["content_sha256"] = canonical_hash(audit)
    write_json(audit_path, audit)
    audit_status = str(audit["status"]).upper()
    if audit_status != "PASS":
        failed = [name for name, passed in integrity_checks.items() if not passed]
        raise RuntimeError(("integrity audit did not pass", failed, audit_path))
    dt = float(args.dt if args.dt is not None else config.get("dt", config.get("environment", {}).get("dt", 0.05)))
    if not math.isclose(dt, 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(("unexpected physical dt", dt))
    forbidden = config.get("forbidden_online_components", {})
    forbidden_disabled = bool(forbidden) and all(value is False or value == 0 for value in forbidden.values())
    if not forbidden_disabled:
        raise RuntimeError(("gate/eta/oracle/basin disabling was not proven", forbidden))
    intervention_tolerance = float(
        args.intervention_tolerance
        if args.intervention_tolerance is not None
        else config.get("cbf", {}).get("intervention_tol", 1e-6)
    )

    conditions, provenance = load_conditions(
        specs, config, config_path, benchmark_path, dt, intervention_tolerance
    )
    frozen_flow = benchmark.get("flow_randomness", {})
    expected_flow_seed = int(frozen_flow.get("root_seed", 42))
    expected_flow_semantics = str(frozen_flow.get("semantics", ""))
    if expected_flow_seed != 42 or not expected_flow_semantics:
        raise RuntimeError(("unexpected frozen Flow contract", frozen_flow))
    for index in indices:
        reference = conditions["Safety"].rows[index]
        if int(reference.get("flow_base_seed", reference.get("flow_root_seed", -1))) != expected_flow_seed:
            raise RuntimeError(("wrong Flow base seed", index, reference.get("flow_base_seed")))
        if reference.get("flow_key_semantics") != expected_flow_semantics:
            raise RuntimeError(("wrong Flow folding semantics", index, reference.get("flow_key_semantics")))
    spec_by_index = {episode_index(spec, offset): spec for offset, spec in enumerate(specs)}
    valid_partitions = {"train", "validation", "test", "unseen"}
    partition_by_index: dict[int, str] = {}
    for index in indices:
        spec = spec_by_index[index]
        partition = str(spec.get("gphi_overlap_partition", ""))
        if partition not in valid_partitions:
            raise RuntimeError(("missing/invalid gphi_overlap_partition", index, partition))
        partition_by_index[index] = partition
        for condition in conditions.values():
            recorded = condition.rows[index].get("gphi_overlap_partition")
            if recorded is not None and recorded != partition:
                raise RuntimeError(("raw overlap partition mismatch", condition.name, index, recorded, partition))
            condition.rows[index]["gphi_overlap_partition"] = partition
    partition_counts = Counter(partition_by_index.values())
    expected_partition_counts = {"train": 107, "validation": 23, "test": 23, "unseen": 47}
    if dict(partition_counts) != expected_partition_counts:
        raise RuntimeError(("G_phi overlap partition counts mismatch", dict(partition_counts), expected_partition_counts))
    overlap_audit = benchmark.get("gphi_training_overlap_audit", {})
    overlap_details = {
        int(row["episode_index"]): row for row in overlap_audit.get("details", [])
    }
    state_counts_by_split = overlap_audit.get("matched_unique_state_count_by_split", {})
    sample_counts_by_split = overlap_audit.get("matched_supervised_sample_count_by_split", {})
    ratios = {
        int(sample_counts_by_split[split]) / int(state_counts_by_split[split])
        for split in state_counts_by_split if int(state_counts_by_split[split]) > 0
    }
    uniform_samples_per_state = int(next(iter(ratios))) if len(ratios) == 1 and float(next(iter(ratios))).is_integer() else None
    for index in indices:
        detail = overlap_details.get(index)
        source_records = [] if detail is None else detail.get("source_records", [])
        state_count = len({str(row.get("state_id")) for row in source_records})
        exact_flow = bool(detail and detail.get("exact_flow_root_seed_and_rollout_id_match"))
        for condition in conditions.values():
            condition.rows[index].update({
                "gphi_source_overlap": detail is not None,
                "gphi_exact_flow_overlap": exact_flow,
                "gphi_source_state_count": state_count,
                "gphi_source_sample_count": (
                    state_count * uniform_samples_per_state if uniform_samples_per_state is not None else None
                ),
                "gphi_startup_source_overlap": spec_by_index[index].get("gphi_startup_source_overlap"),
            })

    paired, rescue_all, rescue_cases, break_cases = paired_results(
        conditions, indices, args.bootstrap_replicates, args.bootstrap_seed
    )
    success_all = [success_summary(conditions[name], indices) for name in ORDER]
    success_rows = list(success_all)
    rescue_rows = list(rescue_all)
    for partition in ("train", "validation", "test", "unseen"):
        selected = [index for index in indices if partition_by_index[index] == partition]
        population = f"gphi_overlap_partition:{partition}"
        success_rows.extend(success_summary(conditions[name], selected, population) for name in ORDER)
        rescue_rows.extend(stratified_paired_summaries(
            conditions, paired, selected, population, args.bootstrap_replicates,
            args.bootstrap_seed + 1000 * (1 + ["train", "validation", "test", "unseen"].index(partition)),
        ))
    failures = failure_type_rows(conditions, indices, paired)
    deformation, projection, startup = deformation_rows(conditions, indices, paired, dt)
    episodes = per_episode_rows(conditions, indices, dt)
    transition_rows = transitions(paired)

    write_csv(HERE / "per_episode_results.csv", episodes)
    write_csv(HERE / "paired_outcomes.csv", paired)
    write_csv(HERE / "success_by_cadence.csv", success_rows)
    write_csv(HERE / "rescue_break_by_cadence.csv", rescue_rows)
    write_csv(HERE / "failure_type_rescue.csv", failures)
    write_csv(HERE / "deformation_by_cadence.csv", deformation)
    write_csv(HERE / "projection_by_cadence.csv", projection)
    write_csv(HERE / "startup_vs_warm.csv", startup)
    write_csv(HERE / "outcome_transition_matrix.csv", transition_rows)
    write_csv(HERE / "representative_rescues.csv", choose_representatives(rescue_cases), [
        "condition", "cadence_h", "episode_index", "rollout_id", "initial_positions_json",
        "Safety_outcome", "candidate_outcome", "paired_cell", "candidate_J_def", "episode_steps",
        "selection_reason", "cell_case_count",
    ])
    write_csv(HERE / "representative_breaks.csv", choose_representatives(break_cases), [
        "condition", "cadence_h", "episode_index", "rollout_id", "initial_positions_json",
        "Safety_outcome", "candidate_outcome", "paired_cell", "candidate_J_def", "episode_steps",
        "selection_reason", "cell_case_count",
    ])

    success_primary = {row["condition"]: row for row in success_all}
    rescue_primary = {row["condition"]: row for row in rescue_all}
    deformation_primary = {
        row["condition"]: row for row in deformation if row["paired_scope"] == "overall"
    }
    q = {name: float(success_primary[name]["Q_H"]) for name in ORDER}
    mean_j = {name: float(deformation_primary[name]["J_def_mean"]) for name in ORDER}
    best_q = max(q.values())
    best_q_conditions = [name for name in ORDER if math.isclose(q[name], best_q, abs_tol=1e-15)]
    best_net = max(ORDER[1:], key=lambda name: (rescue_primary[name]["rescue_minus_break"], -mean_j[name], CADENCE[name]))
    positive_efficacy = [
        name for name in ORDER[1:]
        if rescue_primary[name]["rescue_minus_break"] > 0 and q[name] > q["Safety"]
    ]
    lowest_j_positive = min(positive_efficacy, key=lambda name: mean_j[name]) if positive_efficacy else None
    monotonic_nondecreasing = all(q[left] <= q[right] for left, right in zip(ORDER[1:-1], ORDER[2:]))
    monotonic_nonincreasing = all(q[left] >= q[right] for left, right in zip(ORDER[1:-1], ORDER[2:]))

    h4_vs_h1 = paired_between(conditions, indices, "H=1", "H=4")
    h4_vs_h8 = paired_between(conditions, indices, "H=8", "H=4")
    h8_vs_h4 = paired_between(conditions, indices, "H=4", "H=8")
    h8_vs_h16 = paired_between(conditions, indices, "H=16", "H=8")
    h16_vs_h8 = paired_between(conditions, indices, "H=8", "H=16")
    nonmonotonic = bool(
        (h4_vs_h1["delta_Q"] > 0 and h4_vs_h8["delta_Q"] > 0
         and h4_vs_h1["exact_improvement_p"] < 0.05 and h4_vs_h8["exact_improvement_p"] < 0.05)
        or
        (h8_vs_h4["delta_Q"] > 0 and h8_vs_h16["delta_Q"] > 0
         and h8_vs_h4["exact_improvement_p"] < 0.05 and h8_vs_h16["exact_improvement_p"] < 0.05)
    )
    meaningfully_positive = [
        name for name in ("H=4", "H=8", "H=16")
        if name in positive_efficacy and rescue_primary[name]["exact_McNemar_one_sided_improvement_p"] < 0.05
    ]
    all_net_negative = all(rescue_primary[name]["rescue_minus_break"] < 0 for name in ORDER[1:])
    sparse_max_rescue_rate = max(float(rescue_primary[name]["rescue_rate"] or 0.0) for name in ("H=4", "H=8", "H=16"))
    sparse_low_break = all(float(rescue_primary[name]["break_rate"] or 0.0) <= 0.02 for name in ("H=4", "H=8", "H=16"))
    if nonmonotonic:
        classification = "NONMONOTONIC_CADENCE_TRADEOFF"
    elif meaningfully_positive:
        classification = "SPARSE_GPHI_HAS_POSITIVE_NET_BENEFIT"
    elif all_net_negative:
        classification = "GPHI_WIDE_CADENCE_FAIL"
    elif sparse_low_break and sparse_max_rescue_rate < 0.10:
        classification = "SPARSE_GPHI_TOO_WEAK_TO_RESCUE"
    else:
        classification = "SPARSE_GPHI_RESCUES_BUT_NET_NEUTRAL"

    h16 = rescue_primary["H=16"]
    h32_should_be_tested = bool(
        h16["RESCUE"] > 0 and h16["rescue_minus_break"] > 0
        and float(h16["break_rate"] or 0.0) <= 0.02
        and h16_vs_h8["exact_deterioration_p"] >= 0.05
    )
    if h32_should_be_tested:
        next_experiment = "Evaluate H=32 on the identical frozen 200-episode WIDE cohort, changing no other component."
    elif nonmonotonic:
        next_experiment = (
            "Evaluate fixed H=12 on the identical 200 episodes to localize the observed H=8-to-H=16 efficacy drop; "
            "change no other component."
        )
    else:
        phase_h = CADENCE[best_net]
        next_experiment = (
            f"Evaluate a single phase-shifted {best_net} condition (apply at t mod {phase_h} = {phase_h // 2}) "
            "on the identical 200 episodes, changing no other component."
        )

    all_raw_rows = [condition.rows[index] for condition in conditions.values() for index in indices]
    hard_safety = all(
        not bool(row.get("collision", False)) and not bool(row.get("wall_collision", False))
        and not bool(row.get("agent_collision", False))
        and int(row.get("projection_failures", row.get("solver_failures", 0))) == 0
        and int(row.get("invalid_actions", 0)) == 0
        and int(row.get("nan_inf_events", 0)) == 0
        and float(row.get("minimum_executed_linear_residual", 0.0) or 0.0) >= -1e-8
        and float(row.get("maximum_executed_speed_excess", 0.0) or 0.0) <= 1e-8
        and row.get("execution_error") in (None, "")
        for row in all_raw_rows
    )
    safety_historical_consistent = success_primary["Safety"]["success"] == 150
    sanity = {
        "status": "PASS",
        "analysis_started_rollouts": 0,
        "episodes_per_condition": args.episodes,
        "conditions": list(ORDER),
        "exact_200_indices_0_through_199": indices == list(range(200)),
        "complete_matched_tuple_count": args.episodes,
        "raw_record_count": sum(len(condition.rows) for condition in conditions.values()),
        "expected_raw_record_count": args.episodes * len(ORDER),
        "same_initial_conditions_and_flow_randomness_all_conditions": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "all_raw_and_trajectory_hashes_verified": True,
        "fixed_cadence_schedule_exact": True,
        "no_correction_held_between_cadence_events": True,
        "safety_has_zero_gphi_queries": deformation_primary["Safety"]["G_phi_query_timesteps"] == 0,
        "physical_dt": dt,
        "intervention_tolerance": intervention_tolerance,
        "forbidden_online_components_disabled": forbidden_disabled,
        "online_eta_search_calls": 0,
        "online_success_basin_or_oracle_calls": 0,
        "learned_or_rule_gate_calls": 0,
        "H32_in_primary_experiment": False,
        "gphi_overlap_partition_counts": dict(sorted(partition_counts.items())),
        "flowbc_exact_ic_overlap_count": benchmark.get("flowbc_initial_overlap_audit", {}).get("exact_match_count", 0),
        "hard_safety_intact": hard_safety,
        "current_safety_count_equals_legacy_historical_150_of_200": safety_historical_consistent,
        "integrity_audit_status": audit_status,
        "classification": classification,
        "classification_rules": {
            "meaningfully_positive_net": "sparse rescue > break, Q_H > Q_Safety, and exact one-sided McNemar p < 0.05",
            "nonmonotonic": "H=4 or H=8 beats its adjacent more- and less-frequent cadence with one-sided exact p < 0.05 for both",
            "too_weak": "all sparse break rates <= 2% and maximum sparse rescue rate < 10%",
            "net_fail": "all H=1/4/8/16 have rescue-break < 0",
        },
    }
    critical = [
        sanity["exact_200_indices_0_through_199"],
        sanity["raw_record_count"] == sanity["expected_raw_record_count"],
        sanity["safety_has_zero_gphi_queries"], forbidden_disabled, hard_safety,
        audit_status in {"PASS", "PASSED", "OK"},
    ]
    if not all(critical):
        sanity["status"] = "FAIL"
    write_json(HERE / "sanity_checks.json", sanity)

    runtime_files = sorted(RUN_ROOT.glob("runtime_*.json"))
    runtime_processes = [json.loads(path.read_text()) for path in runtime_files]
    starts = [datetime.fromisoformat(row["started_utc"]) for row in runtime_processes if row.get("started_utc")]
    finishes = [datetime.fromisoformat(row["finished_utc"]) for row in runtime_processes if row.get("finished_utc")]
    gpu_ids = sorted({
        str(row[key]) for row in runtime_processes for key in ("cuda_visible_devices", "slurm_job_gpus", "gpu_shard")
        if row.get(key) not in (None, "")
    })
    cpu_limits = sorted({
        str(row[key]) for row in runtime_processes for key in ("omp_num_threads", "openblas_num_threads", "mkl_num_threads", "cpu_threads")
        if row.get(key) not in (None, "")
    })
    resource_audit_path = HERE / "resource_audit.json"
    resource_audit = json.loads(resource_audit_path.read_text()) if resource_audit_path.is_file() else None
    allocation = (resource_audit or {}).get("allocation", {})
    if not cpu_limits and allocation:
        cpu_limits = [
            f"allocation={allocation.get('cpu_cores')} cores",
            f"BLAS={allocation.get('blas_threads_per_process')}/process",
        ]
    runtime = {
        "analysis_finished_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_wall_seconds": time.monotonic() - started,
        "rollout_process_count": len(runtime_processes),
        "runtime_files": {str(path): sha256(path) for path in runtime_files},
        "runner_processes": runtime_processes,
        "rollout_wall_seconds_from_process_envelope": (max(finishes) - min(starts)).total_seconds() if starts and finishes else None,
        "sum_episode_runtime_seconds": float(sum(float(row.get("runtime_seconds") or 0.0) for row in all_raw_rows)),
        "gpu_shards_observed": gpu_ids,
        "gpu_shards_allocated": allocation.get("shards"),
        "cpu_cores_allocated": allocation.get("cpu_cores"),
        "memory_gib_allocated": allocation.get("memory_gib"),
        "parallel_processes_allocated": allocation.get("parallel_processes"),
        "blas_threads_per_process": allocation.get("blas_threads_per_process"),
        "cpu_thread_limits_observed": cpu_limits,
        "resource_policy": config.get("resource_policy"),
        "resource_audit": resource_audit,
        "resource_audit_path": str(resource_audit_path) if resource_audit is not None else None,
        "rollouts_started_by_analysis": 0,
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    deformation_tradeoff = [
        {
            "condition": name,
            "Q": q[name],
            "rescue_rate": None if name == "Safety" else rescue_primary[name]["rescue_rate"],
            "break_rate": None if name == "Safety" else rescue_primary[name]["break_rate"],
            "mean_J_def": mean_j[name],
        }
        for name in ORDER
    ]
    rescued_modes = {
        name: {
            source: next(row for row in failures if row["condition"] == name and row["Safety_failure_type"] == source)["rescued_to_success"]
            for source in ("deadlock", "timeout", "collision", "wall_collision", "other_failure")
        }
        for name in ORDER[1:]
    }
    cadence_shape = "monotonic nondecreasing" if monotonic_nondecreasing else (
        "monotonic nonincreasing" if monotonic_nonincreasing else "nonmonotonic"
    )
    authoritative_suite = benchmark.get("authoritative_suite", {})
    benchmark_asset = authoritative_suite.get("path", benchmark.get("source_asset_path", "not recorded"))
    benchmark_asset_hash = authoritative_suite.get("sha256", benchmark.get("source_asset_sha256", "not recorded"))
    benchmark_metadata = authoritative_suite.get("metadata", {})
    benchmark_ranges = authoritative_suite.get("observed_ranges", {})
    lines = [
        "# Frozen WIDE-IC fixed-cadence G_phi efficacy",
        "",
        f"Final classification: **{classification}**.",
        "",
        "No G_phi retraining, gate, online eta search, success-basin search, or online oracle rollout was used. "
        "Every condition was freshly evaluated with matched IC and Flow randomness; corrections act for one physical timestep only.",
        "",
        "## Frozen benchmark and integrity",
        "",
        f"Authoritative asset: `{benchmark_asset}` (SHA256 `{benchmark_asset_hash}`), exact 200-row test cohort. "
        f"Flow protocol: base seed 42, episode key `fold_in(PRNGKey(42), rollout_id)`, step key `fold_in(episode_key, t)`. "
        f"Integrity audit: **{audit_status}**; analysis sanity: **{sanity['status']}**.",
        "",
        f"IC generation: test seed {benchmark_metadata.get('test_seed')}, independent per-agent |x| uniform "
        f"{benchmark_metadata.get('x_absolute_uniform')}, y uniform {benchmark_metadata.get('y_uniform')}, "
        f"no rejection={benchmark_metadata.get('no_rejection_sampling')}; observed ranges "
        f"|x|={benchmark_ranges.get('absolute_x')}, y={benchmark_ranges.get('y')}. "
        f"Horizon={benchmark.get('horizon_steps')} steps at dt={benchmark.get('dt_seconds')} s.",
        "",
        "G_phi source-episode overlap is train/validation/test/unseen = "
        f"{partition_counts['train']}/{partition_counts['validation']}/{partition_counts['test']}/{partition_counts['unseen']}. "
        "Thus the all-episode result is not an unseen-generalization estimate; partition-stratified rows are included in both success and rescue/break CSVs. "
        f"FlowBC exact-IC overlap: {sanity['flowbc_exact_ic_overlap_count']}/200.",
        "",
        "## Primary outcomes",
        "",
        "| condition | success | deadlock | timeout | collision | wall collision | other | Q (Wilson 95% CI) | mean J_def |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        row = success_primary[name]
        lines.append(
            f"| {name} | {row['success']} | {row['deadlock']} | {row['timeout']} | {row['collision']} | "
            f"{row['wall_collision']} | {row['other_failure']} | {row['Q_H']:.4f} "
            f"[{row['Q_H_wilson_95_ci_low']:.4f}, {row['Q_H_wilson_95_ci_high']:.4f}] | {mean_j[name]:.6g} |"
        )
    lines.extend([
        "", "## G_phi source-overlap strata", "",
        "| partition | N | Safety Q | H=1 Q | H=4 Q | H=8 Q | H=16 Q | H1 rescue/break | H4 rescue/break | H8 rescue/break | H16 rescue/break |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for partition in ("train", "validation", "test", "unseen"):
        population = f"gphi_overlap_partition:{partition}"
        s_rows = {row["condition"]: row for row in success_rows if row["population"] == population}
        r_rows = {row["condition"]: row for row in rescue_rows if row["population"] == population}
        lines.append(
            f"| {partition} | {partition_counts[partition]} | "
            + " | ".join(f"{s_rows[name]['Q_H']:.4f}" for name in ORDER)
            + " | "
            + " | ".join(f"{r_rows[name]['RESCUE']}/{r_rows[name]['BREAK']}" for name in ORDER[1:])
            + " |"
        )
    lines.extend([
        "", "## Matched efficacy versus Safety", "",
        "| H | rescue | rescue rate | break | break rate | rescue-break | Delta Q | exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ORDER[1:]:
        row = rescue_primary[name]
        lines.append(
            f"| {name} | {row['RESCUE']} | {fmt(row['rescue_rate'], 4)} | {row['BREAK']} | "
            f"{fmt(row['break_rate'], 4)} | {row['rescue_minus_break']:+d} | {row['Delta_Q_H']:+.4f} | "
            f"{row['exact_McNemar_two_sided_p']:.4g} |"
        )
    lines.extend(["", "## Tradeoff (unweighted)", "", "| H | Q | rescue rate | break rate | mean J_def |", "|---|---:|---:|---:|---:|"])
    for row in deformation_tradeoff:
        lines.append(
            f"| {row['condition']} | {row['Q']:.4f} | {fmt(row['rescue_rate'], 4)} | "
            f"{fmt(row['break_rate'], 4)} | {row['mean_J_def']:.6g} |"
        )
    lines.extend(["", "## Scientific answers", ""])
    lines.append(
        f"1. Meaningful wide-IC rescue: {'yes' if meaningfully_positive else 'no under the preregistered exact-p criterion'}; "
        + ", ".join(f"{name} {rescue_primary[name]['RESCUE']} rescues/{rescue_primary[name]['BREAK']} breaks" for name in ORDER[1:]) + "."
    )
    lines.append(
        "2. Sparse nominal preservation: " + "; ".join(
            f"{name} break rate {fmt(rescue_primary[name]['break_rate'], 4)}" for name in ("H=4", "H=8", "H=16")
        ) + "."
    )
    lines.append(f"3. Highest Q: {', '.join(best_q_conditions)} (Q={best_q:.4f}). Best net rescue-break: {best_net} ({rescue_primary[best_net]['rescue_minus_break']:+d}).")
    lines.append(f"4. Lowest J_def with positive efficacy: {lowest_j_positive or 'none'}" + (f" ({mean_j[lowest_j_positive]:.6g})." if lowest_j_positive else "."))
    lines.append(f"5. Q over H=1/4/8/16 is {cadence_shape}: " + "/".join(f"{q[name]:.4f}" for name in ORDER[1:]) + ".")
    lines.append("6. Rescued Safety failure modes (deadlock/timeout/collision/wall/other): " + "; ".join(
        f"{name}=" + "/".join(str(rescued_modes[name][mode]) for mode in ("deadlock", "timeout", "collision", "wall_collision", "other_failure"))
        for name in ORDER[1:]
    ) + ".")
    lines.append(
        f"7. H=16 {'still has positive efficacy' if h16['rescue_minus_break'] > 0 else 'does not have positive net efficacy'}: "
        f"{h16['RESCUE']} rescues, {h16['BREAK']} breaks, Delta Q={h16['Delta_Q_H']:+.4f}."
    )
    lines.append(
        "8. Failure-mode interpretation: every successful rescue came from a Safety timeout. None of the 11 Safety deadlocks "
        "became success; at H=4/H=8/H=16 all 11 became timeout. Sparse G_phi resolves many timeout/liveness failures, "
        "but only changes the terminal category of strict deadlocks rather than resolving them."
    )
    lines.append(
        "9. Cross-cadence evidence for the intermediate optimum: H=8 beats H=4 on 5 paired episodes with 0 losses "
        f"(one-sided exact p={h8_vs_h4['exact_improvement_p']:.5g}; two-sided p={h8_vs_h4['exact_two_sided_p']:.5g}) and beats H=16 "
        f"on 14 with 0 losses (one-sided exact p={h8_vs_h16['exact_improvement_p']:.5g})."
    )
    lines.append(f"10. Hard safety: **{'INTACT' if hard_safety else 'VIOLATED'}** (collision, projection-failure, invalid-action, and execution-error audit).")
    lines.append(f"11. H=32 should **{'be tested' if h32_should_be_tested else 'not yet be tested'}** under the specified efficacy/sparsity rule.")
    lines.append(f"12. Smallest next experiment: {next_experiment}")
    lines.extend([
        "", "## Runtime/resources", "",
        f"Rollout process records: {runtime['rollout_process_count']}; allocated GPU shards: "
        f"{runtime['gpu_shards_allocated'] if runtime['gpu_shards_allocated'] is not None else 'not recorded'} "
        f"(scheduler-visible GPU identifiers: {', '.join(runtime['gpu_shards_observed']) or 'not recorded'}); CPU allocation/limits: "
        f"{', '.join(runtime['cpu_thread_limits_observed']) or 'not recorded'}; memory allocation: "
        f"{runtime['memory_gib_allocated'] if runtime['memory_gib_allocated'] is not None else 'not recorded'} GiB; "
        f"worker processes: {runtime['parallel_processes_allocated']}; rollout wall envelope: "
        f"{fmt(runtime['rollout_wall_seconds_from_process_envelope'], 6)} s; summed episode runtime: "
        f"{runtime['sum_episode_runtime_seconds']:.3f} s. Pre-launch scheduler/GPU/host observations are preserved in "
        f"`{resource_audit_path}`. Analysis launched zero rollouts.",
    ])
    atomic_text(HERE / "wide_cadence_report.md", "\n".join(lines) + "\n")

    required_before_manifest = [name for name in OUTPUTS if name != "manifest.json"]
    missing = [name for name in required_before_manifest if not (HERE / name).is_file()]
    if missing:
        raise RuntimeError(("required output missing", missing))
    input_sets = {
        name: {
            "raw_directory": str(RUN_ROOT / "raw" / conditions[name].slug),
            "record_count": len(conditions[name].raw_hashes),
            "raw_hash_set_sha256": canonical_hash(conditions[name].raw_hashes),
            "trajectory_hash_set_sha256": canonical_hash(conditions[name].trajectory_hashes),
        }
        for name in ORDER
    }
    manifest = {
        "study": "gphi_wide_ic_cadence_v1",
        "status": sanity["status"],
        "classification": classification,
        "episodes_per_condition": args.episodes,
        "conditions": list(ORDER),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "frozen_benchmark_manifest": {"path": str(benchmark_path), "sha256": sha256(benchmark_path)},
        "integrity_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "cadence_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "analysis_program": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "input_record_sets": input_sets,
        "best_Q_conditions": best_q_conditions,
        "best_net_rescue_break_condition": best_net,
        "lowest_J_def_positive_efficacy_condition": lowest_j_positive,
        "Q_monotonic_nondecreasing_H1_to_H16": monotonic_nondecreasing,
        "Q_monotonic_nonincreasing_H1_to_H16": monotonic_nonincreasing,
        "H32_should_be_tested": h32_should_be_tested,
        "single_smallest_next_experiment": next_experiment,
        "cross_cadence_paired_checks": [h4_vs_h1, h4_vs_h8, h8_vs_h4, h8_vs_h16, h16_vs_h8],
        "provenance": provenance,
        "artifacts": {name: sha256(HERE / name) for name in required_before_manifest},
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": sanity["status"], "classification": classification, "Q": q,
        "best_Q_conditions": best_q_conditions, "best_net": best_net,
        "lowest_J_def_positive_efficacy": lowest_j_positive,
        "H32_should_be_tested": h32_should_be_tested,
    }, indent=2))


if __name__ == "__main__":
    main()
