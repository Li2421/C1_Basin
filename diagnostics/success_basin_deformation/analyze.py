"""Compute provisional executed-action deformation from cached SBMA traces.

This module never imports or invokes the controller, Flow policy, projector, or
environment.  It reads the two actions saved in the same physical-step record:

    delta_u[k] = u_exec[k] - u_safe[k]
    J_def       = dt * sum_k ||delta_u[k]||_F**2

Consequently it cannot alter rollout behavior.  Numerical/solver-incomplete
attempts receive no full-horizon J_def and are replaced in aggregates only by
an existing completed retry with the same (state, eta, seed), matching the
frozen SBMA repair semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "diagnostics" / "success_basin_multimodality"
OUT = Path(__file__).resolve().parent
EVENTS = ("success", "deadlock", "timeout", "collision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def eta_key(eta: object) -> tuple[float, float, float]:
    values = np.asarray(eta, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"invalid eta: {eta!r}")
    return tuple(float(x) for x in np.round(values, 12))


def one_sided_interval(k: int, n: int, alpha: float = 0.05) -> list[float]:
    return [
        float(beta.ppf(alpha, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0,
    ]


def classify_cell(successes: int, n: int, unresolved: int) -> tuple[str, list[float]]:
    interval = one_sided_interval(successes, n) if n else [0.0, 1.0]
    if unresolved or n < 16:
        label = "UNKNOWN_CELL"
    elif interval[0] >= 0.80:
        label = "SUCCESS_CELL"
    elif interval[1] <= 0.20:
        label = "FAILURE_CELL"
    else:
        label = "UNKNOWN_CELL"
    return label, interval


def cohort(seed: int, stage: str) -> str:
    if 95101001 <= seed <= 95101016:
        return "phase_a"
    if 95102001 <= seed <= 95102064:
        return "known_validation"
    if 95103001 <= seed <= 95103032:
        return "straight_interpolation"
    if 95104001 <= seed <= 95104016:
        return "basin_margin"
    return stage


def summarize_group(
    state_id: str,
    eta: tuple[float, float, float],
    records: list[dict],
    unresolved: int = 0,
    cohort_name: str | None = None,
) -> dict:
    counts = Counter(row["terminal_outcome"] for row in records)
    n = len(records)
    label, interval = classify_cell(counts["success"], n, unresolved)
    values = np.asarray([row["J_def"] for row in records], dtype=np.float64)
    steps = np.asarray([row["episode_length_steps"] for row in records], dtype=np.float64)
    answer = {
        "state_id": state_id,
        "eta": list(eta),
        "n": n,
        "unresolved": int(unresolved),
        "counts": {event: int(counts[event]) for event in EVENTS},
        "success_rate": float(counts["success"] / n) if n else None,
        "Q_S_one_sided_95": interval,
        "classification": label,
        "mean_J_def": float(np.mean(values)) if n else None,
        "std_J_def": float(np.std(values, ddof=1)) if n > 1 else 0.0 if n else None,
        "min_J_def": float(np.min(values)) if n else None,
        "max_J_def": float(np.max(values)) if n else None,
        "mean_episode_length_steps": float(np.mean(steps)) if n else None,
        "mean_episode_length_seconds": float(np.mean([r["episode_length_seconds"] for r in records])) if n else None,
        "seeds": sorted(int(row["seed"]) for row in records),
        "source_stages": sorted({row["stage"] for row in records}),
    }
    if cohort_name is not None:
        answer["cohort"] = cohort_name
    return answer


def phase_a_boundary(
    state_id: str,
    eta: tuple[float, float, float],
    phase_cells: dict[tuple[str, tuple[float, float, float]], dict],
    axes: list[list[float]],
) -> dict:
    if any(value not in axis for value, axis in zip(eta, axes)):
        return {
            "on_phase_a_grid": False,
            "location": "outside_phase_a_lattice_additional_probe",
        }
    indices = [axis.index(value) for value, axis in zip(eta, axes)]
    on_tested_grid_boundary = any(
        index in (0, len(axis) - 1) for index, axis in zip(indices, axes)
    )
    neighbors = []
    for dimension in range(3):
        for offset in (-1, 1):
            neighbor_indices = list(indices)
            neighbor_indices[dimension] += offset
            if not 0 <= neighbor_indices[dimension] < len(axes[dimension]):
                continue
            neighbor = tuple(
                float(axes[d][neighbor_indices[d]]) for d in range(3)
            )
            cell = phase_cells[(state_id, neighbor)]
            neighbors.append(
                {"eta": list(neighbor), "classification": cell["classification"]}
            )
    touches_non_success = any(
        item["classification"] != "SUCCESS_CELL" for item in neighbors
    )
    if on_tested_grid_boundary:
        location = "tested_grid_boundary"
    elif touches_non_success:
        location = "near_success_failure_or_unknown_boundary"
    else:
        location = "interior_of_empirical_phase_a_success_region"
    return {
        "on_phase_a_grid": True,
        "on_tested_grid_boundary": on_tested_grid_boundary,
        "touches_non_success_neighbor": touches_non_success,
        "location": location,
        "axis_neighbors": neighbors,
    }


def paired_winner_check(
    cells: list[dict], records_by_cell: dict[tuple[str, tuple[float, float, float], str], list[dict]]
) -> dict:
    successful = sorted(
        (cell for cell in cells if cell["classification"] == "SUCCESS_CELL"),
        key=lambda cell: (cell["mean_J_def"], cell["eta"]),
    )
    winner, runner = successful[:2]
    sid = winner["state_id"]
    winner_eta, runner_eta = eta_key(winner["eta"]), eta_key(runner["eta"])
    winner_rows = {
        row["seed"]: row["J_def"]
        for row in records_by_cell[(sid, winner_eta, "phase_a")]
    }
    runner_rows = {
        row["seed"]: row["J_def"]
        for row in records_by_cell[(sid, runner_eta, "phase_a")]
    }
    seeds = sorted(set(winner_rows) & set(runner_rows))
    if set(seeds) != set(winner_rows) or set(seeds) != set(runner_rows):
        raise AssertionError("Phase-A winner/runner do not have identical CRN seeds")
    differences = np.asarray(
        [runner_rows[seed] - winner_rows[seed] for seed in seeds], dtype=np.float64
    )
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    critical = float(student_t.ppf(0.975, len(differences) - 1))
    return {
        "winner_eta": list(winner_eta),
        "winner_mean_J_def": winner["mean_J_def"],
        "runner_up_eta": list(runner_eta),
        "runner_up_mean_J_def": runner["mean_J_def"],
        "paired_seed_count": len(seeds),
        "mean_runner_minus_winner": mean,
        "paired_95_interval": [mean - critical * standard_error, mean + critical * standard_error],
        "winner_lower_on_all_paired_seeds": bool(np.all(differences > 0)),
    }


def format_eta(eta: list[float] | tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:g}" for value in eta) + ")"


def main() -> None:
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    dt = float(protocol["environment"]["dt"])
    axes = [
        [float(x) for x in protocol["phase_a_design"]["axes"][name]]
        for name in ("goal", "safe", "relative")
    ]
    manifest_paths = sorted((SOURCE / "raw").glob("*/manifest.json"))
    attempts = []
    input_paths = set(manifest_paths)
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        stage = manifest_path.parent.name
        for record in manifest["records"]:
            attempts.append((stage, record))
            input_paths.add(SOURCE / record["file"])
    before_stats = {
        str(path.relative_to(ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_paths
    }

    per_rollout = []
    max_w_reconstruction_error = 0.0
    max_integration_error = 0.0
    max_j_loop_error = 0.0
    hash_mismatches = []
    terminal_mismatches = []
    dimension_mismatches = []
    zero_equivalence_failures = []
    actual_zero_j_count = 0
    actual_all_equal_count = 0

    for stage, record in attempts:
        path = SOURCE / record["file"]
        observed_sha = sha256(path)
        if observed_sha != record["sha256"]:
            hash_mismatches.append(str(path.relative_to(ROOT)))
        with np.load(path, allow_pickle=False) as trace:
            u_safe = np.asarray(trace["u_safe"], dtype=np.float64)
            u_exec = np.asarray(trace["u_exec"], dtype=np.float64)
            g = np.asarray(trace["g"], dtype=np.float64)
            w = np.asarray(trace["w"], dtype=np.float64)
            positions_before = np.asarray(trace["positions_before"], dtype=np.float64)
            positions_after = np.asarray(trace["positions_after"], dtype=np.float64)
            events = np.asarray(trace["event"])
        if not (
            u_safe.shape == u_exec.shape == g.shape == w.shape
            and u_safe.ndim == 3
            and u_safe.shape[1:] == (2, 2)
            and len(u_safe) == int(record["steps"])
        ):
            dimension_mismatches.append(str(path.relative_to(ROOT)))
        if len(u_safe):
            max_w_reconstruction_error = max(
                max_w_reconstruction_error,
                float(np.max(np.abs(w - u_safe - g))),
            )
            max_integration_error = max(
                max_integration_error,
                float(np.max(np.abs(positions_after - positions_before - dt * u_exec))),
            )
        delta = u_exec - u_safe
        per_step_squared = np.sum(delta * delta, axis=(1, 2), dtype=np.float64)
        partial_j = float(dt * np.sum(per_step_squared, dtype=np.float64))
        loop_j = float(
            sum(dt * float(np.dot(step.reshape(-1), step.reshape(-1))) for step in delta)
        )
        max_j_loop_error = max(max_j_loop_error, abs(partial_j - loop_j))
        all_equal = bool(np.array_equal(u_exec, u_safe))
        zero_j = partial_j == 0.0
        actual_all_equal_count += int(all_equal)
        actual_zero_j_count += int(zero_j)
        if zero_j != all_equal:
            zero_equivalence_failures.append(str(path.relative_to(ROOT)))
        completed = record["outcome"] is not None
        if completed and (not len(events) or str(events[-1]) != record["outcome"]):
            terminal_mismatches.append(str(path.relative_to(ROOT)))
        row = {
            "state_id": record["state_id"],
            "eta": list(eta_key(record["eta"])),
            "seed": int(record["seed"]),
            "stage": stage,
            "source_file": record["file"],
            "source_sha256": record["sha256"],
            "terminal_outcome": record["outcome"],
            "execution_error": record["execution_error"],
            "episode_length_steps": int(record["steps"]),
            "episode_length_seconds": float(dt * int(record["steps"])),
            "J_def": partial_j if completed else None,
            "partial_J_def_if_unresolved": None if completed else partial_j,
            "max_step_squared_deformation": float(np.max(per_step_squared)) if len(per_step_squared) else 0.0,
            "mean_step_squared_deformation": float(np.mean(per_step_squared)) if len(per_step_squared) else 0.0,
            "all_steps_u_exec_exactly_equal_u_safe": all_equal,
            "cohort": cohort(int(record["seed"]), stage),
            "superseded_by_completed_same_seed_retry": False,
        }
        per_rollout.append(row)

    # The zero identity is also exercised directly even if no cached policy has
    # u_exec == u_safe at every physical step.
    synthetic_equal = np.zeros((7, 2, 2), dtype=np.float64)
    synthetic_zero_j = float(
        dt * np.sum((synthetic_equal - synthetic_equal) ** 2, dtype=np.float64)
    )
    if synthetic_zero_j != 0.0:
        raise AssertionError("synthetic zero-deformation identity failed")

    # Completed same-key retries replace only an incomplete numerical attempt.
    identity_groups = defaultdict(list)
    for row in per_rollout:
        identity_groups[(row["state_id"], eta_key(row["eta"]), row["seed"])].append(row)
    effective = []
    superseded_count = 0
    duplicate_completed = []
    for identity, rows in identity_groups.items():
        completed = [row for row in rows if row["J_def"] is not None]
        if len(completed) > 1:
            duplicate_completed.append({"identity": [identity[0], list(identity[1]), identity[2]], "files": [row["source_file"] for row in completed]})
        if completed:
            chosen = completed[0]
            effective.append(chosen)
            for row in rows:
                if row is not chosen and row["J_def"] is None:
                    row["superseded_by_completed_same_seed_retry"] = True
                    superseded_count += 1
        else:
            effective.append(rows[0])
    if duplicate_completed:
        raise AssertionError(f"duplicate completed retries: {duplicate_completed}")

    records_by_cell = defaultdict(list)
    unresolved_by_cell = Counter()
    all_records_by_cell = defaultdict(list)
    all_unresolved_by_cell = Counter()
    for row in effective:
        key = (row["state_id"], eta_key(row["eta"]), row["cohort"])
        all_key = (row["state_id"], eta_key(row["eta"]))
        if row["J_def"] is None:
            unresolved_by_cell[key] += 1
            all_unresolved_by_cell[all_key] += 1
        else:
            records_by_cell[key].append(row)
            all_records_by_cell[all_key].append(row)

    cohort_cells = []
    for key in sorted(set(records_by_cell) | set(unresolved_by_cell)):
        sid, eta, cohort_name = key
        cohort_cells.append(
            summarize_group(
                sid,
                eta,
                records_by_cell[key],
                unresolved_by_cell[key],
                cohort_name,
            )
        )
    all_cells = []
    for key in sorted(set(all_records_by_cell) | set(all_unresolved_by_cell)):
        sid, eta = key
        all_cells.append(
            summarize_group(
                sid, eta, all_records_by_cell[key], all_unresolved_by_cell[key]
            )
        )

    phase_a_cells = [cell for cell in cohort_cells if cell["cohort"] == "phase_a"]
    phase_a_lookup = {(cell["state_id"], eta_key(cell["eta"])): cell for cell in phase_a_cells}
    eta_stars = {}
    for state in ("D1_pair231", "D2_pair228", "D4_pair227"):
        phase_success = sorted(
            [cell for cell in phase_a_cells if cell["state_id"] == state and cell["classification"] == "SUCCESS_CELL"],
            key=lambda cell: (cell["mean_J_def"], cell["eta"]),
        )
        all_success = sorted(
            [cell for cell in all_cells if cell["state_id"] == state and cell["classification"] == "SUCCESS_CELL"],
            key=lambda cell: (cell["mean_J_def"], cell["eta"]),
        )
        phase_winner = phase_success[0]
        expanded_winner = all_success[0]
        eta_stars[state] = {
            "phase_a_CRN_grid": {
                "eta": phase_winner["eta"],
                "mean_J_def": phase_winner["mean_J_def"],
                "std_J_def": phase_winner["std_J_def"],
                "success_rate": phase_winner["success_rate"],
                "n": phase_winner["n"],
                "boundary": phase_a_boundary(state, eta_key(phase_winner["eta"]), phase_a_lookup, axes),
                "winner_vs_runner_up": paired_winner_check(
                    [cell for cell in phase_a_cells if cell["state_id"] == state], records_by_cell
                ),
            },
            "all_existing_tested_points": {
                "eta": expanded_winner["eta"],
                "mean_J_def": expanded_winner["mean_J_def"],
                "std_J_def": expanded_winner["std_J_def"],
                "success_rate": expanded_winner["success_rate"],
                "n": expanded_winner["n"],
                "source_stages": expanded_winner["source_stages"],
                "boundary": phase_a_boundary(state, eta_key(expanded_winner["eta"]), phase_a_lookup, axes),
            },
        }

    # Find the common Phase-A policy with the smallest worst relative gap from
    # the three state-wise Phase-A minima.  Five percent is reporting-only and
    # never changes the winner or success classification.
    successful_sets = []
    for state in eta_stars:
        successful_sets.append(
            {
                eta_key(cell["eta"])
                for cell in phase_a_cells
                if cell["state_id"] == state and cell["classification"] == "SUCCESS_CELL"
            }
        )
    common_success = set.intersection(*successful_sets)
    common_rows = []
    for eta in sorted(common_success):
        per_state = {}
        gaps = []
        for state in eta_stars:
            cell = phase_a_lookup[(state, eta)]
            minimum = eta_stars[state]["phase_a_CRN_grid"]["mean_J_def"]
            gap = float(cell["mean_J_def"] / minimum - 1.0)
            gaps.append(gap)
            state_success = sorted(
                [c for c in phase_a_cells if c["state_id"] == state and c["classification"] == "SUCCESS_CELL"],
                key=lambda c: (c["mean_J_def"], c["eta"]),
            )
            rank = 1 + next(i for i, c in enumerate(state_success) if eta_key(c["eta"]) == eta)
            per_state[state] = {"mean_J_def": cell["mean_J_def"], "relative_gap": gap, "rank": rank}
        common_rows.append({"eta": list(eta), "maximum_relative_gap": max(gaps), "mean_relative_gap": float(np.mean(gaps)), "states": per_state})
    best_common = min(common_rows, key=lambda row: (row["maximum_relative_gap"], row["mean_relative_gap"], row["eta"]))
    common_assessment = {
        "definition_of_near_minimal": "within 5% of the state-specific Phase-A mean-J minimum; reporting criterion only",
        "same_exact_phase_a_minimizer_all_three": len({tuple(eta_stars[s]["phase_a_CRN_grid"]["eta"]) for s in eta_stars}) == 1,
        "best_common_eta": best_common,
        "near_minimal_all_three_at_5_percent": best_common["maximum_relative_gap"] <= 0.05,
    }

    known = {}
    for state in eta_stars:
        cell = next(
            cell for cell in cohort_cells
            if cell["state_id"] == state
            and cell["cohort"] == "known_validation"
            and eta_key(cell["eta"]) == (1.0, 0.0, 0.25)
        )
        known[state] = cell
    pooled_known_rows = [
        row for row in effective
        if row["J_def"] is not None
        and row["cohort"] == "known_validation"
        and eta_key(row["eta"]) == (1.0, 0.0, 0.25)
    ]
    pooled_values = np.asarray([row["J_def"] for row in pooled_known_rows])
    known["pooled_descriptive_only"] = {
        "n": len(pooled_known_rows),
        "success_count": sum(row["terminal_outcome"] == "success" for row in pooled_known_rows),
        "mean_J_def": float(np.mean(pooled_values)),
        "std_J_def": float(np.std(pooled_values, ddof=1)),
        "min_J_def": float(np.min(pooled_values)),
        "max_J_def": float(np.max(pooled_values)),
        "warning": "Pooled value mixes three different starting states and is descriptive only.",
    }

    successful_all = sorted(
        [cell for cell in all_cells if cell["classification"] == "SUCCESS_CELL"],
        key=lambda cell: (cell["state_id"], cell["mean_J_def"], cell["eta"]),
    )
    pathologies = {}
    for state in eta_stars:
        state_cells = sorted(
            [cell for cell in all_cells if cell["state_id"] == state],
            key=lambda cell: (cell["mean_J_def"], cell["eta"]),
        )
        state_success = [
            cell for cell in state_cells if cell["classification"] == "SUCCESS_CELL"
        ]
        correlation = spearmanr(
            [cell["mean_J_def"] for cell in state_success],
            [cell["mean_episode_length_steps"] for cell in state_success],
        )
        pathologies[state] = {
            "lowest_J_over_all_outcome_classes": state_cells[0],
            "lowest_J_is_success_cell": state_cells[0]["classification"] == "SUCCESS_CELL",
            "successful_cell_J_vs_episode_length_spearman": float(correlation.statistic),
            "successful_cell_J_vs_episode_length_p": float(correlation.pvalue),
            "successful_cell_count": len(state_success),
        }
    with (OUT / "successful_eta_table.csv").open("w", newline="") as stream:
        fields = [
            "state_id", "eta1", "eta2", "eta3", "success_rate", "mean_J_def",
            "std_J_def", "mean_episode_length_steps", "mean_episode_length_seconds",
            "n", "success_count", "deadlock_count", "timeout_count", "collision_count",
            "source_stages",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in successful_all:
            writer.writerow({
                "state_id": cell["state_id"],
                "eta1": cell["eta"][0], "eta2": cell["eta"][1], "eta3": cell["eta"][2],
                "success_rate": cell["success_rate"], "mean_J_def": cell["mean_J_def"],
                "std_J_def": cell["std_J_def"],
                "mean_episode_length_steps": cell["mean_episode_length_steps"],
                "mean_episode_length_seconds": cell["mean_episode_length_seconds"],
                "n": cell["n"], "success_count": cell["counts"]["success"],
                "deadlock_count": cell["counts"]["deadlock"],
                "timeout_count": cell["counts"]["timeout"],
                "collision_count": cell["counts"]["collision"],
                "source_stages": ";".join(cell["source_stages"]),
            })

    with (OUT / "per_rollout_j_def.jsonl").open("w") as stream:
        for row in per_rollout:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    after_stats = {
        str(path.relative_to(ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_paths
    }
    changed_inputs = sorted(path for path in before_stats if before_stats[path] != after_stats[path])
    sanity = {
        "formula": "J_def = dt * sum_k sum_agents,sum_xy (u_exec[k]-u_safe[k])^2",
        "dt": dt,
        "weight": "identity",
        "discount": None,
        "normalization": None,
        "raw_attempts": len(per_rollout),
        "effective_state_eta_seed_records": len(effective),
        "completed_effective_rollouts": sum(row["J_def"] is not None for row in effective),
        "unresolved_effective_rollouts": sum(row["J_def"] is None for row in effective),
        "superseded_incomplete_attempts": superseded_count,
        "source_sha256_mismatches": hash_mismatches,
        "source_size_or_mtime_changes_during_analysis": changed_inputs,
        "dimension_mismatches": dimension_mismatches,
        "terminal_mismatches": terminal_mismatches,
        "maximum_w_minus_u_safe_minus_g_error": max_w_reconstruction_error,
        "maximum_position_integration_error": max_integration_error,
        "maximum_vectorized_vs_step_loop_J_error": max_j_loop_error,
        "actual_full_rollouts_with_exact_zero_J": actual_zero_j_count,
        "actual_full_rollouts_with_u_exec_exactly_equal_u_safe": actual_all_equal_count,
        "zero_iff_exact_action_equality_failures": zero_equivalence_failures,
        "synthetic_exact_equal_action_J": synthetic_zero_j,
        "post_projection_provenance": "u_exec is assigned from project_velocity_with_retry(w,...) before both u_safe and u_exec are appended to the same per-step hist record in run.py lines 51-60",
        "same_state_same_flow_provenance": "u_safe and u_exec are from the same inner-loop iteration, observation, A,b and sampled Flow action in run.py lines 44-60",
        "existing_independent_projection_replay": json.loads((SOURCE / "verification.json").read_text()),
    }
    if any((hash_mismatches, changed_inputs, dimension_mismatches, terminal_mismatches, zero_equivalence_failures)):
        raise AssertionError("deformation audit sanity checks failed")

    results = {
        "status": "PROVISIONAL_EMPIRICAL_TEST_ONLY",
        "new_rollouts": 0,
        "eta_star": eta_stars,
        "same_eta_near_minimal": common_assessment,
        "known_eta_1_0_0p25_validation": known,
        "pathology_audit": pathologies,
        "successful_eta_cell_count": len(successful_all),
        "successful_eta_cells_sorted_by_state_and_mean_J_def": successful_all,
        "all_tested_cells": all_cells,
        "cohort_cells": cohort_cells,
        "sanity_checks": sanity,
    }
    write_json(OUT / "results.json", results)
    write_json(OUT / "sanity_checks.json", sanity)

    top_rows = []
    for state in eta_stars:
        rows = [cell for cell in successful_all if cell["state_id"] == state][:10]
        for rank, cell in enumerate(rows, 1):
            top_rows.append(
                f"| {state} | {rank} | {format_eta(cell['eta'])} | {cell['success_rate']:.3f} | "
                f"{cell['mean_J_def']:.8f} | {cell['std_J_def']:.8f} | "
                f"{cell['mean_episode_length_steps']:.1f} | {cell['n']} |"
            )
    known_rows = []
    for state in eta_stars:
        cell = known[state]
        known_rows.append(
            f"| {state} | {cell['mean_J_def']:.8f} | {cell['std_J_def']:.8f} | "
            f"{cell['min_J_def']:.8f} | {cell['max_J_def']:.8f} | "
            f"{cell['counts']['success']}/{cell['n']} |"
        )
    eta_rows = []
    for state, item in eta_stars.items():
        phase = item["phase_a_CRN_grid"]
        expanded = item["all_existing_tested_points"]
        comparison = phase["winner_vs_runner_up"]
        eta_rows.append(
            f"| {state} | {format_eta(phase['eta'])} | {phase['mean_J_def']:.8f} | "
            f"{phase['boundary']['location']} | {format_eta(expanded['eta'])} | "
            f"{expanded['mean_J_def']:.8f} | {comparison['paired_95_interval'][0]:.8f}, "
            f"{comparison['paired_95_interval'][1]:.8f} |"
        )
    pathology_rows = []
    for state, item in pathologies.items():
        lowest = item["lowest_J_over_all_outcome_classes"]
        pathology_rows.append(
            f"| {state} | {format_eta(lowest['eta'])} | {lowest['classification']} | "
            f"{lowest['success_rate']:.3f} | {lowest['mean_J_def']:.8f} | "
            f"{item['successful_cell_J_vs_episode_length_spearman']:.3f} |"
        )

    report = f"""# Provisional executed-action deformation audit

## Audit outcome

This audit reused every cached SBMA attempt and launched **zero new rollouts**.  The
logger is an offline reader in this sibling diagnostic directory; no frozen
controller, physics, Flow, projection, event, success, or eta source was edited.

The smallest possible online insertion would be immediately after the second
projection in `success_basin_multimodality/run.py`: compute `delta = u - safe`
after line 52 and accumulate `dt * dot(delta.ravel(), delta.ravel())`.  The
offline implementation is exactly equivalent because lines 56--60 save `safe`
and `u` from that same loop iteration in the same trajectory record.

## Implemented quantity

For each completed cached rollout,

$$
J_{{\\rm def}}(z_t,\\eta;\\xi)
= {dt:g}\\sum_{{k=t}}^{{T_\\eta-1}}
  \\left\\|u_{{\\mathrm{{exec}},k}}^\\eta-u_{{\\mathrm{{safe}},k}}\\right\\|_2^2.
$$

The empirical cell value is the arithmetic mean over its stored Flow seeds.
There is no discount, normalization, completion penalty, progress term,
smoothness term, or contribution from raw pre-projection `g`.

## Empirical minima

`Phase-A CRN grid` compares the frozen 80-point lattice using the same 16 Flow
seeds for every eta in a state. `All tested` additionally includes validation,
interpolation, and axis-margin probes; those extra designs use different but
predeclared seed cohorts.

| state | Phase-A eta* | mean J | Phase-A location | all-tested eta* | mean J | paired 95% CI: runner minus Phase-A winner |
|---|---:|---:|---|---:|---:|---:|
{chr(10).join(eta_rows)}

The best single common Phase-A eta is
`{format_eta(best_common['eta'])}`.  Its maximum relative gap from the three
state-specific minima is `{100*best_common['maximum_relative_gap']:.3f}%`;
therefore the answer to “same eta within 5% of minimum for all three states” is
**{str(common_assessment['near_minimal_all_three_at_5_percent']).upper()}**.
This 5% label is descriptive only and did not select or classify cells.

All three empirical minima are on the sampled Phase-A box boundary. D1 also
touches both a FAILURE and an UNKNOWN neighbor; D2 touches an UNKNOWN neighbor.
D4's available axis-neighbors are successful, but its minimum remains at the
lowest tested goal/safe/relative corner, so the success-constrained continuous
minimum is not bracketed.

## eta=(1,0,0.25), dedicated validation seeds

| state | mean J | std | min | max | successes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(known_rows)}

## Lowest-cost successful eta cells

The table below shows the ten lowest per state.  The complete table is
`successful_eta_table.csv`, sorted by state and mean J.  J means include every
terminal outcome in a high-confidence successful cell, not only successful
sample paths.

| state | rank | eta | success rate | mean J | std J | mean steps | n |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(top_rows)}

## Pathology audit

Without the success constraint, the smallest J in every state belongs to a
FAILURE_CELL. This is expected for an action-deformation-only objective:
policies that do almost nothing and terminate in deadlock can be extremely
cheap.

| state | lowest-J eta over all cells | class | success rate | mean J | Spearman(J, episode length) within SUCCESS_CELL |
|---|---:|---|---:|---:|---:|
{chr(10).join(pathology_rows)}

The strong positive rank correlations show that even inside the success set,
the accumulated metric is materially coupled to episode duration. This is not
an added completion-time penalty--it follows from summing a nonnegative running
cost until policy-dependent termination.

## Sanity checks

- Raw attempts read: **{len(per_rollout)}**; completed effective rollouts:
  **{sanity['completed_effective_rollouts']}**; unresolved after same-seed repair:
  **{sanity['unresolved_effective_rollouts']}**.
- All source NPZ SHA256 values matched their manifests, and no source file size
  or mtime changed during analysis.
- `J_def == 0` iff every stored `u_exec` equals `u_safe` bit-for-bit passed for
  every cached attempt. Actual all-step-zero rollouts: **{actual_zero_j_count}**;
  the explicit equal-array unit check also returned exactly `0.0`.
- Maximum `w - u_safe - g` reconstruction error:
  **{max_w_reconstruction_error:.3e}**.
- Maximum dynamics reconstruction error:
  **{max_integration_error:.3e}**.
- Maximum vectorized-versus-step-loop J difference:
  **{max_j_loop_error:.3e}**.
- Existing SBMA independent replay checked 108 projections; its maximum
  recomputation difference was
  **{sanity['existing_independent_projection_replay']['maximum_projection_recompute_difference']:.3e}**.

## Interpretation and recommendation

This is only a provisional empirical test of “minimum executed-action
deformation subject to success.”  It measures the action that survives the
hard projection, which is the intended object, and it preserves CRN pairing on
the primary grid.

Potential pathology: the undiscounted sum depends on termination time.  A
quick failure can have a small cost, and among successful policies a faster
completion accumulates fewer terms. Restricting eta selection to the frozen
high-confidence SUCCESS_CELL set prevents the first issue here, but does not
remove the duration coupling.  Also, large raw `g` that is removed by the
second projection is intentionally free under this definition.

**Recommendation: MODIFY before freezing.** Keep the executed-action,
same-state/same-Flow reference as the core deformation term, but do not yet
freeze it as the sole controller objective: the tested minimum must be treated
as finite-design empirical evidence, especially wherever it lies on a sampled
grid/probe boundary, and the termination-time coupling should be addressed or
explicitly accepted in the eventual specification.
"""
    (OUT / "audit_report.md").write_text(report)

    output_names = [
        "audit_report.md", "per_rollout_j_def.jsonl", "results.json",
        "sanity_checks.json", "successful_eta_table.csv",
    ]
    source_files = [
        SOURCE / "protocol.json", SOURCE / "run.py", SOURCE / "verification.json",
        SOURCE / "success_map.json", *manifest_paths,
    ]
    manifest = {
        "study": "provisional_success_basin_executed_action_deformation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_rollouts": 0,
        "source_attempts": len(per_rollout),
        "formula": sanity["formula"],
        "implementation": "offline read-only computation from same-step cached u_safe and post-second-projection u_exec",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_files
        },
        "implementation_hashes": {
            "__init__.py": sha256(OUT / "__init__.py"),
            "analyze.py": sha256(OUT / "analyze.py"),
        },
        "output_hashes": {
            name: sha256(OUT / name) for name in output_names
        },
        "files_added": ["__init__.py", "analyze.py", *output_names, "manifest.json"],
        "frozen_source_files_modified": [],
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({
        "new_rollouts": 0,
        "raw_attempts": len(per_rollout),
        "completed_effective_rollouts": sanity["completed_effective_rollouts"],
        "successful_eta_cells": len(successful_all),
        "eta_star": {state: item["phase_a_CRN_grid"]["eta"] for state, item in eta_stars.items()},
        "best_common_eta": best_common["eta"],
    }, indent=2))


if __name__ == "__main__":
    main()
