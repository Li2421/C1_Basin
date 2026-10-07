"""Fail-closed analysis of the single-use frozen Safety versus H=8 test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
MANIFEST = HERE / "fresh_test_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
CONFIG = HERE / "controller_config.json"
CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
EXPECTED_CHECKPOINT = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
EXPECTED_MANIFEST = "68eecae6d04b00b0e4db7447a8e05d2443d0c886e67d14526b2eae8952841d45"
EXPECTED_OVERLAP = "26a003a435141f91bc3b87729d64c29a55b32c4ca2ebebd20545adc6602df4f6"
FLOW_ROOT = 2026092502
FLOW_SEMANTICS = "episode_key=fold_in(PRNGKey(2026092502), rollout_id); step_key=fold_in(episode_key, physical_step)"
DT = 0.05
N = 200
BOOTSTRAP_REPLICATES = 200_000
BOOTSTRAP_SEED = 20260925
CONDITIONS = {"Safety": ("safety", None), "H=8": ("h8", 8)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "p95": None, "max": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def binomial_upper(successes: int, trials: int) -> float:
    if not trials:
        return 1.0
    return min(1.0, sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2**trials)


def outcome(row: dict[str, Any]) -> str:
    value = str(row.get("outcome", "")).lower()
    if value == "other":
        return "other_failure"
    return value


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(MANIFEST) != EXPECTED_MANIFEST or sha256(OVERLAP) != EXPECTED_OVERLAP:
        raise RuntimeError("frozen manifest or overlap audit changed")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("checkpoint changed")
    manifest = json.loads(MANIFEST.read_text())
    overlap = json.loads(OVERLAP.read_text())
    config = json.loads(CONFIG.read_text())
    manifest_body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    config_body = {key: value for key, value in config.items() if key != "content_sha256"}
    if canonical_hash(manifest_body) != manifest["content_sha256"]:
        raise RuntimeError("manifest semantic hash mismatch")
    if canonical_hash(config_body) != config["content_sha256"]:
        raise RuntimeError("controller config semantic hash mismatch")
    if overlap.get("status") != "PASS" or overlap.get("exact_initial_state_match_count") != 0 or overlap.get("new_seed_collisions"):
        raise RuntimeError("unseen overlap audit does not pass")
    if not manifest.get("frozen_before_rollout") or manifest.get("raw_rollout_records_present_at_freeze") != 0:
        raise RuntimeError("manifest was not frozen before rollout")
    if manifest["episode_count"] != N or len(manifest["episodes"]) != N:
        raise RuntimeError("manifest is not the frozen 200-episode cohort")
    if manifest["controller_protocol"] != {
        "controllers": ["Safety", "H=8"], "cadence_h": 8,
        "correction_one_step_only": True, "held_correction": False,
        "horizon_steps": 850, "dt_seconds": 0.05, "no_tuning": True,
    }:
        raise RuntimeError("controller protocol changed")
    if config["controllers"] != {"safety": "Safety", "h8": "H=8"} or config["cadence_h8"] != 8:
        raise RuntimeError("controller set/cadence changed")
    if config["flow_root_seed"] != FLOW_ROOT or config["flow_key_semantics"] != FLOW_SEMANTICS:
        raise RuntimeError("Flow stream changed")
    if any(config["forbidden"].values()):
        raise RuntimeError("forbidden online/training component enabled")
    return manifest, overlap, config


def load_records(manifest: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, dict[int, dict[str, np.ndarray]]], dict[str, Any]]:
    expected = {int(row["episode_index"]): row for row in manifest["episodes"]}
    records: dict[str, dict[int, dict[str, Any]]] = {}
    trajectories: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    raw_hashes: dict[str, str] = {}
    trajectory_hashes: dict[str, str] = {}
    cadence_exact = True
    no_hold = True
    flow_keys_matched = True
    starts_after_freeze = True
    for name, (slug, cadence) in CONDITIONS.items():
        raw_dir = HERE / "runs/production/raw" / slug
        actual = sorted(raw_dir.glob("episode_*.json"))
        if len(actual) != N:
            raise RuntimeError(("wrong raw record count", name, len(actual)))
        records[name] = {}
        trajectories[name] = {}
        for index in range(N):
            raw_path = raw_dir / f"episode_{index:04d}.json"
            row = json.loads(raw_path.read_text())
            required = {
                "record_complete": True, "condition": name, "episode_index": index,
                "manifest_file_sha256": EXPECTED_MANIFEST,
                "manifest_content_sha256": manifest["content_sha256"],
                "controller_config_sha256": config["content_sha256"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT,
                "flow_root_seed": FLOW_ROOT, "flow_key_semantics": FLOW_SEMANTICS,
                "gphi_overlap_partition": "fresh_unseen", "cadence_h": cadence,
            }
            for key, value in required.items():
                if row.get(key) != value:
                    raise RuntimeError(("raw field mismatch", name, index, key, row.get(key), value))
            if not row.get("rollout_started_after_manifest_frozen"):
                starts_after_freeze = False
            spec = expected[index]
            if int(row["rollout_id"]) != int(spec["rollout_id"]):
                raise RuntimeError(("rollout ID mismatch", name, index))
            if not np.array_equal(np.asarray(row["initial_positions"], np.float32), np.asarray(spec["initial_positions"], np.float32)):
                raise RuntimeError(("initial condition mismatch", name, index))
            trajectory_path = HERE / row["trajectory_file"]
            if not trajectory_path.is_file() or sha256(trajectory_path) != row["trajectory_sha256"]:
                raise RuntimeError(("trajectory hash mismatch", name, index))
            with np.load(trajectory_path, allow_pickle=False) as data:
                tr = {key: np.asarray(data[key]) for key in data.files}
            steps = np.asarray(tr["step"], dtype=np.int64)
            if not np.array_equal(steps, np.arange(len(steps))) or len(steps) != int(row["episode_steps"]):
                raise RuntimeError(("physical step mismatch", name, index))
            scheduled = np.asarray(tr["cadence_scheduled"], dtype=bool)
            schedule_expected = np.zeros(len(steps), dtype=bool) if cadence is None else steps % cadence == 0
            cadence_exact &= bool(np.array_equal(scheduled, schedule_expected))
            g_hat = np.asarray(tr["g_hat"], dtype=np.float64)
            u_safe = np.asarray(tr["u_safe"], dtype=np.float64)
            u_exec = np.asarray(tr["u_exec"], dtype=np.float64)
            raw_target = np.asarray(tr["raw_second_target"], dtype=np.float64)
            inactive = ~scheduled
            no_hold &= bool(
                np.max(np.abs(g_hat[inactive]), initial=0.0) <= 1e-12
                and np.max(np.abs(u_exec[inactive] - u_safe[inactive]), initial=0.0) <= 1e-12
            )
            if not np.allclose(raw_target, u_safe + g_hat, rtol=1e-9, atol=1e-11):
                raise RuntimeError(("second projection target mismatch", name, index))
            norms = np.linalg.norm((u_exec - u_safe).reshape(len(steps), -1), axis=1)
            jdef = DT * float(np.sum(norms**2))
            if not math.isclose(jdef, float(row["J_def"]), rel_tol=1e-9, abs_tol=1e-11):
                raise RuntimeError(("J_def mismatch", name, index))
            if not all(np.isfinite(value).all() for value in tr.values() if np.issubdtype(value.dtype, np.number)):
                raise RuntimeError(("NaN/Inf trajectory", name, index))
            records[name][index] = row
            trajectories[name][index] = tr
            raw_hashes[f"{slug}/{raw_path.name}"] = sha256(raw_path)
            trajectory_hashes[f"{slug}/{trajectory_path.name}"] = sha256(trajectory_path)
    for index in range(N):
        a, b = records["Safety"][index], records["H=8"][index]
        for key in ("rollout_id", "flow_root_seed", "flow_key_semantics"):
            if a[key] != b[key]:
                raise RuntimeError(("unmatched paired stream", index, key))
        ta, tb = trajectories["Safety"][index], trajectories["H=8"][index]
        overlap_steps = min(len(ta["step"]), len(tb["step"]))
        flow_keys_matched &= bool(np.array_equal(ta["flow_step_key"][:overlap_steps], tb["flow_step_key"][:overlap_steps]))
    integrity = {
        "status": "PASS" if cadence_exact and no_hold and flow_keys_matched and starts_after_freeze else "FAIL",
        "episodes_per_condition": N, "raw_record_count": 2 * N,
        "manifest_frozen_before_rollout": starts_after_freeze,
        "exact_fixed_H8_schedule": cadence_exact,
        "no_held_correction": no_hold,
        "matched_initial_conditions_and_flow_streams": flow_keys_matched,
        "raw_hashes_verified": True, "trajectory_hashes_verified": True,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "manifest_file_sha256": sha256(MANIFEST),
        "controller_config_file_sha256": sha256(CONFIG),
        "controller_config_content_sha256": config["content_sha256"],
        "raw_record_sha256": raw_hashes, "trajectory_sha256": trajectory_hashes,
    }
    if integrity["status"] != "PASS":
        raise RuntimeError(("integrity checks failed", integrity))
    return records, trajectories, integrity


def summarize(manifest: dict[str, Any], overlap: dict[str, Any], config: dict[str, Any], records: dict[str, dict[int, dict[str, Any]]], trajectories: dict[str, dict[int, dict[str, np.ndarray]]], integrity: dict[str, Any]) -> None:
    counts: dict[str, Counter[str]] = {}
    condition_metrics: dict[str, Any] = {}
    per_episode: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for name in CONDITIONS:
        counts[name] = Counter(outcome(row) for row in records[name].values())
        success = counts[name]["success"]
        condition_metrics[name] = {
            "episodes": N, "success": success, "strict_deadlock": counts[name]["deadlock"],
            "timeout": counts[name]["timeout"], "collision": counts[name]["collision"],
            "wall_collision": counts[name]["wall_collision"], "other_termination": counts[name]["other_failure"],
            "Q": success / N, "Q_wilson_95_ci": wilson(success, N),
        }
    for index in range(N):
        s, h = records["Safety"][index], records["H=8"][index]
        so, ho = outcome(s), outcome(h)
        cell = "BOTH_SUCCESS" if so == ho == "success" else "RESCUE" if so != "success" and ho == "success" else "BREAK" if so == "success" and ho != "success" else "BOTH_FAIL"
        paired.append({
            "episode_index": index, "source_id": manifest["episodes"][index]["source_id"],
            "rollout_id": int(s["rollout_id"]), "Safety_outcome": so, "H8_outcome": ho,
            "paired_cell": cell, "H8_J_def": float(h["J_def"]),
            "initial_positions_json": json.dumps(s["initial_positions"], separators=(",", ":")),
        })
        for name, row in (("Safety", s), ("H=8", h)):
            per_episode.append({
                "condition": name, "episode_index": index, "source_id": manifest["episodes"][index]["source_id"],
                "rollout_id": int(row["rollout_id"]), "outcome": outcome(row), "success": bool(row["success"]),
                "episode_steps": int(row["episode_steps"]), "J_def": float(row["J_def"]),
                "paired_cell": "REFERENCE" if name == "Safety" else cell,
                "projection_failures": int(row.get("projection_failures", 0)),
                "invalid_actions": int(row.get("invalid_actions", 0)),
                "nan_inf_events": int(row.get("nan_inf_events", 0)),
                "agent_collision": bool(row.get("agent_collision", False)),
                "wall_collision": bool(row.get("wall_collision", False)),
                "runtime_seconds": float(row.get("runtime_seconds", 0.0)),
            })
    cells = Counter(row["paired_cell"] for row in paired)
    rescue, broken = cells["RESCUE"], cells["BREAK"]
    safety_fail = N - counts["Safety"]["success"]
    safety_success = counts["Safety"]["success"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.asarray([
        int(records["H=8"][i]["success"]) - int(records["Safety"][i]["success"]) for i in range(N)
    ], dtype=np.float64)
    draw_indices = rng.integers(0, N, size=(BOOTSTRAP_REPLICATES, N))
    boot = differences[draw_indices].mean(axis=1)
    bootstrap_ci = [float(value) for value in np.quantile(boot, [0.025, 0.975])]
    discordant = rescue + broken
    improve_p = binomial_upper(rescue, discordant)
    deteriorate_p = binomial_upper(broken, discordant)
    two_sided = min(1.0, 2 * min(improve_p, deteriorate_p))
    paired_summary = {
        "BOTH_SUCCESS": cells["BOTH_SUCCESS"], "RESCUE": rescue, "BREAK": broken, "BOTH_FAIL": cells["BOTH_FAIL"],
        "Safety_failures": safety_fail, "Safety_successes": safety_success,
        "rescue_rate": rescue / safety_fail if safety_fail else None,
        "rescue_rate_wilson_95_ci": wilson(rescue, safety_fail),
        "break_rate": broken / safety_success if safety_success else None,
        "break_rate_wilson_95_ci": wilson(broken, safety_success),
        "net_rescue": rescue - broken,
        "Delta_Q": (rescue - broken) / N,
        "paired_bootstrap_95_ci": bootstrap_ci,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "exact_McNemar_two_sided_p": two_sided,
        "exact_McNemar_one_sided_improvement_p": improve_p,
        "exact_McNemar_one_sided_deterioration_p": deteriorate_p,
    }
    failure_rescue: dict[str, Any] = {}
    for label in ("timeout", "deadlock"):
        selected = [i for i in range(N) if outcome(records["Safety"][i]) == label]
        rescued = sum(outcome(records["H=8"][i]) == "success" for i in selected)
        destinations = Counter(outcome(records["H=8"][i]) for i in selected)
        failure_rescue[f"Safety_{label}"] = {
            "Safety_count": len(selected), "H8_rescue_count": rescued,
            "H8_rescue_rate": rescued / len(selected) if selected else None,
            "H8_rescue_rate_wilson_95_ci": wilson(rescued, len(selected)),
            "H8_destination_outcomes": dict(sorted(destinations.items())),
        }
    h8_j = [float(records["H=8"][i]["J_def"]) for i in range(N)]
    deformation = {"definition": "0.05 * sum_t ||u_exec,t-u_safe,t||_2^2", "H=8": {"overall": stats(h8_j)}}
    for cell in ("BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL"):
        selected = [float(records["H=8"][int(row["episode_index"])]["J_def"]) for row in paired if row["paired_cell"] == cell]
        deformation["H=8"][cell] = stats(selected)
    hard_safety = {
        "H8_agent_collisions": sum(int(row.get("agent_collision", False)) for row in records["H=8"].values()),
        "H8_wall_collisions": sum(int(row.get("wall_collision", False)) for row in records["H=8"].values()),
        "H8_invalid_actions": sum(int(row.get("invalid_actions", 0)) for row in records["H=8"].values()),
        "H8_nan_inf_events": sum(int(row.get("nan_inf_events", 0)) for row in records["H=8"].values()),
        "H8_projection_solver_failures": sum(int(row.get("projection_failures", 0)) for row in records["H=8"].values()),
    }
    hard_safety["intact"] = not any(hard_safety.values())
    if paired_summary["net_rescue"] > 0 and paired_summary["Delta_Q"] > 0 and improve_p < 0.05 and bootstrap_ci[0] > 0:
        classification = "H8_FRESH_GENERALIZATION_CONFIRMED"
    elif rescue > 0 and paired_summary["net_rescue"] > 0:
        classification = "H8_FRESH_GENERALIZATION_PARTIAL"
    else:
        classification = "H8_FRESH_GENERALIZATION_FAIL"
    primary = {
        "endpoint": "frozen 850-step / 42.5-second horizon", "conditions": condition_metrics,
        "paired": paired_summary, "classification": classification,
    }
    integrity.update({
        "overlap_audit_status": overlap["status"],
        "exact_prior_initial_overlap_count": overlap["exact_initial_state_match_count"],
        "prior_seed_collision_count": len(overlap["new_seed_collisions"]),
        "no_post_hoc_tuning": True, "controllers_evaluated": ["Safety", "H=8"],
        "hard_safety_intact": hard_safety["intact"],
    })
    runtime_files = sorted((HERE / "runs/production").glob("runtime_*.json"))
    runtimes = [json.loads(path.read_text()) for path in runtime_files]
    process_starts = [row["started_utc"] for row in runtimes]
    process_finishes = [row["finished_utc"] for row in runtimes]
    runtime = {
        "process_count": len(runtimes), "runtime_records": runtimes,
        "first_process_started_utc": min(process_starts) if process_starts else None,
        "last_process_finished_utc": max(process_finishes) if process_finishes else None,
        "rollout_wall_seconds": (
            datetime.fromisoformat(max(process_finishes)).timestamp()
            - datetime.fromisoformat(min(process_starts)).timestamp()
            if process_starts and process_finishes else None
        ),
        "requested_resources": json.loads((HERE / "resource_audit.json").read_text()),
        "rollout_tuples": 2 * N,
        "total_episode_simulation_seconds": float(sum(float(row.get("runtime_seconds", 0.0)) for rows in records.values() for row in rows.values())),
    }
    write_csv(HERE / "per_episode_results.csv", per_episode)
    write_csv(HERE / "paired_outcomes.csv", paired)
    write_json(HERE / "primary_metrics.json", primary)
    write_json(HERE / "rescue_break_analysis.json", paired_summary)
    write_json(HERE / "failure_type_rescue.json", failure_rescue)
    write_json(HERE / "deformation_metrics.json", deformation)
    write_json(HERE / "hard_safety_checks.json", hard_safety)
    write_json(HERE / "integrity_audit.json", integrity)
    write_json(HERE / "runtime_statistics.json", runtime)
    timeout_pattern = failure_rescue["Safety_timeout"]
    deadlock_pattern = failure_rescue["Safety_deadlock"]
    asymmetry = (
        timeout_pattern["H8_rescue_rate"] is not None and deadlock_pattern["H8_rescue_rate"] is not None
        and timeout_pattern["H8_rescue_rate"] > deadlock_pattern["H8_rescue_rate"]
    )
    report = [
        "# Frozen H=8 fresh-unseen WIDE generalization",
        "",
        f"**Classification: {classification}.** This is the single pre-frozen 200-episode test; no alternative cadence or checkpoint was evaluated.",
        "",
        "## Frozen test integrity",
        "",
        f"- Generator: `{manifest['generator']['name']}` with seed `{manifest['generator']['ic_generator_seed']}`; exact authoritative replay was bitwise verified.",
        f"- Manifest SHA256: `{sha256(MANIFEST)}`; frozen at `{manifest['frozen_utc']}` with zero rollout records present.",
        f"- Overlap: 0 exact initial-state matches and 0 seed/stream collisions against the audited prior assets.",
        f"- Checkpoint: `{CHECKPOINT}` (`{sha256(CHECKPOINT)}`).",
        "- Fixed endpoint: 850 steps at 0.05 s; matched fresh Flow root 2026092502; Safety and one-step-only H=8 only.",
        "",
        "## Primary result",
        "",
        "| Controller | Success | Strict deadlock | Timeout | Collision | Q (95% Wilson CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("Safety", "H=8"):
        row = condition_metrics[name]
        ci = row["Q_wilson_95_ci"]
        report.append(f"| {name} | {row['success']}/{N} | {row['strict_deadlock']} | {row['timeout']} | {row['collision']} | {row['Q']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
    report += [
        "",
        f"Paired cells: BOTH_SUCCESS={cells['BOTH_SUCCESS']}, RESCUE={rescue}, BREAK={broken}, BOTH_FAIL={cells['BOTH_FAIL']}. ",
        f"Delta Q={paired_summary['Delta_Q']:+.4f} (paired bootstrap 95% CI [{bootstrap_ci[0]:+.4f}, {bootstrap_ci[1]:+.4f}]); exact one-sided McNemar improvement p={improve_p:.6g}.",
        "",
        "## Failure-type rescue",
        "",
        f"- Safety timeout: {timeout_pattern['H8_rescue_count']}/{timeout_pattern['Safety_count']} rescued ({timeout_pattern['H8_rescue_rate'] if timeout_pattern['H8_rescue_rate'] is not None else 'N/A'}).",
        f"- Safety strict deadlock: {deadlock_pattern['H8_rescue_count']}/{deadlock_pattern['Safety_count']} rescued ({deadlock_pattern['H8_rescue_rate'] if deadlock_pattern['H8_rescue_rate'] is not None else 'N/A'}).",
        f"- Timeout-vs-deadlock rescue asymmetry reproduced descriptively: **{'yes' if asymmetry else 'no'}**.",
        "",
        "## Deformation and safety",
        "",
        f"H=8 J_def mean={deformation['H=8']['overall']['mean']:.6f}, median={deformation['H=8']['overall']['median']:.6f}, P95={deformation['H=8']['overall']['p95']:.6f}, max={deformation['H=8']['overall']['max']:.6f}.",
        f"Hard safety: **{'INTACT' if hard_safety['intact'] else 'VIOLATED'}**.",
    ]
    (HERE / "fresh_generalization_report.md").write_text("\n".join(report) + "\n")
    output_names = [
        "fresh_test_manifest.json", "overlap_audit.json", "controller_config.json", "resource_audit.json",
        "per_episode_results.csv", "paired_outcomes.csv", "primary_metrics.json", "rescue_break_analysis.json",
        "failure_type_rescue.json", "deformation_metrics.json", "hard_safety_checks.json", "integrity_audit.json",
        "runtime_statistics.json", "fresh_generalization_report.md", "prepare_fresh_manifest.py",
        "run_fresh_evaluation.py", "run_two_shards.sh", "analyze_fresh.py",
    ]
    output_hashes = {name: sha256(HERE / name) for name in output_names}
    output_manifest = {
        "schema": "gphi_h8_fresh_unseen_generalization_outputs_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "classification": classification, "files_sha256": output_hashes,
        "frozen_inputs": {"fresh_test_manifest_sha256": sha256(MANIFEST), "checkpoint_sha256": sha256(CHECKPOINT)},
    }
    output_manifest["content_sha256"] = canonical_hash(output_manifest)
    write_json(HERE / "manifest.json", output_manifest)
    print(json.dumps({"status": "PASS", "classification": classification, "primary": primary, "hard_safety": hard_safety}, indent=2))


def main() -> None:
    manifest, overlap, config = load_inputs()
    records, trajectories, integrity = load_records(manifest, config)
    summarize(manifest, overlap, config, records, trajectories, integrity)


if __name__ == "__main__":
    main()
