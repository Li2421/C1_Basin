"""Fail-closed analysis for the frozen three-way fresh WIDE evaluation."""

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
HERE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"
MANIFEST = HERE / "fresh_wide_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
CONTROLLERS = HERE / "controller_manifest.json"
EXPECTED_MANIFEST_SHA = "274a1c5968c9fe957d58575c0d388099c6707b90a6e421c835378e67d9f95b67"
EXPECTED_CONTROLLER_SHA = "376fee53ee2882ca05893fed37463849dfb859702349433472df70e15da1d840"
EXPECTED_DIRECT_SHA = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
EXPECTED_ETA_SHA = "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"
N = 200
DT = 0.05
BOOTSTRAP_REPLICATES = 200_000
BOOTSTRAP_SEED = 2026092603
CONDITIONS = {
    "Safety": "safety",
    "Direct-g H8": "direct_g_h8",
    "Structured eta": "structured_eta",
}


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


def stats(values: Iterable[float], completion: bool = False) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    base: dict[str, float | int | None] = {"count": int(len(array))}
    if not len(array):
        return {**base, "mean": None, "median": None, "std": None, "p95": None, "max": None}
    base.update({
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.quantile(array, .95)), "max": float(array.max()),
    })
    if completion:
        base["p90"] = float(np.quantile(array, .90))
    return base


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_bootstrap(differences: np.ndarray, seed_offset: int = 0) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values: list[np.ndarray] = []
    left = BOOTSTRAP_REPLICATES
    while left:
        size = min(10_000, left)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        values.append(differences[indices].mean(axis=1))
        left -= size
    boot = np.concatenate(values)
    return [float(value) for value in np.quantile(boot, [.025, .975])]


def binomial_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return min(1.0, sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2**trials)


def paired_test(rescue: int, broken: int, differences: np.ndarray, seed_offset: int) -> dict[str, Any]:
    discordant = rescue + broken
    one_improve = binomial_tail(rescue, discordant)
    one_worse = binomial_tail(broken, discordant)
    return {
        "paired_bootstrap_95_ci": paired_bootstrap(differences, seed_offset),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
        "McNemar_discordant_count": discordant,
        "exact_McNemar_two_sided_p": min(1.0, 2 * min(one_improve, one_worse)),
        "exact_McNemar_one_sided_improvement_p": one_improve,
    }


def normalized_outcome(row: dict[str, Any]) -> str:
    value = str(row["outcome"]).lower()
    return "other" if value == "other_failure" else value


def load_locked() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("frozen fresh manifest changed")
    if sha256(CONTROLLERS) != EXPECTED_CONTROLLER_SHA:
        raise RuntimeError("locked controller manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    overlap = json.loads(OVERLAP.read_text())
    controllers = json.loads(CONTROLLERS.read_text())
    for payload, label in ((manifest, "fresh manifest"), (controllers, "controller manifest")):
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if canonical_hash(body) != payload["content_sha256"]:
            raise RuntimeError(f"{label} semantic hash mismatch")
    if not manifest["frozen_before_rollout"] or manifest["raw_rollout_records_present_at_freeze"] != 0:
        raise RuntimeError("manifest not frozen before rollout")
    if manifest["episode_count"] != N or len(manifest["episodes"]) != N:
        raise RuntimeError("not exactly 200 fresh episodes")
    if overlap["status"] != "PASS" or overlap["exact_initial_state_match_count"] != 0:
        raise RuntimeError("fresh overlap audit failed")
    if controllers["direct_g"]["sha256"] != EXPECTED_DIRECT_SHA:
        raise RuntimeError("direct-g checkpoint mismatch")
    if controllers["structured_eta"]["sha256"] != EXPECTED_ETA_SHA:
        raise RuntimeError("eta checkpoint mismatch")
    if any(controllers["forbidden"].values()):
        raise RuntimeError("forbidden operation enabled")
    return manifest, overlap, controllers


def load_records(manifest: dict[str, Any], controllers: dict[str, Any]):
    specs = {int(row["episode_index"]): row for row in manifest["episodes"]}
    records: dict[str, dict[int, dict[str, Any]]] = {}
    trajectories: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    manifest_sha = sha256(MANIFEST)
    controller_sha = sha256(CONTROLLERS)
    audit = {
        "status": "PASS", "episodes_per_controller": N,
        "manifest_frozen_before_rollout": True,
        "manifest_sha256": manifest_sha, "controller_manifest_sha256": controller_sha,
        "matched_initial_conditions": True, "matched_flow_streams": True,
        "direct_H8_schedule_exact": True, "direct_no_held_correction": True,
        "structured_eta_one_query": True, "structured_eta_fixed": True,
        "structured_eta_dense_feedback": True, "safety_uncorrected": True,
        "trajectory_hashes_verified": True, "all_numeric_values_finite": True,
        "jdef_recomputed_exactly": True, "raw_target_identity_verified": True,
    }
    for condition, slug in CONDITIONS.items():
        files = sorted((HERE / "runs/raw" / slug).glob("episode_*.json"))
        if len(files) != N:
            raise RuntimeError((condition, "record count", len(files)))
        records[condition], trajectories[condition] = {}, {}
        for index in range(N):
            path = HERE / "runs/raw" / slug / f"episode_{index:04d}.json"
            row = json.loads(path.read_text())
            expected = {
                "record_complete": True, "condition": condition, "episode_index": index,
                "fresh_wide_manifest_sha256": manifest_sha,
                "fresh_wide_manifest_content_sha256": manifest["content_sha256"],
                "controller_manifest_sha256": controller_sha,
                "flow_root_seed": manifest["flow_randomness"]["root_seed"],
                "flow_key_semantics": manifest["flow_randomness"]["semantics"],
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise RuntimeError((condition, index, key, row.get(key), value))
            expected_checkpoint = None if condition == "Safety" else EXPECTED_DIRECT_SHA if condition == "Direct-g H8" else EXPECTED_ETA_SHA
            if row.get("learned_checkpoint_sha256") != expected_checkpoint:
                raise RuntimeError((condition, index, "checkpoint"))
            spec = specs[index]
            if row["rollout_id"] != spec["rollout_id"]:
                raise RuntimeError((condition, index, "rollout_id"))
            if not np.array_equal(np.asarray(row["initial_positions"], np.float32), np.asarray(spec["initial_positions"], np.float32)):
                raise RuntimeError((condition, index, "initial positions"))
            if not row["rollout_started_after_manifest_frozen"]:
                raise RuntimeError((condition, index, "rollout before freeze"))
            trajectory_path = HERE / row["trajectory_file"]
            if not trajectory_path.is_file() or sha256(trajectory_path) != row["trajectory_sha256"]:
                raise RuntimeError((condition, index, "trajectory hash"))
            with np.load(trajectory_path, allow_pickle=False) as data:
                tr = {key: np.asarray(data[key]) for key in data.files}
            steps = np.asarray(tr["step"], dtype=np.int64)
            if len(steps) != row["episode_steps"] or not np.array_equal(steps, np.arange(len(steps))):
                raise RuntimeError((condition, index, "physical steps"))
            scheduled = np.asarray(tr["cadence_scheduled"], dtype=bool)
            g = np.asarray(tr["g_hat"], dtype=np.float64)
            u_safe = np.asarray(tr["u_safe"], dtype=np.float64)
            u_exec = np.asarray(tr["u_exec"], dtype=np.float64)
            raw_target = np.asarray(tr["raw_second_target"], dtype=np.float64)
            if not np.allclose(raw_target, u_safe + g, rtol=1e-9, atol=1e-11):
                raise RuntimeError((condition, index, "raw target"))
            if condition == "Safety":
                audit["safety_uncorrected"] &= bool(not scheduled.any() and np.max(np.abs(g), initial=0) <= 1e-12 and np.max(np.abs(u_exec-u_safe), initial=0) <= 1e-12)
            elif condition == "Direct-g H8":
                expected_schedule = steps % 8 == 0
                audit["direct_H8_schedule_exact"] &= bool(np.array_equal(scheduled, expected_schedule))
                inactive = ~scheduled
                audit["direct_no_held_correction"] &= bool(np.max(np.abs(g[inactive]), initial=0) <= 1e-12 and np.max(np.abs(u_exec[inactive]-u_safe[inactive]), initial=0) <= 1e-12)
            else:
                eta = np.asarray(tr["eta_hat"], dtype=np.float64)
                query = np.asarray(tr["model_query_marker"], dtype=bool)
                audit["structured_eta_one_query"] &= bool(query.sum() == 1 and query[0] and row["model_query_count"] == 1)
                audit["structured_eta_fixed"] &= bool(np.allclose(eta, eta[0], atol=0, rtol=0) and np.allclose(eta[0], row["eta_hat"], atol=1e-14, rtol=0))
                audit["structured_eta_dense_feedback"] &= bool(scheduled.all())
            correction = (u_exec - u_safe).reshape(len(steps), -1)
            recomputed_jdef = DT * float(np.sum(np.linalg.norm(correction, axis=1) ** 2))
            if not math.isclose(recomputed_jdef, row["J_def"], rel_tol=1e-9, abs_tol=1e-11):
                raise RuntimeError((condition, index, "J_def"))
            if not all(np.isfinite(array).all() for array in tr.values() if np.issubdtype(array.dtype, np.number)):
                raise RuntimeError((condition, index, "NaN/Inf"))
            records[condition][index] = row
            trajectories[condition][index] = tr
    for index in range(N):
        base = records["Safety"][index]
        base_tr = trajectories["Safety"][index]
        for condition in ("Direct-g H8", "Structured eta"):
            row = records[condition][index]
            for key in ("rollout_id", "flow_root_seed", "flow_key_semantics", "initial_positions"):
                if row[key] != base[key]:
                    raise RuntimeError((condition, index, "pair mismatch", key))
            tr = trajectories[condition][index]
            common = min(len(base_tr["step"]), len(tr["step"]))
            audit["matched_flow_streams"] &= bool(np.array_equal(base_tr["flow_step_key"][:common], tr["flow_step_key"][:common]))
    boolean_checks = [value for key, value in audit.items() if isinstance(value, bool)]
    if not all(boolean_checks):
        audit["status"] = "FAIL"
        raise RuntimeError(("integrity failed", audit))
    return records, trajectories, audit


def pair_category(reference: str, learned: str) -> str:
    if reference == "success" and learned == "success":
        return "BOTH_SUCCESS"
    if reference != "success" and learned == "success":
        return "RESCUE"
    if reference == "success" and learned != "success":
        return "BREAK"
    return "BOTH_FAIL"


def summarize_pair(records, reference: str, learned: str, seed_offset: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index in range(N):
        ro = normalized_outcome(records[reference][index])
        lo = normalized_outcome(records[learned][index])
        rows.append({"episode_index": index, "reference": reference, "learned": learned,
                     "reference_outcome": ro, "learned_outcome": lo,
                     "paired_cell": pair_category(ro, lo)})
    cells = Counter(row["paired_cell"] for row in rows)
    ref_success = sum(normalized_outcome(records[reference][i]) == "success" for i in range(N))
    ref_failure = N - ref_success
    rescue, broken = cells["RESCUE"], cells["BREAK"]
    differences = np.asarray([int(records[learned][i]["success"]) - int(records[reference][i]["success"]) for i in range(N)], dtype=float)
    summary = {
        "reference": reference, "learned": learned, **{key: cells[key] for key in ("BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL")},
        "reference_successes": ref_success, "reference_failures": ref_failure,
        "rescue_count": rescue, "rescue_rate": rescue / ref_failure if ref_failure else None,
        "rescue_rate_wilson_95_ci": wilson(rescue, ref_failure),
        "break_count": broken, "break_rate": broken / ref_success if ref_success else None,
        "break_rate_wilson_95_ci": wilson(broken, ref_success),
        "net_rescue": rescue - broken, "Delta_Q": differences.mean(),
        **paired_test(rescue, broken, differences, seed_offset),
    }
    return summary, rows


def main() -> None:
    manifest, overlap, controllers = load_locked()
    records, trajectories, integrity = load_records(manifest, controllers)

    primary: dict[str, Any] = {"endpoint": "850 physical steps = 42.5 seconds", "episode_count": N, "controllers": {}}
    per_episode: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        counts = Counter(normalized_outcome(row) for row in records[condition].values())
        success = counts["success"]
        primary["controllers"][condition] = {
            "success": success, "strict_deadlock": counts["deadlock"], "timeout": counts["timeout"],
            "collision": counts["collision"], "wall_collision": sum(int(row.get("wall_collision", False)) for row in records[condition].values()),
            "other_termination": counts["other"], "Q": success / N, "Q_wilson_95_ci": wilson(success, N),
        }
        for index, row in records[condition].items():
            per_episode.append({
                "episode_index": index, "source_id": row["source_id"], "condition": condition,
                "outcome": normalized_outcome(row), "success": row["success"], "episode_steps": row["episode_steps"],
                "completion_time_sec": row["episode_steps"] * DT if row["success"] else "",
                "J_def": row["J_def"], "eta1": row.get("eta_hat", [None, None, None])[0] if row.get("eta_hat") else "",
                "eta2": row.get("eta_hat", [None, None, None])[1] if row.get("eta_hat") else "",
                "eta3": row.get("eta_hat", [None, None, None])[2] if row.get("eta_hat") else "",
                "eta_norm": row.get("eta_norm", ""), "eta_clipped": row.get("eta_clipped", ""),
                "agent_collision": row.get("agent_collision", False), "wall_collision": row.get("wall_collision", False),
                "invalid_actions": row.get("invalid_actions", 0), "nan_inf_events": row.get("nan_inf_events", 0),
                "projection_failures": row.get("projection_failures", 0), "runtime_seconds": row.get("runtime_seconds", 0),
            })

    direct_pair, direct_pair_rows = summarize_pair(records, "Safety", "Direct-g H8", 1)
    eta_pair, eta_pair_rows = summarize_pair(records, "Safety", "Structured eta", 2)
    safety_pairs = {"Direct-g H8": direct_pair, "Structured eta": eta_pair}
    rescue_break_rows = [{"controller": name, **value} for name, value in safety_pairs.items()]

    failure_rows: list[dict[str, Any]] = []
    for learned in ("Direct-g H8", "Structured eta"):
        for source_failure in ("timeout", "deadlock"):
            selected = [i for i in range(N) if normalized_outcome(records["Safety"][i]) == source_failure]
            destinations = Counter(normalized_outcome(records[learned][i]) for i in selected)
            rescued = destinations["success"]
            failure_rows.append({
                "controller": learned, "Safety_failure_type": source_failure, "Safety_count": len(selected),
                "rescued": rescued, "rescue_rate": rescued / len(selected) if selected else None,
                "rescue_rate_wilson_low": wilson(rescued, len(selected))[0] if selected else None,
                "rescue_rate_wilson_high": wilson(rescued, len(selected))[1] if selected else None,
                **{f"destination_{key}": destinations[key] for key in ("success", "deadlock", "timeout", "collision", "other")},
            })

    direct_vs_eta_rows: list[dict[str, Any]] = []
    for index in range(N):
        do = normalized_outcome(records["Direct-g H8"][index])
        eo = normalized_outcome(records["Structured eta"][index])
        cell = "BOTH_SUCCESS" if do == eo == "success" else "ETA_ONLY_SUCCESS" if do != "success" and eo == "success" else "DIRECT_ONLY_SUCCESS" if do == "success" and eo != "success" else "BOTH_FAIL"
        direct_vs_eta_rows.append({
            "episode_index": index, "Safety_outcome": normalized_outcome(records["Safety"][index]),
            "Direct_outcome": do, "Eta_outcome": eo, "paired_cell": cell,
        })
    de_cells = Counter(row["paired_cell"] for row in direct_vs_eta_rows)
    de_differences = np.asarray([int(records["Structured eta"][i]["success"]) - int(records["Direct-g H8"][i]["success"]) for i in range(N)], dtype=float)
    de_summary: dict[str, Any] = {
        **{key: de_cells[key] for key in ("BOTH_SUCCESS", "ETA_ONLY_SUCCESS", "DIRECT_ONLY_SUCCESS", "BOTH_FAIL")},
        "Delta_Q_eta_minus_direct": de_differences.mean(),
        **paired_test(de_cells["ETA_ONLY_SUCCESS"], de_cells["DIRECT_ONLY_SUCCESS"], de_differences, 3),
        "by_Safety_outcome": {},
    }
    for safety_outcome in ("success", "timeout", "deadlock"):
        selected = [row for row in direct_vs_eta_rows if row["Safety_outcome"] == safety_outcome]
        cells = Counter(row["paired_cell"] for row in selected)
        de_summary["by_Safety_outcome"][safety_outcome] = {"count": len(selected), **dict(cells)}

    deformation: dict[str, Any] = {"definition": "dt * sum_t ||u_exec-u_safe||_2^2", "dt": DT, "controllers": {}}
    for condition in ("Direct-g H8", "Structured eta"):
        cells = direct_pair_rows if condition == "Direct-g H8" else eta_pair_rows
        entry = {"overall": stats(records[condition][i]["J_def"] for i in range(N))}
        for category in ("BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL"):
            selected = [row["episode_index"] for row in cells if row["paired_cell"] == category]
            entry[category] = stats(records[condition][i]["J_def"] for i in selected)
        deformation["controllers"][condition] = entry

    completion: dict[str, Any] = {"controllers": {}, "paired_completion_time_differences_sec": {}}
    for condition in CONDITIONS:
        completion["controllers"][condition] = stats((records[condition][i]["episode_steps"] * DT for i in range(N) if records[condition][i]["success"]), completion=True)
    for a, b in (("Safety", "Direct-g H8"), ("Safety", "Structured eta"), ("Direct-g H8", "Structured eta")):
        selected = [i for i in range(N) if records[a][i]["success"] and records[b][i]["success"]]
        completion["paired_completion_time_differences_sec"][f"{b}_minus_{a}"] = stats(((records[b][i]["episode_steps"] - records[a][i]["episode_steps"]) * DT for i in selected), completion=True)

    eta_rows = records["Structured eta"]
    eta_vectors = np.asarray([eta_rows[i]["eta_hat"] for i in range(N)], dtype=float)
    eta_norms = np.linalg.norm(eta_vectors, axis=1)
    eta_stats: dict[str, Any] = {
        "coordinate_summary": {f"eta{j+1}": {"mean": float(eta_vectors[:, j].mean()), "std": float(eta_vectors[:, j].std(ddof=1)), "p05": float(np.quantile(eta_vectors[:, j], .05)), "median": float(np.median(eta_vectors[:, j])), "p95": float(np.quantile(eta_vectors[:, j], .95))} for j in range(3)},
        "eta_norm": stats(eta_norms), "clipped_count": sum(row["eta_clipped"] for row in eta_rows.values()),
        "clipping_fraction": sum(row["eta_clipped"] for row in eta_rows.values()) / N,
        "near_zero_threshold": .05, "near_zero_count": int(np.sum(eta_norms <= .05)),
        "near_zero_fraction": float(np.mean(eta_norms <= .05)), "by_structured_outcome": {}, "by_Safety_pair_category": {},
    }
    for category in ("success", "deadlock", "timeout", "collision", "other"):
        selected = [i for i in range(N) if normalized_outcome(eta_rows[i]) == category]
        eta_stats["by_structured_outcome"][category] = {"eta_norm": stats(eta_norms[selected]), "count": len(selected)}
    for category in ("BOTH_SUCCESS", "RESCUE", "BREAK", "BOTH_FAIL"):
        selected = [row["episode_index"] for row in eta_pair_rows if row["paired_cell"] == category]
        eta_stats["by_Safety_pair_category"][category] = {"eta_norm": stats(eta_norms[selected]), "count": len(selected)}

    nominal_rows: list[dict[str, Any]] = []
    nominal_summary: dict[str, Any] = {}
    safety_success = [i for i in range(N) if records["Safety"][i]["success"]]
    for learned in ("Direct-g H8", "Structured eta"):
        breaks = [i for i in safety_success if not records[learned][i]["success"]]
        nominal_summary[learned] = {"Safety_success_count": len(safety_success), "break_count": len(breaks), "break_rate": len(breaks) / len(safety_success), "break_rate_wilson_95_ci": wilson(len(breaks), len(safety_success))}
    for group, selected in (("PRESERVED", [i for i in safety_success if records["Structured eta"][i]["success"]]), ("BROKEN", [i for i in safety_success if not records["Structured eta"][i]["success"]])):
        executed_means = [eta_rows[i]["executed_correction_norm"]["mean"] for i in selected]
        nominal_rows.append({"structured_eta_group": group, "count": len(selected), "mean_eta_norm": float(np.mean(eta_norms[selected])) if selected else None, "mean_executed_correction_norm": float(np.mean(executed_means)) if selected else None, "mean_J_def": float(np.mean([eta_rows[i]["J_def"] for i in selected])) if selected else None})

    hard_safety: dict[str, Any] = {"controllers": {}}
    for condition in CONDITIONS:
        hard_safety["controllers"][condition] = {
            "agent_collisions": sum(int(row.get("agent_collision", False)) for row in records[condition].values()),
            "wall_collisions": sum(int(row.get("wall_collision", False)) for row in records[condition].values()),
            "invalid_actions": sum(int(row.get("invalid_actions", 0)) for row in records[condition].values()),
            "nan_inf_events": sum(int(row.get("nan_inf_events", 0)) for row in records[condition].values()),
            "projection_solver_failures": sum(int(row.get("projection_failures", 0)) for row in records[condition].values()),
        }
    hard_safety["intact"] = not any(value for entry in hard_safety["controllers"].values() for value in entry.values())

    eta_q = primary["controllers"]["Structured eta"]["Q"]
    direct_q = primary["controllers"]["Direct-g H8"]["Q"]
    safety_q = primary["controllers"]["Safety"]["Q"]
    eta_deadlock_rescue = next(row["rescued"] for row in failure_rows if row["controller"] == "Structured eta" and row["Safety_failure_type"] == "deadlock")
    direct_deadlock_rescue = next(row["rescued"] for row in failure_rows if row["controller"] == "Direct-g H8" and row["Safety_failure_type"] == "deadlock")
    eta_vs_direct_ci = de_summary["paired_bootstrap_95_ci"]
    if hard_safety["intact"] and eta_q > direct_q and eta_vs_direct_ci[0] > 0 and eta_deadlock_rescue >= direct_deadlock_rescue and eta_pair["net_rescue"] > 0:
        classification = "STRUCTURED_ETA_DOMINATES_DIRECT_G"
    elif hard_safety["intact"] and eta_q > safety_q and eta_pair["paired_bootstrap_95_ci"][0] > 0 and eta_q >= direct_q - .03 and eta_deadlock_rescue > direct_deadlock_rescue and eta_pair["break_rate"] <= .10:
        classification = "STRUCTURED_ETA_FULL_EPISODE_CONFIRMED"
    elif eta_deadlock_rescue > direct_deadlock_rescue and eta_q < direct_q:
        classification = "STRUCTURED_ETA_STRICT_ONLY"
    else:
        classification = "STRUCTURED_ETA_NO_GENERAL_WIDE_ADVANTAGE"
    primary["classification"] = classification
    primary["safety_referenced"] = safety_pairs
    primary["direct_vs_structured_eta"] = de_summary

    runtime_files = sorted((HERE / "runs").glob("runtime_*.json"))
    runtimes = [json.loads(path.read_text()) for path in runtime_files]
    starts = [datetime.fromisoformat(row["started_utc"]) for row in runtimes]
    finishes = [datetime.fromisoformat(row["finished_utc"]) for row in runtimes]
    resource = json.loads((HERE / "resource_audit.json").read_text())
    runtime = {
        "worker_count": len(runtimes), "rollout_wall_seconds": (max(finishes) - min(starts)).total_seconds(),
        "first_worker_started_utc": min(starts).isoformat(), "last_worker_finished_utc": max(finishes).isoformat(),
        "controller_episode_count": 3 * N, "sum_controller_episode_runtime_seconds": sum(row["runtime_seconds"] for condition in CONDITIONS for row in records[condition].values()),
        "resources": resource, "runtime_records": runtimes,
    }
    integrity.update({"overlap_status": overlap["status"], "exact_overlap": overlap["exact_initial_state_match_count"], "hard_safety_intact": hard_safety["intact"], "no_training_or_tuning": True})

    write_csv(HERE / "per_episode_results.csv", per_episode)
    write_json(HERE / "primary_metrics.json", primary)
    write_csv(HERE / "safety_rescue_break.csv", rescue_break_rows)
    write_csv(HERE / "failure_type_rescue.csv", failure_rows)
    write_csv(HERE / "direct_vs_eta_paired.csv", direct_vs_eta_rows)
    write_csv(HERE / "eta_prediction_statistics.csv", [{"scope": "overall", **{f"eta{j+1}_mean": float(eta_vectors[:, j].mean()) for j in range(3)}, "eta_norm_mean": eta_stats["eta_norm"]["mean"], "eta_norm_median": eta_stats["eta_norm"]["median"], "eta_norm_p95": eta_stats["eta_norm"]["p95"], "clipping_fraction": eta_stats["clipping_fraction"], "near_zero_fraction": eta_stats["near_zero_fraction"]}] + [{"scope": f"outcome_{key}", "count": value["count"], "eta_norm_mean": value["eta_norm"]["mean"], "eta_norm_median": value["eta_norm"]["median"], "eta_norm_p95": value["eta_norm"]["p95"]} for key, value in eta_stats["by_structured_outcome"].items()])
    write_json(HERE / "eta_prediction_statistics.json", eta_stats)
    write_csv(HERE / "nominal_preservation.csv", [{"controller": key, **value} for key, value in nominal_summary.items()] + nominal_rows)
    write_json(HERE / "deformation_metrics.json", deformation)
    deformation_rows: list[dict[str, Any]] = []
    for condition, scopes in deformation["controllers"].items():
        for scope, values in scopes.items():
            deformation_rows.append({"controller": condition, "scope": scope, **values})
    write_csv(HERE / "deformation_metrics.csv", deformation_rows)
    write_json(HERE / "completion_time_metrics.json", completion)
    completion_rows = [
        {"comparison": condition, "scope": "successful_episodes", **values}
        for condition, values in completion["controllers"].items()
    ] + [
        {"comparison": comparison, "scope": "paired_both_success", **values}
        for comparison, values in completion["paired_completion_time_differences_sec"].items()
    ]
    write_csv(HERE / "completion_time_metrics.csv", completion_rows)
    write_json(HERE / "hard_safety_checks.json", hard_safety)
    write_json(HERE / "integrity_audit.json", integrity)
    write_json(HERE / "runtime_statistics.json", runtime)

    def pct(value: float) -> str:
        return f"{100*value:.1f}%"
    report = f"""# Structured eta fresh WIDE full-episode audit

## Frozen protocol

- Manifest frozen before rollout: `{manifest['frozen_utc']}`; file SHA256 `{sha256(MANIFEST)}`.
- Authoritative WIDE generator: `{manifest['generator']['implementation']}` with IC root `{manifest['generator']['ic_generator_seed']}` and Flow root `{manifest['flow_randomness']['root_seed']}`.
- Fresh overlap audit: **PASS**, exact overlap **0** against {overlap['prior_initial_records_checked']} prior records ({overlap['prior_initial_unique_count']} unique ICs).
- Safety, direct-g H8 one-step, and one-shot persistent structured eta used identical ICs and Flow streams.
- No training, eta search, oracle query, gate, cadence/model selection, or post-hoc replacement was performed.

## Primary outcome (N=200)

| Controller | Success | Strict deadlock | Timeout | Collision | Q (95% Wilson CI) | Mean J_def | Median completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for condition in CONDITIONS:
        metric = primary["controllers"][condition]
        mean_j = 0.0 if condition == "Safety" else deformation["controllers"][condition]["overall"]["mean"]
        report += f"| {condition} | {metric['success']} | {metric['strict_deadlock']} | {metric['timeout']} | {metric['collision']} | {metric['Q']:.3f} [{metric['Q_wilson_95_ci'][0]:.3f}, {metric['Q_wilson_95_ci'][1]:.3f}] | {mean_j:.4f} | {completion['controllers'][condition]['median']:.2f} |\n"
    report += "\n## Safety-referenced efficacy\n\n"
    report += "| Learned controller | Rescue | Rescue rate | Break | Break rate | Net rescue | Delta Q (paired 95% CI) | McNemar p |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for condition, pair in safety_pairs.items():
        report += f"| {condition} | {pair['rescue_count']} | {pct(pair['rescue_rate'])} | {pair['break_count']} | {pct(pair['break_rate'])} | {pair['net_rescue']} | {pair['Delta_Q']:+.3f} [{pair['paired_bootstrap_95_ci'][0]:+.3f}, {pair['paired_bootstrap_95_ci'][1]:+.3f}] | {pair['exact_McNemar_two_sided_p']:.3g} |\n"
    report += "\n## Failure-mode rescue\n\n| Method | Safety failure | Count | Rescued | Rescue rate |\n|---|---|---:|---:|---:|\n"
    for row in failure_rows:
        report += f"| {row['controller']} | {row['Safety_failure_type']} | {row['Safety_count']} | {row['rescued']} | {pct(row['rescue_rate']) if row['rescue_rate'] is not None else 'n/a'} |\n"
    report += f"""

## Direct-g versus structured eta

- BOTH_SUCCESS: {de_cells['BOTH_SUCCESS']}; ETA_ONLY_SUCCESS: {de_cells['ETA_ONLY_SUCCESS']}; DIRECT_ONLY_SUCCESS: {de_cells['DIRECT_ONLY_SUCCESS']}; BOTH_FAIL: {de_cells['BOTH_FAIL']}.
- Paired Delta Q (eta - direct): {de_summary['Delta_Q_eta_minus_direct']:+.3f}, 95% paired bootstrap CI [{de_summary['paired_bootstrap_95_ci'][0]:+.3f}, {de_summary['paired_bootstrap_95_ci'][1]:+.3f}].
- Structured-eta nominal break rate among Safety successes: {pct(nominal_summary['Structured eta']['break_rate'])}; direct-g: {pct(nominal_summary['Direct-g H8']['break_rate'])}.

## One-shot eta and safety

- Eta clipping: {eta_stats['clipped_count']}/{N} ({pct(eta_stats['clipping_fraction'])}); near-zero (norm <= 0.05): {eta_stats['near_zero_count']}/{N}.
- Eta norm mean/median/P95: {eta_stats['eta_norm']['mean']:.3f} / {eta_stats['eta_norm']['median']:.3f} / {eta_stats['eta_norm']['p95']:.3f}.
- Eta norm by outcome (mean): success {eta_stats['by_structured_outcome']['success']['eta_norm']['mean']:.3f}, deadlock {eta_stats['by_structured_outcome']['deadlock']['eta_norm']['mean']:.3f}, timeout {eta_stats['by_structured_outcome']['timeout']['eta_norm']['mean']:.3f}. Successful episodes tended to receive larger modes; this is descriptive, not a tuned threshold.
- Hard safety intact: **{hard_safety['intact']}**. No collision, invalid action, NaN/Inf, or projection solver failure was accepted.

## Conclusion

**{classification}**

The frozen test result is reported without model or protocol changes. Detailed paired cells, deformation by outcome, completion-time comparisons, eta-outcome associations, and integrity checks are in the machine-readable artifacts.
"""
    (HERE / "fresh_wide_report.md").write_text(report)

    created = datetime.now(timezone.utc).isoformat()
    required = [
        "fresh_wide_manifest.json", "overlap_audit.json", "integrity_audit.json", "controller_manifest.json",
        "per_episode_results.csv", "primary_metrics.json", "safety_rescue_break.csv", "failure_type_rescue.csv",
        "direct_vs_eta_paired.csv", "eta_prediction_statistics.csv", "nominal_preservation.csv",
        "deformation_metrics.csv", "deformation_metrics.json", "completion_time_metrics.csv", "completion_time_metrics.json", "hard_safety_checks.json",
        "runtime_statistics.json", "fresh_wide_report.md",
    ]
    output_hashes = {name: sha256(HERE / name) for name in required}
    final_manifest = {
        "schema": "structured_eta_fresh_wide_final_manifest_v1", "created_utc": created,
        "classification": classification, "episode_count": N, "controllers": list(CONDITIONS),
        "fresh_manifest_frozen_before_rollout": True, "exact_overlap_count": 0,
        "direct_g_checkpoint_sha256": EXPECTED_DIRECT_SHA, "structured_eta_checkpoint_sha256": EXPECTED_ETA_SHA,
        "no_training_or_tuning": True, "output_sha256": output_hashes,
    }
    final_manifest["content_sha256"] = canonical_hash(final_manifest)
    write_json(HERE / "manifest.json", final_manifest)
    print(json.dumps({"status": "PASS", "classification": classification, "primary": primary, "hard_safety": hard_safety, "runtime": {key: value for key, value in runtime.items() if key != "runtime_records"}}, indent=2))


if __name__ == "__main__":
    main()
