"""Freeze the 17 source states, selected etas, and H1 reuse contract."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
SOURCE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from single_integrator.cbf import barrier_constraints  # noqa: E402
from single_integrator.environment import GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from single_integrator.outcomes import first_event  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def eta_key(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(round(float(item), 8) for item in value)  # type: ignore[arg-type]


def main() -> None:
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    reproduction = json.loads((SOURCE / "reproduction_audit.json").read_text())
    if reproduction.get("all_17_exact") is not True:
        raise RuntimeError("source strict-deadlock reproduction did not pass")
    with (SOURCE / "episode_capacity_classification.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with (SOURCE / "queried_states.csv").open(newline="") as handle:
        query_by_id = {row["state_id"]: row for row in csv.DictReader(handle)}
    with (SOURCE / "eta_search_results.csv").open(newline="") as handle:
        exact_rows = list(csv.DictReader(handle))
    source_cases = {
        row["case_id"]: row
        for row in json.loads((SOURCE / "strict_deadlock_manifest.json").read_text())["cases"]
    }

    state_rows = []
    eta_rows = []
    reuse = []
    expected_seeds = list(range(95310001, 95310065))
    for episode in episodes:
        state_id = episode["earliest_robust_state"]
        eta = eta_key(episode["best_eta_at_capacity_onset"])
        query = query_by_id[state_id]
        state_file = Path(query["state_file"])
        if sha256(state_file) != query["state_sha256"]:
            raise RuntimeError((state_id, "state hash mismatch"))
        robust_path = SOURCE / "raw/robust" / f"{state_id}.jsonl"
        robust_all = [json.loads(line) for line in robust_path.read_text().splitlines() if line]
        robust = [
            row for row in robust_all
            if not row.get("state_complete") and eta_key(row["eta"]) == eta
        ]
        robust.sort(key=lambda row: int(row["seed"]))
        exact = [
            row for row in exact_rows
            if row["state_id"] == state_id and eta_key(row["eta"]) == eta
        ]
        seeds = [int(row["seed"]) for row in robust]
        if seeds != expected_seeds or len(exact) != 1:
            raise RuntimeError((state_id, "reuse tuple mismatch", len(robust), len(exact)))
        if sum(row["outcome"] == "success" for row in robust) < 63:
            raise RuntimeError((state_id, "selected eta no longer B63"))
        if exact[0]["outcome"] != "success":
            raise RuntimeError((state_id, "selected eta exact-flow did not succeed"))
        case = source_cases[episode["case_id"]]
        state_rows.append({
            "case_id": episode["case_id"], "benchmark": episode["benchmark"],
            "episode_index": int(episode["episode_index"]), "state_id": state_id,
            "query_label": query["query_label"], "query_step": int(query["query_step"]),
            "seconds_before_deadlock": float(query["seconds_before_deadlock"]),
            "remaining_global_steps": int(query["remaining_global_steps"]),
            "state_file": str(state_file), "state_sha256": query["state_sha256"],
            "rng_namespace": int(query["rng_namespace"]),
            "exact_flow_root_seed": int(query["exact_flow_root_seed"]),
            "exact_flow_rollout_id": int(query["exact_flow_rollout_id"]),
            "source_trajectory": case["source_trajectory"],
            "source_trajectory_sha256": case["source_trajectory_sha256"],
            "global_cadence_phase_mod8": int(query["query_step"]) % 8,
        })
        eta_rows.append({
            "case_id": episode["case_id"], "state_id": state_id,
            "eta": list(eta), "selection": "frozen minimum-J_def robust B63 eta at earliest robust state",
            "source_success_count": int(episode["success_count"]),
            "source_evaluated": int(episode["evaluated"]),
            "source_dense_J_def": float(episode["J_def"]),
        })
        reuse.append({
            "case_id": episode["case_id"], "state_id": state_id,
            "eta": list(eta), "robust_path": str(robust_path),
            "robust_path_sha256": sha256(robust_path),
            "robust_seeds": seeds, "robust_successes": sum(row["outcome"] == "success" for row in robust),
            "exact_source_row": exact[0],
        })

    source_files = {
        "environment": Path(inspect.getsourcefile(GiveWayEnv)).resolve(),
        "projection": Path(inspect.getsourcefile(barrier_constraints)).resolve(),
        "retry": Path(inspect.getsourcefile(project_velocity_with_retry)).resolve(),
        "corrector": Path(inspect.getsourcefile(DiagnosticCorrector)).resolve(),
        "flow_loader": Path(inspect.getsourcefile(load_policy)).resolve(),
        "outcome": Path(inspect.getsourcefile(first_event)).resolve(),
    }
    capacity_frozen = json.loads((SOURCE / "strict_deadlock_manifest.json").read_text())
    for name in ("environment", "projection", "retry"):
        expected = capacity_frozen["source_sha256"][name]
        if sha256(source_files[name]) != expected:
            raise RuntimeError((name, "source hash changed"))
    write_json(HERE / "source_state_manifest.json", {
        "schema": "strict_deadlock_oracle_cadence_source_states_v1",
        "count": len(state_rows), "states": state_rows,
        "starting_rule": "earliest robust queried augmented state",
        "expected_start_labels": {"S0": 16, "S_8s": 1},
    })
    write_json(HERE / "eta_manifest.json", {
        "schema": "strict_deadlock_oracle_cadence_eta_v1",
        "search_performed": False, "etas": eta_rows,
    })
    write_json(HERE / "integrity_audit.json", {
        "status": "PREPARED_REUSE_GATE_PASS",
        "source_capacity_manifest": str(SOURCE / "manifest.json"),
        "source_capacity_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "source_capacity_status": source_manifest.get("status"),
        "states": len(state_rows),
        "historical": sum(row["benchmark"] == "historical" for row in state_rows),
        "fresh_unseen": sum(row["benchmark"] == "fresh_unseen" for row in state_rows),
        "state_hashes_pass": True, "eta_search_performed": False,
        "H1_robust_reuse": reuse,
        "H1_robust_reuse_count": 17 * 64,
        "H1_all_selected_etas_B63": True,
        "H1_all_selected_etas_exact_flow_success": True,
        "robust_seeds": expected_seeds,
        "continuation_horizon": "remaining global horizon through absolute step 850",
        "cadence_phase": "absolute global physical timestep; never reset at query",
        "source_files": {name: str(path) for name, path in source_files.items()},
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
    })
    print(json.dumps({"status": "PASS", "states": len(state_rows), "reused_H1_robust": 1088}, indent=2))


if __name__ == "__main__":
    main()
