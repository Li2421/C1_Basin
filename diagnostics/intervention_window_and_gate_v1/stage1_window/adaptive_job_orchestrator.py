"""Freeze and safely execute Stage-1 adaptive delay jobs.

The coarse statistics script is the sole owner of adaptive_selection.json.
This module never chooses states, delays, or seeds.  It validates that frozen
selection, converts it into runner-native selections, and optionally executes
the resulting jobs sequentially.  The common rollout runner remains the only
code that evaluates physics and it provides append-only, per-batch resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/zhihan/research/Basin_C1")
PYTHON = Path("/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python")
PLAN_SHA = "0d4a8ac8696b5284c577d764c28c5200e776b26aa8ce400724575f6ba5741a0f"
INDEX_SHA = "c30a73b57bf77d75555cb6495be725f203d93c92a5aafc831b9d24fa32bc0ab9"
RUNNER_SHA = "517f408b08572dd3b271cabf715d61a23d49e21474ccf8ac1e516c3ef06f5e55"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def frozen_inputs() -> tuple[dict, dict]:
    plan_path = HERE / "audit_plan.json"
    index_path = HERE / "tuple_index.jsonl"
    runner_path = HERE / "run_delay_rollouts.py"
    actual = {
        "audit_plan_sha256": sha(plan_path),
        "tuple_index_sha256": sha(index_path),
        "runner_sha256": sha(runner_path),
    }
    expected = {
        "audit_plan_sha256": PLAN_SHA,
        "tuple_index_sha256": INDEX_SHA,
        "runner_sha256": RUNNER_SHA,
    }
    if actual != expected:
        raise AssertionError({"frozen_input_hash_mismatch": {"expected": expected, "actual": actual}})
    return json.loads(plan_path.read_text()), actual


def validate_coarse_chain(selection: dict) -> None:
    if not selection.get("selection_frozen_before_adaptive_rollout"):
        raise AssertionError("adaptive selection was not frozen before rollouts")
    if selection.get("audit_plan_sha256") != PLAN_SHA:
        raise AssertionError("adaptive selection plan hash mismatch")
    hashes = selection.get("coarse_manifest_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise AssertionError("adaptive selection has no coarse manifest chain")
    required_names = {"raw/coarse_short_shard0/manifest.json", "raw/coarse_long_shard0/manifest.json"}
    if not required_names <= set(hashes):
        raise AssertionError({"missing_coarse_manifests": sorted(required_names - set(hashes))})
    for relative, expected_hash in hashes.items():
        path = HERE / relative
        if not path.exists() or sha(path) != expected_hash:
            raise AssertionError(f"coarse manifest hash mismatch: {relative}")


def validate_state_ids(plan: dict, state_ids: list[str]) -> list[str]:
    known = {state["state_id"] for state in plan["states"]}
    values = [str(value) for value in state_ids]
    if len(values) != len(set(values)) or not set(values) <= known:
        raise AssertionError("adaptive job contains duplicate or unknown state ids")
    return sorted(values)


def expected_new_physical(
    plan: dict,
    state_ids: list[str],
    seeds_by_state: dict[str, list[int]],
    delay_count: int,
    zero_canonical_is_preregistered: bool,
) -> int:
    by_id = {state["state_id"]: state for state in plan["states"]}
    total = 0
    for state_id in state_ids:
        if by_id[state_id]["eta_best"] == [0.0, 0.0, 0.0]:
            # Initial 64 seed sets have a preregistered canonical tuple in the
            # coarse index; genuinely new escalation seeds require exactly one
            # physical canonical continuation for all requested delays.
            multiplier = 0 if zero_canonical_is_preregistered else 1
        else:
            multiplier = delay_count
        total += multiplier * len(seeds_by_state[state_id])
    return total


def build_jobs() -> dict:
    plan, frozen = frozen_inputs()
    selection_path = HERE / "adaptive_selection.json"
    if not selection_path.exists():
        raise RuntimeError("stage1_statistics.py coarse must create adaptive_selection.json first")
    selection_hash = sha(selection_path)
    selection = json.loads(selection_path.read_text())
    validate_coarse_chain(selection)
    initial = {state["state_id"]: [int(value) for value in state["seeds_initial"]] for state in plan["states"]}
    derived_root = HERE / "adaptive_jobs"
    jobs: list[dict] = []

    def add_job(stage: str, delays: list[int], state_ids: list[str], state_seed_map=None, rule: str = "") -> None:
        ids = validate_state_ids(plan, state_ids)
        if not ids:
            return
        if any(delay < 0 or delay > 128 for delay in delays):
            raise AssertionError((stage, delays))
        payload = {
            "audit_plan_sha256": PLAN_SHA,
            "parent_adaptive_selection_sha256": selection_hash,
            "selection_frozen_before_rollout": True,
            "selection_rule": rule,
            "state_ids": ids,
        }
        if state_seed_map is not None:
            seed_map = {state_id: [int(value) for value in state_seed_map[state_id]] for state_id in ids}
            if any(not values or len(values) != len(set(values)) for values in seed_map.values()):
                raise AssertionError(f"empty or duplicate seed list in {stage}")
            payload["state_seed_map"] = seed_map
            seeds = seed_map
        else:
            seeds = {state_id: initial[state_id] for state_id in ids}
        path = derived_root / f"{stage}_selection.json"
        write_json_atomic(path, payload)
        expected_records = sum(len(seeds[state_id]) for state_id in ids) * len(delays)
        physical = expected_new_physical(
            plan, ids, seeds, len(delays),
            zero_canonical_is_preregistered=state_seed_map is None,
        )
        jobs.append({
            "stage": stage,
            "delays": sorted(delays),
            "state_count": len(ids),
            "selection_path": str(path),
            "selection_sha256": sha(path),
            "expected_records": expected_records,
            "expected_new_physical_rollouts": physical,
        })

    add_job(
        "adaptive_d128", [128], selection.get("d128_state_ids", []),
        rule="frozen d128_state_ids from adaptive_selection.json",
    )
    for delay_text, state_ids in sorted(selection.get("refinement_by_delay", {}).items(), key=lambda item: int(item[0])):
        delay = int(delay_text)
        add_job(
            f"adaptive_refine_d{delay}", [delay], state_ids,
            rule=f"frozen refinement_by_delay[{delay}] from adaptive_selection.json",
        )
    for delay_text, item in sorted(selection.get("seed_escalation_by_delay", {}).items(), key=lambda pair: int(pair[0])):
        delay = int(delay_text)
        ids = item["state_ids"]
        if set(ids) != set(item["state_seed_map"]):
            raise AssertionError(f"seed escalation state/map mismatch at d={delay}")
        # d0 is deliberately included: every extra delayed continuation must
        # have a matched new-seed baseline.  For eta=0 the common runner groups
        # the two logical records into one canonical physical continuation.
        add_job(
            f"adaptive_escalate_d{delay}", [0, delay], ids,
            state_seed_map=item["state_seed_map"],
            rule=f"frozen seed_escalation_by_delay[{delay}] with matched new-seed d0",
        )

    manifest = {
        **frozen,
        "adaptive_selection_path": str(selection_path),
        "adaptive_selection_sha256": selection_hash,
        "coarse_manifest_hashes": selection["coarse_manifest_hashes"],
        "job_count": len(jobs),
        "jobs": jobs,
        "execution_order_is_sequential": True,
        "gpu_shards_total_if_executed": 1,
        "prepared_only_no_rollout": True,
    }
    path = HERE / "adaptive_job_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise AssertionError("existing adaptive job manifest differs; refusing overwrite")
    write_json_atomic(path, manifest)
    return manifest


def execute_jobs(manifest: dict) -> None:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.12",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    for job in manifest["jobs"]:
        selection_path = Path(job["selection_path"])
        if sha(selection_path) != job["selection_sha256"]:
            raise AssertionError(f"derived selection changed: {selection_path}")
        output = HERE / "raw" / f"{job['stage']}_shard0"
        completed = output / "manifest.json"
        if completed.exists():
            completed_manifest = json.loads(completed.read_text())
            if completed_manifest.get("selection_sha256") != job["selection_sha256"]:
                raise AssertionError(f"completed stage selection mismatch: {job['stage']}")
            subprocess.run([
                str(PYTHON), str(HERE / "check_completed_stage.py"),
                "--stage", job["stage"], "--shard", "0",
                "--expected-delays", ",".join(map(str, job["delays"])),
                "--expected-new", str(job["expected_new_physical_rollouts"]),
            ], check=True, cwd=ROOT, env=env)
            continue
        subprocess.run([
            str(PYTHON), str(HERE / "run_delay_rollouts.py"),
            "--delays", ",".join(map(str, job["delays"])),
            "--stage", job["stage"], "--shard", "0", "--shards", "1",
            "--device", "gpu", "--batch", "16",
            "--selection-json", str(selection_path),
        ], check=True, cwd=ROOT, env=env)
        completed_manifest = json.loads(completed.read_text())
        if completed_manifest.get("selection_sha256") != job["selection_sha256"]:
            raise AssertionError(f"new stage selection mismatch: {job['stage']}")
        subprocess.run([
            str(PYTHON), str(HERE / "check_completed_stage.py"),
            "--stage", job["stage"], "--shard", "0",
            "--expected-delays", ",".join(map(str, job["delays"])),
            "--expected-new", str(job["expected_new_physical_rollouts"]),
        ], check=True, cwd=ROOT, env=env)
    subprocess.run([
        str(PYTHON), str(HERE / "check_adaptive_execution.py")
    ], check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute"))
    args = parser.parse_args()
    manifest = build_jobs()
    if args.command == "execute":
        execute_jobs(manifest)
        manifest = dict(manifest)
        manifest["prepared_only_no_rollout"] = False
        manifest["all_jobs_executed_and_checked"] = True
        write_json_atomic(HERE / "adaptive_execution_complete.json", manifest)
    print(json.dumps({
        "command": args.command,
        "adaptive_selection_sha256": manifest["adaptive_selection_sha256"],
        "job_count": manifest["job_count"],
        "jobs": [job["stage"] for job in manifest["jobs"]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
