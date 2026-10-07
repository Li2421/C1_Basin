"""Plan and execute result-dependent Stage-1 adaptive follow-up rounds.

The preregistered coarse/phase-1 selection remains immutable.  This module is
only entered after that phase has completed.  At each round it recomputes the
paired success evidence from completed, plan-bound manifests and does exactly
two kinds of follow-up:

* an unresolved transition point advances from 64 -> 128 -> 256 matched
  continuations (never beyond 256); and
* a state whose d=64 result becomes resolved-safe after escalation becomes
  eligible for the preregistered d=128 extension.

Every round is hash-bound to the complete evidence that preceded it.  Derived
selection files contain only tuples absent from all preceding completed
manifests.  The common frozen rollout runner remains the sole physics path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage1_statistics as stats


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/zhihan/research/Basin_C1")
PYTHON = Path("/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python")
FOLLOWUP = HERE / "adaptive_followup"

# These hashes are deliberately repeated here rather than inferred from a
# mutable upstream manifest.  The follow-up is invalid if any frozen input is
# edited.
PLAN_SHA = "0d4a8ac8696b5284c577d764c28c5200e776b26aa8ce400724575f6ba5741a0f"
INDEX_SHA = "c30a73b57bf77d75555cb6495be725f203d93c92a5aafc831b9d24fa32bc0ab9"
RUNNER_SHA = "517f408b08572dd3b271cabf715d61a23d49e21474ccf8ac1e516c3ef06f5e55"
MAX_MATCHED = 256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def frozen_inputs() -> tuple[dict, dict]:
    paths = {
        "audit_plan_sha256": HERE / "audit_plan.json",
        "tuple_index_sha256": HERE / "tuple_index.jsonl",
        "runner_sha256": HERE / "run_delay_rollouts.py",
    }
    actual = {name: sha(path) for name, path in paths.items()}
    expected = {
        "audit_plan_sha256": PLAN_SHA,
        "tuple_index_sha256": INDEX_SHA,
        "runner_sha256": RUNNER_SHA,
    }
    if actual != expected:
        raise AssertionError({"frozen_input_hash_mismatch": {"expected": expected, "actual": actual}})
    return json.loads((HERE / "audit_plan.json").read_text()), actual


def require_phase1() -> dict:
    selection = HERE / "adaptive_selection.json"
    manifest = HERE / "adaptive_job_manifest.json"
    if not selection.exists() or not manifest.exists():
        raise RuntimeError("coarse/phase-1 adaptive selection has not been prepared")
    jobs = json.loads(manifest.read_text())
    if jobs.get("job_count", 0):
        complete = HERE / "adaptive_execution_complete.json"
        if not complete.exists():
            raise RuntimeError("phase-1 adaptive jobs are not complete")
        done = json.loads(complete.read_text())
        if done.get("adaptive_selection_sha256") != sha(selection):
            raise AssertionError("phase-1 completion/selection hash mismatch")
        if not done.get("all_jobs_executed_and_checked"):
            raise AssertionError("phase-1 adaptive integrity was not certified")
    return {
        "adaptive_selection_sha256": sha(selection),
        "adaptive_job_manifest_sha256": sha(manifest),
        "adaptive_execution_complete_sha256": (
            sha(HERE / "adaptive_execution_complete.json")
            if (HERE / "adaptive_execution_complete.json").exists() else None
        ),
    }


def record_key_digest(keys: set[tuple[str, int, int]]) -> str:
    return canonical_hash([[sid, delay, seed] for sid, delay, seed in sorted(keys)])


def paired_effects(plan: dict, records: dict[tuple[str, int, int], dict]) -> dict[str, dict[int, dict]]:
    """Compute the same paired Q status as stage1_statistics, without J work."""
    output: dict[str, dict[int, dict]] = defaultdict(dict)
    for state in plan["states"]:
        sid = state["state_id"]
        delays = sorted({delay for ss, delay, _ in records if ss == sid})
        if 0 not in delays:
            raise AssertionError(f"{sid}: missing d0 evidence")
        seeds0 = {seed for ss, delay, seed in records if ss == sid and delay == 0}
        output[sid][0] = {"q_status": "REFERENCE", "n_matched": len(seeds0)}
        for delay in delays:
            if delay == 0:
                continue
            common = sorted(seeds0 & {seed for ss, dd, seed in records if ss == sid and dd == delay})
            if not common:
                continue
            s0 = np.asarray([bool(records[(sid, 0, seed)]["success"]) for seed in common], np.int8)
            sh = np.asarray([bool(records[(sid, delay, seed)]["success"]) for seed in common], np.int8)
            diff = s0 - sh
            lo, hi = stats.bootstrap_mean(diff, stats.stable_seed(f"Q:{sid}:{delay}:{len(common)}"))
            output[sid][delay] = {
                "q_status": stats.q_status(diff, lo, hi),
                "n_matched": len(common),
                "Q0": float(s0.mean()),
                "QH": float(sh.mean()),
                "Delta_Q0_minus_QH": float(diff.mean()),
                "paired_bootstrap95_lower": lo,
                "paired_bootstrap95_upper": hi,
            }
    return output


def reserve_seeds(state: dict, target_n: int) -> list[int]:
    initial = [int(value) for value in state["seeds_initial"]]
    if len(initial) != 64 or len(set(initial)) != 64:
        raise AssertionError(f"{state['state_id']}: expected exactly 64 frozen initial seeds")
    if target_n not in (128, 256):
        raise ValueError(target_n)
    extra = target_n - 64
    base = 100_000_000 + (int(state["rng_namespace"]) % 1_000_000) * 256
    values = list(range(base, base + extra))
    if set(values) & set(initial):
        raise AssertionError(f"{state['state_id']}: reserve/initial seed collision")
    return initial + values


def next_target(n: int) -> int | None:
    if n < 64:
        raise AssertionError(f"adaptive point has fewer than 64 matched seeds: {n}")
    if n < 128:
        return 128
    if n < MAX_MATCHED:
        return MAX_MATCHED
    return None


def select_round(plan: dict, effects: dict[str, dict[int, dict]], records: dict) -> tuple[list[dict], list[dict]]:
    """Return requested tuple groups and terminal ambiguous diagnostics."""
    groups: list[dict] = []
    terminal = []
    for state in plan["states"]:
        sid = state["state_id"]
        values = effects[sid]

        # Preserve the original d128 eligibility rule.  This is re-evaluated
        # after every round so a d64 point that becomes stable at n=128/256 is
        # no longer permanently excluded by its original n=64 status.
        d64 = values.get(64)
        if d64 and stats.resolved_zero(d64["q_status"]):
            initial = [int(seed) for seed in state["seeds_initial"]]
            missing = [seed for seed in initial if (sid, 128, seed) not in records]
            if missing:
                groups.append({
                    "kind": "newly_eligible_d128",
                    "delay": 128,
                    "state_id": sid,
                    "seeds": missing,
                    "requested_delays": [128],
                    "selection_reason": "d64 is now resolved-safe; complete frozen initial 64 seeds at d128",
                })

        bad = sorted(
            delay for delay, item in values.items()
            if delay > 0 and item["q_status"] == "SUPPORTED_RECOVERABILITY_LOSS"
        )
        first_bad = min(bad) if bad else None
        prior_safe = [0] + [
            delay for delay, item in values.items()
            if delay > 0 and (first_bad is None or delay < first_bad)
            and stats.resolved_zero(item["q_status"])
        ]
        last_safe = max(prior_safe)
        candidates = sorted(
            delay for delay, item in values.items()
            if delay > last_safe and item["q_status"] == "H_AMBIGUOUS"
            and (first_bad is None or delay <= first_bad)
        )
        if not candidates:
            continue
        delay = candidates[0]
        n = int(values[delay]["n_matched"])
        target = next_target(n)
        if target is None:
            terminal.append({
                "state_id": sid,
                "delay_steps": delay,
                "n_matched": n,
                "q_status": values[delay]["q_status"],
                "reason": "first transition-obstructing ambiguous point reached the 256-sample cap",
            })
            continue

        target_seeds = reserve_seeds(state, target)
        matched = {
            seed for seed in target_seeds
            if (sid, 0, seed) in records and (sid, delay, seed) in records
        }
        if len(matched) != n:
            raise AssertionError({
                "state_id": sid, "delay": delay, "observed_n": n,
                "target_namespace_matched": len(matched),
                "message": "matched evidence escaped the frozen initial/reserve namespace",
            })
        needed = [seed for seed in target_seeds if seed not in matched]
        if len(needed) != target - n:
            raise AssertionError((sid, delay, n, target, len(needed)))

        # Split by missing tuple pattern.  In particular, a d0 reserve seed
        # produced while escalating an earlier delay is reused as evidence,
        # never rerun in a later raw stage.
        patterns: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for seed in needed:
            missing_delays = tuple(
                d for d in (0, delay) if (sid, d, seed) not in records
            )
            if not missing_delays:
                raise AssertionError((sid, delay, seed, "needed seed has no missing tuple"))
            patterns[missing_delays].append(seed)
        for requested_delays, seeds in sorted(patterns.items()):
            groups.append({
                "kind": "ambiguous_transition_escalation",
                "delay": delay,
                "state_id": sid,
                "seeds": seeds,
                "requested_delays": list(requested_delays),
                "n_before": n,
                "target_n": target,
                "selection_reason": (
                    "first H_AMBIGUOUS point after the last resolved-safe delay and not beyond "
                    "the first supported-bad delay; advance only 64->128->256"
                ),
            })
    return groups, terminal


def index_reuse_inventory(plan: dict) -> tuple[set[tuple[str, int, int]], set[tuple[str, int]]]:
    exact, zero = set(), set()
    state = {row["state_id"]: row for row in plan["states"]}
    for line in (HERE / "tuple_index.jsonl").read_text().splitlines():
        item = json.loads(line)
        sid, delay, seed = item["state_id"], int(item["delay_steps"]), int(item["seed"])
        if item["status"] == "REUSE":
            exact.add((sid, delay, seed))
            if state[sid]["eta_best"] == [0.0, 0.0, 0.0]:
                zero.add((sid, seed))
    return exact, zero


def expected_physical(plan: dict, requested: set[tuple[str, int, int]]) -> int:
    exact, zero = index_reuse_inventory(plan)
    state = {row["state_id"]: row for row in plan["states"]}
    physical_nonzero = 0
    physical_zero = set()
    for sid, delay, seed in requested:
        if (sid, delay, seed) in exact:
            continue
        if state[sid]["eta_best"] == [0.0, 0.0, 0.0]:
            if (sid, seed) not in zero:
                physical_zero.add((sid, seed))
        else:
            physical_nonzero += 1
    return physical_nonzero + len(physical_zero)


def prepare_round(round_number: int) -> dict:
    if round_number < 1:
        raise ValueError(round_number)
    plan, frozen = frozen_inputs()
    phase1 = require_phase1()
    records, manifest_hashes = stats.completed_manifests(PLAN_SHA)
    preexisting_keys = set(records)
    effects = paired_effects(plan, records)
    groups, terminal = select_round(plan, effects, records)

    round_dir = FOLLOWUP / f"round_{round_number:02d}"
    manifest_path = round_dir / "job_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        # Preparing a round is immutable.  This makes supervisor restarts safe
        # even after some/all jobs from the round have completed.
        print(json.dumps({
            "round": round_number, "job_count": existing["job_count"],
            "reused_existing_frozen_manifest": True,
        }, indent=2))
        return existing

    if round_number > 1:
        previous = FOLLOWUP / f"round_{round_number - 1:02d}" / "execution_complete.json"
        if not previous.exists():
            raise RuntimeError(f"previous follow-up round is not complete: {previous}")
        previous_hash = sha(previous)
    else:
        previous_hash = None

    evidence = {
        **frozen,
        **phase1,
        "round": round_number,
        "previous_round_execution_complete_sha256": previous_hash,
        "completed_manifest_hashes": manifest_hashes,
        "preexisting_tuple_count": len(preexisting_keys),
        "preexisting_tuple_key_sha256": record_key_digest(preexisting_keys),
        "selection_protocol": (
            "first transition-obstructing H_AMBIGUOUS point per state; 64->128->256 only; "
            "re-evaluate resolved-safe d64 eligibility for initial-seed d128 every round"
        ),
    }
    evidence_hash = canonical_hash(evidence)
    selections_dir = round_dir / "selections"

    # Merge state requests having the same kind/delay/missing-delay pattern.
    merged: dict[tuple[str, int, tuple[int, ...]], dict[str, list[int]]] = defaultdict(dict)
    metadata: dict[tuple[str, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for item in groups:
        key = (item["kind"], int(item["delay"]), tuple(item["requested_delays"]))
        if item["state_id"] in merged[key]:
            raise AssertionError(("duplicate state request in merged job", key, item["state_id"]))
        merged[key][item["state_id"]] = [int(seed) for seed in item["seeds"]]
        metadata[key].append({k: v for k, v in item.items() if k != "seeds"})

    jobs = []
    requested_all: set[tuple[str, int, int]] = set()
    for job_index, (key, seed_map) in enumerate(sorted(merged.items(), key=lambda item: item[0])):
        kind, transition_delay, requested_delays = key
        stage = f"adaptive_followup_r{round_number:02d}_{job_index:02d}_{kind}_d{transition_delay}"
        selection = {
            "audit_plan_sha256": PLAN_SHA,
            "followup_round": round_number,
            "parent_evidence_sha256": evidence_hash,
            "selection_frozen_before_rollout": True,
            "kind": kind,
            "transition_delay": transition_delay,
            "requested_delays": list(requested_delays),
            "state_ids": sorted(seed_map),
            "state_seed_map": {sid: seed_map[sid] for sid in sorted(seed_map)},
            "state_decisions": sorted(metadata[key], key=lambda row: row["state_id"]),
        }
        selection_path = selections_dir / f"{stage}.json"
        atomic_json(selection_path, selection)
        requested = {
            (sid, delay, seed)
            for sid, seeds in seed_map.items()
            for seed in seeds for delay in requested_delays
        }
        if requested & preexisting_keys:
            raise AssertionError({"stage": stage, "preexisting_tuple_overlap": len(requested & preexisting_keys)})
        if requested & requested_all:
            raise AssertionError({"stage": stage, "cross_job_tuple_overlap": len(requested & requested_all)})
        requested_all.update(requested)
        jobs.append({
            "stage": stage,
            "kind": kind,
            "transition_delay": transition_delay,
            "delays": list(requested_delays),
            "state_count": len(seed_map),
            "selection_path": str(selection_path),
            "selection_sha256": sha(selection_path),
            "expected_records": len(requested),
            "expected_new_physical_rollouts": expected_physical(plan, requested),
            "requested_tuple_key_sha256": record_key_digest(requested),
        })

    manifest = {
        **evidence,
        "evidence_sha256": evidence_hash,
        "job_count": len(jobs),
        "jobs": jobs,
        "requested_tuple_count": len(requested_all),
        "requested_tuple_key_sha256": record_key_digest(requested_all),
        "terminal_ambiguous_count": len(terminal),
        "terminal_ambiguous_points": terminal,
        "all_jobs_execute_sequentially_on_one_gpu_shard": True,
        "prepared_only_no_rollout": True,
    }
    atomic_json(manifest_path, manifest)

    if not jobs:
        convergence = {
            **frozen,
            **phase1,
            "converged_after_round_planning": round_number,
            "terminal_ambiguous_points_at_256": terminal,
            "terminal_ambiguous_count": len(terminal),
            "round_job_manifest": str(manifest_path),
            "round_job_manifest_sha256": sha(manifest_path),
            "completed_manifest_hashes": manifest_hashes,
            "stop_rule": (
                "no newly eligible d128 tuples and no relevant transition-obstructing "
                "ambiguous point below 256 matched continuations"
            ),
        }
        atomic_json(FOLLOWUP / "convergence.json", convergence)
    print(json.dumps({
        "round": round_number,
        "job_count": len(jobs),
        "requested_tuple_count": len(requested_all),
        "terminal_ambiguous_at_256": len(terminal),
        "converged": not jobs,
    }, indent=2, sort_keys=True))
    return manifest


def validate_input_evidence(manifest: dict) -> set[tuple[str, int, int]]:
    plan, frozen = frozen_inputs()
    if any(manifest.get(key) != value for key, value in frozen.items()):
        raise AssertionError("round manifest frozen-input hash mismatch")
    require_phase1()
    records = {}
    for relative, expected in manifest["completed_manifest_hashes"].items():
        path = HERE / relative
        if not path.exists() or sha(path) != expected:
            raise AssertionError(f"input evidence manifest changed: {relative}")
        records_path = path.with_name("records.jsonl")
        source_manifest = json.loads(path.read_text())
        if source_manifest.get("records_sha256") and sha(records_path) != source_manifest["records_sha256"]:
            raise AssertionError(f"input evidence records changed: {records_path}")
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
            if key in records:
                old = records[key]
                for field in ("success", "outcome", "steps", "J_total"):
                    if old.get(field) != row.get(field):
                        raise AssertionError(f"conflicting input tuple {key}")
            records[key] = row
    keys = set(records)
    if len(keys) != manifest["preexisting_tuple_count"]:
        raise AssertionError("preexisting tuple count changed")
    if record_key_digest(keys) != manifest["preexisting_tuple_key_sha256"]:
        raise AssertionError("preexisting tuple digest changed")
    return keys


def execute_round(round_number: int) -> dict:
    round_dir = FOLLOWUP / f"round_{round_number:02d}"
    manifest_path = round_dir / "job_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"round was not prepared: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["round"] != round_number or not manifest["job_count"]:
        raise AssertionError("cannot execute empty/mismatched follow-up round")
    preexisting = validate_input_evidence(manifest)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.12",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    seen = set(preexisting)
    output_hashes = {}
    for job in manifest["jobs"]:
        selection = Path(job["selection_path"])
        if sha(selection) != job["selection_sha256"]:
            raise AssertionError(f"follow-up selection changed: {selection}")
        output = HERE / "raw" / f"{job['stage']}_shard0"
        completed = output / "manifest.json"
        if not completed.exists():
            subprocess.run([
                str(PYTHON), str(HERE / "run_delay_rollouts.py"),
                "--delays", ",".join(map(str, job["delays"])),
                "--stage", job["stage"], "--shard", "0", "--shards", "1",
                "--device", "gpu", "--batch", "16", "--seed-stop", "1000000",
                "--selection-json", str(selection),
            ], check=True, cwd=ROOT, env=env)
        subprocess.run([
            str(PYTHON), str(HERE / "check_completed_stage.py"),
            "--stage", job["stage"], "--shard", "0",
            "--expected-delays", ",".join(map(str, job["delays"])),
            "--expected-new", str(job["expected_new_physical_rollouts"]),
        ], check=True, cwd=ROOT, env=env)
        completed_manifest = json.loads(completed.read_text())
        records_path = output / "records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        keys = {(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in rows}
        if len(rows) != len(keys) or len(rows) != job["expected_records"]:
            raise AssertionError(f"follow-up output cardinality mismatch: {job['stage']}")
        if record_key_digest(keys) != job["requested_tuple_key_sha256"]:
            raise AssertionError(f"follow-up output tuple digest mismatch: {job['stage']}")
        if keys & seen:
            raise AssertionError(f"duplicate tuple across evidence/jobs: {job['stage']}")
        if completed_manifest.get("selection_sha256") != job["selection_sha256"]:
            raise AssertionError(f"selection provenance mismatch: {job['stage']}")
        seen.update(keys)
        output_hashes[str(completed.relative_to(HERE))] = sha(completed)

    completion = {
        "round": round_number,
        "job_manifest_sha256": sha(manifest_path),
        "evidence_sha256": manifest["evidence_sha256"],
        "job_count": manifest["job_count"],
        "completed_output_manifest_hashes": output_hashes,
        "all_jobs_executed_and_checked": True,
        "no_duplicate_tuples_against_preexisting_or_within_round": True,
        "post_round_tuple_count": len(seen),
        **{key: manifest[key] for key in ("audit_plan_sha256", "tuple_index_sha256", "runner_sha256")},
    }
    atomic_json(round_dir / "execution_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return completion


def verify_convergence() -> dict:
    plan, frozen = frozen_inputs()
    del plan
    path = FOLLOWUP / "convergence.json"
    if not path.exists():
        raise RuntimeError("follow-up convergence marker is missing")
    value = json.loads(path.read_text())
    if any(value.get(key) != expected for key, expected in frozen.items()):
        raise AssertionError("convergence frozen-input hash mismatch")
    round_manifest = Path(value["round_job_manifest"])
    if sha(round_manifest) != value["round_job_manifest_sha256"]:
        raise AssertionError("convergence round manifest changed")
    manifest = json.loads(round_manifest.read_text())
    if manifest["job_count"] != 0:
        raise AssertionError("convergence round unexpectedly contains jobs")
    for relative, expected in value["completed_manifest_hashes"].items():
        path_manifest = HERE / relative
        if not path_manifest.exists() or sha(path_manifest) != expected:
            raise AssertionError(f"convergence evidence changed: {relative}")
    print(json.dumps({
        "convergence_verified": True,
        "terminal_ambiguous_count": value["terminal_ambiguous_count"],
        "round": value["converged_after_round_planning"],
    }, indent=2, sort_keys=True))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "verify-convergence"))
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    if args.command == "verify-convergence":
        verify_convergence()
        return
    if args.round is None:
        parser.error("--round is required for prepare/execute")
    if args.command == "prepare":
        prepare_round(args.round)
    else:
        execute_round(args.round)


if __name__ == "__main__":
    main()
