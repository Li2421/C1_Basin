"""Index reusable delay tuples and preregister missing coarse rollouts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
DIAG = ROOT / "diagnostics"
HERE = Path(__file__).resolve().parent
PRIOR = DIAG / "intervention_delay_window_audit"
TOL = 1e-7
FILE_HASH_CACHE: dict[Path, str] = {}


def sha(path: Path) -> str:
    if path not in FILE_HASH_CACHE:
        FILE_HASH_CACHE[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return FILE_HASH_CACHE[path]


def compatible(a: dict, b: dict, deformation_field: str) -> bool:
    exact = ("outcome", "steps", "terminal_step")
    if any(a.get(field) != b.get(field) for field in exact):
        return False
    return abs(float(a.get(deformation_field, 0.0)) - float(b.get(deformation_field, 0.0))) <= TOL


def add_candidate(store, key, candidate, deformation_field):
    if key in store:
        old = store[key]
        if not compatible(old["record"], candidate["record"], deformation_field):
            raise AssertionError(f"inconsistent duplicate source tuple: {key}")
        return
    store[key] = candidate


def candidate(path: Path, line_number: int, line: str, record: dict, source_kind: str) -> dict:
    return {
        "source_path": str(path), "source_line_number": line_number,
        "source_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
        "source_file_sha256": sha(path), "source_kind": source_kind,
        "record": record,
    }


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan = json.loads(plan_path.read_text())
    state_by_id = {state["state_id"]: state for state in plan["states"]}
    eta_by_id = {state_id: tuple(state["eta_best"]) for state_id, state in state_by_id.items()}

    dataset = {}
    for version in ("v1", "v2", "v3", "v4"):
        for path in sorted((DIAG / f"gphi_training_dataset_{version}/raw").rglob("records.jsonl")):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                record = json.loads(line)
                state_id = record.get("state_id")
                if state_id not in state_by_id or record.get("execution_error") is not None:
                    continue
                eta = tuple(float(value) for value in record.get("eta", []))
                if eta != eta_by_id[state_id]:
                    continue
                key = (state_id, int(record["seed"]))
                add_candidate(dataset, key, candidate(path, line_number, line, record, f"dataset_{version}_eta_best"), "J_def")

    prior_plan_path = PRIOR / "audit_plan.json"
    prior_plan_hash = sha(prior_plan_path)
    prior_plan = json.loads(prior_plan_path.read_text())
    if prior_plan["checkpoint_sha256"] != plan["checkpoint_sha256"] or prior_plan["environment"] != plan["environment"]:
        raise AssertionError("prior audit physics/checkpoint mismatch")
    prior = {}
    valid_prior_stages = []
    for stage in sorted((PRIOR / "raw").iterdir()):
        manifest_path = stage / "manifest.json"
        path = stage / "records.jsonl"
        if not manifest_path.exists() or not path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("audit_plan_sha256") != prior_plan_hash:
            continue
        if manifest.get("records_sha256") != sha(path):
            raise AssertionError(f"prior source hash mismatch: {stage}")
        valid_prior_stages.append(stage.name)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            record = json.loads(line)
            state_id = record.get("state_id")
            if state_id not in state_by_id or record.get("execution_error") is not None:
                continue
            key = (state_id, int(record["delay_steps"]), int(record["seed"]))
            add_candidate(prior, key, candidate(path, line_number, line, record, "prior_delay_audit"), "J_total")

    index_path = HERE / "tuple_index.jsonl"
    inventory = Counter()
    state_inventory = defaultdict(Counter)
    with index_path.open("w") as handle:
        for state in plan["states"]:
            state_id = state["state_id"]
            zero_eta = state["eta_best"] == [0.0, 0.0, 0.0]
            for seed in state["seeds_initial"]:
                canonical = dataset.get((state_id, int(seed)))
                if canonical is None:
                    for candidate_delay in (0, 4):
                        canonical = prior.get((state_id, candidate_delay, int(seed)))
                        if canonical is not None:
                            break
                for delay in plan["coarse_delays"]:
                    exact = prior.get((state_id, int(delay), int(seed)))
                    selected = exact
                    reuse_kind = f"prior_exact_d{delay}" if exact is not None else None
                    if selected is None and delay == 0 and (state_id, int(seed)) in dataset:
                        selected = dataset[(state_id, int(seed))]
                        reuse_kind = "dataset_exact_eta_best_d0"
                    if selected is None and zero_eta and canonical is not None:
                        selected = canonical
                        reuse_kind = "zero_eta_canonical_all_delays"
                    status = "REUSE" if selected is not None else "NEW"
                    row = {
                        "state_id": state_id, "delay_steps": int(delay), "seed": int(seed),
                        "status": status, "reuse_kind": reuse_kind,
                        "source_path": None if selected is None else selected["source_path"],
                        "source_line_number": None if selected is None else selected["source_line_number"],
                        "source_line_sha256": None if selected is None else selected["source_line_sha256"],
                        "source_file_sha256": None if selected is None else selected["source_file_sha256"],
                        "source_kind": None if selected is None else selected["source_kind"],
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    inventory[(delay, status)] += 1
                    state_inventory[state_id][(delay, status)] += 1

    expected = len(plan["states"]) * 64 * len(plan["coarse_delays"])
    if sum(inventory.values()) != expected:
        raise AssertionError((sum(inventory.values()), expected))
    summary = {
        "audit_plan_sha256": sha(plan_path),
        "tuple_index_sha256": sha(index_path),
        "tuple_count": expected,
        "valid_prior_stages": valid_prior_stages,
        "valid_prior_stage_count": len(valid_prior_stages),
        "dataset_canonical_tuple_count": len(dataset),
        "prior_delay_tuple_count": len(prior),
        "by_delay": {
            str(delay): {status.lower(): inventory[(delay, status)] for status in ("REUSE", "NEW")}
            for delay in plan["coarse_delays"]
        },
        "short_coarse_d0_d4_d8_d16": {
            "reuse": sum(inventory[(delay, "REUSE")] for delay in (0, 4, 8, 16)),
            "new": sum(inventory[(delay, "NEW")] for delay in (0, 4, 8, 16)),
        },
        "long_coarse_d32_d64": {
            "reuse": sum(inventory[(delay, "REUSE")] for delay in (32, 64)),
            "new": sum(inventory[(delay, "NEW")] for delay in (32, 64)),
        },
        "float_duplicate_tolerance": TOL,
    }
    (HERE / "tuple_inventory.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
