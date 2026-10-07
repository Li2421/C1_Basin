"""Freeze the eta-zero audit cohort and plan only missing continuations."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
ETA_ZERO = (0.0, 0.0, 0.0)
OLD_HARD = {
    "R_D4_s95101008_p134", "R_D4_s95105001_p116",
    "R_D4_s95101014_p123", "R_D4_s95105004_p114",
}
EASY_ZERO = {"N_r143_s105", "N_r151_s124", "R_D2_s95101009_p112", "N_r066_s107"}
EASY_ONE = {"Q_pair228_m120", "P_r198_m120", "R_D4_s95101013_p027", "R_D1_s95106017_p018"}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader((DATA / "matched_boundary_pairs.csv").open()))
    pair_ids = {row[key] for row in pairs for key in ("zero_state_id", "nonzero_state_id")}
    state_rows = read_jsonl(DATA / "state_manifest.jsonl")
    state_by_id = {row["state_id"]: row for row in state_rows}
    metadata = read_jsonl(DATA / "sample_metadata.jsonl")
    state_label = {}; state_group = {}; state_split = {}; state_category = {}
    for row in metadata:
        state_label[row["state_id"]] = 0 if bool(row["zero_label"]) else 1
        state_group[row["state_id"]] = row["leakage_group"]
        state_split[row["state_id"]] = row["split"]
        state_category[row["state_id"]] = row["category"]
    heldout = {state_id for state_id in pair_ids if state_split[state_id] in {"validation", "test"}}
    target_ids = pair_ids | OLD_HARD | EASY_ZERO | EASY_ONE
    missing = target_ids - set(state_by_id)
    if missing:
        raise RuntimeError(("states missing from V4", sorted(missing)))

    source_paths = sorted((ROOT / "diagnostics").glob("gphi_training_dataset_v[234]/raw/*/records.jsonl"))
    effective = {}; provenance = {}; conflicts = []
    for path in source_paths:
        for record in read_jsonl(path):
            if record["state_id"] not in target_ids or tuple(float(value) for value in record["eta"]) != ETA_ZERO:
                continue
            key = (record["state_id"], int(record["seed"]))
            if key in effective and (effective[key]["outcome"], int(effective[key]["steps"])) != (record["outcome"], int(record["steps"])):
                conflicts.append({"state_id": key[0], "seed": key[1], "old": provenance[key], "new": str(path)})
            effective[key] = record; provenance[key] = str(path.relative_to(ROOT))
    if conflicts:
        raise RuntimeError(("conflicting cached tuples", conflicts))
    existing = []
    for key in sorted(effective):
        record = dict(effective[key]); record["cache_source"] = provenance[key]; record["audit_stage"] = "existing"
        existing.append(record)
    write_jsonl(HERE / "existing_eta_zero_rollouts.jsonl", existing)

    by_state = defaultdict(list)
    for record in existing: by_state[record["state_id"]].append(record)
    audited = []
    for state_id in sorted(target_ids):
        tags = []
        if state_id in pair_ids: tags.append("MATCHED_BOUNDARY")
        if state_id in heldout: tags.append("HELDOUT_CLOSE_BOUNDARY")
        if state_id in OLD_HARD: tags.append("OLD_HARD_ZERO")
        if state_id in EASY_ZERO: tags.append("EASY_GATE0_CONTROL")
        if state_id in EASY_ONE: tags.append("EASY_GATE1_CONTROL")
        records = by_state[state_id]; counts = Counter(row["outcome"] for row in records)
        expected_label = 0 if state_id in EASY_ZERO else 1 if state_id in EASY_ONE else state_label[state_id]
        if expected_label != state_label[state_id]: raise AssertionError((state_id, expected_label, state_label[state_id]))
        audited.append({
            "state_id": state_id, "audit_tags": tags, "original_gate_label": state_label[state_id],
            "category": state_category[state_id], "split": state_split[state_id], "source_group": state_group[state_id],
            "source_trajectory": state_by_id[state_id]["source_trajectory"], "state_file": state_by_id[state_id]["state_file"],
            "rng_namespace": state_by_id[state_id]["rng_namespace"], "step": state_by_id[state_id]["step"],
            "inter_agent_distance": state_by_id[state_id].get("inter_agent_distance"),
            "relative_velocity_norm": state_by_id[state_id].get("relative_velocity_norm"),
            "previous_evaluated": len(records), "previous_successes": counts["success"],
            "previous_failures": len(records) - counts["success"], "previous_outcomes": dict(counts),
            "previous_rule_evidence": f"{counts['success']}/{len(records)}" if len(records) == 64 else f"early-stop after {len(records)} with {len(records)-counts['success']} failures",
        })
    write_jsonl(HERE / "audited_state_manifest.jsonl", audited)

    # Common new seed namespace across states/pairs.  Each state receives only
    # enough missing independent seeds to reach exactly 256 total observations.
    common = list(range(96000001, 96000600))
    used = {(row["state_id"], int(row["seed"])) for row in existing}
    arms = []
    for state in audited:
        needed = 256 - state["previous_evaluated"]
        seeds = [seed for seed in common if (state["state_id"], seed) not in used][:needed]
        if len(seeds) != needed: raise AssertionError((state["state_id"], needed, len(seeds)))
        arms.append({"arm_id": state["state_id"] + "__eta0_to256", "state_id": state["state_id"], "eta": [0.0, 0.0, 0.0], "seeds": seeds})
    # Greedy workload assignment leaves both shards useful without splitting a state.
    loads = [0, 0]; shards = [[], []]
    for arm in sorted(arms, key=lambda row: -(850 - state_by_id[row["state_id"]]["step"]) * len(row["seeds"])):
        shard = 0 if loads[0] <= loads[1] else 1
        shards[shard].append(arm); loads[shard] += (850 - state_by_id[arm["state_id"]]["step"]) * len(arm["seeds"])
    write_json(HERE / "to256_arms.json", arms)
    for shard in (0, 1): write_json(HERE / f"to256_shard{shard}_arms.json", shards[shard])
    write_json(HERE / "audit_plan.json", {
        "states": len(audited), "matched_boundary_states": len(pair_ids), "old_hard_states": len(OLD_HARD),
        "heldout_boundary_states": len(heldout), "easy_gate0_controls": len(EASY_ZERO), "easy_gate1_controls": len(EASY_ONE),
        "existing_eta0_tuples_reused": len(existing), "new_to256_tuples": sum(len(arm["seeds"]) for arm in arms),
        "shard_planned_max_physical_steps": loads, "source_cache_files": [str(path.relative_to(ROOT)) for path in source_paths],
        "duplicate_or_conflicting_tuples": 0, "new_seed_namespace": "96000001+; disjoint from prior 952/957/959 namespaces",
    })
    print(json.dumps(read_json := json.loads((HERE / "audit_plan.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
