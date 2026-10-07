"""Plan confidence continuations for every Dataset V4 state without spatial selection."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
PRIOR = ROOT / "diagnostics/oracle_boundary_confidence_audit"
ETA0 = (0.0, 0.0, 0.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    states = read_jsonl(DATA / "state_manifest.jsonl")
    state_by_id = {row["state_id"]: row for row in states}
    metadata = read_jsonl(DATA / "sample_metadata.jsonl")
    state_meta = {}
    for row in metadata:
        state_meta[row["state_id"]] = {
            "original_gate_label": 0 if bool(row["zero_label"]) else 1,
            "category": row["category"], "split": row["split"],
            "source_group": row["leakage_group"], "source_trajectory": row["source_trajectory"],
        }
    if set(state_meta) != set(state_by_id) or len(states) != 324:
        raise AssertionError((len(states), len(state_meta)))

    prior_stability = {row["state_id"]: row for row in csv.DictReader((PRIOR / "b63_resampling_stability.csv").open())}
    prior_rollouts = read_jsonl(PRIOR / "eta_zero_rollouts.jsonl")
    effective = {}; provenance = {}; conflicts = []
    # First scan the original oracle caches for all V4 states.
    cache_paths = sorted((ROOT / "diagnostics").glob("gphi_training_dataset_v[234]/raw/*/records.jsonl"))
    for path in cache_paths:
        for record in read_jsonl(path):
            if record["state_id"] not in state_by_id or tuple(float(v) for v in record["eta"]) != ETA0:
                continue
            key = (record["state_id"], int(record["seed"]))
            if key in effective and (effective[key]["outcome"], int(effective[key]["steps"])) != (record["outcome"], int(record["steps"])):
                conflicts.append((key, provenance[key], str(path)))
            effective[key] = record; provenance[key] = str(path.relative_to(ROOT))
    # The prior enlarged audit supersedes identical cached tuples and adds independent seeds.
    for record in prior_rollouts:
        key = (record["state_id"], int(record["seed"]))
        if key in effective and (effective[key]["outcome"], int(effective[key]["steps"])) != (record["outcome"], int(record["steps"])):
            conflicts.append((key, provenance[key], str(PRIOR / "eta_zero_rollouts.jsonl")))
        effective[key] = record; provenance[key] = str((PRIOR / "eta_zero_rollouts.jsonl").relative_to(ROOT))
    if conflicts:
        raise RuntimeError(("conflicting cached tuples", conflicts[:10]))

    existing = []
    for key in sorted(effective):
        row = dict(effective[key]); row["cache_source"] = provenance[key]; row["confidence_stage"] = "existing"
        existing.append(row)
    write_jsonl(HERE / "existing_eta_zero_rollouts.jsonl", existing)
    by_state = defaultdict(list)
    for row in existing:
        by_state[row["state_id"]].append(row)

    confidence_manifest = []
    for state in states:
        state_id = state["state_id"]; rows = by_state[state_id]; counts = Counter(row["outcome"] for row in rows)
        confidence_manifest.append({
            "state_id": state_id, **state_meta[state_id], "state_file": state["state_file"],
            "state_sha256": state["state_sha256"], "rng_namespace": state["rng_namespace"],
            "step": state["step"], "prior_confidence_audit": state_id in prior_stability,
            "existing_evaluated": len(rows), "existing_successes": counts["success"],
            "existing_failures": len(rows)-counts["success"], "existing_outcomes": dict(counts),
        })
    write_jsonl(HERE / "confidence_state_manifest.jsonl", confidence_manifest)

    used = set(effective)
    common_seeds = list(range(96100001, 96102000))
    arms = []
    for state in confidence_manifest:
        needed = max(0, 256-int(state["existing_evaluated"]))
        seeds = [seed for seed in common_seeds if (state["state_id"], seed) not in used][:needed]
        if len(seeds) != needed:
            raise AssertionError((state["state_id"], needed, len(seeds)))
        if seeds:
            arms.append({"arm_id": state["state_id"]+"__eta0_to256", "state_id": state["state_id"], "eta": [0., 0., 0.], "seeds": seeds})

    loads = [0, 0, 0, 0]; shards = [[], [], [], []]
    for arm in sorted(arms, key=lambda row: -(850-int(state_by_id[row["state_id"]]["step"]))*len(row["seeds"])):
        shard = min(range(4), key=lambda index: loads[index])
        shards[shard].append(arm)
        loads[shard] += (850-int(state_by_id[arm["state_id"]]["step"]))*len(arm["seeds"])
    write_json(HERE / "to256_arms.json", arms)
    for shard in range(4):
        write_json(HERE / f"to256_shard{shard}_arms.json", shards[shard])

    summary = {
        "selection_rule": "all 324 existing Dataset V4 states; no spatial/semantic/source/test-location selection",
        "states": len(states), "prior_enlarged_audit_states_reused": len(prior_stability),
        "existing_unique_eta0_tuples_reused": len(existing),
        "new_to256_tuples": sum(len(arm["seeds"]) for arm in arms),
        "states_already_at_or_above_256": sum(len(by_state[state["state_id"]]) >= 256 for state in states),
        "state_counts": {
            "by_original_label": dict(Counter(str(value["original_gate_label"]) for value in state_meta.values())),
            "by_category": dict(Counter(value["category"] for value in state_meta.values())),
            "by_split": dict(Counter(value["split"] for value in state_meta.values())),
        },
        "planned_max_physical_steps_by_shard": loads,
        "cache_paths": [str(path.relative_to(ROOT)) for path in cache_paths],
        "duplicate_or_conflicting_tuples": 0,
        "new_seed_namespace": "96100001+; independent of earlier 952/957/959/960 namespaces",
        "dataset_manifest_sha256": sha(DATA / "manifest.json"),
    }
    write_json(HERE / "confidence_plan.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
