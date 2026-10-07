"""Freeze baseline and targeted eta-arm plans; maximize compatible cache reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "diagnostics/gphi_training_dataset_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def seed_set(state_id: str, default: list[int]) -> list[int]:
    if state_id == "D1_pair231":
        return list(range(95106001, 95106033)) + list(range(95107001, 95107033))
    if state_id in ("D2_pair228", "D4_pair227"):
        return list(range(95101001, 95101017)) + list(range(95105001, 95105017)) + list(range(95108001, 95108033))
    return default


def shard_arms(arms: list[dict], states: dict[str, dict], max_steps: int) -> list[list[dict]]:
    """Greedy two-way balance by the maximum remaining physical horizon."""
    shards = [[], []]
    loads = [0, 0]
    weighted = sorted(
        arms,
        key=lambda arm: (max_steps - int(states[arm["state_id"]]["step"])) * len(arm["seeds"]),
        reverse=True,
    )
    for arm in weighted:
        index = 0 if loads[0] <= loads[1] else 1
        shards[index].append(arm)
        loads[index] += (max_steps - int(states[arm["state_id"]]["step"])) * len(arm["seeds"])
    return shards


def write_v1_reuse(valid_state_ids: set[str]) -> int:
    """Export exact audited v1 tuples for the 67 retained state identities."""
    from diagnostics.gphi_training_dataset_v1.finalize_dataset import load_effective_records

    effective, _ = load_effective_records()
    rows = []
    for (state_id, _, _), row in sorted(effective.items()):
        if state_id not in valid_state_ids or row.get("execution_error") is not None:
            continue
        rows.append({**row, "reuse_source": "gphi_training_dataset_v1"})
    (HERE / "v1_reuse.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline",))
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    state_rows = read_jsonl(HERE / "state_manifest.jsonl")
    states = {row["state_id"]: row for row in state_rows}
    arms = [{
        "arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"],
        "eta": [0.0, 0.0, 0.0], "seeds": seed_set(row["state_id"], protocol["oracle_seed_default"]),
    } for row in state_rows]
    (HERE / "baseline_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    shards = shard_arms(arms, states, int(protocol["environment"]["max_steps"]))
    for index, shard in enumerate(shards):
        (HERE / f"baseline_shard{index}_arms.json").write_text(json.dumps(shard, indent=2) + "\n")
    for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
        ids = {row["state_id"] for row in state_rows if row["category"] == category}
        selected = [arm for arm in arms if arm["state_id"] in ids]
        (HERE / f"baseline_{category.lower()}_arms.json").write_text(json.dumps(selected, indent=2) + "\n")
    reused = write_v1_reuse({row["state_id"] for row in state_rows if row.get("retained_from_v1")})
    print(json.dumps({
        "arms": len(arms), "maximum_new_rollouts": 64 * len(arms),
        "maximum_physical_steps": sum((protocol["environment"]["max_steps"] - states[arm["state_id"]]["step"]) * 64 for arm in arms),
        "shard_arm_counts": [len(shard) for shard in shards],
        "v1_exact_tuples_available": reused,
        "by_category": {category: sum(states[arm["state_id"]]["category"] == category for arm in arms) for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")},
    }, indent=2))


if __name__ == "__main__":
    main()
