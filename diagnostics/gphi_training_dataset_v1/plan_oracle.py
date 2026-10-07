"""Freeze baseline and targeted eta-arm plans; maximize compatible cache reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def seed_set(state_id: str, default: list[int]) -> list[int]:
    if state_id == "D1_pair231":
        return list(range(95106001, 95106033)) + list(range(95107001, 95107033))
    if state_id in ("D2_pair228", "D4_pair227"):
        return list(range(95101001, 95101017)) + list(range(95105001, 95105017)) + list(range(95108001, 95108033))
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline",))
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = read_jsonl(HERE / "state_manifest.jsonl")
    arms = [{
        "arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"],
        "eta": [0.0, 0.0, 0.0], "seeds": seed_set(row["state_id"], protocol["oracle_seed_default"]),
    } for row in states]
    (HERE / "baseline_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
        ids = {row["state_id"] for row in states if row["category"] == category}
        selected = [arm for arm in arms if arm["state_id"] in ids]
        (HERE / f"baseline_{category.lower()}_arms.json").write_text(json.dumps(selected, indent=2) + "\n")
    catalog = {row["state_id"]: row for row in states}
    print(json.dumps({
        "arms": len(arms), "maximum_new_rollouts": 64 * len(arms),
        "maximum_physical_steps": sum((protocol["environment"]["max_steps"] - catalog[arm["state_id"]]["step"]) * 64 for arm in arms),
        "by_category": {category: sum(catalog[arm["state_id"]]["category"] == category for arm in arms) for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")},
    }, indent=2))


if __name__ == "__main__":
    main()
