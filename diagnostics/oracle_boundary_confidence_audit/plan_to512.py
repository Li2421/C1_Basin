"""Select only statistically unclear states for adaptive extension to 512."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
REFERENCE = 63 / 64


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n; denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def p64(p: float) -> float:
    return p ** 64 + 64 * (1 - p) * p ** 63


def main() -> None:
    audit_states = read_jsonl(HERE / "audited_state_manifest.jsonl")
    paths = [HERE / "existing_eta_zero_rollouts.jsonl", HERE / "raw/to256_shard0/records.jsonl", HERE / "raw/to256_shard1/records.jsonl"]
    effective = {}
    for path in paths:
        for row in read_jsonl(path):
            key = (row["state_id"], int(row["seed"]))
            if key in effective: raise RuntimeError(("duplicate tuple", key))
            effective[key] = row
    by_state = defaultdict(list)
    for (state_id, _), row in effective.items(): by_state[state_id].append(row)
    selection = []; selected = []
    for state in audit_states:
        rows = by_state[state["state_id"]]
        if len(rows) != 256: raise AssertionError((state["state_id"], len(rows)))
        successes = sum(row["outcome"] == "success" for row in rows); estimate = successes / 256
        lower, upper = wilson(successes, 256); probability = p64(estimate)
        # This is an adaptive sampling criterion only, never a new oracle rule.
        unclear = (lower <= REFERENCE <= upper) or (0.05 < probability < 0.95)
        reason = []
        if lower <= REFERENCE <= upper: reason.append("Wilson95 includes 63/64 reference")
        if 0.05 < probability < 0.95: reason.append("random-64 B63 probability between 0.05 and 0.95")
        selection.append({"state_id": state["state_id"], "successes_256": successes, "p_hat_256": estimate,
                          "wilson95_lower": lower, "wilson95_upper": upper, "P64_B63": probability,
                          "extend_to_512": unclear, "reasons": reason})
        if unclear: selected.append(state)
    write_json(HERE / "to512_selection.json", {"criterion": "Wilson95 contains 63/64 OR 0.05 < analytic P64(B63) < 0.95; adaptive sampling only", "states_at_256": len(audit_states), "selected_count": len(selected), "states": selection})
    current_seeds = {key for key in effective}
    common = list(range(96001001, 96002000))
    arms = []
    for state in selected:
        seeds = [seed for seed in common if (state["state_id"], seed) not in current_seeds][:256]
        arms.append({"arm_id": state["state_id"] + "__eta0_to512", "state_id": state["state_id"], "eta": [0.0, 0.0, 0.0], "seeds": seeds})
    state_by_id = {row["state_id"]: row for row in read_jsonl(DATA / "state_manifest.jsonl")}
    loads = [0, 0]; shards = [[], []]
    for arm in sorted(arms, key=lambda row: -(850 - state_by_id[row["state_id"]]["step"]) * len(row["seeds"])):
        shard = 0 if loads[0] <= loads[1] else 1; shards[shard].append(arm); loads[shard] += (850 - state_by_id[arm["state_id"]]["step"]) * len(arm["seeds"])
    write_json(HERE / "to512_arms.json", arms)
    for shard in (0, 1): write_json(HERE / f"to512_shard{shard}_arms.json", shards[shard])
    print(json.dumps({"states_at_256": len(audit_states), "extend_to_512": len(selected), "new_rollouts": 256 * len(selected), "planned_max_steps_by_shard": loads}, indent=2))


if __name__ == "__main__":
    main()
