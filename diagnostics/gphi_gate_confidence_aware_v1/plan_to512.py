"""Apply the prior audit's adaptive 512 rule verbatim to the full V4 pool."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
REFERENCE = 63/64


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return center-half, center+half


def p64(p: float) -> float:
    return p**64 + 64*(1-p)*p**63


def main() -> None:
    states = read_jsonl(HERE/"confidence_state_manifest.jsonl")
    state_by_id = {row["state_id"]: row for row in states}
    paths = [HERE/"existing_eta_zero_rollouts.jsonl"] + [HERE/f"raw/to256_shard{shard}/records.jsonl" for shard in range(4)]
    effective = {}
    for path in paths:
        for row in read_jsonl(path):
            key = (row["state_id"], int(row["seed"]))
            if key in effective:
                old = effective[key]
                if (old["outcome"], int(old["steps"])) != (row["outcome"], int(row["steps"])):
                    raise RuntimeError(("conflicting tuple", key))
                continue
            effective[key] = row
    by_state = defaultdict(list)
    for (state_id, _), row in effective.items(): by_state[state_id].append(row)
    selected = []; decisions = []
    for state in states:
        rows = by_state[state["state_id"]]; n = len(rows)
        if n not in (256, 512): raise AssertionError((state["state_id"], n))
        successes = sum(row["outcome"] == "success" for row in rows); estimate = successes/n
        lower, upper = wilson(successes, n); probability = p64(estimate)
        unclear = n == 256 and ((lower <= REFERENCE <= upper) or (0.05 < probability < 0.95))
        reasons = []
        if lower <= REFERENCE <= upper: reasons.append("Wilson95 includes 63/64 reference")
        if 0.05 < probability < 0.95: reasons.append("random-64 B63 probability between 0.05 and 0.95")
        decisions.append({"state_id": state["state_id"], "n": n, "successes": successes, "p_hat": estimate,
                          "wilson95_lower": lower, "wilson95_upper": upper, "P64_B63": probability,
                          "extend_to_512": unclear, "already_512_from_prior_audit": n == 512, "reasons": reasons})
        if unclear: selected.append(state)
    used = set(effective); common = list(range(96102001, 96103000)); arms = []
    for state in selected:
        seeds = [seed for seed in common if (state["state_id"], seed) not in used][:256]
        if len(seeds) != 256: raise AssertionError(state["state_id"])
        arms.append({"arm_id": state["state_id"]+"__eta0_to512", "state_id": state["state_id"], "eta": [0.,0.,0.], "seeds": seeds})
    loads = [0,0,0,0]; shards = [[],[],[],[]]
    for arm in sorted(arms, key=lambda row: -(850-int(state_by_id[row["state_id"]]["step"]))*len(row["seeds"])):
        shard = min(range(4), key=lambda index: loads[index])
        shards[shard].append(arm); loads[shard] += (850-int(state_by_id[arm["state_id"]]["step"]))*len(arm["seeds"])
    write_json(HERE/"to512_selection.json", {
        "criterion": "Wilson95 contains 63/64 OR 0.05 < analytic P64(B63) < 0.95; adaptive sampling only; copied from prior audit",
        "states": len(states), "already_512": sum(row["already_512_from_prior_audit"] for row in decisions),
        "selected_count": len(selected), "new_rollouts": 256*len(selected), "planned_max_physical_steps_by_shard": loads,
        "state_decisions": decisions,
    })
    write_json(HERE/"to512_arms.json", arms)
    for shard in range(4): write_json(HERE/f"to512_shard{shard}_arms.json", shards[shard])
    print(json.dumps({"states": len(states), "already_512": sum(row["already_512_from_prior_audit"] for row in decisions),
                      "extend_to_512": len(selected), "new_rollouts": 256*len(selected), "loads": loads}, indent=2))


if __name__ == "__main__":
    main()
