"""Freeze eta=0-only B_63 arms for all restoration-audited candidates."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    restoration = json.loads((HERE / "restoration_checks.json").read_text())
    if restoration["status"] != "PASS":
        raise RuntimeError("restoration did not pass")
    states = [json.loads(line) for line in (HERE / "candidate_state_manifest.jsonl").read_text().splitlines() if line]
    arms = [{
        "arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"],
        "eta": [0.0, 0.0, 0.0], "seeds": protocol["oracle_seed_default"],
    } for row in states]
    (HERE / "zero_oracle_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    # adaptive_run consumes this canonical name; it contains candidates only at this stage.
    (HERE / "state_manifest.jsonl").write_text((HERE / "candidate_state_manifest.jsonl").read_text())
    print(json.dumps({
        "arms": len(arms), "eta": [0.0, 0.0, 0.0], "seeds_per_viable_arm": 64,
        "maximum_rollouts": 64 * len(arms), "source_groups": len({row["source_group"] for row in states}),
    }, indent=2))


if __name__ == "__main__":
    main()
