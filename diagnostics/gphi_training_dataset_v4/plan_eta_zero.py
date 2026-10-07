"""Freeze the eta-zero-first B_63 evaluation for all V4 candidates."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    restoration = json.loads((HERE / "restoration_checks.json").read_text())
    if restoration["status"] != "PASS":
        raise RuntimeError("restoration failed")
    states = [json.loads(line) for line in (HERE / "candidate_state_manifest.jsonl").read_text().splitlines() if line]
    arms = [{"arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"], "eta": [0.0, 0.0, 0.0], "seeds": protocol["oracle_seed_default"]} for row in states]
    (HERE / "eta_zero_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    # Deterministic two-way state partition: no (state, eta, seed) overlap.
    for shard in (0, 1):
        selected = [arm for index, arm in enumerate(arms) if index % 2 == shard]
        (HERE / f"eta_zero_shard{shard}_arms.json").write_text(json.dumps(selected, indent=2) + "\n")
    (HERE / "state_manifest.jsonl").write_text((HERE / "candidate_state_manifest.jsonl").read_text())
    print(json.dumps({"arms": len(arms), "shard_arm_counts": [len(arms[::2]), len(arms[1::2])], "maximum_rollouts": 64 * len(arms), "source_groups": len({row["source_group"] for row in states})}, indent=2))


if __name__ == "__main__":
    main()
