"""Plan targeted nonzero eta arms after the baseline gate, with cache reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

from diagnostics.gphi_training_dataset_v2.plan_oracle import read_jsonl, seed_set, shard_arms
from diagnostics.success_basin_deformation_decomposition.analyze import eta_key, load_inventory


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    states = {row["state_id"]: row for row in read_jsonl(HERE / "state_manifest.jsonl")}
    baseline = []
    for path in sorted((HERE / "raw").glob("baseline_*/manifest.json")):
        baseline.extend(json.loads(path.read_text())["arm_results"])
    failed_states = sorted(row["state_id"] for row in baseline if not row["B_63_member"])
    candidates = [eta_key(eta) for eta in protocol["candidate_etas"]]
    arms = []
    for state_id in failed_states:
        seeds = seed_set(state_id, protocol["oracle_seed_default"])
        for index, eta in enumerate(candidates):
            arms.append({
                "arm_id": f"{state_id}__eta_{index}", "state_id": state_id,
                "eta": list(eta), "seeds": seeds, "candidate_priority": index,
            })
    (HERE / "candidate_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    shards = shard_arms(arms, states, int(protocol["environment"]["max_steps"]))
    for index, shard in enumerate(shards):
        (HERE / f"candidate_shard{index}_arms.json").write_text(json.dumps(shard, indent=2) + "\n")
    for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
        selected = [arm for arm in arms if states[arm["state_id"]]["category"] == category]
        (HERE / f"candidate_{category.lower()}_arms.json").write_text(json.dumps(selected, indent=2) + "\n")

    # Reuse only exact state/eta/seed tuples from prior audited diagnostics.
    _, effective, _, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    arm_lookup = {(arm["state_id"], eta_key(arm["eta"])): arm for arm in arms}
    reused_by_tuple = {}
    # First reuse the complete audited v1 effective tuple inventory.  State
    # identity, RNG namespace, and oracle seed sets were retained exactly.
    for row in read_jsonl(HERE / "v1_reuse.jsonl"):
        key = (row["state_id"], eta_key(row["eta"]))
        arm = arm_lookup.get(key)
        if arm is None or int(row["seed"]) not in arm["seeds"] or row.get("execution_error") is not None:
            continue
        reused_by_tuple[(row["state_id"], key[1], int(row["seed"]))] = row
    for row in effective:
        key = (row["state_id"], eta_key(row["eta"]))
        arm = arm_lookup.get(key)
        if arm is None or int(row["seed"]) not in arm["seeds"] or row["execution_error"] is not None:
            continue
        with np.load(row["path"]) as data:
            first = {
                "u_flow": np.asarray(data["u_flow"][0]).reshape(-1).tolist(),
                "u_safe": np.asarray(data["u_safe"][0]).reshape(-1).tolist(),
                "g_raw": np.asarray(data["g"][0]).reshape(-1).tolist(),
                "u_exec": np.asarray(data["u_exec"][0]).reshape(-1).tolist(),
                "first_projection_status": "reused_audited_cache",
                "second_projection_status": "reused_audited_cache",
                "first_retry": False, "second_retry": False,
                "first_min_linear_residual": None, "second_min_linear_residual": None,
            }
        record = {
            "state_id": row["state_id"], "eta": list(key[1]), "seed": int(row["seed"]),
            "outcome": row["outcome"], "execution_error": None, "steps": int(row["steps"]),
            "terminal_step": int(states[row["state_id"]]["step"] + row["steps"]),
            "J_def": float(row["stored_J_def"]), "first_step": first,
            "source": row["source"], "source_file": str(row["path"]), "source_sha256": row["sha256"],
        }
        reused_by_tuple.setdefault((row["state_id"], key[1], int(row["seed"])), record)
    reused = [reused_by_tuple[key] for key in sorted(reused_by_tuple)]
    reuse_path = HERE / "candidate_reuse.jsonl"
    reuse_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reused))
    print(json.dumps({
        "failed_zero_states": len(failed_states), "candidate_arms": len(arms),
        "maximum_rollouts_without_reuse_or_stopping": len(arms) * 64,
        "reused_exact_tuples": len(reused), "shard_arm_counts": [len(shard) for shard in shards],
        "by_category": {category: sum(states[arm["state_id"]]["category"] == category for arm in arms) for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")},
    }, indent=2))


if __name__ == "__main__":
    main()
