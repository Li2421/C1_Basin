"""Select matched eta-zero/nonzero boundary pairs and plan nonzero oracle arms."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
V3_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.gphi_training_dataset_v2.finalize_dataset import FeatureBuilder
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config


TARGET_PAIRS = {"train": 9, "validation": 2, "test": 4}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    candidates = read_jsonl(HERE / "candidate_state_manifest.jsonl")
    by_id = {row["state_id"]: row for row in candidates}
    records = []
    manifests = []
    for path in sorted((HERE / "raw").glob("eta_zero*/records.jsonl")):
        records.extend(read_jsonl(path))
        manifests.append(json.loads((path.parent / "manifest.json").read_text()))
    grouped = defaultdict(list)
    for row in records:
        grouped[row["state_id"]].append(row)
    labels = {}
    results = []
    for state in candidates:
        rows = grouped[state["state_id"]]
        counts = Counter(row["outcome"] for row in rows)
        zero = len(rows) == 64 and counts["success"] >= 63 and not any(row.get("execution_error") for row in rows)
        labels[state["state_id"]] = "zero" if zero else "nonzero"
        results.append({"state_id": state["state_id"], "source_group": state["source_group"], "split": state["provisional_split"], "evaluated": len(rows), "success": counts["success"], "failures": len(rows) - counts["success"], "eta_zero_B_63": zero})

    builder = FeatureBuilder()
    reps = {}
    envs = {}
    for state in candidates:
        env = restore_full(HERE / state["state_file"], config)
        with np.load(HERE / state["audit_reference_file"]) as ref:
            feature, _ = builder.build(env, {"u_flow": ref["u_flow"], "u_safe": ref["u_safe"]}, config, cbf)
        reps[state["state_id"]] = feature
        envs[state["state_id"]] = env
    normalization = json.loads((V3_TRAIN / "normalization.json").read_text())
    mean = np.asarray(normalization["mean"]); scale = np.asarray(normalization["scale"])
    norm = {state_id: (feature - mean) / scale for state_id, feature in reps.items()}

    def distance(a: str, b: str) -> float:
        return float(np.linalg.norm(norm[a] - norm[b]) / math.sqrt(len(mean)))

    chosen_pairs = []
    used = set()
    # First secure within-root examples on both oracle sides.
    for split in ("train", "validation", "test"):
        groups = sorted({row["source_group"] for row in candidates if row["provisional_split"] == split})
        per_group_target = 2 if split in ("validation", "test") else 1
        for group in groups:
            zeros = [row["state_id"] for row in candidates if row["source_group"] == group and labels[row["state_id"]] == "zero"]
            nonzeros = [row["state_id"] for row in candidates if row["source_group"] == group and labels[row["state_id"]] == "nonzero"]
            options = sorted(((distance(z, n), z, n) for z in zeros for n in nonzeros), key=lambda item: (item[0], item[1], item[2]))
            for _, zero_id, nonzero_id in options:
                if len([pair for pair in chosen_pairs if pair["split"] == split and pair["source_group_zero"] == group and pair["source_group_nonzero"] == group]) >= per_group_target:
                    break
                if zero_id in used or nonzero_id in used:
                    continue
                chosen_pairs.append({"split": split, "zero_state_id": zero_id, "nonzero_state_id": nonzero_id, "source_group_zero": group, "source_group_nonzero": group, "same_source_group": True, "feature_distance": distance(zero_id, nonzero_id)})
                used.update((zero_id, nonzero_id))
        # Fill the split target with the closest unused cross-root pair if a
        # root lacks both labels; source trajectories remain distinct.
        while sum(pair["split"] == split for pair in chosen_pairs) < TARGET_PAIRS[split]:
            zeros = [row["state_id"] for row in candidates if row["provisional_split"] == split and labels[row["state_id"]] == "zero" and row["state_id"] not in used]
            nonzeros = [row["state_id"] for row in candidates if row["provisional_split"] == split and labels[row["state_id"]] == "nonzero" and row["state_id"] not in used]
            if not zeros or not nonzeros:
                break
            _, zero_id, nonzero_id = min((distance(z, n), z, n) for z in zeros for n in nonzeros)
            chosen_pairs.append({"split": split, "zero_state_id": zero_id, "nonzero_state_id": nonzero_id, "source_group_zero": by_id[zero_id]["source_group"], "source_group_nonzero": by_id[nonzero_id]["source_group"], "same_source_group": by_id[zero_id]["source_group"] == by_id[nonzero_id]["source_group"], "feature_distance": distance(zero_id, nonzero_id)})
            used.update((zero_id, nonzero_id))

    if len(chosen_pairs) < 10:
        raise RuntimeError(("fewer than ten matched boundary pairs", len(chosen_pairs)))
    selected_ids = {state_id for pair in chosen_pairs for state_id in (pair["zero_state_id"], pair["nonzero_state_id"])}
    selected = []
    for state in candidates:
        if state["state_id"] not in selected_ids:
            continue
        row = dict(state)
        row["split"] = row.pop("provisional_split")
        row["oracle_boundary_class"] = labels[row["state_id"]]
        selected.append(row)
    source_groups = {row["source_group"] for row in selected}
    if not 20 <= len(selected) <= 40 or len(source_groups) < 6:
        raise RuntimeError(("selection target failed", len(selected), len(source_groups)))

    pair_rows = []
    for index, pair in enumerate(chosen_pairs):
        zero = by_id[pair["zero_state_id"]]; nonzero = by_id[pair["nonzero_state_id"]]
        zero_env = envs[zero["state_id"]]; nonzero_env = envs[nonzero["state_id"]]
        zero_history = np.asarray(zero_env.distance_history[-41:])
        nonzero_history = np.asarray(nonzero_env.distance_history[-41:])
        pair_rows.append({
            "pair_id": f"boundary_pair_{index:02d}", **pair,
            "zero_inter_agent_distance": zero["inter_agent_distance"],
            "nonzero_inter_agent_distance": nonzero["inter_agent_distance"],
            "absolute_inter_agent_distance_difference": abs(zero["inter_agent_distance"] - nonzero["inter_agent_distance"]),
            "zero_relative_velocity_norm": zero["relative_velocity_norm"],
            "nonzero_relative_velocity_norm": nonzero["relative_velocity_norm"],
            "relative_velocity_norm_difference": abs(zero["relative_velocity_norm"] - nonzero["relative_velocity_norm"]),
            "history_RMS_difference_m": float(np.linalg.norm(zero_history - nonzero_history) / math.sqrt(zero_history.size)),
            "stuck_timer_difference_s": abs(zero["stuck_timer"] - nonzero["stuck_timer"]),
            "candidate_active_zero": zero["candidate_since"] >= 0,
            "candidate_active_nonzero": nonzero["candidate_since"] >= 0,
            "zero_oracle_class": "zero", "nonzero_oracle_class": "nonzero",
        })
    write_jsonl(HERE / "selected_state_manifest.jsonl", selected)
    write_jsonl(HERE / "state_manifest.jsonl", selected)
    write_csv(HERE / "matched_boundary_pairs_preoracle.csv", pair_rows)
    (HERE / "eta_zero_classification.json").write_text(json.dumps({
        "candidate_results": results,
        "candidate_class_counts": dict(Counter(labels.values())),
        "selected_state_count": len(selected),
        "selected_class_counts": dict(Counter(row["oracle_boundary_class"] for row in selected)),
        "selected_source_groups": sorted(source_groups),
        "pair_count": len(pair_rows),
        "eta_zero_runtime": {"new_rollouts": sum(item["new_rollouts"] for item in manifests), "physical_steps": sum(item["physical_steps"] for item in manifests), "elapsed_s_max_parallel": max(item["elapsed_s"] for item in manifests)},
    }, indent=2) + "\n")

    # The eight frozen validated eta regions from V3 are the complete targeted
    # library. Eta zero was already rejected for this selected class.
    arms = []
    for state in selected:
        if state["oracle_boundary_class"] != "nonzero":
            continue
        for eta_index, eta in enumerate(protocol["candidate_etas"]):
            arms.append({"arm_id": f"{state['state_id']}__candidate{eta_index:02d}", "state_id": state["state_id"], "eta": eta, "seeds": protocol["oracle_seed_default"]})
    states_by_id = {row["state_id"]: row for row in selected}
    loads = [0, 0]; shards = [[], []]
    for arm in sorted(arms, key=lambda item: (config.max_steps - states_by_id[item["state_id"]]["step"]) * 64, reverse=True):
        shard = 0 if loads[0] <= loads[1] else 1
        shards[shard].append(arm)
        loads[shard] += (config.max_steps - states_by_id[arm["state_id"]]["step"]) * 64
    (HERE / "nonzero_candidate_arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    for shard, values in enumerate(shards):
        (HERE / f"nonzero_candidate_shard{shard}_arms.json").write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps({
        "candidate_labels": dict(Counter(labels.values())),
        "selected_states": len(selected), "selected_labels": dict(Counter(row["oracle_boundary_class"] for row in selected)),
        "selected_source_groups": len(source_groups), "matched_pairs": len(chosen_pairs),
        "split_pairs": dict(Counter(pair["split"] for pair in chosen_pairs)),
        "nonzero_oracle_arms": len(arms), "shard_arms": [len(value) for value in shards],
    }, indent=2))


if __name__ == "__main__":
    main()
