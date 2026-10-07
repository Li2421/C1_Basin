"""Read-only source-family, eta and label-semantics inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, dump_json


def inspect(version: str):
    directory = ROOT / f"datasets/orthoflow3_basin_dataset_{version}"
    state_cols = ["scenario", "split", "state_uid", "source_initial_state_id", "parent_episode_id", "timestep", "controller_uid"]
    states = pq.read_table(directory / "states.parquet", columns=state_cols).to_pylist()
    schema = pq.read_schema(directory / "eta_labels.parquet")
    label_cols = ["scenario", "split", "state_uid", "eta_uid", "robust_15of16", "seed_count", "success_count", "numerical_failure_count", "controller_uid"]
    if "label_semantics_version" in schema.names:
        label_cols.append("label_semantics_version")
    labels = pq.read_table(directory / "eta_labels.parquet", columns=label_cols).to_pylist()
    out = {}
    for sc in sorted(set(s["scenario"] for s in states)):
        ss = [s for s in states if s["scenario"] == sc]
        rr = [r for r in labels if r["scenario"] == sc]
        bysplit = {}
        for split in sorted(set(s["split"] for s in ss)):
            sids = {s["state_uid"] for s in ss if s["split"] == split}
            groups = {s["source_initial_state_id"] for s in ss if s["split"] == split}
            srr = [r for r in rr if r["state_uid"] in sids]
            per_state = Counter(r["state_uid"] for r in srr)
            bysplit[split] = {"states": len(sids), "source_groups": len(groups), "pairs": len(srr),
                              "eta_unique": len({r["eta_uid"] for r in srr}),
                              "robust_true": sum(r["robust_15of16"] is True for r in srr),
                              "robust_false": sum(r["robust_15of16"] is False for r in srr),
                              "robust_unknown": sum(r["robust_15of16"] is None for r in srr),
                              "full_Q16_or_more": sum(r["seed_count"] >= 16 and r["numerical_failure_count"] == 0 for r in srr),
                              "eta_per_state_min_median_max": [min(per_state.values()), float(np.median(list(per_state.values()))), max(per_state.values())],
                              "controller_uids": sorted({r["controller_uid"] for r in srr})}
        groups_by_split = {k: {s["source_initial_state_id"] for s in ss if s["split"] == k} for k in bysplit}
        splits = list(groups_by_split)
        leakage = sum(len(groups_by_split[splits[i]] & groups_by_split[splits[j]])
                      for i in range(len(splits)) for j in range(i + 1, len(splits)))
        item = {"splits": bysplit, "source_group_overlap": leakage,
                "state_timestep_counts": dict(Counter(s["timestep"] for s in ss)),
                "label_semantics_versions": dict(Counter(r.get("label_semantics_version", "not_recorded_v1") for r in rr))}
        out[sc] = item
    return out


def main():
    paths = {
        "generator": ROOT / "diagnostics/orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack",
        "critic": ROOT / "diagnostics/orthoflow3_generator_critic_v1/critic/seed23/checkpoint.msgpack",
        "normalization": ROOT / "diagnostics/orthoflow3_generator_critic_v1/normalization.json",
        "v1_train_script": ROOT / "diagnostics/orthoflow3_generator_critic_v1/train_evaluate.py",
    }
    ckpt = {k: {"path": str(v), "exists": v.exists(), "sha256": hashlib.sha256(v.read_bytes()).hexdigest() if v.exists() else None}
            for k, v in paths.items()}
    report = {"v1_used_by_frozen_models": inspect("v1"),
              "v2_audited_current_safety": inspect("v2_audited"),
              "frozen_artifacts": ckpt,
              "critical_target_semantics": "Original v1 critic constructs q=success_count/seed_count even for early-stopped labels. This is not an unbiased empirical Q16 target; corrected v2 full-Q16 evidence must be kept distinct from logical B15 early-stop labels.",
              "new_rollout": 0}
    dump_json("lineage_inventory.json", report)
    print(json.dumps({"v1": report["v1_used_by_frozen_models"], "v2": report["v2_audited_current_safety"]}, indent=2))


if __name__ == "__main__":
    main()
