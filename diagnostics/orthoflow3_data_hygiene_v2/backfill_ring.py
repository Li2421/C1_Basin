#!/usr/bin/env python3
"""Backfill every v1 Ring (state, eta) under the fixed canonical safety adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from new_benchmark_common.basin_dataset_v1 import TrainingRuntime, sampled_state_rows
from new_benchmark_common.safety_eta3 import STANDARD_SEEDS, execute_batch, uid, canonical
from shared_rollout_db.src.rollout_db import connect


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "datasets/orthoflow3_basin_dataset_v1"
STAGE = "data_hygiene_v2_ring_fixed_safety"


def runtime() -> TrainingRuntime:
    rt = TrainingRuntime("ring_exchange", sampled_state_rows("ring_exchange"), parent=False)
    rt.output = HERE / "work/ring_exchange"
    rt.output.mkdir(parents=True, exist_ok=True)
    rt.experiment_uid = uid("exp", {"path": str(HERE.resolve()), "stage": STAGE})
    with connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
            (rt.experiment_uid, STAGE, str(HERE.resolve()),
             "canonical_15of16_seeds_0_to_15_v1", None,
             canonical({"purpose": "replace stale Ring safety evidence", "training": False})),
        )
        con.commit()
    return rt


def jobs(rt: TrainingRuntime) -> list[dict]:
    labels = pq.read_table(V1 / "eta_labels.parquet").to_pylist()
    by_uid = {row["uid"]: row for row in rt.states}
    result = []
    for row in labels:
        if row["scenario"] != "ring_exchange":
            continue
        result.append({
            "state": by_uid[row["state_uid"]],
            "eta": json.loads(row["eta_raw"]),
            "chain": "orthoflow3",
            "seeds": list(STANDARD_SEEDS),
            "metadata": {
                "audit_source": "orthoflow3_basin_dataset_v1",
                "old_controller_uid": row["controller_uid"],
                "old_eta_label_robust": row["robust_15of16"],
                "dataset_state_id": row["state_id"],
            },
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    rt = runtime()
    result = execute_batch(
        rt, STAGE, jobs(rt), shard_index=args.shard, num_shards=args.shards,
        exact_robust_15of16=True, exact_early_acceptance=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
