"""Diagnostic: do corrected-safety interaction models resolve TRAIN-chosen preference reversals?

The source DEV states were used for checkpoint selection, so this is a model
diagnostic, not independent confirmation. Pair identities came from TRAIN only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, ring_matrices
from train_interaction import InteractionCritic, SPLIT, parse2, prepare
from train_eta_only_mlp import EtaOnly


def main() -> None:
    pairs = list(csv.DictReader((OUT / "ring_exchange_reversal_pairs.csv").open()))
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "state_uid", "eta_uid", "eta_raw",
                                    "robust_15of16", "label_semantics_version"]).to_pylist()
    ring = [r for r in labels if r["scenario"] == "ring_exchange"]
    assert all(r["label_semantics_version"] == "ring_current_safety_v2" for r in ring)
    train, val = ring_matrices(labels)
    eta_idx = {uid: i for i, uid in enumerate(train["eta_ids"])}
    state_idx = {uid: i for i, uid in enumerate(val["states"])}
    state_rows = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                               columns=["scenario", "split", "state_uid", "conditioning"]).to_pylist()
    by_state = {r["state_uid"]: r for r in state_rows
                if r["scenario"] == "ring_exchange" and r["state_uid"] in state_idx}
    data = prepare()
    mean, std = data["ring_exchange"]["h_mean"], data["ring_exchange"]["h_std"]
    center = np.asarray(SPLIT["eta_coord_center"], np.float32)
    radius = np.asarray(SPLIT["eta_coord_radius"], np.float32)
    model = InteractionCritic()
    dims = [len(data[s]["h_mean"]) for s in ("four_way_intersection", "ring_exchange")]
    init = model.init(jax.random.PRNGKey(0), jnp.zeros((1, dims[0])), jnp.zeros((1, 3)),
                      jnp.zeros((1, dims[1])), jnp.zeros((1, 3)))
    eta_model = EtaOnly()
    eta_init = eta_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    rows = []
    models = {}
    for seed in (17, 23, 41):
        for run in ("ring_full", "joint_full", "ring_25_scratch", "ring_25_transfer_fixed", "ring_25_finetune"):
            path = OUT / "interaction_models" / f"seed{seed}" / run / "checkpoint.msgpack"
            if path.exists():
                models[f"{run}_seed{seed}"] = (model, serialization.from_bytes(init, path.read_bytes()), "ring")
        path = OUT / "eta_only_mlp" / f"seed{seed}" / "ring_exchange" / "checkpoint.msgpack"
        if path.exists():
            models[f"eta_only_mlp_seed{seed}"] = (eta_model, serialization.from_bytes(eta_init, path.read_bytes()), "eta")
    for pair_no, pair in enumerate(pairs):
        ia, ib = eta_idx[pair["eta_a_uid"]], eta_idx[pair["eta_b_uid"]]
        eta = (np.asarray([train["x"][ia], train["x"][ib]], np.float32) - center) / radius
        epred = {}
        for name, (mod, params, kind) in models.items():
            if kind == "eta":
                epred[name] = np.asarray(mod.apply(params, jnp.asarray(eta)))
        for state_uid, si in state_idx.items():
            ya, yb = val["y"][si, ia], val["y"][si, ib]
            if not (np.isfinite(ya) and np.isfinite(yb)) or ya == yb:
                continue
            raw_h = np.asarray(parse2(by_state[state_uid]["conditioning"])["flat"], np.float32)
            h = np.repeat(((raw_h - mean) / std)[None], 2, axis=0)
            row = {"pair_index": pair_no, "state_uid": state_uid,
                   "true_better": int(np.argmax([ya, yb]))}
            for name, (mod, params, kind) in models.items():
                scores = epred[name] if kind == "eta" else np.asarray(
                    mod.apply(params, jnp.asarray(h), jnp.asarray(eta), method=mod.ring))
                row[name] = int(np.argmax(scores)) == row["true_better"]
            rows.append(row)
    output = OUT / "ring_reversal_corrected_model_comparison.csv"
    with output.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    totals = Counter({name: sum(int(r[name]) for r in rows) for name in models})
    summary = {"n_decisive_comparisons": len(rows), "train_selected_eta_pairs": len(pairs),
               "corrected_safety_version": "ring_current_safety_v2", "model_correct": dict(totals),
               "validation_used_for_checkpoint_selection": True, "new_rollout": 0}
    (OUT / "ring_reversal_corrected_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
