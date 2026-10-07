"""Frozen v1 critic versus eta-only prior on TRAIN-selected Ring reversal pairs."""

from __future__ import annotations

import csv
import json

import flax.serialization as serialization
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, dump_json, ring_matrices
from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as base


def decode_twice(value):
    out = json.loads(value)
    return json.loads(out) if isinstance(out, str) else out


def main():
    pair_rows = list(csv.DictReader((OUT / "ring_exchange_reversal_pairs.csv").open()))
    assert len(pair_rows) == 10
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "split", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version"]).to_pylist()
    ring = [r for r in labels if r["scenario"] == "ring_exchange"]
    assert all(r["label_semantics_version"] == "ring_current_safety_v2" for r in ring)
    train, val = ring_matrices(labels)
    eidx = {e: i for i, e in enumerate(train["eta_ids"])}
    norm = json.loads((ROOT / "diagnostics/orthoflow3_generator_critic_v1/normalization.json").read_text())
    n = norm["scenarios"]["ring_exchange"]
    rows = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                         columns=["scenario", "split", "state_uid", "conditioning", "environment_descriptor"]).to_pylist()
    by = {r["state_uid"]: r for r in rows if r["scenario"] == "ring_exchange" and r["split"] in ("val", "validation")}
    assert set(by) == set(val["states"])
    model = base.Critic()
    dims = {s: len(norm["scenarios"][s]["h_mean"]) for s in base.SCENARIOS}
    cdim = len(norm["environment_keys"]) + 3
    checkpoint = ROOT / "diagnostics/orthoflow3_generator_critic_v1/critic/seed23/checkpoint.msgpack"
    params = serialization.from_bytes(base.merge_initialized(model, dims, cdim, critic=True), checkpoint.read_bytes())
    eta_scores = np.load(OUT / "eta_only_kernel_model.npz")
    eta_prior = float(eta_scores["prior"])
    eta_bw = float(eta_scores["bandwidth"])
    all_eta = train["z"]
    from scipy.spatial.distance import cdist
    k = np.exp(-cdist(all_eta, all_eta, "sqeuclidean") / (2 * eta_bw**2))
    eta_prob = np.clip(eta_prior + k @ eta_scores["alpha"], 0, 1)
    out = []
    for pairno, pair in enumerate(pair_rows):
        ia, ib = eidx[pair["eta_a_uid"]], eidx[pair["eta_b_uid"]]
        eta = np.asarray([train["x"][ia], train["x"][ib]], np.float32)
        eta_norm = (eta - base.CENTER) / base.RADIUS
        for state_uid in val["states"]:
            si = val["states"].index(state_uid)
            ya, yb = val["y"][si, ia], val["y"][si, ib]
            if not (np.isfinite(ya) and np.isfinite(yb)) or ya == yb:
                continue
            s = by[state_uid]
            h = np.asarray(decode_twice(s["conditioning"])["flat"], np.float32)
            h = (h - np.asarray(n["h_mean"], np.float32)) / np.asarray(n["h_std"], np.float32)
            fields = base.scalar_environment_fields(decode_twice(s["environment_descriptor"]))
            c = np.zeros(cdim, np.float32)
            for j, key in enumerate(norm["environment_keys"]):
                c[j] = fields.get(key, 0.0)
            c[len(norm["environment_keys"]) + base.SCENARIOS.index("ring_exchange")] = 1.0
            c = (c - np.asarray(n["c_mean"], np.float32)) / np.asarray(n["c_std"], np.float32)
            logits = np.asarray(model.apply(params, jnp.asarray(np.repeat(h[None], 2, axis=0)),
                                            jnp.asarray(np.repeat(c[None], 2, axis=0)),
                                            jnp.asarray(eta_norm), method=model.ring))
            truth = int(np.argmax([ya, yb]))
            out.append({"pair_index": pairno, "state_uid": state_uid, "truth_better": truth,
                        "frozen_critic_better": int(np.argmax(logits)),
                        "frozen_critic_correct": int(np.argmax(logits)) == truth,
                        "eta_only_better": int(np.argmax([eta_prob[ia], eta_prob[ib]])),
                        "eta_only_correct": int(np.argmax([eta_prob[ia], eta_prob[ib]])) == truth,
                        "critic_logit_margin_a_minus_b": float(logits[0] - logits[1]),
                        "eta_only_margin_a_minus_b": float(eta_prob[ia] - eta_prob[ib])})
    with (OUT / "ring_reversal_model_predictions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    summary = {"scenario": "ring_exchange", "frozen_model": str(checkpoint),
               "safety_labels": "ring_current_safety_v2", "pair_selection": "TRAIN only",
               "validation_source_family_states": len(val["states"]), "train_selected_pairs": 10,
               "decisive_state_pair_comparisons": len(out),
               "frozen_critic_correct": sum(r["frozen_critic_correct"] for r in out),
               "eta_only_correct": sum(r["eta_only_correct"] for r in out),
               "status": "diagnostic validation; historical v1 critic saw prior-safety Ring validation during checkpoint selection",
               "new_rollout": 0}
    dump_json("ring_reversal_model_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
