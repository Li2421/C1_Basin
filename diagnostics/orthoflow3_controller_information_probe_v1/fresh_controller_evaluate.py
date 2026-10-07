"""Frozen v8-controller selection after source-only model training."""
from __future__ import annotations

import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from diagnostics.orthoflow3_controller_intervention_generalization_v1.intervention import OLD
from shared_rollout_db.src.rollout_db import canonical, connect

from .fresh_controller import DEST
from .probe import OUT, Critic, load_scene, read, write


def outcomes(data):
    pairs = read(DEST / "pairs.json")
    by_physical = {(r["state_uid"], tuple(np.asarray(r["eta"], np.float32))): r for r in pairs}
    val_ix = np.flatnonzero(data["split"] == "validation")
    rows = []
    with connect(True) as db:
        for i in val_ix:
            uid = data["rows"][i]["state_uid"]
            eta = tuple(np.asarray(data["d"]["eta"][data["ids"][i]], np.float32))
            pair = by_physical[(uid, eta)]
            rec = db.execute("""SELECT seed_key,success,numerical_failure,conflict_quarantined,
                compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                (uid, pair["eta_uid"], pair["alternate_controller_uid"])).fetchall()
            valid = [r for r in rec if r["seed_key"] in
                     {canonical({"future_index": k}) for k in range(16)}
                     and not r["numerical_failure"] and not r["conflict_quarantined"]
                     and r["compatibility_quality"] == "EXACT_REUSE"]
            rows.append({"row_index": int(i), "state_uid": uid, "eta_uid": pair["eta_uid"],
                         "observed_trials": len(valid),
                         "success": sum(int(r["success"]) for r in valid),
                         "failure": sum(1-int(r["success"]) for r in valid)})
    assert len(rows) == 16
    return rows


def local_h20_context(data, val_ix):
    pairs = read(DEST / "pairs.json")
    mapping = {(r["state_uid"], tuple(np.asarray(r["eta"],np.float32))):r for r in pairs}
    states = read(OLD / "states.json")
    runtime = h20.rc.RichRuntime("ring_exchange", read(DEST / "protocol.json")["profiles"]["ring_exchange"]["alternate_path"])
    vectors = []
    for i in val_ix:
        uid = data["rows"][i]["state_uid"]
        eta = tuple(np.asarray(data["d"]["eta"][data["ids"][i]],np.float32))
        pair = mapping[(uid,eta)]
        response = h20.rc.cached(runtime,{"state_uid":uid,"physical":states[pair["state_index"]]["physical"]},pair["eta"])
        assert response["valid"],response.get("error")
        vectors.append(response["features"]["mean"])
    return np.asarray(vectors,np.float32)


def score(path, kind, seed, data, val_ix, vector, h20_context):
    use_state = kind in ("fingerprint_full","physical_context")
    use_context = kind != "eta_only"
    model = Critic(use_state, use_context, False)
    state_index = data["state_index"]
    x = data["x"]
    template = model.init(jax.random.PRNGKey(seed), gather(x, state_index[val_ix[:1]]),
                          jnp.zeros((1, 3)), jnp.zeros((1, 24)), jnp.zeros((1, 3)))
    params = serialization.from_bytes(template, (path / "checkpoint.msgpack").read_bytes())
    norm = read(path / "normalization.json")
    center = np.asarray(norm["context_center"], np.float32)
    scale = np.asarray(norm["context_scale"], np.float32)
    raw = h20_context if kind == "physical_context" else np.broadcast_to(vector,(len(val_ix),24))
    c = jnp.asarray((raw-center)/scale)
    identity = jnp.zeros((len(val_ix), 3))
    z = np.asarray(model.apply(params, gather(x, state_index[val_ix]),
                               jnp.asarray(data["eta"][val_ix]), c, identity))
    return 1/(1+np.exp(-z))


def run():
    data = load_scene("ring_exchange")
    y = outcomes(data)
    val_ix = np.flatnonzero(data["split"] == "validation")
    assert [r["row_index"] for r in y] == val_ix.tolist()
    vector = np.asarray(read(DEST / "v8_fingerprint.json")["vector"], np.float32)
    h20_context = local_h20_context(data,val_ix)
    by_state = {}
    for j, r in enumerate(y):
        by_state.setdefault(r["state_uid"], []).append(j)
    oracle_B15 = int(sum(any(y[j]["success"] >= 15 for j in ix) for ix in by_state.values()))
    results = []
    specs = [("eta_only", "eta_only", True),
             ("fingerprint_eta", "fingerprint_eta", True),
             ("fingerprint_full", "fingerprint_full", True),
             ("physical_context", "physical_context", True),
             ("fingerprint_eta", "fingerprint_eta_heldout_controller2_crossmatrix", False),
             ("fingerprint_full", "fingerprint_full_heldout_controller2_crossmatrix", False),
             ("fingerprint_eta", "fingerprint_eta_crossmatrix", True),
             ("fingerprint_full", "fingerprint_full_crossmatrix", True)]
    for kind, label, trained_on_third in specs:
        for seed in (17, 23, 41):
            path = OUT / "models/ring_exchange" / label / f"seed{seed}"
            p = score(path, kind, seed, data, val_ix, vector, h20_context)
            s = np.asarray([r["success"] for r in y], float)
            f = np.asarray([r["failure"] for r in y], float)
            nll = float((-s*np.log(np.clip(p,1e-8,1)) - f*np.log(np.clip(1-p,1e-8,1))).sum() /
                        max(1,(s+f).sum()))
            chosen = [ix[int(np.argmax(p[ix]))] for ix in by_state.values()]
            b15 = int(sum(y[j]["success"]>=15 for j in chosen))
            severe = int(sum(p[j]>.9 and y[j]["success"]/max(1,y[j]["observed_trials"])<=.5 for j in chosen))
            results.append({"kind": kind, "checkpoint_group": label, "seed": seed,
                            "trained_on_existing_third_v7_controller": trained_on_third,
                            "fresh_v8_labels_used_for_training_or_checkpoint": False,
                            "observed_NLL": nll, "B15_selected": b15,
                            "B15_oracle": oracle_B15,
                            "mean_selected_empirical_Q": float(np.mean([y[j]["success"]/max(1,y[j]["observed_trials"])
                                                                  for j in chosen])),
                            "severe_high_confidence_false_top1": severe,
                            "selected_row_indices": [y[j]["row_index"] for j in chosen]})
    with (DEST / "fresh_v8_selection.csv").open("w", newline="") as fh:
        fields = [x for x in results[0] if x != "selected_row_indices"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:v for k,v in r.items() if k in fields} for r in results])
    write(DEST / "fresh_v8_detail.json", {"oracle_B15": oracle_B15,
          "states": len(by_state), "candidate_Q16": y, "models": results,
          "new_controller_v8_never_in_model_training": True,
          "source_validation_states_were_used_for_checkpoint_selection_on_other_controllers": True})
    print(json.dumps({"oracle_B15": oracle_B15,"states":len(by_state),
                      "models": [{k:v for k,v in r.items() if k in ("checkpoint_group","seed","B15_selected","observed_NLL")}
                                 for r in results]},indent=2))


if __name__ == "__main__":
    run()
