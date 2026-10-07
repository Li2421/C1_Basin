"""Freeze source-trained critic predictions before held-controller Q16 is opened."""
from __future__ import annotations

import argparse
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from diagnostics.orthoflow3_controller_intervention_generalization_v1.intervention import OLD
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from shared_rollout_db.src.rollout_db import connect

from .held_robust_benchmark import DEST, FLOW
from .probe import OUT, Critic, load_scene, read, write


def feature_shard(shard, shards):
    if (DEST / f"context_shard{shard}of{shards}.json").exists():
        raise FileExistsError("Context shard already exists")
    h20.rc.OUT = DEST
    base = h20.rc.RichRuntime("ring_exchange")
    held = h20.rc.RichRuntime("ring_exchange", FLOW)
    rows = read(DEST / "pairs.json")
    states = read(OLD / "states.json")
    output = []
    for i, row in enumerate(rows):
        if i % shards != shard:
            continue
        state = {"state_uid": row["state_uid"], "physical": states[row["state_index"]]["physical"]}
        assert states[row["state_index"]]["state_uid"] == row["state_uid"]
        a = h20.rc.cached(held, state, row["eta"])
        b = h20.rc.cached(base, state, row["eta"])
        output.append({"pair_index": i, "held_valid": a["valid"], "base_valid": b["valid"],
                       "held": a["features"]["mean"] if a["valid"] else None,
                       "wrong_base": b["features"]["mean"] if b["valid"] else None,
                       "held_error": a.get("error"), "base_error": b.get("error")})
        if len(output) % 16 == 0:
            print(json.dumps({"shard": shard, "done": len(output)}), flush=True)
    write(DEST / f"context_shard{shard}of{shards}.json", output)
    print(json.dumps({"shard": shard, "complete": len(output),
                      "valid": sum(r["held_valid"] and r["base_valid"] for r in output)}), flush=True)


def freeze_scores(shards):
    path = DEST / "frozen_predictions.json"
    if path.exists():
        raise FileExistsError("Held-controller predictions already frozen")
    pairs = read(DEST / "pairs.json")
    profile = read(DEST / "protocol.json")["profiles"]["ring_exchange"]
    with connect(True) as db:
        n = db.execute("SELECT COUNT(*) FROM rollout WHERE controller_uid=?",
                       (profile["alternate_controller_uid"],)).fetchone()[0]
    assert n == 0, "Prediction freezing must precede held-controller task outcomes"
    contexts = [r for shard in range(shards)
                for r in read(DEST / f"context_shard{shard}of{shards}.json")]
    assert len(contexts) == len(pairs) and len({r["pair_index"] for r in contexts}) == len(pairs)
    contexts.sort(key=lambda r: r["pair_index"])
    assert all(r["held_valid"] and r["base_valid"] for r in contexts)
    held = np.asarray([r["held"] for r in contexts], np.float32)
    wrong = np.asarray([r["wrong_base"] for r in contexts], np.float32)
    assert held.shape == wrong.shape == (len(pairs), 24)
    scene_data = load_scene("ring_exchange")
    x = scene_data["x"]
    state_index = np.asarray([r["state_index"] for r in pairs], int)
    predictions = []
    for kind in ("eta_only", "physical_context"):
        model = Critic(kind != "eta_only", kind != "eta_only", False)
        for seed in (17, 23, 41):
            folder = OUT / "models/ring_exchange" / kind / f"seed{seed}"
            norm = read(folder / "normalization.json")
            eta = (np.asarray([r["eta"] for r in pairs], np.float32)
                   - np.asarray(norm["eta_center"], np.float32)) / np.asarray(norm["eta_scale"], np.float32)
            c0 = np.asarray(norm["context_center"], np.float32)
            cs = np.asarray(norm["context_scale"], np.float32)
            template = model.init(jax.random.PRNGKey(seed), gather(x, state_index[:1]),
                                  jnp.zeros((1, 3)), jnp.zeros((1, 24)), jnp.zeros((1, 3)))
            params = serialization.from_bytes(template, (folder / "checkpoint.msgpack").read_bytes())
            for variant, raw in (("correct", held), ("wrong_base", wrong)) if kind != "eta_only" else (("none", held),):
                context = (raw - c0) / cs
                logits = np.asarray(model.apply(params, gather(x, state_index), jnp.asarray(eta),
                                                jnp.asarray(context), jnp.zeros((len(pairs), 3))))
                p = np.asarray(jax.nn.sigmoid(logits))
                predictions.append({"kind": kind, "seed": seed, "context": variant,
                                    "checkpoint_sha256": hashlib.sha256((folder / "checkpoint.msgpack").read_bytes()).hexdigest(),
                                    "probabilities": p.astype(float).tolist()})
    write(path, {"schema": "pre_outcome_frozen_critic_scores_v1",
                 "held_task_outcomes_present_at_score_time": False,
                 "source_controller_checkpoints_only": True,
                 "physical_state_families_absent_from_probe_train_val": True,
                 "candidate_pool_is_score_blind": True,
                 "pair_keys": [{"state_uid": r["state_uid"], "eta_uid": r["eta_uid"]} for r in pairs],
                 "predictions": predictions,
                 "context_feature_distance_median": float(np.median(np.linalg.norm(held-wrong, axis=1)))})
    print(json.dumps({"pairs": len(pairs), "models": len(predictions),
                      "context_feature_distance_median": float(np.median(np.linalg.norm(held-wrong, axis=1)))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("features", "freeze"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.action == "features": feature_shard(args.shard, args.shards)
    else: freeze_scores(args.shards)
