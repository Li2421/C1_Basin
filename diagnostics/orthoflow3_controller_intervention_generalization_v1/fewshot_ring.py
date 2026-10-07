"""Post-hoc Ring TRAIN-label adaptation diagnostic, never zero-shot evidence.

Starts from source-only frozen H8 controller-conditioned critic. Ring TRAIN
families are selected outcome-blind by a stable hash. Ring VAL selects the
checkpoint. The already-opened frozen Ring K16 TEST is used once per frozen
curve point only to diagnose source-support versus capacity limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from scipy.special import expit

from diagnostics.orthoflow3_controller_context_loso_v1.evaluate import TARGET
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .train import Critic, OUT, OLD, read, write, sha

DEST = OUT / "fewshot_ring"
SIZES = (2, 8, 32, 64)
SEEDS = (17, 23, 41)


def ranked_families(rows):
    family = sorted({r["family"] for r in rows if r["scene"] == "ring_exchange"
                     and r["split"] == "train" and r["role"] == "canonical_historical"})
    return sorted(family, key=lambda s: hashlib.sha256(("ring-adaptation-v1|" + s).encode()).hexdigest())


def train(size, seed):
    dest = DEST / f"n{size}" / f"seed{seed}"
    if (dest / "training.json").exists(): return
    d = dict(np.load(OUT / "training_dataset.npz"))
    rows = read(OUT / "training_rows.json")
    families = ranked_families(rows)
    assert len(families) >= max(SIZES)
    chosen = set(families[:size])
    tr = np.asarray([i for i,r in enumerate(rows) if r["scene"] == "ring_exchange"
                     and r["role"] == "canonical_historical" and r["split"] == "train"
                     and r["family"] in chosen], int)
    va = np.asarray([i for i,r in enumerate(rows) if r["scene"] == "ring_exchange"
                     and r["role"] == "canonical_historical" and r["split"] == "validation"], int)
    assert len(tr) > 0 and len(va) > 0
    norm = read(OUT / "models/ring/contrast_full" / f"seed{seed}" / "normalization.json")
    assert not norm["target_data_used"] and norm["target_scene"] == "ring_exchange"
    center = np.asarray(norm["eta_center"], np.float32)
    radius = np.asarray(norm["eta_radius"], np.float32)
    ccenter = np.asarray(norm["context_center"], np.float32)
    cscale = np.asarray(norm["context_scale"], np.float32)
    eta = (d["eta"] - center) / radius
    context = (d["context"] - ccenter) / cscale
    x = {k: d[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    model = Critic(False)
    empty = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, 10)))
    start = OUT / "models/ring/contrast_full" / f"seed{seed}" / "checkpoint.msgpack"
    params = serialization.from_bytes(empty, start.read_bytes())
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(2e-4, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, o, xx, ee, cc, ss, ff):
        def loss(pp):
            z = model.apply(pp, xx, ee, cc)
            return jnp.mean((ss * jax.nn.softplus(-z) + ff * jax.nn.softplus(z)) / 16)
        value, grads = jax.value_and_grad(loss)(p)
        updates, o = optimizer.update(grads, o, p)
        return optax.apply_updates(p, updates), o, value

    predict = jax.jit(lambda p, xx, ee, cc: model.apply(p, xx, ee, cc))
    by_state = {}
    for i in tr:
        by_state.setdefault(rows[i]["state_uid"], []).append(i)
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    stale = 0
    history = []
    for it in range(1, 1601):
        state_keys = rng.choice(list(by_state), min(4, len(by_state)), replace=True)
        ix = np.concatenate([rng.choice(by_state[key], 16) for key in state_keys])
        params, opt_state, loss = step(params, opt_state, gather(x, d["state_index"][ix]),
                                       jnp.asarray(eta[ix]), jnp.asarray(context[ix]),
                                       jnp.asarray(d["s"][ix]), jnp.asarray(d["f"][ix]))
        if it % 50: continue
        logits = np.concatenate([np.asarray(predict(params, gather(x, d["state_index"][chunk]),
                                                jnp.asarray(eta[chunk]), jnp.asarray(context[chunk])))
                                 for chunk in np.array_split(va, max(1, int(np.ceil(len(va) / 256))))])
        val_nll = float((d["s"][va] * np.logaddexp(0,-logits) + d["f"][va] * np.logaddexp(0,logits)).sum() /
                        (d["s"][va] + d["f"][va]).sum())
        history.append({"step": it, "train_batch_loss": float(loss), "ring_val_observed_count_NLL": val_nll})
        if val_nll < best[0] - 1e-5:
            best = (val_nll, serialization.to_bytes(params), it)
            stale = 0
        else:
            stale += 1
        if stale >= 8 and it >= 400: break
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "history.json", history)
    write(dest / "training.json", {"size_train_source_families": size, "seed": seed,
                                   "train_pairs": len(tr), "ring_val_pairs": len(va),
                                   "best_step": best[2], "best_ring_val_NLL": best[0],
                                   "source_only_initial_checkpoint_sha256": sha(start),
                                   "source_normalization_unchanged": True,
                                   "target_scene_train_labels_used": True,
                                   "classification": "POST_HOC_FEW_SHOT_ADAPTATION_NOT_ZERO_SHOT",
                                   "frozen_K16_TEST_not_used_for_checkpoint_selection": True,
                                   "selected_families": families[:size]})
    print(json.dumps({"size": size, "seed": seed, "best_step": best[2], "val_NLL": best[0]}))


def evaluate():
    from .evaluate import save_csv

    manifest = read(TARGET / "ring" / "manifest.json")
    truth = read(OLD / "cached_truth.json")["ring"]
    assert [m["state_uid"] for m in manifest] == [t["state_uid"] for t in truth]
    x = dict(np.load(TARGET / "ring" / "entities.npz"))
    context = np.load(OUT / "target_nominal_ring.npz")["context"]
    eta = np.asarray([m["eta"] for m in manifest], np.float32).reshape(-1,3)
    state_ix = np.repeat(np.arange(len(manifest)), 16)
    robust = np.asarray([[v is True for v in t["robust"]] for t in truth], bool)
    unknown = np.asarray([[v is None for v in t["robust"]] for t in truth], bool)
    lower = np.asarray([t["lower"] for t in truth])
    output = []
    for size in SIZES:
        for seed in SEEDS:
            folder = DEST / f"n{size}" / f"seed{seed}"
            training = read(folder / "training.json")
            assert training["classification"] == "POST_HOC_FEW_SHOT_ADAPTATION_NOT_ZERO_SHOT"
            norm = read(OUT / "models/ring/contrast_full" / f"seed{seed}" / "normalization.json")
            e = (eta - np.asarray(norm["eta_center"], np.float32)) / np.asarray(norm["eta_radius"], np.float32)
            c = (context - np.asarray(norm["context_center"], np.float32)) / np.asarray(norm["context_scale"], np.float32)
            model = Critic(False)
            empty = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1,3)), jnp.zeros((1,10)))
            params = serialization.from_bytes(empty, (folder / "checkpoint.msgpack").read_bytes())
            fn = jax.jit(lambda xx, ee, cc: model.apply(params, xx, ee, cc))
            z = np.concatenate([np.asarray(fn(gather(x, state_ix[j:j+128]), jnp.asarray(e[j:j+128]),
                                             jnp.asarray(c[state_ix[j:j+128]])))
                                for j in range(0, len(eta), 128)]).reshape(len(manifest),16)
            choice = z.argmax(1)
            ii = np.arange(len(manifest))
            output.append({"train_families": size, "seed": seed,
                           "B15": int(robust[ii,choice].sum()), "unresolved": int(unknown[ii,choice].sum()),
                           "oracle_B15": int(robust.any(1).sum()),
                           "mean_selected_Q_lower": float(lower[ii,choice].mean()),
                           "mean_selected_probability": float(expit(z[ii,choice]).mean()),
                           "severe_FP": int(((expit(z[ii,choice]) > .9) & (lower[ii,choice] <= .5)).sum()),
                           "POST_HOC_NOT_ZERO_SHOT": True})
    save_csv("fewshot_ring/results.csv", output)
    write(DEST / "decision.json", {"classification": "POST_HOC_FEW_SHOT_ADAPTATION_NOT_ZERO_SHOT",
                                   "source_only_normalization": True, "target_train_and_val_labels_used": True,
                                   "K16_TEST_opened_previously_for_diagnostic": True,
                                   "results": output})
    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--size", type=int, choices=SIZES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "train": train(args.size, args.seed)
    else: evaluate()
