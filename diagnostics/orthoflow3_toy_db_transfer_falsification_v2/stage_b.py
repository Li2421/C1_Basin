#!/usr/bin/env python3
"""Frozen-head mode-ID permutation control (no rollout acquisition)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

H = Path(__file__).parent
D = H.parent
TOY = D / "orthoflow3_shared_eta_codebook_v1"
DB = D / "orthoflow3_db_shared_mode_transfer_v1"
JOINT = D / "orthoflow3_toy_db_joint_selector_v1"
SEEDS = (17, 23, 41)
M = 12


def rows(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, data):
    path = H / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(data[0]) if data else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def dump(path, obj):
    path = H / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_data(root, split):
    z = np.load(root / "state_features.npz")
    mask = z["splits"].astype(str) == split
    return z["state_ids"].astype(str)[mask], z["x"].astype(np.float32)[mask]


def label_data(root, split):
    fn = {"train": "train_mode_counts.csv", "val": "val_mode_counts.csv", "test": "test_mode_q64.csv"}[split]
    ids, _ = feature_data(root, split)
    idx = {s: i for i, s in enumerate(ids)}
    y = np.full((len(ids), M), np.nan, np.float32)
    for r in rows(root / fn):
        y[idx[r["state_id"]], int(r["mode_id"])] = float(r["successes"]) / float(r["trials"])
    if np.isnan(y).any():
        raise RuntimeError(f"incomplete {root.name}/{split} matrix")
    return ids, y


def metric(logits, y, perm):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    perm = np.asarray(perm, int)
    target = y[:, perm]
    eps = 1e-7
    nll = float(np.mean(-(target * np.log(p + eps) + (1 - target) * np.log(1 - p + eps))))
    brier = float(np.mean((p - target) ** 2))
    out_col = np.argmax(p, axis=1)
    actual_mode = perm[out_col]
    q = y[np.arange(len(y)), actual_mode]
    oracle = y.max(axis=1)
    return {
        "nll": nll,
        "brier": brier,
        "mean_selected_Q": float(q.mean()),
        "robust15_states": int(np.sum(q >= 15 / 16)),
        "robust15_fraction": float(np.mean(q >= 15 / 16)),
        "mean_regret": float(np.mean(oracle - q)),
        "mode_counts": json.dumps({str(int(k)): int(v) for k, v in zip(*np.unique(actual_mode, return_counts=True))}, sort_keys=True),
    }


class ToySemantic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(64, name="adapter")(x))
        x = nn.silu(nn.Dense(64, name="trunk")(x))
        return nn.Dense(M, name="head")(x)


def train_toy():
    tid, tx = feature_data(TOY, "train")
    _, ty = label_data(TOY, "train")
    _, vx = feature_data(TOY, "val")
    _, vy = label_data(TOY, "val")
    model = ToySemantic()
    summaries, saved = [], {}
    for seed in SEEDS:
        params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, tx.shape[1]), jnp.float32))
        opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
        state = opt.init(params)

        @jax.jit
        def step(p, s, x, y):
            def loss(q):
                return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(q, x), y))
            val, grad = jax.value_and_grad(loss)(p)
            upd, s = opt.update(grad, s, p)
            return optax.apply_updates(p, upd), s, val

        rng = np.random.default_rng(seed)
        best_params, best_key, best_epoch, wait = None, None, 0, 0
        for epoch in range(2400):
            order = rng.permutation(len(tx))
            for start in range(0, len(tx), 32):
                ii = order[start:start + 32]
                params, state, _ = step(params, state, jnp.asarray(tx[ii]), jnp.asarray(ty[ii]))
            vm = metric(np.asarray(model.apply(params, jnp.asarray(vx))), vy, np.arange(M))
            key = (vm["nll"], -vm["mean_selected_Q"], vm["brier"])
            if best_key is None or key < best_key:
                best_params = jax.tree_util.tree_map(np.asarray, params)
                best_key, best_epoch, wait = key, epoch, 0
            else:
                wait += 1
            if wait >= 180:
                break
        vm = metric(np.asarray(model.apply(best_params, jnp.asarray(vx))), vy, np.arange(M))
        out = H / "toy_semantic_model" / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        ck = out / "checkpoint.msgpack"
        ck.write_bytes(serialization.to_bytes(best_params))
        summaries.append({"seed": seed, "best_epoch": best_epoch, "epochs": epoch + 1,
                          "checkpoint": str(ck), "checkpoint_sha256": sha(ck), **vm})
        saved[seed] = best_params
    write_csv("toy_semantic_model/training_summary.csv", summaries)
    selected = min(summaries, key=lambda r: (r["nll"], -r["mean_selected_Q"], r["brier"], r["seed"]))
    dump("toy_semantic_model/selected.json", selected)
    return saved[int(selected["seed"])]["params"], selected


def adapter_init(seed, in_dim):
    key = jax.random.PRNGKey(seed)
    kernel = jax.nn.initializers.lecun_normal()(key, (in_dim, 64), jnp.float32)
    return {"kernel": kernel, "bias": jnp.zeros((64,), jnp.float32)}


def main():
    frozen, toy_selected = train_toy()
    rng = np.random.default_rng(20260930)
    permutations = []
    while len(permutations) < 10:
        p = rng.permutation(M)
        if np.all(p != np.arange(M)) and p.tolist() not in permutations:
            permutations.append(p.tolist())
    dump("permutations.json", {"seed": 20260930, "constraint": "10 deterministic derangements; no fixed points", "permutations": permutations})

    train_ids, train_x = feature_data(DB, "train")
    _, train_y = label_data(DB, "train")
    _, val_x = feature_data(DB, "val")
    _, val_y = label_data(DB, "val")
    _, test_x = feature_data(DB, "test")
    _, test_y = label_data(DB, "test")
    subsets = json.load(open(JOINT / "db_train_subsets.json"))
    conditions = [("correct", list(range(M)))] + [(f"perm_{i:02d}", p) for i, p in enumerate(permutations)]
    results = []

    trunk, head = frozen["trunk"], frozen["head"]
    def apply_adapter(p, x):
        a = nn.silu(x @ p["kernel"] + p["bias"])
        z = nn.silu(a @ trunk["kernel"] + trunk["bias"])
        return z @ head["kernel"] + head["bias"]

    for fraction in (25, 50):
        keep_ids = set(subsets[str(fraction)])
        keep = np.asarray([sid in keep_ids for sid in train_ids])
        x, y = train_x[keep], train_y[keep]
        for condition, perm in conditions:
            target = y[:, np.asarray(perm)]
            for seed in SEEDS:
                params = adapter_init(seed, x.shape[1])
                opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
                state = opt.init(params)

                @jax.jit
                def step(p, s, xb, yb):
                    def loss(q):
                        return jnp.mean(optax.sigmoid_binary_cross_entropy(apply_adapter(q, xb), yb))
                    value, grad = jax.value_and_grad(loss)(p)
                    upd, s = opt.update(grad, s, p)
                    return optax.apply_updates(p, upd), s, value

                order_rng = np.random.default_rng(seed)
                for epoch in range(1200):
                    order = order_rng.permutation(len(x))
                    for start in range(0, len(x), 16):
                        ii = order[start:start + 16]
                        params, state, _ = step(params, state, jnp.asarray(x[ii]), jnp.asarray(target[ii]))
                val_m = metric(np.asarray(apply_adapter(params, jnp.asarray(val_x))), val_y, perm)
                test_m = metric(np.asarray(apply_adapter(params, jnp.asarray(test_x))), test_y, perm)
                base = "db_correct_adaptation" if condition == "correct" else "db_permutation_controls"
                out = H / base / f"db{fraction}_{condition}_seed{seed}"
                out.mkdir(parents=True, exist_ok=True)
                ck = out / "adapter_checkpoint.msgpack"
                ck.write_bytes(serialization.to_bytes(params))
                results.append({"db_fraction": fraction, "condition": condition,
                                "permutation": json.dumps(perm, separators=(",", ":")), "seed": seed,
                                "train_states": len(x), "epochs": 1200,
                                "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
                                **{f"val_{k}": v for k, v in val_m.items()},
                                **{f"test_{k}": v for k, v in test_m.items()}})
    write_csv("permutation_results.csv", results)

    comparison = []
    lower_better = {"test_nll", "test_brier", "test_mean_regret"}
    metrics = ("test_nll", "test_brier", "test_mean_selected_Q", "test_robust15_states", "test_mean_regret")
    for fraction in (25, 50):
        block = [r for r in results if r["db_fraction"] == fraction]
        correct = [r for r in block if r["condition"] == "correct"]
        shuffled = [r for r in block if r["condition"] != "correct"]
        out = {"db_fraction": fraction, "correct_models": len(correct), "shuffled_models": len(shuffled)}
        for name in metrics:
            c = np.asarray([float(r[name]) for r in correct])
            s = np.asarray([float(r[name]) for r in shuffled])
            out.update({f"correct_{name}_median": float(np.median(c)),
                        f"correct_{name}_min": float(c.min()), f"correct_{name}_max": float(c.max()),
                        f"shuffled_{name}_median": float(np.median(s)),
                        f"shuffled_{name}_q25": float(np.quantile(s, .25)),
                        f"shuffled_{name}_q75": float(np.quantile(s, .75)),
                        f"shuffled_{name}_best": float(s.min() if name in lower_better else s.max()),
                        f"shuffled_{name}_worst": float(s.max() if name in lower_better else s.min())})
        comparison.append(out)
    write_csv("correct_vs_shuffled.csv", comparison)
    dump("stage_b_summary.json", {"toy_selected": toy_selected, "models": len(results), "comparison": comparison,
                                  "new_rollouts": 0, "robust_definition": "selected empirical Q >= 15/16"})
    print(json.dumps({"models": len(results), "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
