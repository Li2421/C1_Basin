#!/usr/bin/env python3
"""Mode-free conditional eta density, trained on existing canonical pairs only."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq

ROOT = Path("/home/zhihan/research/Basin_C1")
OUT = Path(__file__).resolve().parent
PAIR = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1/pair_table.parquet"
MAN = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json"
SEEDS = (17, 23, 41)
CENTER = np.array([0.625, 0.0, 0.375], np.float32)
RADIUS = np.array([0.625, 0.5, 0.375], np.float32)


class Generator(nn.Module):
    @nn.compact
    def __call__(self, h, scenario):
        # One model q(eta|h,c); c selects an input adapter, not an eta mode.
        if scenario == "Toy":
            x = nn.silu(nn.Dense(64, name="toy_adapter")(h))
        else:
            x = nn.silu(nn.Dense(64, name="db_adapter")(h))
        x = nn.silu(nn.Dense(64, name="shared1")(x))
        x = nn.silu(nn.Dense(64, name="shared2")(x))
        return nn.Dense(6, name="out")(x)


def dist_params(raw):
    mu = raw[..., :3]
    sigma = 0.025 + 0.275 * jax.nn.sigmoid(raw[..., 3:])
    return mu, sigma


def mean_eta(raw):
    return jnp.asarray(CENTER) + jnp.asarray(RADIUS) * jnp.tanh(raw[..., :3])


def log_prob(raw, eta):
    u = jnp.clip((eta - jnp.asarray(CENTER)) / jnp.asarray(RADIUS), -0.999999, 0.999999)
    z = jnp.arctanh(u)
    mu, sigma = dist_params(raw)
    gaussian = -0.5 * (((z - mu) / sigma) ** 2 + 2 * jnp.log(sigma) + jnp.log(2 * jnp.pi))
    jac = jnp.log(jnp.asarray(RADIUS) * (1 - u * u) + 1e-9)
    return jnp.sum(gaussian - jac, axis=-1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    rows = pq.read_table(PAIR).to_pylist()
    manifest = json.loads(MAN.read_text())
    stats = manifest["state_normalization"]
    data = {}
    for scen in ("Toy", "DB"):
        mean = np.asarray(stats[scen]["mean"], np.float32)
        std = np.asarray(stats[scen]["std"], np.float32)
        data[scen] = {}
        for split in ("train", "val"):
            selected = [r for r in rows if r["scenario"] == scen and r["state_split"] == split]
            selected = [r for r in selected if 0 <= r["eta1"] <= 1.25 and -0.5 <= r["eta2"] <= 0.5 and 0 <= r["eta3"] <= 0.75]
            unique = {}
            for r in selected:
                key = (r["state_uid"], r["eta_uid"], r["controller_uid"])
                if key not in unique or r["n_trials"] > unique[key]["n_trials"]:
                    unique[key] = r
            selected = list(unique.values())
            positives = [r for r in selected if r["empirical_q"] >= 0.9]
            negatives = [r for r in selected if r["empirical_q"] <= 0.5]
            pos_state = defaultdict(list)
            for r in positives:
                pos_state[r["state_uid"]].append(r)
            paired = []
            for r in negatives:
                if r["state_uid"] in pos_state:
                    p = min(pos_state[r["state_uid"]], key=lambda x: (-x["empirical_q"], x["eta_uid"]))
                    paired.append((r, p))
            def aeta(seq):
                return np.asarray([[r["eta1"], r["eta2"], r["eta3"]] for r in seq], np.float32)
            def ah(seq):
                return (np.asarray([r["h_raw"] for r in seq], np.float32) - mean) / std
            pos_counts = defaultdict(int)
            for r in positives: pos_counts[r["state_uid"]] += 1
            d = {
                "h": ah(positives), "eta": aeta(positives),
                "weight": np.asarray([(0.5 + 0.5 * r["empirical_q"]) / pos_counts[r["state_uid"]] for r in positives], np.float32),
                "neg_h": ah([r for r, _ in paired]),
                "neg_eta": aeta([r for r, _ in paired]),
                "paired_pos_eta": aeta([p for _, p in paired]),
            }
            data[scen][split] = d
            print(scen, split, "states", len(pos_counts), "positive", len(positives), "paired negative", len(paired), flush=True)
    return data


def run(seed, data):
    model = Generator()
    rng = jax.random.PRNGKey(seed)
    toy = model.init(rng, jnp.zeros((1, 214), jnp.float32), "Toy")
    db = model.init(rng, jnp.zeros((1, 80), jnp.float32), "DB")
    from flax.core import freeze, unfreeze
    params = unfreeze(toy)
    params["params"]["db_adapter"] = unfreeze(db)["params"]["db_adapter"]
    params = freeze(params)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(5e-4, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, th, te, tw, tnh, tne, tpe, dh, de, dw, dnh, dne, dpe):
        def branch(p, scen, h, eta, w, nh, ne, pe):
            raw = model.apply(p, h, scen)
            pos = -jnp.sum(w * log_prob(raw, eta)) / jnp.maximum(jnp.sum(w), 1e-8)
            nraw = model.apply(p, nh, scen)
            neg = jnp.mean(jax.nn.softplus(1.0 + log_prob(nraw, ne) - log_prob(nraw, pe)))
            return pos + neg
        def objective(p):
            return 0.5 * branch(p, "Toy", th, te, tw, tnh, tne, tpe) + 0.5 * branch(p, "DB", dh, de, dw, dnh, dne, dpe)
        value, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    def val_loss(p):
        scores = []
        for scen in ("Toy", "DB"):
            d = data[scen]["val"]
            raw = model.apply(p, jnp.asarray(d["h"]), scen)
            lp = np.asarray(log_prob(raw, jnp.asarray(d["eta"])))
            pos = -float(np.sum(d["weight"] * lp) / np.sum(d["weight"]))
            if len(d["neg_h"]):
                nr = model.apply(p, jnp.asarray(d["neg_h"]), scen)
                gap = 1 + np.asarray(log_prob(nr, jnp.asarray(d["neg_eta"]))) - np.asarray(log_prob(nr, jnp.asarray(d["paired_pos_eta"])))
                neg = float(np.mean(np.logaddexp(0, gap)))
            else: neg = 0.0
            scores.append(pos + neg)
        return float(np.mean(scores))

    sample_rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    history = []
    for it in range(1, 2001):
        args = []
        for scen in ("Toy", "DB"):
            d = data[scen]["train"]
            ix = sample_rng.integers(0, len(d["h"]), 256)
            ni = sample_rng.integers(0, len(d["neg_h"]), 128)
            args.extend([jnp.asarray(d["h"][ix]), jnp.asarray(d["eta"][ix]), jnp.asarray(d["weight"][ix]),
                         jnp.asarray(d["neg_h"][ni]), jnp.asarray(d["neg_eta"][ni]), jnp.asarray(d["paired_pos_eta"][ni])])
        params, opt_state, train = step(params, opt_state, *args)
        if it % 100 == 0:
            val = val_loss(params)
            history.append({"step": it, "train": float(train), "val": val})
            if val < best[0]: best = (val, serialization.to_bytes(params), it)
    name = f"seed{seed}"
    (OUT / name).mkdir(exist_ok=True)
    (OUT / name / "checkpoint.msgpack").write_bytes(best[1])
    (OUT / name / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"seed": seed, "val_loss": best[0], "step": best[2], "checkpoint": str(OUT / name / "checkpoint.msgpack"), "sha256": sha(OUT / name / "checkpoint.msgpack")}


def main():
    OUT.mkdir(exist_ok=True)
    data = load_data()
    results = [run(seed, data) for seed in SEEDS]
    selected = min(results, key=lambda x: (x["val_loss"], x["seed"]))
    (OUT / "generator_training.json").write_text(json.dumps({"results": results, "selected": selected,
        "source_pair_table": str(PAIR), "source_sha256": sha(PAIR), "mode_ids_used": False,
        "test_outcomes_used": False, "eta_center": CENTER.tolist(), "eta_radius": RADIUS.tolist()}, indent=2) + "\n")
    print(json.dumps({"selected": selected}, indent=2))


if __name__ == "__main__": main()
