"""Source-only concat Q(s,η) and capacity-matched Q(η) training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, read, sha
from diagnostics.physical_state_critic_formal_v1.train_unified import metrics as metrics_q16, select_key

SEEDS = (17, 23)
HERE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE)))
STEPS = tuple(range(200, int(os.environ.get("TRANSFER_Q_MAX_STEPS", "6000")) + 1, 200))
NEAR_TIE = 1/512


def metrics(logits, success, failure, numerical, scenes):
    """Keep validation selection scores comparable for exact Q4 or Q16 rows.

    The shared scorer expresses empirical Q on a 16-seed scale. Multiplying
    all outcome counts by 16/n preserves NLL, ranking and selection regret;
    its B15 field is only a provisional n/n diagnostic when n < 16.
    """
    exposure = np.asarray(success)+np.asarray(failure)+np.asarray(numerical)
    assert np.all(exposure == exposure.flat[0])
    n = int(exposure.flat[0])
    assert n in (4, 16)
    factor = 16 // n
    score = metrics_q16(logits, np.asarray(success)*factor,
                        np.asarray(failure)*factor,
                        np.asarray(numerical)*factor, scenes)
    score["validation_rollout_seeds_per_pair"] = n
    return score


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def fit(panel, kind, seed):
    assert panel in PANELS and kind in ("state_concat", "eta_only_wide") and seed in SEEDS
    data_path = HERE / "source_data" / panel / "trainval.npz"
    manifest = read(HERE / "source_data" / panel / "manifest.json")
    assert sha(data_path) == manifest["trainval_sha256"] and not manifest["test_labels_opened"]
    folder = HERE / "fits" / panel / kind / f"seed{seed}"
    if (folder / "complete.json").exists():
        result = read(folder / "complete.json")
        assert result["trainval_sha256"] == sha(data_path)
        assert result["checkpoint_sha256"] == sha(folder / "checkpoint.msgpack")
        return result
    folder.mkdir(parents=True, exist_ok=True)
    d = np.load(data_path)
    train = np.flatnonzero(d["split"] == "train")
    val = np.flatnonzero(d["split"] == "validation")
    assert len(train) == manifest["train_states"] and len(val) == manifest["validation_states"]
    e = jnp.asarray(d["eta"], jnp.float32)
    success = jnp.asarray(d["success"], jnp.float32)
    failure = jnp.asarray(d["failure"], jnp.float32)
    x = {key[7:]: jnp.asarray(d[key]) for key in d.files if key.startswith("entity_")}
    net = rep.Critic() if kind == "state_concat" else WideEtaOnly()

    def gather(ii):
        return {k: v[ii] for k, v in x.items()}

    params = net.init(jax.random.PRNGKey(seed), gather(jnp.array([0])), e[:1, :1].reshape(1, 3))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
    optimizer_state = optimizer.init(params)
    mean_exposure = float(np.mean(np.asarray(success + failure)[train]))
    assert mean_exposure > 0

    @jax.jit
    def train_block(weights, opt_state, draws):
        def one_step(carry, sample):
            p, o = carry
            ii, jj = sample[:, 0], sample[:, 1]

            def loss(pp):
                z = net.apply(pp, gather(ii), e[ii, jj])
                return jnp.mean(success[ii, jj] * jax.nn.softplus(-z) +
                                failure[ii, jj] * jax.nn.softplus(z)) / mean_exposure

            value, grad = jax.value_and_grad(loss)(p)
            updates, o = optimizer.update(grad, o, p)
            return (optax.apply_updates(p, updates), o), value

        return jax.lax.scan(one_step, (weights, opt_state), draws)

    @jax.jit
    def predict(weights, ii, jj):
        return net.apply(weights, gather(ii), e[ii, jj])

    scenes = np.asarray(d["scene"])
    pools = {sc: np.asarray([(i, j) for i in train[scenes[train] == sc]
                             for j in range(16) if d["success"][i, j] + d["failure"][i, j] > 0], np.int32)
             for sc in sorted(set(scenes[train]))}
    assert all(len(v) >= 500 for v in pools.values())
    val_i = np.repeat(val, 16)
    val_j = np.tile(np.arange(16), len(val))
    rng = np.random.default_rng(seed)
    best = None
    best_bytes = None
    history = []
    prior_step = 0
    started = time.monotonic()
    for step in STEPS:
        parts = []
        for k, pool in enumerate(pools.values()):
            n = 128 // len(pools) + (k < 128 % len(pools))
            parts.append(pool[rng.integers(len(pool), size=(step-prior_step, n))])
        draws = np.concatenate(parts, axis=1)
        (params, optimizer_state), _ = train_block(params, optimizer_state, jnp.asarray(draws))
        pred = []
        for a in range(0, len(val_i), 256):
            pred.append(np.asarray(predict(params, jnp.asarray(val_i[a:a+256]),
                                           jnp.asarray(val_j[a:a+256]))))
        logits = np.concatenate(pred).reshape(len(val), 16)
        score = metrics(logits, d["success"][val], d["failure"][val],
                        d["numerical"][val], scenes[val])
        row = {"step": step, **score}
        history.append(row)
        if best is None or select_key(row, NEAR_TIE) < select_key(best, NEAR_TIE):
            best = row
            best_bytes = serialization.to_bytes(params)
        prior_step = step
        if step % 1000 == 0:
            print(json.dumps({"panel": panel, "kind": kind, "seed": seed, "step": step,
                              "val_regret": row["selection_regret"], "best_step": best["step"],
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
    (folder / "checkpoint.msgpack").write_bytes(best_bytes)
    write(folder / "history.json", history)
    result = {"panel": panel, "kind": kind, "seed": seed, "selected": best,
              "optimization_steps_max": max(STEPS),
              "trainval_sha256": sha(data_path), "checkpoint_sha256": sha(folder / "checkpoint.msgpack"),
              "feature_sha256": manifest["feature_code_sha256"],
              "checkpoint_rule": "min source-scene mean selection regret; pairwise then NLL near-ties",
              "target_labels_used": False, "elapsed_seconds": time.monotonic()-started,
              "backend": jax.default_backend()}
    write(folder / "complete.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", choices=tuple(PANELS), required=True)
    p.add_argument("--kind", choices=("state_concat", "eta_only_wide"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    a = p.parse_args()
    print(json.dumps(fit(a.panel, a.kind, a.seed)), flush=True)
