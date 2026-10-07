"""Source-family test of nominal H8 versus eta-conditioned H8 information.

Only existing matched controller interventions are used. This diagnostic does
not touch frozen K16 target labels or modify the deployment generator.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .train import Critic, OUT, read, sha, write
from .rich_context import key

SCENES = ("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange")


def context(scene, uid, eta, sha_alt):
    row = read(OUT / "rich_response_cache" / f"{key(scene, uid, eta, sha_alt)}.json")
    assert row["valid"], (scene, uid, row.get("error"))
    return row["features"]["mean"]


def materialize():
    if (OUT / "rich_probe_dataset.npz").exists():
        return
    npz = dict(np.load(OUT / "training_dataset.npz"))
    rows = read(OUT / "training_rows.json")
    index = {}
    for r in rows:
        index.setdefault(r["state_uid"], r["state_index"])
    balanced = list(csv.DictReader((OUT / "balanced_expansion/intervention_pairs.csv").open()))
    protocol = read(OUT / "balanced_expansion/protocol.json")
    result = []
    for r in balanced:
        if int(r["alternate_valid"]) != 16:
            continue
        eta = json.loads(r["eta"])
        a = context(r["scene"], r["state_uid"], eta, None)
        b = context(r["scene"], r["state_uid"], eta,
                    protocol["profiles"][r["scene"]]["alternate_flow_sha256"])
        result.append({"scene": r["scene"], "state_uid": r["state_uid"],
                       "state_index": index[r["state_uid"]], "split": r["split"],
                       "eta": eta, "base_s": int(r["base_observed_success"]),
                       "base_f": int(r["base_observed_failure"]),
                       "alt_s": int(r["alternate_success"]), "alt_f": 16 - int(r["alternate_success"]),
                       "base_c": a, "alt_c": b})
    toy = OUT.parent / "orthoflow3_controller_conditioning_probe_v1"
    toy_protocol = read(toy / "protocol.json")
    toy_rows = list(csv.DictReader((toy / "controller_pair_results.csv").open()))
    order = sorted({r["source_group"] for r in toy_rows},
                   key=lambda f: __import__("hashlib").sha256(f"intervention-family-v1|{f}".encode()).hexdigest())
    val = set(order[:4])
    for r in toy_rows:
        eta = json.loads(r["eta"])
        result.append({"scene": "toy_giveway", "state_uid": r["state_uid"],
                       "state_index": index[r["state_uid"]],
                       "split": "validation" if r["source_group"] in val else "train",
                       "eta": eta, "base_s": int(r["base_success"]),
                       "base_f": 16 - int(r["base_success"]),
                       "alt_s": int(r["alternate_success"]),
                       "alt_f": 16 - int(r["alternate_success"]),
                       "base_c": context("toy_giveway", r["state_uid"], eta, None),
                       "alt_c": context("toy_giveway", r["state_uid"], eta,
                                        toy_protocol["flow_sha256"]["1"])})
    arrays = {name: np.asarray([r[name] for r in result], np.float32)
              for name in ("eta", "base_s", "base_f", "alt_s", "alt_f", "base_c", "alt_c")}
    arrays["state_index"] = np.asarray([r["state_index"] for r in result], np.int32)
    np.savez_compressed(OUT / "rich_probe_dataset.npz", **arrays,
                        **{k: npz[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")})
    write(OUT / "rich_probe_rows.json", [{k: r[k] for k in ("scene", "state_uid", "split")} for r in result])
    print(json.dumps({"pairs": len(result), "states": len({r["state_uid"] for r in result})}))


def materialize_h20():
    if (OUT / "rich_probe_dataset_h20.npz").exists():
        return
    from . import h20_features as h20

    old = dict(np.load(OUT / "rich_probe_dataset.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    protocol = read(OUT / "balanced_expansion/protocol.json")
    toy_protocol = read(OUT.parent / "orthoflow3_controller_conditioning_probe_v1/protocol.json")
    exact_eta = {}
    for item in read(OUT / "balanced_expansion/pairs.json"):
        exact_eta[(item["scene"], item["state_uid"], tuple(np.asarray(item["eta"], np.float32)))] = item["eta"]
    for item in read(OUT.parent / "orthoflow3_controller_conditioning_probe_v1/pair_manifest.json"):
        exact_eta[("toy_giveway", item["state_uid"], tuple(np.asarray(item["eta"], np.float32)))] = item["eta"]
    base, alt = [], []
    invalid = []
    for i, r in enumerate(rows):
        scene, uid = r["scene"], r["state_uid"]
        eta = exact_eta[(scene, uid, tuple(old["eta"][i]))]
        alt_sha = (toy_protocol["flow_sha256"]["1"] if scene == "toy_giveway"
                   else protocol["profiles"][scene]["alternate_flow_sha256"])
        def get(sha_alt):
            path = h20.rc.OUT / "rich_response_cache" / f"{h20.rc.key(scene, uid, eta, sha_alt)}.json"
            result = read(path)
            if not result["valid"]:
                invalid.append({"index": i, "scene": scene, "state_uid": uid,
                                "error": result.get("error")})
                return [0.] * 24
            return result["features"]["mean"]
        base.append(get(None));alt.append(get(alt_sha))
    assert not invalid, invalid
    old["base_c"] = np.asarray(base, np.float32)
    old["alt_c"] = np.asarray(alt, np.float32)
    np.savez_compressed(OUT / "rich_probe_dataset_h20.npz", **old)
    write(OUT / "h20_feature_audit.json", {"pairs": len(rows), "invalid": invalid,
                                           "same_pairs_and_splits_as_H8": True,
                                           "max_context_steps": 20,
                                           "frozen_target_labels_used": False})
    print(json.dumps({"pairs": len(rows), "invalid": len(invalid)}))


def train(kind, seed):
    dest = OUT / "rich_probe_models" / kind / f"seed{seed}"
    if (dest / "result.json").exists():
        return
    d = dict(np.load(OUT / ("rich_probe_dataset_h20.npz" if kind.endswith("20") else "rich_probe_dataset.npz")))
    rows = read(OUT / "rich_probe_rows.json")
    x = {k: d[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    scenes = {sc: {sp: np.asarray([i for i, r in enumerate(rows) if r["scene"] == sc and r["split"] == sp], int)
                   for sp in ("train", "validation")} for sc in SCENES}
    eta = d["eta"]
    train_ix = np.concatenate([scenes[sc]["train"] for sc in SCENES])
    eta_center = eta[train_ix].mean(0)
    eta_scale = np.maximum(eta[train_ix].std(0), .1)
    eta = (eta - eta_center) / eta_scale
    c0 = d["base_c"] if kind.startswith("rich") else d["base_c"][:, :10]
    c1 = d["alt_c"] if kind.startswith("rich") else d["alt_c"][:, :10]
    raw = np.concatenate((c0[train_ix], c1[train_ix]))
    center = raw.mean(0)
    scale = np.maximum(raw.std(0), .05)
    c0, c1 = (c0 - center) / scale, (c1 - center) / scale
    index = d["state_index"]
    base_s, base_f, alt_s, alt_f = (d[k] for k in ("base_s", "base_f", "alt_s", "alt_f"))
    flip = np.zeros(len(rows), np.float32)
    for i in range(len(rows)):
        base_yes = base_s[i] + base_f[i] >= 16 and base_f[i] <= 1
        alt_yes = alt_s[i] + alt_f[i] >= 16 and alt_f[i] <= 1
        flip[i] = 1 if alt_yes and base_f[i] >= 2 else -1 if base_yes and alt_f[i] >= 2 else 0
    model = Critic(False)
    params = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, c0.shape[1])))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4))
    optim_state = optimizer.init(params)

    @jax.jit
    def step(p, os, xx, ee, aa, bb, bs, bf, ats, atf, yy):
        def objective(pp):
            z0 = model.apply(pp, xx, ee, aa)
            z1 = model.apply(pp, xx, ee, bb)
            nll = jnp.mean((bs * jax.nn.softplus(-z0) + bf * jax.nn.softplus(z0) +
                            ats * jax.nn.softplus(-z1) + atf * jax.nn.softplus(z1)) / 16.)
            contrast = (jnp.sum(jnp.where(yy != 0, jax.nn.softplus(-yy * (z1 - z0)), 0.)) /
                        jnp.maximum(jnp.sum(yy != 0), 1.))
            return nll + contrast
        value, grads = jax.value_and_grad(objective)(p)
        updates, os = optimizer.update(grads, os, p)
        return optax.apply_updates(p, updates), os, value

    predict = jax.jit(lambda p, xx, ee, cc: model.apply(p, xx, ee, cc))
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    stale = 0
    for it in range(1, 2501):
        ix = np.concatenate([rng.choice(scenes[sc]["train"], 32) for sc in SCENES])
        params, optim_state, _ = step(params, optim_state, gather(x, index[ix]), jnp.asarray(eta[ix]),
                                      jnp.asarray(c0[ix]), jnp.asarray(c1[ix]),
                                      jnp.asarray(base_s[ix]), jnp.asarray(base_f[ix]),
                                      jnp.asarray(alt_s[ix]), jnp.asarray(alt_f[ix]), jnp.asarray(flip[ix]))
        if it % 100:
            continue
        stats = []
        for sc in SCENES:
            jj = scenes[sc]["validation"]
            z0 = np.asarray(predict(params, gather(x, index[jj]), jnp.asarray(eta[jj]), jnp.asarray(c0[jj])))
            z1 = np.asarray(predict(params, gather(x, index[jj]), jnp.asarray(eta[jj]), jnp.asarray(c1[jj])))
            nll = ((base_s[jj] * np.logaddexp(0, -z0) + base_f[jj] * np.logaddexp(0, z0) +
                    alt_s[jj] * np.logaddexp(0, -z1) + alt_f[jj] * np.logaddexp(0, z1)).sum() /
                   max(1, (base_s[jj] + base_f[jj] + alt_s[jj] + alt_f[jj]).sum()))
            have = flip[jj] != 0
            contrast = float(np.logaddexp(0, -flip[jj][have] * (z1[have] - z0[have])).mean()) if have.any() else 0.
            stats.append((nll, contrast))
        score = float(np.mean([a + .3 * b for a, b in stats]))
        if score < best[0] - 1e-5:
            best = (score, serialization.to_bytes(params), it)
            stale = 0
        else:
            stale += 1
        if stale >= 8 and it >= 800:
            break
    params = serialization.from_bytes(model.init(jax.random.PRNGKey(seed), gather(x, [0]),
                                                  jnp.zeros((1, 3)), jnp.zeros((1, c0.shape[1]))), best[1])
    details = []
    for sc in SCENES:
        jj = scenes[sc]["validation"]
        z0 = np.asarray(predict(params, gather(x, index[jj]), jnp.asarray(eta[jj]), jnp.asarray(c0[jj])))
        z1 = np.asarray(predict(params, gather(x, index[jj]), jnp.asarray(eta[jj]), jnp.asarray(c1[jj])))
        strong = flip[jj] != 0
        details.append({"scene": sc, "validation_pairs": len(jj), "B15_flips": int(strong.sum()),
                        "flip_direction_correct": int((np.sign(z1[strong] - z0[strong]) == flip[jj][strong]).sum()),
                        "median_abs_prediction_change": float(np.median(np.abs(jax.nn.sigmoid(z1) - jax.nn.sigmoid(z0))))})
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "result.json", {"kind": kind, "seed": seed, "best_step": best[2],
                                 "source_val_selection_score": best[0], "scenes": details,
                                 "target_K16_labels_used": False,
                                 "architecture": "same physical encoder/eta encoder/structured full critic; context dimension differs"})
    print(json.dumps({"kind": kind, "seed": seed, "score": best[0], "scenes": details}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize", "materialize_h20", "train"))
    parser.add_argument("--kind", choices=("nominal", "rich", "nominal20", "rich20"))
    parser.add_argument("--seed", type=int, choices=(17, 23, 41))
    args = parser.parse_args()
    if args.action == "materialize": materialize()
    elif args.action == "materialize_h20": materialize_h20()
    else: train(args.kind, args.seed)
