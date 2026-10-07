"""Second, source-only correction: matched-controller contrast supervision.

The architecture and ordinary observed-count NLL are unchanged. A confirmed
B15 switch for the SAME (h,eta) adds a logistic controller-order constraint.
This is not eta-ranking loss and uses no target-scene labels.
"""
import argparse
import fcntl
import json
import time
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .train import Critic, FOLDS, OUT, OLD, SEEDS, fold_data, read, sha, write

KIND = "contrast_full"
LAMBDA = 1.0


def contrast_pairs(rows, ids, scene, split):
    groups = defaultdict(dict)
    for j, i in enumerate(ids):
        r = rows[i]
        if r["scene"] == scene and r["split"] == split:
            groups[(r["state_uid"], r["eta_uid"])][r["role"]] = j
    pairs = []
    for group in groups.values():
        alt = group.get("future_flow_intervention")
        canonical = group.get("canonical_historical", group.get("toy_flow0_counterfactual_base"))
        if alt is None or canonical is None:
            continue
        a, b = rows[ids[alt]], rows[ids[canonical]]
        alt_b15, base_b15 = a["s"] + a["f"] >= 16 and a["f"] <= 1, b["s"] + b["f"] >= 16 and b["f"] <= 1
        alt_negative, base_negative = a["f"] >= 2, b["f"] >= 2
        if alt_b15 and base_negative:
            pairs.append((canonical, alt, 1.))
        elif base_b15 and alt_negative:
            pairs.append((canonical, alt, -1.))
    return np.asarray(pairs, np.float32).reshape(-1, 3)


def train(fold, seed):
    dest = OUT / "models" / fold / KIND / f"seed{seed}"
    if (dest / "training.json").exists():
        return
    assert not (OUT / "target_predictions.json").exists()
    npz, rows, ids, state, eta, context, groups, info = fold_data(fold)
    x = {k: npz[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    s, f = npz["s"][ids], npz["f"][ids]
    source = list(groups)
    matched = {sc: {sp: contrast_pairs(rows, ids, sc, sp) for sp in ("train", "validation")}
               for sc in source}
    active = [sc for sc in source if len(matched[sc]["train"]) > 0]
    assert active
    model = Critic(False)
    params = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, 10)))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4))
    optimizer_state = optimizer.init(params)
    mean_n = {sc: float(np.mean(s[g["old"]["train"]] + f[g["old"]["train"]])) for sc, g in groups.items()}
    weights = np.asarray([1. / mean_n[rows[i]["scene"]] for i in ids], np.float32)

    @jax.jit
    def step(p, opt_state, bx, be, bc, bs, bf, bw, px, pe, pc0, pc1, py):
        def loss(q):
            z = model.apply(q, bx, be, bc)
            nll = jnp.mean((bs * jax.nn.softplus(-z) + bf * jax.nn.softplus(z)) * bw)
            z0 = model.apply(q, px, pe, pc0)
            z1 = model.apply(q, px, pe, pc1)
            contrast = jnp.mean(jax.nn.softplus(-py * (z1 - z0)))
            return nll + LAMBDA * contrast, (nll, contrast)
        (value, details), grad = jax.value_and_grad(loss, has_aux=True)(p)
        update, opt_state = optimizer.update(grad, opt_state, p)
        return optax.apply_updates(p, update), opt_state, value, details

    predict = jax.jit(lambda p, bx, be, bc: model.apply(p, bx, be, bc))
    rng = np.random.default_rng(seed)
    rng_contrast = np.random.default_rng(seed + 99001)
    history = []
    best = (float("inf"), None, 0)
    stale = 0
    started = time.time()
    for it in range(1, 3501):
        ix = np.concatenate([rng.choice(groups[sc][role]["train"], 64 if role == "old" else 32)
                             for sc in source for role in ("old", "intervention")])
        sampled = np.concatenate([matched[sc]["train"][rng_contrast.choice(len(matched[sc]["train"]), 16)]
                                  for sc in active])
        a = sampled[:, 0].astype(int)
        b = sampled[:, 1].astype(int)
        y = sampled[:, 2]
        params, optimizer_state, value, details = step(
            params, optimizer_state, gather(x, state[ix]), jnp.asarray(eta[ix]), jnp.asarray(context[ix]),
            jnp.asarray(s[ix]), jnp.asarray(f[ix]), jnp.asarray(weights[ix]),
            gather(x, state[a]), jnp.asarray(eta[a]), jnp.asarray(context[a]),
            jnp.asarray(context[b]), jnp.asarray(y))
        if it % 100:
            continue
        validation = {}
        for sc in source:
            validation[sc] = {}
            for role in ("old", "intervention"):
                jj = groups[sc][role]["validation"]
                z = np.concatenate([np.asarray(predict(params, gather(x, state[q]),
                                                      jnp.asarray(eta[q]), jnp.asarray(context[q])))
                                    for q in (jj[k:k + 256] for k in range(0, len(jj), 256))])
                validation[sc][role] = float((s[jj] * np.logaddexp(0, -z) + f[jj] * np.logaddexp(0, z)).sum() /
                                             max(1, (s[jj] + f[jj]).sum()))
            contrast = matched[sc]["validation"]
            if len(contrast):
                aa = contrast[:, 0].astype(int);bb = contrast[:, 1].astype(int);yy = contrast[:, 2]
                z0 = np.asarray(predict(params, gather(x, state[aa]), jnp.asarray(eta[aa]), jnp.asarray(context[aa])))
                z1 = np.asarray(predict(params, gather(x, state[bb]), jnp.asarray(eta[bb]), jnp.asarray(context[bb])))
                validation[sc]["contrast"] = float(np.logaddexp(0, -yy * (z1 - z0)).mean())
            else:
                validation[sc]["contrast"] = None
        nll = float(np.mean([.5 * v["old"] + .5 * v["intervention"] for v in validation.values()]))
        possible = [v["contrast"] for v in validation.values() if v["contrast"] is not None]
        contrast_score = float(np.mean(possible))
        score = nll + .3 * contrast_score
        history.append({"step": it, "training_objective": float(value),
                        "nll_train": float(details[0]), "contrast_train": float(details[1]),
                        "source_val_nll": nll, "source_val_contrast": contrast_score,
                        "selection_score": score, "source_scenes": validation})
        if score < best[0] - 1e-5:
            best = (score, serialization.to_bytes(params), it)
            stale = 0
        else:
            stale += 1
        if it % 500 == 0:
            print(json.dumps({"fold": fold, "kind": KIND, "seed": seed, "step": it,
                              "source_val_nll": nll, "source_val_contrast": contrast_score,
                              "seconds": time.time() - started}), flush=True)
        if stale >= 10 and it >= 1500:
            break
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "history.json", history)
    write(dest / "normalization.json", info)
    write(dest / "training.json", {"fold": fold, "kind": KIND, "seed": seed,
                                   "best_source_val": best[0], "best_step": best[2], "steps": it,
                                   "seconds": time.time() - started,
                                   "checkpoint_sha256": sha(dest / "checkpoint.msgpack"),
                                   "dataset_sha256": sha(OUT / "training_dataset.npz"),
                                   "target_labels_used": False,
                                   "contrast_lambda_train": LAMBDA,
                                   "contrast_pairs_train": {sc: len(matched[sc]["train"]) for sc in source},
                                   "contrast_pairs_val": {sc: len(matched[sc]["validation"]) for sc in source},
                                   "selection": "source VAL NLL +0.3 matched-controller contrast"})


def freeze():
    path = OUT / "models_frozen.json"
    data = read(path)
    assert not (OUT / "target_predictions.json").exists()
    for fold in FOLDS:
        runs = [read(OUT / "models" / fold / KIND / f"seed{s}" / "training.json") for s in SEEDS]
        data["folds"][fold][KIND] = {"runs": runs,
                                      "selected_seed": min(runs, key=lambda r: (r["best_source_val"], r["seed"]))["seed"]}
    data["source_only_contrast_branch_frozen"] = True
    write(path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "freeze"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        lock = OUT / "models" / args.fold / KIND / f"seed{args.seed}" / "run.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            train(args.fold, args.seed)
