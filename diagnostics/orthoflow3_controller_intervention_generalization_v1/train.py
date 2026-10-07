"""Source-only LOSO critic with genuine controller interventions.

The context is H8 nominal Flow+safety response.  Canonical historical partial
counts and matched alternate-controller Q16 evidence are both observed-count
NLL supervision.  No target scene outcome is used before model freeze.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from . import intervention as base
from .rich_context import key

ROOT, OUT, OLD = base.ROOT, base.OUT, base.OLD
BAL = OUT / "balanced_expansion"
FOLDS = {"toy": "toy_giveway", "db": "double_bottleneck",
         "four": "four_way_intersection", "ring": "ring_exchange"}
SEEDS = (17, 23, 41)
KINDS = ("additive", "structured_full")
jax.config.update("jax_default_matmul_precision", "highest")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature(scene, state_uid, eta, alternate_sha):
    path = OUT / "rich_response_cache" / f"{key(scene, state_uid, eta, alternate_sha)}.json"
    row = read(path)
    if not row["valid"]:
        return None
    return row["features"]["mean"][:10]


def materialize():
    if (OUT / "training_dataset.npz").exists():
        return
    states = read(OLD / "states.json")
    old = pq.read_table(OLD / "all_pairs.parquet").to_pylist()
    normal = {}
    for path in sorted((OUT / "source_nominal").glob("*.json")):
        for r in read(path):
            assert r["valid"], (r["state_uid"], r["error"])
            normal[r["state_index"]] = r["context"]
    assert len(normal) == len(states)
    physical = [s["physical"] for s in states]
    rows = [{"state_index": r["state_index"], "state_uid": r["state_uid"],
             "family": states[r["state_index"]]["family"], "scene": r["scenario"],
             "split": r["split"], "eta": r["eta"], "eta_uid": r["eta_uid"],
             "controller_uid": r["controller_uid"], "role": "canonical_historical",
             "s": int(r["s"]), "f": int(r["f"]), "context": normal[r["state_index"]]}
            for r in old]
    protocol = read(BAL / "protocol.json")
    seen = {(r["state_uid"], r["eta_uid"], r["controller_uid"]) for r in rows}
    alt_rows = list(csv.DictReader((BAL / "intervention_pairs.csv").open()))
    for r in alt_rows:
        if int(r["alternate_valid"]) != 16:
            continue
        key_tuple = (r["state_uid"], r["eta_uid"], r["alternate_controller_uid"])
        assert key_tuple not in seen
        seen.add(key_tuple)
        eta = json.loads(r["eta"])
        context = feature(r["scene"], r["state_uid"], eta,
                          protocol["profiles"][r["scene"]]["alternate_flow_sha256"])
        if context is None:
            continue
        s = int(r["alternate_success"])
        rows.append({"state_index": int(r["state_index"]), "state_uid": r["state_uid"],
                     "family": r["family"], "scene": r["scene"], "split": r["split"],
                     "eta": eta, "eta_uid": r["eta_uid"],
                     "controller_uid": r["alternate_controller_uid"], "role": "future_flow_intervention",
                     "s": s, "f": 16 - s, "context": context})
    # Existing Toy matched Flow0/Flow1 counterfactuals add intervention
    # diversity to source folds without touching the frozen Toy K16 target.
    toy = ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1"
    toy_protocol = read(toy / "protocol.json")
    toy_pairs = list(csv.DictReader((toy / "controller_pair_results.csv").open()))
    toy_order = sorted({r["source_group"] for r in toy_pairs},
                       key=lambda f: hashlib.sha256(f"intervention-family-v1|{f}".encode()).hexdigest())
    toy_val = set(toy_order[:4])
    from diagnostics.orthoflow3_controller_context_loso_v1.context import Runtime
    import jax

    runtime = Runtime("toy_giveway")
    template = dict(next(s["physical"] for s in states if s["scenario"] == "toy_giveway"))
    by_uid = {}
    for r in toy_pairs:
        uid = r["state_uid"]
        if uid not in by_uid:
            env = runtime.make()
            env.reset(np.asarray(json.loads(r["initial_positions"])))
            flow = runtime.flow(env, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), int(r["rollout_id"])), 0))
            p = dict(template)
            p.update(positions=env.positions.tolist(), velocities=env.velocities.tolist(),
                     goals=env.goals.tolist(), flow=flow.tolist())
            by_uid[uid] = len(physical)
            physical.append(p)
        eta = json.loads(r["eta"])
        split = "validation" if r["source_group"] in toy_val else "train"
        for role, controller_uid, q, flow_sha in (
                ("toy_flow0_counterfactual_base", toy_protocol["base_controller_uid"],
                 int(r["base_success"]), None),
                ("future_flow_intervention", toy_protocol["alternate_controller_uid"],
                 int(r["alternate_success"]), toy_protocol["flow_sha256"]["1"])):
            context = feature("toy_giveway", uid, eta, flow_sha)
            if context is None:
                continue
            rows.append({"state_index": by_uid[uid], "state_uid": uid,
                         "family": r["source_group"], "scene": "toy_giveway", "split": split,
                         "eta": eta, "eta_uid": r["eta_uid"], "controller_uid": controller_uid,
                         "role": role, "s": q, "f": 16 - q, "context": context})
    x = rep.batch([rep.entities(p) for p in physical])
    previous = dict(np.load(OLD / "entities.npz"))
    for name in previous:
        np.testing.assert_allclose(x[name][:len(states)], previous[name], atol=2e-6, rtol=2e-6)
    np.savez_compressed(OUT / "training_dataset.npz", **x,
                        state_index=np.asarray([r["state_index"] for r in rows], np.int32),
                        eta=np.asarray([r["eta"] for r in rows], np.float32),
                        context=np.asarray([r["context"] for r in rows], np.float32),
                        s=np.asarray([r["s"] for r in rows], np.float32),
                        f=np.asarray([r["f"] for r in rows], np.float32))
    compact = [{k: v for k, v in r.items() if k not in ("eta", "context")}
               for r in rows]
    write(OUT / "training_rows.json", compact)
    split_audit = {scene: {sp: {role: sum(r["scene"] == scene and r["split"] == sp and r["role"] == role for r in rows)
                                  for role in sorted({r["role"] for r in rows})}
                            for sp in ("train", "validation")}
                   for scene in FOLDS.values()}
    write(OUT / "dataset_manifest.json",
          {"state_count": len(physical), "pair_count": len(rows), "split_counts": split_audit,
           "old_pairs_sha256": sha(OLD / "all_pairs.parquet"),
           "balanced_protocol_sha256": sha(BAL / "protocol.json"),
           "balanced_results_sha256": sha(BAL / "intervention_pairs.csv"),
           "toy_swap_results_sha256": sha(toy / "controller_pair_results.csv"),
           "target_scene_labels_used": False, "source_family_split_disjoint": True,
           "h_replay_matches_frozen_entities": True,
           "loss": "observed-count Bernoulli NLL; no unrun-seed imputation"})
    print(json.dumps({"states": len(physical), "pairs": len(rows), "roles": dict(__import__("collections").Counter(r["role"] for r in rows))}))


class Critic(nn.Module):
    additive: bool = False

    @nn.compact
    def __call__(self, x, eta, context):
        h = rep.Encoder(name="physical_encoder")(x)
        e = nn.silu(nn.Dense(32, name="eta_encoder")(eta))
        c = nn.silu(nn.Dense(32, name="context_encoder")(context))
        if self.additive:
            a = nn.silu(nn.Dense(128, name="difficulty1")(jnp.concatenate((h, c), -1)))
            a = nn.silu(nn.Dense(64, name="difficulty2")(a))
            a = nn.Dense(1, name="difficulty_out")(a)[..., 0]
            b = nn.silu(nn.Dense(64, name="eta_only1")(e))
            b = nn.Dense(1, name="eta_only_out")(b)[..., 0]
            return a + b
        prior = nn.silu(nn.Dense(128, name="prior1")(jnp.concatenate((e, c), -1)))
        prior = nn.silu(nn.Dense(64, name="prior2")(prior))
        prior = nn.Dense(1, name="prior_out")(prior)[..., 0]
        inter = nn.silu(nn.Dense(128, name="interaction1")(jnp.concatenate((h, e, c), -1)))
        inter = nn.silu(nn.Dense(64, name="interaction2")(inter))
        inter = nn.Dense(1, name="interaction_out", kernel_init=nn.initializers.zeros_init())(inter)[..., 0]
        return prior + inter


def fold_data(fold):
    npz = dict(np.load(OUT / "training_dataset.npz"))
    rows = read(OUT / "training_rows.json")
    source = [sc for key, sc in FOLDS.items() if key != fold]
    ids = np.asarray([i for i, r in enumerate(rows) if r["scene"] in source], int)
    mask = np.asarray([rows[i]["split"] == "train" for i in ids], bool)
    state = npz["state_index"][ids]
    context = npz["context"][ids]
    # Use one entry per state/controller rather than pair-count weighting.
    first = {}
    for j in np.flatnonzero(mask):
        first.setdefault((int(state[j]), rows[ids[j]]["controller_uid"]), j)
    train_context = context[list(first.values())]
    center = train_context.mean(0)
    scale = np.maximum(train_context.std(0), .05)
    context = (context - center) / scale
    norm = read(OLD / fold / "normalization.json")
    eta = (npz["eta"][ids] - np.asarray(norm["eta_center"], np.float32)) / np.asarray(norm["eta_radius"], np.float32)
    selection = {scene: {role: {sp: np.asarray([j for j, i in enumerate(ids)
                                                   if rows[i]["scene"] == scene and rows[i]["split"] == sp
                                                   and (rows[i]["role"] == "canonical_historical") == (role == "old")], int)
                          for sp in ("train", "validation")}
                  for role in ("old", "intervention")}
                 for scene in source}
    info = {"context_center": center.tolist(), "context_scale": scale.tolist(),
            "eta_center": norm["eta_center"], "eta_radius": norm["eta_radius"],
            "source_scenes": source, "target_scene": FOLDS[fold],
            "target_data_used": False}
    return npz, rows, ids, state, eta.astype(np.float32), context.astype(np.float32), selection, info


def train(fold, kind, seed):
    dest = OUT / "models" / fold / kind / f"seed{seed}"
    if (dest / "training.json").exists():
        return
    assert not (OUT / "target_predictions.json").exists(), "target predictions already opened"
    npz, rows, ids, state, eta, context, groups, info = fold_data(fold)
    model = Critic(additive=kind == "additive")
    x = {k: npz[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    params = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, 10)))
    optim = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4))
    state_opt = optim.init(params)
    success = npz["s"][ids]
    failure = npz["f"][ids]
    mean_n = {sc: float(np.mean(success[g["old"]["train"]] + failure[g["old"]["train"]]))
              for sc, g in groups.items()}
    weight = np.asarray([1. / mean_n[rows[i]["scene"]] for i in ids], np.float32)

    @jax.jit
    def step(params, state_opt, bx, be, bc, bs, bf, bw):
        def loss(p):
            z = model.apply(p, bx, be, bc)
            return jnp.mean((bs * jax.nn.softplus(-z) + bf * jax.nn.softplus(z)) * bw)
        value, grad = jax.value_and_grad(loss)(params)
        updates, state_opt = optim.update(grad, state_opt, params)
        return optax.apply_updates(params, updates), state_opt, value

    predict = jax.jit(lambda p, bx, be, bc: model.apply(p, bx, be, bc))
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    stale = 0
    history = []
    started = time.time()
    for it in range(1, 3501):
        ix = np.concatenate([rng.choice(groups[sc][role]["train"], 64 if role == "old" else 32)
                             for sc in groups for role in ("old", "intervention")])
        params, state_opt, value = step(params, state_opt, gather(x, state[ix]),
                                        jnp.asarray(eta[ix]), jnp.asarray(context[ix]),
                                        jnp.asarray(success[ix]), jnp.asarray(failure[ix]),
                                        jnp.asarray(weight[ix]))
        if it % 100:
            continue
        validation = {}
        for sc in groups:
            validation[sc] = {}
            for role in ("old", "intervention"):
                jj = groups[sc][role]["validation"]
                z = np.concatenate([np.asarray(predict(params, gather(x, state[q]),
                                                      jnp.asarray(eta[q]), jnp.asarray(context[q])))
                                    for q in (jj[k:k + 256] for k in range(0, len(jj), 256))])
                loss = success[jj] * np.logaddexp(0, -z) + failure[jj] * np.logaddexp(0, z)
                validation[sc][role] = float(loss.sum() / max(1, (success[jj] + failure[jj]).sum()))
        score = float(np.mean([.5 * d["old"] + .5 * d["intervention"] for d in validation.values()]))
        history.append({"step": it, "training_objective": float(value), "validation": validation, "score": score})
        if score < best[0] - 1e-5:
            best = (score, serialization.to_bytes(params), it)
            stale = 0
        else:
            stale += 1
        if it % 500 == 0:
            print(json.dumps({"fold": fold, "kind": kind, "seed": seed, "step": it,
                              "source_val": score, "seconds": time.time() - started}), flush=True)
        if stale >= 10 and it >= 1500:
            break
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "history.json", history)
    write(dest / "normalization.json", info)
    write(dest / "training.json", {"fold": fold, "kind": kind, "seed": seed,
                                   "best_source_val": best[0], "best_step": best[2],
                                   "steps": it, "seconds": time.time() - started,
                                   "checkpoint_sha256": sha(dest / "checkpoint.msgpack"),
                                   "dataset_sha256": sha(OUT / "training_dataset.npz"),
                                   "target_labels_used": False,
                                   "mean_observed_trials_by_scene": mean_n,
                                   "sampling": "per-source-scene batch: 64 canonical historical +32 intervention",
                                   "checkpoint_selection": "source VAL mean of 0.5 canonical NLL +0.5 intervention NLL"})


def freeze():
    result = {}
    for fold in FOLDS:
        result[fold] = {}
        for kind in KINDS:
            runs = [read(OUT / "models" / fold / kind / f"seed{s}" / "training.json") for s in SEEDS]
            result[fold][kind] = {"runs": runs,
                                  "selected_seed": min(runs, key=lambda r: (r["best_source_val"], r["seed"]))["seed"]}
    write(OUT / "models_frozen.json", {"folds": result,
                                        "selection": "source family VAL only",
                                        "target_labels_used": False,
                                        "dataset_sha256": sha(OUT / "training_dataset.npz")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize", "train", "freeze"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "materialize":
        materialize()
    elif args.action == "freeze":
        freeze()
    else:
        lock = OUT / "models" / args.fold / args.kind / f"seed{args.seed}" / "run.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            train(args.fold, args.kind, args.seed)
