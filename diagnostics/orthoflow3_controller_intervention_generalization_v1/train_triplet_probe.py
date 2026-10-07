"""Matched three-controller, source-family-heldout H20 critic probe.

This preserves the same physical encoder and pure observed-count NLL plus
matched-controller contrast used by the two-condition source probe. The only
new supervision is the compatible third Flow condition on TRAIN families.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
import flax.linen as nn

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from . import h20_features as h20
from .rich_intervention_probe import SCENES
from .second_variant import OUT as SECOND
from .train import Critic, OUT, read, write


class BoundedResidualCritic(nn.Module):
    max_logit_residual: float

    @nn.compact
    def __call__(self, x, eta, context):
        h = rep.Encoder(name="physical_encoder")(x)
        e = nn.silu(nn.Dense(32, name="eta_encoder")(eta))
        nominal = nn.silu(nn.Dense(32, name="nominal_context_encoder")(context[:, :10]))
        response = nn.silu(nn.Dense(32, name="eta_response_encoder")(context[:, 10:]))
        difficulty = nn.silu(nn.Dense(128, name="difficulty1")(jnp.concatenate((h, nominal), -1)))
        difficulty = nn.silu(nn.Dense(64, name="difficulty2")(difficulty))
        difficulty = nn.Dense(1, name="difficulty_out")(difficulty)[..., 0]
        prior = nn.silu(nn.Dense(64, name="eta_prior1")(e))
        prior = nn.Dense(1, name="eta_prior_out")(prior)[..., 0]
        residual = nn.silu(nn.Dense(128, name="interaction1")(jnp.concatenate((h, e, nominal, response), -1)))
        residual = nn.silu(nn.Dense(64, name="interaction2")(residual))
        residual = nn.Dense(1, name="interaction_out", kernel_init=nn.initializers.zeros_init())(residual)[..., 0]
        return difficulty + prior + self.max_logit_residual * jnp.tanh(residual)


def make_model(kind):
    if kind in ("bounded1", "bounded3", "bounded3_interaction"):
        return BoundedResidualCritic(1. if kind == "bounded1" else 3.)
    return Critic(kind == "additive")


def reversal_quadruplets(rows, s, f, split):
    by = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == split:
            by[(row["scene"], row["state_uid"])].append(i)
    result = []
    for _, indices in by.items():
        for i, j in combinations(indices, 2):
            signs, margins = [], []
            for sj, fj in zip(s, f):
                lo = (float(sj[i]) - (16 - float(fj[j]))) / 16
                hi = ((16 - float(fj[i])) - float(sj[j])) / 16
                signs.append(1 if lo > 0 else -1 if hi < 0 else 0)
                margins.append(max(lo, -hi))
            for a, b in combinations(range(3), 2):
                if signs[a] * signs[b] < 0 and max(margins[a], margins[b]) >= .25:
                    result.append((i, j, a, b, signs[a], signs[b]))
    return np.asarray(result, np.int32).reshape(-1, 6)


def materialize():
    import csv

    target = SECOND / "triplet_dataset_h20.npz"
    if target.exists():
        return
    data = dict(np.load(OUT / "rich_probe_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    exact = {(r["scene"], r["state_uid"], tuple(np.asarray(r["eta"], np.float32))): r
             for r in read(SECOND / "pairs.json")}
    outcome = {(r["scene"], r["state_uid"], r["eta_uid"]): r
               for r in csv.DictReader((SECOND / "intervention_pairs.csv").open())}
    profiles = read(SECOND / "protocol.json")["profiles"]
    context, successes, failures, full = [], [], [], []
    for i, row in enumerate(rows):
        if row["scene"] == "toy_giveway":
            context.append(data["base_c"][i]);successes.append(0);failures.append(0);full.append(False)
            continue
        pair = exact[(row["scene"], row["state_uid"], tuple(data["eta"][i]))]
        result = outcome[(row["scene"], row["state_uid"], pair["eta_uid"])]
        path = h20.rc.OUT / "rich_response_cache" / f"{h20.rc.key(row['scene'], row['state_uid'], pair['eta'], profiles[row['scene']]['alternate_flow_sha256'])}.json"
        response = read(path)
        assert response["valid"], (row, response.get("error"))
        n = int(result["second_valid"])
        s = int(result["second_success"])
        context.append(response["features"]["mean"])
        successes.append(s);failures.append(n - s);full.append(n == 16)
    data["second_c"] = np.asarray(context, np.float32)
    data["second_s"] = np.asarray(successes, np.float32)
    data["second_f"] = np.asarray(failures, np.float32)
    data["second_full"] = np.asarray(full, bool)
    np.savez_compressed(target, **data)
    write(SECOND / "triplet_dataset_audit.json",
          {"pairs": len(rows), "second_controller_pairs": int(np.sum(data["second_s"] + data["second_f"] > 0)),
           "second_complete_Q16": int(data["second_full"].sum()),
           "numerical_unresolved_kept_as_partial_observed_counts": True,
           "frozen_target_labels_used": False})


def train(seed, kind="full"):
    folder = {"full": "triplet_models", "additive": "triplet_nominal_additive_models",
              "bounded1": "triplet_bounded1_models", "bounded3": "triplet_bounded3_models",
              "bounded3_interaction": "triplet_bounded3_interaction_models"}[kind]
    dest = SECOND / folder / f"seed{seed}"
    if (dest / "result.json").exists():
        return
    data = dict(np.load(SECOND / "triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    x = {k: data[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    scene_ix = {sc: {sp: np.asarray([i for i, r in enumerate(rows) if r["scene"] == sc and r["split"] == sp], int)
                     for sp in ("train", "validation")} for sc in SCENES}
    train_ix = np.concatenate([scene_ix[sc]["train"] for sc in SCENES])
    eta_center = data["eta"][train_ix].mean(0)
    eta_scale = np.maximum(data["eta"][train_ix].std(0), .1)
    eta = (data["eta"] - eta_center) / eta_scale
    width = 10 if kind == "additive" else 24
    context_stack = np.concatenate([data["base_c"][train_ix, :width], data["alt_c"][train_ix, :width],
                                   data["second_c"][train_ix][data["second_full"][train_ix], :width]])
    center = context_stack.mean(0)
    scale = np.maximum(context_stack.std(0), .05)
    contexts = [(data[k][:, :width] - center) / scale for k in ("base_c", "alt_c", "second_c")]
    s = [data[k] for k in ("base_s", "alt_s", "second_s")]
    f = [data[k] for k in ("base_f", "alt_f", "second_f")]
    known = [(ss + ff) >= 16 for ss, ff in zip(s, f)]
    yes = [known[j] & (f[j] <= 1) for j in range(3)]
    no = [f[j] >= 2 for j in range(3)]
    flip = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        flip.append((yes[b] & no[a]).astype(np.float32) - (yes[a] & no[b]).astype(np.float32))
    use_interaction = kind == "bounded3_interaction"
    train_quad = reversal_quadruplets(rows, s, f, "train") if use_interaction else np.zeros((0, 6), np.int32)
    val_quad = reversal_quadruplets(rows, s, f, "validation") if use_interaction else np.zeros((0, 6), np.int32)
    if use_interaction:
        assert len(train_quad) >= 30 and len(val_quad) >= 5
    model = make_model(kind)
    params = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, width)))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, os, xx, ee, c0, c1, c2, s0, f0, s1, f1, s2, f2, y01, y02, y12,
             qx, qe_i, qe_j, qc_ai, qc_aj, qc_bi, qc_bj, qsa, qsb):
        def objective(pp):
            z = [model.apply(pp, xx, ee, c) for c in (c0, c1, c2)]
            nll = jnp.mean(sum(sj * jax.nn.softplus(-zj) + fj * jax.nn.softplus(zj)
                                for sj, fj, zj in zip((s0, s1, s2), (f0, f1, f2), z)) / 16)
            contrast = 0.
            for a, b, y in ((0, 1, y01), (0, 2, y02), (1, 2, y12)):
                contrast += jnp.sum(jnp.where(y != 0, jax.nn.softplus(-y * (z[b] - z[a])), 0.)) / jnp.maximum(jnp.sum(y != 0), 1.)
            if use_interaction:
                zai = model.apply(pp, qx, qe_i, qc_ai)
                zaj = model.apply(pp, qx, qe_j, qc_aj)
                zbi = model.apply(pp, qx, qe_i, qc_bi)
                zbj = model.apply(pp, qx, qe_j, qc_bj)
                interaction = jnp.mean(jax.nn.softplus(-qsa * (zai - zaj)) +
                                       jax.nn.softplus(-qsb * (zbi - zbj)))
                return nll + contrast + interaction
            return nll + contrast
        loss, grads = jax.value_and_grad(objective)(p)
        updates, os = optimizer.update(grads, os, p)
        return optax.apply_updates(p, updates), os, loss

    pred = jax.jit(lambda pp, xx, ee, cc: model.apply(pp, xx, ee, cc))
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, 0)
    stale = 0
    for it in range(1, 2501):
        ix = np.concatenate([rng.choice(scene_ix[sc]["train"], 32) for sc in SCENES])
        if use_interaction:
            quad = train_quad[rng.choice(len(train_quad), 32)]
            qi, qj, qa, qb, qsa, qsb = quad.T
            pick = lambda condition, index: np.stack([contexts[int(c)][int(k)] for c, k in zip(condition, index)])
            q_input = (gather(x, data["state_index"][qi]), jnp.asarray(eta[qi]), jnp.asarray(eta[qj]),
                       jnp.asarray(pick(qa, qi)), jnp.asarray(pick(qa, qj)),
                       jnp.asarray(pick(qb, qi)), jnp.asarray(pick(qb, qj)),
                       jnp.asarray(qsa, jnp.float32), jnp.asarray(qsb, jnp.float32))
        else:
            q_input = (gather(x, data["state_index"][ix[:1]]),
                       *[jnp.zeros((1, 3), jnp.float32) for _ in range(2)],
                       *[jnp.zeros((1, width), jnp.float32) for _ in range(4)],
                       *[jnp.zeros((1,), jnp.float32) for _ in range(2)])
        params, opt_state, _ = step(params, opt_state, gather(x, data["state_index"][ix]),
                                    jnp.asarray(eta[ix]), *[jnp.asarray(c[ix]) for c in contexts],
                                    *[jnp.asarray(v[ix]) for pair in zip(s, f) for v in pair],
                                    *[jnp.asarray(y[ix]) for y in flip], *q_input)
        if it % 100:
            continue
        scores = []
        for sc in SCENES:
            ixv = scene_ix[sc]["validation"]
            zz = [np.asarray(pred(params, gather(x, data["state_index"][ixv]),
                                  jnp.asarray(eta[ixv]), jnp.asarray(c[ixv]))) for c in contexts]
            nll = sum((s[j][ixv] * np.logaddexp(0, -zz[j]) + f[j][ixv] * np.logaddexp(0, zz[j])).sum()
                      for j in range(3)) / max(1, sum((s[j][ixv] + f[j][ixv]).sum() for j in range(3)))
            contrast = []
            for (a, b), y in zip(((0, 1), (0, 2), (1, 2)), flip):
                active = y[ixv] != 0
                if active.any():
                    contrast.append(float(np.logaddexp(0, -y[ixv][active] * (zz[b][active] - zz[a][active])).mean()))
            scores.append(float(nll + .3 * np.mean(contrast)) if contrast else float(nll))
        score = float(np.mean(scores))
        if use_interaction:
            qa, qb = val_quad[:, 2], val_quad[:, 3]
            vi, vj = val_quad[:, 0], val_quad[:, 1]
            pick = lambda condition, index: np.stack([contexts[int(c)][int(k)] for c, k in zip(condition, index)])
            xxv = gather(x, data["state_index"][vi])
            za_i = np.asarray(pred(params, xxv, jnp.asarray(eta[vi]), jnp.asarray(pick(qa, vi))))
            za_j = np.asarray(pred(params, xxv, jnp.asarray(eta[vj]), jnp.asarray(pick(qa, vj))))
            zb_i = np.asarray(pred(params, xxv, jnp.asarray(eta[vi]), jnp.asarray(pick(qb, vi))))
            zb_j = np.asarray(pred(params, xxv, jnp.asarray(eta[vj]), jnp.asarray(pick(qb, vj))))
            reversal_loss = np.mean(np.logaddexp(0, -val_quad[:, 4] * (za_i - za_j)) +
                                    np.logaddexp(0, -val_quad[:, 5] * (zb_i - zb_j)))
            score += .3 * float(reversal_loss)
        if score < best[0] - 1e-5:
            best = (score, serialization.to_bytes(params), it)
            stale = 0
        else:
            stale += 1
        if stale >= 8 and it >= 800:
            break
    params = serialization.from_bytes(params, best[1])
    details = []
    for sc in SCENES:
        ix = scene_ix[sc]["validation"]
        zz = [np.asarray(pred(params, gather(x, data["state_index"][ix]),
                              jnp.asarray(eta[ix]), jnp.asarray(c[ix]))) for c in contexts]
        condition = []
        for j in range(3):
            observed = s[j][ix] + f[j][ix]
            if observed.sum() == 0:
                continue
            condition.append({"controller": j, "observed_trials": int(observed.sum()),
                              "NLL": float((s[j][ix] * np.logaddexp(0, -zz[j]) + f[j][ix] * np.logaddexp(0, zz[j])).sum() / observed.sum())})
        changes = []
        for (a, b), y in zip(((0, 1), (0, 2), (1, 2)), flip):
            active = y[ix] != 0
            changes.append({"comparison": f"{a}->{b}", "B15_flips": int(active.sum()),
                            "correct_direction": int((np.sign(zz[b][active] - zz[a][active]) == y[ix][active]).sum()),
                            "median_abs_probability_change": float(np.median(np.abs(jax.nn.sigmoid(zz[b]) - jax.nn.sigmoid(zz[a]))))})
        details.append({"scene": sc, "conditions": condition, "changes": changes})
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "result.json", {"seed": seed, "kind": kind, "best_step": best[2],
                                 "source_val_selection_score": best[0], "scenes": details,
                                 "architecture": {"full": "24D H20 eta-conditioned structured full critic",
                                                  "additive": "10D H20 eta-independent A(h,C(h))+B(eta) additive critic",
                                                  "bounded1": "A(h,C_nom)+B(eta)+1*tanh(interaction(h,eta,C_rich))",
                                                  "bounded3": "A(h,C_nom)+B(eta)+3*tanh(interaction(h,eta,C_rich))",
                                                  "bounded3_interaction": "same bounded3 plus source-only certified controller eta-order reversal constraint"}[kind],
                                 "observed_count_NLL": True, "matched_controller_contrast": True,
                                 "source_family_heldout": True, "target_labels_used": False,
                                 "train_certified_controller_rank_reversals": int(len(train_quad)),
                                 "val_certified_controller_rank_reversals": int(len(val_quad))})
    print(json.dumps({"seed": seed, "score": best[0], "details": details}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize", "train"))
    parser.add_argument("--seed", type=int, choices=(17, 23, 41))
    parser.add_argument("--kind", choices=("full", "additive", "bounded1", "bounded3", "bounded3_interaction"), default="full")
    args = parser.parse_args()
    if args.action == "materialize": materialize()
    else: train(args.seed, args.kind)
