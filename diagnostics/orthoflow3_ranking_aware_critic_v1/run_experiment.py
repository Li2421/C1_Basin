#!/usr/bin/env python3
"""Ranking-aware continuous-Q critic experiment (no controller rollouts)."""
from __future__ import annotations

import csv
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
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

ROOT = Path("/home/zhihan/research/Basin_C1")
SRC = ROOT / "diagnostics/orthoflow3_structured_continuous_q_data_v1"
OLD = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1"
OUT = Path(__file__).resolve().parent
SEEDS = (17, 23, 41)
LAMBDAS = (0.1, 0.3, 1.0)


def dump(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(name, rows):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["empty"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stable_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SingleCritic(nn.Module):
    @nn.compact
    def __call__(self, h, eta):
        z = nn.silu(nn.Dense(128, name="state1")(h))
        z = nn.silu(nn.Dense(64, name="state2")(z))
        e = nn.silu(nn.Dense(32, name="eta1")(eta))
        x = jnp.concatenate([z, e], axis=-1)
        x = nn.silu(nn.Dense(128, name="c1")(x))
        x = nn.silu(nn.Dense(64, name="c2")(x))
        return nn.Dense(1, name="out")(x)[..., 0]


class DualCritic(nn.Module):
    @nn.compact
    def __call__(self, h, eta):
        z = nn.silu(nn.Dense(128, name="state1")(h))
        z = nn.silu(nn.Dense(64, name="state2")(z))
        e = nn.silu(nn.Dense(32, name="eta1")(eta))
        x = jnp.concatenate([z, e], axis=-1)
        x = nn.silu(nn.Dense(128, name="c1")(x))
        x = nn.silu(nn.Dense(64, name="c2")(x))
        return nn.Dense(1, name="prob_head")(x)[..., 0], nn.Dense(1, name="rank_head")(x)[..., 0]


def normalize(rows):
    man = json.load(open(OLD / "dataset_manifest.json"))
    sn = man["state_normalization"]["Toy"]
    ec = np.asarray(man["eta_normalization"]["center"], np.float32)
    es = np.asarray(man["eta_normalization"]["scale"], np.float32)
    h = np.asarray([r["h_raw"] for r in rows], np.float32)
    eta = np.asarray([r["eta"] for r in rows], np.float32)
    return {
        "rows": rows,
        "h": (h - np.asarray(sn["mean"], np.float32)) / np.asarray(sn["std"], np.float32),
        "eta": (eta - ec) / es,
        "y": np.asarray([r["empirical_q"] for r in rows], np.float32),
        "w": np.asarray([min(int(r["n_trials"]), 16) for r in rows], np.float32),
    }


def canonical_wide(row):
    z = dict(row)
    z["eta"] = [float(row["eta1"]), float(row["eta2"]), float(row["eta3"])]
    z["data_source"] = "wide"
    return z


def build_data():
    structured = pq.read_table(SRC / "structured_pair_table.parquet").to_pylist()
    train_struct = [dict(r, data_source="structured") for r in structured if r["matrix_partition"] == "TRAIN_TRAIN"]
    val = [dict(r, data_source="structured") for r in structured if r["matrix_partition"] == "VAL_VAL"]
    test = [dict(r, data_source="structured") for r in structured if r["matrix_partition"] == "TESTSTATE_TESTETA"]
    wide0 = [canonical_wide(r) for r in pq.read_table(SRC / "sparse_matched_control.parquet").to_pylist()]
    skeys = {(r["state_uid"], r["eta_uid"]) for r in train_struct}
    wide = [r for r in wide0 if (r["state_uid"], r["eta_uid"]) not in skeys]
    union = train_struct + wide
    assert len({(r["state_uid"], r["eta_uid"]) for r in union}) == len(union)
    test_eta = {r["eta_uid"] for r in test}
    assert not test_eta.intersection({r["eta_uid"] for r in union})
    return train_struct, wide, union, val, test


def make_pairs(rows, min_gap=0.25):
    by_state = defaultdict(list)
    for i, r in enumerate(rows):
        by_state[r["state_uid"]].append(i)
    pairs = []
    for sid, ix in sorted(by_state.items()):
        for ai in range(len(ix)):
            for bi in range(ai + 1, len(ix)):
                i, j = ix[ai], ix[bi]
                qi, qj = rows[i]["empirical_q"], rows[j]["empirical_q"]
                if abs(qi - qj) < min_gap:
                    continue
                if qi < qj:
                    i, j, qi, qj = j, i, qj, qi
                src = "structured" if rows[i]["data_source"] == rows[j]["data_source"] == "structured" else "wide_or_mixed"
                pairs.append({
                    "state_uid": sid,
                    "i": i,
                    "j": j,
                    "q_hi": qi,
                    "q_lo": qj,
                    "q_gap": qi - qj,
                    "strong_b15_failure": bool(qi >= 15 / 16 and qj <= 0.5),
                    "pair_source": src,
                })
    return pairs


def nll_np(p, y, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.sum(w * (-y * np.log(p) - (1 - y) * np.log(1 - p))) / np.sum(w))


def auc_ap(y, score):
    y = np.asarray(y, bool)
    score = np.asarray(score)
    pos, neg = y.sum(), (~y).sum()
    if not pos or not neg:
        return None, None
    ranks = rankdata(score)
    auc = (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
    order = np.argsort(-score)
    ys = y[order]
    precision = np.cumsum(ys) / (np.arange(len(ys)) + 1)
    return float(auc), float(np.sum(precision * ys) / pos)


def calibration_fit(p, y, w):
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def objective(ab):
        pred = sigmoid(ab[0] + ab[1] * x)
        return nll_np(pred, y, w)

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        use = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not use.any():
            continue
        ece += use.mean() * abs(float(p[use].mean() - y[use].mean()))
        rel.append({"lo": lo, "hi": hi, "n": int(use.sum()), "pred_mean": float(p[use].mean()), "q_mean": float(y[use].mean())})
    return float(fit.x[0]), float(fit.x[1]), float(ece), rel


def probability_metrics(p, data):
    y, w = data["y"], data["w"]
    rho = spearmanr(p, y).statistic
    b15 = y >= 15 / 16
    auc, ap = auc_ap(b15, p)
    pred_b15 = p >= 15 / 16
    intercept, slope, ece, rel = calibration_fit(p, y, w)
    return {
        "nll": nll_np(p, y, w),
        "mae": float(np.mean(abs(p - y))),
        "brier": float(np.mean((p - y) ** 2)),
        "spearman": float(rho),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece10": ece,
        "b15_auroc": auc,
        "b15_auprc": ap,
        "b15_precision": float(np.sum(pred_b15 & b15) / max(np.sum(pred_b15), 1)),
        "b15_recall": float(np.sum(pred_b15 & b15) / max(np.sum(b15), 1)),
        "reliability": rel,
    }


def ordering_metrics(scores, data, min_gap=0.25):
    by = defaultdict(list)
    for i, r in enumerate(data["rows"]):
        by[r["state_uid"]].append(i)
    correct = total = 0
    rhos = []
    for ix in by.values():
        if len(ix) > 1:
            rho = spearmanr(scores[ix], data["y"][ix]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if abs(data["y"][i] - data["y"][j]) < min_gap:
                    continue
                correct += int((scores[i] - scores[j]) * (data["y"][i] - data["y"][j]) > 0)
                total += 1
    return {"within_state_spearman": float(np.mean(rhos)), "pairwise_order_accuracy": correct / max(total, 1), "ordering_pairs": total}


def candidate_indices(data, K):
    by = defaultdict(list)
    for i, r in enumerate(data["rows"]):
        by[r["state_uid"]].append(i)
    result = {}
    for sid, ix in by.items():
        if len(ix) < K:
            continue
        e = data["eta"][ix]
        first = min(range(len(ix)), key=lambda j: stable_hash(f"rank-{K}|{data['rows'][ix[j]]['eta_uid']}"))
        selected = [first]
        dist = np.linalg.norm(e - e[first], axis=1)
        dist[first] = -1
        while len(selected) < K:
            j = int(np.argmax(dist))
            selected.append(j)
            dist = np.minimum(dist, np.linalg.norm(e - e[j], axis=1))
            dist[selected] = -1
        result[sid] = np.asarray(ix)[selected]
    return result


def ranking_metrics(scores, data, Ks=(8, 16, 32)):
    order = ordering_metrics(scores, data)
    out = []
    for K in Ks:
        vals = []
        for _, ix in candidate_indices(data, K).items():
            pred_order = ix[np.argsort(-scores[ix])]
            chosen = pred_order[0]
            oracle = ix[np.argmax(data["y"][ix])]
            has = bool(np.any(data["y"][ix] >= 15 / 16))
            vals.append({
                "chosen_q": data["y"][chosen],
                "oracle_q": data["y"][oracle],
                "has": has,
                "chosen_b15": bool(data["y"][chosen] >= 15 / 16),
                "top3": bool(np.any(data["y"][pred_order[:3]] >= 15 / 16)),
            })
        cover = [v for v in vals if v["has"]]
        out.append({
            "K": K,
            "states": len(vals),
            "mean_selected_q": float(np.mean([v["chosen_q"] for v in vals])),
            "mean_oracle_q": float(np.mean([v["oracle_q"] for v in vals])),
            "regret": float(np.mean([v["oracle_q"] - v["chosen_q"] for v in vals])),
            "b15_selection_rate": float(np.mean([v["chosen_b15"] for v in cover])),
            "top3_b15_hit_rate": float(np.mean([v["top3"] for v in cover])),
            **order,
        })
    return out


def threshold_then_rank(prob, score, data, tau, K):
    rows = []
    all_true, all_pred = [], []
    for _, ix in candidate_indices(data, K).items():
        passed = ix[prob[ix] >= tau]
        all_true.extend((data["y"][ix] >= 15 / 16).tolist())
        all_pred.extend((prob[ix] >= tau).tolist())
        oracle = float(np.max(data["y"][ix]))
        if not len(passed):
            rows.append({"has_candidate": False, "oracle": oracle})
            continue
        chosen = passed[np.argmax(score[passed])]
        rows.append({"has_candidate": True, "q": float(data["y"][chosen]), "oracle": oracle, "b15": bool(data["y"][chosen] >= 15 / 16)})
    pred = np.asarray(all_pred, bool)
    true = np.asarray(all_true, bool)
    selected = [r for r in rows if r["has_candidate"]]
    return {
        "tau": tau,
        "K": K,
        "states": len(rows),
        "candidate_pass_rate": float(pred.mean()),
        "true_b15_precision": float(np.sum(pred & true) / max(np.sum(pred), 1)),
        "true_b15_recall": float(np.sum(pred & true) / max(np.sum(true), 1)),
        "no_candidate_rate": float(np.mean([not r["has_candidate"] for r in rows])),
        "selected_true_q": float(np.mean([r["q"] for r in selected])) if selected else None,
        "selection_b15_rate": float(np.mean([r["b15"] for r in selected])) if selected else None,
        "conditional_regret": float(np.mean([r["oracle"] - r["q"] for r in selected])) if selected else None,
    }


def bce(logits, y, w):
    return jnp.sum(w * optax.sigmoid_binary_cross_entropy(logits, y)) / jnp.maximum(jnp.sum(w), 1)


def rank_loss(hi, lo, gap):
    return jnp.mean(gap * jax.nn.softplus(-(hi - lo)))


def predict(model, params, data, dual=False):
    out = model.apply(params, jnp.asarray(data["h"]), jnp.asarray(data["eta"]))
    if dual:
        a, b = out
        return sigmoid(np.asarray(a)), np.asarray(b)
    z = np.asarray(out)
    return sigmoid(z), z


def save_checkpoint(model_name, seed, tag, params, summary):
    d = OUT / model_name / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.msgpack").write_bytes(serialization.to_bytes(params))
    (d / f"{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def train_model(model_kind, seed, lam, structured, wide, union, val, train_pairs, val_pairs):
    dual = model_kind == "C"
    model = DualCritic() if dual else SingleCritic()
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, union["h"].shape[1])), jnp.zeros((1, 3)))
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    state = opt.init(params)

    if dual:
        @jax.jit
        def step(p, s, hs, es, ys, ws, hw, ew, yw, ww, hhi, ehi, hlo, elo, gap):
            def loss(q):
                ps, _ = model.apply(q, hs, es)
                pw, _ = model.apply(q, hw, ew)
                _, hi = model.apply(q, hhi, ehi)
                _, lo = model.apply(q, hlo, elo)
                return 0.5 * bce(ps, ys, ws) + 0.5 * bce(pw, yw, ww) + lam * rank_loss(hi, lo, gap)
            value, grad = jax.value_and_grad(loss)(p)
            update, s = opt.update(grad, s, p)
            return optax.apply_updates(p, update), s, value
    else:
        @jax.jit
        def step(p, s, hs, es, ys, ws, hw, ew, yw, ww, hhi, ehi, hlo, elo, gap):
            def loss(q):
                ps = model.apply(q, hs, es)
                pw = model.apply(q, hw, ew)
                hi = model.apply(q, hhi, ehi)
                lo = model.apply(q, hlo, elo)
                rank = rank_loss(hi, lo, gap) if model_kind in ("B", "D") else 0.0
                prob = 0.0 if model_kind == "D" else 0.5 * bce(ps, ys, ws) + 0.5 * bce(pw, yw, ww)
                return prob + lam * rank
            value, grad = jax.value_and_grad(loss)(p)
            update, s = opt.update(grad, s, p)
            return optax.apply_updates(p, update), s, value

    rng = np.random.default_rng(seed)
    sp = np.asarray([i for i, p in enumerate(train_pairs) if p["pair_source"] == "structured"])
    wp = np.asarray([i for i, p in enumerate(train_pairs) if p["pair_source"] != "structured"])
    best_prob = (np.inf, None, 0)
    best_rank = (np.inf, None, 0)
    stale = 0
    for it in range(1, 4001):
        si = rng.integers(0, len(structured["y"]), 64)
        wi = rng.integers(0, len(wide["y"]), 64)
        if len(sp) and len(wp):
            pi = np.concatenate([rng.choice(sp, 64, replace=True), rng.choice(wp, 64, replace=True)])
        else:
            pi = rng.integers(0, len(train_pairs), 128)
        hi = np.asarray([train_pairs[i]["i"] for i in pi])
        lo = np.asarray([train_pairs[i]["j"] for i in pi])
        gap = np.asarray([train_pairs[i]["q_gap"] for i in pi], np.float32)
        args = [
            jnp.asarray(structured[k][si]) for k in ("h", "eta", "y", "w")
        ] + [
            jnp.asarray(wide[k][wi]) for k in ("h", "eta", "y", "w")
        ] + [
            jnp.asarray(union["h"][hi]), jnp.asarray(union["eta"][hi]),
            jnp.asarray(union["h"][lo]), jnp.asarray(union["eta"][lo]), jnp.asarray(gap),
        ]
        params, state, _ = step(params, state, *args)
        if it % 50:
            continue
        pp, ss = predict(model, params, val, dual)
        pnll = nll_np(pp, val["y"], val["w"])
        om = ordering_metrics(ss, val)
        rscore = 1.0 - om["pairwise_order_accuracy"]
        if pnll < best_prob[0] - 1e-6:
            best_prob = (pnll, jax.tree_util.tree_map(np.asarray, params), it)
            stale = 0
        else:
            stale += 1
        if rscore < best_rank[0] - 1e-6:
            best_rank = (rscore, jax.tree_util.tree_map(np.asarray, params), it)
        if stale >= 30 and it >= 1000:
            break
    if model_kind == "D":
        best_prob = best_rank
    return model, best_prob, best_rank, {"seed": seed, "lambda": lam, "steps": it, "best_prob_val_nll": best_prob[0], "best_prob_step": best_prob[2], "best_rank_val_error": best_rank[0], "best_rank_step": best_rank[2]}


def mean_std(rows, keys):
    out = {}
    for k in keys:
        x = np.asarray([r[k] for r in rows], float)
        out[k + "_mean"] = float(x.mean())
        out[k + "_std"] = float(x.std())
        out[k + "_median"] = float(np.median(x))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_struct_rows, wide_rows, union_rows, val_rows, test_rows = build_data()
    struct = normalize(train_struct_rows)
    wide = normalize(wide_rows)
    union = normalize(union_rows)
    val = normalize(val_rows)
    test = normalize(test_rows)
    train_pairs = make_pairs(union_rows)
    val_pairs = make_pairs(val_rows)

    pair_audit = []
    for scope, pairs in (("train", train_pairs), ("val", val_pairs)):
        for src in sorted({p["pair_source"] for p in pairs}):
            z = [p for p in pairs if p["pair_source"] == src]
            pair_audit.append({"scope": scope, "pair_source": src, "pairs": len(z), "strong_b15_vs_failure": sum(p["strong_b15_failure"] for p in z), "mean_q_gap": float(np.mean([p["q_gap"] for p in z]))})
    write_csv("ranking_pair_audit.csv", pair_audit)

    manifest = {
        "task": "ORTHOFLOW3_RANKING_AWARE_CRITIC_V1",
        "new_rollout": 0,
        "structured_source": str(SRC / "structured_pair_table.parquet"),
        "wide_source": str(SRC / "sparse_matched_control.parquet"),
        "frozen_state_split": str(SRC / "toy_state_split.json"),
        "frozen_eta_split": str(SRC / "toy_eta_split.json"),
        "structured_train_pairs": len(train_struct_rows),
        "wide_rows_before_dedup": 1180,
        "wide_unique_added_pairs": len(wide_rows),
        "union_train_pairs": len(union_rows),
        "train_states": len({r["state_uid"] for r in union_rows}),
        "train_eta": len({r["eta_uid"] for r in union_rows}),
        "val_pairs": len(val_rows),
        "test_pairs": len(test_rows),
        "test_states": len({r["state_uid"] for r in test_rows}),
        "test_eta": len({r["eta_uid"] for r in test_rows}),
        "test_eta_overlap_train": len({r["eta_uid"] for r in test_rows}.intersection({r["eta_uid"] for r in union_rows})),
    }
    dump("dataset_manifest.json", manifest)
    dump("train_config.json", {"seeds": SEEDS, "lambdas": LAMBDAS, "rank_q_gap": 0.25, "n_eff_cap": 16, "probability_batch": {"structured": 64, "wide": 64}, "ranking_batch": {"structured": 64, "wide_or_mixed": 64}, "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "max_steps": 4000, "checkpoint_interval": 50})

    runs = []
    models = {}
    # A: probability-only. Lambda is zero.
    for seed in SEEDS:
        model, bp, br, summary = train_model("A", seed, 0.0, struct, wide, union, val, train_pairs, val_pairs)
        summary.update(model="A", variant="nll_only")
        runs.append(summary)
        models[("A", 0.0, seed)] = (model, bp, br, False)
        save_checkpoint("model_A_nll_only", seed, "best_probability", bp[1], summary)
    for kind, dirname in (("B", "model_B_single_head"), ("C", "model_C_dual_head")):
        for lam in LAMBDAS:
            for seed in SEEDS:
                model, bp, br, summary = train_model(kind, seed, lam, struct, wide, union, val, train_pairs, val_pairs)
                summary.update(model=kind, variant=f"lambda_{lam}")
                runs.append(summary)
                models[(kind, lam, seed)] = (model, bp, br, kind == "C")
                save_checkpoint(dirname, seed, f"lambda_{lam}_best_probability", bp[1], summary)
                save_checkpoint(dirname, seed, f"lambda_{lam}_best_ranking", br[1], summary)
    # D: independent ranker; probability is model A.
    for seed in SEEDS:
        model, bp, br, summary = train_model("D", seed, 1.0, struct, wide, union, val, train_pairs, val_pairs)
        summary.update(model="D", variant="rank_only")
        runs.append(summary)
        models[("D", 1.0, seed)] = (model, bp, br, False)
        save_checkpoint("model_D_separate_ranker", seed, "best_ranking", br[1], summary)

    # Select lambda globally across seeds using VAL only, separately by role.
    selected = {"A": {"prob_lambda": 0.0, "rank_lambda": 0.0}, "D": {"prob_lambda": 0.0, "rank_lambda": 1.0}}
    for kind in ("B", "C"):
        score = {}
        for lam in LAMBDAS:
            rr = [r for r in runs if r["model"] == kind and r["lambda"] == lam]
            score[lam] = {"prob": float(np.mean([r["best_prob_val_nll"] for r in rr])), "rank": float(np.mean([r["best_rank_val_error"] for r in rr]))}
        selected[kind] = {"prob_lambda": min(score, key=lambda x: score[x]["prob"]), "rank_lambda": min(score, key=lambda x: score[x]["rank"]), "val_scores": score}
    dump("selected_variants.json", selected)

    probability_rows, calibration_rows, ranking_rows, threshold_rows, stability = [], [], [], [], []
    cache = {}
    for kind in ("A", "B", "C", "D"):
        for seed in SEEDS:
            # Probability source: A for D, selected probability checkpoint otherwise.
            pkind = "A" if kind == "D" else kind
            plam = selected[pkind]["prob_lambda"]
            pm, pbp, _, pdual = models[(pkind, plam, seed)]
            prob, _ = predict(pm, pbp[1], test, pdual)
            # Ranking source/checkpoint.
            rlam = selected[kind]["rank_lambda"]
            rm, _, rbr, rdual = models[(kind, rlam, seed)]
            _, score = predict(rm, rbr[1], test, rdual)
            cache[(kind, seed)] = (prob, score)
            met = probability_metrics(prob, test)
            rel = met.pop("reliability")
            probability_rows.append({"model": kind, "seed": seed, "prob_lambda": plam, **met})
            for b in rel:
                calibration_rows.append({"model": kind, "seed": seed, "prob_lambda": plam, **b})
            for rr in ranking_metrics(score, test):
                ranking_rows.append({"model": kind, "seed": seed, "rank_lambda": rlam, **rr})
            for tau in (0.8, 0.9, 0.95):
                for K in (8, 16, 32):
                    threshold_rows.append({"model": kind, "seed": seed, "prob_lambda": plam, "rank_lambda": rlam, **threshold_then_rank(prob, score, test, tau, K)})

    write_csv("probability_metrics.csv", probability_rows)
    write_csv("calibration_metrics.csv", calibration_rows)
    write_csv("ranking_metrics.csv", ranking_rows)
    write_csv("threshold_then_rank.csv", threshold_rows)
    write_csv("training_runs.csv", runs)

    for kind in ("A", "B", "C", "D"):
        pr = [r for r in probability_rows if r["model"] == kind]
        stability.append({"model": kind, "metric_family": "probability", **mean_std(pr, ["nll", "mae", "brier", "spearman", "ece10", "b15_auroc", "b15_auprc", "b15_precision", "b15_recall"])})
        for K in (8, 16, 32):
            rr = [r for r in ranking_rows if r["model"] == kind and r["K"] == K]
            stability.append({"model": kind, "metric_family": f"ranking_K{K}", **mean_std(rr, ["mean_selected_q", "regret", "b15_selection_rate", "top3_b15_hit_rate", "within_state_spearman", "pairwise_order_accuracy"])})
    write_csv("seed_stability.csv", stability)

    # Compact decision payload; classification/final report are finalized below.
    agg_prob = {k: next(r for r in stability if r["model"] == k and r["metric_family"] == "probability") for k in "ABCD"}
    agg_rank = {k: {K: next(r for r in stability if r["model"] == k and r["metric_family"] == f"ranking_K{K}") for K in (8, 16, 32)} for k in "ABCD"}
    a_sel = np.mean([agg_rank["A"][K]["b15_selection_rate_mean"] for K in (8, 16, 32)])
    best_kind = min(("B", "C", "D"), key=lambda k: np.mean([agg_rank[k][K]["regret_mean"] for K in (8, 16, 32)]))
    overall_best_kind = min(("A", "B", "C", "D"), key=lambda k: np.mean([agg_rank[k][K]["regret_mean"] for K in (8, 16, 32)]))
    best_sel = np.mean([agg_rank[best_kind][K]["b15_selection_rate_mean"] for K in (8, 16, 32)])
    best_reg = np.mean([agg_rank[best_kind][K]["regret_mean"] for K in (8, 16, 32)])
    a_reg = np.mean([agg_rank["A"][K]["regret_mean"] for K in (8, 16, 32)])
    single_nll_damage = agg_prob["B"]["nll_mean"] - agg_prob["A"]["nll_mean"]
    decoupled_preserves = abs(agg_prob[best_kind]["nll_mean"] - agg_prob["A"]["nll_mean"]) < 0.05 if best_kind in ("C", "D") else False
    selection_gain = best_sel - a_sel
    regret_gain = a_reg - best_reg
    if best_sel >= 0.88 and selection_gain >= 0.08 and regret_gain >= 0.05 and single_nll_damage <= 0.05:
        decision = "RANKING_OBJECTIVE_STRONGLY_SUPPORTED"
    elif best_kind in ("C", "D") and best_sel > a_sel + 0.08 and decoupled_preserves:
        decision = "DECOUPLED_PROBABILITY_RANKING_SUPPORTED"
    elif best_sel <= a_sel + 0.05 or best_reg >= a_reg - 0.03:
        decision = "OBJECTIVE_MISMATCH_NOT_MAIN_BOTTLENECK"
    else:
        decision = "RANKING_AUGMENTATION_NOT_SUPPORTED" if agg_prob[best_kind]["nll_mean"] > agg_prob["A"]["nll_mean"] + 0.1 else "DECOUPLED_PROBABILITY_RANKING_SUPPORTED"
    final = {
        "classification": decision,
        "new_rollout": 0,
        "ranking_pairs": {"train": len(train_pairs), "val": len(val_pairs), "strong_train": sum(p["strong_b15_failure"] for p in train_pairs)},
        "selected_variants": selected,
        "best_ranking_model": best_kind,
        "overall_best_ranking_model": overall_best_kind,
        "probability_summary": agg_prob,
        "ranking_summary": agg_rank,
        "baseline_mean_selection": a_sel,
        "best_mean_selection": best_sel,
        "baseline_mean_regret": a_reg,
        "best_mean_regret": best_reg,
        "augmentation_selection_gain": selection_gain,
        "augmentation_regret_gain": regret_gain,
        "single_head_nll_damage": single_nll_damage,
    }
    dump("final_decision.json", final)
    make_report(final, threshold_rows)
    print(json.dumps(final, indent=2))


def make_report(final, threshold_rows):
    p = final["probability_summary"]
    r = final["ranking_summary"]
    lines = [
        "# ORTHOFLOW3_RANKING_AWARE_CRITIC_V1",
        "",
        f"Classification: **{final['classification']}**",
        "",
        f"NEW ROLLOUT = **{final['new_rollout']}**.",
        "",
        f"Training ranking pairs: {final['ranking_pairs']['train']:,}; VAL ranking pairs: {final['ranking_pairs']['val']:,}; strong B15-vs-failure pairs: {final['ranking_pairs']['strong_train']:,}.",
        "",
        "## Probability metrics (three-seed mean)",
        "",
        "| Model | NLL | MAE | Spearman | ECE10 | B15 AUROC | B15 recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in "ABCD":
        lines.append(f"| {k} | {p[k]['nll_mean']:.4f} | {p[k]['mae_mean']:.4f} | {p[k]['spearman_mean']:.4f} | {p[k]['ece10_mean']:.4f} | {p[k]['b15_auroc_mean']:.4f} | {p[k]['b15_recall_mean']:.4f} |")
    lines += ["", "## Ranking metrics (three-seed mean)", "", "| Model | K | Regret | B15 selection | Top-3 B15 | Within-state rho | Pair order |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in "ABCD":
        for K in (8, 16, 32):
            x = r[k].get(K, r[k].get(str(K)))
            lines.append(f"| {k} | {K} | {x['regret_mean']:.4f} | {x['b15_selection_rate_mean']:.4f} | {x['top3_b15_hit_rate_mean']:.4f} | {x['within_state_spearman_mean']:.4f} | {x['pairwise_order_accuracy_mean']:.4f} |")
    lines += ["", "## Selected variants", "", "```json", json.dumps(final["selected_variants"], indent=2, sort_keys=True), "```", "", "## Threshold then rank", ""]
    for model in ("C", "D"):
        lines += [f"### Model {model}", "", "| tau | K | candidate pass | B15 precision | B15 recall | no candidate | selected Q | B15 selection | regret |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for tau in (0.8, 0.9, 0.95):
            for K in (8, 16, 32):
                z = [x for x in threshold_rows if x["model"] == model and x["tau"] == tau and x["K"] == K]
                avg = lambda key: float(np.mean([x[key] for x in z]))
                lines.append(f"| {tau:.2f} | {K} | {avg('candidate_pass_rate'):.4f} | {avg('true_b15_precision'):.4f} | {avg('true_b15_recall'):.4f} | {avg('no_candidate_rate'):.4f} | {avg('selected_true_q'):.4f} | {avg('selection_b15_rate'):.4f} | {avg('conditional_regret'):.4f} |")
        lines.append("")
    lines += ["## Interpretation", "", f"Overall best ranking model: **{final['overall_best_ranking_model']}**; best ranking-augmented model: **{final['best_ranking_model']}**. Relative to NLL-only A, the best augmentation changed mean B15 top-1 over K=8/16/32 from {final['baseline_mean_selection']:.3f} to {final['best_mean_selection']:.3f}, and mean regret from {final['baseline_mean_regret']:.3f} to {final['best_mean_regret']:.3f}.", ""]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
