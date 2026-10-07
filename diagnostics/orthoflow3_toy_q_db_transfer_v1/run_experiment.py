#!/usr/bin/env python3
"""Zero-rollout transfer of the best Toy continuous-Q critic to DB."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flax import core, serialization
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

ROOT = Path("/home/zhihan/research/Basin_C1")
OLD = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1"
TOY = ROOT / "diagnostics/orthoflow3_nll_weighting_ablation_v1"
STRUCT = ROOT / "diagnostics/orthoflow3_structured_continuous_q_data_v1"
RANK = ROOT / "diagnostics/orthoflow3_ranking_aware_critic_v1"
OUT = Path(__file__).resolve().parent
SEEDS = (17, 23, 41)
MODELS = ("eta_only", "db_scratch", "toy_state_adapter", "toy_state_eta_adapter", "toy_full_finetune")

spec = importlib.util.spec_from_file_location("ranklib", RANK / "run_experiment.py")
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
SingleCritic = lib.SingleCritic


class EtaOnly(nn.Module):
    @nn.compact
    def __call__(self, eta):
        x = nn.silu(nn.Dense(32, name="eta1")(eta))
        x = nn.silu(nn.Dense(128, name="c1")(x))
        x = nn.silu(nn.Dense(64, name="c2")(x))
        return nn.Dense(1, name="out")(x)[..., 0]


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


def load_rows():
    rows = pq.read_table(OLD / "pair_table.parquet").to_pylist()
    db = [r for r in rows if r["scenario"] == "DB"]
    train = [r for r in db if r["sampled_train"]]
    val = [r for r in db if r["sampled_val"]]
    test_d = [r for r in db if r["sampled_eval"] and r["regime"] == "D_unseen_state_unseen_eta"]
    rank_test = [r for r in db if r["state_split"] == "test" and r["n_trials"] >= 16]
    return train, val, test_d, rank_test


def normalize_db(rows):
    man = json.load(open(OLD / "dataset_manifest.json"))
    sn = man["state_normalization"]["DB"]
    ec = np.asarray(man["eta_normalization"]["center"], np.float32)
    es = np.asarray(man["eta_normalization"]["scale"], np.float32)
    h = np.asarray([r["h_raw"] for r in rows], np.float32)
    eta = np.asarray([[r["eta1"], r["eta2"], r["eta3"]] for r in rows], np.float32)
    return {
        "rows": rows,
        "h": (h - np.asarray(sn["mean"], np.float32)) / np.asarray(sn["std"], np.float32),
        "eta": (eta - ec) / es,
        "y": np.asarray([r["empirical_q"] for r in rows], np.float32),
        "w": np.ones(len(rows), np.float32),
    }


def load_toy_params(seed):
    model = SingleCritic()
    init = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 214)), jnp.zeros((1, 3)))
    path = TOY / "models" / "secondary_combined" / "W1" / f"seed{seed}" / "checkpoint.msgpack"
    return serialization.from_bytes(init, path.read_bytes()), path


def db_init_from_toy(seed, toy_params):
    model = SingleCritic()
    params = core.unfreeze(model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 80)), jnp.zeros((1, 3))))
    source = core.unfreeze(toy_params)
    for key in ("eta1", "c1", "c2", "out"):
        params["params"][key] = source["params"][key]
    return model, core.freeze(params)


def make_optimizer(params, train_keys):
    if train_keys is None:
        return optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    labels = core.unfreeze(jax.tree_util.tree_map(lambda _: "freeze", params))
    for key in train_keys:
        labels["params"][key] = jax.tree_util.tree_map(lambda _: "train", labels["params"][key])
    return optax.multi_transform({
        "train": optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4)),
        "freeze": optax.set_to_zero(),
    }, core.freeze(labels))


def nll(prob, y):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(np.mean(-y * np.log(prob) - (1 - y) * np.log(1 - prob)))


def train_state_model(name, seed, train, val, toy_params=None):
    if name == "db_scratch":
        model = SingleCritic()
        params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 80)), jnp.zeros((1, 3)))
        keys = None
    else:
        model, params = db_init_from_toy(seed, toy_params)
        keys = {
            "toy_state_adapter": ("state1", "state2"),
            "toy_state_eta_adapter": ("state1", "state2", "eta1"),
            "toy_full_finetune": None,
        }[name]
    optimizer = make_optimizer(params, keys)
    state = optimizer.init(params)

    @jax.jit
    def step(p, s, h, eta, y):
        def loss(q):
            logits = model.apply(q, h, eta)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
        value, grad = jax.value_and_grad(loss)(p)
        update, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, update), s, value

    rng = np.random.default_rng(seed)
    best = (np.inf, None, 0)
    stale = 0
    for it in range(1, 4001):
        ix = rng.integers(0, len(train["y"]), 128)
        params, state, _ = step(params, state, jnp.asarray(train["h"][ix]), jnp.asarray(train["eta"][ix]), jnp.asarray(train["y"][ix]))
        if it % 50:
            continue
        prob = lib.sigmoid(np.asarray(model.apply(params, jnp.asarray(val["h"]), jnp.asarray(val["eta"]))))
        score = nll(prob, val["y"])
        if score < best[0] - 1e-6:
            best = (score, jax.tree_util.tree_map(np.asarray, params), it)
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    return model, best, it


def train_eta_only(seed, train, val):
    model = EtaOnly()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 3)))
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    state = optimizer.init(params)

    @jax.jit
    def step(p, s, eta, y):
        def loss(q): return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(q, eta), y))
        value, grad = jax.value_and_grad(loss)(p)
        update, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, update), s, value

    rng = np.random.default_rng(seed)
    best = (np.inf, None, 0)
    stale = 0
    for it in range(1, 4001):
        ix = rng.integers(0, len(train["y"]), 128)
        params, state, _ = step(params, state, jnp.asarray(train["eta"][ix]), jnp.asarray(train["y"][ix]))
        if it % 50:
            continue
        prob = lib.sigmoid(np.asarray(model.apply(params, jnp.asarray(val["eta"]))))
        score = nll(prob, val["y"])
        if score < best[0] - 1e-6:
            best = (score, jax.tree_util.tree_map(np.asarray, params), it)
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    return model, best, it


def predict(name, model, params, data):
    if name == "eta_only":
        logits = np.asarray(model.apply(params, jnp.asarray(data["eta"])))
    else:
        logits = np.asarray(model.apply(params, jnp.asarray(data["h"]), jnp.asarray(data["eta"])))
    return lib.sigmoid(logits), logits


def calibration(prob, y):
    x = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1 - 1e-6))
    def obj(ab): return nll(lib.sigmoid(ab[0] + ab[1] * x), y)
    fit = minimize(obj, np.asarray([0.0, 1.0]), method="BFGS")
    ece = 0.0
    for lo in np.arange(0, 1, .1):
        use = (prob >= lo) & ((prob < lo + .1) if lo < .9 else (prob <= 1))
        if use.any(): ece += use.mean() * abs(prob[use].mean() - y[use].mean())
    return float(fit.x[0]), float(fit.x[1]), float(ece)


def auc_ap(y, score):
    y = np.asarray(y, bool)
    pos, neg = y.sum(), (~y).sum()
    if not pos or not neg: return None, None
    ranks = rankdata(score)
    auc = (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
    order = np.argsort(-score); ys = y[order]; precision = np.cumsum(ys) / (np.arange(len(ys)) + 1)
    return float(auc), float(np.sum(precision * ys) / pos)


def probability_metrics(prob, data):
    y = data["y"]
    b15 = y >= 15 / 16
    auroc, auprc = auc_ap(b15, prob)
    pred = prob >= 15 / 16
    ci, cs, ece = calibration(prob, y)
    return {
        "nll": nll(prob, y),
        "mae": float(np.mean(abs(prob - y))),
        "brier": float(np.mean((prob - y) ** 2)),
        "spearman": float(spearmanr(prob, y).statistic),
        "calibration_intercept": ci,
        "calibration_slope": cs,
        "ece10": ece,
        "b15_auroc": auroc,
        "b15_auprc": auprc,
        "b15_accuracy": float(np.mean(pred == b15)),
        "b15_precision": float(np.sum(pred & b15) / max(np.sum(pred), 1)),
        "b15_recall": float(np.sum(pred & b15) / max(np.sum(b15), 1)),
    }


def candidate_sets(rows, data, K):
    by = defaultdict(list)
    for i, r in enumerate(rows): by[r["state_uid"]].append(i)
    result = {}
    for sid, ix in by.items():
        if len(ix) < K: continue
        ix = sorted(ix, key=lambda i: rows[i]["eta_uid"])
        eta = data["eta"][ix]
        chosen = [0]
        dist = np.linalg.norm(eta - eta[0], axis=1)
        while len(chosen) < K:
            j = int(np.argmax(dist)); chosen.append(j)
            dist = np.minimum(dist, np.linalg.norm(eta - eta[j], axis=1)); dist[chosen] = -1
        result[sid] = np.asarray(ix)[chosen]
    return result


def ranking_metrics(score, data, Ks=(8, 16, 32)):
    out = []
    for K in Ks:
        vals = []
        for _, ix in candidate_sets(data["rows"], data, K).items():
            order = ix[np.argsort(-score[ix])]
            chosen = order[0]; oracle = ix[np.argmax(data["y"][ix])]
            has = bool(np.any(data["y"][ix] >= 15 / 16))
            vals.append({
                "selected": float(data["y"][chosen]), "oracle": float(data["y"][oracle]), "has": has,
                "b15": bool(data["y"][chosen] >= 15 / 16), "top3": bool(np.any(data["y"][order[:3]] >= 15 / 16)),
                "unseen_fraction": float(np.mean([data["rows"][i]["eta_split"] == "test" for i in ix])),
            })
        cover = [v for v in vals if v["has"]]
        out.append({
            "K": K, "eligible_states": len(vals),
            "mean_selected_q": float(np.mean([v["selected"] for v in vals])),
            "mean_oracle_q": float(np.mean([v["oracle"] for v in vals])),
            "regret": float(np.mean([v["oracle"] - v["selected"] for v in vals])),
            "b15_selection_rate": float(np.mean([v["b15"] for v in cover])),
            "top3_b15_hit_rate": float(np.mean([v["top3"] for v in cover])),
            "mean_unseen_eta_fraction": float(np.mean([v["unseen_fraction"] for v in vals])),
        })
    return out


def mean_std(rows, keys):
    out = {}
    for key in keys:
        x = np.asarray([r[key] for r in rows], float)
        out[key + "_mean"] = float(x.mean()); out[key + "_std"] = float(x.std()); out[key + "_median"] = float(np.median(x))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, td, rt = load_rows()
    train, val, test, rank_test = map(normalize_db, (tr, va, td, rt))
    manifest = {
        "task": "ORTHOFLOW3_TOY_Q_DB_TRANSFER_V1", "new_rollout": 0,
        "toy_source": str(TOY / "models/secondary_combined/W1"),
        "db_pair_source": str(OLD / "pair_table.parquet"),
        "db_train_pairs": len(tr), "db_train_states": len({r['state_uid'] for r in tr}), "db_train_eta": len({r['eta_uid'] for r in tr}),
        "db_val_pairs": len(va), "db_val_states": len({r['state_uid'] for r in va}),
        "frozen_D_test_pairs": len(td), "frozen_D_test_states": len({r['state_uid'] for r in td}), "frozen_D_test_eta": len({r['eta_uid'] for r in td}),
        "ranking_test_pairs": len(rt), "ranking_test_states": len({r['state_uid'] for r in rt}),
        "loss": "W1 pair-equal pure NLL for all DB training",
        "mode_correspondence": False, "T_DB": False,
    }
    dump("dataset_manifest.json", manifest)
    dump("train_config.json", {"architecture": "80->128->64 state; 3->32 eta; 96->128->64->1 interaction", "optimizer": "AdamW 1e-3, wd 1e-4, clip 5", "seeds": SEEDS, "max_steps": 4000, "eval_every": 50, "patience_evals": 20, "loss": "pair-equal BCE/NLL"})

    probability_rows, ranking_rows, training_rows = [], [], []
    artifacts = {}
    for seed in SEEDS:
        toy_params, toy_path = load_toy_params(seed)
        for name in MODELS:
            if name == "eta_only": model, best, steps = train_eta_only(seed, train, val)
            else: model, best, steps = train_state_model(name, seed, train, val, toy_params)
            d = OUT / name / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
            (d / "checkpoint.msgpack").write_bytes(serialization.to_bytes(best[1]))
            summary = {"model": name, "seed": seed, "best_step": best[2], "steps": steps, "val_nll": best[0], "toy_source_checkpoint": str(toy_path) if name.startswith('toy_') else None}
            (d / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            training_rows.append(summary)
            artifacts[(name, seed)] = (model, best[1])
            prob, score = predict(name, model, best[1], test)
            probability_rows.append({"model": name, "seed": seed, **probability_metrics(prob, test)})
            rprob, rscore = predict(name, model, best[1], rank_test)
            for z in ranking_metrics(rscore, rank_test): ranking_rows.append({"model": name, "seed": seed, **z})
    write_csv("training_summary.csv", training_rows)
    write_csv("probability_metrics.csv", probability_rows)
    write_csv("ranking_metrics.csv", ranking_rows)

    stability = []
    for name in MODELS:
        p = [r for r in probability_rows if r["model"] == name]
        stability.append({"model": name, "family": "probability", **mean_std(p, ["nll", "mae", "brier", "spearman", "ece10", "b15_auroc", "b15_auprc", "b15_accuracy", "b15_precision", "b15_recall", "calibration_intercept", "calibration_slope"])})
        for K in (8, 16, 32):
            rr = [r for r in ranking_rows if r["model"] == name and r["K"] == K]
            stability.append({"model": name, "family": f"ranking_K{K}", **mean_std(rr, ["mean_selected_q", "mean_oracle_q", "regret", "b15_selection_rate", "top3_b15_hit_rate", "mean_unseen_eta_fraction"])})
    write_csv("seed_stability.csv", stability)

    # Audit whether DB's frozen unseen eta were seen numerically by the Toy
    # source. This does not alter the frozen TEST; it qualifies the claim.
    structured_source = [r for r in pq.read_table(STRUCT / "structured_pair_table.parquet").to_pylist() if r["matrix_partition"] == "TRAIN_TRAIN"]
    wide_source = pq.read_table(STRUCT / "sparse_matched_control.parquet").to_pylist()
    toy_eta = {r["eta_uid"]: np.asarray(r["eta"], float) for r in structured_source}
    for r in wide_source: toy_eta[r["eta_uid"]] = np.asarray([r["eta1"], r["eta2"], r["eta3"]], float)
    toy_coords = np.stack(list(toy_eta.values()))
    man = json.load(open(OLD / "dataset_manifest.json")); eta_scale = np.asarray(man["eta_normalization"]["scale"], float)
    d_eta = {r["eta_uid"]: np.asarray([r["eta1"], r["eta2"], r["eta3"]], float) for r in td}
    overlap_rows = []
    for uid, eta_value in sorted(d_eta.items()):
        distance = np.linalg.norm((toy_coords - eta_value) / eta_scale, axis=1)
        overlap_rows.append({"eta_uid": uid, "eta1": eta_value[0], "eta2": eta_value[1], "eta3": eta_value[2], "exact_uid_in_toy_source": uid in toy_eta, "exact_coordinate_in_toy_source": bool(np.any(np.max(np.abs(toy_coords - eta_value), axis=1) < 1e-12)), "nearest_toy_source_eta_distance": float(distance.min())})
    write_csv("eta_overlap_audit.csv", overlap_rows)
    exact_seen = {r["eta_uid"] for r in overlap_rows if r["exact_coordinate_in_toy_source"]}
    keep = np.asarray([r["eta_uid"] not in exact_seen for r in td])
    strict_test = {k: (v[keep] if k in ("h", "eta", "y", "w") else [x for x, use in zip(v, keep) if use]) for k, v in test.items()}
    strict_rows = []
    for name in MODELS:
        for seed in SEEDS:
            model, params = artifacts[(name, seed)]
            prob, _ = predict(name, model, params, strict_test)
            strict_rows.append({"model": name, "seed": seed, "pairs": len(strict_test["y"]), **probability_metrics(prob, strict_test)})
    write_csv("strict_global_unseen_probability.csv", strict_rows)
    strict_summary = {m: mean_std([r for r in strict_rows if r["model"] == m], ["nll", "mae", "spearman", "b15_auroc", "b15_accuracy"]) for m in MODELS}

    psummary = {m: next(r for r in stability if r["model"] == m and r["family"] == "probability") for m in MODELS}
    rsummary = {m: {K: next(r for r in stability if r["model"] == m and r["family"] == f"ranking_K{K}") for K in (8, 16, 32)} for m in MODELS}
    db = psummary["db_scratch"]
    eta = psummary["eta_only"]
    gains = {}
    for m in MODELS[2:]:
        gains[m] = {
            "nll_vs_db_scratch": float(db["nll_mean"] - psummary[m]["nll_mean"]),
            "mae_vs_db_scratch": float(db["mae_mean"] - psummary[m]["mae_mean"]),
            "spearman_vs_db_scratch": float(psummary[m]["spearman_mean"] - db["spearman_mean"]),
            "nll_vs_eta_only": float(eta["nll_mean"] - psummary[m]["nll_mean"]),
            "mean_regret": float(np.mean([rsummary[m][K]["regret_mean"] for K in (8, 16, 32)])),
            "mean_b15_selection": float(np.mean([rsummary[m][K]["b15_selection_rate_mean"] for K in (8, 16, 32)])),
        }
    best_transfer = min(MODELS[2:], key=lambda m: (psummary[m]["nll_mean"], np.mean([rsummary[m][K]["regret_mean"] for K in (8,16,32)])))
    positive_vs_db = gains[best_transfer]["nll_vs_db_scratch"] > 0.03 and gains[best_transfer]["mae_vs_db_scratch"] > 0.02
    positive_vs_eta = gains[best_transfer]["nll_vs_eta_only"] > 0.03
    decision = "POSITIVE_TRANSFER" if positive_vs_db and positive_vs_eta else ("PARTIAL_TRANSFER" if positive_vs_db or positive_vs_eta else "TRANSFER_NOT_DEMONSTRATED")
    final = {"classification": decision, "new_rollout": 0, "best_transfer_model": best_transfer, "probability_summary": psummary, "ranking_summary": rsummary, "transfer_gains": gains, "raw_eta_obstruction_diagnostic": {"state_adapter_nll": psummary['toy_state_adapter']['nll_mean'], "state_eta_adapter_nll": psummary['toy_state_eta_adapter']['nll_mean']}, "toy_source_eta_overlap": {"db_D_eta": len(d_eta), "exact_coordinate_overlap": len(exact_seen)}, "strict_global_unseen_probability": strict_summary}
    dump("final_decision.json", final)
    make_report(final)
    print(json.dumps({"classification": decision, "best_transfer": best_transfer, "gains": gains[best_transfer], "new_rollout": 0}, indent=2))


def make_report(final):
    p, r = final["probability_summary"], final["ranking_summary"]
    lines = ["# ORTHOFLOW3_TOY_Q_DB_TRANSFER_V1", "", f"Classification: **{final['classification']}**", "", "NEW ROLLOUT = **0**.", "", "## Frozen DB unseen-state + unseen-eta probability", "", "| Model | NLL | MAE | Spearman | B15 AUROC | B15 AUPRC | B15 accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        x = p[m]
        lines.append(f"| {m} | {x['nll_mean']:.4f} | {x['mae_mean']:.4f} | {x['spearman_mean']:.4f} | {x['b15_auroc_mean']:.4f} | {x['b15_auprc_mean']:.4f} | {x['b15_accuracy_mean']:.4f} |")
    lines += ["", "## Finite-candidate ranking on frozen DB TEST states", "", "| Model | K | Eligible | Selected Q | Oracle Q | Regret | B15 selection | Top-3 B15 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        for K in (8, 16, 32):
            x = r[m].get(K, r[m].get(str(K)))
            # Eligible count is deterministic: 16/16/10 for K=8/16/32.
            eligible = 10 if K == 32 else 16
            lines.append(f"| {m} | {K} | {eligible} | {x['mean_selected_q_mean']:.4f} | {x['mean_oracle_q_mean']:.4f} | {x['regret_mean']:.4f} | {x['b15_selection_rate_mean']:.4f} | {x['top3_b15_hit_rate_mean']:.4f} |")
    lines += ["", "## Transfer gains", ""]
    for m, g in final["transfer_gains"].items():
        lines.append(f"- {m}: NLL gain vs DB scratch={g['nll_vs_db_scratch']:.4f}; MAE gain={g['mae_vs_db_scratch']:.4f}; Spearman gain={g['spearman_vs_db_scratch']:.4f}; NLL gain vs eta-only={g['nll_vs_eta_only']:.4f}.")
    lines += ["", "## Toy-source eta overlap audit", "", f"Of the four frozen DB D-test eta values, {final['toy_source_eta_overlap']['exact_coordinate_overlap']} appeared exactly in the Toy source. On the remaining three globally unseen eta values:", "", "| Model | NLL | MAE | Spearman |", "|---|---:|---:|---:|"]
    for m in MODELS:
        x = final["strict_global_unseen_probability"][m]
        lines.append(f"| {m} | {x['nll_mean']:.4f} | {x['mae_mean']:.4f} | {x['spearman_mean']:.4f} |")
    lines += ["", f"Best transfer model: **{final['best_transfer_model']}**.", ""]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
