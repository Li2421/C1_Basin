#!/usr/bin/env python3
"""Pure-NLL weighting ablation using frozen Toy structured data."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/zhihan/research/Basin_C1")
SRC = ROOT / "diagnostics/orthoflow3_structured_continuous_q_data_v1"
RANK = ROOT / "diagnostics/orthoflow3_ranking_aware_critic_v1"
OUT = Path(__file__).resolve().parent
SEEDS = (17, 23, 41)
VARIANTS = ("W0", "W1", "W2", "W3")
DATASETS = ("primary_structured", "secondary_combined")
MAX_STEPS = 4000
EVAL_EVERY = 50

# Reuse the audited architecture, normalization, and evaluation definitions.
spec = importlib.util.spec_from_file_location("ranklib", RANK / "run_experiment.py")
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
SingleCritic = lib.SingleCritic


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


def prepare():
    sr = pq.read_table(SRC / "structured_pair_table.parquet").to_pylist()
    structured_rows = [dict(r, data_source="structured") for r in sr if r["matrix_partition"] == "TRAIN_TRAIN"]
    val_rows = [dict(r, data_source="structured") for r in sr if r["matrix_partition"] == "VAL_VAL"]
    test_rows = [dict(r, data_source="structured") for r in sr if r["matrix_partition"] == "TESTSTATE_TESTETA"]
    wide0 = []
    for r in pq.read_table(SRC / "sparse_matched_control.parquet").to_pylist():
        z = dict(r, eta=[r["eta1"], r["eta2"], r["eta3"]], data_source="wide")
        wide0.append(z)
    sk = {(r["state_uid"], r["eta_uid"]) for r in structured_rows}
    wide_rows = [r for r in wide0 if (r["state_uid"], r["eta_uid"]) not in sk]
    combined_rows = structured_rows + wide_rows
    assert len(combined_rows) == len({(r["state_uid"], r["eta_uid"]) for r in combined_rows})
    assert not ({r["eta_uid"] for r in combined_rows} & {r["eta_uid"] for r in test_rows})
    return structured_rows, wide_rows, combined_rows, val_rows, test_rows


def weights(rows, variant):
    n = np.asarray([min(int(r["n_trials"]), 16) for r in rows], np.float32)
    if variant == "W0":
        return n
    if variant == "W1":
        return np.ones(len(rows), np.float32)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["state_uid"]].append(i)
    w = np.zeros(len(rows), np.float32)
    for ix in by.values():
        ix = np.asarray(ix)
        base = np.ones(len(ix)) if variant == "W2" else np.sqrt(n[ix])
        w[ix] = base / base.sum()
    return w


def weighted_nll(p, y, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.sum(w * (-y * np.log(p) - (1 - y) * np.log(1 - p))) / np.sum(w))


def objective_metrics(p, data, row_weights):
    y = data["y"]
    pe = np.ones(len(y), np.float32)
    sb = weights(data["rows"], "W2")
    return {
        "training_objective": weighted_nll(p, y, row_weights),
        "pair_equal_nll": weighted_nll(p, y, pe),
        "state_balanced_nll": weighted_nll(p, y, sb),
    }


def val_diagnostics(p, data):
    base = lib.probability_metrics(p, data)
    base.pop("reliability")
    score = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    order = lib.ordering_metrics(score, data)
    rank = lib.ranking_metrics(score, data, Ks=(8, 12))
    out = {**base, **order}
    for z in rank:
        out.update({f"K{z['K']}_regret": z["regret"], f"K{z['K']}_b15_selection": z["b15_selection_rate"], f"K{z['K']}_top3": z["top3_b15_hit_rate"]})
    return out


def train_one(dataset, variant, seed, structured, wide, combined, val):
    train = structured if dataset == "primary_structured" else combined
    tw = weights(train["rows"], variant)
    vw = weights(val["rows"], variant)
    sw = weights(structured["rows"], variant)
    ww = weights(wide["rows"], variant) if len(wide["rows"]) else np.zeros(0, np.float32)
    if dataset == "secondary_combined":
        # Exact frozen 50/50 source objective. Normalize each source internally;
        # this prevents either pair count from defining its total contribution.
        sw = sw / sw.mean()
        ww = ww / ww.mean()
        tw = np.concatenate([0.5 * sw / sw.sum(), 0.5 * ww / ww.sum()]).astype(np.float32)
    model = SingleCritic()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, train["h"].shape[1])), jnp.zeros((1, 3)))
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    state = optimizer.init(params)

    @jax.jit
    def step(p, s, h, e, y, w):
        def loss(q):
            logits = model.apply(q, h, e)
            per = optax.sigmoid_binary_cross_entropy(logits, y)
            return jnp.sum(w * per) / jnp.maximum(jnp.sum(w), 1e-8)
        value, grad = jax.value_and_grad(loss)(p)
        update, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, update), s, value

    rng = np.random.default_rng(seed)
    trajectory = []
    best = (np.inf, None, 0)
    stale = 0
    selection_frozen = False
    for it in range(1, MAX_STEPS + 1):
        if dataset == "primary_structured":
            ix = rng.integers(0, len(train["y"]), 128)
            bh, be, by, bw = train["h"][ix], train["eta"][ix], train["y"][ix], tw[ix]
        else:
            # Frozen 50/50 source balance; all variants receive identical indices.
            si = rng.integers(0, len(structured["y"]), 64)
            wi = rng.integers(0, len(wide["y"]), 64)
            bh = np.concatenate([structured["h"][si], wide["h"][wi]])
            be = np.concatenate([structured["eta"][si], wide["eta"][wi]])
            by = np.concatenate([structured["y"][si], wide["y"][wi]])
            bw = np.concatenate([sw[si], ww[wi]])
        params, state, _ = step(params, state, jnp.asarray(bh), jnp.asarray(be), jnp.asarray(by), jnp.asarray(bw))
        if it % EVAL_EVERY:
            continue
        train_p, _ = lib.predict(model, params, train, False)
        val_p, _ = lib.predict(model, params, val, False)
        tm = objective_metrics(train_p, train, tw)
        vm = objective_metrics(val_p, val, vw)
        vd = val_diagnostics(val_p, val)
        row = {"dataset": dataset, "variant": variant, "seed": seed, "step": it, **{f"train_{k}": v for k, v in tm.items()}, **{f"val_{k}": v for k, v in vm.items()}, **{f"val_{k}": v for k, v in vd.items()}}
        trajectory.append(row)
        score = vm["training_objective"]
        # Reproduce the frozen early-stopping framework (20 evaluations of
        # patience), while continuing all runs to the same 4,000 steps solely
        # so trajectory correlations remain comparable across weightings.
        if not selection_frozen:
            if score < best[0] - 1e-6:
                best = (score, jax.tree_util.tree_map(np.asarray, params), it)
                stale = 0
            else:
                stale += 1
            if stale >= 20:
                selection_frozen = True
    return model, best, trajectory, tw


def test_group_metrics(prob, rows):
    y = np.asarray([r["empirical_q"] for r in rows])
    groups = {
        "clear_failure": y == 0,
        "intermediate": (y > 0) & (y < 15 / 16),
        "b15_robust": y >= 15 / 16,
    }
    result = {}
    for name, use in groups.items():
        result[name] = {
            "pairs": int(use.sum()),
            "mae": float(np.mean(abs(prob[use] - y[use]))),
            "nll": weighted_nll(prob[use], y[use], np.ones(use.sum())),
        }
    return result


def correlations(trajectories):
    rows = []
    xs = ("train_training_objective", "val_pair_equal_nll", "val_state_balanced_nll")
    ys = ("val_within_state_spearman", "val_K8_regret", "val_K12_regret")
    grouped = defaultdict(list)
    for r in trajectories:
        grouped[(r["dataset"], r["variant"], r["seed"])].append(r)
    for (dataset, variant, seed), rr in grouped.items():
        for xk in xs:
            for yk in ys:
                x = np.asarray([r[xk] for r in rr]); y = np.asarray([r[yk] for r in rr])
                rows.append({"dataset": dataset, "variant": variant, "seed": seed, "x": xk, "y": yk, "checkpoints": len(rr), "pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)})
    return rows


def mean_std(rows, metrics):
    out = {}
    for m in metrics:
        z = np.asarray([r[m] for r in rows], float)
        out[m + "_mean"] = float(z.mean())
        out[m + "_std"] = float(z.std())
        out[m + "_median"] = float(np.median(z))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sr, wr, cr, vr, ter = prepare()
    structured, wide, combined, val, test = map(lib.normalize, (sr, wr, cr, vr, ter))
    dump("dataset_manifest.json", {
        "new_rollout": 0,
        "primary_pairs": len(sr),
        "secondary_pairs": len(cr),
        "secondary_structured_pairs": len(sr),
        "secondary_wide_pairs": len(wr),
        "train_states": len({r["state_uid"] for r in cr}),
        "test_pairs": len(ter),
        "frozen_source": str(SRC),
    })
    (OUT / "weighting_definitions.md").write_text("""# Frozen weighting definitions\n\n- W0: pair weight `min(n_trials,16)`.\n- W1: every canonical pair has weight 1.\n- W2: pair weight `1/M_state`, so every state has equal total weight.\n- W3: within each state, pair weight is `sqrt(min(n,16))`, normalized to state total 1.\n\nAll variants use the same architecture, seed-specific initialization, minibatch indices, optimizer, learning rate, 4,000 steps, and 50-step evaluation cadence. Checkpoints minimize the variant's VAL probability objective. No ranking loss is used.\n""")

    # State-weight audit is deterministic and independent of training.
    state_audit = []
    for dataset, rows in (("primary_structured", sr), ("secondary_combined", cr)):
        by = defaultdict(list)
        for i, r in enumerate(rows):
            by[r["state_uid"]].append(i)
        allw = {v: weights(rows, v) for v in VARIANTS}
        for sid, ix in sorted(by.items()):
            state_audit.append({
                "dataset": dataset,
                "state_uid": sid,
                "eta_pairs": len(ix),
                "total_trials": sum(rows[i]["n_trials"] for i in ix),
                **{f"{v}_total_weight": float(allw[v][ix].sum()) for v in VARIANTS},
            })
    write_csv("state_weight_audit.csv", state_audit)

    trajectories, probability_rows, ranking_rows, group_rows = [], [], [], []
    checkpoints = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                model, best, traj, tw = train_one(dataset, variant, seed, structured, wide, combined, val)
                trajectories.extend(traj)
                checkpoints[(dataset, variant, seed)] = (model, best)
                d = OUT / "models" / dataset / variant / f"seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "checkpoint.msgpack").write_bytes(serialization.to_bytes(best[1]))
                dump(Path("models") / dataset / variant / f"seed{seed}" / "summary.json", {"best_val_objective": best[0], "best_step": best[2]})
                prob, logits = lib.predict(model, best[1], test, False)
                pm = lib.probability_metrics(prob, test)
                pm.pop("reliability")
                pe_nll = weighted_nll(prob, test["y"], np.ones(len(test["y"])))
                sb_nll = weighted_nll(prob, test["y"], weights(test["rows"], "W2"))
                probability_rows.append({"dataset": dataset, "variant": variant, "seed": seed, "best_step": best[2], "pair_equal_nll": pe_nll, "state_balanced_nll": sb_nll, **pm})
                for z in lib.ranking_metrics(logits, test):
                    ranking_rows.append({"dataset": dataset, "variant": variant, "seed": seed, **z})
                gm = test_group_metrics(prob, ter)
                # Final gradient proxy uses |p-q| times effective training weight.
                train_data = structured if dataset == "primary_structured" else combined
                train_rows = sr if dataset == "primary_structured" else cr
                train_prob, _ = lib.predict(model, best[1], train_data, False)
                group_y = train_data["y"]
                group_masks = {"clear_failure": group_y == 0, "intermediate": (group_y > 0) & (group_y < 15 / 16), "b15_robust": group_y >= 15 / 16}
                for name, use in group_masks.items():
                    group_rows.append({
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "label_group": name,
                        "train_pair_count": int(use.sum()),
                        "effective_weight_fraction": float(tw[use].sum() / tw.sum()),
                        "gradient_proxy_fraction": float(np.sum(tw[use] * abs(train_prob[use] - group_y[use])) / np.sum(tw * abs(train_prob - group_y))),
                        "test_pairs": gm[name]["pairs"],
                        "test_mae": gm[name]["mae"],
                        "test_nll": gm[name]["nll"],
                    })
    write_csv("training_trajectories.csv", trajectories)
    write_csv("probability_metrics.csv", probability_rows)
    write_csv("ranking_metrics.csv", ranking_rows)
    write_csv("label_group_weight_audit.csv", group_rows)
    corr = correlations(trajectories)
    write_csv("nll_vs_ranking_correlations.csv", corr)

    stability = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            pp = [r for r in probability_rows if r["dataset"] == dataset and r["variant"] == variant]
            stability.append({"dataset": dataset, "variant": variant, "family": "probability", **mean_std(pp, ["pair_equal_nll", "state_balanced_nll", "mae", "brier", "spearman", "ece10", "b15_auroc", "b15_auprc"])})
            for K in (8, 16, 32):
                rr = [r for r in ranking_rows if r["dataset"] == dataset and r["variant"] == variant and r["K"] == K]
                stability.append({"dataset": dataset, "variant": variant, "family": f"ranking_K{K}", **mean_std(rr, ["mean_selected_q", "regret", "b15_selection_rate", "top3_b15_hit_rate", "within_state_spearman", "pairwise_order_accuracy"])})
    write_csv("seed_stability.csv", stability)

    # State domination aggregate.
    domination = {}
    for dataset in DATASETS:
        z = [r for r in state_audit if r["dataset"] == dataset]
        domination[dataset] = {}
        for v in VARIANTS:
            x = np.asarray([r[f"{v}_total_weight"] for r in z])
            domination[dataset][v] = {"max_median_ratio": float(x.max() / np.median(x)), "min_max_ratio": float(x.min() / x.max()), "median": float(np.median(x)), "max": float(x.max())}

    # Compare with frozen ranking-aware experiment; no retraining.
    rank_stability = list(csv.DictReader(open(RANK / "seed_stability.csv")))
    comparison = []
    for dataset in DATASETS:
        candidates = []
        for v in VARIANTS:
            q = next(r for r in stability if r["dataset"] == dataset and r["variant"] == v and r["family"] == "probability")
            rs = [next(r for r in stability if r["dataset"] == dataset and r["variant"] == v and r["family"] == f"ranking_K{K}") for K in (8, 16, 32)]
            candidates.append((v, float(q["pair_equal_nll_mean"]), np.mean([float(x["regret_mean"]) for x in rs]), np.mean([float(x["b15_selection_rate_mean"]) for x in rs])))
        best = min(candidates, key=lambda x: (x[2], x[1]))
        comparison.append({"source": dataset, "model": f"pure_nll_{best[0]}", "nll": best[1], "mean_regret_K8_16_32": best[2], "mean_b15_selection_K8_16_32": best[3]})
    for model in ("A", "B", "C", "D"):
        q = next(r for r in rank_stability if r["model"] == model and r["metric_family"] == "probability")
        rs = [next(r for r in rank_stability if r["model"] == model and r["metric_family"] == f"ranking_K{K}") for K in (8, 16, 32)]
        comparison.append({"source": "ranking_aware_v1", "model": model, "nll": float(q["nll_mean"]), "mean_regret_K8_16_32": np.mean([float(x["regret_mean"]) for x in rs]), "mean_b15_selection_K8_16_32": np.mean([float(x["b15_selection_rate_mean"]) for x in rs])})
    write_csv("comparison_with_ranking_aware.csv", comparison)

    # Classification is based on primary, as required.
    prim = {v: {
        "prob": next(r for r in stability if r["dataset"] == "primary_structured" and r["variant"] == v and r["family"] == "probability"),
        "rank": {K: next(r for r in stability if r["dataset"] == "primary_structured" and r["variant"] == v and r["family"] == f"ranking_K{K}") for K in (8, 16, 32)},
    } for v in VARIANTS}
    def mean_rank(v, key): return float(np.mean([float(prim[v]["rank"][K][key + "_mean"]) for K in (8, 16, 32)]))
    base_reg, base_sel = mean_rank("W0", "regret"), mean_rank("W0", "b15_selection_rate")
    best_v = min(VARIANTS, key=lambda v: (mean_rank(v, "regret"), float(prim[v]["prob"]["pair_equal_nll_mean"])))
    reg_gain = base_reg - mean_rank(best_v, "regret")
    sel_gain = mean_rank(best_v, "b15_selection_rate") - base_sel
    prob_not_worse = float(prim[best_v]["prob"]["pair_equal_nll_mean"]) <= float(prim["W0"]["prob"]["pair_equal_nll_mean"]) + 0.02
    if best_v in ("W1", "W2") and prob_not_worse and reg_gain >= 0.04 and sel_gain >= 0.05:
        decision = "NLL_WEIGHTING_MISMATCH_SUPPORTED"
    elif best_v == "W3" and prob_not_worse and reg_gain >= 0.04 and sel_gain >= 0.05:
        decision = "CONFIDENCE_WEIGHTING_USEFUL_BUT_GLOBAL_TRIAL_WEIGHTING_HARMFUL"
    else:
        # The prompt's case D takes precedence: if no alternative produces a
        # material selection gain over W0, weighting is not the main cause.
        decision = "CURRENT_NLL_WEIGHTING_NOT_MAIN_CAUSE"

    final = {
        "classification": decision,
        "new_rollout": 0,
        "primary_best_variant": best_v,
        "primary_regret_gain_vs_W0": reg_gain,
        "primary_selection_gain_vs_W0": sel_gain,
        "state_domination": domination,
        "primary": prim,
        "comparison_with_ranking_aware": comparison,
        "ranking_trajectory_summary": summarize_correlations(corr),
    }
    dump("final_decision.json", final)
    make_report(final, stability, group_rows, comparison)
    print(json.dumps({"classification": decision, "best": best_v, "regret_gain": reg_gain, "selection_gain": sel_gain, "new_rollout": 0}, indent=2))


def summarize_correlations(rows):
    out = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            z = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant and r["x"] == "val_pair_equal_nll" and r["y"] == "val_K8_regret"]
            out.append({"dataset": dataset, "variant": variant, "pearson_mean": float(np.mean([r["pearson"] for r in z])), "spearman_mean": float(np.mean([r["spearman"] for r in z]))})
    return out


def make_report(final, stability, groups, comparison):
    lines = ["# ORTHOFLOW3_NLL_WEIGHTING_ABLATION_V1", "", f"Classification: **{final['classification']}**", "", "NEW ROLLOUT = **0**.", "", "## Primary structured-only TEST", "", "| Weight | NLL | MAE | Spearman | K8 regret/B15/top3 | K16 regret/B15/top3 | K32 regret/B15/top3 |", "|---|---:|---:|---:|---|---|---|"]
    for v in VARIANTS:
        p = next(r for r in stability if r["dataset"] == "primary_structured" and r["variant"] == v and r["family"] == "probability")
        ks = []
        for K in (8, 16, 32):
            r = next(x for x in stability if x["dataset"] == "primary_structured" and x["variant"] == v and x["family"] == f"ranking_K{K}")
            ks.append(f"{r['regret_mean']:.3f}/{r['b15_selection_rate_mean']:.3f}/{r['top3_b15_hit_rate_mean']:.3f}")
        lines.append(f"| {v} | {p['pair_equal_nll_mean']:.4f} | {p['mae_mean']:.4f} | {p['spearman_mean']:.4f} | {ks[0]} | {ks[1]} | {ks[2]} |")
    lines += ["", "## Secondary structured + wide TEST", "", "| Weight | NLL | MAE | Spearman | Mean regret | Mean B15 top-1 |", "|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        p = next(r for r in stability if r["dataset"] == "secondary_combined" and r["variant"] == v and r["family"] == "probability")
        rr = [next(x for x in stability if x["dataset"] == "secondary_combined" and x["variant"] == v and x["family"] == f"ranking_K{K}") for K in (8, 16, 32)]
        lines.append(f"| {v} | {p['pair_equal_nll_mean']:.4f} | {p['mae_mean']:.4f} | {p['spearman_mean']:.4f} | {np.mean([x['regret_mean'] for x in rr]):.4f} | {np.mean([x['b15_selection_rate_mean'] for x in rr]):.4f} |")
    lines += ["", "## State domination", ""]
    for dataset, dd in final["state_domination"].items():
        lines.append(f"- {dataset}: " + ", ".join(f"{v} max/median={dd[v]['max_median_ratio']:.3f}" for v in VARIANTS))
    lines += ["", "## NLL trajectory versus K8 regret", ""]
    for r in final["ranking_trajectory_summary"]:
        lines.append(f"- {r['dataset']} {r['variant']}: Pearson={r['pearson_mean']:.3f}, Spearman={r['spearman_mean']:.3f}")
    lines += ["", "## Comparison with ranking-aware critic", "", "| Source | Model | NLL | Mean regret | Mean B15 top-1 |", "|---|---|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['source']} | {r['model']} | {r['nll']:.4f} | {r['mean_regret_K8_16_32']:.4f} | {r['mean_b15_selection_K8_16_32']:.4f} |")
    lines += ["", "## Interpretation", "", f"Primary best weighting: **{final['primary_best_variant']}**. Relative to W0, mean regret gain={final['primary_regret_gain_vs_W0']:.4f}, mean B15-selection gain={final['primary_selection_gain_vs_W0']:.4f}.", ""]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
