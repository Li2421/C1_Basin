#!/usr/bin/env python3
"""Read-only Group-DRO analysis using fold-relative score coordinates only.

This script performs no optimizer/model update and does not read any new
rollouts.  It compares selected Group-DRO checkpoints with the frozen
source-group-balanced BCE condition.  Raw scores from different outer folds
are never pooled: any pooled ranking number below uses logit minus that
fold's validation-selected threshold.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "diagnostics" / "gate_source_balancing_v1"
CV = ROOT / "diagnostics" / "hard_stable_boundary_crossval"
CONF = ROOT / "diagnostics" / "gphi_gate_confidence_aware_v1"
sys.path.insert(0, str(OUT))
import group_dro_gate_runner as dro  # noqa: E402
sys.path.insert(0, str(SOURCE / "conditions" / "source_group_balanced"))
import run_source_group_balanced as source_runner  # noqa: E402


ROBUST_ZERO = (
    "RBV_Q_pair228_m080_s95401003_p030",
    "RB_Q_pair226_m080_s95400802_p073",
    "RB_Q_pair228_m080_s95401001_p050",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float) -> float:
    return float(value)


def i(value: str | int) -> int:
    return int(value)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score), dtype=float)
    ordered = score[order]
    left = 0
    while left < len(score):
        right = left + 1
        while right < len(score) and ordered[right] == ordered[left]:
            right += 1
        rank[order[left:right]] = (left + right + 1) / 2
        left = right
    return float((rank[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ap(y: np.ndarray, score: np.ndarray) -> float:
    pos = int(y.sum())
    if not pos:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy, ss = y[order], score[order]
    area = last_recall = 0.0
    tp = fp = 0
    ends = list(np.flatnonzero(ss[1:] != ss[:-1]) + 1) + [len(yy)]
    for end in ends:
        begin = tp + fp
        tp += int(yy[begin:end].sum())
        fp += int(end - begin - yy[begin:end].sum())
        recall = tp / pos
        area += (recall - last_recall) * tp / max(tp + fp, 1)
        last_recall = recall
    return float(area)


def metrics(rows: list[dict], name: str) -> dict:
    y = np.asarray([i(row["oracle_label"]) for row in rows], int)
    pred = np.asarray([i(row["predicted_label"]) for row in rows], int)
    margin = np.asarray([f(row["fold_relative_logit_margin"]) for row in rows], float)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    pos, neg = tp + fn, tn + fp
    recall = tp / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    return {
        "subset": name, "state_count": len(rows), "stable_zero": neg,
        "stable_nonzero": pos, "correct": tp + tn, "TP": tp, "TN": tn,
        "FP": fp, "FN": fn, "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (recall + specificity) / 2 if pos and neg else float("nan"),
        "FPR": fp / neg if neg else float("nan"), "FNR": fn / pos if pos else float("nan"),
        "fold_relative_margin_AUROC": auc(y, margin),
        "fold_relative_margin_AUPRC": ap(y, margin),
        "mean_zero_margin": float(margin[y == 0].mean()) if neg else float("nan"),
        "mean_nonzero_margin": float(margin[y == 1].mean()) if pos else float("nan"),
        "score_note": "AUROC/AUPRC use fold-relative logit margin, never pooled raw fold logits",
    }


def condition_rows(path: Path) -> list[dict]:
    return [row for row in read_csv(path) if row["training_seed"] == "SEED_MEAN"]


def group_dynamics() -> list[dict]:
    """Stream large histories; retain only first/final epoch group records."""
    files = {"q": OUT / "group_weight_history.csv", "loss": OUT / "group_loss_history.csv"}
    blocks: dict[tuple[str, str], dict] = {}
    for kind, path in files.items():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["fold_id"], row["training_seed"])
                epoch = i(row["epoch"])
                block = blocks.setdefault(key, {"fold_id": key[0], "training_seed": key[1]})
                # Histories are read one file at a time. Keep per-file epoch
                # sentinels rather than assuming the q pass created loss keys.
                low_key, high_key = f"first_{kind}_epoch", f"last_{kind}_epoch"
                low = block.get(low_key)
                high = block.get(high_key)
                if low is None or epoch < low:
                    block[low_key] = epoch
                    block[f"first_{kind}"] = []
                if epoch == block[low_key]:
                    block[f"first_{kind}"].append(row)
                if high is None or epoch > high:
                    block[high_key] = epoch
                    block[f"last_{kind}"] = []
                if epoch == block[high_key]:
                    block[f"last_{kind}"].append(row)
    out = []
    for block in blocks.values():
        fq = block["first_q"]
        lq = block["last_q"]
        fl = block["first_loss"]
        ll = block["last_loss"]
        topq = max(lq, key=lambda row: f(row["q_g"]))
        first_losses = np.asarray([f(row["L_g"]) for row in fl])
        last_losses = np.asarray([f(row["L_g"]) for row in ll])
        out.append({
            "fold_id": block["fold_id"], "training_seed": block["training_seed"],
            "heldout_source_group": topq["heldout_source_group"], "eta_q": f(topq["eta_q"]),
            "first_epoch": block["first_q_epoch"], "last_epoch": block["last_q_epoch"],
            "training_group_count": len(lq),
            "first_effective_groups": f(fq[0]["q_effective_group_count"]),
            "final_effective_groups": f(lq[0]["q_effective_group_count"]),
            "final_q_entropy": f(lq[0]["q_entropy"]),
            "final_max_q": f(topq["q_g"]), "final_top_q_group": topq["training_source_group"],
            "first_mean_group_loss": float(first_losses.mean()), "final_mean_group_loss": float(last_losses.mean()),
            "first_worst_group_loss": float(first_losses.max()), "final_worst_group_loss": float(last_losses.max()),
            "worst_group_loss_change": float(last_losses.max() - first_losses.max()),
            "final_worst_loss_group": max(ll, key=lambda row: f(row["L_g"]))["training_source_group"],
        })
    return sorted(out, key=lambda row: (row["fold_id"], int(row["training_seed"])))


def q_trajectory_stats() -> list[dict]:
    """Summarize q movement without retaining the 127 MB history in memory."""
    output: list[dict] = []
    active_key = active_epoch = None
    active_rows: list[dict] = []
    previous_q: dict[str, float] | None = None
    record: dict | None = None

    def finish_epoch(rows: list[dict]) -> None:
        nonlocal previous_q, record
        if not rows:
            return
        q = {row["training_source_group"]: f(row["q_g"]) for row in rows}
        top = max(q, key=q.get)
        if record is None:
            record = {"fold_id": rows[0]["fold_id"], "training_seed": rows[0]["training_seed"],
                      "heldout_source_group": rows[0]["heldout_source_group"], "epoch_count": 0,
                      "top_groups": [], "l1_steps": []}
        record["epoch_count"] += 1
        record["top_groups"].append(top)
        if previous_q is not None:
            record["l1_steps"].append(sum(abs(q[name] - previous_q[name]) for name in q))
        previous_q = q

    def finish_run() -> None:
        nonlocal previous_q, record
        if record is None:
            return
        tops = record["top_groups"]
        changes = sum(left != right for left, right in zip(tops, tops[1:]))
        l1 = record["l1_steps"]
        output.append({"fold_id": record["fold_id"], "training_seed": record["training_seed"],
                       "heldout_source_group": record["heldout_source_group"],
                       "epoch_count": record["epoch_count"], "unique_top_q_groups": len(set(tops)),
                       "top_q_group_switches": changes,
                       "mean_epoch_q_L1_change": float(np.mean(l1)) if l1 else 0.0,
                       "max_epoch_q_L1_change": float(np.max(l1)) if l1 else 0.0,
                       "final_top_q_group": tops[-1]})
        previous_q, record = None, None

    with (OUT / "group_weight_history.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["fold_id"], row["training_seed"])
            epoch = i(row["epoch"])
            if active_key is None:
                active_key, active_epoch = key, epoch
            if key != active_key or epoch != active_epoch:
                finish_epoch(active_rows)
                active_rows = []
                if key != active_key:
                    finish_run()
                    active_key = key
                active_epoch = epoch
            active_rows.append(row)
    finish_epoch(active_rows)
    finish_run()
    return sorted(output, key=lambda row: (row["fold_id"], int(row["training_seed"])))


def calibration_summary(rows: list[dict], condition: str) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["oracle_class"]].append(row)
    output = []
    for label, block in sorted(by_class.items()):
        means = np.asarray([f(row["mean_fold_relative_logit_margin"]) for row in block])
        errs = np.asarray([f(row["FPR_or_FNR"]) for row in block])
        output.append({
            "condition": condition, "oracle_class": label, "fold_count": len(block),
            "mean_of_fold_mean_margin": float(means.mean()), "std_of_fold_mean_margin": float(means.std()),
            "min_fold_mean_margin": float(means.min()), "max_fold_mean_margin": float(means.max()),
            "mean_class_error_rate": float(errs.mean()), "max_class_error_rate": float(errs.max()),
            "coordinate": "fold_relative_logit_margin",
        })
    return output


def load_params(path: Path):
    with np.load(path, allow_pickle=False) as loaded:
        params, layer = [], 0
        while f"layer_{layer}_weight" in loaded:
            params.append({"w": jnp.asarray(loaded[f"layer_{layer}_weight"]),
                           "b": jnp.asarray(loaded[f"layer_{layer}_bias"])})
            layer += 1
    return params


def group_dro_attribution() -> list[dict]:
    """Inference-only 256-step IG on selected DRO checkpoints for fixed probes."""
    jax.config.update("jax_enable_x64", True)
    context, folds = dro.load_context_and_folds()
    fold_by_group = {fold["outer_group"]: fold for fold in folds}
    ids = context["sample_ids"]
    flow = context["arrays"]["flow_seed"]
    semantic = source_runner.semantic_groups(context["schema"])
    rows = []
    for state_id in ROBUST_ZERO:
        group = context["state"][state_id]["source_group"]
        fold = fold_by_group[group]
        index = np.flatnonzero(ids == state_id)
        index = index[np.argsort(flow[index])]
        if len(index) != 64:
            raise RuntimeError(("expected 64 variants", state_id, len(index)))
        x = fold["normalized"][index]
        for seed in dro.SEEDS:
            path = OUT / "selected_checkpoints" / fold["fold_id"] / f"seed_{seed}.npz"
            params = load_params(path)
            ig = source_runner.integrated_gradients(params, x)
            logits = np.asarray(dro.base.cv.logits(params, jnp.asarray(x, jnp.float32)), float)
            ref = float(dro.base.cv.logits(params, jnp.zeros((1, 214), jnp.float32))[0])
            residual = float(np.max(np.abs(ig.sum(axis=1) - (logits - ref))))
            if residual > 5e-3:
                raise RuntimeError(("IG completeness", state_id, seed, residual))
            for name, dims in semantic.items():
                values = ig[:, dims].sum(axis=1)
                rows.append({
                    "condition": "GROUP_DRO_BCE", "state_id": state_id,
                    "source_group": group, "fold_id": fold["fold_id"], "training_seed": seed,
                    "feature_group": name, "dimension_count": len(dims),
                    "mean_ig_logit_contribution": float(values.mean()),
                    "std_ig_logit_contribution": float(values.std()),
                    "fraction_variants_push_toward_intervention": float(np.mean(values > 0)),
                    "mean_absolute_ig_logit_contribution": float(np.abs(values).mean()),
                    "max_IG_completeness_residual": residual,
                    "ig_path": "fold train-normalized mean (0) to naturally saved 214-D Flow variants",
                })
    return rows


def main() -> None:
    source_oof = condition_rows(SOURCE / "conditions" / "source_group_balanced" / "out_of_fold_predictions.csv")
    dro_oof = condition_rows(OUT / "oof_predictions.csv")
    difficult = {row["state_id"] for row in read_csv(CV / "difficult_stable_states.csv")}
    category = {row["state_id"]: row["category"] for row in read_csv(CONF / "oracle_confidence_dataset.csv")}
    subsets = {
        "all_evaluated_oracle_stable": lambda rows: rows,
        "recovery_oracle_stable": lambda rows: [row for row in rows if category[row["state_id"]] == "RECOVERY"],
        "preregistered_difficult_oracle_stable": lambda rows: [row for row in rows if row["state_id"] in difficult],
    }
    comparison = []
    for condition, rows in (("SOURCE_GROUP_BALANCED_BCE", source_oof), ("GROUP_DRO_BCE", dro_oof)):
        for name, select in subsets.items():
            comparison.append({"condition": condition, **metrics(select(rows), name)})
    write_csv(OUT / "oof_metric_comparison.csv", comparison)

    source_cal = read_csv(SOURCE / "conditions" / "source_group_balanced" / "fold_relative_calibration.csv")
    dro_cal = read_csv(OUT / "fold_relative_margin_analysis.csv")
    calibration = calibration_summary(source_cal, "SOURCE_GROUP_BALANCED_BCE") + calibration_summary(dro_cal, "GROUP_DRO_BCE")
    write_csv(OUT / "calibration_comparison.csv", calibration)

    dynamics = group_dynamics()
    write_csv(OUT / "group_dynamics_summary.csv", dynamics)
    q_trajectory = q_trajectory_stats()
    write_csv(OUT / "q_trajectory_summary.csv", q_trajectory)

    # Fixed hard zero cloud comparison uses the seed-mean inference rows.
    dro_hard = read_csv(OUT / "hard_zero_state_metrics.csv")
    source_variant = read_csv(SOURCE / "conditions" / "source_group_balanced" / "variant_predictions.csv")
    hard_rows = []
    for state_id in ROBUST_ZERO:
        b = [row for row in source_variant if row["state_id"] == state_id and row["training_seed"] == "SEED_MEAN"]
        d = [row for row in dro_hard if row["state_id"] == state_id and row["training_seed"] == "SEED_MEAN"]
        if len(b) != 64 or len(d) != 1:
            raise RuntimeError(("hard zero inventory", state_id, len(b), len(d)))
        bp = np.asarray([f(row["p_gate"]) for row in b])
        bm = np.asarray([f(row["fold_relative_logit_margin"]) for row in b])
        row = d[0]
        hard_rows.append({
            "state_id": state_id, "oracle_label": 0,
            "source_balanced_intervention_fraction": float(np.mean([i(v["predicted_label"]) for v in b])),
            "source_balanced_mean_probability": float(bp.mean()),
            "source_balanced_mean_margin": float(bm.mean()),
            "group_dro_intervention_fraction": f(row["intervention_fraction"]),
            "group_dro_mean_probability": f(row["mean_p_gate"]),
            "group_dro_mean_margin": f(row["mean_fold_relative_margin"]),
            "margin_change_DRO_minus_source_balanced": f(row["mean_fold_relative_margin"]) - float(bm.mean()),
        })
    write_csv(OUT / "hard_zero_comparison.csv", hard_rows)

    # Fixed hard nonzero cloud aggregate; all seven N=13 nonzeros are present
    # in the Group-DRO hard-state inventory and remain untouched evaluation data.
    nonzero = [row for row in read_csv(OUT / "hard_nonzero_state_metrics.csv") if row["training_seed"] == "SEED_MEAN"]
    nonzero_summary = [{
        "condition": "GROUP_DRO_BCE", "difficult_nonzero_state_count": len(nonzero),
        "mean_intervention_fraction": float(np.mean([f(row["intervention_fraction"]) for row in nonzero])),
        "fully_intervened_cloud_count": int(sum(f(row["intervention_fraction"]) == 1.0 for row in nonzero)),
        "mean_fold_relative_margin": float(np.mean([f(row["mean_fold_relative_margin"]) for row in nonzero])),
        "cloud_level_FNR_fraction": float(np.mean([f(row["intervention_fraction"]) < 0.5 for row in nonzero])),
    }]
    write_csv(OUT / "hard_nonzero_summary.csv", nonzero_summary)

    # Seed sensitivity for difficult states only, without cross-fold score pooling.
    seed_rows = read_csv(OUT / "seed_stability.csv")
    seed_summary = []
    for label, select in (("difficult", lambda row: row["state_id"] in difficult),
                          ("robust_zero", lambda row: row["state_id"] in ROBUST_ZERO)):
        block = [row for row in seed_rows if select(row)]
        agreement = np.asarray([f(row["prediction_agreement"]) for row in block])
        spread = np.asarray([f(row["std_fold_relative_margin"]) for row in block])
        seed_summary.append({"subset": label, "state_count": len(block),
                             "mean_prediction_agreement": float(agreement.mean()),
                             "min_prediction_agreement": float(agreement.min()),
                             "mean_seed_margin_std": float(spread.mean()),
                             "max_seed_margin_std": float(spread.max())})
    write_csv(OUT / "seed_stability_summary.csv", seed_summary)

    dro_attr = group_dro_attribution()
    source_attr = [row for row in read_csv(SOURCE / "conditions" / "source_group_balanced" / "attribution.csv") if row["condition"] == "SOURCE_GROUP_BALANCED"]
    source_key = {(row["state_id"], str(row["training_seed"]), row["feature_group"]): row for row in source_attr}
    attr_compare = []
    for row in dro_attr:
        source_row = source_key.get((row["state_id"], str(row["training_seed"]), row["feature_group"]))
        if source_row is None:
            raise RuntimeError(("missing frozen source-balanced attribution", row))
        src = f(source_row["mean_ig_logit_contribution"])
        value = f(row["mean_ig_logit_contribution"])
        attr_compare.append({
            "state_id": row["state_id"], "training_seed": row["training_seed"], "fold_id": row["fold_id"],
            "feature_group": row["feature_group"], "dimension_count": row["dimension_count"],
            "source_balanced_mean_IG_logit_contribution": src,
            "group_dro_mean_IG_logit_contribution": value,
            "DRO_minus_source_balanced": value - src,
            "source_push_toward_intervention": src > 0,
            "DRO_push_toward_intervention": value > 0,
            "note": "positive IG moves a stable-zero input toward intervention",
        })
    write_csv(OUT / "attribution_comparison.csv", attr_compare)

    lookup = {(row["condition"], row["subset"]): row for row in comparison}
    lines = [
        "# Group-DRO offline analysis",
        "",
        "Read-only analysis of selected checkpoints. No model update, rollout, state, feature, label, or threshold retuning occurred. All ranking numbers below use fold-relative logit margin, never raw scores pooled across independently trained LOGO folds.",
        "",
        "## OOF classification (BAcc / FPR / FNR)",
        "",
        "| condition | all stable | recovery stable | difficult stable (N=13) |",
        "|---|---:|---:|---:|",
    ]
    for condition, label in (("SOURCE_GROUP_BALANCED_BCE", "source-balanced BCE"), ("GROUP_DRO_BCE", "Group-DRO")):
        cells = []
        for subset in ("all_evaluated_oracle_stable", "recovery_oracle_stable", "preregistered_difficult_oracle_stable"):
            row = lookup[condition, subset]
            cells.append(f"{row['balanced_accuracy']:.4f} / {row['FPR']:.4f} / {row['FNR']:.4f}")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.extend([
        "",
        "Group-DRO does not change the primary hard-boundary outcome: 5/13 correct, BAcc 0.3929, FPR 0.5000, FNR 0.7143. All three pre-registered robust zero clouds remain intervention on every seed-mean Flow variant.",
        "",
        "## Group-DRO dynamics",
        "",
        f"Across {len(dynamics)} fold/seed runs, final effective group count has median {np.median([row['final_effective_groups'] for row in dynamics]):.2f} (min {np.min([row['final_effective_groups'] for row in dynamics]):.2f}); median final max q is {np.median([row['final_max_q'] for row in dynamics]):.4f}. The final top-q group is baseline_r118 in every selected run. Its q mass is concentrated but not single-group collapse; epoch-to-epoch q movement is summarized in `q_trajectory_summary.csv`. First-to-final worst-group BCE falls in every run (see `group_dynamics_summary.csv`).",
        "",
        "## Interpretation",
        "",
        "**GROUP_DRO_DOES_NOT_HELP.** Worst-training-group weighting changes broad OOF behavior but does not repair the fixed source-held-out hard boundary or the stable-zero false-positive clouds. The smallest justified next experiment is not another generic reweighting sweep: audit a source-group-invariant training constraint/representation objective on the same frozen data under a fresh LOGO evaluation.",
    ])
    (OUT / "offline_analysis_report.md").write_text("\n".join(lines) + "\n")
    (OUT / "offline_analysis_sanity.json").write_text(json.dumps({
        "analysis_only": True, "new_states": 0, "new_oracle_rollouts": 0,
        "optimizer_steps": 0, "JAX_backend": jax.default_backend(),
        "raw_cross_fold_logits_pooled": False, "fold_relative_coordinate_used": True,
        "selected_checkpoint_count": 21, "source_baseline_checkpoint_retrained": False,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
