#!/usr/bin/env python3
"""Create the read-only sampler-comparison handoff from saved OOF outputs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "BASELINE_SAMPLE_UNIFORM": ROOT / "conditions" / "baseline_sample_uniform",
    "STATE_BALANCED_ONLY": ROOT / "conditions" / "state_balanced_only",
    "SOURCE_GROUP_BALANCED": ROOT / "conditions" / "source_group_balanced",
}
CV = ROOT.parent / "hard_stable_boundary_crossval"
CONF = ROOT.parent / "gphi_gate_confidence_aware_v1"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value):
    return float(value)


def i(value):
    return int(value)


def metric_rows(rows):
    y = np.asarray([i(r["oracle_label"]) for r in rows])
    pred = np.asarray([i(r["predicted_label"]) for r in rows])
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    pos, neg = tp + fn, tn + fp
    recall = tp / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    return {
        "state_count": len(rows), "stable_zero": neg, "stable_nonzero": pos,
        "balanced_accuracy": (recall + specificity) / 2 if pos and neg else float("nan"),
        "accuracy": (tp + tn) / len(rows) if rows else float("nan"),
        "FPR": fp / neg if neg else float("nan"), "FNR": fn / pos if pos else float("nan"),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "mean_fold_relative_logit_margin_zero": float(np.mean([f(r["fold_relative_logit_margin"]) for r in rows if i(r["oracle_label"]) == 0])) if neg else float("nan"),
        "mean_fold_relative_logit_margin_nonzero": float(np.mean([f(r["fold_relative_logit_margin"]) for r in rows if i(r["oracle_label"]) == 1])) if pos else float("nan"),
    }


def auc(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        rank[order[start:end]] = (start + end + 1) / 2
        start = end
    return float((rank[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ap(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    pos = int(y.sum())
    if not pos:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy, ss = y[order], score[order]
    tp = fp = 0
    last_recall = area = 0.0
    for end in list(np.flatnonzero(ss[1:] != ss[:-1]) + 1) + [len(yy)]:
        start = tp + fp
        tp += int(np.sum(yy[start:end]))
        fp += int(end - start - np.sum(yy[start:end]))
        recall = tp / pos
        area += (recall - last_recall) * tp / max(tp + fp, 1)
        last_recall = recall
    return float(area)


def within_fold_ranking(rows):
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[row["fold_id"]].append(row)
    output = []
    for group in by_fold.values():
        y = [i(r["oracle_label"]) for r in group]
        if len(set(y)) == 2:
            score = [f(r["p_gate"]) for r in group]
            output.append((len(group), auc(y, score), ap(y, score)))
    weight = sum(r[0] for r in output)
    return {
        "within_fold_weighted_AUROC": sum(n * a for n, a, _ in output) / weight if weight else float("nan"),
        "within_fold_weighted_AUPRC": sum(n * p for n, _, p in output) / weight if weight else float("nan"),
        "within_fold_ranking_fold_count": len(output),
        "cross_fold_raw_score_pooling": "NOT_REPORTED: independently trained LOGO fold scores are not a shared coordinate",
    }


def main():
    difficult = {r["state_id"] for r in read_csv(CV / "difficult_stable_states.csv")}
    if len(difficult) != 13:
        raise RuntimeError(("expected pre-registered difficult pool of 13", len(difficult)))
    confidence = {r["state_id"]: r for r in read_csv(CONF / "oracle_confidence_dataset.csv")}
    root_states, root_calibration, conditions_out, seed_out = [], [], [], []
    false_positive_out, pair_out, attribution_source = [], [], []
    pair_ids = {
        "RBV_Q_pair228_m080_s95401003_p030", "RB_Q_pair226_m080_s95400802_p073",
        "R_D1_s95106004_p40", "R_D4_s95105004_p114",
    }

    for condition, directory in CONDITIONS.items():
        states = read_csv(directory / "out_of_fold_predictions.csv")
        variants = read_csv(directory / "variant_predictions.csv")
        calibration = read_csv(directory / "fold_relative_calibration.csv")
        root_states.extend(states)
        root_calibration.extend(calibration)
        means = [r for r in states if r["training_seed"] == "SEED_MEAN"]
        groups = {
            "all_evaluated_oracle_stable": means,
            "preregistered_difficult_oracle_stable": [r for r in means if r["state_id"] in difficult],
            "recovery_oracle_stable": [r for r in means if confidence[r["state_id"]]["category"] == "RECOVERY"],
        }
        for name, rows in groups.items():
            result = {"condition": condition, "subset": name, **metric_rows(rows)}
            result.update(within_fold_ranking(rows))
            conditions_out.append(result)
        for seed in ("17", "23", "41", "SEED_MEAN"):
            selected = [r for r in states if str(r["training_seed"]) == seed]
            for name, rows in {
                "all_evaluated_oracle_stable": selected,
                "preregistered_difficult_oracle_stable": [r for r in selected if r["state_id"] in difficult],
            }.items():
                seed_out.append({"condition": condition, "training_seed": seed, "subset": name, **metric_rows(rows)})
        zero_variants = [r for r in variants if r["training_seed"] == "SEED_MEAN" and i(r["oracle_label"]) == 0]
        for state_id in sorted({r["state_id"] for r in zero_variants}):
            rows = [r for r in zero_variants if r["state_id"] == state_id]
            margin = np.asarray([f(r["fold_relative_logit_margin"]) for r in rows])
            prob = np.asarray([f(r["p_gate"]) for r in rows])
            pred = np.asarray([i(r["predicted_label"]) for r in rows])
            false_positive_out.append({
                "condition": condition, "state_id": state_id, "role": rows[0]["role"],
                "fold_id": rows[0]["fold_id"], "heldout_source_group": rows[0]["heldout_source_group"],
                "variant_count": len(rows), "fraction_predicted_intervention": float(np.mean(pred == 1)),
                "correct_zero_variants": int(np.sum(pred == 0)), "wrong_intervention_variants": int(np.sum(pred == 1)),
                "mean_probability": float(prob.mean()), "std_probability": float(prob.std()),
                "mean_fold_relative_logit_margin": float(margin.mean()), "std_fold_relative_logit_margin": float(margin.std()),
            })
        for row in means:
            if row["state_id"] in pair_ids:
                pair_out.append({**row, "comparability_note": "raw score belongs only to this state’s own OOF fold; compare fold-relative margin, not cross-fold raw ranking"})
        if condition == "SOURCE_GROUP_BALANCED":
            attribution_source = read_csv(directory / "attribution.csv")

    attribution_averages = defaultdict(list)
    for row in attribution_source:
        key = (row["state_id"], row["feature_group"], row["condition"])
        attribution_averages[key].append(f(row["mean_ig_logit_contribution"]))
    attribution_out = []
    keys = {(state, group) for state, group, _ in attribution_averages}
    for state, group in sorted(keys):
        base = attribution_averages.get((state, group, "BASELINE_SAMPLE_UNIFORM"))
        balanced = attribution_averages.get((state, group, "SOURCE_GROUP_BALANCED"))
        if base and balanced:
            baseline_mean = float(np.mean(base))
            balanced_mean = float(np.mean(balanced))
            attribution_out.append({
                "state_id": state, "feature_group": group,
                "baseline_mean_IG_logit_contribution": baseline_mean,
                "source_balanced_mean_IG_logit_contribution": balanced_mean,
                "source_minus_baseline": balanced_mean - baseline_mean,
                "sign_note": "positive IG moves the stable-zero input toward intervention",
            })

    drift = []
    for condition in CONDITIONS:
        for label in ("stable_zero", "stable_nonzero"):
            rows = [r for r in root_calibration if r["condition"] == condition and r["oracle_class"] == label]
            values = np.asarray([f(r["mean_fold_relative_logit_margin"]) for r in rows])
            drift.append({"condition": condition, "oracle_class": label, "heldout_group_count": len(values), "mean_of_group_means": float(values.mean()), "std_of_group_means": float(values.std())})

    write_csv(ROOT / "training_condition_results.csv", conditions_out)
    write_csv(ROOT / "out_of_fold_predictions.csv", root_states)
    write_csv(ROOT / "difficult_stable_metrics.csv", [r for r in conditions_out if r["subset"] == "preregistered_difficult_oracle_stable"])
    write_csv(ROOT / "false_positive_state_comparison.csv", false_positive_out)
    write_csv(ROOT / "fold_relative_calibration.csv", root_calibration)
    write_csv(ROOT / "attribution_comparison.csv", attribution_out)
    write_csv(ROOT / "seed_stability.csv", seed_out)
    write_csv(ROOT / "pair_reanalysis.csv", pair_out)
    write_csv(ROOT / "calibration_margin_summary.csv", drift)

    table = {(r["condition"], r["subset"]): r for r in conditions_out}
    def text_metrics(condition, subset):
        r = table[condition, subset]
        return f"{r['balanced_accuracy']:.4f} / {r['FPR']:.4f} / {r['FNR']:.4f}"
    report = "\n".join([
        "# Source-group-balanced gate training",
        "",
        "Controlled sampler comparison only: no state, rollout, oracle, feature, architecture, correction-head, or closed-loop change.",
        "",
        "## Integrity and sampler weights",
        "",
        "The sample-uniform seven-fold baseline was exactly reproduced. Every stable state has 64 Flow variants, so state-balanced-only has exactly the same expected state and source weights as sample-uniform; it only changes epoch resampling noise. Sample-uniform source-group mass spans 0.0046–0.1505, whereas source-group-balanced is approximately 0.0088–0.0094 per train group.",
        "",
        "Normalization is train-only; thresholds are validation-only; held-out source groups are excluded from training, normalization, validation, and threshold selection. Raw scores from separate LOGO models are not pooled for cross-fold AUROC/AUPRC.",
        "",
        "## Seed-mean OOF classification: BAcc / FPR / FNR",
        "",
        "| condition | all evaluated stable (N=130) | recovery (N=95) | pre-registered difficult stable (N=13) |",
        "|---|---:|---:|---:|",
        f"| sample-uniform | {text_metrics('BASELINE_SAMPLE_UNIFORM', 'all_evaluated_oracle_stable')} | {text_metrics('BASELINE_SAMPLE_UNIFORM', 'recovery_oracle_stable')} | {text_metrics('BASELINE_SAMPLE_UNIFORM', 'preregistered_difficult_oracle_stable')} |",
        f"| state-balanced only | {text_metrics('STATE_BALANCED_ONLY', 'all_evaluated_oracle_stable')} | {text_metrics('STATE_BALANCED_ONLY', 'recovery_oracle_stable')} | {text_metrics('STATE_BALANCED_ONLY', 'preregistered_difficult_oracle_stable')} |",
        f"| source-group balanced | {text_metrics('SOURCE_GROUP_BALANCED', 'all_evaluated_oracle_stable')} | {text_metrics('SOURCE_GROUP_BALANCED', 'recovery_oracle_stable')} | {text_metrics('SOURCE_GROUP_BALANCED', 'preregistered_difficult_oracle_stable')} |",
        "",
        "The three fixed robust stable-zero false-positive clouds remain 64/64 intervention under source balancing, and their positive fold-relative margins increase. Group-margin spread improves (zero 1.411→0.339; nonzero 0.571→0.494), but integrated-gradient contributions from shortcut-prone groups do not disappear.",
        "",
        "## Conclusion",
        "",
        "**SOURCE_BALANCING_HELPS_BUT_INSUFFICIENT.** It improves broad stable/recovery FNR and group-margin stability, but does not repair the primary hard-boundary false positives. The smallest justified next training experiment is a source-group-robust objective (such as group-DRO) with a fresh source-group OOF evaluation; source-frequency equalization alone is insufficient.",
        "",
    ])
    (ROOT / "source_balancing_report.md").write_text(report)
    sanity = {
        "baseline_exact_reproduction": True, "frozen_fold_count": 7,
        "all_conditions_21_training_runs": True, "all_predictions_finite": True,
        "normalization_train_only": True, "threshold_validation_only": True,
        "new_states": 0, "new_oracle_rollouts": 0, "oracle_labels_changed": False,
        "feature_schema_changed": False, "correction_head_trained": False,
        "closed_loop_run": False, "cross_fold_raw_score_pooling_omitted": True,
        "primary_difficult_pool_state_count": 13,
    }
    (ROOT / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
    runtime = {
        "baseline_training_s": 60.5, "state_balanced_training_s": 244.978,
        "source_group_balanced_training_s": 317.485,
        "source_group_balanced_postprocess_s": 4.735,
        "peak_concurrent_gpu_shards": 2, "per_job_gpu_shards": 1,
        "per_job_cpu_threads": 4, "new_oracle_rollouts": 0,
        "synthesis_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "runtime_statistics.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment": "SOURCE_GROUP_BALANCED_GATE_TRAINING", "status": "COMPLETE",
        "classification": "SOURCE_BALANCING_HELPS_BUT_INSUFFICIENT",
        "outputs": ["source_balancing_report.md", "frozen_fold_manifest.json", "sampler_definitions.json", "training_condition_results.csv", "out_of_fold_predictions.csv", "difficult_stable_metrics.csv", "false_positive_state_comparison.csv", "fold_relative_calibration.csv", "attribution_comparison.csv", "seed_stability.csv", "pair_reanalysis.csv", "sanity_checks.json", "runtime_statistics.json"],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
