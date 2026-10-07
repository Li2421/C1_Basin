"""Validate all preregistered Stage-2 runs and pool strict OOF results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CV = Path("/home/zhihan/research/Basin_C1/diagnostics/hard_stable_boundary_crossval")
sys.path.insert(0, str(CV))
import run_crossval as cv  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, float)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)
    return np.log(x / (1 - x))


def relative_metrics(rows: list[dict]) -> dict:
    y = np.asarray([int(row["y_H"]) for row in rows])
    margin = np.asarray([float(row["fold_relative_logit_margin"]) for row in rows])
    return cv.metrics(y, sigmoid(margin), 0.5)


def grouped_bootstrap(rows: list[dict], seed_text: str, draws: int = 5000) -> dict:
    """Source-group bootstrap CIs for the pooled fold-relative metrics."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    groups = sorted(grouped)
    keys = ("balanced_accuracy", "AUROC", "FPR", "FNR")
    if len(groups) < 2:
        return {f"{key}_group_bootstrap95_lower": float("nan") for key in keys} | {
            f"{key}_group_bootstrap95_upper": float("nan") for key in keys
        }
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in keys}
    for _ in range(draws):
        selected = rng.integers(0, len(groups), size=len(groups))
        boot = [row for index in selected for row in grouped[groups[int(index)]]]
        value = relative_metrics(boot)
        for key in keys:
            if math.isfinite(float(value[key])):
                samples[key].append(float(value[key]))
    output = {}
    for key in keys:
        values = samples[key]
        lower, upper = (np.quantile(values, [0.025, 0.975]) if values else (float("nan"), float("nan")))
        output[f"{key}_group_bootstrap95_lower"] = float(lower)
        output[f"{key}_group_bootstrap95_upper"] = float(upper)
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    ready_path = HERE / "TRAINING_READY.json"
    ready = json.loads(ready_path.read_text())
    fold_manifest = json.loads((HERE / "fold_manifest.json").read_text())
    if ready["fold_manifest_sha256"] != sha(HERE / "fold_manifest.json"):
        raise AssertionError("fold manifest changed")
    all_rows, result_rows = [], []
    missing = []
    for fold in fold_manifest["folds"]:
        for model in ("LINEAR", "MLP_64x64"):
            for seed in (17,):
                run = HERE / "runs" / f"H{fold['H']}" / model / fold["fold_id"] / f"seed_{seed}"
                if not (run / "manifest.json").exists():
                    missing.append(str(run))
                    continue
                manifest = json.loads((run / "manifest.json").read_text())
                if manifest["training_ready_sha256"] != sha(ready_path):
                    raise AssertionError(f"stale run: {run}")
                if manifest["predictions_sha256"] != sha(run / "predictions.csv"):
                    raise AssertionError(f"prediction hash mismatch: {run}")
                result = json.loads((run / "result.json").read_text())
                result_rows.append(result)
                all_rows.extend(read_csv(run / "predictions.csv"))
    if missing:
        raise RuntimeError(f"missing {len(missing)} preregistered runs; first: {missing[:10]}")

    # The single preregistered seed is re-thresholded from its validation
    # records under the explicit PRIMARY_SEED_17 label for uniform reporting.
    primary_rows = []
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(int(row["H"]), row["model"], row["fold_id"], row["split"], row["state_id"])].append(row)
    by_fold = defaultdict(lambda: {"validation": [], "test": []})
    for (H, model, fold_id, split, state_id), rows in grouped.items():
        if len(rows) != 1:
            raise AssertionError((H, model, fold_id, split, state_id, len(rows)))
        exemplar = rows[0]
        by_fold[(H, model, fold_id)][split].append({**exemplar, "seed": "PRIMARY_SEED_17", "p_gate": float(rows[0]["p_gate"])})
    for (H, model, fold_id), split in by_fold.items():
        val = sorted(split["validation"], key=lambda x: x["state_id"])
        test = sorted(split["test"], key=lambda x: x["state_id"])
        threshold = cv.select_threshold(np.asarray([int(row["y_H"]) for row in val]), np.asarray([float(row["p_gate"]) for row in val]))
        for row in val + test:
            p = float(row["p_gate"])
            row.update({"seed": "PRIMARY_SEED_17", "threshold": threshold, "predicted": int(p >= threshold), "correct": int((p >= threshold) == int(row["y_H"]))})
            primary_rows.append(row)
    # `PRIMARY_SEED_17` is the one preregistered real run, re-thresholded from
    # its validation records.  Do not append the raw seed-17 alias to OOF
    # outputs: every state/model/fold prediction must appear exactly once.
    oof = []
    for row in primary_rows:
        if row["split"] != "test":
            continue
        margin = float(logit(float(row["p_gate"])) - logit(float(row["threshold"])))
        oof.append({**row, "fold_relative_logit_margin": margin, "fold_relative_probability": float(sigmoid(margin))})
    write_csv(HERE / "oof_predictions.csv", oof)
    for H in sorted({int(row["H"]) for row in oof}):
        write_csv(HERE / f"oof_predictions_H{H}.csv", [row for row in oof if int(row["H"]) == H])

    metrics_rows = []
    horizons = sorted({int(row["H"]) for row in oof})
    for H in horizons:
        for model in ("LINEAR", "MLP_64x64"):
            for seed in ("PRIMARY_SEED_17",):
                base = [row for row in oof if int(row["H"]) == H and row["model"] == model and str(row["seed"]) == seed]
                scopes = {
                    "ALL_RESOLVED": base,
                    "NORMAL": [row for row in base if row["category"] == "NORMAL"],
                    "PRE_DEADLOCK": [row for row in base if row["category"] == "PRE_DEADLOCK"],
                    "RECOVERY": [row for row in base if row["category"] == "RECOVERY"],
                    "HARD13_RESOLVED": [row for row in base if str(row["is_hard13_anchor"]).lower() == "true"],
                    "EVENTUALLY_NEEDED_Y_LONG1": [row for row in base if int(row["y_long_diagnostic_only"]) == 1],
                    "RECOVERY_Y_LONG1": [row for row in base if row["category"] == "RECOVERY" and int(row["y_long_diagnostic_only"]) == 1],
                }
                for scope, rows in scopes.items():
                    if not rows:
                        continue
                    value = relative_metrics(rows)
                    uncertainty = grouped_bootstrap(rows, f"H={H}|model={model}|scope={scope}")
                    metrics_rows.append({"H": H, "model": model, "seed": seed, "scope": scope, **value, **uncertainty, "source_group_count": len({row['source_group'] for row in rows}), "score_definition": "sigmoid(logit(p)-fold_validation_threshold_logit); threshold=0.5"})
    write_csv(HERE / "metrics.csv", metrics_rows)
    write_csv(HERE / "training_results.csv", result_rows)
    model_comparison = [
        row for row in metrics_rows
        if row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "ALL_RESOLVED"
    ]
    write_csv(HERE / "model_comparison.csv", model_comparison)
    for H in horizons:
        payload = {
            "H": H,
            "metrics": [row for row in metrics_rows if int(row["H"]) == H],
            "primary_cross_fold_score": "fold-relative logit margin",
            "raw_cross_fold_logits_used_as_common_coordinate": False,
        }
        (HERE / f"metrics_H{H}.json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")

    # Target nesting is a diagnostic for binary-vs-ordinal interpretation.
    targets = read_csv(HERE / "window_targets.csv")
    by_state = defaultdict(dict)
    for row in targets:
        by_state[row["state_id"]][int(row["H"])] = int(row["y_H"])
    violations = []
    for state_id, values in by_state.items():
        ordered = sorted(values)
        for lower, upper in zip(ordered, ordered[1:]):
            if values[lower] == 1 and values[upper] == 0:
                violations.append({"state_id": state_id, "lower_H": lower, "upper_H": upper, "lower_label": 1, "upper_label": 0})
    if violations:
        write_csv(HERE / "ordinal_consistency_violations.csv", violations)
    else:
        (HERE / "ordinal_consistency_violations.csv").write_text("state_id,lower_H,upper_H,lower_label,upper_label\n")

    summary = {
        "candidate_windows_sha256": ready["candidate_windows_sha256"],
        "training_ready_sha256": sha(ready_path),
        "required_run_count": len(fold_manifest["folds"]) * 2,
        "completed_run_count": len(result_rows),
        "oof_prediction_rows": len(oof),
        "ordinal_resolved_label_violations": len(violations),
        "raw_probabilities_across_folds_not_used_as_common_ranking_coordinate": True,
        "primary_cross_fold_score": "fold-relative logit margin",
    }
    (HERE / "stage2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    comparison_lines = []
    for row in model_comparison:
        comparison_lines.append(
            f"| {row['H']} | {row['model']} | {int(row['state_count'])} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['AUROC']):.4f} | "
            f"{float(row['AUPRC']):.4f} | {float(row['FPR']):.4f} | {float(row['FNR']):.4f} |"
        )
    urgency_lines = []
    for row in metrics_rows:
        if row["seed"] == "PRIMARY_SEED_17" and row["scope"] == "EVENTUALLY_NEEDED_Y_LONG1":
            urgency_lines.append(
                f"| {row['H']} | {row['model']} | {int(row['state_count'])} | "
                f"{float(row['balanced_accuracy']):.4f} | {float(row['AUROC']):.4f} | "
                f"{float(row['FPR']):.4f} | {float(row['FNR']):.4f} |"
            )
    report = f"""# Stage 2 window-aligned gate audit

- Strict source-group LOGO folds: **{len(fold_manifest['folds'])}**.
- Completed preregistered runs: **{len(result_rows)}**.
- Architectures: linear `214->1`; SiLU MLP `214->64->64->1`.
- Loss: ordinary BCE on resolved `y_H` only; `H_AMBIGUOUS` excluded.
- Preregistered training seed: 17 only, fixed uniformly before any Stage-2 performance; normalization and threshold selection are fold-train/validation only.
- Cross-fold AUROC/AUPRC use fold-relative logit margins, not incomparable raw logits.

| H | Model | Resolved states | BAcc | AUROC | AUPRC | FPR | FNR |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

`y_long=1` urgency-only diagnostic (old label is metadata, never a target):

| H | Model | Resolved states | BAcc | AUROC | FPR | FNR |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(urgency_lines)}

Resolved ordinal label violations: **{len(violations)}**.
"""
    (HERE / "stage2_report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
