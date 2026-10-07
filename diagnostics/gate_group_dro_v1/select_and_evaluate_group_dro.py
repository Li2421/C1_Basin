"""Validation-only eta selection, followed by selected-model OOF evaluation.

This program is deliberately ordered in two phases.  Phase 1 reads only the
candidate validation predictions and selects one pre-registered eta_q per
frozen outer fold.  Phase 2 then loads only those selected checkpoints and
performs the single allowed OOF/variant inference pass.  Candidate test
predictions are neither written nor read by the selection phase.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import group_dro_gate_runner as dro  # noqa: E402


ETAS = (0.01, 0.05, 0.1)
CONDITION = "GROUP_DRO_BCE"


def eta_tag(eta: float) -> str:
    return f"eta_{eta:.2f}".replace(".", "p")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(fold_id: str, eta: float, seed: int) -> Path:
    return OUT / "candidate_runs" / eta_tag(eta) / fold_id / f"seed_{seed}" / "checkpoint.npz"


def candidate_validation_path(fold_id: str, eta: float, seed: int) -> Path:
    return OUT / "candidate_runs" / eta_tag(eta) / fold_id / f"seed_{seed}" / "validation_predictions.csv"


def candidate_history_path(fold_id: str, eta: float, seed: int) -> Path:
    return OUT / "candidate_runs" / eta_tag(eta) / fold_id / f"seed_{seed}" / "group_history.npz"


def load_checkpoint(path: Path, fold: dict):
    with np.load(path, allow_pickle=False) as loaded:
        params = []
        layer = 0
        while f"layer_{layer}_weight" in loaded:
            params.append({"w": jnp.asarray(loaded[f"layer_{layer}_weight"]), "b": jnp.asarray(loaded[f"layer_{layer}_bias"])})
            layer += 1
        metadata = json.loads(str(loaded["metadata_json"]))
        mean = np.asarray(loaded["normalization_mean"], float)
        scale = np.asarray(loaded["normalization_scale"], float)
        binary = np.asarray(loaded["normalization_binary_mask"], bool)
    if metadata.get("loss_mode") != "group_dro_bce":
        raise RuntimeError(("not a Group-DRO checkpoint", path, metadata.get("loss_mode")))
    if not (np.allclose(mean, fold["mean"]) and np.allclose(scale, fold["scale"]) and np.array_equal(binary, fold["binary"])):
        raise RuntimeError(("checkpoint normalization mismatch", path))
    return params, metadata


def validation_candidate_rows(fold: dict, eta: float) -> tuple[list[dict], dict]:
    """Read validation only and produce the pre-registered selection summary."""
    by_seed = []
    for seed in dro.SEEDS:
        path = candidate_validation_path(fold["fold_id"], eta, seed)
        if not path.exists():
            raise RuntimeError(("missing candidate validation", path))
        rows = read_csv(path)
        if not rows or any(row["fold_id"] != fold["fold_id"] or float(row["eta_q"]) != eta or int(row["training_seed"]) != seed for row in rows):
            raise RuntimeError(("candidate validation metadata mismatch", path))
        by_seed.append(rows)
    ids = [row["state_id"] for row in by_seed[0]]
    labels = np.asarray([int(row["oracle_label"]) for row in by_seed[0]], int)
    if any([row["state_id"] for row in rows] != ids or [int(row["oracle_label"]) for row in rows] != labels.tolist() for rows in by_seed[1:]):
        raise RuntimeError(("candidate validation aggregation mismatch", fold["fold_id"], eta))
    probability = np.mean([[float(row["p_gate"]) for row in rows] for rows in by_seed], axis=0)
    threshold = float(dro.base.cv.select_threshold(labels, probability))
    metrics = dro.base.cv.metrics(labels, probability, threshold)
    record = {
        "fold_id": fold["fold_id"],
        "heldout_source_group": fold["outer_group"],
        "eta_q": eta,
        "selection_data": "validation stable states only",
        "seed_aggregation": "mean probability over fixed seeds 17,23,41",
        "validation_state_count": int(len(ids)),
        "validation_selected_threshold": threshold,
        **{f"validation_{key}": value for key, value in metrics.items()},
        "validation_state_BCE": float(dro.base.cv.state_bce(labels, probability)),
    }
    return by_seed, record


def select_etas(folds: list[dict]) -> tuple[dict[str, float], list[dict]]:
    """Phase 1: predeclare choice rule, read validation only, choose eta per fold."""
    all_rows = []
    selected = {}
    for fold in folds:
        candidates = []
        for eta in ETAS:
            _, row = validation_candidate_rows(fold, eta)
            candidates.append(row)
        # Pre-registered validation-only model-selection rule.  It mirrors the
        # frozen threshold's preference: BAcc, then lower worst error, then
        # discrimination/calibration ties, then lower eta as deterministic tie.
        def key(row: dict):
            return (
                float(row["validation_balanced_accuracy"]),
                -max(float(row["validation_FPR"]), float(row["validation_FNR"])),
                float(row["validation_AUROC"]),
                float(row["validation_AUPRC"]),
                -float(row["validation_Brier"]),
                -float(row["eta_q"]),
            )
        chosen = max(candidates, key=key)
        selected[fold["fold_id"]] = float(chosen["eta_q"])
        for row in candidates:
            row["selected_by_validation_only"] = bool(row is chosen)
            row["selection_rule"] = "maximize validation BAcc; minimize max(FPR,FNR); maximize AUROC/AUPRC; minimize Brier; lower eta tie-break"
            row["outer_test_read_during_selection"] = False
            all_rows.append(row)
    return selected, all_rows


def ensemble_result(fold: dict, context: dict, eta: float) -> dict:
    """Phase 2: load only selected checkpoints and perform frozen OOF inference."""
    sample_ids = context["sample_ids"]
    label = context["label"]
    validation, test, models = [], [], {}
    for seed in dro.SEEDS:
        params, metadata = load_checkpoint(checkpoint_path(fold["fold_id"], eta, seed), fold)
        val_ids, val_p, val_y = dro.base.cv.aggregate_state(
            dro.base.cv.predict(params, fold["normalized"][fold["validation_index"]]),
            sample_ids[fold["validation_index"]], label,
        )
        saved = read_csv(candidate_validation_path(fold["fold_id"], eta, seed))
        saved_ids = [row["state_id"] for row in saved]
        saved_p = np.asarray([float(row["p_gate"]) for row in saved])
        if saved_ids != val_ids.tolist() or not np.allclose(saved_p, val_p, atol=2e-6, rtol=0):
            raise RuntimeError(("selected checkpoint validation reproduction failed", fold["fold_id"], eta, seed))
        test_ids, test_p, test_y = dro.base.cv.aggregate_state(
            dro.base.cv.predict(params, fold["normalized"][fold["test_index"]]),
            sample_ids[fold["test_index"]], label,
        )
        threshold = float(metadata["validation_selected_threshold"])
        recomputed_threshold = float(dro.base.cv.select_threshold(val_y, val_p))
        if abs(threshold - recomputed_threshold) > dro.base.TOLERANCE:
            raise RuntimeError(("checkpoint validation-only threshold reproduction failed", fold["fold_id"], eta, seed, threshold, recomputed_threshold))
        models[seed] = {"params": params, "threshold": threshold, "best_epoch": int(metadata["best_epoch"]), "validation_bce": float(metadata.get("validation_state_BCE", math.nan))}
        validation.append((val_ids, val_p, val_y))
        test.append((test_ids, test_p, test_y))
    if any(not np.array_equal(validation[0][0], item[0]) for item in validation) or any(not np.array_equal(test[0][0], item[0]) for item in test):
        raise RuntimeError(("selected model aggregation IDs differ", fold["fold_id"]))
    val_mean = np.mean([item[1] for item in validation], axis=0)
    test_mean = np.mean([item[1] for item in test], axis=0)
    ensemble_threshold = float(dro.base.cv.select_threshold(validation[0][2], val_mean))
    return {
        "fold": fold, "eta_q": eta, "seed_models": models, "validation": validation, "test": test,
        "val_ids": validation[0][0], "test_ids": test[0][0], "validation_mean": val_mean,
        "test_mean": test_mean, "ensemble_threshold": ensemble_threshold,
    }


def state_rows(result: dict) -> list[dict]:
    rows = []
    for position, seed in enumerate(dro.SEEDS):
        threshold = result["seed_models"][seed]["threshold"]
        t_logit = float(dro.base.logit(threshold))
        for state_id, probability, label in zip(result["test"][position][0], result["test"][position][1], result["test"][position][2]):
            logit = float(dro.base.logit(probability))
            predicted = int(probability >= threshold)
            rows.append({
                "condition": CONDITION, "loss_mode": "group_dro_bce", "eta_q": result["eta_q"],
                "fold_id": result["fold"]["fold_id"], "heldout_source_group": result["fold"]["outer_group"],
                "state_id": str(state_id), "oracle_label": int(label), "p_gate": float(probability), "gate_logit": logit,
                "validation_selected_threshold": threshold, "validation_threshold_logit": t_logit,
                "fold_relative_logit_margin": logit - t_logit, "predicted_label": predicted,
                "correct": bool(predicted == int(label)), "training_seed": seed,
            })
    threshold = result["ensemble_threshold"]
    t_logit = float(dro.base.logit(threshold))
    for state_id, probability, label in zip(result["test_ids"], result["test_mean"], result["test"][0][2]):
        logit = float(dro.base.logit(probability))
        predicted = int(probability >= threshold)
        rows.append({
            "condition": CONDITION, "loss_mode": "group_dro_bce", "eta_q": result["eta_q"],
            "fold_id": result["fold"]["fold_id"], "heldout_source_group": result["fold"]["outer_group"],
            "state_id": str(state_id), "oracle_label": int(label), "p_gate": float(probability), "gate_logit": logit,
            "validation_selected_threshold": threshold, "validation_threshold_logit": t_logit,
            "fold_relative_logit_margin": logit - t_logit, "predicted_label": predicted,
            "correct": bool(predicted == int(label)), "training_seed": "SEED_MEAN",
        })
    return rows


def infer_variants(state_id: str, result: dict, context: dict, role: str) -> list[dict]:
    arrays = context["arrays"]
    ids = context["sample_ids"]
    idx = np.flatnonzero(ids == state_id)
    idx = idx[np.argsort(arrays["flow_seed"][idx])]
    if len(idx) != 64:
        raise RuntimeError(("expected exactly 64 saved variants", state_id, len(idx)))
    x = result["fold"]["normalized"][idx]
    rows, seed_p = [], []
    for seed in dro.SEEDS:
        model = result["seed_models"][seed]
        logits = np.asarray(dro.base.cv.logits(model["params"], jnp.asarray(x, jnp.float32)), float)
        probabilities = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)), float)
        seed_p.append(probabilities)
        threshold = model["threshold"]
        t_logit = float(dro.base.logit(threshold))
        for position, data_index in enumerate(idx):
            predicted = int(probabilities[position] >= threshold)
            rows.append({
                "condition": CONDITION, "role": role, "eta_q": result["eta_q"], "fold_id": result["fold"]["fold_id"],
                "heldout_source_group": result["fold"]["outer_group"], "state_id": state_id,
                "oracle_label": int(context["label"][state_id]), "flow_seed": int(arrays["flow_seed"][data_index]),
                "sample_id": str(arrays["sample_id"][data_index]), "training_seed": seed,
                "p_gate": float(probabilities[position]), "gate_logit": float(logits[position]),
                "validation_selected_threshold": threshold, "validation_threshold_logit": t_logit,
                "fold_relative_logit_margin": float(logits[position] - t_logit),
                "predicted_label": predicted, "correct": bool(predicted == int(context["label"][state_id])),
            })
    mean_p = np.mean(seed_p, axis=0)
    threshold = result["ensemble_threshold"]
    t_logit = float(dro.base.logit(threshold))
    for position, data_index in enumerate(idx):
        logit = float(dro.base.logit(mean_p[position]))
        predicted = int(mean_p[position] >= threshold)
        rows.append({
            "condition": CONDITION, "role": role, "eta_q": result["eta_q"], "fold_id": result["fold"]["fold_id"],
            "heldout_source_group": result["fold"]["outer_group"], "state_id": state_id,
            "oracle_label": int(context["label"][state_id]), "flow_seed": int(arrays["flow_seed"][data_index]),
            "sample_id": str(arrays["sample_id"][data_index]), "training_seed": "SEED_MEAN",
            "p_gate": float(mean_p[position]), "gate_logit": logit,
            "validation_selected_threshold": threshold, "validation_threshold_logit": t_logit,
            "fold_relative_logit_margin": logit - t_logit,
            "predicted_label": predicted, "correct": bool(predicted == int(context["label"][state_id])),
        })
    return rows


def relative_metrics(rows: list[dict]) -> dict:
    """Pooled metrics with fold-relative margin as the cross-fold score axis."""
    if not rows:
        return {"state_count": 0}
    y = np.asarray([int(row["oracle_label"]) for row in rows], int)
    score = np.asarray([float(row["fold_relative_logit_margin"]) for row in rows], float)
    predicted = np.asarray([int(row["predicted_label"]) for row in rows], int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    pos, neg = tp + fn, tn + fp
    return {
        "score_coordinate": "fold_relative_logit_margin = logit - validation_threshold_logit (not raw cross-fold logits)",
        "state_count": int(len(rows)), "stable_zero": neg, "stable_nonzero": pos,
        "balanced_accuracy": float(((tp / pos) + (tn / neg)) / 2) if pos and neg else None,
        "AUROC": float(dro.base.cv.auroc(y, score)) if pos and neg else None,
        "AUPRC": float(dro.base.cv.auprc(y, score)) if pos and neg else None,
        "accuracy": float(np.mean(predicted == y)), "FPR": float(fp / neg) if neg else None,
        "FNR": float(fn / pos) if pos else None, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "mean_zero_margin": float(score[y == 0].mean()) if neg else None,
        "mean_nonzero_margin": float(score[y == 1].mean()) if pos else None,
    }


def subset_ids_from_difficult() -> tuple[set[str], set[str], set[str]]:
    source = ROOT / "diagnostics" / "hard_stable_boundary_crossval"
    difficult_rows = read_csv(source / "difficult_stable_states.csv")
    difficult = {row["state_id"] for row in difficult_rows}
    robust_rows = read_csv(source / "repeated_errors.csv")
    robust_zero = {row["state_id"] for row in robust_rows if int(row["oracle_label"]) == 0}
    expected = {
        "RBV_Q_pair228_m080_s95401003_p030",
        "RB_Q_pair226_m080_s95400802_p073",
        "RB_Q_pair228_m080_s95401001_p050",
    }
    if not expected <= robust_zero:
        raise RuntimeError(("fixed robust zero reference mismatch", expected - robust_zero))
    return difficult, robust_zero, {row["state_id"] for row in difficult_rows if int(row["oracle_label"]) == 1}


def variant_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_id"], str(row["training_seed"]))].append(row)
    summary = []
    for (state_id, seed), block in sorted(grouped.items()):
        label = int(block[0]["oracle_label"])
        margin = np.asarray([float(row["fold_relative_logit_margin"]) for row in block])
        p = np.asarray([float(row["p_gate"]) for row in block])
        predicted = np.asarray([int(row["predicted_label"]) for row in block])
        summary.append({
            "state_id": state_id, "oracle_label": label, "training_seed": seed,
            "fold_id": block[0]["fold_id"], "heldout_source_group": block[0]["heldout_source_group"],
            "eta_q": block[0]["eta_q"], "variant_count": len(block),
            "correct_variants": int(np.sum(predicted == label)), "wrong_variants": int(np.sum(predicted != label)),
            "intervention_fraction": float(np.mean(predicted == 1)), "mean_p_gate": float(p.mean()), "std_p_gate": float(p.std()),
            "mean_fold_relative_margin": float(margin.mean()), "std_fold_relative_margin": float(margin.std()),
            "min_fold_relative_margin": float(margin.min()), "max_fold_relative_margin": float(margin.max()),
        })
    return summary


def selected_history_rows(folds: list[dict], chosen: dict[str, float]) -> tuple[list[dict], list[dict]]:
    q_rows, loss_rows = [], []
    for fold in folds:
        eta = chosen[fold["fold_id"]]
        for seed in dro.SEEDS:
            path = candidate_history_path(fold["fold_id"], eta, seed)
            with np.load(path, allow_pickle=False) as values:
                # Materialize each compressed archive member exactly once.
                # Indexing an ``NpzFile`` member inside the row loop would
                # repeatedly decompress it, which is both needlessly slow and
                # obscures that this is a pure audit-export stage.
                epochs = values["epoch"]
                groups = values["group"]
                q_values = values["q_g"]
                losses = values["L_g"]
                objectives = values["objective"]
                entropy_values = values["q_entropy"]
                effective_counts = values["q_effective_group_count"]
                validation_bces = values["validation_state_BCE"]
                length = len(epochs)
                for position in range(length):
                    common = {
                        "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"],
                        "training_seed": seed, "eta_q": eta, "epoch": int(epochs[position]),
                        "training_source_group": str(groups[position]), "objective": float(objectives[position]),
                        "q_entropy": float(entropy_values[position]), "q_effective_group_count": float(effective_counts[position]),
                        "validation_state_BCE": None if not np.isfinite(validation_bces[position]) else float(validation_bces[position]),
                    }
                    q_rows.append({**common, "q_g": float(q_values[position])})
                    loss_rows.append({**common, "L_g": float(losses[position])})
    return q_rows, loss_rows


def make_selected_links(folds: list[dict], chosen: dict[str, float]) -> list[dict]:
    link_root = OUT / "selected_checkpoints"
    if link_root.exists():
        shutil.rmtree(link_root)
    links = []
    for fold in folds:
        eta = chosen[fold["fold_id"]]
        for seed in dro.SEEDS:
            source = checkpoint_path(fold["fold_id"], eta, seed)
            destination = link_root / fold["fold_id"] / f"seed_{seed}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
            links.append({"fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "eta_q": eta, "training_seed": seed, "checkpoint": str(destination), "target": str(source)})
    return links


def main() -> None:
    started = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    ready = json.loads((OUT / "BASELINE_DRO_READY.json").read_text())
    if ready.get("status") != "BASELINE_DRO_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline condition not ready", ready))
    context, folds = dro.load_context_and_folds()
    expected_candidate_count = len(folds) * len(ETAS) * len(dro.SEEDS)
    done = list((OUT / "candidate_runs").glob("eta_*/*/seed_*/DONE.json"))
    if len(done) != expected_candidate_count or any(json.loads(path.read_text()).get("status") != "COMPLETE" for path in done):
        raise RuntimeError(("candidate training incomplete", len(done), expected_candidate_count))

    # Phase 1: the only code path before eta selection reads candidate
    # validation CSVs.  No candidate test prediction is persisted or accessed.
    chosen, hyper_rows = select_etas(folds)
    write_csv(OUT / "hyperparameter_selection.csv", hyper_rows)
    dro.base.write_json(OUT / "selected_eta_by_fold.json", {
        "selection_data": "validation stable states only",
        "outer_test_used_for_selection": False,
        "eta_q_by_fold": chosen,
    })

    # Phase 2 begins only after the selection artifact is immutable on disk.
    selected = {fold["fold_id"]: ensemble_result(fold, context, chosen[fold["fold_id"]]) for fold in folds}
    oof = []
    for result in selected.values():
        oof.extend(state_rows(result))
    write_csv(OUT / "oof_predictions.csv", oof)

    difficult, robust_zero, difficult_nonzero = subset_ids_from_difficult()
    fold_by_group = {result["fold"]["outer_group"]: result for result in selected.values()}
    variant_rows = []
    for state_id in sorted(difficult):
        group = context["state"][state_id]["source_group"]
        role = "difficult_stable_nonzero" if context["label"][state_id] else "difficult_stable_zero"
        if state_id in robust_zero:
            role += "|robust_false_positive_zero_reference"
        variant_rows.extend(infer_variants(state_id, fold_by_group[group], context, role))
    write_csv(OUT / "hard_state_variant_predictions.csv", variant_rows)
    variants = variant_summary(variant_rows)
    write_csv(OUT / "hard_zero_state_metrics.csv", [row for row in variants if row["state_id"] in robust_zero])
    write_csv(OUT / "hard_nonzero_state_metrics.csv", [row for row in variants if row["state_id"] in difficult_nonzero])

    seed_mean = [row for row in oof if row["training_seed"] == "SEED_MEAN"]
    state_lookup = context["state"]
    all_metrics = relative_metrics(seed_mean)
    recovery_metrics = relative_metrics([row for row in seed_mean if state_lookup[row["state_id"]]["category"] == "RECOVERY"])
    difficult_metrics = relative_metrics([row for row in seed_mean if row["state_id"] in difficult])
    for blob, name in ((all_metrics, "all_stable_metrics.json"), (recovery_metrics, "recovery_metrics.json"), (difficult_metrics, "difficult_stable_metrics.json")):
        blob.update({"condition": CONDITION, "fold_count": len(folds), "selected_eta_by_fold": chosen})
        dro.base.write_json(OUT / name, blob)

    q_rows, loss_rows = selected_history_rows(folds, chosen)
    write_csv(OUT / "group_weight_history.csv", q_rows)
    write_csv(OUT / "group_loss_history.csv", loss_rows)
    link_rows = make_selected_links(folds, chosen)
    write_csv(OUT / "selected_checkpoint_manifest.csv", link_rows)

    # Per-fold, per-class margins use only each fold's own validation frame.
    margin_rows = dro.base.calibration_rows([row for row in oof if row["training_seed"] == "SEED_MEAN"], CONDITION)
    write_csv(OUT / "fold_relative_margin_analysis.csv", margin_rows)
    seed_stability = []
    for state_id in sorted({row["state_id"] for row in oof}):
        block = [row for row in oof if row["state_id"] == state_id and row["training_seed"] != "SEED_MEAN"]
        margins = np.asarray([float(row["fold_relative_logit_margin"]) for row in block])
        predicted = np.asarray([int(row["predicted_label"]) for row in block])
        seed_stability.append({"state_id": state_id, "oracle_label": int(block[0]["oracle_label"]), "fold_id": block[0]["fold_id"], "eta_q": block[0]["eta_q"], "seed_count": len(block), "mean_fold_relative_margin": float(margins.mean()), "std_fold_relative_margin": float(margins.std()), "prediction_agreement": float(max(np.mean(predicted == 0), np.mean(predicted == 1)))})
    write_csv(OUT / "seed_stability.csv", seed_stability)

    checks = {
        "status": "DRO_TRAINING_COMPLETE",
        "candidate_training_count": expected_candidate_count,
        "candidate_eta_values_exactly_preregistered": True,
        "selected_eta_using_validation_only": True,
        "outer_test_read_during_eta_selection": False,
        "outer_test_excluded_from_training_normalization_validation_threshold_and_q": True,
        "selected_checkpoint_validation_reproduction": True,
        "all_selected_outputs_finite": all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in oof + variant_rows),
        "frozen_fold_count": len(folds), "feature_schema_changed": False, "oracle_labels_changed": False,
        "new_states": 0, "new_oracle_rollouts": 0, "correction_head_trained": False, "closed_loop_run": False,
    }
    dro.base.write_json(OUT / "sanity_checks.json", checks)
    task_times = [float(json.loads(path.read_text())["elapsed_s"]) for path in done]
    dro.base.write_json(OUT / "runtime_statistics.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "selection_and_selected_evaluation_wall_s": time.perf_counter() - started,
        "candidate_training_runs": expected_candidate_count, "sum_candidate_task_wall_s": float(sum(task_times)),
        "selected_folds": len(folds), "selected_checkpoints": len(link_rows), "GPU_shards_candidate_training_peak": 2,
        "CPU_threads_per_shard": 4, "new_states": 0, "new_oracle_rollouts": 0,
    })
    dro.base.write_json(OUT / "manifest.json", {
        "condition": CONDITION, "baseline_readiness": ready, "candidate_eta_q": list(ETAS),
        "selected_eta_by_fold": chosen, "selection": "validation-state performance only; no candidate outer-test output used",
        "outputs": ["hyperparameter_selection.csv", "selected_eta_by_fold.json", "oof_predictions.csv", "hard_state_variant_predictions.csv", "group_weight_history.csv", "group_loss_history.csv", "selected_checkpoint_manifest.csv", "all_stable_metrics.json", "recovery_metrics.json", "difficult_stable_metrics.json"],
    })
    print({"status": "DRO_TRAINING_COMPLETE", "selected_eta_by_fold": chosen, "all_stable": all_metrics, "recovery": recovery_metrics, "difficult": difficult_metrics}, flush=True)


if __name__ == "__main__":
    main()
