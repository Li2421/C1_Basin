"""Source-balanced BCE with class-conditional source-logit invariance.

This is a deliberately constrained follow-on to the frozen source-balanced
gate experiment.  The 214-D inputs, oracle-stable labels, folds, SiLU MLP,
optimizer, train-only normalization, validation early stopping, and
validation-only threshold selection are all imported from the prior audit.

Only for ``lambda_invariance > 0``, the loss adds a generic *training
source-group* regularizer.  For each oracle class c and every training source
group g containing that class, let mu[g,c] be the mean model logit over the
complete saved group/class cell.  The penalty is the mean, over eligible
classes, of the equally-cell-weighted variance of mu[g,c] across source
groups.  A class with fewer than two nonempty training source-group cells is
skipped.  This preserves class separation: no term compares c=0 logits with
c=1 logits.

No validation/test state, group, loss, threshold, or statistic is used in the
regularizer.  Lambda=0 takes the exact old source-balanced BCE execution path
and is therefore a strict reproduction control, not merely a numerically
similar objective.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_AUDIT = ROOT / "diagnostics" / "gate_source_balancing_v1"
sys.path.insert(0, str(SOURCE_AUDIT))
from shared import gate_condition_runner as base  # noqa: E402


SEEDS = base.SEEDS
HIDDEN = base.HIDDEN
LAMBDA_GRID = (0.0, 0.01, 0.05, 0.1, 0.5)
LOSS_MODES = ("baseline_bce", "source_balanced_bce", "conditional_source_invariance_bce")


def load_context_and_folds() -> tuple[dict, list[dict]]:
    context = base.load_context()
    folds = base.prepare_folds(context)
    if len(folds) != 7:
        raise RuntimeError(("expected seven frozen LOGO folds", len(folds)))
    return context, folds


def class_cell_layout(fold: dict, context: dict) -> dict[str, Any]:
    """Create group×class membership from only frozen *training* rows."""
    state = context["state"]
    sample_ids = context["sample_ids"]
    labels = context["sample_label"][fold["train_index"]].astype(np.int32)
    if not np.isin(labels, (0, 1)).all():
        raise RuntimeError(("nonbinary stable training label", fold["fold_id"]))
    group_per_sample = np.asarray([state[str(state_id)]["source_group"] for state_id in sample_ids[fold["train_index"]]], dtype=object)
    groups = np.asarray(sorted(set(group_per_sample.tolist())), dtype=object)
    expected_groups = np.asarray(sorted(fold["train_groups"]), dtype=object)
    if not np.array_equal(groups, expected_groups):
        raise RuntimeError(("train group mismatch", fold["fold_id"], len(groups), len(expected_groups)))
    if fold["outer_group"] in set(groups) or fold["outer_group"] in set(fold["validation_groups"]):
        raise RuntimeError(("outer test group leakage", fold["fold_id"], fold["outer_group"]))
    lookup = {str(group): i for i, group in enumerate(groups)}
    group_index = np.asarray([lookup[str(group)] for group in group_per_sample], np.int32)
    cell_counts = np.zeros((len(groups), 2), np.int32)
    np.add.at(cell_counts, (group_index, labels), 1)
    active_cell = cell_counts > 0
    eligible_class = active_cell.sum(axis=0) >= 2
    if not eligible_class.any():
        raise RuntimeError(("no multi-source class cells for invariance", fold["fold_id"], cell_counts.tolist()))
    return {
        "groups": groups,
        "group_index": group_index,
        "labels": labels,
        "cell_counts": cell_counts,
        "active_cell": active_cell,
        "eligible_class": eligible_class,
        "eligible_group_count_by_class": active_cell.sum(axis=0).astype(int),
        "eligible_class_count": int(eligible_class.sum()),
    }


def cell_eligibility_rows(folds: list[dict], context: dict) -> list[dict]:
    """Auditable eligibility inventory; it contains no learned/test statistic."""
    rows: list[dict] = []
    for fold in folds:
        layout = class_cell_layout(fold, context)
        for group_i, group in enumerate(layout["groups"]):
            for class_value in (0, 1):
                rows.append({
                    "fold_id": fold["fold_id"],
                    "heldout_source_group": fold["outer_group"],
                    "training_source_group": str(group),
                    "oracle_class": class_value,
                    "complete_training_cell_sample_count": int(layout["cell_counts"][group_i, class_value]),
                    "cell_nonempty": bool(layout["active_cell"][group_i, class_value]),
                    "class_has_at_least_two_training_source_cells": bool(layout["eligible_class"][class_value]),
                    "included_in_conditional_invariance": bool(layout["active_cell"][group_i, class_value] and layout["eligible_class"][class_value]),
                    "uses_validation_or_outer_test_data": False,
                })
    return rows


def conditional_invariance_numpy(logits: np.ndarray, group_index: np.ndarray, labels: np.ndarray, group_count: int) -> tuple[float, dict]:
    """Reference formula for unit tests and transparent inspection.

    Every nonempty group×class cell is a full training-epoch cell.  Eligible
    classes need ≥2 nonempty source cells.  We average cell deviations within
    a class, then average the eligible *classes* equally.
    """
    logits = np.asarray(logits, float).reshape(-1)
    group_index = np.asarray(group_index, int).reshape(-1)
    labels = np.asarray(labels, int).reshape(-1)
    if not (len(logits) == len(group_index) == len(labels)) or not np.isin(labels, (0, 1)).all():
        raise ValueError("invalid logits/group_index/binary labels")
    counts = np.zeros((group_count, 2), int)
    sums = np.zeros((group_count, 2), float)
    np.add.at(counts, (group_index, labels), 1)
    np.add.at(sums, (group_index, labels), logits)
    active = counts > 0
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=active)
    eligible = active.sum(axis=0) >= 2
    per_class = np.full(2, np.nan)
    for class_value in (0, 1):
        present = active[:, class_value]
        if eligible[class_value]:
            center = means[present, class_value].mean()
            per_class[class_value] = np.mean((means[present, class_value] - center) ** 2)
    penalty = float(np.nanmean(per_class[eligible]))
    return penalty, {
        "cell_counts": counts,
        "cell_means": means,
        "active_cell": active,
        "eligible_class": eligible,
        "per_class_variance": per_class,
        "eligible_class_count": int(eligible.sum()),
    }


def _make_source_balanced_step(optimizer):
    """Exact BCE baseline path retained in the new training module."""
    @jax.jit
    def step(current, optimizer_state, batch_x, batch_y):
        def loss_fn(candidate):
            # BASELINE_BCE_PRESERVED:
            # loss = jnp.mean(base.cv.optax.sigmoid_binary_cross_entropy(
            #     base.cv.logits(candidate, batch_x), batch_y
            # ))
            # The formerly active ordinary BCE line is intentionally retained
            # above; this source-balanced path executes the same objective.
            loss = jnp.mean(base.cv.optax.sigmoid_binary_cross_entropy(
                base.cv.logits(candidate, batch_x), batch_y
            ))
            return loss
        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, current)
        return base.cv.optax.apply_updates(current, updates), next_optimizer_state, loss
    return step


def _make_conditional_invariance_step(optimizer, layout: dict, lambda_invariance: float):
    """BCE on source-balanced draw plus complete-training-cell invariance term."""
    group_count = len(layout["groups"])
    group_index = jnp.asarray(layout["group_index"], jnp.int32)
    labels = jnp.asarray(layout["labels"], jnp.int32)
    cell_counts = jnp.asarray(layout["cell_counts"], jnp.float32)
    active = jnp.asarray(layout["active_cell"])
    eligible = jnp.asarray(layout["eligible_class"])
    cell_id = group_index * 2 + labels
    lam = jnp.asarray(float(lambda_invariance), jnp.float32)
    eligible_count = int(layout["eligible_class_count"])

    def invariance(logits):
        # Full train-fold logits only: every source×class cell uses all its
        # naturally saved variants, never a validation/test or sparse sampled
        # batch cell.
        sums = jnp.zeros((group_count * 2,), dtype=logits.dtype).at[cell_id].add(logits)
        # Empty cells use a harmless denominator/value and are removed by
        # ``active`` below.  This avoids NaN gradients from 0/0 before the
        # missing-cell mask is applied.
        safe_cell_counts = jnp.where(active, cell_counts, 1.0)
        means = (sums.reshape(group_count, 2) / safe_cell_counts)
        masked_means = jnp.where(active, means, 0.0)
        count_by_class = jnp.sum(active, axis=0)
        safe_class_count = jnp.maximum(count_by_class, 1)
        class_centers = jnp.sum(masked_means, axis=0) / safe_class_count
        squared = (means - class_centers[None, :]) ** 2
        per_class = jnp.sum(jnp.where(active, squared, 0.0), axis=0) / safe_class_count
        # Class-conditioning is intentional: the average below includes no
        # cross-class mean/logit comparison, so it cannot reward a single
        # class-unconditional logit collapse.
        penalty = jnp.sum(jnp.where(eligible, per_class, 0.0)) / eligible_count
        return penalty, per_class, means

    @jax.jit
    def step(current, optimizer_state, sampled_x, sampled_y, full_train_x):
        def loss_fn(candidate):
            sampled_logits = base.cv.logits(candidate, sampled_x)
            per_example_bce = base.cv.optax.sigmoid_binary_cross_entropy(sampled_logits, sampled_y)
            # BASELINE_BCE_PRESERVED:
            # bce = jnp.mean(per_example_bce)
            # The per-example BCE is unchanged.  Only the explicitly
            # configured generic class-conditional source aggregation below
            # is added for lambda > 0.
            bce = jnp.mean(per_example_bce)
            penalty, per_class, means = invariance(base.cv.logits(candidate, full_train_x))
            loss = bce + lam * penalty
            return loss, (bce, penalty, per_class, means)
        (loss, aux), gradients = jax.value_and_grad(loss_fn, has_aux=True)(current)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, current)
        return base.cv.optax.apply_updates(current, updates), next_optimizer_state, loss, aux
    return step


def _state_predictions(params, fold: dict, context: dict, partition: str):
    index = fold[f"{partition}_index"]
    return base.cv.aggregate_state(
        base.cv.predict(params, fold["normalized"][index]), context["sample_ids"][index], context["label"]
    )


def train_one_seed(
    fold: dict,
    context: dict,
    *,
    loss_mode: str,
    seed: int,
    lambda_invariance: float,
    checkpoint_path: Path | None = None,
    keep_history: bool = True,
) -> dict[str, Any]:
    """Train one frozen fold and seed.  λ=0 exact old source-BCE execution."""
    if loss_mode not in LOSS_MODES:
        raise ValueError(("unsupported loss mode", loss_mode))
    if lambda_invariance not in LAMBDA_GRID:
        raise ValueError(("lambda not pre-registered", lambda_invariance, LAMBDA_GRID))
    if loss_mode == "baseline_bce" and lambda_invariance != 0.0:
        raise ValueError("baseline_bce requires lambda=0")
    if loss_mode == "source_balanced_bce" and lambda_invariance != 0.0:
        raise ValueError("source_balanced_bce requires lambda=0")
    if loss_mode == "conditional_source_invariance_bce" and lambda_invariance == 0.0:
        # Makes the λ=0 control unambiguous and prevents accidentally
        # compiling a mathematically equivalent but non-identical path.
        raise ValueError("lambda=0 must use source_balanced_bce exact baseline path")

    layout = class_cell_layout(fold, context)
    sample_label = context["sample_label"]
    train_index = fold["train_index"]
    full_train_x = jnp.asarray(fold["normalized"][train_index], jnp.float32)
    params = base.cv.init_params(jax.random.PRNGKey(seed), [214, *HIDDEN, 1])
    optimizer = base.cv.optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    optimizer_state = optimizer.init(params)
    sample_rng = np.random.default_rng(seed)
    history: list[dict] = []
    if loss_mode in ("baseline_bce", "source_balanced_bce"):
        step = _make_source_balanced_step(optimizer)
    else:
        step = _make_conditional_invariance_step(optimizer, layout, lambda_invariance)

    best = params
    best_validation_bce = float("inf")
    best_epoch = 0
    stale = 0
    epochs_ran = 0
    for epoch in range(1, 1201):
        epochs_ran = epoch
        if loss_mode == "baseline_bce":
            # Named retained baseline mode: the original sample-uniform full
            # training set BCE.  It is not a conditional-invariance candidate
            # but remains executable rather than a deleted/reconstructed path.
            sampled_x = full_train_x
            sampled_y = jnp.asarray(sample_label[train_index], jnp.float32)
        else:
            sampled = base.hierarchical_epoch_indices(fold, context, "SOURCE_GROUP_BALANCED", sample_rng)
            sampled_x = jnp.asarray(fold["normalized"][sampled], jnp.float32)
            sampled_y = jnp.asarray(sample_label[sampled], jnp.float32)
        validation_bce = None
        if loss_mode in ("baseline_bce", "source_balanced_bce"):
            params, optimizer_state, total_loss = step(params, optimizer_state, sampled_x, sampled_y)
            bce_value = float(total_loss)
            penalty_value = 0.0
            per_class = np.asarray([np.nan, np.nan])
        else:
            params, optimizer_state, total_loss, aux = step(params, optimizer_state, sampled_x, sampled_y, full_train_x)
            bce_raw, penalty_raw, per_class_raw, _ = aux
            bce_value = float(bce_raw)
            penalty_value = float(penalty_raw)
            per_class = np.asarray(per_class_raw, float)
        if epoch == 1 or epoch % 10 == 0:
            _, validation_probability, validation_y = _state_predictions(params, fold, context, "validation")
            validation_bce = float(base.cv.state_bce(validation_y, validation_probability))
            if validation_bce < best_validation_bce - 1e-6:
                best_validation_bce = validation_bce
                best_epoch = epoch
                best = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
                stale = 0
            else:
                stale += 1
        if keep_history:
            for class_value in (0, 1):
                history.append({
                    "fold_id": fold["fold_id"],
                    "heldout_source_group": fold["outer_group"],
                    "training_seed": seed,
                    "loss_mode": loss_mode,
                    "lambda_invariance": lambda_invariance,
                    "epoch": epoch,
                    "oracle_class": class_value,
                    "eligible_training_source_group_count": int(layout["eligible_group_count_by_class"][class_value]),
                    "class_included_in_penalty": bool(layout["eligible_class"][class_value]),
                    "sampled_source_balanced_BCE": bce_value,
                    "conditional_invariance_penalty": penalty_value,
                    "class_cell_logit_variance": float(per_class[class_value]),
                    "total_loss": float(total_loss),
                    "validation_state_BCE": validation_bce,
                    "uses_validation_or_outer_test_data": False,
                })
        if stale >= 35:
            break

    validation_ids, validation_probability, validation_y = _state_predictions(best, fold, context, "validation")
    test_ids, test_probability, test_y = _state_predictions(best, fold, context, "test")
    threshold = float(base.cv.select_threshold(validation_y, validation_probability))
    if not (np.isfinite(validation_probability).all() and np.isfinite(test_probability).all()):
        raise RuntimeError(("nonfinite output", fold["fold_id"], seed, loss_mode, lambda_invariance))
    metadata = {
        "condition": "CONDITIONAL_SOURCE_INVARIANCE_GATE",
        "loss_mode": loss_mode,
        "lambda_invariance": lambda_invariance,
        "architecture": [214, 64, 64, 1],
        "activation": "SiLU",
        "per_example_loss": "ordinary sigmoid BCE",
        "base_sampling": "uniform train source group -> uniform state in group -> uniform saved Flow variant",
        "invariance": ("disabled exact lambda=0 baseline" if lambda_invariance == 0 else
                       "mean over eligible classes of variance across nonempty complete training source-group/class mean logits"),
        "invariance_uses_training_groups_only": True,
        "eligible_group_counts_by_class": layout["eligible_group_count_by_class"].tolist(),
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)",
        "fold_id": fold["fold_id"], "outer_group": fold["outer_group"], "seed": seed,
        "best_epoch": best_epoch, "epochs_ran": epochs_ran,
        "validation_selected_threshold": threshold,
        "normalization": "existing train-only mean/std; boolean semantics preserved",
    }
    if checkpoint_path is not None:
        base.serialize_checkpoint(checkpoint_path, best, fold["mean"], fold["scale"], fold["binary"], metadata)
    return {
        "fold": fold, "seed": seed, "loss_mode": loss_mode, "lambda_invariance": lambda_invariance,
        "params": best, "threshold": threshold, "best_epoch": best_epoch, "epochs_ran": epochs_ran,
        "validation_bce": best_validation_bce,
        "validation": (validation_ids, validation_probability, validation_y),
        "test": (test_ids, test_probability, test_y), "layout": layout, "history": history,
    }


def seed_prediction_rows(result: dict, condition: str) -> list[dict]:
    state_ids, probabilities, labels = result["test"]
    threshold = float(result["threshold"])
    threshold_logit = float(base.logit(threshold))
    rows = []
    for state_id, probability, label in zip(state_ids, probabilities, labels):
        value_logit = float(base.logit(probability))
        predicted = int(probability >= threshold)
        rows.append({
            "condition": condition, "loss_mode": result["loss_mode"], "lambda_invariance": result["lambda_invariance"],
            "fold_id": result["fold"]["fold_id"], "heldout_source_group": result["fold"]["outer_group"],
            "state_id": str(state_id), "oracle_label": int(label), "p_gate": float(probability), "gate_logit": value_logit,
            "validation_selected_threshold": threshold, "validation_threshold_logit": threshold_logit,
            "fold_relative_logit_margin": value_logit - threshold_logit,
            "predicted_label": predicted, "correct": bool(predicted == int(label)), "training_seed": result["seed"],
        })
    return rows


def write_config(path: Path) -> None:
    base.write_json(path, {
        "architecture": [214, 64, 64, 1], "activation": "SiLU",
        "optimizer": {"family": "AdamW", "lr": 1e-3, "weight_decay": 1e-5},
        "seeds": list(SEEDS), "lambda_invariance_grid": list(LAMBDA_GRID),
        "loss_mode": list(LOSS_MODES),
        "base_loss": "ordinary per-example BCE under frozen hierarchical source-balanced sampling",
        "regularizer": {
            "name": "class_conditional_training_source_logit_invariance",
            "formula": "mean_{c: |G_c|>=2} mean_{g in G_c}(mu_{g,c}-mean_{h in G_c}mu_{h,c})^2",
            "mu": "mean gate logit across the complete current epoch training source-group/class cell",
            "cell_weighting": "equal nonempty source-group/class cells within each class; equal eligible classes",
            "missing_cell_policy": "skip empty cells; skip class entirely if fewer than two nonempty training source cells",
            "uses_validation_or_outer_test": False,
            "cross_class_logit_matching": False,
        },
        "normalization": "frozen train-only normalization", "threshold": "frozen validation-only selection",
        "new_states": 0, "new_oracle_rollouts": 0, "feature_schema_changed": False,
        "oracle_labels_changed": False, "correction_head_trained": False, "closed_loop_run": False,
    })


def write_preserved_loss_audit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""# Preserved BCE and conditional-invariance audit

The source-balanced BCE loss line remains in
`conditional_invariance_gate_runner.py` under the explicit marker
`BASELINE_BCE_PRESERVED`; it is commented, not deleted.  The
`source_balanced_bce` / lambda=0 branch executes the same original ordinary
per-example BCE after the frozen source-group→state→Flow-variant sampler.

For positive lambda, each full training epoch computes group/class mean logits
over the complete training fold.  It adds the variance of these means across
nonempty **training** source groups separately for oracle class 0 and oracle
class 1.  Empty cells are skipped, and a class with fewer than two training
source cells does not enter the penalty.  There is intentionally no penalty
between the two class means, no class-unconditional source statistic, and no
validation or outer-test statistic in either the regularizer or its eligibility
calculation.
""")
