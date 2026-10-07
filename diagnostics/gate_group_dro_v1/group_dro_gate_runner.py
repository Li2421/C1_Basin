"""Frozen source-group Group-DRO gate trainer.

This module is deliberately narrow: it reuses the seven pre-registered
source-group-held-out folds, 214-D inputs, stable oracle labels, model,
normalization, optimizer, validation rule, and threshold rule from
``gate_source_balancing_v1``.  The only new objective is the *aggregation*
of the unchanged per-example sigmoid BCE over **training source groups**.

Loss modes
----------
``source_balanced_bce``
    Exact existing hierarchical source-group -> state -> Flow-variant
    resampling condition.  It exists here solely to make the pre-change
    baseline reproducible after adding Group-DRO.
``group_dro_bce``
    Per-source-group mean BCEs are aggregated with exponentiated-gradient
    adversarial weights q.  q is updated only from training-source losses.

No state-specific weight, test loss, validation loss, or oracle/feature
change enters either objective.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SOURCE_AUDIT = ROOT / "diagnostics" / "gate_source_balancing_v1"
sys.path.insert(0, str(SOURCE_AUDIT))
from shared import gate_condition_runner as base  # noqa: E402


SEEDS = base.SEEDS
HIDDEN = base.HIDDEN
LOSS_MODES = ("baseline_bce", "source_balanced_bce", "group_dro_bce")
DEFAULT_ETA_Q = (0.01, 0.05, 0.1)


def load_context_and_folds() -> tuple[dict, list[dict]]:
    """Load the unmodified frozen data and enforce all seven exact folds."""
    context = base.load_context()
    folds = base.prepare_folds(context)
    if len(folds) != 7:
        raise RuntimeError(("expected seven frozen LOGO folds", len(folds)))
    return context, folds


def source_group_layout(fold: dict, context: dict) -> dict[str, Any]:
    """Map each frozen training row to one and only one training source group."""
    sample_ids = context["sample_ids"]
    source_by_state = {state_id: row["source_group"] for state_id, row in context["state"].items()}
    group_per_row = np.asarray([source_by_state[str(state_id)] for state_id in sample_ids[fold["train_index"]]], dtype=object)
    groups = np.asarray(sorted(set(group_per_row.tolist())), dtype=object)
    expected = np.asarray(sorted(fold["train_groups"]), dtype=object)
    if not np.array_equal(groups, expected):
        raise RuntimeError(("training source-group mismatch", fold["fold_id"], groups.tolist(), expected.tolist()))
    lookup = {str(group): position for position, group in enumerate(groups)}
    group_index = np.asarray([lookup[str(group)] for group in group_per_row], dtype=np.int32)
    counts = np.bincount(group_index, minlength=len(groups)).astype(np.float32)
    if np.any(counts <= 0):
        raise RuntimeError(("empty training source group", fold["fold_id"], groups.tolist(), counts.tolist()))
    outer = fold["outer_group"]
    if outer in lookup or outer in fold["validation_groups"]:
        raise RuntimeError(("outer group leaked into Group-DRO layout", fold["fold_id"], outer))
    return {"groups": groups, "group_index": group_index, "counts": counts}


def _copy_params(params):
    return jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)


def _state_predictions(params, fold: dict, context: dict, partition: str):
    """Use the frozen state aggregation semantics verbatim."""
    if partition not in ("validation", "test"):
        raise ValueError(partition)
    index = fold[f"{partition}_index"]
    return base.cv.aggregate_state(
        base.cv.predict(params, fold["normalized"][index]),
        context["sample_ids"][index],
        context["label"],
    )


def _validation_bce(params, fold: dict, context: dict) -> float:
    _, probability, target = _state_predictions(params, fold, context, "validation")
    return float(base.cv.state_bce(target, probability))


def _make_source_balanced_step(optimizer):
    """Exact old active BCE path, explicitly preserved for reproducibility."""
    @jax.jit
    def step(current, optimizer_state, batch_x, batch_y):
        def loss_fn(candidate):
            # BASELINE_BCE_PRESERVED:
            # loss = jnp.mean(base.cv.optax.sigmoid_binary_cross_entropy(
            #     base.cv.logits(candidate, batch_x), batch_y
            # ))
            # The former active BCE line is intentionally retained above.
            # In source_balanced_bce mode it remains the actual objective.
            loss = jnp.mean(base.cv.optax.sigmoid_binary_cross_entropy(
                base.cv.logits(candidate, batch_x), batch_y
            ))
            return loss
        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, new_state = optimizer.update(gradients, optimizer_state, current)
        return base.cv.optax.apply_updates(current, updates), new_state, loss
    return step


def _make_group_dro_step(optimizer, group_index: np.ndarray, counts: np.ndarray, eta_q: float):
    """One full-train-set Group-DRO update with log-space EG q updates.

    Updating log_q and applying softmax is algebraically identical to
    q <- q*exp(eta_q L_g); normalize, but avoids overflow for long runs.
    ``group_index``/``counts`` cover only frozen training samples/groups.
    """
    group_index_j = jnp.asarray(group_index, jnp.int32)
    counts_j = jnp.asarray(counts, jnp.float32)
    group_count = len(counts)
    eta = jnp.asarray(float(eta_q), jnp.float32)

    def group_losses(candidate, train_x, train_y):
        per_example = base.cv.optax.sigmoid_binary_cross_entropy(base.cv.logits(candidate, train_x), train_y)
        totals = jnp.zeros((group_count,), dtype=per_example.dtype).at[group_index_j].add(per_example)
        return totals / counts_j

    @jax.jit
    def step(current, optimizer_state, q, log_q, train_x, train_y):
        def loss_fn(candidate):
            losses_by_group = group_losses(candidate, train_x, train_y)
            # BASELINE_BCE_PRESERVED:
            # loss = jnp.mean(base.cv.optax.sigmoid_binary_cross_entropy(
            #     base.cv.logits(candidate, train_x), train_y
            # ))
            # The baseline per-example BCE remains unchanged; Group-DRO only
            # changes its aggregation below.
            loss = jnp.sum(q * losses_by_group)
            return loss, losses_by_group
        (loss, losses_by_group), gradients = jax.value_and_grad(loss_fn, has_aux=True)(current)
        updates, new_state = optimizer.update(gradients, optimizer_state, current)
        new_params = base.cv.optax.apply_updates(current, updates)
        # stop_gradient makes q an adversarial moving statistic, never a
        # differentiated recurrent optimization path.
        updated_log_q = log_q + eta * jax.lax.stop_gradient(losses_by_group)
        updated_log_q = updated_log_q - jax.scipy.special.logsumexp(updated_log_q)
        updated_q = jnp.exp(updated_log_q)
        return new_params, new_state, updated_q, updated_log_q, loss, losses_by_group
    return step


def train_one_seed(
    fold: dict,
    context: dict,
    *,
    loss_mode: str,
    seed: int,
    eta_q: float | None = None,
    checkpoint_path: Path | None = None,
    keep_group_history: bool = True,
) -> dict[str, Any]:
    """Train one frozen fold/seed, without ever reading validation/test loss in q.

    ``source_balanced_bce`` preserves the prior sampled full-batch condition
    exactly.  ``group_dro_bce`` uses the same number of full-batch optimizer
    updates and early-stop cadence, but each update evaluates L_g over each
    training source group and aggregates by q.
    """
    if loss_mode not in LOSS_MODES:
        raise ValueError(("unknown loss mode", loss_mode))
    if loss_mode == "group_dro_bce" and eta_q not in DEFAULT_ETA_Q:
        raise ValueError(("eta_q must be pre-registered", eta_q, DEFAULT_ETA_Q))

    layout = source_group_layout(fold, context)
    sample_label = context["sample_label"]
    train_index = fold["train_index"]
    train_x = jnp.asarray(fold["normalized"][train_index], jnp.float32)
    train_y = jnp.asarray(sample_label[train_index], jnp.float32)
    params = base.cv.init_params(jax.random.PRNGKey(seed), [214, *HIDDEN, 1])
    optimizer = base.cv.optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    optimizer_state = optimizer.init(params)
    sample_rng = np.random.default_rng(seed)
    group_history: list[dict] = []

    if loss_mode in ("baseline_bce", "source_balanced_bce"):
        step = _make_source_balanced_step(optimizer)
        q = None
        log_q = None
    else:
        step = _make_group_dro_step(optimizer, layout["group_index"], layout["counts"], float(eta_q))
        q = jnp.full((len(layout["groups"]),), 1.0 / len(layout["groups"]), dtype=jnp.float32)
        log_q = jnp.full((len(layout["groups"]),), -math.log(len(layout["groups"])), dtype=jnp.float32)

    best = params
    best_validation_bce = float("inf")
    best_epoch = 0
    stale = 0
    epochs_ran = 0
    for epoch in range(1, 1201):
        epochs_ran = epoch
        if loss_mode in ("baseline_bce", "source_balanced_bce"):
            if loss_mode == "baseline_bce":
                # Exact ordinary full-sample BCE condition; the retained
                # baseline branch never changes sample or source weights.
                batch_x, batch_y = train_x, train_y
            else:
                sampled = base.hierarchical_epoch_indices(fold, context, "SOURCE_GROUP_BALANCED", sample_rng)
                batch_x = jnp.asarray(fold["normalized"][sampled], jnp.float32)
                batch_y = jnp.asarray(sample_label[sampled], jnp.float32)
            params, optimizer_state, objective = step(params, optimizer_state, batch_x, batch_y)
            losses_by_group = None
        else:
            params, optimizer_state, q, log_q, objective, losses_by_group = step(
                params, optimizer_state, q, log_q, train_x, train_y
            )

        validation_bce: float | None = None
        if epoch == 1 or epoch % 10 == 0:
            validation_bce = _validation_bce(params, fold, context)
            if validation_bce < best_validation_bce - 1e-6:
                best_validation_bce = validation_bce
                best_epoch = epoch
                best = _copy_params(params)
                stale = 0
            else:
                stale += 1
        if keep_group_history and loss_mode == "group_dro_bce":
            q_np = np.asarray(q, float)
            loss_np = np.asarray(losses_by_group, float)
            entropy = float(-np.sum(np.where(q_np > 0, q_np * np.log(q_np), 0.0)))
            effective_group_count = float(np.exp(entropy))
            for position, group in enumerate(layout["groups"]):
                group_history.append({
                    "fold_id": fold["fold_id"],
                    "heldout_source_group": fold["outer_group"],
                    "training_seed": seed,
                    "loss_mode": loss_mode,
                    "eta_q": float(eta_q),
                    "epoch": epoch,
                    "training_source_group": str(group),
                    "q_g": float(q_np[position]),
                    "L_g": float(loss_np[position]),
                    "objective": float(objective),
                    "q_entropy": entropy,
                    "q_effective_group_count": effective_group_count,
                    "validation_state_BCE": validation_bce,
                })
        if stale >= 35:
            break

    validation_ids, validation_probability, validation_y = _state_predictions(best, fold, context, "validation")
    test_ids, test_probability, test_y = _state_predictions(best, fold, context, "test")
    threshold = float(base.cv.select_threshold(validation_y, validation_probability))
    if not (np.isfinite(validation_probability).all() and np.isfinite(test_probability).all()):
        raise RuntimeError(("nonfinite output", loss_mode, fold["fold_id"], seed))

    metadata = {
        "condition": "GROUP_DRO_GATE_TRAINING",
        "loss_mode": loss_mode,
        "eta_q": eta_q,
        "architecture": [214, 64, 64, 1],
        "activation": "SiLU",
        "per_example_loss": "ordinary sigmoid BCE",
        "group_aggregation": ("ordinary sample-uniform mean BCE" if loss_mode == "baseline_bce" else
                              "uniform sampled source-group BCE" if loss_mode == "source_balanced_bce" else
                              "sum_g q_g * mean_{i in g} BCE_i"),
        "q_update": None if loss_mode in ("baseline_bce", "source_balanced_bce") else "q <- normalize(q*exp(eta_q*stop_gradient(L_g))); log-space stable equivalent",
        "training_groups_only": True,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)",
        "seed": seed,
        "fold_id": fold["fold_id"],
        "outer_group": fold["outer_group"],
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "validation_selected_threshold": threshold,
        "normalization": "train-only mean/std; boolean dimensions literal",
    }
    if checkpoint_path is not None:
        base.serialize_checkpoint(checkpoint_path, best, fold["mean"], fold["scale"], fold["binary"], metadata)
    return {
        "fold": fold,
        "seed": seed,
        "loss_mode": loss_mode,
        "eta_q": eta_q,
        "params": best,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "validation_bce": best_validation_bce,
        "validation": (validation_ids, validation_probability, validation_y),
        "test": (test_ids, test_probability, test_y),
        "group_layout": layout,
        "group_history": group_history,
    }


def seed_prediction_rows(result: dict, context: dict, condition: str) -> list[dict]:
    """State-level OOF rows with fold-relative margins; raw fold scores remain separate."""
    state_ids, probabilities, labels = result["test"]
    threshold = float(result["threshold"])
    threshold_logit = float(base.logit(threshold))
    rows = []
    for state_id, probability, label in zip(state_ids, probabilities, labels):
        raw_logit = float(base.logit(probability))
        predicted = int(probability >= threshold)
        rows.append({
            "condition": condition,
            "loss_mode": result["loss_mode"],
            "eta_q": result["eta_q"],
            "fold_id": result["fold"]["fold_id"],
            "heldout_source_group": result["fold"]["outer_group"],
            "state_id": str(state_id),
            "oracle_label": int(label),
            "p_gate": float(probability),
            "gate_logit": raw_logit,
            "validation_selected_threshold": threshold,
            "validation_threshold_logit": threshold_logit,
            "fold_relative_logit_margin": raw_logit - threshold_logit,
            "predicted_label": predicted,
            "correct": bool(predicted == int(label)),
            "training_seed": result["seed"],
        })
    return rows


def write_config_template(path: Path) -> None:
    """Write frozen configuration so the training worker cannot silently drift."""
    base.write_json(path, {
        "architecture": [214, 64, 64, 1],
        "activation": "SiLU",
        "optimizer": {"family": "AdamW", "lr": 1e-3, "weight_decay": 1e-5},
        "seeds": list(SEEDS),
        "loss_mode": ["baseline_bce", "source_balanced_bce", "group_dro_bce"],
        "group_dro_eta_q_candidates": list(DEFAULT_ETA_Q),
        "per_example_loss": "BCE(logit, y)",
        "group_dro": {
            "L_g": "mean BCE over examples in training source group g",
            "initial_q": "uniform over training source groups",
            "update": "q <- normalize(q * exp(eta_q * stop_gradient(L_g)))",
            "loss": "sum_g q_g L_g",
            "q_uses_validation_or_outer_test": False,
        },
        "folds": "exact seven frozen source-group-held-out folds copied unchanged",
        "normalization": "train stable states only; existing semantics",
        "early_stopping": "existing validation-state BCE, check epoch 1 then every 10 epochs, patience 35 checks",
        "threshold": "existing validation-only rule",
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "feature_schema_changed": False,
        "oracle_labels_changed": False,
        "correction_head_trained": False,
        "closed_loop_run": False,
    })


def write_preserved_loss_audit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""# Preserved BCE-loss audit

`group_dro_gate_runner.py` preserves the former active source-balanced BCE
line verbatim under the marker `BASELINE_BCE_PRESERVED` in both JIT step
implementations.  The original line was commented rather than deleted.

`loss_mode=source_balanced_bce` executes that same ordinary per-example BCE
after the already frozen source-group → state → Flow-variant sampler.

`loss_mode=group_dro_bce` leaves the per-example BCE unchanged and changes
only aggregation: it computes a mean loss for every **training** source
group, then uses `sum_g q_g L_g`.  `q` starts uniform and is updated in
log-space by the documented exponentiated-gradient equivalent of
`q_g <- q_g exp(eta_q stop_gradient(L_g))`, normalized over training groups.

No loss, q, normalization, validation threshold, or early-stop operation
reads validation or outer-test groups except the pre-existing validation
evaluation used strictly for early stopping and threshold selection.
""")
