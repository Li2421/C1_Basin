"""Safe orchestration for single-segment entry/exit decision heads.

This module is intentionally data-driven.  Importing it launches no rollout,
training, or final-test evaluation.  Training functions fail closed unless all
paired branch records are complete, source-grouped, and restricted to the
train/validation splits.  Full-loop analysis defaults to validation or
calibration and refuses final-test rows unless the caller explicitly opts in.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from decision_learning import (
    DEFAULT_EQUIVALENCE_CI_LEVEL,
    DEFAULT_EQUIVALENCE_MARGIN,
    DEFAULT_MIN_MATCHED_PAIRS,
    LAMBDA_BREAK_FAMILY,
    DecisionEvidence,
    DeformationNormalizer,
    MatchedBranchPair,
    pair_branch_records,
    paired_success_difference_interval,
    predict_head_logit,
    predict_head_probability,
    save_decision_head_checkpoint,
    summarize_decision_evidence,
    train_decision_head,
)


TRAINING_SEEDS: tuple[int, ...] = (17, 23, 41)
MAXIMUM_EPOCHS = 1200
PREDECLARED_SCORE_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75)
ALLOWED_TRAINING_SPLITS = frozenset({"train", "validation"})
ALLOWED_ANALYSIS_SPLITS = frozenset({"validation", "calibration"})
HEAD_INPUT_DIMENSIONS = {
    "entry_g": 214,
    "entry_eta": 214,
    "exit_g": 214,
    "exit_eta": 217,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_jsonable
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_threshold_variant(
    source_checkpoint: str | Path,
    destination_checkpoint: str | Path,
    *,
    threshold_probability: float,
) -> dict[str, Any]:
    """Create an immutable, hashed deployment variant at a frozen threshold.

    Network weights and train-only normalization are copied byte-for-value;
    only the scalar deployment threshold and its audit metadata change.  This
    makes every predeclared entry/exit threshold pair a distinct policy hash
    before complete full-loop validation.
    """

    threshold_probability = float(threshold_probability)
    if threshold_probability not in PREDECLARED_SCORE_THRESHOLDS:
        raise ValueError(
            f"threshold must be one of {PREDECLARED_SCORE_THRESHOLDS}, "
            f"got {threshold_probability}"
        )
    source_checkpoint = Path(source_checkpoint)
    destination_checkpoint = Path(destination_checkpoint)
    if destination_checkpoint.exists():
        raise FileExistsError(destination_checkpoint)
    threshold_logit = float(
        np.log(threshold_probability) - np.log1p(-threshold_probability)
    )
    with np.load(source_checkpoint, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "normalization_mean", "normalization_scale",
        "layer_0_weight", "layer_0_bias",
        "layer_1_weight", "layer_1_bias",
        "layer_2_weight", "layer_2_bias",
        "metadata_json",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint is not a FrozenHead-compatible head: {missing}")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    metadata.update(
        {
            "threshold_probability": threshold_probability,
            "threshold_logit": threshold_logit,
            "threshold_variant_source_sha256": file_hash(source_checkpoint),
            "threshold_variant_selection_split": "validation",
        }
    )
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    payload["threshold_logit"] = np.asarray(threshold_logit, dtype=np.float64)
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination_checkpoint, **payload)
    manifest = {
        "schema": "single_segment_threshold_variant_v1",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_sha256": file_hash(source_checkpoint),
        "checkpoint_path": str(destination_checkpoint.resolve()),
        "checkpoint_sha256": file_hash(destination_checkpoint),
        "threshold_probability": threshold_probability,
        "threshold_logit": threshold_logit,
        "predeclared_threshold_grid": list(PREDECLARED_SCORE_THRESHOLDS),
        "weights_changed": False,
        "normalization_changed": False,
        "test_data_used": False,
    }
    manifest["policy_sha256"] = content_hash(manifest)
    return manifest


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n"
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class HeadDataset:
    """Complete paired evidence for one head/system across train and validation."""

    head_kind: str
    input_dimension: int
    train_features: np.ndarray
    train_evidence: tuple[DecisionEvidence, ...]
    train_rows: tuple[dict[str, Any], ...]
    validation_features: np.ndarray
    validation_evidence: tuple[DecisionEvidence, ...]
    validation_rows: tuple[dict[str, Any], ...]
    deformation_normalizer: DeformationNormalizer
    manifest: dict[str, Any]


def _require_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def target_lineage_id(row: Mapping[str, Any]) -> str:
    """Keep policy-iteration targets distinct in every aggregation."""

    decision_id = _require_text(row, "decision_id")
    pass_id = _require_text(row, "pass_id")
    downstream = _require_text(row, "downstream_policy_hash")
    return f"{decision_id}::pass={pass_id}::downstream={downstream}"


def _feature_from_row(row: Mapping[str, Any], head_kind: str) -> np.ndarray:
    if head_kind not in HEAD_INPUT_DIMENSIONS:
        raise ValueError(f"unknown head_kind {head_kind!r}")
    feature = np.asarray(row["feature"], dtype=np.float32).reshape(-1)
    if head_kind == "exit_eta" and feature.size == 214:
        eta = np.asarray(row.get("eta_latched", []), dtype=np.float32).reshape(-1)
        if eta.size != 3:
            raise ValueError("exit_eta requires feature[214] plus eta_latched[3], or feature[217]")
        feature = np.concatenate([feature, eta])
    expected = HEAD_INPUT_DIMENSIONS[head_kind]
    if feature.size != expected:
        raise ValueError(f"{head_kind} expects {expected} features, received {feature.size}")
    if not np.all(np.isfinite(feature)):
        raise ValueError("decision features must be finite")
    return feature


def _audit_source_split(rows: Sequence[Mapping[str, Any]], *, training: bool) -> dict[str, str]:
    root_to_split: dict[str, str] = {}
    allowed = ALLOWED_TRAINING_SPLITS if training else ALLOWED_ANALYSIS_SPLITS
    for row in rows:
        source = _require_text(row, "root_source_id")
        split = _require_text(row, "split")
        if split not in allowed:
            raise ValueError(
                f"split {split!r} is forbidden in this {'training' if training else 'analysis'} operation"
            )
        previous = root_to_split.setdefault(source, split)
        if previous != split:
            raise ValueError(f"root source {source} crosses splits: {previous}, {split}")
    return root_to_split


def build_head_dataset(
    decision_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    head_kind: str,
    expected_futures_per_input: int = DEFAULT_MIN_MATCHED_PAIRS,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    equivalence_ci_level: float = DEFAULT_EQUIVALENCE_CI_LEVEL,
    require_validation: bool = True,
) -> HeadDataset:
    """Validate paired labels and construct a leakage-free head dataset.

    ``decision_rows`` has one row per decision input/policy pass.  Branch rows
    have one row per action and future stream.  The composite target lineage
    includes the downstream policy hash; records from different policy passes
    are never silently merged.
    """

    if expected_futures_per_input <= 0:
        raise ValueError("expected_futures_per_input must be positive")
    if head_kind not in HEAD_INPUT_DIMENSIONS:
        raise ValueError(f"unknown head_kind {head_kind!r}")
    decision_rows = list(decision_rows)
    branch_rows = list(branch_rows)
    if not decision_rows or not branch_rows:
        raise ValueError("decision and paired-branch files must both be complete and nonempty")
    _audit_source_split(decision_rows, training=True)
    _audit_source_split(branch_rows, training=True)

    decisions: dict[str, dict[str, Any]] = {}
    source_split: dict[str, str] = {}
    for original in decision_rows:
        row = dict(original)
        target_id = target_lineage_id(row)
        if target_id in decisions:
            raise ValueError(f"duplicate decision lineage: {target_id}")
        split = _require_text(row, "split")
        source = _require_text(row, "root_source_id")
        prior_split = source_split.setdefault(source, split)
        if prior_split != split:
            raise ValueError(f"root source {source} crosses splits")
        row["target_lineage_id"] = target_id
        row["feature_array"] = _feature_from_row(row, head_kind)
        decisions[target_id] = row

    normalized_branches: list[dict[str, Any]] = []
    for original in branch_rows:
        row = dict(original)
        target_id = target_lineage_id(row)
        if target_id not in decisions:
            raise ValueError(f"branch has no decision-state row: {target_id}")
        decision = decisions[target_id]
        for key in ("root_source_id", "split", "pass_id", "downstream_policy_hash"):
            if str(row.get(key, "")) != str(decision.get(key, "")):
                raise ValueError(f"{target_id}: branch/decision mismatch for {key}")
        row["target_lineage_id"] = target_id
        normalized_branches.append(row)

    grouped = pair_branch_records(
        normalized_branches,
        decision_key="target_lineage_id",
        invariant_keys=(
            "root_source_id",
            "split",
            "state_hash",
            "absolute_step",
            "mode",
            "current_flow_realization_id",
            "future_rng_state_hash",
            "eta_latched_hash",
            "pass_id",
            "downstream_policy_hash",
        ),
    )
    if set(grouped) != set(decisions):
        missing = sorted(set(decisions) - set(grouped))
        extra = sorted(set(grouped) - set(decisions))
        raise ValueError(f"incomplete decision/branch coverage; missing={missing}, extra={extra}")
    for target_id, pairs in grouped.items():
        if len(pairs) != expected_futures_per_input:
            raise ValueError(
                f"{target_id}: expected {expected_futures_per_input} matched futures, got {len(pairs)}"
            )

    training_pairs = [
        pair
        for target_id, pairs in grouped.items()
        if decisions[target_id]["split"] == "train"
        for pair in pairs
    ]
    if not training_pairs:
        raise ValueError("no training paired branches")
    deformation_normalizer = DeformationNormalizer.fit(training_pairs, split="train")

    by_split: dict[str, list[tuple[dict[str, Any], DecisionEvidence]]] = {
        "train": [],
        "validation": [],
    }
    for target_id in sorted(decisions):
        decision = decisions[target_id]
        evidence = summarize_decision_evidence(
            target_id,
            grouped[target_id],
            deformation_normalizer=deformation_normalizer,
            equivalence_margin=equivalence_margin,
            equivalence_ci_level=equivalence_ci_level,
            minimum_pairs=expected_futures_per_input,
            downstream_policy_hash=str(decision["downstream_policy_hash"]),
        )
        by_split[decision["split"]].append((decision, evidence))
    if not by_split["train"]:
        raise ValueError("training decision sources are required")
    if require_validation and not by_split["validation"]:
        raise ValueError("both train and validation decision sources are required")

    def unpack(split: str):
        if not by_split[split]:
            return (
                np.empty((0, HEAD_INPUT_DIMENSIONS[head_kind]), dtype=np.float32),
                tuple(),
                tuple(),
            )
        rows = tuple(row for row, _ in by_split[split])
        evidence = tuple(item for _, item in by_split[split])
        features = np.stack([row["feature_array"] for row in rows]).astype(np.float32)
        clean_rows = tuple(
            {
                key: value
                for key, value in row.items()
                if key not in {"feature", "feature_array"}
            }
            for row in rows
        )
        return features, evidence, clean_rows

    train_features, train_evidence, train_rows = unpack("train")
    validation_features, validation_evidence, validation_rows = unpack("validation")
    train_sources = sorted({row["root_source_id"] for row in train_rows})
    validation_sources = sorted({row["root_source_id"] for row in validation_rows})
    if set(train_sources) & set(validation_sources):
        raise AssertionError("train/validation root-source leakage")
    lineage_counts: dict[str, int] = {}
    for row in (*train_rows, *validation_rows):
        key = str(row["downstream_policy_hash"])
        lineage_counts[key] = lineage_counts.get(key, 0) + 1
    manifest = {
        "schema": "single_segment_head_dataset_v1",
        "head_kind": head_kind,
        "input_dimension": HEAD_INPUT_DIMENSIONS[head_kind],
        "expected_futures_per_input": expected_futures_per_input,
        "complete_paired_branches": True,
        "train_decision_inputs": len(train_rows),
        "validation_decision_inputs": len(validation_rows),
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "train_validation_source_overlap": 0,
        "branch_rollouts": len(normalized_branches),
        "downstream_policy_lineage_counts": lineage_counts,
        "lineage_retained_not_merged": True,
        "equivalence_margin": equivalence_margin,
        "equivalence_ci_level": equivalence_ci_level,
        "deformation_normalization": {
            "fit_split": deformation_normalizer.fit_split,
            "method": deformation_normalizer.method,
            "scale": deformation_normalizer.scale,
            "pair_count": deformation_normalizer.pair_count,
        },
        "unresolved_train": sum(item.unresolved for item in train_evidence),
        "unresolved_validation": sum(item.unresolved for item in validation_evidence),
    }
    manifest["content_sha256"] = content_hash(manifest)
    return HeadDataset(
        head_kind=head_kind,
        input_dimension=HEAD_INPUT_DIMENSIONS[head_kind],
        train_features=train_features,
        train_evidence=train_evidence,
        train_rows=train_rows,
        validation_features=validation_features,
        validation_evidence=validation_evidence,
        validation_rows=validation_rows,
        deformation_normalizer=deformation_normalizer,
        manifest=manifest,
    )


def threshold_candidates(probability: np.ndarray) -> np.ndarray:
    """Return the threshold grid frozen before paired outcomes were observed."""

    probability = np.sort(np.unique(np.asarray(probability, dtype=np.float64)))
    if probability.size == 0 or not np.all(np.isfinite(probability)):
        raise ValueError("finite validation scores are required")
    return np.asarray(PREDECLARED_SCORE_THRESHOLDS, dtype=np.float64)


def validation_threshold_table(
    probability: np.ndarray,
    evidence: Sequence[DecisionEvidence],
    *,
    lambda_break: float,
) -> list[dict[str, Any]]:
    """Evaluate thresholds using validation counterfactual evidence only."""

    if lambda_break not in LAMBDA_BREAK_FAMILY:
        raise ValueError(f"lambda_break must be in {LAMBDA_BREAK_FAMILY}")
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if probability.size != len(evidence):
        raise ValueError("one score is required per validation decision input")
    r = np.asarray([row.r for row in evidence], dtype=np.float64)
    b = np.asarray([row.b for row in evidence], dtype=np.float64)
    unresolved = np.asarray([row.unresolved for row in evidence], dtype=bool)
    deform_target = np.asarray([row.deformation_target for row in evidence], dtype=np.float64)
    deform_weight = np.asarray([row.deformation_weight for row in evidence], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(probability):
        event = probability >= threshold
        rescue = float(np.mean(event * r))
        breaks = float(np.mean(event * b))
        # This is a validation ranking statistic, not task success probability.
        cost_sensitive_gain = rescue - (1.0 + lambda_break) * breaks
        deformation_preference_agreement = float(
            np.mean(
                event * deform_weight * deform_target
                + (~event) * deform_weight * (1.0 - deform_target)
            )
        )
        rows.append(
            {
                "threshold": float(threshold),
                "event_count": int(np.sum(event)),
                "event_fraction": float(np.mean(event)),
                "estimated_rescue": rescue,
                "estimated_break": breaks,
                "estimated_net_success": rescue - breaks,
                "cost_sensitive_gain": cost_sensitive_gain,
                "deformation_preference_agreement": deformation_preference_agreement,
                "triggered_unresolved_count": int(np.sum(event & unresolved)),
                "triggered_unresolved_fraction": float(np.mean(event & unresolved)),
            }
        )
    return rows


def choose_validation_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic validation-only threshold selection.

    The frozen protocol maximizes paired net-success evidence ``r-b``.  Ties
    resolve by less observed break evidence and then the higher threshold.
    """

    if not rows:
        raise ValueError("threshold table is empty")
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row["estimated_net_success"]),
                -float(row["estimated_break"]),
                float(row["threshold"]),
            ),
        )
    )


def run_training_grid(
    dataset: HeadDataset,
    output_directory: str | Path,
    *,
    seeds: Sequence[int] = TRAINING_SEEDS,
    lambda_break_family: Sequence[float] = LAMBDA_BREAK_FAMILY,
    epochs: int = MAXIMUM_EPOCHS,
    allow_test_configuration: bool = False,
) -> dict[str, Any]:
    """Train all predeclared heads and prepare full-loop validation candidates.

    Epoch and threshold choices use validation decision branches only.  The
    returned provisional choice is *not* a final policy selection: candidate
    systems must subsequently be compared by complete validation episodes via
    :func:`select_policy_by_full_loop_validation`.
    """

    if epochs <= 0 or epochs > MAXIMUM_EPOCHS:
        raise ValueError(f"epochs must lie in [1, {MAXIMUM_EPOCHS}]")
    if not allow_test_configuration and tuple(seeds) != TRAINING_SEEDS:
        raise ValueError(f"production seeds must be {TRAINING_SEEDS}")
    if not allow_test_configuration and tuple(lambda_break_family) != LAMBDA_BREAK_FAMILY:
        raise ValueError(f"production lambda_break family must be {LAMBDA_BREAK_FAMILY}")
    if any(value not in LAMBDA_BREAK_FAMILY for value in lambda_break_family):
        raise ValueError(f"lambda_break family must be a subset of {LAMBDA_BREAK_FAMILY}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if dataset.manifest.get("complete_paired_branches") is not True:
        raise ValueError("refusing to train against incomplete/mutating label manifests")

    candidates: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    for lambda_break in lambda_break_family:
        for seed in seeds:
            params, normalization, history, metadata = train_decision_head(
                dataset.train_features,
                dataset.train_evidence,
                dataset.validation_features,
                dataset.validation_evidence,
                seed=int(seed),
                lambda_break=float(lambda_break),
                epochs=epochs,
            )
            run_id = f"{dataset.head_kind}_lambda{lambda_break:g}_seed{seed}"
            checkpoint_path = output_directory / f"{run_id}.npz"
            checkpoint_metadata = {
                **metadata,
                "run_id": run_id,
                "head_kind": dataset.head_kind,
                "dataset_content_sha256": dataset.manifest["content_sha256"],
                "test_data_used": False,
            }
            probability = predict_head_probability(
                params, dataset.validation_features, normalization
            )
            validation_logits = predict_head_logit(
                params, dataset.validation_features, normalization
            )
            table = validation_threshold_table(
                probability, dataset.validation_evidence, lambda_break=float(lambda_break)
            )
            selected_threshold = choose_validation_threshold(table)
            # FrozenHead consumes logits.  Select a finite logit threshold
            # that exactly reproduces the validation event set, including
            # saturated sigmoid scores and never/always-event candidates.
            probability_threshold = float(selected_threshold["threshold"])
            selected_event = np.asarray(probability) >= probability_threshold
            if np.all(selected_event):
                threshold_logit = float(np.nextafter(np.min(validation_logits), -np.inf))
            elif not np.any(selected_event):
                threshold_logit = float(np.nextafter(np.max(validation_logits), np.inf))
            else:
                highest_wait = float(np.max(validation_logits[~selected_event]))
                lowest_event = float(np.min(validation_logits[selected_event]))
                if not highest_wait < lowest_event:
                    raise AssertionError("probability threshold is not monotone in logits")
                threshold_logit = (highest_wait + lowest_event) / 2.0
            checkpoint_metadata["threshold_probability"] = probability_threshold
            checkpoint_metadata["threshold_logit"] = threshold_logit
            validation_break_term = float(np.mean(
                np.asarray([row.b for row in dataset.validation_evidence], dtype=np.float64)
                * np.logaddexp(0.0, validation_logits)
            ))
            checkpoint_metadata["validation_break_term"] = validation_break_term
            save_decision_head_checkpoint(
                checkpoint_path, params, normalization, checkpoint_metadata,
                threshold_logit=threshold_logit,
            )
            checkpoint_sha256 = file_hash(checkpoint_path)
            write_csv(output_directory / f"{run_id}_thresholds.csv", table)
            all_history.extend({"run_id": run_id, **row} for row in history)
            candidate = {
                **checkpoint_metadata,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "threshold": probability_threshold,
                "threshold_probability": probability_threshold,
                "threshold_logit": threshold_logit,
                "validation_threshold_metrics": selected_threshold,
                "epoch_and_threshold_selection_split": "validation",
            }
            policy_payload = {
                "schema": "single_segment_decision_policy_candidate_v1",
                "head_kind": dataset.head_kind,
                "checkpoint_sha256": checkpoint_sha256,
                "threshold_probability": probability_threshold,
                "threshold_logit": threshold_logit,
                "lambda_break": float(lambda_break),
                "seed": int(seed),
                "best_epoch": metadata["best_epoch"],
                "dataset_content_sha256": dataset.manifest["content_sha256"],
                "downstream_policy_hashes": sorted(
                    dataset.manifest["downstream_policy_lineage_counts"]
                ),
                "selection_split": "validation",
                "test_data_used": False,
            }
            policy_payload["policy_sha256"] = content_hash(policy_payload)
            candidate["policy_manifest"] = policy_payload
            candidate["policy_sha256"] = policy_payload["policy_sha256"]
            candidates.append(candidate)

    # Frozen selection protocol: validation decision-surrogate loss, then the
    # validation break term, lower seed, and earlier epoch.  Full-loop
    # validation remains mandatory before deployment acceptance.
    provisional = max(
        candidates,
        key=lambda row: (
            -row["validation_total_loss"],
            -row["validation_break_term"],
            -row["seed"],
            -row["best_epoch"],
            -row["lambda_break"],
        ),
    )
    result = {
        "schema": "single_segment_head_training_grid_v1",
        "dataset_manifest": dataset.manifest,
        "candidate_count": len(candidates),
        "seeds": list(seeds),
        "lambda_break_family": list(lambda_break_family),
        "epochs": epochs,
        "candidates": candidates,
        "provisional_branch_evidence_choice": provisional,
        "final_selected_policy": None,
        "requires_full_episode_validation": True,
        "epoch_and_threshold_selection_data": "validation only",
        "calibration_or_test_used": False,
    }
    write_csv(output_directory / "training_history.csv", all_history)
    write_json(output_directory / "checkpoint_manifest.json", result)
    return result


def _wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / n
    denominator = 1.0 + z * z / n
    center = (estimate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(estimate * (1 - estimate) / n + z * z / (4 * n * n)) / denominator
    return float(center - half), float(center + half)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _outcome_success(row: Mapping[str, Any], prefix: str) -> bool:
    explicit = row.get(f"{prefix}_success")
    if explicit is not None and explicit != "":
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            raise ValueError(f"invalid success indicator: {explicit!r}")
        return bool(int(explicit)) if isinstance(explicit, (int, np.integer)) else bool(explicit)
    return str(row[f"{prefix}_outcome"]) == "success"


def analyze_full_loop_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    policy_hash: str,
    allow_final_test: bool = False,
) -> dict[str, Any]:
    """Compute source-episode metrics for one frozen full-loop policy.

    Final-test analysis requires an explicit opt-in so development scripts
    cannot consume it accidentally.  Each root source contributes exactly one
    record; multiple decision states or branch continuations are rejected.
    """

    if split == "final_test" and not allow_final_test:
        raise ValueError("final-test analysis is locked until the selected system is frozen")
    if split != "final_test" and split not in ALLOWED_ANALYSIS_SPLITS:
        raise ValueError(f"unsupported full-loop split: {split}")
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("full-loop rows are empty")
    seen_sources: set[str] = set()
    for row in rows:
        if row.get("record_complete") is not True or row.get("execution_error") is not None:
            raise ValueError("incomplete/errored full-loop outcome cannot enter analysis")
        if str(row.get("split")) != split:
            raise ValueError("mixed or mislabeled full-loop split")
        if str(row.get("policy_hash")) != policy_hash:
            raise ValueError("mixed full-loop policy hashes")
        source = _require_text(row, "root_source_id")
        if source in seen_sources:
            raise ValueError("source-level inference requires exactly one row per root source")
        seen_sources.add(source)
        if int(row.get("recovery_segment_count", 0)) > 1:
            raise ValueError("single-segment controller produced multiple recovery segments")
        if str(row.get("safety_outcome")) not in {
            "success", "timeout", "deadlock", "strict_deadlock", "collision"
        }:
            raise ValueError("unknown Safety outcome")
        if str(row.get("learned_outcome")) not in {
            "success", "timeout", "deadlock", "strict_deadlock", "collision"
        }:
            raise ValueError("unknown learned outcome")

    def homogeneous_threshold(key: str) -> float | None:
        present = [row.get(key) not in (None, "") for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"mixed availability for frozen {key}")
        if not any(present):
            return None
        values = {float(row[key]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"mixed frozen {key} values")
        value = values.pop()
        if value not in PREDECLARED_SCORE_THRESHOLDS:
            raise ValueError(f"{key} is outside the predeclared threshold grid")
        return value

    entry_threshold = homogeneous_threshold("entry_threshold_probability")
    exit_threshold = homogeneous_threshold("exit_threshold_probability")

    safety = np.asarray([_outcome_success(row, "safety") for row in rows], dtype=bool)
    learned = np.asarray([_outcome_success(row, "learned") for row in rows], dtype=bool)
    n = len(rows)
    both_success = int(np.sum(safety & learned))
    rescue = int(np.sum(~safety & learned))
    breaks = int(np.sum(safety & ~learned))
    both_fail = int(np.sum(~safety & ~learned))
    q_safe = float(np.mean(safety))
    q_learned = float(np.mean(learned))
    q_safe_ci = _wilson(int(np.sum(safety)), n)
    q_learned_ci = _wilson(int(np.sum(learned)), n)
    delta_ci = paired_success_difference_interval(rescue, breaks, n, confidence=0.95)
    safety_failures = int(np.sum(~safety))
    safety_successes = int(np.sum(safety))
    rescue_ci = _wilson(rescue, safety_failures)
    break_ci = _wilson(breaks, safety_successes)

    timeout_mask = np.asarray([str(row["safety_outcome"]) == "timeout" for row in rows])
    deadlock_mask = np.asarray(
        [str(row["safety_outcome"]) in {"deadlock", "strict_deadlock"} for row in rows]
    )
    jdef = np.asarray([float(row.get("jdef", 0.0)) for row in rows])
    entered = np.asarray(
        [row.get("entry_step") not in (None, "", -1, "-1") for row in rows], dtype=bool
    )
    exited = np.asarray(
        [row.get("exit_step") not in (None, "", -1, "-1") for row in rows], dtype=bool
    )
    entry_step = np.asarray(
        [float(row["entry_step"]) if entered[index] else np.nan for index, row in enumerate(rows)]
    )
    exit_step = np.asarray(
        [float(row["exit_step"]) if exited[index] else np.nan for index, row in enumerate(rows)]
    )
    terminal_step = np.asarray([float(row["terminal_step"]) for row in rows])
    recovery_steps = np.asarray(
        [
            int(row.get("recovery_transition_count", 0))
            if entered[index]
            else 0
            for index, row in enumerate(rows)
        ]
    )
    if np.any(entered & (recovery_steps < 1)):
        raise ValueError("entry must execute at least one recovery transition")
    if np.any(exited & ~entered):
        raise ValueError("exit without an entry")
    if np.any(exited & (exit_step <= entry_step)):
        raise ValueError("zero-duration recovery segment")

    finite_success = entered & exited & learned
    takeover_success = entered & ~exited & learned
    returned_then_failed = entered & exited & ~learned
    completed_after_return = int(np.sum(finite_success))
    collisions = sum(int(row.get("agent_collision", 0)) for row in rows)
    wall_collisions = sum(int(row.get("wall_collision", 0)) for row in rows)
    invalid_actions = sum(int(row.get("invalid_action", 0)) for row in rows)
    nan_inf = sum(int(row.get("nan_inf", 0)) for row in rows)
    solver_failures = sum(int(row.get("projection_solver_failure", 0)) for row in rows)

    entry_values = entry_step[entered]
    exit_values = exit_step[exited]
    segment_values = recovery_steps[entered]
    late_entry = entered & (terminal_step - entry_step <= 1.0)
    defer_until_failure = (~learned) & ((~entered) | late_entry)
    support_values = [
        value
        for row in rows
        for value in (row.get("entry_score_supported"), row.get("exit_score_supported"))
        if value not in (None, "")
    ]
    unsupported_scores = sum(not _coerce_bool(value) for value in support_values)
    result = {
        "schema": "single_segment_full_loop_metrics_v1",
        "split": split,
        "policy_hash": policy_hash,
        "entry_threshold_probability": entry_threshold,
        "exit_threshold_probability": exit_threshold,
        "source_episode_count": n,
        "source_level_inference": True,
        "q_safe": q_safe,
        "q_safe_ci95": list(q_safe_ci),
        "q_learned": q_learned,
        "q_learned_ci95": list(q_learned_ci),
        "delta_q": q_learned - q_safe,
        "delta_q_paired_ci95": list(delta_ci),
        "both_success": both_success,
        "rescue": rescue,
        "break": breaks,
        "both_fail": both_fail,
        "rescue_rate": rescue / safety_failures if safety_failures else None,
        "rescue_rate_ci95": list(rescue_ci),
        "break_rate": breaks / safety_successes if safety_successes else None,
        "break_rate_ci95": list(break_ci),
        "timeout_failures": int(np.sum(timeout_mask)),
        "timeout_rescues": int(np.sum(timeout_mask & learned)),
        "strict_deadlock_failures": int(np.sum(deadlock_mask)),
        "strict_deadlock_rescues": int(np.sum(deadlock_mask & learned)),
        "jdef": _summary(jdef),
        "jdef_rescues": _summary(jdef[(~safety) & learned]),
        "jdef_breaks": _summary(jdef[safety & (~learned)]),
        "segment_statistics": {
            "entered_count": int(np.sum(entered)),
            "entered_fraction": float(np.mean(entered)),
            "exited_count": int(np.sum(exited)),
            "exited_fraction_among_entered": float(np.sum(exited) / max(np.sum(entered), 1)),
            "successful_finite_segment_then_safety_count": completed_after_return,
            "successful_takeover_until_completion_count": int(np.sum(takeover_success)),
            "returned_to_safety_then_failed_count": int(np.sum(returned_then_failed)),
            "entry_step": _summary(entry_values),
            "exit_step": _summary(exit_values),
            "recovery_transition_count": _summary(segment_values),
        },
        "degeneracy": {
            "never_enter": bool(np.sum(entered) == 0),
            "enter_almost_everywhere": bool(np.mean(entered) >= 0.95),
            "exit_after_one_step_almost_everywhere": bool(
                np.sum(entered) > 0 and np.mean(recovery_steps[entered] == 1) >= 0.95
            ),
            "never_exit_before_termination": bool(np.sum(entered) > 0 and np.sum(exited) == 0),
            "repeatedly_defer_entry_until_failure_count": int(np.sum(defer_until_failure)),
            "exit_then_lose_task_progress_count": int(np.sum(returned_then_failed)),
            "confidence_support_observations": len(support_values),
            "confidence_unsupported_count": unsupported_scores,
            "multiple_segment_count": 0,
        },
        "hard_safety": {
            "agent_collisions": collisions,
            "wall_collisions": wall_collisions,
            "invalid_actions": invalid_actions,
            "nan_inf": nan_inf,
            "projection_solver_failures": solver_failures,
            "intact": collisions + wall_collisions + invalid_actions + nan_inf + solver_failures == 0,
        },
        "final_test_explicitly_unlocked": bool(split == "final_test" and allow_final_test),
    }
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


def select_policy_by_full_loop_validation(
    candidate_metrics: Sequence[Mapping[str, Any]],
    *,
    exploratory_break_budget: float = 0.05,
) -> dict[str, Any]:
    """Select a frozen candidate from complete validation episodes only.

    This is the mandatory global-policy selection step.  Head loss, local
    action accuracy, and branch-surrogate scores are intentionally absent from
    its ordering.  The frozen ordering is task success, conditional break,
    deformation, higher entry threshold, then lower exit threshold.  The
    exploratory break budget is reported but is not a validation-time hard
    filter; its uncertainty is audited once on the separate calibration set.
    """

    metrics = [dict(row) for row in candidate_metrics]
    if not metrics:
        raise ValueError("no full-loop validation candidates")
    policy_hashes: set[str] = set()
    for row in metrics:
        if row.get("split") != "validation":
            raise ValueError("final policy selection must use full-episode validation only")
        policy_hash = _require_text(row, "policy_hash")
        if policy_hash in policy_hashes:
            raise ValueError("duplicate full-loop validation policy")
        policy_hashes.add(policy_hash)
        if int(row.get("source_episode_count", 0)) <= 0:
            raise ValueError("candidate lacks source-level full-loop support")
    safe = [row for row in metrics if row.get("hard_safety", {}).get("intact") is True]
    if not safe:
        raise ValueError("no hard-safety-intact validation candidate")
    within_break = [
        row
        for row in safe
        if row.get("break_rate") is not None
        and float(row["break_rate"]) <= exploratory_break_budget
    ]
    selected = max(
        safe,
        key=lambda row: (
            float(row["q_learned"]),
            -float(row["break_rate"] if row.get("break_rate") is not None else 1.0),
            -float(row.get("jdef", {}).get("mean") or 0.0),
            float(row.get("entry_threshold_probability", 0.5)),
            -float(row.get("exit_threshold_probability", 0.5)),
            str(row["policy_hash"]),
        ),
    )
    return {
        "schema": "single_segment_full_loop_validation_selection_v1",
        "selected_policy_hash": selected["policy_hash"],
        "selected_metrics": selected,
        "candidate_count": len(metrics),
        "hard_safety_intact_candidate_count": len(safe),
        "point_break_budget_candidate_count": len(within_break),
        "exploratory_break_budget": exploratory_break_budget,
        "selection_split": "validation",
        "selection_basis": "complete source-level full-episode outcomes",
        "selection_order": [
            "maximum_q", "minimum_conditional_break", "minimum_mean_jdef",
            "higher_entry_threshold", "lower_exit_threshold",
        ],
        "break_budget_used_as_selection_filter": False,
        "local_head_loss_used_as_global_substitute": False,
        "calibration_required_after_selection": True,
        "test_data_used": False,
    }


def calibration_break_budget_audit(
    metrics: Mapping[str, Any],
    *,
    break_budget: float = 0.05,
) -> dict[str, Any]:
    """Audit, but never certify, the exploratory conditional break budget."""

    if str(metrics.get("split")) != "calibration":
        raise ValueError("break-budget audit requires the separate calibration cohort")
    estimate = metrics.get("break_rate")
    upper = metrics.get("break_rate_ci95", [None, None])[1]
    supported = bool(upper is not None and np.isfinite(upper) and upper <= break_budget)
    return {
        "break_budget": float(break_budget),
        "conditional_break_estimate": estimate,
        "conditional_break_ci95": metrics.get("break_rate_ci95"),
        "point_estimate_within_budget": bool(estimate is not None and estimate <= break_budget),
        "ci_upper_within_budget": supported,
        "pilot_supported_not_certified": supported,
        "certification_claimed": False,
        "insufficient_support": not supported,
    }
