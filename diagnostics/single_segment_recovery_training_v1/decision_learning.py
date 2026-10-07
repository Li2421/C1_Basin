"""Decision-label, uncertainty, loss, and small-head training utilities.

The single-segment recovery experiment does not use hard hindsight labels.
For a decision state and matched future stream, action ``0`` and action ``1``
are complete-continuation counterfactuals.  The sufficient success evidence is

    r = P(S0=0, S1=1)   (the event rescues),
    b = P(S0=1, S1=0)   (the event breaks).

The event is ENTER for an entry head and EXIT for an exit head.  The primary
loss is therefore

    r * softplus(-logit) + (1 + lambda_break) * b * softplus(logit).

Deformation is only allowed to break a tie after a conservative paired
confidence interval supports the predeclared practical-equivalence region.
This module deliberately represents inadequate evidence as unresolved; it
never turns absence of observed discordance into a hard decision label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LAMBDA_BREAK_FAMILY: tuple[float, ...] = (0.0, 1.0, 3.0)
DEFAULT_EQUIVALENCE_MARGIN = 0.02
DEFAULT_EQUIVALENCE_CI_LEVEL = 0.90
DEFAULT_MIN_MATCHED_PAIRS = 32
DEFAULT_MAX_DEFORMATION_WEIGHT = 0.10
DEFAULT_DEFORMATION_TO_SUCCESS_CAP = 0.10


@dataclass(frozen=True)
class MatchedBranchPair:
    """One paired future realization from a fixed decision input."""

    future_stream_id: str
    success0: int
    success1: int
    jdef0: float
    jdef1: float

    def __post_init__(self) -> None:
        if self.success0 not in (0, 1) or self.success1 not in (0, 1):
            raise ValueError("success indicators must be exactly 0 or 1")
        if not np.isfinite(self.jdef0) or not np.isfinite(self.jdef1):
            raise ValueError("remaining J_def must be finite")
        if self.jdef0 < 0.0 or self.jdef1 < 0.0:
            raise ValueError("remaining J_def must be nonnegative")


@dataclass(frozen=True)
class DeformationNormalizer:
    """TRAIN-only robust scale for paired remaining-deformation differences."""

    scale: float
    fit_split: str = "train"
    method: str = "median_absolute_paired_difference"
    pair_count: int = 0

    @classmethod
    def fit(
        cls,
        pairs: Iterable[MatchedBranchPair],
        *,
        split: str = "train",
        floor: float = 1e-8,
    ) -> "DeformationNormalizer":
        if split != "train":
            raise ValueError("deformation normalization must be fit on training sources only")
        pairs = tuple(pairs)
        if not pairs:
            raise ValueError("cannot fit deformation scale without paired branches")
        differences = np.asarray([abs(p.jdef1 - p.jdef0) for p in pairs], dtype=np.float64)
        positive = differences[differences > floor]
        if positive.size:
            scale = float(np.median(positive))
        else:
            pooled = np.asarray(
                [value for p in pairs for value in (p.jdef0, p.jdef1) if value > floor],
                dtype=np.float64,
            )
            scale = float(np.median(pooled)) if pooled.size else 1.0
        return cls(scale=max(scale, floor), pair_count=len(pairs))


@dataclass(frozen=True)
class DecisionEvidence:
    """Soft success evidence and uncertainty for one decision input."""

    decision_id: str
    n_pairs: int
    both_fail: int
    rescue: int
    break_count: int
    both_success: int
    r: float
    b: float
    paired_delta: float
    paired_ci_level: float
    paired_ci_lower: float
    paired_ci_upper: float
    equivalence_margin: float
    equivalence_supported: bool
    status: str
    unresolved: bool
    deformation_target: float
    deformation_weight: float
    mean_jdef0: float
    mean_jdef1: float
    downstream_policy_hash: str | None = None

    @property
    def success_weight(self) -> float:
        """Coefficient mass in the primary loss before break asymmetry."""

        return self.r + self.b

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson_interval(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval with deterministic NormalDist quantiles."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial count")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / n
    denominator = 1.0 + z * z / n
    center = (estimate + z * z / (2.0 * n)) / denominator
    half = (
        z
        * np.sqrt(estimate * (1.0 - estimate) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def paired_success_difference_interval(
    rescue: int,
    break_count: int,
    n_pairs: int,
    *,
    confidence: float = DEFAULT_EQUIVALENCE_CI_LEVEL,
) -> tuple[float, float]:
    """Conservative CI for paired success difference ``P01 - P10``.

    Rescue and break are two cells of the paired multinomial.  We construct
    simultaneous Wilson intervals for their marginal cell probabilities via
    Bonferroni and subtract the endpoints.  It is conservative, including in
    the no-discordance case where an ordinary bootstrap would incorrectly
    collapse to a zero-width interval.
    """

    if rescue < 0 or break_count < 0 or rescue + break_count > n_pairs:
        raise ValueError("invalid paired-discordance counts")
    alpha = 1.0 - confidence
    per_cell_confidence = 1.0 - alpha / 2.0
    rescue_lower, rescue_upper = _wilson_interval(rescue, n_pairs, per_cell_confidence)
    break_lower, break_upper = _wilson_interval(break_count, n_pairs, per_cell_confidence)
    return (
        max(-1.0, rescue_lower - break_upper),
        min(1.0, rescue_upper - break_lower),
    )


def pair_branch_records(
    records: Iterable[Mapping[str, Any]],
    *,
    decision_key: str = "decision_id",
    action_key: str = "action",
    stream_key: str = "future_stream_id",
    success_key: str = "success",
    jdef_key: str = "remaining_jdef",
    invariant_keys: Sequence[str] = (
        "root_source_id",
        "split",
        "state_hash",
        "absolute_step",
        "mode",
        "current_flow_realization_id",
        "future_rng_state_hash",
        "eta_latched_hash",
        "downstream_policy_hash",
    ),
) -> dict[str, list[MatchedBranchPair]]:
    """Validate and pair action-0/action-1 branch records by future stream.

    Pairing is strict: duplicates, missing counterfactuals, or mismatched
    state/current-Flow/policy identifiers raise instead of being dropped.
    Keys absent from *both* records are ignored, permitting compact unit-test
    and bootstrap files; a key present on just one side is a mismatch.
    """

    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for row in records:
        decision_id = str(row[decision_key])
        stream_id = str(row[stream_key])
        action = int(row[action_key])
        if action not in (0, 1):
            raise ValueError(f"{decision_id}/{stream_id}: action must be 0 or 1")
        cell = grouped.setdefault((decision_id, stream_id), {})
        if action in cell:
            raise ValueError(f"{decision_id}/{stream_id}: duplicate action {action}")
        cell[action] = row

    result: dict[str, list[MatchedBranchPair]] = {}
    for (decision_id, stream_id), actions in grouped.items():
        if set(actions) != {0, 1}:
            raise ValueError(f"{decision_id}/{stream_id}: missing matched action branch")
        row0, row1 = actions[0], actions[1]
        for key in invariant_keys:
            present0, present1 = key in row0, key in row1
            if present0 != present1 or (present0 and row0[key] != row1[key]):
                raise ValueError(f"{decision_id}/{stream_id}: paired invariant differs: {key}")
        pair = MatchedBranchPair(
            future_stream_id=stream_id,
            success0=int(row0[success_key]),
            success1=int(row1[success_key]),
            jdef0=float(row0[jdef_key]),
            jdef1=float(row1[jdef_key]),
        )
        result.setdefault(decision_id, []).append(pair)

    for decision_id in result:
        result[decision_id].sort(key=lambda pair: pair.future_stream_id)
    return result


def summarize_decision_evidence(
    decision_id: str,
    pairs: Sequence[MatchedBranchPair],
    *,
    deformation_normalizer: DeformationNormalizer | None = None,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    equivalence_ci_level: float = DEFAULT_EQUIVALENCE_CI_LEVEL,
    minimum_pairs: int = DEFAULT_MIN_MATCHED_PAIRS,
    max_deformation_weight: float = DEFAULT_MAX_DEFORMATION_WEIGHT,
    deformation_to_success_cap: float = DEFAULT_DEFORMATION_TO_SUCCESS_CAP,
    downstream_policy_hash: str | None = None,
) -> DecisionEvidence:
    """Compute r/b targets and guarded deformation preference for one input."""

    if equivalence_margin <= 0.0:
        raise ValueError("equivalence margin must be positive and predeclared")
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    if not 0.0 <= max_deformation_weight <= 1.0:
        raise ValueError("max_deformation_weight must be in [0, 1]")
    if not 0.0 <= deformation_to_success_cap <= 1.0:
        raise ValueError("deformation_to_success_cap must be in [0, 1]")
    if not pairs:
        raise ValueError("at least one matched pair is required")
    stream_ids = [pair.future_stream_id for pair in pairs]
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("future stream IDs must be unique within a decision input")

    s0 = np.asarray([pair.success0 for pair in pairs], dtype=np.int8)
    s1 = np.asarray([pair.success1 for pair in pairs], dtype=np.int8)
    j0 = np.asarray([pair.jdef0 for pair in pairs], dtype=np.float64)
    j1 = np.asarray([pair.jdef1 for pair in pairs], dtype=np.float64)
    n_pairs = len(pairs)
    both_fail = int(np.sum((s0 == 0) & (s1 == 0)))
    rescue = int(np.sum((s0 == 0) & (s1 == 1)))
    break_count = int(np.sum((s0 == 1) & (s1 == 0)))
    both_success = int(np.sum((s0 == 1) & (s1 == 1)))
    r = rescue / n_pairs
    b = break_count / n_pairs
    lower, upper = paired_success_difference_interval(
        rescue, break_count, n_pairs, confidence=equivalence_ci_level
    )
    enough_pairs = n_pairs >= minimum_pairs
    equivalence_supported = bool(
        enough_pairs and lower >= -equivalence_margin and upper <= equivalence_margin
    )

    if not enough_pairs:
        status = "INSUFFICIENT_MATCHED_FUTURES"
        unresolved = True
    elif lower > 0.0:
        status = "ACTION1_SUCCESS_FAVORED"
        unresolved = False
    elif upper < 0.0:
        status = "ACTION0_SUCCESS_FAVORED"
        unresolved = False
    elif equivalence_supported:
        status = "SUCCESS_EQUIVALENT_WITHIN_PREDECLARED_MARGIN"
        unresolved = False
    elif rescue == 0 and break_count == 0:
        status = "UNRESOLVED_NO_DISCORDANCE"
        unresolved = True
    else:
        status = "UNRESOLVED_PAIRED_SUCCESS_DIFFERENCE"
        unresolved = True

    deformation_target = 0.5
    deformation_weight = 0.0
    if equivalence_supported and deformation_normalizer is not None:
        # Positive advantage means action 1 has smaller remaining deformation.
        normalized_advantage = float(
            np.tanh(abs(float(np.mean(j0) - np.mean(j1))) / deformation_normalizer.scale)
        )
        if np.mean(j1) < np.mean(j0):
            deformation_target = 1.0
        elif np.mean(j1) > np.mean(j0):
            deformation_target = 0.0
        else:
            deformation_target = 0.5
            normalized_advantage = 0.0
        proposed = max_deformation_weight * normalized_advantage
        success_weight = r + b
        if success_weight > 0.0:
            proposed = min(proposed, deformation_to_success_cap * success_weight)
        deformation_weight = float(proposed)

    return DecisionEvidence(
        decision_id=str(decision_id),
        n_pairs=n_pairs,
        both_fail=both_fail,
        rescue=rescue,
        break_count=break_count,
        both_success=both_success,
        r=float(r),
        b=float(b),
        paired_delta=float(r - b),
        paired_ci_level=float(equivalence_ci_level),
        paired_ci_lower=float(lower),
        paired_ci_upper=float(upper),
        equivalence_margin=float(equivalence_margin),
        equivalence_supported=equivalence_supported,
        status=status,
        unresolved=unresolved,
        deformation_target=deformation_target,
        deformation_weight=deformation_weight,
        mean_jdef0=float(np.mean(j0)),
        mean_jdef1=float(np.mean(j1)),
        downstream_policy_hash=downstream_policy_hash,
    )


def softplus(value: np.ndarray | float) -> np.ndarray:
    """Stable NumPy softplus."""

    return np.logaddexp(0.0, np.asarray(value, dtype=np.float64))


def decision_loss_components(
    logits: np.ndarray | Sequence[float],
    evidence: Sequence[DecisionEvidence],
    *,
    lambda_break: float,
) -> dict[str, np.ndarray | float]:
    """Return primary and bounded deformation loss components.

    The returned ``total`` is the mean over decision inputs, so inputs with
    many future streams do not masquerade as independent decision states.
    """

    if lambda_break not in LAMBDA_BREAK_FAMILY:
        raise ValueError(f"lambda_break must be selected from {LAMBDA_BREAK_FAMILY}")
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.size != len(evidence):
        raise ValueError("one logit is required per decision input")
    r = np.asarray([row.r for row in evidence], dtype=np.float64)
    b = np.asarray([row.b for row in evidence], dtype=np.float64)
    deformation_target = np.asarray([row.deformation_target for row in evidence], dtype=np.float64)
    deformation_weight = np.asarray([row.deformation_weight for row in evidence], dtype=np.float64)
    success = r * softplus(-logits) + (1.0 + lambda_break) * b * softplus(logits)
    deformation = deformation_weight * (
        deformation_target * softplus(-logits)
        + (1.0 - deformation_target) * softplus(logits)
    )
    combined = success + deformation
    return {
        "success_per_input": success,
        "deformation_per_input": deformation,
        "combined_per_input": combined,
        "success_mean": float(np.mean(success)),
        "deformation_mean": float(np.mean(deformation)),
        "total": float(np.mean(combined)),
    }


def evidence_arrays(evidence: Sequence[DecisionEvidence]) -> dict[str, np.ndarray]:
    """Convert evidence objects to arrays consumed by a JAX trainer."""

    return {
        "r": np.asarray([row.r for row in evidence], dtype=np.float32),
        "b": np.asarray([row.b for row in evidence], dtype=np.float32),
        "deformation_target": np.asarray(
            [row.deformation_target for row in evidence], dtype=np.float32
        ),
        "deformation_weight": np.asarray(
            [row.deformation_weight for row in evidence], dtype=np.float32
        ),
    }


def fit_feature_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature normalization; callers must pass training features only."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] not in (214, 217):
        raise ValueError("head inputs must have dimension 214 (G/entry) or 217 (ETA exit)")
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("training features must be nonempty and finite")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if features.shape[-1] != mean.size or mean.shape != scale.shape:
        raise ValueError("feature/normalization dimension mismatch")
    return (features - mean) / scale


def _lazy_jax():
    """Import training dependencies lazily so labeling stays NumPy-only."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import optax  # pylint: disable=import-outside-toplevel

    return jax, jnp, optax


def init_head_parameters(seed: int, input_dim: int) -> list[dict[str, Any]]:
    """Initialize the fixed ``input -> 64 -> 64 -> 1`` SiLU head."""

    if input_dim not in (214, 217):
        raise ValueError("input_dim must be 214 or 217")
    jax, jnp, _ = _lazy_jax()
    dimensions = (input_dim, 64, 64, 1)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(dimensions) - 1)
    params: list[dict[str, Any]] = []
    for key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        params.append(
            {
                "w": jax.random.uniform(
                    key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=jnp.float32
                ),
                "b": jnp.zeros((fan_out,), dtype=jnp.float32),
            }
        )
    return params


def head_logits(params: Sequence[Mapping[str, Any]], features: Any) -> Any:
    """Forward pass for the fixed small decision head."""

    _, jnp, _ = _lazy_jax()
    value = features
    for layer in params[:-1]:
        value = jax_silu(value @ layer["w"] + layer["b"], jnp=jnp)
    return (value @ params[-1]["w"] + params[-1]["b"]).reshape(-1)


def jax_silu(value: Any, *, jnp: Any | None = None) -> Any:
    if jnp is None:
        _, jnp, _ = _lazy_jax()
    return value / (1.0 + jnp.exp(-value))


def train_decision_head(
    train_features: np.ndarray,
    train_evidence: Sequence[DecisionEvidence],
    validation_features: np.ndarray,
    validation_evidence: Sequence[DecisionEvidence],
    *,
    seed: int,
    lambda_break: float,
    epochs: int = 1200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], list[dict[str, float]], dict[str, Any]]:
    """Train a small head and select its checkpoint by validation loss only.

    This is intentionally a compact full-batch utility.  Source-level split
    construction and threshold selection remain responsibilities of the
    experiment driver.  No test outcomes enter this function.
    """

    if lambda_break not in LAMBDA_BREAK_FAMILY:
        raise ValueError(f"lambda_break must be selected from {LAMBDA_BREAK_FAMILY}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    x_train = np.asarray(train_features, dtype=np.float32)
    x_validation = np.asarray(validation_features, dtype=np.float32)
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train/validation feature matrices must have the same input dimension")
    if x_train.shape[0] != len(train_evidence) or x_validation.shape[0] != len(validation_evidence):
        raise ValueError("one evidence object is required per feature row")
    mean, scale = fit_feature_normalization(x_train)
    x_train = normalize_features(x_train, mean, scale)
    x_validation = normalize_features(x_validation, mean, scale)
    train_targets = evidence_arrays(train_evidence)
    validation_targets = evidence_arrays(validation_evidence)

    jax, jnp, optax = _lazy_jax()
    params = init_head_parameters(seed, x_train.shape[1])
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer_state = optimizer.init(params)

    x_train_jax = jnp.asarray(x_train)
    x_validation_jax = jnp.asarray(x_validation)
    train_targets_jax = {key: jnp.asarray(value) for key, value in train_targets.items()}
    validation_targets_jax = {key: jnp.asarray(value) for key, value in validation_targets.items()}

    def loss_terms(candidate: Any, features: Any, targets: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        values = head_logits(candidate, features)
        success = (
            targets["r"] * jnp.logaddexp(0.0, -values)
            + (1.0 + lambda_break) * targets["b"] * jnp.logaddexp(0.0, values)
        )
        deformation = targets["deformation_weight"] * (
            targets["deformation_target"] * jnp.logaddexp(0.0, -values)
            + (1.0 - targets["deformation_target"]) * jnp.logaddexp(0.0, values)
        )
        return jnp.mean(success + deformation), jnp.mean(success), jnp.mean(deformation)

    @jax.jit
    def step(candidate: Any, state: Any) -> tuple[Any, Any, Any, Any, Any]:
        def objective(value: Any) -> tuple[Any, tuple[Any, Any]]:
            total, success, deformation = loss_terms(value, x_train_jax, train_targets_jax)
            return total, (success, deformation)

        (loss, (success_loss, deformation_loss)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(candidate)
        updates, state = optimizer.update(gradients, state, candidate)
        return optax.apply_updates(candidate, updates), state, loss, success_loss, deformation_loss

    best_params = None
    best_epoch = 0
    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        params, optimizer_state, train_loss, train_success, train_deformation = step(
            params, optimizer_state
        )
        validation_loss, validation_success, validation_deformation = loss_terms(
            params, x_validation_jax, validation_targets_jax
        )
        row = {
            "epoch": float(epoch),
            "train_total_loss": float(train_loss),
            "train_success_loss": float(train_success),
            "train_deformation_loss": float(train_deformation),
            "validation_total_loss": float(validation_loss),
            "validation_success_loss": float(validation_success),
            "validation_deformation_loss": float(validation_deformation),
        }
        history.append(row)
        if row["validation_total_loss"] < best_validation_loss:
            best_validation_loss = row["validation_total_loss"]
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)

    assert best_params is not None
    normalization = {"mean": mean, "scale": scale}
    metadata = {
        "architecture": [int(x_train.shape[1]), 64, 64, 1],
        "activation": "SiLU",
        "seed": int(seed),
        "lambda_break": float(lambda_break),
        "best_epoch": int(best_epoch),
        "validation_total_loss": float(best_validation_loss),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "normalization_fit_split": "train",
        "parameter_count": int(sum(value.size for layer in best_params for value in layer.values())),
    }
    return best_params, normalization, history, metadata


def predict_head_probability(
    params: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    normalization: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Predict event probabilities; scores are not task-success probabilities."""

    jax, jnp, _ = _lazy_jax()
    normalized = normalize_features(features, normalization["mean"], normalization["scale"])
    return np.asarray(jax.nn.sigmoid(head_logits(params, jnp.asarray(normalized))))


def predict_head_logit(
    params: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    normalization: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return raw decision logits under the frozen training normalization."""

    _, jnp, _ = _lazy_jax()
    normalized = normalize_features(features, normalization["mean"], normalization["scale"])
    return np.asarray(head_logits(params, jnp.asarray(normalized)), dtype=np.float64)


def save_decision_head_checkpoint(
    path: str | Path,
    params: Sequence[Mapping[str, Any]],
    normalization: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    threshold_logit: float | None = None,
) -> None:
    """Save an auditable NumPy checkpoint using the deployment-head contract."""

    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "normalization_mean": np.asarray(normalization["mean"], dtype=np.float32),
        "normalization_scale": np.asarray(normalization["scale"], dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    if threshold_logit is not None:
        if not np.isfinite(threshold_logit):
            raise ValueError("threshold_logit must be finite")
        payload["threshold_logit"] = np.asarray(threshold_logit, dtype=np.float64)
    for index, layer in enumerate(params):
        payload[f"layer_{index}_weight"] = np.asarray(layer["w"], dtype=np.float32)
        payload[f"layer_{index}_bias"] = np.asarray(layer["b"], dtype=np.float32)
    np.savez(path, **payload)
