"""Trajectory-first pilot dataset support for the centralized 4-agent expert.

This module is deliberately scenario-local.  It records the complete joint
trajectory, independently replays every rollout through the plant, keeps all
initial-condition relatives in one train/validation split, and derives
coordination modes from actual bottleneck crossing events rather than planner
labels.

The plant is first order, but its last applied velocity is part of the Markov
observation.  Pilot nonzero initial values are installed through the existing
exact augmented-state restoration API; they have no inertial effect after the
first expert action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .environment import Config, DoubleBottleneckEnv
from .expert import (
    CoordinationHypothesis,
    ExpertPlan,
    all_coordination_hypotheses,
)
from .scenario import AGENT_NAMES, INITIAL_REGIMES, initial_positions


DATASET_SCHEMA = "double_bottleneck_expert_dataset_v1"
ROLLOUT_SCHEMA = "double_bottleneck_expert_trajectory_v1"
OBSERVATION_SCHEMA = (
    "per-agent [p_i(2), last_applied_velocity_i(2), goal_i-p_i(2), "
    "(p_j-p_i(2), v_j-v_i(2)) for j != i in ascending global index]"
)
ACTION_SCHEMA = "joint executed velocity [agent=A1,A2,B1,B2, xy], shape [T,4,2]"
INITIAL_VELOCITY_PROTOCOL = "augmented_state_last_applied_velocity_v1"
TERMINAL_REASONS = frozenset(("success", "collision", "deadlock", "timeout", "planner_exhausted"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def environment_descriptor(env: DoubleBottleneckEnv) -> dict[str, Any]:
    """Return the complete fixed-geometry/dynamics contract used by a record."""

    return {
        "config": env.config.to_dict(),
        "config_fingerprint": env.config.fingerprint,
        "walls": env.walls.tolist(),
        "wall_names": list(env.wall_names),
        "goals": env.goals.tolist(),
        "agent_names": list(AGENT_NAMES),
        "pair_indices": [list(pair) for pair in env.pair_indices],
        "observation_shape": [4, 18],
        "action_shape": [4, 2],
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "initial_velocity_protocol": INITIAL_VELOCITY_PROTOCOL,
    }


def environment_fingerprint(env_or_descriptor: DoubleBottleneckEnv | Mapping[str, Any]) -> str:
    """Hash config, derived geometry, ordering, and tensor semantics together."""

    descriptor = (
        environment_descriptor(env_or_descriptor)
        if isinstance(env_or_descriptor, DoubleBottleneckEnv)
        else _jsonable(env_or_descriptor)
    )
    return hashlib.sha256(_canonical_json(descriptor).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InitialConditionSpec:
    """One pilot initial condition and its leakage-control family."""

    condition_id: str
    family_id: str
    regime: str
    positions: np.ndarray
    initial_velocities: np.ndarray
    perturbation: Mapping[str, Any]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.shape != (4, 2) or not np.isfinite(positions).all():
            raise ValueError("initial positions must be finite with shape [4,2]")
        velocities = np.asarray(self.initial_velocities, dtype=np.float64)
        if velocities.shape != (4, 2) or not np.isfinite(velocities).all():
            raise ValueError("initial velocities must be finite with shape [4,2]")
        if self.regime not in INITIAL_REGIMES:
            raise ValueError(f"unknown initial regime {self.regime!r}")
        if not self.condition_id or not self.family_id:
            raise ValueError("condition_id and family_id must be nonempty")
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "initial_velocities", velocities.copy())
        object.__setattr__(self, "perturbation", dict(self.perturbation))


@dataclass(frozen=True)
class TrajectoryRecord:
    """One full, variable-length centralized-expert rollout."""

    rollout_id: str
    family_id: str
    condition_id: str
    regime: str
    split: str
    initial_positions: np.ndarray
    initial_velocities: np.ndarray
    positions: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    terminal_reason: str
    success: bool
    collision: bool
    wall_collision: bool
    agent_collision: bool
    deadlock: bool
    timeout: bool
    episode_steps: int
    min_swept_pair_surface_distance: float
    min_swept_wall_clearance: float
    path_length: float
    solve_time_seconds: float
    perturbation: Mapping[str, Any]
    hypothesis: Mapping[str, Any]
    coordination_mode: Mapping[str, Any]
    environment: Mapping[str, Any]
    environment_fingerprint: str
    config_fingerprint: str
    trajectory_digest: str

    @property
    def training_eligible(self) -> bool:
        return bool(
            self.success
            and self.terminal_reason == "success"
            and not self.collision
            and not self.deadlock
            and not self.timeout
        )

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def trajectory_digest(
    positions: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray | None = None,
) -> str:
    """Content hash used to reject exact duplicated full-state trajectories.

    ``observations`` is included for dataset records because last-applied
    velocity is state: two first-order rollouts can share position/actions but
    have different initial Markov states.  The optional form is retained for
    lightweight geometry-only analysis helpers.
    """

    digest = hashlib.sha256()
    arrays = (positions, actions) if observations is None else (positions, observations, actions)
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _first_crossing_time(values: np.ndarray, plane: float, direction: int) -> float | None:
    signed = direction * (np.asarray(values, dtype=np.float64) - float(plane))
    for index in range(len(signed) - 1):
        if signed[index] <= 1e-12 and signed[index + 1] >= -1e-12:
            delta = signed[index + 1] - signed[index]
            alpha = 0.0 if abs(delta) <= 1e-15 else float(-signed[index] / delta)
            if -1e-9 <= alpha <= 1.0 + 1e-9:
                return index + min(max(alpha, 0.0), 1.0)
    return None


def infer_coordination_mode(
    positions: np.ndarray,
    goals: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Infer passage orders from the executed geometry, independent of labels."""

    positions = np.asarray(positions, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (4, 2):
        raise ValueError("positions must have shape [T+1,4,2]")
    if goals.shape != (4, 2):
        raise ValueError("goals must have shape [4,2]")
    directions = np.sign(goals[:, 0] - positions[0, :, 0]).astype(np.int64)
    # Degenerate already-at-goal unit fixtures still have an unambiguous side
    # destination in this fixed left/right task.
    directions = np.where(directions == 0, np.sign(goals[:, 0]).astype(np.int64), directions)
    if np.any(directions == 0):
        raise ValueError("cannot infer travel direction from initial state and goal")

    passage_center = config.chamber_half_length + 0.5 * config.bottleneck_length
    planes = {"left": -passage_center, "center": 0.0, "right": passage_center}
    times: dict[str, dict[str, float | None]] = {}
    orders: dict[str, list[str]] = {}
    complete = True
    for name, plane in planes.items():
        by_agent = {
            AGENT_NAMES[agent]: _first_crossing_time(
                positions[:, agent, 0], plane, int(directions[agent])
            )
            for agent in range(4)
        }
        times[name] = by_agent
        complete = complete and all(value is not None for value in by_agent.values())
        orders[name] = [
            agent
            for agent, value in sorted(
                by_agent.items(),
                key=lambda item: (math.inf if item[1] is None else item[1], item[0]),
            )
            if value is not None
        ]

    center = times["center"]
    center_order = orders["center"]
    first_direction = None
    if center_order:
        first_index = AGENT_NAMES.index(center_order[0])
        first_direction = "left_to_right" if directions[first_index] > 0 else "right_to_left"
    ltr_order = [name for name in center_order if directions[AGENT_NAMES.index(name)] > 0]
    rtl_order = [name for name in center_order if directions[AGENT_NAMES.index(name)] < 0]
    signature = (
        f"left={'>'.join(orders['left'])};center={'>'.join(center_order)};"
        f"right={'>'.join(orders['right'])}"
    )
    return {
        "protocol": "directed_plane_crossing_v1",
        "planes_x": planes,
        "crossing_times_steps": times,
        "left_passage_order": orders["left"],
        "center_crossing_order": center_order,
        "right_passage_order": orders["right"],
        "left_to_right_order": ltr_order,
        "right_to_left_order": rtl_order,
        "first_direction": first_direction,
        "complete": bool(complete),
        "signature": signature,
    }


def initialize_environment(config: Config, spec: InitialConditionSpec) -> DoubleBottleneckEnv:
    """Create an exact fresh initial Markov state for an expert solve."""

    env = DoubleBottleneckEnv(config)
    env.reset(spec.positions, regime=spec.regime)
    if np.linalg.norm(spec.initial_velocities, axis=-1).max(initial=0.0) > config.max_speed + 1e-9:
        raise ValueError("initial last-applied velocity exceeds the plant speed bound")
    state = env.augmented_state()
    state["last_applied_velocity"] = spec.initial_velocities.copy()
    env.restore_augmented_state(state)
    return env


def _replay(
    config: Config,
    regime: str,
    initial_positions_value: np.ndarray,
    initial_velocities_value: np.ndarray,
    actions: np.ndarray,
) -> dict[str, Any]:
    replay_spec = InitialConditionSpec(
        condition_id="replay",
        family_id="replay",
        regime=regime,
        positions=initial_positions_value,
        initial_velocities=initial_velocities_value,
        perturbation={},
    )
    env = initialize_environment(config, replay_spec)
    positions = [env.positions.copy()]
    observations = [env.observation().copy()]
    min_pair = math.inf
    min_wall = math.inf
    info = None
    for index, action in enumerate(actions):
        if env.done:
            raise ValueError(f"trajectory contains action {index} after terminal event")
        observation, _, _, info = env.step(action)
        positions.append(env.positions.copy())
        observations.append(observation.copy())
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
    if info is None:
        raise ValueError("trajectory must contain at least one action")
    terminal_reason = str(info["termination"]) if env.done else "planner_exhausted"
    summary = env.summary()
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float32),
        "terminal_reason": terminal_reason,
        "success": bool(summary["collision_free_success"]),
        "wall_collision": bool(summary["wall_collision"]),
        "agent_collision": bool(summary["agent_collision"]),
        "deadlock": bool(summary["deadlock"]),
        "timeout": terminal_reason == "timeout",
        "episode_steps": int(summary["episode_steps"]),
        "min_pair": float(min_pair),
        "min_wall": float(min_wall),
        "goals": env.goals.copy(),
    }


def record_from_plan(
    plan: ExpertPlan,
    source_env: DoubleBottleneckEnv,
    spec: InitialConditionSpec,
    rollout_id: str,
) -> TrajectoryRecord:
    """Convert an expert plan after an independent plant replay."""

    if source_env.step_count != 0 or source_env.done:
        raise ValueError("source environment must be freshly reset")
    if source_env.initial_regime != spec.regime:
        raise ValueError("source environment regime does not match initial-condition spec")
    if not np.array_equal(source_env.positions, spec.positions):
        raise ValueError("source environment positions do not match initial-condition spec")
    if not np.array_equal(source_env.velocities, spec.initial_velocities):
        raise ValueError("source environment velocity state does not match initial-condition spec")

    actions = np.asarray(plan.actions, dtype=np.float64)
    replay = _replay(
        source_env.config,
        spec.regime,
        spec.positions,
        spec.initial_velocities,
        actions,
    )
    if not np.allclose(plan.positions, replay["positions"], rtol=0.0, atol=2e-12):
        raise ValueError("expert plan positions disagree with independent dynamics replay")
    if not np.allclose(plan.observations, replay["observations"], rtol=0.0, atol=2e-6):
        raise ValueError("expert plan observations disagree with independent replay")

    descriptor = environment_descriptor(source_env)
    mode = infer_coordination_mode(replay["positions"], replay["goals"], source_env.config)
    hypothesis = {
        "label": plan.hypothesis.label,
        "first_direction": plan.hypothesis.first_direction,
        "left_to_right_order": list(plan.hypothesis.left_to_right_order),
        "right_to_left_order": list(plan.hypothesis.right_to_left_order),
        "candidate_count": int(plan.candidate_count),
    }
    positions = replay["positions"]
    digest = trajectory_digest(positions, actions, replay["observations"])
    return TrajectoryRecord(
        rollout_id=str(rollout_id),
        family_id=spec.family_id,
        condition_id=spec.condition_id,
        regime=spec.regime,
        split="unassigned",
        initial_positions=spec.positions.copy(),
        initial_velocities=spec.initial_velocities,
        positions=positions,
        observations=replay["observations"],
        actions=actions.copy(),
        terminal_reason=replay["terminal_reason"],
        success=replay["success"],
        collision=bool(replay["wall_collision"] or replay["agent_collision"]),
        wall_collision=replay["wall_collision"],
        agent_collision=replay["agent_collision"],
        deadlock=replay["deadlock"],
        timeout=replay["timeout"],
        episode_steps=replay["episode_steps"],
        min_swept_pair_surface_distance=replay["min_pair"],
        min_swept_wall_clearance=replay["min_wall"],
        path_length=float(np.linalg.norm(actions, axis=-1).sum() * source_env.config.dt),
        solve_time_seconds=float(plan.solve_time_seconds),
        perturbation=dict(spec.perturbation),
        hypothesis=hypothesis,
        coordination_mode=mode,
        environment=descriptor,
        environment_fingerprint=environment_fingerprint(descriptor),
        config_fingerprint=source_env.config.fingerprint,
        trajectory_digest=digest,
    )


def validate_record(record: TrajectoryRecord) -> None:
    """Raise ``ValueError`` if a rollout is corrupt or semantically mislabeled."""

    prefix = f"record {record.rollout_id!r}: "
    arrays = {
        "initial_positions": np.asarray(record.initial_positions),
        "initial_velocities": np.asarray(record.initial_velocities),
        "positions": np.asarray(record.positions),
        "observations": np.asarray(record.observations),
        "actions": np.asarray(record.actions),
    }
    expected = {
        "initial_positions": (4, 2),
        "initial_velocities": (4, 2),
        "positions": (record.episode_steps + 1, 4, 2),
        "observations": (record.episode_steps + 1, 4, 18),
        "actions": (record.episode_steps, 4, 2),
    }
    for name, array in arrays.items():
        if array.shape != expected[name]:
            raise ValueError(prefix + f"{name} has shape {array.shape}, expected {expected[name]}")
        if not np.isfinite(array).all():
            raise ValueError(prefix + f"{name} contains non-finite values")
    if record.episode_steps <= 0:
        raise ValueError(prefix + "empty trajectory")
    if record.terminal_reason not in TERMINAL_REASONS:
        raise ValueError(prefix + f"unknown terminal reason {record.terminal_reason!r}")
    if record.split not in ("unassigned", "train", "val"):
        raise ValueError(prefix + f"invalid split {record.split!r}")
    if not np.array_equal(arrays["initial_positions"], arrays["positions"][0]):
        raise ValueError(prefix + "initial_positions differs from positions[0]")

    descriptor = _jsonable(record.environment)
    if record.environment_fingerprint != environment_fingerprint(descriptor):
        raise ValueError(prefix + "environment fingerprint mismatch")
    config_data = descriptor.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(prefix + "missing environment config")
    config = Config(**config_data)
    if record.config_fingerprint != config.fingerprint:
        raise ValueError(prefix + "config fingerprint mismatch")
    reference_env = DoubleBottleneckEnv(config)
    if descriptor != environment_descriptor(reference_env):
        raise ValueError(prefix + "environment descriptor differs from reconstructed plant")
    speed = np.linalg.norm(arrays["actions"], axis=-1)
    if speed.max(initial=0.0) > config.max_speed + 1e-9:
        raise ValueError(prefix + "action exceeds speed bound")
    initial_speed = np.linalg.norm(arrays["initial_velocities"], axis=-1)
    if initial_speed.max(initial=0.0) > config.max_speed + 1e-9:
        raise ValueError(prefix + "initial last-applied velocity exceeds speed bound")

    replay = _replay(
        config,
        record.regime,
        arrays["initial_positions"],
        arrays["initial_velocities"],
        arrays["actions"],
    )
    if not np.allclose(arrays["positions"], replay["positions"], rtol=0.0, atol=2e-12):
        raise ValueError(prefix + "positions violate single-integrator dynamics")
    if not np.allclose(arrays["observations"], replay["observations"], rtol=0.0, atol=2e-6):
        raise ValueError(prefix + "observations disagree with plant observation semantics")
    labels = {
        "terminal_reason": replay["terminal_reason"],
        "success": replay["success"],
        "wall_collision": replay["wall_collision"],
        "agent_collision": replay["agent_collision"],
        "deadlock": replay["deadlock"],
        "timeout": replay["timeout"],
        "episode_steps": replay["episode_steps"],
    }
    for name, expected_value in labels.items():
        if getattr(record, name) != expected_value:
            raise ValueError(prefix + f"{name} label disagrees with replay")
    if record.collision != bool(record.wall_collision or record.agent_collision):
        raise ValueError(prefix + "combined collision label is inconsistent")
    if record.success and (record.collision or record.terminal_reason != "success"):
        raise ValueError(prefix + "hidden collision/non-success trajectory labeled successful")
    if not math.isclose(
        record.min_swept_pair_surface_distance, replay["min_pair"], rel_tol=0.0, abs_tol=2e-12
    ):
        raise ValueError(prefix + "minimum swept pair clearance disagrees with replay")
    if not math.isclose(
        record.min_swept_wall_clearance, replay["min_wall"], rel_tol=0.0, abs_tol=2e-12
    ):
        raise ValueError(prefix + "minimum swept wall clearance disagrees with replay")
    expected_length = float(np.linalg.norm(arrays["actions"], axis=-1).sum() * config.dt)
    if not math.isclose(record.path_length, expected_length, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(prefix + "path length is inconsistent")
    if record.trajectory_digest != trajectory_digest(
        arrays["positions"], arrays["actions"], arrays["observations"]
    ):
        raise ValueError(prefix + "trajectory digest mismatch")


def deduplicate_records(
    records: Sequence[TrajectoryRecord],
) -> tuple[list[TrajectoryRecord], list[dict[str, str]]]:
    """Keep the first exact trajectory and report later duplicate provenance."""

    unique: list[TrajectoryRecord] = []
    owner: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for record in records:
        first = owner.get(record.trajectory_digest)
        if first is None:
            owner[record.trajectory_digest] = record.rollout_id
            unique.append(record)
        else:
            duplicates.append(
                {
                    "dropped_rollout_id": record.rollout_id,
                    "duplicate_of": first,
                    "trajectory_digest": record.trajectory_digest,
                }
            )
    return unique, duplicates


def _seeded_family_order(families: Iterable[str], seed: int) -> list[str]:
    return sorted(
        families,
        key=lambda family: hashlib.sha256(f"{seed}:{family}".encode("utf-8")).hexdigest(),
    )


def assign_grouped_split(
    records: Sequence[TrajectoryRecord],
    val_fraction: float = 0.25,
    seed: int = 17,
) -> list[TrajectoryRecord]:
    """Assign whole initial-condition families, stratified by regime."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between zero and one")
    family_regime: dict[str, str] = {}
    for record in records:
        previous = family_regime.setdefault(record.family_id, record.regime)
        if previous != record.regime:
            raise ValueError(f"family {record.family_id!r} spans multiple regimes")
    validation: set[str] = set()
    for regime in INITIAL_REGIMES:
        families = _seeded_family_order(
            (family for family, value in family_regime.items() if value == regime), seed
        )
        if len(families) < 2:
            continue
        count = min(len(families) - 1, max(1, int(round(len(families) * val_fraction))))
        validation.update(families[:count])
    return [
        replace(record, split="val" if record.family_id in validation else "train")
        for record in records
    ]


def validate_records(records: Sequence[TrajectoryRecord]) -> dict[str, Any]:
    """Validate a record collection and summarize Gate-C-relevant properties."""

    if not records:
        raise ValueError("dataset contains no records")
    rollout_ids: set[str] = set()
    digests: set[str] = set()
    family_splits: dict[str, str] = {}
    for record in records:
        validate_record(record)
        if record.rollout_id in rollout_ids:
            raise ValueError(f"duplicate rollout id {record.rollout_id!r}")
        if record.trajectory_digest in digests:
            raise ValueError(f"exact duplicate trajectory {record.trajectory_digest}")
        rollout_ids.add(record.rollout_id)
        digests.add(record.trajectory_digest)
        if record.split != "unassigned":
            previous = family_splits.setdefault(record.family_id, record.split)
            if previous != record.split:
                raise ValueError(f"initial-condition family {record.family_id!r} leaks across splits")
    fingerprints = {record.environment_fingerprint for record in records}
    if len(fingerprints) != 1:
        raise ValueError("dataset mixes environment fingerprints")

    successes = sum(record.training_eligible for record in records)
    collisions = sum(record.collision for record in records)
    timeouts = sum(record.timeout for record in records)
    deadlocks = sum(record.deadlock for record in records)
    regime_counts = {
        regime: sum(record.regime == regime for record in records) for regime in INITIAL_REGIMES
    }
    split_counts = {
        split: sum(record.split == split for record in records) for split in ("train", "val")
    }
    return {
        "rollouts": len(records),
        "transitions": int(sum(record.length for record in records)),
        "training_eligible_rollouts": int(successes),
        "success_rate": successes / len(records),
        "collision_rate": collisions / len(records),
        "timeout_rate": timeouts / len(records),
        "deadlock_rate": deadlocks / len(records),
        "regime_counts": regime_counts,
        "split_counts": split_counts,
        "families": len(family_splits or {record.family_id for record in records}),
        "environment_fingerprint": next(iter(fingerprints)),
        "initial_velocity_protocol": INITIAL_VELOCITY_PROTOCOL,
    }


def analyze_mode_diversity(records: Sequence[TrajectoryRecord]) -> dict[str, Any]:
    """Report actual discrete crossing modes, emphasizing identical starts."""

    successful = [record for record in records if record.training_eligible]
    signatures: dict[str, int] = {}
    for record in successful:
        signature = str(record.coordination_mode.get("signature", "incomplete"))
        signatures[signature] = signatures.get(signature, 0) + 1

    exact_groups: dict[str, list[TrajectoryRecord]] = {}
    for record in successful:
        start_key = hashlib.sha256(
            np.ascontiguousarray(
                np.concatenate((record.initial_positions, record.initial_velocities), axis=-1),
                dtype=np.float64,
            ).tobytes()
        ).hexdigest()
        exact_groups.setdefault(start_key, []).append(record)
    multimodal_groups = []
    for start_key, group in exact_groups.items():
        group_signatures = sorted(
            {str(record.coordination_mode.get("signature", "incomplete")) for record in group}
        )
        if len(group_signatures) > 1:
            multimodal_groups.append(
                {
                    "initial_state_sha256": start_key,
                    "condition_ids": sorted({record.condition_id for record in group}),
                    "rollout_ids": sorted(record.rollout_id for record in group),
                    "mode_signatures": group_signatures,
                }
            )
    return {
        "protocol": "actual_directed_plane_crossing_orders_v1",
        "successful_rollouts": len(successful),
        "unique_successful_mode_signatures": len(signatures),
        "mode_counts": dict(sorted(signatures.items())),
        "identical_initial_states_with_multiple_successful_modes": len(multimodal_groups),
        "multimodal_initial_state_groups": multimodal_groups,
        "genuine_multimodality_observed": bool(multimodal_groups),
        "interpretation": (
            "Multiple signatures count only when actual crossing orders differ for an exactly "
            "identical initial state; hypothesis labels alone are not evidence."
        ),
    }


def pilot_initial_condition_specs(config: Config | None = None) -> tuple[InitialConditionSpec, ...]:
    """Twelve restrained pilot starts (four per regime); no solves occur here."""

    config = config or Config()
    specs: list[InitialConditionSpec] = []
    travel = np.asarray((1.0, 1.0, -1.0, -1.0))
    lateral_pattern = np.asarray((0.025, -0.020, -0.025, 0.020))
    symmetry_delta = np.asarray(
        ((0.018, 0.012), (-0.014, -0.010), (0.018, -0.012), (-0.014, 0.010))
    )
    velocity_pattern = np.asarray(
        ((0.035, 0.010), (0.020, -0.012), (-0.030, -0.008), (-0.018, 0.010))
    )

    def add(
        regime: str,
        family: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        details: dict[str, Any],
    ) -> None:
        condition_id = f"{regime}__{family}"
        spec = InitialConditionSpec(
            condition_id=condition_id,
            # The two opposite-first-direction solves of this exact state must
            # remain together.  No near-copy of this state uses another ID.
            family_id=condition_id,
            regime=regime,
            positions=positions,
            initial_velocities=velocities,
            perturbation={"family": family, **details},
        )
        initialize_environment(config, spec)
        specs.append(spec)

    for regime_index, regime in enumerate(INITIAL_REGIMES):
        base = initial_positions(config, regime)
        zeros = np.zeros((4, 2), dtype=np.float64)
        add(
            regime,
            "nominal",
            base,
            zeros,
            {"delta_positions": zeros.tolist(), "delta_initial_velocities": zeros.tolist()},
        )

        # Regime-dependent signs cover both nearer and farther starts while
        # individual magnitudes alter longitudinal offsets within each wave.
        distance_sign = 1.0 if regime_index != 1 else -1.0
        longitudinal_amount = np.asarray((0.070, 0.040, 0.060, 0.035))
        distance_delta = np.zeros((4, 2), dtype=np.float64)
        distance_delta[:, 0] = distance_sign * travel * longitudinal_amount
        add(
            regime,
            "bottleneck_distance_longitudinal_offsets",
            base + distance_delta,
            zeros,
            {
                "axis": "longitudinal",
                "toward_or_away": "toward" if distance_sign > 0 else "away",
                "delta_positions": distance_delta.tolist(),
                "delta_initial_velocities": zeros.tolist(),
            },
        )

        lateral_delta = np.zeros((4, 2), dtype=np.float64)
        lateral_delta[:, 1] = (1.0 if regime_index % 2 == 0 else -1.0) * lateral_pattern
        add(
            regime,
            "lateral_offsets",
            base + lateral_delta,
            zeros,
            {
                "axis": "lateral",
                "delta_positions": lateral_delta.tolist(),
                "delta_initial_velocities": zeros.tolist(),
            },
        )

        # Weak/near regimes receive a reflection-skew perturbation; the clear
        # regime keeps its geometry and isolates velocity-state coverage.
        state_delta = zeros if regime == "clearly_asymmetric" else symmetry_delta
        initial_velocity = (1.0 if regime_index != 1 else -1.0) * velocity_pattern
        add(
            regime,
            "velocity_and_symmetry_probe",
            base + state_delta,
            initial_velocity,
            {
                "axis": "last_applied_velocity_and_optional_reflection_skew",
                "weak_symmetry_perturbation": regime != "clearly_asymmetric",
                "delta_positions": state_delta.tolist(),
                "delta_initial_velocities": initial_velocity.tolist(),
            },
        )
    return tuple(specs)


def pilot_rollout_requests(
    specs: Sequence[InitialConditionSpec] | None = None,
) -> tuple[tuple[InitialConditionSpec, CoordinationHypothesis], ...]:
    """Probe both opposite first-direction modes for every identical start."""

    specs = tuple(pilot_initial_condition_specs() if specs is None else specs)
    directional = (
        CoordinationHypothesis("left_to_right"),
        CoordinationHypothesis("right_to_left"),
    )
    return tuple((spec, hypothesis) for spec in specs for hypothesis in directional)


def all_mode_rollout_requests(
    specs: Sequence[InitialConditionSpec] | None = None,
) -> tuple[tuple[InitialConditionSpec, CoordinationHypothesis], ...]:
    """Probe all eight planner hypotheses for every identical pilot start.

    The hypotheses are only candidate modes.  Dataset generation must still
    infer the executed crossing signatures and reject the collection unless
    every start realizes eight distinct successful modes.
    """

    specs = tuple(pilot_initial_condition_specs() if specs is None else specs)
    hypotheses = all_coordination_hypotheses()
    return tuple((spec, hypothesis) for spec in specs for hypothesis in hypotheses)


def _record_metadata(record: TrajectoryRecord) -> dict[str, Any]:
    excluded = {"initial_positions", "initial_velocities", "positions", "observations", "actions"}
    return {
        key: _jsonable(value)
        for key, value in asdict(record).items()
        if key not in excluded
    }


def save_dataset(
    output: Path | str,
    records: Sequence[TrajectoryRecord],
    duplicates_dropped: Sequence[Mapping[str, str]] = (),
    generation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically create a new pilot dataset directory."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset path {output}")
    report = validate_records(records)
    mode_analysis = analyze_mode_diversity(records)
    output.mkdir(parents=True, exist_ok=False)
    rollouts_dir = output / "rollouts"
    rollouts_dir.mkdir()
    file_entries = []
    for index, record in enumerate(records):
        filename = f"episode_{index:04d}.npz"
        metadata = _record_metadata(record)
        np.savez_compressed(
            rollouts_dir / filename,
            schema=np.asarray(ROLLOUT_SCHEMA),
            metadata_json=np.asarray(_canonical_json(metadata)),
            initial_positions=np.asarray(record.initial_positions, dtype=np.float64),
            initial_velocities=np.asarray(record.initial_velocities, dtype=np.float64),
            positions=np.asarray(record.positions, dtype=np.float64),
            observations=np.asarray(record.observations, dtype=np.float32),
            actions=np.asarray(record.actions, dtype=np.float64),
        )
        file_entries.append(
            {
                "file": f"rollouts/{filename}",
                "rollout_id": record.rollout_id,
                "family_id": record.family_id,
                "split": record.split,
                "training_eligible": record.training_eligible,
                "trajectory_digest": record.trajectory_digest,
            }
        )
    manifest = {
        "schema": DATASET_SCHEMA,
        "complete": True,
        "trajectory_first": True,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "initial_velocity_protocol": INITIAL_VELOCITY_PROTOCOL,
        "environment": _jsonable(records[0].environment),
        "environment_fingerprint": records[0].environment_fingerprint,
        "quality_report": report,
        "mode_analysis": mode_analysis,
        "duplicates_dropped": [_jsonable(item) for item in duplicates_dropped],
        "generation_metadata": _jsonable(generation_metadata or {}),
        "files": file_entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_record(path: Path | str) -> TrajectoryRecord:
    """Load one rollout without pickle/object arrays."""

    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != ROLLOUT_SCHEMA:
            raise ValueError(f"unsupported rollout schema in {path}")
        metadata = json.loads(str(data["metadata_json"]))
        return TrajectoryRecord(
            **metadata,
            initial_positions=data["initial_positions"].copy(),
            initial_velocities=data["initial_velocities"].copy(),
            positions=data["positions"].copy(),
            observations=data["observations"].copy(),
            actions=data["actions"].copy(),
        )


__all__ = (
    "ACTION_SCHEMA",
    "DATASET_SCHEMA",
    "INITIAL_VELOCITY_PROTOCOL",
    "InitialConditionSpec",
    "OBSERVATION_SCHEMA",
    "ROLLOUT_SCHEMA",
    "TrajectoryRecord",
    "all_mode_rollout_requests",
    "analyze_mode_diversity",
    "assign_grouped_split",
    "deduplicate_records",
    "environment_descriptor",
    "environment_fingerprint",
    "infer_coordination_mode",
    "load_record",
    "pilot_initial_condition_specs",
    "pilot_rollout_requests",
    "record_from_plan",
    "save_dataset",
    "trajectory_digest",
    "validate_record",
    "validate_records",
)
