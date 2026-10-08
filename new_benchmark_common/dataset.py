"""Auditable trajectory-first data storage and recovery collection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .protocol import ExpertContinuation, ScenarioProtocol


DATASET_SCHEMA = "new_benchmark_joint_stage1_dataset_v1"
TRAJECTORY_SCHEMA = "new_benchmark_joint_stage1_trajectory_v1"
_SOURCES = {"nominal", "uniform_recovery", "gate_local_recovery", "targeted_wall_obstacle", "targeted_agent"}


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        hasher.update(str(value.shape).encode())
        hasher.update(value.dtype.str.encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class RecoveryAudit:
    """Required provenance for every recovery trajectory.

    Targeted samples must name a train/dev rollout and a valid pre-collision
    state; a post-collision/inside-obstacle state cannot be admitted.
    """

    source_rollout_id: str
    source_split: str
    source_time: int
    collision_type: str | None
    collision_identity: str | None
    distance_to_collision: float | None
    perturbation_seed: int
    expert_success: bool


@dataclass(frozen=True)
class Trajectory:
    rollout_id: str
    split: str
    source: str
    initial_state: Any
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    recovery_audit: RecoveryAudit | None = None

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class DatasetWriter:
    """Write independently split, digest-protected data without mode labels."""

    def __init__(
        self, root: str | Path, scenario: ScenarioProtocol, *, scenario_config: Mapping[str, Any]
    ):
        self.root = Path(root)
        self.scenario = scenario
        self.scenario_config = dict(scenario_config)
        self.records: list[dict[str, Any]] = []
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"dataset directory must be empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, trajectory: Trajectory) -> Path:
        if trajectory.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if trajectory.source not in _SOURCES:
            raise ValueError(f"unknown data source {trajectory.source!r}")
        if trajectory.source == "targeted_wall_obstacle":
            if trajectory.recovery_audit is None or trajectory.recovery_audit.collision_type not in {"wall", "obstacle"}:
                raise ValueError("wall/obstacle recovery requires matching audit record")
        if trajectory.source != "nominal" and trajectory.recovery_audit is None:
            raise ValueError("all recovery trajectories require provenance")
        if trajectory.recovery_audit and trajectory.recovery_audit.source_split == "test":
            raise ValueError("untouched test trajectories may never source recovery data")
        if trajectory.length <= 0 or trajectory.observations.shape[0] != trajectory.length + 1:
            raise ValueError("trajectory must have T actions and T+1 observations")
        if trajectory.observations.shape[1:] != self.scenario.observation_shape:
            raise ValueError("observation shape conflicts with scenario contract")
        if trajectory.actions.shape[1:] != self.scenario.action_shape:
            raise ValueError("action shape conflicts with scenario contract")
        if not all(np.isfinite(a).all() for a in (trajectory.states, trajectory.observations, trajectory.actions)):
            raise ValueError("non-finite trajectory data")
        if not bool(trajectory.metadata.get("success", True)):
            raise ValueError("only successful expert continuations are training eligible")
        # Digest the exact on-disk dtypes, rather than the caller's potentially
        # float64 observations/actions.
        stored_states = np.asarray(trajectory.states)
        stored_observations = np.asarray(trajectory.observations, dtype=np.float32)
        stored_actions = np.asarray(trajectory.actions, dtype=np.float32)
        digest = _digest(stored_states, stored_observations, stored_actions)
        relative = Path("rollouts") / trajectory.split / f"{trajectory.rollout_id}.npz"
        path = self.root / relative
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(trajectory.metadata)
        metadata.update(rollout_id=trajectory.rollout_id, split=trajectory.split,
                        source=trajectory.source, trajectory_digest=digest, success=True)
        np.savez_compressed(
            path, schema=np.asarray(TRAJECTORY_SCHEMA), states=stored_states,
            observations=stored_observations, actions=stored_actions,
            initial_state_json=np.asarray(json.dumps(_jsonable(trajectory.initial_state), sort_keys=True)),
            metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
            recovery_audit_json=np.asarray(json.dumps(asdict(trajectory.recovery_audit) if trajectory.recovery_audit else None, sort_keys=True)),
        )
        self.records.append({"file": str(relative), "rollout_id": trajectory.rollout_id,
                             "split": trajectory.split, "source": trajectory.source,
                             "trajectory_digest": digest, "length": trajectory.length})
        return path

    def finalize(self, *, extra_report: Mapping[str, Any] | None = None) -> Path:
        split_counts = {split: sum(r["split"] == split for r in self.records) for split in ("train", "dev", "test")}
        source_counts = {source: sum(r["source"] == source for r in self.records) for source in sorted(_SOURCES)}
        manifest = {
            "schema": DATASET_SCHEMA, "complete": True, "scenario": self.scenario.name,
            "environment_fingerprint": self.scenario.environment_fingerprint,
            "agent_order": list(self.scenario.agent_order),
            "observation_shape": list(self.scenario.observation_shape),
            "action_shape": list(self.scenario.action_shape), "scenario_config": _jsonable(self.scenario_config),
            "files": self.records, "counts": {"split": split_counts, "source": source_counts},
            "recovery_protocol": getattr(self.scenario, "recovery_protocol",
                "uniform anchors + local perturb/re-query; targeted sources limited to train/dev valid pre-collision states"),
            "extra_report": _jsonable(extra_report or {}),
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return path


class JointTransitionDataset:
    """Uniform-transition sampler, with source accounting retained for ablations."""

    def __init__(self, root: str | Path, split: str, *, seed: int = 0):
        root = Path(root)
        manifest_path = root / "manifest.json"
        self.manifest_bytes = manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.manifest = json.loads(self.manifest_bytes)
        if self.manifest.get("schema") != DATASET_SCHEMA or not self.manifest.get("complete"):
            raise ValueError("incomplete or wrong dataset schema")
        if split not in {"train", "dev", "test", "all"}:
            raise ValueError("invalid split")
        self.root, self.split = root, split
        self.environment_fingerprint = str(self.manifest["environment_fingerprint"])
        self.agent_order = tuple(self.manifest["agent_order"])
        self.observation_shape = tuple(self.manifest["observation_shape"])
        self.action_shape = tuple(self.manifest["action_shape"])
        self.trajectories = tuple(self._load(row) for row in self.manifest["files"] if split == "all" or row["split"] == split)
        if not self.trajectories:
            raise ValueError(f"no trajectories in split {split}")
        self.observations = np.concatenate([t.observations[:-1] for t in self.trajectories]).astype(np.float32)
        self.actions = np.concatenate([t.actions for t in self.trajectories]).astype(np.float32)
        self.sources = np.concatenate([np.repeat(t.source, t.length) for t in self.trajectories])
        self._rng = np.random.default_rng(seed)

    def _load(self, row: Mapping[str, Any]) -> Trajectory:
        path = self.root / row["file"]
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"].item()) != TRAJECTORY_SCHEMA:
                raise ValueError(f"wrong trajectory schema: {path}")
            states, observations, actions = data["states"], data["observations"], data["actions"]
            metadata = json.loads(str(data["metadata_json"].item()))
            audit_raw = json.loads(str(data["recovery_audit_json"].item()))
            initial_state = json.loads(str(data["initial_state_json"].item()))
        if observations.shape[1:] != self.observation_shape or actions.shape[1:] != self.action_shape:
            raise ValueError(f"shape mismatch: {path}")
        if observations.shape[0] != actions.shape[0] + 1 or len(actions) == 0:
            raise ValueError(f"invalid transition count: {path}")
        digest = _digest(states, observations, actions)
        if digest != row["trajectory_digest"] or digest != metadata.get("trajectory_digest"):
            raise ValueError(f"digest mismatch: {path}")
        audit = RecoveryAudit(**audit_raw) if audit_raw else None
        return Trajectory(str(row["rollout_id"]), str(row["split"]), str(row["source"]), initial_state,
                          states, observations, actions, metadata, audit)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        index = self._rng.integers(len(self.actions), size=batch_size)
        return {"observations": self.observations[index], "actions": self.actions[index]}

    def all_transitions(self):
        return self.observations.copy(), self.actions.copy()

    def source_counts(self) -> dict[str, int]:
        return {name: int((self.sources == name).sum()) for name in sorted(_SOURCES)}

    def __len__(self):
        return len(self.actions)


def fit_train_normalization(dataset: JointTransitionDataset, floor: float = 0.01,
                            shared_agents: bool = False,
                            active_action_scale: bool = False) -> dict[str, tuple[float, ...]]:
    if dataset.split != "train":
        raise ValueError("normalization must be fit exclusively on train")
    if active_action_scale and not shared_agents:
        raise ValueError("active-action normalization requires shared-agent normalization")
    result = {}
    for name, values in (("obs", dataset.observations), ("act", dataset.actions)):
        flat = values.reshape(len(values), -1).astype(np.float64)
        if shared_agents:
            pooled = values.reshape(-1, values.shape[-1]).astype(np.float64)
            if name == "act" and active_action_scale:
                # At large N, safe one-way data contain many waiting robots.
                # Scaling actions by the all-agent standard deviation makes
                # the same physical 0.5 m/s command grow with 1/sqrt(active
                # fraction). Fit units on executed moving commands instead.
                pooled = pooled[np.linalg.norm(pooled, axis=1) > 0.1]
                if not len(pooled):
                    raise ValueError("no moving actions for active normalization")
            mean = np.tile(pooled.mean(axis=0), values.shape[1])
            scale = np.tile(np.maximum(pooled.std(axis=0), floor), values.shape[1])
        else:
            mean = flat.mean(axis=0)
            scale = np.maximum(flat.std(axis=0), floor)
        result[f"{name}_mean"] = tuple(mean)
        result[f"{name}_scale"] = tuple(scale)
    return result


def continuation_to_trajectory(continuation: ExpertContinuation, *, rollout_id: str, split: str,
                               source: str, initial_state: Any, metadata: Mapping[str, Any] | None = None,
                               recovery_audit: RecoveryAudit | None = None) -> Trajectory:
    if not continuation.success:
        raise ValueError("failed expert continuation may not become a training trajectory")
    details = dict(metadata or {})
    details.update(continuation.metadata, success=True, terminal_reason=continuation.terminal_reason)
    audit = replace(recovery_audit, expert_success=True) if recovery_audit else None
    return Trajectory(rollout_id, split, source, initial_state, np.asarray(continuation.states),
                      np.asarray(continuation.observations), np.asarray(continuation.actions), details, audit)


def uniform_requery_candidates(trajectory: Trajectory, scenario: ScenarioProtocol, *, anchors: int,
                               seed: int) -> Iterable[tuple[int, Any, RecoveryAudit]]:
    """Uniformly spaced generic anchors; scenario invokes its expert for each state."""
    if anchors <= 0:
        raise ValueError("anchors must be positive")
    indices = np.unique(np.linspace(0, trajectory.length - 1, min(anchors, trajectory.length), dtype=int))
    root = np.random.default_rng(seed)
    for time in indices:
        perturb_seed = int(root.integers(0, 2**31 - 1))
        state = _recovery_state_at(scenario, trajectory, int(time))
        state = scenario.perturb_state(state, np.random.default_rng(perturb_seed))
        if scenario.valid_state(state):
            yield int(time), state, RecoveryAudit(trajectory.rollout_id, trajectory.split, int(time), None, None, None,
                                                   perturb_seed, False)


def targeted_requery_candidates(trajectory: Trajectory, scenario: ScenarioProtocol, *, precursor_steps: Iterable[int],
                                collision_type: str, collision_identity: str | None, distance_to_collision: float,
                                seed: int) -> Iterable[tuple[int, Any, RecoveryAudit]]:
    """Generate only valid pre-collision candidates from train/dev rollout evidence."""
    if trajectory.split not in {"train", "dev"}:
        raise ValueError("targeted recovery may use train/dev failures only")
    if collision_type not in {"wall", "obstacle", "agent"}:
        raise ValueError("unsupported collision type")
    root = np.random.default_rng(seed)
    for time in precursor_steps:
        if not 0 <= time < trajectory.length:
            raise ValueError("precursor index outside valid action window")
        perturb_seed = int(root.integers(0, 2**31 - 1))
        state = _recovery_state_at(scenario, trajectory, int(time))
        state = scenario.perturb_state(state, np.random.default_rng(perturb_seed))
        if scenario.valid_state(state):
            yield int(time), state, RecoveryAudit(trajectory.rollout_id, trajectory.split, int(time), collision_type,
                                                   collision_identity, float(distance_to_collision), perturb_seed, False)


def _recovery_state_at(scenario: ScenarioProtocol, trajectory: Trajectory, time: int) -> Any:
    """Recover an opaque plant state while retaining legacy array-only support.

    Most single-integrator scenarios can re-query from ``trajectory.states[t]``.
    A plant whose goals, velocities, timers, or other hidden physical quantities
    must survive a recovery query implements the optional
    ``recovery_state_at(trajectory, time)`` hook.  This remains physical state,
    not a coordination label.
    """
    hook = getattr(scenario, "recovery_state_at", None)
    return hook(trajectory, time) if hook is not None else trajectory.states[time]


def collect_nominal_and_uniform(
    scenario: ScenarioProtocol,
    root: str | Path,
    *,
    scenario_config: Mapping[str, Any],
    nominal_counts: Mapping[str, int],
    uniform_anchors: int,
    seed: int,
    rollout_prefix: str = "",
) -> dict[str, Any]:
    """Materialize broad independent nominal states and train/dev recovery data.

    ``nominal_counts`` must explicitly include train, dev, and test.  The test
    split gets independently sampled nominal expert states only: neither a test
    trajectory nor a test failure can source recovery data.  Retry counts are
    reported rather than silently hidden, making Gate A/C acquisition auditable.

    For a Base-U+W ablation, invoke this function twice with the same arguments
    and seed (once to a Base-U root and once to a Base-U+W root), then add
    target-derived continuations to the second writer before its ``finalize``.
    Scenarios that need that workflow should use ``DatasetWriter`` directly so
    the same deterministic nominal/uniform loop can be retained verbatim.
    """
    required = {"train", "dev", "test"}
    if set(nominal_counts) != required or any(int(nominal_counts[s]) <= 0 for s in required):
        raise ValueError("nominal_counts must contain positive train/dev/test counts")
    writer = DatasetWriter(root, scenario, scenario_config=scenario_config)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"nominal_requested": dict(nominal_counts), "rollout_prefix": rollout_prefix, "nominal_success": {},
                              "expert_failures": {}, "uniform_recovery_success": 0,
                              "uniform_recovery_failures": 0, "targeted_recovery_transitions": 0}
    serial = 0
    for split in ("train", "dev", "test"):
        successful = failures = 0
        # Bound retries so an unreliable expert cannot silently yield a narrow
        # cherry-picked distribution.
        attempts = 0
        while successful < int(nominal_counts[split]):
            attempts += 1
            if attempts > 20 * int(nominal_counts[split]):
                raise RuntimeError(f"expert failed too often while collecting {split}")
            initial = scenario.sample_initial_state(split, rng)
            continuation = scenario.expert(initial, rng)
            if not continuation.success:
                failures += 1
                continue
            nominal = continuation_to_trajectory(
            continuation, rollout_id=f"{rollout_prefix}{split}_nominal_{serial:06d}", split=split,
                source="nominal", initial_state=initial,
            )
            serial += 1
            writer.add(nominal)
            successful += 1
            if split == "test":
                continue
            for time, perturbed, audit in uniform_requery_candidates(
                nominal, scenario, anchors=uniform_anchors, seed=int(rng.integers(0, 2**31 - 1))
            ):
                recovery = scenario.expert(perturbed, np.random.default_rng(audit.perturbation_seed))
                if not recovery.success:
                    report["uniform_recovery_failures"] += 1
                    continue
                writer.add(continuation_to_trajectory(
                    recovery, rollout_id=f"{rollout_prefix}{split}_uniform_{serial:06d}", split=split,
                    source="uniform_recovery", initial_state=perturbed,
                    metadata={"anchor_rollout_id": nominal.rollout_id, "anchor_time": time},
                    recovery_audit=audit,
                ))
                serial += 1
                report["uniform_recovery_success"] += 1
        report["nominal_success"][split], report["expert_failures"][split] = successful, failures
    writer.finalize(extra_report=report)
    return report
