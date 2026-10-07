"""Strict trajectory-first adapter for the validated 4-agent expert pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .flowbc_4a_agent import AGENT_ORDER, ACT_DIM, NUM_AGENTS, OBS_DIM


DATASET_SCHEMA = "double_bottleneck_expert_dataset_v1"
ROLLOUT_SCHEMA = "double_bottleneck_expert_trajectory_v1"


def _trajectory_digest(positions, observations, actions) -> str:
    digest = hashlib.sha256()
    for array in (positions, observations, actions):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _observations(positions, initial_velocity, actions, goals) -> np.ndarray:
    count = positions.shape[0]
    velocities = np.empty_like(positions)
    velocities[0] = initial_velocity
    velocities[1:] = actions
    rows = np.empty((count, NUM_AGENTS, OBS_DIM), dtype=np.float64)
    for time in range(count):
        for agent in range(NUM_AGENTS):
            relative = []
            for other in range(NUM_AGENTS):
                if other != agent:
                    relative.extend(
                        (
                            positions[time, other] - positions[time, agent],
                            velocities[time, other] - velocities[time, agent],
                        )
                    )
            rows[time, agent] = np.concatenate(
                (
                    positions[time, agent],
                    velocities[time, agent],
                    goals[agent] - positions[time, agent],
                    *relative,
                )
            )
    return rows


@dataclass(frozen=True)
class Episode:
    path: Path
    family_id: str
    split: str
    regime: str
    rollout_id: str
    mode_signature: str
    initial_positions: np.ndarray
    initial_velocities: np.ndarray
    positions: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class FlowBC4ADataset:
    """Validated episodes with Toy-identical uniform-transition sampling."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "all"] = "train",
        seed: int = 0,
    ):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.manifest = json.loads(self.manifest_bytes)
        self._validate_manifest()
        if split not in ("train", "val", "all"):
            raise ValueError("split must be train, val, or all")
        self.split = split
        self.environment = dict(self.manifest["environment"])
        self.environment_fingerprint = str(self.manifest["environment_fingerprint"])
        self.config = dict(self.environment["config"])
        self.goals = np.asarray(self.environment["goals"], dtype=np.float64)
        self._rng = np.random.default_rng(seed)

        family_splits: dict[str, str] = {}
        digests: set[str] = set()
        selected = []
        for entry in self.manifest["files"]:
            family = str(entry["family_id"])
            entry_split = str(entry["split"])
            previous = family_splits.setdefault(family, entry_split)
            if previous != entry_split:
                raise ValueError(f"family {family!r} leaks across splits")
            digest = str(entry["trajectory_digest"])
            if digest in digests:
                raise ValueError(f"duplicate trajectory digest {digest}")
            digests.add(digest)
            if not bool(entry.get("training_eligible", False)):
                raise ValueError("pilot manifest contains a noneligible rollout")
            if split == "all" or entry_split == split:
                selected.append(self._load_episode(entry))
        if not selected:
            raise ValueError(f"no episodes selected for split {split}")
        self.episodes = tuple(selected)
        by_family: dict[str, list[Episode]] = {}
        for episode in self.episodes:
            by_family.setdefault(episode.family_id, []).append(episode)
        self.by_family = {name: tuple(items) for name, items in sorted(by_family.items())}
        self.family_names = tuple(self.by_family)
        self.n_transitions = sum(episode.length for episode in self.episodes)
        self.observations = np.concatenate(
            [episode.observations[:-1] for episode in self.episodes]
        ).astype(np.float32)
        self.actions = np.concatenate(
            [episode.actions for episode in self.episodes]
        ).astype(np.float32)
        if len(self.observations) != self.n_transitions or len(self.actions) != self.n_transitions:
            raise AssertionError("transition concatenation disagrees with episode lengths")

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema") != DATASET_SCHEMA or not manifest.get("complete", False):
            raise ValueError("dataset manifest is incomplete or has the wrong schema")
        environment = manifest.get("environment", {})
        if tuple(environment.get("observation_shape", ())) != (NUM_AGENTS, OBS_DIM):
            raise ValueError("manifest observation shape is not [4,18]")
        if tuple(environment.get("action_shape", ())) != (NUM_AGENTS, ACT_DIM):
            raise ValueError("manifest action shape is not [4,2]")
        if tuple(environment.get("agent_names", ())) != AGENT_ORDER:
            raise ValueError("manifest does not use canonical agent order")
        fingerprint = manifest.get("environment_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("invalid fixed-geometry fingerprint")
        report = manifest.get("quality_report", {})
        if report.get("environment_fingerprint") != fingerprint:
            raise ValueError("quality report fingerprint mismatch")
        if any(float(report.get(name, 1.0)) != 0.0 for name in ("collision_rate", "deadlock_rate", "timeout_rate")):
            raise ValueError("dataset quality report admits unsafe or incomplete rollouts")
        if float(report.get("success_rate", 0.0)) != 1.0:
            raise ValueError("dataset quality report is not all-success")

    def _load_episode(self, entry: dict[str, Any]) -> Episode:
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest file path escapes dataset root")
        path = self.root / relative
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != ROLLOUT_SCHEMA:
                raise ValueError(f"wrong rollout schema in {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            initial_positions = np.asarray(archive["initial_positions"], dtype=np.float64)
            initial_velocities = np.asarray(archive["initial_velocities"], dtype=np.float64)
            positions = np.asarray(archive["positions"], dtype=np.float64)
            observations = np.asarray(archive["observations"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float64)

        length = actions.shape[0] if actions.ndim == 3 else -1
        expected = {
            "initial_positions": (NUM_AGENTS, ACT_DIM),
            "initial_velocities": (NUM_AGENTS, ACT_DIM),
            "positions": (length + 1, NUM_AGENTS, ACT_DIM),
            "observations": (length + 1, NUM_AGENTS, OBS_DIM),
            "actions": (length, NUM_AGENTS, ACT_DIM),
        }
        arrays = {
            "initial_positions": initial_positions,
            "initial_velocities": initial_velocities,
            "positions": positions,
            "observations": observations,
            "actions": actions,
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"invalid {name} in {path}: {arrays[name].shape} != {shape}")
        if length <= 0 or int(metadata.get("episode_steps", -1)) != length:
            raise ValueError(f"episode length mismatch in {path}")
        if metadata.get("environment_fingerprint") != self.manifest["environment_fingerprint"]:
            raise ValueError(f"environment fingerprint mismatch in {path}")
        for name in ("family_id", "split", "rollout_id", "trajectory_digest"):
            if str(metadata.get(name)) != str(entry.get(name)):
                raise ValueError(f"manifest/rollout {name} mismatch in {path}")
        if not (
            metadata.get("success") is True
            and metadata.get("terminal_reason") == "success"
            and metadata.get("collision") is False
            and metadata.get("deadlock") is False
            and metadata.get("timeout") is False
        ):
            raise ValueError(f"non-clean rollout presented as expert data: {path}")

        config = self.manifest["environment"]["config"]
        max_speed = float(config["max_speed"])
        dt = float(config["dt"])
        if np.linalg.norm(actions, axis=-1).max() > max_speed + 1e-9:
            raise ValueError(f"overspeed expert action in {path}")
        residual = positions[1:] - positions[:-1] - dt * actions
        if np.abs(residual).max() > 5e-12:
            raise ValueError(f"action/state dynamics mismatch in {path}")
        if not np.allclose(positions[0], initial_positions, rtol=0, atol=1e-12):
            raise ValueError(f"initial position mismatch in {path}")
        reconstructed = _observations(positions, initial_velocities, actions, self.goals)
        if not np.allclose(observations, reconstructed, rtol=0, atol=8e-7):
            raise ValueError(f"stored observation does not match canonical construction in {path}")
        final_errors = np.linalg.norm(self.goals - positions[-1], axis=-1)
        if np.any(final_errors > float(config["goal_tolerance"]) + 1e-9):
            raise ValueError(f"success endpoint outside goal tolerance in {path}")
        if float(metadata["min_swept_pair_surface_distance"]) <= float(config["agent_collision_margin"]):
            raise ValueError(f"hidden swept agent collision in {path}")
        if float(metadata["min_swept_wall_clearance"]) <= float(config["wall_collision_margin"]):
            raise ValueError(f"hidden swept wall collision in {path}")
        digest = _trajectory_digest(positions, observations, actions)
        if digest != entry["trajectory_digest"] or digest != metadata["trajectory_digest"]:
            raise ValueError(f"trajectory digest mismatch in {path}")

        mode = metadata.get("coordination_mode", {})
        return Episode(
            path=path,
            family_id=str(entry["family_id"]),
            split=str(entry["split"]),
            regime=str(metadata["regime"]),
            rollout_id=str(entry["rollout_id"]),
            mode_signature=str(mode.get("signature", "")),
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
            positions=positions,
            observations=observations,
            actions=actions.astype(np.float32),
            metadata=metadata,
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._rng.integers(0, self.n_transitions, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
        }

    def all_transitions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.observations.copy(), self.actions.copy()

    def __len__(self) -> int:
        return self.n_transitions


def fit_train_normalization(dataset: FlowBC4ADataset, floor: float = 0.01) -> dict[str, tuple[float, ...]]:
    if dataset.split != "train":
        raise ValueError("normalization statistics must be fit on the train split")
    if floor <= 0:
        raise ValueError("normalization floor must be positive")
    observations, actions = dataset.all_transitions()
    result = {}
    for name, values in (("obs", observations), ("act", actions)):
        flat = values.reshape((values.shape[0], -1)).astype(np.float64)
        result[f"{name}_mean"] = tuple(flat.mean(axis=0).tolist())
        result[f"{name}_scale"] = tuple(np.maximum(flat.std(axis=0), floor).tolist())
    return result


__all__ = ("Episode", "FlowBC4ADataset", "fit_train_normalization")
