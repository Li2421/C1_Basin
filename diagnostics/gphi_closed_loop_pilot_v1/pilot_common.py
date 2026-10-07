"""Shared, fail-closed utilities for the startup-complete G_phi pilot.

The pilot deliberately imports the original Toy Give-Way tree before any
same-named modules from the Basin workspace.  No oracle or eta implementation
is imported here; deployment consists only of FlowBC, two frozen projections,
and the deterministic 214 -> 128 -> 128 -> 4 network.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
TRAINING = ROOT / "diagnostics/gphi_training_startup_complete_v1"
PARETO = ROOT / "diagnostics/gphi_startup_warm_pareto_v1"
V3_DATASET = ROOT / "diagnostics/gphi_training_dataset_v3"

FLOW_CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
STARTUP_WRAPPER = DATASET / "startup_feature_builder.py"
EXACT_PROJECTOR = ROOT / "diagnostics/success_basin_multimodality/exact_projector.py"

FROZEN_SOURCES = {
    SYSROOT / "single_integrator/cbf.py": "841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48",
    SYSROOT / "single_integrator/environment.py": "427b1c0db1d68698e095a95e50a4404bae6c806a0f87351bfac431f6eeefd49b",
    FLOW_CHECKPOINT: "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
    EXACT_PROJECTOR: "e29d510dc1752f138bdfcc491f8f5008dbcd215282301396852bdbfd754ac544",
    STARTUP_WRAPPER: "c15d351a5ac89daa8e51eaf6965498ff58d16bdf4d6e9ef271dfc179afd8bb84",
    ROOT / "diagnostics/gphi_training_dataset_v2/finalize_dataset.py": "5ec2ba5ab447a707022f693719731595a76d92e6352b3f66c19caedf2baf2341",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_frozen_sources() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for path, expected in FROZEN_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        observed[str(path)] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen source mismatch: {path}; expected {expected}, got {actual}"
            )
    # Ensure the authoritative implementation wins all module resolution.
    for path in (ROOT, SYSROOT):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(SYSROOT))
    sys.path.insert(1, str(ROOT))
    return {"status": "PASS", "observed_sha256": observed}


def load_environment_config(dataset_dir: Path = DATASET) -> dict[str, Any]:
    candidates = (dataset_dir / "protocol.json", V3_DATASET / "protocol.json")
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text())
            if "environment" in payload:
                return payload["environment"]
    raise FileNotFoundError("no frozen environment configuration found")


def audit_startup_training_artifacts(
    checkpoint: Path, normalization: Path, *, require_startup_complete: bool,
) -> dict[str, Any]:
    """Bind production inference to a completed, offline-gated training run."""
    expected_checkpoint = (PARETO / "best_balanced_checkpoint.npz").resolve()
    expected_normalization = (TRAINING / "artifacts/normalization.json").resolve()
    checkpoint = checkpoint.resolve(); normalization = normalization.resolve()
    if checkpoint != expected_checkpoint or normalization != expected_normalization:
        if require_startup_complete:
            raise RuntimeError((
                "production must use startup-complete training artifacts",
                str(checkpoint), str(normalization),
            ))
        return {
            "status": "EXTERNAL_CHECKPOINT_SMOKE_ONLY", "checkpoint": str(checkpoint),
            "normalization": str(normalization),
        }
    artifact_root = expected_checkpoint.parent
    sanity_path = artifact_root / "sanity_checks.json"
    manifest_path = artifact_root / "manifest.json"
    selection_path = artifact_root / "selected_checkpoints.json"
    projection_path = artifact_root / "projection_replay_metrics.json"
    base_sanity_path = TRAINING / "artifacts/sanity_checks.json"
    for path in (expected_checkpoint, expected_normalization, sanity_path, manifest_path,
                 selection_path, projection_path, base_sanity_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sanity = json.loads(sanity_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    projection = json.loads(projection_path.read_text())
    base_sanity = json.loads(base_sanity_path.read_text())
    expected_hash = manifest.get("primary_checkpoint_sha256")
    actual_hash = sha256(expected_checkpoint)
    primary = selection.get("global_balanced_primary", {})
    checks = {
        "pareto_manifest_completed": manifest.get("status") == "COMPLETE",
        "pareto_ready_for_pilot": manifest.get("ready_for_full_episode_pilot") is True,
        "training_sanity_passed": sanity.get("passed") is True,
        "base_training_sanity_passed": base_sanity.get("passed") is True,
        "projection_replay_passed": (
            not projection["validation"]["projection_failures"]
            and not projection["test"]["projection_failures"]
        ),
        "architecture_optimizer_loss_unchanged": sanity.get("architecture_optimizer_loss_unchanged") is True,
        "every_epoch_checkpointed": sanity.get("every_epoch_checkpointed") is True,
        "validation_only_selection": sanity.get("selection_validation_only") is True and selection.get("test_used_for_selection") is False,
        "selected_seed_epoch_exact": primary.get("seed") == 23 and primary.get("epoch") == 1171,
        "checkpoint_manifest_hash_matches": expected_hash == actual_hash,
    }
    if not all(checks.values()):
        raise RuntimeError(("startup-complete training audit failed", checks))
    return {
        "status": "PASS", "checks": checks, "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": actual_hash, "normalization": str(expected_normalization),
        "sanity_checks_sha256": sha256(sanity_path), "training_manifest_sha256": sha256(manifest_path),
        "selection_audit_sha256": sha256(selection_path),
        "projection_replay_sha256": sha256(projection_path),
        "selected_seed": primary.get("seed"), "selected_epoch": primary.get("epoch"),
        "training_initialization": "fresh random initialization per registered seed; no V3 checkpoint initialization in pipeline",
    }


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


class DeterministicGphi:
    """Numpy inference for the frozen checkpoint serialization contract."""

    def __init__(self, checkpoint: Path, normalization_json: Path | None = None) -> None:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as values:
            required = {
                "architecture_json", "normalization_mean", "normalization_scale",
                "normalization_binary_mask", "layer_0_weight", "layer_0_bias",
                "layer_1_weight", "layer_1_bias", "layer_2_weight", "layer_2_bias",
            }
            missing = required - set(values.files)
            if missing:
                raise RuntimeError(("checkpoint fields missing", sorted(missing)))
            self.architecture = json.loads(str(values["architecture_json"].item()))
            self.mean = np.asarray(values["normalization_mean"], dtype=np.float64)
            self.scale = np.asarray(values["normalization_scale"], dtype=np.float64)
            self.binary = np.asarray(values["normalization_binary_mask"], dtype=bool)
            self.layers = [
                (
                    np.asarray(values[f"layer_{index}_weight"], dtype=np.float32),
                    np.asarray(values[f"layer_{index}_bias"], dtype=np.float32),
                )
                for index in range(3)
            ]
        shapes = [(weight.shape, bias.shape) for weight, bias in self.layers]
        expected = [((214, 128), (128,)), ((128, 128), (128,)), ((128, 4), (4,))]
        if shapes != expected:
            raise RuntimeError(("architecture mismatch", shapes, expected))
        if self.mean.shape != (214,) or self.scale.shape != (214,) or self.binary.shape != (214,):
            raise RuntimeError("normalization shape mismatch")
        if np.any(self.scale <= 0) or not all(
            np.isfinite(value).all()
            for value in (self.mean, self.scale, *(item for layer in self.layers for item in layer))
        ):
            raise RuntimeError("checkpoint contains invalid/nonfinite values")
        if normalization_json is not None and normalization_json.exists():
            external = json.loads(normalization_json.read_text())
            external_mean = np.asarray(external["mean"], dtype=np.float64)
            external_scale = np.asarray(external["scale"], dtype=np.float64)
            if not np.array_equal(external_mean, self.mean) or not np.array_equal(external_scale, self.scale):
                raise RuntimeError("checkpoint and normalization.json disagree")
        self.path = checkpoint
        self.sha256 = sha256(checkpoint)

    @staticmethod
    def _silu(values: np.ndarray) -> np.ndarray:
        return values * stable_sigmoid(values)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[-1] != 214 or not np.isfinite(features).all():
            raise ValueError(("invalid G_phi feature", features.shape))
        return ((features - self.mean) / self.scale).astype(np.float32)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        values = self.normalize(features)
        for index, (weight, bias) in enumerate(self.layers):
            values = values @ weight + bias
            if index < 2:
                values = self._silu(values)
        result = np.asarray(values, dtype=np.float64)
        if result.shape[-1] != 4 or not np.isfinite(result).all():
            raise RuntimeError(("invalid G_phi output", result.shape))
        return result

    def audit(self) -> dict[str, Any]:
        parameter_count = int(sum(weight.size + bias.size for weight, bias in self.layers))
        return {
            "status": "PASS", "checkpoint": str(self.path), "checkpoint_sha256": self.sha256,
            "architecture": [214, 128, 128, 4], "activation": "SiLU",
            "parameter_count": parameter_count, "normalization_dimension": len(self.mean),
            "random_initialization_provenance": self.architecture,
        }


class TrainingReference:
    """Nearest normalized train-state-centroid OOD diagnostic."""

    def __init__(self, samples_path: Path, model: DeterministicGphi) -> None:
        if not samples_path.is_file():
            raise FileNotFoundError(samples_path)
        with np.load(samples_path, allow_pickle=False) as values:
            features = np.asarray(values["features"], dtype=np.float64)
            state_ids = np.asarray(values["state_id"]).astype(str)
            splits = np.asarray(values["split"]).astype(str)
        train = splits == "train"
        if not np.any(train):
            raise RuntimeError("training dataset contains no train samples")
        normalized = model.normalize(features[train])
        train_ids = state_ids[train]
        unique_ids = np.unique(train_ids)
        self.centroids = np.stack(
            [normalized[train_ids == state_id].mean(axis=0) for state_id in unique_ids]
        ).astype(np.float32)
        self.state_ids = unique_ids
        lookup = {state_id: index for index, state_id in enumerate(unique_ids)}
        own = np.asarray([lookup[state_id] for state_id in train_ids], dtype=np.int64)
        own_distance = np.linalg.norm(normalized - self.centroids[own], axis=1) / np.sqrt(214.0)
        self.reference_p95 = float(np.quantile(own_distance, 0.95))
        self.samples_path = samples_path.resolve()
        self.samples_sha256 = sha256(samples_path)

    def distances(self, normalized_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(normalized_features, dtype=np.float32)
        q2 = np.sum(query * query, axis=1, keepdims=True)
        c2 = np.sum(self.centroids * self.centroids, axis=1)[None, :]
        squared = np.maximum(0.0, q2 + c2 - 2.0 * (query @ self.centroids.T))
        nearest = np.argmin(squared, axis=1)
        return np.sqrt(squared[np.arange(len(query)), nearest]) / np.sqrt(214.0), nearest

    def audit(self) -> dict[str, Any]:
        return {
            "metric": "RMS Euclidean distance in checkpoint-normalized 214-D space to nearest train-state centroid",
            "reference_threshold": "P95 train-sample distance to its own state centroid",
            "reference_p95": self.reference_p95,
            "train_state_centroids": len(self.centroids),
            "samples_path": str(self.samples_path), "samples_sha256": self.samples_sha256,
        }


def contiguous_true_run_lengths(mask: Iterable[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            runs.append(current); current = 0
    if current:
        runs.append(current)
    return runs
