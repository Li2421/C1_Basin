"""Freeze one full-episode, one-segment policy evaluation manifest.

This builder performs no rollout.  Validation threshold pairs are explicit
members of the predeclared {0.25, 0.50, 0.75} grid and are incorporated into
the immutable policy hash.  Calibration/final evaluation therefore consumes
one already-frozen pair rather than selecting a threshold online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_paired_branch_manifest import (
    MAX_CONTINUATIONS, MAX_STEPS, canonical_hash, existing_budget,
    sha256, source_collection_budget, verify_semantic_hash,
)


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1")
SOURCE_MANIFEST = HERE / "source_split_manifest.json"
FINAL_MANIFEST = HERE / "final_test_manifest.json"
HASHES = HERE / "controller_and_projection_hashes.json"
MANIFEST_DIR = HERE / "full_loop_manifests"
STATE_MACHINE = HERE / "state_machine.py"
FULL_LOOP_RUNNER = HERE / "run_full_episode_system.py"
HORIZON = 850
THRESHOLDS = (0.25, 0.50, 0.75)
SYSTEM_DIMENSIONS = {"G": (214, 214), "ETA": (214, 217)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def inspect_head(path: Path, expected_dimension: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "normalization_mean", "normalization_scale",
            "layer_0_weight", "layer_0_bias", "layer_1_weight", "layer_1_bias",
            "layer_2_weight", "layer_2_bias", "threshold_logit",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError((path, "not a frozen decision head", missing))
        mean = np.asarray(archive["normalization_mean"], dtype=np.float64)
        scale = np.asarray(archive["normalization_scale"], dtype=np.float64)
        weights = [np.asarray(archive[f"layer_{i}_weight"], dtype=np.float64) for i in range(3)]
        biases = [np.asarray(archive[f"layer_{i}_bias"], dtype=np.float64) for i in range(3)]
        checkpoint_threshold = float(np.asarray(archive["threshold_logit"]).item())
        shapes = [item.shape for item in weights]
        expected_shapes = [(expected_dimension, 64), (64, 64), (64, 1)]
        expected_biases = [(64,), (64,), (1,)]
        if (mean.shape != (expected_dimension,) or scale.shape != mean.shape
                or shapes != expected_shapes or [item.shape for item in biases] != expected_biases):
            raise RuntimeError((path, "head architecture mismatch", mean.shape, scale.shape,
                                shapes, [item.shape for item in biases]))
        if np.any(scale <= 0) or not np.isfinite(checkpoint_threshold) or not all(
            np.isfinite(item).all() for item in (mean, scale, *weights, *biases)
        ):
            raise RuntimeError((path, "head contains invalid/nonfinite values"))
    return {"path": str(path.resolve()), "sha256": sha256(path), "input_dimension": expected_dimension}


def _load_sources(split: str, *, allow_final_test: bool) -> tuple[dict, list[dict], dict | None]:
    source = json.loads(SOURCE_MANIFEST.read_text())
    verify_semantic_hash(source, SOURCE_MANIFEST)
    if source.get("status") != "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION":
        raise RuntimeError("source split manifest is not frozen")
    if split not in {"train", "validation", "calibration", "final_test"}:
        raise ValueError(split)
    if split == "final_test" and not allow_final_test:
        raise RuntimeError("final test is locked; explicit authorization is required")
    rows = [dict(row) for row in source["sources"][split]]
    final = None
    if split == "final_test":
        final = json.loads(FINAL_MANIFEST.read_text())
        verify_semantic_hash(final, FINAL_MANIFEST)
        if (final.get("status") != "RESERVED_UNTOUCHED_BEFORE_MODEL_SELECTION_AND_CALIBRATION"
                or final.get("outcomes_observed") is not False):
            raise RuntimeError("final-test manifest is no longer pristine")
        if final.get("source_split_manifest_sha256") != sha256(SOURCE_MANIFEST):
            raise RuntimeError("final/source manifest mismatch")
        if final.get("episodes") != rows:
            raise RuntimeError("final episode rows differ from the frozen source split")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate full-loop source")
    if any(row["split"] != split for row in rows):
        raise RuntimeError("mislabeled full-loop source split")
    return source, rows, final


def _existing_full_loop_budget(exclude: Path) -> tuple[int, int, list[str]]:
    continuations = steps = 0
    assets: list[str] = []
    if not MANIFEST_DIR.exists():
        return continuations, steps, assets
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        if path.resolve() == exclude.resolve():
            continue
        payload = json.loads(path.read_text())
        verify_semantic_hash(payload, path)
        if payload.get("schema") != "single_segment_full_loop_evaluation_manifest_v1":
            continue
        continuations += int(payload["budget"]["new_continuations"])
        steps += int(payload["budget"]["maximum_physical_steps"])
        assets.append(str(path))
    return continuations, steps, assets


def build_manifest(
    *, system: str, split: str, entry_checkpoint: Path, exit_checkpoint: Path,
    entry_threshold_probability: float, exit_threshold_probability: float,
    output: Path, allow_final_test: bool = False, final_gate: Path | None = None,
) -> dict[str, Any]:
    system = str(system).upper()
    if system not in SYSTEM_DIMENSIONS:
        raise ValueError(system)
    if output.exists():
        raise RuntimeError((output, "refusing to overwrite frozen full-loop manifest"))
    for value in (entry_threshold_probability, exit_threshold_probability):
        if float(value) not in THRESHOLDS:
            raise ValueError((value, "threshold is outside the predeclared grid", THRESHOLDS))
    source, rows, final = _load_sources(split, allow_final_test=allow_final_test)
    entry_dim, exit_dim = SYSTEM_DIMENSIONS[system]
    entry = inspect_head(entry_checkpoint, entry_dim)
    exit_record = inspect_head(exit_checkpoint, exit_dim)
    entry["threshold_probability"] = float(entry_threshold_probability)
    entry["threshold_logit"] = float(math.log(entry_threshold_probability / (1-entry_threshold_probability)))
    exit_record["threshold_probability"] = float(exit_threshold_probability)
    exit_record["threshold_logit"] = float(math.log(exit_threshold_probability / (1-exit_threshold_probability)))
    policy = {
        "system": system,
        "entry_head": entry,
        "exit_head": exit_record,
        "one_segment_only": True,
        "safety_before_and_after": True,
        "threshold_grid": list(THRESHOLDS),
    }
    policy_hash = canonical_hash(policy)
    final_gate_record = None
    if split == "final_test":
        if final_gate is None or not final_gate.is_file():
            raise RuntimeError("final test requires a frozen validation+calibration gate artifact")
        gate = json.loads(final_gate.read_text())
        verify_semantic_hash(gate, final_gate)
        required_gate = {
            "schema": "single_segment_final_test_gate_v1",
            "status": "FROZEN_APPROVED_FOR_FINAL_TEST",
            "selected_policy_hash": policy_hash,
            "validation_complete": True,
            "calibration_complete": True,
            "test_data_used": False,
            "final_test_unopened": True,
        }
        for key, expected in required_gate.items():
            if gate.get(key) != expected:
                raise RuntimeError((final_gate, "invalid final-test gate", key, gate.get(key), expected))
        final_gate_record = {
            "path": str(final_gate.resolve()), "sha256": sha256(final_gate),
            "content_sha256": gate["content_sha256"],
        }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for prior_path in sorted(MANIFEST_DIR.glob("*.json")):
        prior = json.loads(prior_path.read_text())
        verify_semantic_hash(prior, prior_path)
        if (prior.get("schema") == "single_segment_full_loop_evaluation_manifest_v1"
                and prior.get("split") == split and prior.get("policy_hash") == policy_hash):
            raise RuntimeError(("duplicate policy/split manifest", prior_path))

    # Every source needs one learned rollout.  Development Safety is reused;
    # final-test Safety must be materialized once and cached for later policies.
    safety_new = len(rows) if split == "final_test" else 0
    new_continuations = len(rows) + safety_new
    maximum_steps = new_continuations * HORIZON
    source_continuations, source_steps, source_assets = source_collection_budget()
    branch_continuations, branch_steps, branch_assets = existing_budget(Path("/__no_branch_exclusion__"))
    full_continuations, full_steps, full_assets = _existing_full_loop_budget(output)
    total_continuations = source_continuations + branch_continuations + full_continuations + new_continuations
    total_steps = source_steps + branch_steps + full_steps + maximum_steps
    if total_continuations > MAX_CONTINUATIONS or total_steps > MAX_STEPS:
        raise RuntimeError(("global experiment budget exceeded", total_continuations, total_steps))
    tasks = [{
        "task_index": index,
        "source_id": row["source_id"],
        "root_source_id": row["source_id"],
        "split": split,
        "episode_index": int(row["episode_index"]),
        "rollout_id": int(row["rollout_id"]),
        "flow_root_seed": int(row["flow_root_seed"]),
        "initial_positions": row["initial_positions"],
    } for index, row in enumerate(rows)]
    frozen_hashes = json.loads(HASHES.read_text())
    if (frozen_hashes.get("schema") != "single_segment_recovery_frozen_hashes_v1"
            or frozen_hashes.get("status") != "PASS"):
        raise RuntimeError("controller/projection integrity audit is not PASS")
    required_assets = {
        "flowbc", "environment", "projection_constraints", "projection_retry",
        "event_priority_and_monitor", "startup_feature_builder", "eta_basis",
        "direct_g_recovery", "structured_eta_recovery",
    }
    if not required_assets.issubset(frozen_hashes):
        raise RuntimeError(("controller/projection integrity audit is incomplete",
                            sorted(required_assets - set(frozen_hashes))))
    for name in sorted(required_assets):
        record = frozen_hashes[name]
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError((name, "frozen asset mismatch during manifest construction"))
    result = {
        "schema": "single_segment_full_loop_evaluation_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "system": system,
        "policy": policy,
        "policy_hash": policy_hash,
        "source_manifest_path": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_manifest_content_sha256": source["content_sha256"],
        "final_test_manifest_path": str(FINAL_MANIFEST) if final is not None else None,
        "final_test_manifest_sha256": sha256(FINAL_MANIFEST) if final is not None else None,
        "final_test_gate": final_gate_record,
        "controller_hash_manifest_path": str(HASHES),
        "controller_hash_manifest_sha256": sha256(HASHES),
        "implementation_hashes": {
            "state_machine": {"path": str(STATE_MACHINE), "sha256": sha256(STATE_MACHINE)},
            "full_loop_runner": {"path": str(FULL_LOOP_RUNNER), "sha256": sha256(FULL_LOOP_RUNNER)},
        },
        "threshold_pair_selected_by_this_run": False,
        "threshold_pair_is_predeclared_validation_candidate": split == "validation",
        "final_test_explicitly_unlocked": bool(split == "final_test" and allow_final_test),
        "matched_flow_semantics": "episode_key=fold_in(PRNGKey(flow_root_seed),rollout_id); step_key=fold_in(episode_key,absolute_step)",
        "safety_baseline": {
            "development_splits": "reuse frozen collect_safety_sources result",
            "final_test": "materialize once then exact cache reuse",
        },
        "source_count": len(rows),
        "tasks": tasks,
        "budget": {
            "new_continuations": new_continuations,
            "new_learned_continuations": len(rows),
            "new_safety_continuations_upper_bound": safety_new,
            "maximum_physical_steps": maximum_steps,
            "global_after_continuations": total_continuations,
            "global_after_maximum_steps": total_steps,
            "global_limits": {"continuations": MAX_CONTINUATIONS, "physical_steps": MAX_STEPS},
            "source_collection_assets": source_assets,
            "branch_manifest_assets": branch_assets,
            "prior_full_loop_manifests": full_assets,
        },
    }
    result["content_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("G", "ETA"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "final_test"), required=True)
    parser.add_argument("--entry-checkpoint", type=Path, required=True)
    parser.add_argument("--exit-checkpoint", type=Path, required=True)
    parser.add_argument("--entry-threshold", type=float, required=True)
    parser.add_argument("--exit-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--final-gate", type=Path)
    args = parser.parse_args()
    payload = build_manifest(
        system=args.system, split=args.split,
        entry_checkpoint=args.entry_checkpoint, exit_checkpoint=args.exit_checkpoint,
        entry_threshold_probability=args.entry_threshold,
        exit_threshold_probability=args.exit_threshold,
        output=args.output, allow_final_test=args.allow_final_test,
        final_gate=args.final_gate,
    )
    atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
