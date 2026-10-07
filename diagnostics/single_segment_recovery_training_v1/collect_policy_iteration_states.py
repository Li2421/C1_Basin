"""Collect one bounded on-policy state batch from the selected ETA hierarchy.

All 40 frozen TRAIN sources are rolled out once.  Only their two previously
frozen, outcome-blind absolute anchor requests are inspected.  At an anchor,
the exact pre-action state is eligible for the entry dataset iff the current
mode is SAFETY_BEFORE, or for the exit dataset iff it is RECOVERY.  Selection
is a deterministic state-ID hash, one state per source, with no terminal
outcome, counterfactual result, or replacement sampling involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SOURCE_MANIFEST = HERE / "source_split_manifest.json"
SELECTED_POLICY = HERE / "full_loop_manifests/calibration/selected_eta.json"
OUTPUT_ENTRY = HERE / "policy_iteration_entry_state_manifest.json"
OUTPUT_EXIT = HERE / "policy_iteration_exit_state_manifest.json"
OUTPUT_INTEGRITY = HERE / "policy_iteration_state_collection_integrity.json"
OUTPUT_RUNTIME = HERE / "runs/policy_iteration_state_collection_runtime.json"
MAX_PER_KIND = 12

for value in (str(SYSROOT), str(ROOT), str(PILOT), str(HERE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(HERE)]

from pilot_common import sha256  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from full_state_io import restore_full_history  # noqa: E402
from run_full_episode_system import (  # noqa: E402
    ThresholdedFrozenHead, canonical_hash, classify, load_and_verify_assets,
    verify_semantic_hash,
)
from single_integrator.environment import GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from state_machine import (  # noqa: E402
    AuthoritativeStepKernel, Mode, PreparedStep, RecoverySystem,
    SingleSegmentRecoveryMachine,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def snapshot_environment(env: GiveWayEnv) -> dict[str, np.ndarray]:
    history = np.asarray(env.distance_history, dtype=np.float64)
    if bool(env.done) or history.shape != (int(env.step_count) + 1, 2) or not np.isfinite(history).all():
        raise RuntimeError(("invalid decision snapshot", env.step_count, env.done, history.shape))
    def optional(value: Any) -> np.ndarray:
        return np.asarray(-1 if value is None else value)
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "positions": np.asarray(env.positions, dtype=np.float64).copy(),
        "velocities": np.asarray(env.velocities, dtype=np.float64).copy(),
        "step": np.asarray(env.step_count, dtype=np.int64),
        "error_history": history.copy(),
        "history_start_step": np.asarray(0, dtype=np.int64),
        "candidate_since": optional(env.candidate_since),
        "stuck_timer": np.asarray(env.stuck_timer, dtype=np.float64),
        "max_stuck_timer": np.asarray(env.max_stuck_timer, dtype=np.float64),
        "ever_candidate_deadlock": np.asarray(env.ever_candidate_deadlock),
        "first_success_step": optional(env.first_success_step),
        "first_deadlock_step": optional(env.first_deadlock_step),
        "first_wall_collision_step": optional(env.first_wall_collision_step),
        "first_agent_collision_step": optional(env.first_agent_collision_step),
        "done": np.asarray(env.done),
    }


class ObservedKernel(AuthoritativeStepKernel):
    """Expose the already-prepared context without another Flow draw."""

    def __init__(self, *args: Any, observer: Callable[[PreparedStep], None], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.observer = observer

    def prepare(self, *, need_feature: bool) -> PreparedStep:
        context = super().prepare(need_feature=need_feature)
        self.observer(context)
        return context


def selection_rank(kind: str, source_id: str, slot: int, step: int) -> str:
    return hashlib.sha256(
        f"single_segment_policy_iteration_state_v1|{kind}|{source_id}|{slot}|{step}".encode()
    ).hexdigest()


def capture_candidate(kind: str, source: dict, slot: int, context: PreparedStep,
                      env: GiveWayEnv, machine: SingleSegmentRecoveryMachine,
                      policy_hash: str) -> dict[str, Any]:
    memory = machine.memory
    if kind == "entry" and memory.mode is not Mode.SAFETY_BEFORE:
        raise AssertionError(memory.mode)
    if kind == "exit":
        if memory.mode is not Mode.RECOVERY or memory.eta_latched is None or memory.recovery_transitions < 1:
            raise AssertionError((memory.mode, memory.eta_latched, memory.recovery_transitions))
    feature = np.asarray(context.feature, dtype=np.float64)
    if feature.shape != (214,) or not np.isfinite(feature).all():
        raise RuntimeError((source["source_id"], context.step, "invalid feature"))
    key = np.asarray(context.flow_key, dtype=np.uint32)
    state_id = (
        f"policy_iteration_{kind}_eta_{source['source_id']}_a{slot}_t{int(context.step):04d}"
    )
    return {
        "kind": kind, "state_id": state_id, "root_source_id": source["source_id"],
        "split": "train", "episode_index": int(source["episode_index"]),
        "rollout_id": int(source["rollout_id"]), "anchor_slot": int(slot),
        "requested_global_step": int(context.step), "absolute_step": int(context.step),
        "selection_rank": selection_rank(kind, source["source_id"], slot, int(context.step)),
        "source_state": snapshot_environment(env),
        "feature": feature.copy(), "observation": np.asarray(context.observation, dtype=np.float64).copy(),
        "current_u_flow": np.asarray(context.u_flow, dtype=np.float64).copy(),
        "current_u_safe": np.asarray(context.u_safe, dtype=np.float64).copy(),
        "current_flow_key_data": key.copy(),
        "current_flow_realization_id": (
            f"root={int(source['flow_root_seed'])}|rollout={int(source['rollout_id'])}"
            f"|absolute_step={int(context.step)}|key={key.tolist()}"
        ),
        "flow_root_seed": int(source["flow_root_seed"]),
        "mode": memory.mode.value, "entry_step": memory.entry_step,
        "exit_step": memory.exit_step, "recovery_used": bool(memory.recovery_used),
        "recovery_transitions": int(memory.recovery_transitions),
        "eta_latched": (
            None if memory.eta_latched is None
            else np.asarray(memory.eta_latched, dtype=np.float64).copy()
        ),
        "collection_policy_hash": policy_hash,
    }


def select_bounded(candidates: list[dict[str, Any]], maximum: int = MAX_PER_KIND) -> list[dict[str, Any]]:
    """Outcome-blind, source-diverse deterministic selection."""
    best_by_source: dict[str, dict[str, Any]] = {}
    for row in candidates:
        source = row["root_source_id"]
        if source not in best_by_source or row["selection_rank"] < best_by_source[source]["selection_rank"]:
            best_by_source[source] = row
    return sorted(best_by_source.values(), key=lambda row: row["selection_rank"])[:maximum]


def save_selected(kind: str, selected: list[dict[str, Any]], candidates: list[dict[str, Any]],
                  source_manifest: dict, policy_manifest: dict, config: Any) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for candidate in selected:
        state_id = candidate["state_id"]
        state_path = HERE / "states/policy_iteration" / kind / "train" / f"{state_id}.npz"
        input_path = HERE / "decision_inputs/policy_iteration" / kind / "train" / f"{state_id}.npz"
        if state_path.exists() or input_path.exists():
            raise RuntimeError((state_id, "refusing to overwrite policy-iteration state"))
        atomic_npz(state_path, candidate["source_state"])
        restored = restore_full_history(state_path, config)
        source_state = candidate["source_state"]
        diffs = {
            "positions": float(np.max(np.abs(restored.positions - source_state["positions"]))),
            "velocities": float(np.max(np.abs(restored.velocities - source_state["velocities"]))),
            "history": float(np.max(np.abs(
                np.asarray(restored.distance_history) - source_state["error_history"]
            ))),
        }
        if max(diffs.values()) != 0.0 or restored.step_count != candidate["absolute_step"]:
            raise RuntimeError((state_id, "full-state round trip mismatch", diffs))
        atomic_npz(input_path, {
            "feature": candidate["feature"], "observation": candidate["observation"],
            "u_flow": candidate["current_u_flow"], "u_safe": candidate["current_u_safe"],
            "flow_step_key": candidate["current_flow_key_data"],
            "requested_global_step": np.asarray(candidate["absolute_step"], dtype=np.int64),
        })
        eta = candidate["eta_latched"]
        row = {
            "schema": "single_segment_policy_iteration_decision_state_v1",
            "state_id": state_id, "root_source_id": candidate["root_source_id"],
            "split": "train", "state_file": str(state_path), "state_sha256": sha256(state_path),
            "decision_input_file": str(input_path), "decision_input_sha256": sha256(input_path),
            "absolute_step": candidate["absolute_step"], "feature": candidate["feature"].tolist(),
            "current_u_flow": candidate["current_u_flow"].tolist(),
            "current_u_safe": candidate["current_u_safe"].tolist(),
            "current_flow_key_data": candidate["current_flow_key_data"].astype(np.uint32).tolist(),
            "current_flow_realization_id": candidate["current_flow_realization_id"],
            "flow_root_seed": candidate["flow_root_seed"], "mode": candidate["mode"],
            "entry_step": candidate["entry_step"], "exit_step": candidate["exit_step"],
            "recovery_used": candidate["recovery_used"],
            "recovery_transitions": candidate["recovery_transitions"],
            "eta_latched": None if eta is None else eta.tolist(),
            "eta_latched_hash": None if eta is None else canonical_hash(eta.tolist()),
            "selection_rank": candidate["selection_rank"],
            "selection_outcome_blind": True, "selection_source_diverse": True,
            "complete_real_monitor_history": True, "history_start_step": 0,
            "round_trip_max_abs_difference": max(diffs.values()),
            "collection_policy_hash": candidate["collection_policy_hash"],
        }
        states.append(row)
    manifest = {
        "schema": "single_segment_policy_iteration_state_manifest_v1",
        "status": "COMPLETE_FROZEN", "decision_kind": kind, "system": "ETA",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "train", "source_count_rolled_out": 40,
        "source_manifest_path": str(SOURCE_MANIFEST), "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "collection_policy_manifest_path": str(SELECTED_POLICY),
        "collection_policy_manifest_sha256": sha256(SELECTED_POLICY),
        "collection_policy_manifest_content_sha256": policy_manifest["content_sha256"],
        "collection_policy_hash": policy_manifest["policy_hash"],
        "anchor_rule": source_manifest["anchor_rule"],
        "selection": {
            "maximum_states": MAX_PER_KIND, "eligible_state_count": len(candidates),
            "eligible_unique_source_count": len({row["root_source_id"] for row in candidates}),
            "selected_state_count": len(states),
            "method": "best deterministic hash-ranked eligible anchor per source, then first 12 sources",
            "uses_terminal_outcome": False, "uses_counterfactual_outcome": False,
            "replaces_unavailable_or_wrong_mode_anchors": False,
        },
        "states": states,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    for path in (OUTPUT_ENTRY, OUTPUT_EXIT, OUTPUT_INTEGRITY, OUTPUT_RUNTIME):
        if path.exists():
            raise RuntimeError((path, "refusing to overwrite policy-iteration collection"))
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    verify_semantic_hash(source_manifest, SOURCE_MANIFEST)
    policy_manifest = json.loads(SELECTED_POLICY.read_text())
    verify_semantic_hash(policy_manifest, SELECTED_POLICY)
    if (policy_manifest.get("system") != "ETA" or policy_manifest.get("policy_hash")
            != "3cecf01735c76c36bc71b822f3a233fa760d97a8b2926906224b14cb18bbf761"):
        raise RuntimeError("selected ETA hierarchy mismatch")
    hashes, _, config, cbf = load_and_verify_assets(policy_manifest)
    entry_head = ThresholdedFrozenHead(policy_manifest["policy"]["entry_head"], 214)
    exit_head = ThresholdedFrozenHead(policy_manifest["policy"]["exit_head"], 217)
    eta_model = FixedDEtaPredictor(Path(hashes["structured_eta_recovery"]["path"]))
    flow_policy, provenance = load_policy(Path(hashes["flowbc"]["path"]))
    reference = json.loads(
        (ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json").read_text()
    )
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: flow_policy.sample_actions(observation[None], seed=key)[0])
    sources = list(source_manifest["sources"]["train"])
    if len(sources) != 40 or any(row["split"] != "train" for row in sources):
        raise RuntimeError("expected exactly frozen TRAIN40")
    entry_candidates: list[dict[str, Any]] = []
    exit_candidates: list[dict[str, Any]] = []
    anchor_observations: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    started_utc = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    total_steps = 0

    for source in sources:
        env = GiveWayEnv(config)
        env.reset(np.asarray(source["initial_positions"], dtype=np.float64))
        requests = {int(step): slot for slot, step in enumerate(source["requested_anchor_steps"])}
        seen: dict[int, str] = {}
        machine_box: dict[str, SingleSegmentRecoveryMachine] = {}

        def observe(context: PreparedStep, *, source: dict = source) -> None:
            if int(context.step) not in requests:
                return
            machine = machine_box["machine"]
            slot = int(requests[int(context.step)])
            mode = machine.memory.mode
            eligible_kind = None
            if mode is Mode.SAFETY_BEFORE:
                eligible_kind = "entry"
            elif mode is Mode.RECOVERY:
                eligible_kind = "exit"
            seen[int(context.step)] = mode.value
            anchor_observations.append({
                "root_source_id": source["source_id"], "anchor_slot": slot,
                "absolute_step": int(context.step), "mode": mode.value,
                "eligible_kind": eligible_kind,
            })
            if eligible_kind is not None:
                candidate = capture_candidate(
                    eligible_kind, source, slot, context, env, machine,
                    policy_manifest["policy_hash"],
                )
                (entry_candidates if eligible_kind == "entry" else exit_candidates).append(candidate)

        episode_key = jax.random.fold_in(
            jax.random.PRNGKey(np.uint32(source["flow_root_seed"])), int(source["rollout_id"])
        )
        kernel = ObservedKernel(
            env=env, config=config, cbf=cbf, episode_key=episode_key,
            sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
            project=project_velocity_with_retry, observer=observe,
        )
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
            entry_head=entry_head, exit_head=exit_head, eta_model=eta_model,
        )
        machine_box["machine"] = machine
        error = None
        try:
            machine.run_to_terminal()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
        if error is not None:
            raise RuntimeError((source["source_id"], "policy-iteration source failed", error))
        total_steps += int(kernel.physical_transition_count)
        for step, slot in requests.items():
            if step not in seen:
                anchor_observations.append({
                    "root_source_id": source["source_id"], "anchor_slot": int(slot),
                    "absolute_step": int(step), "mode": "UNAVAILABLE_TERMINAL_BEFORE_ANCHOR",
                    "eligible_kind": None,
                })
        episode_rows.append({
            "root_source_id": source["source_id"], "outcome": classify(env, None),
            "terminal_step": int(env.step_count), "entry_step": machine.memory.entry_step,
            "exit_step": machine.memory.exit_step,
            "recovery_transitions": int(machine.memory.recovery_transitions),
            "flow_sample_count": int(kernel.flow_sample_count),
            "physical_transition_count": int(kernel.physical_transition_count),
        })

    entry_selected = select_bounded(entry_candidates)
    exit_selected = select_bounded(exit_candidates)
    entry_manifest = save_selected(
        "entry", entry_selected, entry_candidates, source_manifest, policy_manifest, config
    )
    exit_manifest = save_selected(
        "exit", exit_selected, exit_candidates, source_manifest, policy_manifest, config
    )
    atomic_json(OUTPUT_ENTRY, entry_manifest)
    atomic_json(OUTPUT_EXIT, exit_manifest)
    runtime = {
        "schema": "single_segment_policy_iteration_state_runtime_v1",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - began, "device": args.device,
        "jax_devices": [str(item) for item in jax.devices()],
        "new_continuations": len(sources), "new_physical_steps": total_steps,
        "cpu_thread_limit": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "episode_rows": episode_rows,
    }
    atomic_json(OUTPUT_RUNTIME, runtime)
    integrity = {
        "schema": "single_segment_policy_iteration_state_collection_integrity_v1",
        "status": "PASS",
        "selected_policy_hash": policy_manifest["policy_hash"],
        "entry_checkpoint_sha256": policy_manifest["policy"]["entry_head"]["sha256"],
        "exit_checkpoint_sha256": policy_manifest["policy"]["exit_head"]["sha256"],
        "entry_threshold_probability": policy_manifest["policy"]["entry_head"]["threshold_probability"],
        "exit_threshold_probability": policy_manifest["policy"]["exit_head"]["threshold_probability"],
        "train_sources_rolled_out": len(sources), "anchor_requests": len(anchor_observations),
        "anchor_mode_counts": {
            mode: sum(row["mode"] == mode for row in anchor_observations)
            for mode in sorted({row["mode"] for row in anchor_observations})
        },
        "eligible_entry_states": len(entry_candidates), "eligible_exit_states": len(exit_candidates),
        "selected_entry_states": len(entry_selected), "selected_exit_states": len(exit_selected),
        "entry_unique_sources": len({row["root_source_id"] for row in entry_selected}),
        "exit_unique_sources": len({row["root_source_id"] for row in exit_selected}),
        "outcome_used_for_selection": False, "replacement_sampling": False,
        "entry_manifest": {"path": str(OUTPUT_ENTRY), "sha256": sha256(OUTPUT_ENTRY),
                           "content_sha256": entry_manifest["content_sha256"]},
        "exit_manifest": {"path": str(OUTPUT_EXIT), "sha256": sha256(OUTPUT_EXIT),
                          "content_sha256": exit_manifest["content_sha256"]},
        "budget_use": {"new_continuations": len(sources), "new_physical_steps": total_steps},
    }
    integrity["content_sha256"] = canonical_hash(integrity)
    atomic_json(OUTPUT_INTEGRITY, integrity)
    print(json.dumps({key: value for key, value in integrity.items() if key != "content_sha256"}, indent=2))


if __name__ == "__main__":
    main()
