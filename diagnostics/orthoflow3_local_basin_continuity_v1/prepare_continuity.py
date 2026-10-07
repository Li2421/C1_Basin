"""Freeze real near-trajectory states and the common OrthoFlow3 probe cloud.

This preparation step intentionally reads no eta labels until anchors have been
selected from the already-uniform 32-state OrthoFlow3 migration subset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_local_basin_continuity_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
MIGRATION = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
STARTUP = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
V2 = ROOT / "diagnostics/gphi_training_dataset_v2"
ETA_CHECKPOINT = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
OFFSETS = (-4, -1, 1, 4)
ANCHOR_COUNT = 12
ANCHOR_PERMUTATION_SEED = 2026092603
PROBE_PERMUTATION_SEED = 2026092703
SCREEN_SEEDS_PER_STATE_ETA = 8

sys.path[:0] = [str(SYSROOT), str(ROOT)]
from diagnostics.gphi_training_dataset_v2.build_states import restore_full, save_full  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import (  # noqa: E402
    StartupAwareFeatureBuilder,
)
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError((path, "no rows"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def actions_for(row: dict) -> tuple[GiveWayEnv, np.ndarray, np.ndarray, np.ndarray]:
    """Return the true pre-source environment and original stepwise actions."""
    source = Path(row["source_path"])
    if not source.is_absolute():
        for base in (ROOT / "diagnostics/gphi_training_dataset_v3", STARTUP):
            candidate = base / source
            if candidate.exists():
                source = candidate
                break
    with np.load(source, allow_pickle=False) as data:
        source_type = str(row["source_type"])
        if source_type in {"baseline", "frozen_safety_baseline_episode"}:
            env = GiveWayEnv(CONFIG)
            env.reset(np.asarray(data["initial_positions"], dtype=np.float64))
            return (
                env,
                np.asarray(data["executed_velocity"], dtype=np.float64),
                np.asarray(data["nominal_velocity"], dtype=np.float64),
                np.asarray(data["u_safe"], dtype=np.float64),
            )
        if source_type == "qualification":
            env = GiveWayEnv(CONFIG)
            env.reset(np.asarray(data["positions_before"][0], dtype=np.float64))
            return (
                env,
                np.asarray(data["u_exec"], dtype=np.float64),
                np.asarray(data["u_flow"], dtype=np.float64),
                np.asarray(data["u_safe"], dtype=np.float64),
            )
        if source_type in {"recovery", "v3_corrected_recovery"}:
            anchor_id = str(row["anchor_state"])
            anchor = V2_STATES.get(anchor_id)
            if anchor is None:
                raise RuntimeError((row["state_id"], "missing v2 recovery anchor", anchor_id))
            env = restore_full(V2 / str(anchor["state_file"]), CONFIG)
            return (
                env,
                np.asarray(data["u_exec"], dtype=np.float64),
                np.asarray(data["u_flow"], dtype=np.float64),
                np.asarray(data["u_safe"], dtype=np.float64),
            )
    raise RuntimeError((row["state_id"], "unsupported source_type", row["source_type"]))


def replay_to(row: dict, source_local_step: int) -> tuple[GiveWayEnv, np.ndarray, np.ndarray]:
    env, executed, flow, safe = actions_for(row)
    if source_local_step < 0 or source_local_step >= len(executed):
        raise RuntimeError((row["state_id"], source_local_step, len(executed)))
    for index, action in enumerate(executed[:source_local_step]):
        _, _, done, _ = env.step(action)
        if done:
            raise RuntimeError((row["state_id"], "source terminal before requested state", index, env.step_count))
    if env.done:
        raise RuntimeError((row["state_id"], "requested terminal state"))
    return env, flow[source_local_step], safe[source_local_step]


def env_agrees(a: GiveWayEnv, b: GiveWayEnv) -> bool:
    scalar = (
        a.step_count == b.step_count
        and a.candidate_since == b.candidate_since
        and a.first_success_step == b.first_success_step
        and a.first_deadlock_step == b.first_deadlock_step
        and a.first_wall_collision_step == b.first_wall_collision_step
        and a.first_agent_collision_step == b.first_agent_collision_step
        and a.done == b.done
        and a.ever_candidate_deadlock == b.ever_candidate_deadlock
    )
    arrays = (
        np.allclose(a.positions, b.positions, atol=1e-12, rtol=0.0)
        and np.allclose(a.velocities, b.velocities, atol=1e-12, rtol=0.0)
        # Saved dataset snapshots intentionally retain only the deployment
        # 41-sample tail and mark older entries NaN.  Replay keeps the complete
        # real history; compare the authoritative feature/monitor-relevant tail.
        and np.allclose(np.asarray(a.distance_history[-41:]), np.asarray(b.distance_history[-41:]), atol=1e-12, rtol=0.0, equal_nan=True)
        and np.isclose(a.stuck_timer, b.stuck_timer, atol=1e-12, rtol=0.0)
        and np.isclose(a.max_stuck_timer, b.max_stuck_timer, atol=1e-12, rtol=0.0)
    )
    return scalar and arrays


def monitor_signature(env: GiveWayEnv) -> dict:
    return {
        "candidate_since": -1 if env.candidate_since is None else int(env.candidate_since),
        "stuck_timer": float(env.stuck_timer),
        "max_stuck_timer": float(env.max_stuck_timer),
        "ever_candidate_deadlock": bool(env.ever_candidate_deadlock),
        "first_success_step": -1 if env.first_success_step is None else int(env.first_success_step),
        "first_deadlock_step": -1 if env.first_deadlock_step is None else int(env.first_deadlock_step),
        "first_wall_collision_step": -1 if env.first_wall_collision_step is None else int(env.first_wall_collision_step),
        "first_agent_collision_step": -1 if env.first_agent_collision_step is None else int(env.first_agent_collision_step),
        "done": bool(env.done),
    }


def feature(env: GiveWayEnv, flow: np.ndarray, safe: np.ndarray) -> np.ndarray:
    # Historical snapshots encode unavailable pre-tail history as NaN.  The
    # frozen feature contract consumes only real available observations and
    # then applies causal left padding; expose that finite suffix without
    # mutating the physical/monitor state retained in the snapshot.
    original = env.distance_history
    finite = [np.asarray(value, dtype=np.float64).copy() for value in original if np.isfinite(value).all()]
    if not finite:
        raise RuntimeError("no finite goal-error history available")
    env.distance_history = finite
    try:
        vector, _ = BUILDER.build(env, {"u_flow": flow, "u_safe": safe}, CONFIG, CBF)
    finally:
        env.distance_history = original
    return np.asarray(vector, dtype=np.float64)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    # Source scripts may already be present, but preparation must never mutate
    # a previously frozen study manifest or generated state directory.
    if (HERE / "anchor_manifest.json").exists() or (HERE / "states").exists():
        raise RuntimeError(f"refusing to overwrite prepared study {HERE}")
    (HERE / "states").mkdir()

    subset_doc = json.loads((MIGRATION / "migration_subset_manifest.json").read_text())
    subset_ids = [str(row["state_id"]) for row in subset_doc["selected_states"]]
    all_state_rows = {row["state_id"]: row for row in read_jsonl(STARTUP / "state_manifest.jsonl")}
    candidate_rows = [all_state_rows[state_id] for state_id in subset_ids]

    # Eligibility checks intentionally use only provenance/replay validity.
    eligible = []
    reconstruction_failures = []
    for row in candidate_rows:
        try:
            local = int(row.get("source_local_step", row.get("physical_step")))
            source_env, executed, _, _ = actions_for(row)
            if local < 4 or local + 4 >= len(executed):
                continue
            rebuilt, _, _ = replay_to(row, local)
            stored = restore_full(STARTUP / str(row["state_file"]), CONFIG)
            if not env_agrees(rebuilt, stored):
                reconstruction_failures.append({"state_id": row["state_id"], "reason": "anchor_snapshot_mismatch"})
                continue
            # Verify all requested neighbors are real nonterminal trajectory states.
            for offset in OFFSETS:
                neighbor, _, _ = replay_to(row, local + offset)
                if neighbor.done:
                    raise RuntimeError("terminal neighbor")
            eligible.append(row)
        except Exception as exc:
            reconstruction_failures.append({"state_id": row["state_id"], "reason": repr(exc)})
    if len(eligible) < ANCHOR_COUNT:
        raise RuntimeError(("insufficient provenance-valid anchors", len(eligible), reconstruction_failures))

    ordering = np.random.default_rng(ANCHOR_PERMUTATION_SEED).permutation(len(eligible))
    anchors = [eligible[index] for index in ordering[:ANCHOR_COUNT]]

    # Only after the selection is frozen may we attach prior B63 canonical data.
    basin_rows = {row["state_id"]: row for row in csv.DictReader((MIGRATION / "subset_basin_summary.csv").open())}
    migration_rows = {row["state_id"]: row for row in subset_doc["selected_states"]}
    selected = []
    neighbor_rows = []
    all_states = []
    for anchor_rank, row in enumerate(anchors):
        state_id = str(row["state_id"])
        local = int(row.get("source_local_step", row.get("physical_step")))
        env, flow, safe = replay_to(row, local)
        stored = restore_full(STARTUP / str(row["state_file"]), CONFIG)
        if not env_agrees(env, stored):
            raise RuntimeError((state_id, "post-selection anchor mismatch"))
        h = feature(env, flow, safe)
        anchor_file = HERE / "states" / f"A{anchor_rank:02d}__{state_id}.npz"
        save_full(anchor_file, env)
        migration = migration_rows[state_id]
        basin = basin_rows[state_id]
        anchor = {
            "anchor_rank": anchor_rank,
            "state_id": state_id,
            "category": row.get("category"),
            "source_type": row.get("source_type"),
            "source_trajectory": row.get("source_trajectory"),
            "source_path": row.get("source_path"),
            "source_path_sha256": sha(Path(row["source_path"]) if Path(row["source_path"]).is_absolute() else (ROOT / "diagnostics/gphi_training_dataset_v3" / row["source_path"])),
            "source_local_step": local,
            "absolute_step": int(env.step_count),
            "state_file": str(anchor_file),
            "state_sha256": sha(anchor_file),
            "feature": h.tolist(),
            "feature_sha256": hashlib.sha256(h.tobytes()).hexdigest(),
            "monitor": monitor_signature(env),
            "matched_flow_seeds": migration["matched_flow_seeds"],
            "rng_namespace": int(migration["rng_namespace"]),
            "canonical_eta_provisional": json.loads(basin["orthoflow3_canonical_eta"]),
            "canonical_eta_B63_source": "orthoflow3_representation_migration_v1 budgeted promoted set",
            "canonical_eta_B63": True,
        }
        selected.append(anchor)
        all_states.append({**anchor, "role": "anchor", "offset": 0, "anchor_rank": anchor_rank})
        for offset in OFFSETS:
            neighbor_env, neighbor_flow, neighbor_safe = replay_to(row, local + offset)
            h_neighbor = feature(neighbor_env, neighbor_flow, neighbor_safe)
            neighbor_id = f"A{anchor_rank:02d}__{state_id}__d{offset:+d}"
            neighbor_file = HERE / "states" / f"{neighbor_id}.npz"
            save_full(neighbor_file, neighbor_env)
            monitor = monitor_signature(neighbor_env)
            neighbor = {
                "neighbor_id": neighbor_id,
                "anchor_rank": anchor_rank,
                "anchor_state_id": state_id,
                "offset_steps": offset,
                "offset_seconds": float(offset * CONFIG.dt),
                "absolute_step": int(neighbor_env.step_count),
                "source_local_step": local + offset,
                "state_file": str(neighbor_file),
                "state_sha256": sha(neighbor_file),
                "feature": h_neighbor.tolist(),
                "feature_sha256": hashlib.sha256(h_neighbor.tobytes()).hexdigest(),
                "monitor": monitor,
                "matched_flow_seeds": migration["matched_flow_seeds"],
                "rng_namespace": int(migration["rng_namespace"]),
                "category": row.get("category"),
                "source_trajectory": row.get("source_trajectory"),
            }
            neighbor_rows.append(neighbor)
            all_states.append({**neighbor, "role": "neighbor"})

    # One common shared cloud, independent of state labels/outcomes.  The 23
    # nonzero probes are a seeded permutation of the exact frozen 256 Sobol design.
    design = json.loads((MIGRATION / "orthoflow3_search_config.json").read_text())
    points_path = Path(design["candidate_design"]["path"])
    points = json.loads(points_path.read_text())["points"]
    order = np.random.default_rng(PROBE_PERMUTATION_SEED).permutation(len(points))[:23]
    cloud = [{"probe_id": "ZERO", "eta_index": None, "eta": [0.0, 0.0, 0.0], "source": "explicit_zero"}]
    cloud.extend(
        {
            "probe_id": f"S{int(index):03d}",
            "eta_index": int(index),
            "eta": points[int(index)]["theta"],
            "source": "frozen_256_sobol_subset",
        }
        for index in order
    )
    write_csv(HERE / "common_eta_probe_cloud.csv", cloud)

    with np.load(ETA_CHECKPOINT, allow_pickle=False) as checkpoint:
        norm_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64)
        norm_scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float64)
    distance_rows = []
    anchor_by_rank = {row["anchor_rank"]: row for row in selected}
    for neighbor in neighbor_rows:
        anchor = anchor_by_rank[neighbor["anchor_rank"]]
        a_env = restore_full(Path(anchor["state_file"]), CONFIG)
        b_env = restore_full(Path(neighbor["state_file"]), CONFIG)
        position_delta = b_env.positions - a_env.positions
        goal_error_a = np.linalg.norm(a_env.goals - a_env.positions, axis=-1)
        goal_error_b = np.linalg.norm(b_env.goals - b_env.positions, axis=-1)
        rel_a = a_env.positions[0] - a_env.positions[1]
        rel_b = b_env.positions[0] - b_env.positions[1]
        h_a = np.asarray(anchor["feature"], dtype=np.float64)
        h_b = np.asarray(neighbor["feature"], dtype=np.float64)
        ma, mb = anchor["monitor"], neighbor["monitor"]
        discrete = any(ma[key] != mb[key] for key in (
            "candidate_since", "ever_candidate_deadlock", "first_success_step", "first_deadlock_step",
            "first_wall_collision_step", "first_agent_collision_step", "done",
        ))
        distance_rows.append({
            "anchor_rank": neighbor["anchor_rank"], "anchor_state_id": anchor["state_id"],
            "neighbor_id": neighbor["neighbor_id"], "offset_steps": neighbor["offset_steps"],
            "offset_seconds": neighbor["offset_seconds"],
            "per_agent_position_displacement": json.dumps(np.linalg.norm(position_delta, axis=-1).tolist(), separators=(",", ":")),
            "aggregate_position_displacement": float(np.linalg.norm(position_delta)),
            "relative_geometry_change": float(np.linalg.norm(rel_b - rel_a)),
            "goal_error_change_per_agent": json.dumps((goal_error_b - goal_error_a).tolist(), separators=(",", ":")),
            "goal_error_change_l2": float(np.linalg.norm(goal_error_b - goal_error_a)),
            "normalized_feature_distance": float(np.linalg.norm((h_b - norm_mean) / norm_scale - (h_a - norm_mean) / norm_scale)),
            "stuck_timer_change": float(mb["stuck_timer"] - ma["stuck_timer"]),
            "max_stuck_timer_change": float(mb["max_stuck_timer"] - ma["max_stuck_timer"]),
            "candidate_since_anchor": ma["candidate_since"], "candidate_since_neighbor": mb["candidate_since"],
            "discrete_monitor_transition": discrete,
        })
    write_csv(HERE / "state_pair_distances.csv", distance_rows)

    # Every state/probe initially receives eight same-identity future streams.
    screen_arms = []
    for state in all_states:
        for probe in cloud:
            rollout_state_id = state["state_id"] if state["role"] == "anchor" else state["neighbor_id"]
            screen_arms.append({
                "arm_id": f"SCREEN__{state['role'][0].upper()}{state['anchor_rank']:02d}__{state.get('offset', state.get('offset_steps', 0)):+03d}__{probe['probe_id']}",
                "state_id": rollout_state_id, "role": state["role"],
                "anchor_rank": state["anchor_rank"], "state_file": state["state_file"], "state_sha256": state["state_sha256"],
                "absolute_step": state["absolute_step"], "rng_namespace": state["rng_namespace"],
                "eta": probe["eta"], "probe_id": probe["probe_id"], "eta_index": probe["eta_index"],
                "seeds": state["matched_flow_seeds"][:SCREEN_SEEDS_PER_STATE_ETA],
            })
    plan = {
        "schema": "orthoflow3_local_basin_screen_plan_v1", "basis_family": "orthoflow3",
        "seed_policy": "same inherited eight future Flow identities for every anchor-neighbor family",
        "common_cloud_hash": canonical_hash(cloud), "arms": screen_arms,
        "new_continuation_upper_bound": len(screen_arms) * SCREEN_SEEDS_PER_STATE_ETA,
        "physical_step_upper_bound": sum((850 - int(arm["absolute_step"])) * len(arm["seeds"]) for arm in screen_arms),
    }
    plan["content_sha256"] = canonical_hash(plan)
    write_json(HERE / "screen_plan.json", plan)

    protocol = {
        "study": "orthoflow3_local_basin_continuity_v1", "status": "PREPARED_NO_ROLLOUTS",
        "scientific_scope": "model-free fixed-eta local basin continuity", "no_training": True,
        "basis_source": str(ROOT / "diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py"),
        "basis_sha256": "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
        "basis_interface": str(ROOT / "shared_control/basis_families.py"),
        "anchor_selection": {"parent_subset": "orthoflow3 migration frozen 32", "seed": ANCHOR_PERMUTATION_SEED, "count": ANCHOR_COUNT, "offsets": list(OFFSETS)},
        "common_probe_cloud": {"total": 24, "zero": 1, "sobol": 23, "seed": PROBE_PERMUTATION_SEED, "source": str(points_path), "source_sha256": sha(points_path)},
        "screening": {"seeds_per_state_eta": SCREEN_SEEDS_PER_STATE_ETA, "B63_claim_allowed": False},
        "hard_budget": {"new_continuations": 15000, "physical_steps": 8000000, "wall_minutes": 60},
        "state_reconstruction_failures": reconstruction_failures,
        "normalization_checkpoint": str(ETA_CHECKPOINT), "normalization_checkpoint_sha256": sha(ETA_CHECKPOINT),
        "state_manifest_sha256": sha(STARTUP / "state_manifest.jsonl"),
        "migration_manifest_sha256": sha(MIGRATION / "manifest.json"),
    }
    write_json(HERE / "protocol.md.json", protocol)
    (HERE / "protocol.md").write_text(
        "# OrthoFlow3 local basin continuity protocol\n\n"
        "Anchors are selected by a seeded permutation from the prior uniformly selected 32-state migration subset before attaching eta labels. "
        "Neighbors are exact replayed augmented states at t-4, t-1, t+1, and t+4 on the same original trajectory. "
        "The fixed shared probe cloud is eta=0 plus 23 points from the frozen authoritative 256-point Sobol design. "
        "Each state/probe uses eight matched future Flow identities; 8/8 is screening only.\n"
    )
    write_json(HERE / "authoritative_orthoflow3_manifest.json", {
        "implementation": protocol["basis_source"], "sha256": protocol["basis_sha256"],
        "interface": protocol["basis_interface"], "search_config": str(MIGRATION / "orthoflow3_search_config.json"),
        "search_config_sha256": sha(MIGRATION / "orthoflow3_search_config.json"),
    })
    write_json(HERE / "anchor_manifest.json", {"anchors": selected, "eligible_count": len(eligible), "permutation": ordering.tolist()})
    write_json(HERE / "neighbor_manifest.json", {"neighbors": neighbor_rows})
    write_json(HERE / "preflight_budget.json", {
        "screen_new_continuation_upper_bound": plan["new_continuation_upper_bound"],
        "screen_physical_step_upper_bound": plan["physical_step_upper_bound"],
        "remaining_after_screen_continuations": 15000 - plan["new_continuation_upper_bound"],
        "remaining_after_screen_steps": 8000000 - plan["physical_step_upper_bound"],
        "screen_within_budget": plan["new_continuation_upper_bound"] <= 15000 and plan["physical_step_upper_bound"] <= 8000000,
    })
    print(json.dumps({"eligible": len(eligible), "anchors": len(selected), "neighbors": len(neighbor_rows), "screen_arms": len(screen_arms), "screen_rollouts_upper": plan["new_continuation_upper_bound"], "screen_steps_upper": plan["physical_step_upper_bound"]}, indent=2))


if __name__ == "__main__":
    integrity = json.loads((ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text())
    CONFIG = Config(**integrity["environment"])
    CBF = CBFConfig(**integrity["cbf"])
    BUILDER = StartupAwareFeatureBuilder()
    V2_STATES = {row["state_id"]: row for row in read_jsonl(V2 / "state_manifest.jsonl")}
    main()
