"""Predeclare and materialize exact restorable augmented training states."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
BASE = SYSROOT / "results/risk_audit_seed0_baselines/mac_cbf"
BASE_CONFIG = BASE.parent / "config.json"
QUAL = ROOT / "diagnostics/cl_fhcb_qualification/raw/stage1"
GEOMETRY = ROOT / "diagnostics/success_basin_geometry"
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import snapshot_augmented
from diagnostics.success_basin_deformation_decomposition.analyze import eta_key, load_inventory
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.evaluate import load_policy


ANCHOR_BEST = {
    "D1_pair231": (0.5703125, -0.375, -0.125),
    "D2_pair228": (0.40625, -0.5, 0.0),
    "D4_pair227": (0.375, -0.4375, -0.0625),
}
RECOVERY_COUNTS = {"D1_pair231": 7, "D2_pair228": 7, "D4_pair227": 6}
RECOVERY_OFFSETS = (10, 20, 30, 40, 50, 60, 70)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def restore_full(path: Path, config: Config) -> GiveWayEnv:
    with np.load(path) as data:
        env = GiveWayEnv(config)
        env.positions = np.asarray(data["positions"], dtype=np.float64).copy()
        env.velocities = np.asarray(data["velocities"], dtype=np.float64).copy()
        env.step_count = int(data["step"])
        start = int(data["history_start_step"])
        env.distance_history = [np.full(2, np.nan) for _ in range(env.step_count + 1)]
        for offset, value in enumerate(np.asarray(data["error_history"], dtype=np.float64)):
            env.distance_history[start + offset] = value.copy()
        candidate = int(data["candidate_since"])
        env.candidate_since = None if candidate < 0 else candidate
        env.stuck_timer = float(data["stuck_timer"])
        env.max_stuck_timer = float(data["max_stuck_timer"])
        env.ever_candidate_deadlock = bool(data["ever_candidate_deadlock"])
        for field in (
            "first_success_step", "first_deadlock_step", "first_wall_collision_step",
            "first_agent_collision_step",
        ):
            value = int(data[field])
            setattr(env, field, None if value < 0 else value)
        env.done = bool(data["done"])
    return env


def save_full(path: Path, env: GiveWayEnv) -> None:
    aug = snapshot_augmented(env)
    np.savez_compressed(
        path,
        positions=env.positions,
        velocities=env.velocities,
        step=np.asarray(env.step_count),
        error_history=aug.error_history,
        history_start_step=np.asarray(aug.history_start_step),
        candidate_since=np.asarray(-1 if env.candidate_since is None else env.candidate_since),
        stuck_timer=np.asarray(env.stuck_timer),
        max_stuck_timer=np.asarray(env.max_stuck_timer),
        ever_candidate_deadlock=np.asarray(env.ever_candidate_deadlock),
        first_success_step=np.asarray(-1 if env.first_success_step is None else env.first_success_step),
        first_deadlock_step=np.asarray(-1 if env.first_deadlock_step is None else env.first_deadlock_step),
        first_wall_collision_step=np.asarray(-1 if env.first_wall_collision_step is None else env.first_wall_collision_step),
        first_agent_collision_step=np.asarray(-1 if env.first_agent_collision_step is None else env.first_agent_collision_step),
        done=np.asarray(env.done),
    )


def replay_actions(initial: np.ndarray, actions: np.ndarray, stop: int, config: Config) -> GiveWayEnv:
    env = GiveWayEnv(config)
    env.reset(initial)
    for action in actions[:stop]:
        _, _, done, _ = env.step(action)
        if done:
            raise RuntimeError(("source terminated before selected state", stop, env.step_count))
    return env


def anchor_source(state_id: str) -> tuple[Path, int, int]:
    catalog = {row["state_id"]: row for row in json.loads((ROOT / "diagnostics/true_q_geometry/raw/q_map/state_catalog.json").read_text())}
    record_id = catalog[state_id]["source_trace"]
    manifest = json.loads((QUAL / "manifest.json").read_text())
    record = next(row for row in manifest["records"] if row["id"] == record_id)
    return QUAL / record["relative_path"], int(catalog[state_id]["start_step"]), int(record["pair_id"])


def source_step_payload(source_type: str, path: Path, local_step: int) -> dict:
    with np.load(path) as data:
        if source_type == "baseline":
            return {
                "u_flow": np.asarray(data["nominal_velocity"][local_step]),
                "u_safe": np.asarray(data["executed_velocity"][local_step]),
                "u_exec": np.asarray(data["executed_velocity"][local_step]),
                "positions_after": np.asarray(data["positions"][local_step]),
                "stuck_timer_after": float(data["stuck_timer"][local_step]),
                "window_progress_after": np.asarray(data["window_progress"][local_step]),
                "event": (
                    "collision" if bool(data["wall_collision"][local_step] or data["agent_collision"][local_step])
                    else "success" if bool(data["task_success"][local_step])
                    else "deadlock" if bool(data["deadlock"][local_step]) else "running"
                ),
            }
        return {
            "u_flow": np.asarray(data["u_flow"][local_step]),
            "u_safe": np.asarray(data["u_safe"][local_step]),
            "u_exec": np.asarray(data["u_exec"][local_step]),
            "positions_after": np.asarray(data["positions_after"][local_step]),
            "stuck_timer_after": float(data["timer"][local_step]) if "timer" in data.files else np.nan,
            "window_progress_after": np.full(2, np.nan),
            "event": str(data["event"][local_step]),
        }


def assign_splits(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["leakage_group"]].append(row)
    # Keep all recovery descendants and their anchor in one split per root.
    # These three large groups already occupy 8/8/7 states; distribute the
    # remaining source trajectories deterministically without consulting any
    # oracle outcome.  Every split then contains NORMAL, PRE_DEADLOCK, and
    # RECOVERY while remaining close to 70/15/15 (46/10/11 states).
    fixed = {"anchor_D1_pair231": "train", "anchor_D2_pair228": "validation", "anchor_D4_pair227": "test"}
    assignment = dict(fixed)
    normal_groups = sorted(
        group for group, members in groups.items()
        if group not in fixed and {row["category"] for row in members} == {"NORMAL"}
    )
    pre_groups = sorted(
        group for group, members in groups.items()
        if group not in fixed and {row["category"] for row in members} == {"PRE_DEADLOCK"}
    )
    for group in normal_groups[:2]:
        assignment[group] = "validation"
    for group in normal_groups[2:4]:
        assignment[group] = "test"
    # The only two-state baseline pre-deadlock trajectory fits in test without
    # fragmenting a source trajectory; larger pre-deadlock groups stay train.
    smallest_pre = min(pre_groups, key=lambda group: (len(groups[group]), group))
    assignment[smallest_pre] = "test"
    for group in groups:
        assignment.setdefault(group, "train")
    return assignment


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    states_dir = HERE / "states"; refs_dir = HERE / "audit_refs"
    states_dir.mkdir(exist_ok=False); refs_dir.mkdir(exist_ok=False)
    base_config = json.loads(BASE_CONFIG.read_text())
    config = Config(**base_config["policy_provenance"]["evaluation_environment"])
    if sha(SYSROOT / "single_integrator/environment.py") != base_config["policy_provenance"]["source_sha256"]["single_integrator/environment.py"]:
        raise RuntimeError("baseline source does not match frozen environment")
    summary = json.loads((BASE / "summary.json").read_text())["rollouts"]
    by_id = {int(row["rollout_id"]): row for row in summary}
    success_ids = [int(row["rollout_id"]) for row in summary if row["outcome"] == "success"][:24]
    deadlock_ids = [int(row["rollout_id"]) for row in summary if row["outcome"] == "safe_deadlock"]
    if len(success_ids) != 24 or len(deadlock_ids) != 7:
        raise AssertionError((len(success_ids), len(deadlock_ids)))

    planned = []
    for rid in success_ids:
        terminal = int(by_id[rid]["episode_steps"])
        planned.append({
            "state_id": f"N_r{rid:03d}_m120", "category": "NORMAL", "source_type": "baseline",
            "source_path": str(BASE / f"rollout_{rid:04d}.npz"), "source_trajectory": f"baseline_r{rid:03d}",
            "leakage_group": f"baseline_r{rid:03d}", "source_seed": 42, "source_rng_id": rid,
            "source_local_step": terminal - 120, "absolute_step": terminal - 120,
        })
    pre_specs = []
    for index, rid in enumerate(deadlock_ids):
        offsets = (160, 80, 40) if index < 6 else (160, 80)
        terminal = int(by_id[rid]["episode_steps"])
        for offset in offsets:
            pre_specs.append({
                "state_id": f"P_r{rid:03d}_m{offset:03d}", "category": "PRE_DEADLOCK", "source_type": "baseline",
                "source_path": str(BASE / f"rollout_{rid:04d}.npz"), "source_trajectory": f"baseline_r{rid:03d}",
                "leakage_group": f"baseline_r{rid:03d}", "source_seed": 42, "source_rng_id": rid,
                "source_local_step": terminal - offset, "absolute_step": terminal - offset,
                "lead_steps_to_source_deadlock": offset,
            })
    planned.extend(pre_specs)

    # Anchors are explicit sanity states and share leakage groups with their recovery descendants.
    for anchor in ANCHOR_BEST:
        trace, start, pair_id = anchor_source(anchor)
        planned.append({
            "state_id": anchor, "category": "PRE_DEADLOCK", "source_type": "qualification",
            "source_path": str(trace), "source_trajectory": f"anchor_{anchor}",
            "leakage_group": f"anchor_{anchor}", "source_seed": 19073, "source_rng_id": pair_id,
            "source_local_step": start, "absolute_step": start,
        })

    # Pick distinct already-successful corrected trajectories without inspecting their future labels.
    _, effective, _, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    inventory = defaultdict(list)
    for row in effective:
        inventory[(row["state_id"], eta_key(row["eta"]))].append(row)
    recovery_sources = []
    for anchor, count in RECOVERY_COUNTS.items():
        eta = ANCHOR_BEST[anchor]
        options = sorted(
            (row for row in inventory[(anchor, eta)] if row["outcome"] == "success" and not row["execution_error"]),
            key=lambda row: int(row["seed"]),
        )[:count]
        if len(options) != count:
            raise AssertionError((anchor, len(options), count))
        anchor_trace, anchor_step, pair_id = anchor_source(anchor)
        with np.load(anchor_trace) as source:
            anchor_initial = np.asarray(source["positions_before"][0])
            anchor_actions = np.asarray(source["u_exec"])
        anchor_env = replay_actions(anchor_initial, anchor_actions, anchor_step, config)
        for index, row in enumerate(options):
            local = RECOVERY_OFFSETS[index]
            with np.load(row["path"]) as source:
                if len(source["u_exec"]) <= local:
                    raise AssertionError((anchor, row["seed"], len(source["u_exec"]), local))
                env = restore_full_from_env(anchor_env, config)
                for j, action in enumerate(np.asarray(source["u_exec"][:local])):
                    if not np.allclose(env.positions, source["positions_before"][j], atol=1e-12, rtol=0):
                        raise AssertionError((anchor, row["seed"], j, "recovery replay mismatch"))
                    _, _, done, _ = env.step(action)
                    if done:
                        raise AssertionError((anchor, row["seed"], j, "premature recovery termination"))
            recovery_sources.append({
                "state_id": f"R_{anchor.split('_')[0]}_s{int(row['seed'])}_p{local:02d}",
                "category": "RECOVERY", "source_type": "recovery",
                "source_path": str(row["path"]), "source_trajectory": f"{anchor}_eta{eta}_seed{int(row['seed'])}",
                "leakage_group": f"anchor_{anchor}", "source_seed": int(row["seed"]),
                "source_rng_id": pair_id, "source_local_step": local,
                "absolute_step": anchor_step + local, "anchor_state": anchor,
                "source_eta": list(eta), "prebuilt_env": env,
            })
    planned.extend(recovery_sources)
    if Counter(row["category"] for row in planned) != Counter({"NORMAL": 24, "PRE_DEADLOCK": 23, "RECOVERY": 20}):
        raise AssertionError(Counter(row["category"] for row in planned))

    manifest = []
    for index, spec in enumerate(planned):
        source_path = Path(spec["source_path"])
        if spec["source_type"] == "baseline":
            with np.load(source_path) as source:
                env = replay_actions(np.asarray(source["initial_positions"]), np.asarray(source["executed_velocity"]), int(spec["source_local_step"]), config)
        elif spec["source_type"] == "qualification":
            with np.load(source_path) as source:
                env = replay_actions(np.asarray(source["positions_before"][0]), np.asarray(source["u_exec"]), int(spec["source_local_step"]), config)
        else:
            env = spec.pop("prebuilt_env")
        if env.done:
            raise AssertionError((spec["state_id"], "selected terminal state"))
        state_path = states_dir / f"{spec['state_id']}.npz"
        save_full(state_path, env)
        payload = source_step_payload("baseline" if spec["source_type"] == "baseline" else "corrected", source_path, int(spec["source_local_step"]))
        ref_path = refs_dir / f"{spec['state_id']}.npz"
        np.savez_compressed(ref_path, **payload)
        manifest.append({
            **spec, "state_index": index, "rng_namespace": 100000 + index,
            "state_file": str(state_path.relative_to(HERE)), "state_sha256": sha(state_path),
            "audit_reference_file": str(ref_path.relative_to(HERE)), "audit_reference_sha256": sha(ref_path),
            "source_sha256": sha(source_path), "step": env.step_count,
            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
            "ever_candidate_deadlock": env.ever_candidate_deadlock,
        })

    assignment = assign_splits(manifest)
    for row in manifest:
        row["split"] = assignment[row["leakage_group"]]
    write_jsonl(HERE / "state_manifest.jsonl", manifest)
    split_manifest = {
        "assignment_unit": "leakage_group (source trajectory; all descendants of D1/D2/D4 anchor grouped)",
        "state_ids": {split: [row["state_id"] for row in manifest if row["split"] == split] for split in ("train", "validation", "test")},
        "leakage_groups": {split: sorted({row["leakage_group"] for row in manifest if row["split"] == split}) for split in ("train", "validation", "test")},
        "category_counts": {split: dict(Counter(row["category"] for row in manifest if row["split"] == split)) for split in ("train", "validation", "test")},
    }
    (HERE / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    protocol = {
        "study": "gphi_training_dataset_v1", "training_performed": False,
        "state_selection_frozen_before_oracle": True,
        "state_count": len(manifest), "category_counts": dict(Counter(row["category"] for row in manifest)),
        "baseline_source": str(BASE), "baseline_seed": 42,
        "environment": config.to_dict(),
        "checkpoint": str(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"),
        "checkpoint_sha256": sha(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"),
        "oracle_seed_default": list(range(95210001, 95210065)),
        "oracle_rule": "success_count >= 63/64; stop after second physical failure",
        "candidate_etas": [
            [0.5703125, -0.375, -0.125], [0.578125, -0.40625, -0.125], [0.578125, -0.34375, -0.125],
            [0.40625, -0.5, 0.0], [0.4375, -0.53125, 0.0],
            [0.375, -0.4375, -0.0625], [0.40625, -0.4375, -0.0625], [1.0, 0.0, 0.25],
        ],
        "frozen_hashes": {
            "environment": sha(SYSROOT / "single_integrator/environment.py"),
            "projection": sha(SYSROOT / "single_integrator/cbf.py"),
            "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
            "retry": sha(ROOT / "diagnostics/success_basin_multimodality/exact_projector.py"),
            "checkpoint": sha(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"),
        },
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"states": len(manifest), "categories": protocol["category_counts"], "splits": {k: len(v) for k, v in split_manifest["state_ids"].items()}}, indent=2))


def restore_full_from_env(source: GiveWayEnv, config: Config) -> GiveWayEnv:
    """Exact in-memory clone used only while extracting immutable snapshots."""
    env = GiveWayEnv(config)
    env.positions = source.positions.copy(); env.velocities = source.velocities.copy()
    env.step_count = source.step_count
    env.distance_history = [value.copy() for value in source.distance_history]
    env.candidate_since = source.candidate_since
    env.stuck_timer = source.stuck_timer; env.max_stuck_timer = source.max_stuck_timer
    env.ever_candidate_deadlock = source.ever_candidate_deadlock
    env.first_success_step = source.first_success_step; env.first_deadlock_step = source.first_deadlock_step
    env.first_wall_collision_step = source.first_wall_collision_step; env.first_agent_collision_step = source.first_agent_collision_step
    env.done = source.done
    return env


if __name__ == "__main__":
    main()
