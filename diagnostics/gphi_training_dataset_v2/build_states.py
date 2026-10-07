"""Materialize the frozen state-coverage expansion for G_phi dataset v2.

The controller, environment, projection, event, and oracle semantics are
imported unchanged.  Dataset v2 retains every v1 state and split, then adds
label-blind temporal/source diversity up to 246 exact augmented states.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "diagnostics/gphi_training_dataset_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
BASE = SYSROOT / "results/risk_audit_seed0_baselines/mac_cbf"
BASE_CONFIG = BASE.parent / "config.json"
QUAL = ROOT / "diagnostics/cl_fhcb_qualification/raw/stage1"
CATALOG = ROOT / "diagnostics/true_q_geometry/raw/q_map/state_catalog.json"
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import snapshot_augmented
from diagnostics.success_basin_deformation_decomposition.analyze import eta_key, load_inventory
from single_integrator.environment import Config, GiveWayEnv


ANCHOR_BEST = {
    "D1_pair231": (0.5703125, -0.375, -0.125),
    "D2_pair228": (0.40625, -0.5, 0.0),
    "D4_pair227": (0.375, -0.4375, -0.0625),
}
RECOVERY_COUNTS = {"D1_pair231": 22, "D2_pair228": 22, "D4_pair227": 22}
V1_RECOVERY_COUNTS = {"D1_pair231": 7, "D2_pair228": 7, "D4_pair227": 6}
V1_RECOVERY_OFFSETS = (10, 20, 30, 40, 50, 60, 70)
PRE_OFFSETS = (40, 80, 120, 160, 240)
QUAL_OFFSETS = (20, 40, 80, 120, 160)


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
    if env.step_count != stop:
        raise AssertionError((env.step_count, stop))
    return env


def clone_env(source: GiveWayEnv, config: Config) -> GiveWayEnv:
    env = GiveWayEnv(config)
    env.positions = source.positions.copy()
    env.velocities = source.velocities.copy()
    env.step_count = source.step_count
    env.distance_history = [value.copy() for value in source.distance_history]
    env.candidate_since = source.candidate_since
    env.stuck_timer = source.stuck_timer
    env.max_stuck_timer = source.max_stuck_timer
    env.ever_candidate_deadlock = source.ever_candidate_deadlock
    env.first_success_step = source.first_success_step
    env.first_deadlock_step = source.first_deadlock_step
    env.first_wall_collision_step = source.first_wall_collision_step
    env.first_agent_collision_step = source.first_agent_collision_step
    env.done = source.done
    return env


def qualification_sources() -> dict[int, dict]:
    manifest = json.loads((QUAL / "manifest.json").read_text())
    return {
        int(row["pair_id"]): row
        for row in manifest["records"]
        if row["outcome"] == "deadlock"
    }


def anchor_source(state_id: str) -> tuple[Path, int, int]:
    catalog = {row["state_id"]: row for row in json.loads(CATALOG.read_text())}
    row = catalog[state_id]
    record = next(
        item for item in json.loads((QUAL / "manifest.json").read_text())["records"]
        if item["id"] == row["source_trace"]
    )
    return QUAL / record["relative_path"], int(row["start_step"]), int(record["pair_id"])


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


def phase_step(terminal: int, rank: int) -> tuple[int, str]:
    """Choose well-separated normal phases without using oracle labels."""
    phase = rank % 3
    if phase == 0:
        return max(60, int(round(0.20 * terminal))), "early"
    if phase == 1:
        return max(60, int(round(0.50 * terminal))), "mid"
    return terminal - 120, "late"


def assign_splits(rows: list[dict]) -> dict[str, str]:
    """Preserve all v1 memberships, then balance only new source groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["leakage_group"]].append(row)
    v1_rows = [json.loads(line) for line in (V1 / "state_manifest.jsonl").read_text().splitlines()]
    assignment = {row["leakage_group"]: row["split"] for row in v1_rows}
    assignment.update({"qual_pair226": "train", "qual_pair225": "train"})

    new_normal = sorted(
        group for group, members in groups.items()
        if group not in assignment and {row["category"] for row in members} == {"NORMAL"}
    )
    # Existing NORMAL memberships are 20/2/2.  Add 80/8/8 to reach 100/10/10.
    for index, group in enumerate(new_normal):
        assignment[group] = "train" if index < 80 else "validation" if index < 88 else "test"
    for group in groups:
        if group not in assignment:
            raise AssertionError(("unassigned leakage group", group))
    return assignment


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    states_dir = HERE / "states"
    refs_dir = HERE / "audit_refs"
    states_dir.mkdir(exist_ok=False)
    refs_dir.mkdir(exist_ok=False)
    base_config = json.loads(BASE_CONFIG.read_text())
    config = Config(**base_config["policy_provenance"]["evaluation_environment"])
    frozen_environment = base_config["policy_provenance"]["source_sha256"]["single_integrator/environment.py"]
    if sha(SYSROOT / "single_integrator/environment.py") != frozen_environment:
        raise RuntimeError("baseline source does not match frozen environment")

    summary = json.loads((BASE / "summary.json").read_text())["rollouts"]
    by_id = {int(row["rollout_id"]): row for row in summary}
    success_ids = [int(row["rollout_id"]) for row in summary if row["outcome"] == "success"]
    deadlock_ids = [int(row["rollout_id"]) for row in summary if row["outcome"] == "safe_deadlock"]
    if len(success_ids) != 150 or len(deadlock_ids) != 7:
        raise AssertionError((len(success_ids), len(deadlock_ids)))

    planned: list[dict] = []
    # Retain the 24 v1 late-normal states exactly.
    for rid in success_ids[:24]:
        terminal = int(by_id[rid]["episode_steps"])
        planned.append({
            "state_id": f"N_r{rid:03d}_m120", "category": "NORMAL", "normal_phase": "late",
            "source_type": "baseline", "source_path": str(BASE / f"rollout_{rid:04d}.npz"),
            "source_trajectory": f"baseline_r{rid:03d}", "leakage_group": f"baseline_r{rid:03d}",
            "source_seed": 42, "source_rng_id": rid, "source_local_step": terminal - 120,
            "absolute_step": terminal - 120, "retained_from_v1": True,
        })
    # Add 96 distinct successful trajectories with early/mid/late coverage.
    for rank, rid in enumerate(success_ids[24:120]):
        terminal = int(by_id[rid]["episode_steps"])
        step, phase = phase_step(terminal, rank)
        if step >= terminal:
            raise AssertionError((rid, terminal, step))
        planned.append({
            "state_id": f"N_r{rid:03d}_s{step:03d}", "category": "NORMAL", "normal_phase": phase,
            "source_type": "baseline", "source_path": str(BASE / f"rollout_{rid:04d}.npz"),
            "source_trajectory": f"baseline_r{rid:03d}", "leakage_group": f"baseline_r{rid:03d}",
            "source_seed": 42, "source_rng_id": rid, "source_local_step": step,
            "absolute_step": step, "retained_from_v1": False,
        })

    # Five separated lead times from each of seven baseline-deadlock traces.
    for rid in deadlock_ids:
        terminal = int(by_id[rid]["episode_steps"])
        for offset in PRE_OFFSETS:
            step = terminal - offset
            if step < 41:
                raise AssertionError((rid, terminal, offset, step))
            planned.append({
                "state_id": f"P_r{rid:03d}_m{offset:03d}", "category": "PRE_DEADLOCK",
                "source_type": "baseline", "source_path": str(BASE / f"rollout_{rid:04d}.npz"),
                "source_trajectory": f"baseline_r{rid:03d}", "leakage_group": f"baseline_r{rid:03d}",
                "source_seed": 42, "source_rng_id": rid, "source_local_step": step,
                "absolute_step": step, "lead_steps_to_source_deadlock": offset,
                "retained_from_v1": (offset in (40, 80, 160) and (rid != deadlock_ids[-1] or offset != 40)),
            })

    # Five lead times from each qualification deadlock trace.  Reuse the exact
    # D8/D6/D4/D2/D1 catalog identity at its canonical lead time.
    qual = qualification_sources()
    catalog_by_pair = {
        int(row["pair_id"]): row
        for row in json.loads(CATALOG.read_text()) if row["source_outcome"] == "deadlock"
    }
    for pair_id in sorted(set(qual) & set(catalog_by_pair)):
        record = qual[pair_id]
        terminal = int(record["steps"])
        canonical = catalog_by_pair[pair_id]
        canonical_offset = terminal - int(canonical["start_step"])
        source_path = QUAL / record["relative_path"]
        for offset in QUAL_OFFSETS:
            step = terminal - offset
            if step < 41:
                raise AssertionError((pair_id, terminal, offset, step))
            is_canonical = offset == canonical_offset
            state_id = canonical["state_id"] if is_canonical else f"Q_pair{pair_id}_m{offset:03d}"
            canonical_state = canonical["state_id"]
            group = f"anchor_{canonical_state}" if canonical_state in ANCHOR_BEST else f"qual_pair{pair_id}"
            planned.append({
                "state_id": state_id, "category": "PRE_DEADLOCK", "source_type": "qualification",
                "source_path": str(source_path), "source_trajectory": f"qual_pair{pair_id}",
                "leakage_group": group,
                "source_seed": int(record["flow_seed"]), "source_rng_id": pair_id,
                "source_local_step": step, "absolute_step": step,
                "lead_steps_to_source_deadlock": offset,
                "retained_from_v1": state_id in ANCHOR_BEST,
            })

    # Add three source-diverse recovery cohorts.  Exact v1 offsets are retained
    # for their first seeds; additions span onset/active/post-clearance phases.
    _, effective, _, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    inventory = defaultdict(list)
    for row in effective:
        inventory[(row["state_id"], eta_key(row["eta"]))].append(row)
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
            anchor_env = replay_actions(
                np.asarray(source["positions_before"][0]), np.asarray(source["u_exec"]), anchor_step, config)
        for index, row in enumerate(options):
            with np.load(row["path"]) as source:
                length = len(source["u_exec"])
                if index < V1_RECOVERY_COUNTS[anchor]:
                    local = V1_RECOVERY_OFFSETS[index]
                    phase = "v1_progressive"
                else:
                    phase_index = (index - V1_RECOVERY_COUNTS[anchor]) % 3
                    fractions = (0.08, 0.40, 0.72)
                    phase = ("onset", "active", "post_clearance")[phase_index]
                    local = max(5, min(length - 1, int(round(fractions[phase_index] * length))))
                env = clone_env(anchor_env, config)
                for j, action in enumerate(np.asarray(source["u_exec"][:local])):
                    if not np.allclose(env.positions, source["positions_before"][j], atol=1e-12, rtol=0):
                        raise AssertionError((anchor, row["seed"], j, "recovery replay mismatch"))
                    _, _, done, _ = env.step(action)
                    if done:
                        raise AssertionError((anchor, row["seed"], j, "premature recovery termination"))
            retained = index < V1_RECOVERY_COUNTS[anchor]
            local_suffix = str(local) if retained else f"{local:03d}"
            planned.append({
                "state_id": f"R_{anchor.split('_')[0]}_s{int(row['seed'])}_p{local_suffix}",
                "category": "RECOVERY", "recovery_phase": phase, "source_type": "recovery",
                "source_path": str(row["path"]),
                "source_trajectory": f"{anchor}_eta{eta}_seed{int(row['seed'])}",
                "leakage_group": f"anchor_{anchor}", "source_seed": int(row["seed"]),
                "source_rng_id": pair_id, "source_local_step": local,
                "absolute_step": anchor_step + local, "anchor_state": anchor,
                "source_eta": list(eta), "prebuilt_env": env,
                "retained_from_v1": retained,
            })

    expected = Counter({"NORMAL": 120, "PRE_DEADLOCK": 60, "RECOVERY": 66})
    if Counter(row["category"] for row in planned) != expected:
        raise AssertionError(Counter(row["category"] for row in planned))
    if len({row["state_id"] for row in planned}) != len(planned):
        duplicates = [key for key, value in Counter(row["state_id"] for row in planned).items() if value > 1]
        raise AssertionError(("duplicate state ids", duplicates))

    v1_namespace = {
        row["state_id"]: int(row["rng_namespace"])
        for row in (json.loads(line) for line in (V1 / "state_manifest.jsonl").read_text().splitlines())
    }
    manifest = []
    for index, spec in enumerate(planned):
        source_path = Path(spec["source_path"])
        if spec["source_type"] == "baseline":
            with np.load(source_path) as source:
                env = replay_actions(
                    np.asarray(source["initial_positions"]), np.asarray(source["executed_velocity"]),
                    int(spec["source_local_step"]), config)
        elif spec["source_type"] == "qualification":
            with np.load(source_path) as source:
                env = replay_actions(
                    np.asarray(source["positions_before"][0]), np.asarray(source["u_exec"]),
                    int(spec["source_local_step"]), config)
        else:
            env = spec.pop("prebuilt_env")
        if env.done:
            raise AssertionError((spec["state_id"], "selected terminal state"))
        state_path = states_dir / f"{spec['state_id']}.npz"
        save_full(state_path, env)
        payload = source_step_payload(
            "baseline" if spec["source_type"] == "baseline" else "corrected",
            source_path, int(spec["source_local_step"]))
        ref_path = refs_dir / f"{spec['state_id']}.npz"
        np.savez_compressed(ref_path, **payload)
        manifest.append({
            **spec, "state_index": index,
            # Preserve matched-randomness identity for all retained v1 tuples;
            # new namespaces are disjoint from the old 100000-series.
            "rng_namespace": v1_namespace.get(spec["state_id"], 200000 + index),
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
        "assignment_unit": "leakage_group/source trajectory; all common-anchor recovery and pre-deadlock descendants grouped",
        "v1_state_memberships_preserved": True,
        "state_ids": {split: [row["state_id"] for row in manifest if row["split"] == split]
                      for split in ("train", "validation", "test")},
        "leakage_groups": {split: sorted({row["leakage_group"] for row in manifest if row["split"] == split})
                           for split in ("train", "validation", "test")},
        "category_counts": {split: dict(Counter(row["category"] for row in manifest if row["split"] == split))
                            for split in ("train", "validation", "test")},
        "retained_v1_counts": {split: sum(row["retained_from_v1"] and row["split"] == split for row in manifest)
                               for split in ("train", "validation", "test")},
    }
    (HERE / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")

    protocol = {
        "study": "gphi_training_dataset_v2", "training_performed": False,
        "state_selection_frozen_before_oracle": True,
        "state_count": len(manifest), "category_counts": dict(Counter(row["category"] for row in manifest)),
        "retained_v1_state_count": sum(row["retained_from_v1"] for row in manifest),
        "baseline_source": str(BASE), "baseline_seed": 42,
        "environment": config.to_dict(),
        "checkpoint": str(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"),
        "checkpoint_sha256": sha(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"),
        "oracle_seed_default": list(range(95210001, 95210065)),
        "oracle_rule": "success_count >= 63/64; stop after second physical failure",
        "candidate_etas": [
            [0.5703125, -0.375, -0.125], [0.578125, -0.40625, -0.125],
            [0.578125, -0.34375, -0.125], [0.40625, -0.5, 0.0],
            [0.4375, -0.53125, 0.0], [0.375, -0.4375, -0.0625],
            [0.40625, -0.4375, -0.0625], [1.0, 0.0, 0.25],
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
    print(json.dumps({
        "states": len(manifest), "categories": protocol["category_counts"],
        "splits": {key: len(value) for key, value in split_manifest["state_ids"].items()},
        "split_categories": split_manifest["category_counts"],
        "retained_v1": protocol["retained_v1_state_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
