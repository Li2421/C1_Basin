"""Freeze a generic, label-blind startup-state cohort before oracle rollout.

The frozen rule is one state from each of baseline rollout IDs 0..122.  The
physical step is ``rollout_id mod 41``, giving exactly three independent source
episodes at every startup step 0..40.  No terminal outcome, oracle result,
geometry, or prior gate error participates in selection.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    BASE, BASE_CONFIG, FROZEN_FILES, HERE, ROOT, SYSROOT, V3,
    assert_frozen_sources, sha256, write_json, write_jsonl,
)


STATE_COUNT = 123
STARTUP_LAST_STEP = 40
RNG_NAMESPACE_BASE = 410_000


def _save_full(path: Path, env) -> None:
    # Keep the real, unpadded causal history in the restorable snapshot.  The
    # isolated startup-aware FeatureBuilder owns left-padding at feature time.
    history = np.asarray(env.distance_history[-41:], dtype=np.float64).copy()
    np.savez_compressed(
        path,
        positions=env.positions,
        velocities=env.velocities,
        step=np.asarray(env.step_count),
        error_history=history,
        history_start_step=np.asarray(env.step_count - len(history) + 1),
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


def _assign_new_groups(existing: dict[str, str], new_groups: list[str]) -> dict[str, str]:
    """Greedily approach 70/15/15 across the combined V3 group inventory."""
    target = {"train": 0.70, "validation": 0.15, "test": 0.15}
    counts = Counter(existing.values())
    assignment: dict[str, str] = {}
    final_total = len(existing) + len(new_groups)
    desired = {split: target[split] * final_total for split in target}
    for group in sorted(new_groups):
        split = max(target, key=lambda item: (desired[item] - counts[item], item))
        assignment[group] = split
        counts[split] += 1
    return assignment


def main() -> None:
    observed_hashes = assert_frozen_sources()
    if len(list(BASE.glob("rollout_*.npz"))) < STATE_COUNT:
        raise RuntimeError("baseline cache does not contain the frozen 123 episodes")

    # Imports occur only after byte-level provenance is accepted.
    sys.path.insert(0, str(SYSROOT))
    from single_integrator.environment import Config, GiveWayEnv

    HERE.mkdir(parents=True, exist_ok=True)
    # Safe unattended restart: accept only a complete, hash-consistent frozen
    # cohort.  Any partial/inconsistent directory still fails closed below.
    existing_manifest = HERE / "startup_state_manifest.jsonl"
    if existing_manifest.exists() and (HERE / "protocol.json").exists():
        existing = [json.loads(line) for line in existing_manifest.read_text().splitlines() if line]
        valid = (
            len(existing) == STATE_COUNT
            and [row["source_rollout_id"] for row in existing] == list(range(STATE_COUNT))
            and all(row["step"] == row["source_rollout_id"] % 41 for row in existing)
            and all((HERE / row["state_file"]).is_file() and sha256(HERE / row["state_file"]) == row["state_sha256"] for row in existing)
        )
        if not valid:
            raise RuntimeError("pre-existing frozen startup cohort is incomplete/inconsistent")
        print(json.dumps({"status": "RESUME_EXISTING_FROZEN_COHORT", "states": len(existing)}, indent=2))
        return
    states_dir = HERE / "states"
    if states_dir.exists() and any(states_dir.iterdir()):
        raise RuntimeError("states/ is nonempty; refusing to overwrite frozen cohort")
    states_dir.mkdir(exist_ok=True)

    v3_protocol = json.loads((V3 / "protocol.json").read_text())
    config = Config(**v3_protocol["environment"])
    v3_states = [json.loads(line) for line in (V3 / "state_manifest.jsonl").read_text().splitlines() if line]
    existing_split = {row["leakage_group"]: row["split"] for row in v3_states}
    selected_groups = [f"baseline_r{rid:03d}" for rid in range(STATE_COUNT)]
    new_groups = [group for group in selected_groups if group not in existing_split]
    new_assignment = _assign_new_groups(existing_split, new_groups)

    rows = []
    for rollout_id in range(STATE_COUNT):
        source = BASE / f"rollout_{rollout_id:04d}.npz"
        step = rollout_id % (STARTUP_LAST_STEP + 1)
        with np.load(source, allow_pickle=False) as data:
            initial = np.asarray(data["initial_positions"], dtype=np.float64)
            actions = np.asarray(data["executed_velocity"], dtype=np.float64)
            if len(actions) <= STARTUP_LAST_STEP:
                raise RuntimeError((source, len(actions)))
            env = GiveWayEnv(config)
            env.reset(initial)
            for action in actions[:step]:
                _, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError((rollout_id, step, "source terminated during startup"))
            group = f"baseline_r{rollout_id:03d}"
            # Do not use dict.get(..., new_assignment[group]): Python eagerly
            # evaluates the default and would KeyError for inherited groups.
            split = existing_split[group] if group in existing_split else new_assignment[group]
            state_id = f"S_r{rollout_id:03d}_p{step:02d}"
            state_path = states_dir / f"{state_id}.npz"
            _save_full(state_path, env)
            row = {
                "state_id": state_id,
                "state_index_startup": rollout_id,
                "category": "STARTUP",
                "physical_step": step,
                "step": step,
                "source_type": "frozen_safety_baseline_episode",
                "source_episode": f"baseline_r{rollout_id:03d}",
                "source_trajectory": f"baseline_r{rollout_id:03d}",
                "source_group": group,
                "leakage_group": group,
                "source_rollout_id": rollout_id,
                "source_seed": 42,
                "source_rng_id": rollout_id,
                "source_path": str(source),
                "source_sha256": sha256(source),
                "rng_namespace": RNG_NAMESPACE_BASE + rollout_id,
                "real_history_length": step + 1,
                "left_padding_count": 40 - step,
                "history_rule": "left-pad earliest available goal error to length 41",
                "split": split,
                "state_file": str(state_path.relative_to(HERE)),
                "state_sha256": sha256(state_path),
                "selection_rule": "rollout_id in [0,122]; physical_step=rollout_id mod 41",
            }
            rows.append(row)

    write_jsonl(HERE / "startup_state_manifest.jsonl", rows)
    # Canonical oracle runner manifest contains startup states only.  The V3
    # states are merged only after startup labels pass all gates.
    write_jsonl(HERE / "state_manifest.jsonl", rows)
    with (HERE / "startup_states.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    protocol = {
        "study": "startup_complete_gphi_dataset_v1",
        "state_selection_frozen_before_oracle": True,
        "selection_rule": "baseline rollout IDs 0..122; one state/episode; step=rollout_id mod 41",
        "selection_uses_terminal_outcome": False,
        "selection_uses_oracle_or_gate_results": False,
        "startup_state_count": len(rows),
        "states_per_physical_step": {str(step): sum(row["step"] == step for row in rows) for step in range(41)},
        "source_episode_count": len({row["source_episode"] for row in rows}),
        "environment": v3_protocol["environment"],
        "checkpoint": v3_protocol["checkpoint"],
        "checkpoint_sha256": v3_protocol["checkpoint_sha256"],
        "oracle_seed_default": v3_protocol["oracle_seed_default"],
        "candidate_etas": v3_protocol["candidate_etas"],
        "oracle_rule": "B_63 is >=63 successes in the complete matched 64-seed cohort; exact stop after second physical failure",
        "target_semantics": "mean executed first-step correction across compact E_near, exactly as V3",
        # Preserve the V3 provenance contract consumed by downstream training.
        "frozen_hashes": {
            "environment": observed_hashes[str(SYSROOT / "single_integrator/environment.py")],
            "projection": observed_hashes[str(SYSROOT / "single_integrator/cbf.py")],
            "corrector": observed_hashes[str(ROOT / "diagnostics/cl_fhcb/closed_loop.py")],
            "retry": observed_hashes[str(ROOT / "diagnostics/success_basin_multimodality/exact_projector.py")],
            "checkpoint": observed_hashes[str(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")],
        },
        "base_v3_manifest_sha256": sha256(V3 / "manifest.json"),
        "base_v3_samples_sha256": sha256(V3 / "samples.npz"),
        "base_v3_state_manifest_sha256": sha256(V3 / "state_manifest.jsonl"),
        "resource_plan_not_execution_authorization": {"gpu_shards": "1 normally; up to policy limit", "cpu_workers": 4},
    }
    write_json(HERE / "protocol.json", protocol)
    write_json(HERE / "frozen_source_audit.json", {
        "status": "PASS", "absolute_path_sha256": observed_hashes,
    })
    write_json(HERE / "startup_selection_audit.json", {
        "passed": True,
        "states": len(rows),
        "source_episodes": len({row["source_episode"] for row in rows}),
        "step_counts": dict(Counter(row["step"] for row in rows)),
        "startup_split_state_counts": dict(Counter(row["split"] for row in rows)),
        "combined_v3_plus_startup_state_counts": {
            split: sum(row["split"] == split for row in v3_states) + sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "combined_group_counts": dict(Counter({**existing_split, **new_assignment}.values())),
        "no_episode_split_overlap": all(len({row["split"] for row in rows if row["source_group"] == group}) == 1 for group in selected_groups),
    })
    print(json.dumps({"states": len(rows), "startup_split_counts": dict(Counter(row["split"] for row in rows)), "new_groups": len(new_groups)}, indent=2))


if __name__ == "__main__":
    main()
