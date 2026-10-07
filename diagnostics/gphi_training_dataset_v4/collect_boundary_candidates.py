"""Mine one close-range recovery candidate from every reusable V3 source trace."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v2.build_states import restore_full, save_full
from single_integrator.environment import Config


PHASE_FRACTIONS = (0.18, 0.25, 0.32, 0.38, 0.42, 0.46, 0.50, 0.54)
CLOSE_LIMIT = 0.55


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    started = time.monotonic()
    for directory in (HERE / "candidate_states", HERE / "audit_refs", HERE / "raw"):
        directory.mkdir(parents=True, exist_ok=True)
    if any((HERE / "candidate_states").iterdir()) or any((HERE / "audit_refs").iterdir()):
        raise RuntimeError("candidate outputs are nonempty; refusing overwrite")
    v3_protocol = json.loads((V3 / "protocol.json").read_text())
    config = Config(**v3_protocol["environment"])
    v3_sources = read_jsonl(V3 / "candidate_state_manifest.jsonl")
    v3_states = {row["state_id"]: row for row in read_jsonl(V3 / "state_manifest.jsonl")}
    rows = []
    for index, source_row in enumerate(v3_sources):
        trajectory_rank = int(source_row["candidate_rank_within_group"])
        fraction = PHASE_FRACTIONS[trajectory_rank]
        source_path = V3 / source_row["source_path"]
        with np.load(source_path) as source:
            actions = np.asarray(source["u_exec"])
            positions = np.asarray(source["positions_before"])
            distances = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)
            length = len(actions)
            desired = int(round(fraction * length))
            eligible = np.flatnonzero((distances <= CLOSE_LIMIT) & (np.arange(length) >= 10) & (np.arange(length) < length - 2))
            if not len(eligible):
                raise RuntimeError((source_row["source_trajectory"], "no close-range state"))
            local = int(eligible[np.argmin(np.abs(eligible - desired))])
            anchor = v3_states[source_row["anchor_state"]]
            env = restore_full(V3 / anchor["state_file"], config)
            for action in actions[:local]:
                _, _, done, _ = env.step(action)
                if done:
                    raise AssertionError((source_row["source_trajectory"], local, "premature source termination"))
            if not np.allclose(env.positions, positions[local], atol=1e-12, rtol=0):
                raise AssertionError((source_row["source_trajectory"], local, "replay mismatch"))
            state_id = f"RB_{source_row['anchor_state']}_s{source_row['source_seed']}_p{local:03d}"
            state_path = HERE / "candidate_states" / f"{state_id}.npz"
            save_full(state_path, env)
            ref_path = HERE / "audit_refs" / f"{state_id}.npz"
            np.savez_compressed(
                ref_path,
                u_flow=np.asarray(source["u_flow"])[local],
                u_safe=np.asarray(source["u_safe"])[local],
                g_raw=np.asarray(source["g_raw"])[local],
                u_exec=np.asarray(source["u_exec"])[local],
                positions_before=positions[local],
                positions_after=np.asarray(source["positions_after"])[local],
            )
        rows.append({
            "state_id": state_id, "category": "RECOVERY", "boundary_candidate": True,
            "recovery_phase": "early_active" if fraction < 0.32 else "boundary_approach" if fraction < 0.46 else "boundary_late",
            "planned_phase_fraction": fraction, "actual_phase_fraction": local / length,
            "inter_agent_distance": float(distances[local]),
            "relative_velocity_norm": float(np.linalg.norm(env.velocities[1] - env.velocities[0])),
            "source_type": "reused_v3_corrected_recovery_trace",
            "source_path": str(source_path), "source_sha256": sha(source_path),
            "source_trajectory": source_row["source_trajectory"],
            "source_group": source_row["source_group"], "leakage_group": source_row["leakage_group"],
            "source_seed": source_row["source_seed"], "source_rng_id": source_row["source_rng_id"],
            "source_local_step": local, "source_length": length,
            "absolute_step": env.step_count, "anchor_state": source_row["anchor_state"],
            "anchor_step": source_row["anchor_step"], "source_eta": source_row["source_eta"],
            "anchor_rng_namespace": source_row["anchor_rng_namespace"],
            "rng_namespace": 400000 + index,
            "state_file": str(state_path.relative_to(HERE)), "state_sha256": sha(state_path),
            "audit_reference_file": str(ref_path.relative_to(HERE)), "audit_reference_sha256": sha(ref_path),
            "step": env.step_count, "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
            "ever_candidate_deadlock": env.ever_candidate_deadlock,
            "retained_from_v3": False, "provisional_split": anchor["split"],
        })
    if len(rows) != 96 or len({row["source_trajectory"] for row in rows}) != 96:
        raise AssertionError((len(rows), len({row["source_trajectory"] for row in rows})))
    write_jsonl(HERE / "candidate_state_manifest.jsonl", rows)
    protocol = {
        **v3_protocol,
        "study": "gphi_training_dataset_v4_recovery_boundary_coverage",
        "state_selection_frozen_before_oracle": True,
        "candidate_state_count": len(rows),
        "candidate_source_group_count": len({row["source_group"] for row in rows}),
        "candidate_phase_fractions": list(PHASE_FRACTIONS),
        "candidate_close_range_collection_limit_m": CLOSE_LIMIT,
        "oracle_seed_default": list(range(95910001, 95910065)),
        "oracle_rule": "success_count >= 63/64; eta zero first; stop at second physical failure",
        "resource_plan": {"GPU_shards": 2, "CPU_workers": 8, "reason": "GPU was at 2 MiB and 0% utilization; user explicitly selected the light-load two-shard allowance"},
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    distances = np.asarray([row["inter_agent_distance"] for row in rows])
    print(json.dumps({
        "candidates": len(rows), "source_groups": len({row["source_group"] for row in rows}),
        "source_trajectories": len({row["source_trajectory"] for row in rows}),
        "distance_min_median_max": [float(distances.min()), float(np.median(distances)), float(distances.max())],
        "elapsed_s": time.monotonic() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
