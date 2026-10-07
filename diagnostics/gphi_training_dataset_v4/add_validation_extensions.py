"""Add four early D2 states so validation contains both boundary sides."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v2.build_states import restore_full, save_full
from single_integrator.environment import Config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    manifest_path = HERE / "candidate_state_manifest.jsonl"
    candidates = rows(manifest_path)
    if any(row["state_id"].startswith("RBV_") for row in candidates):
        raise RuntimeError("validation extensions already exist")
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"])
    v3_sources = [row for row in rows(V3 / "candidate_state_manifest.jsonl") if row["source_group"] == "anchor_D2_pair228"][:4]
    v3_states = {row["state_id"]: row for row in rows(V3 / "state_manifest.jsonl")}
    additions = []
    for index, (source_row, local) in enumerate(zip(v3_sources, (18, 24, 30, 36))):
        source_path = V3 / source_row["source_path"]
        anchor = v3_states[source_row["anchor_state"]]
        env = restore_full(V3 / anchor["state_file"], config)
        with np.load(source_path) as source:
            actions = np.asarray(source["u_exec"])
            for action in actions[:local]:
                env.step(action)
            positions = np.asarray(source["positions_before"])
            if not np.allclose(env.positions, positions[local], atol=1e-12, rtol=0):
                raise AssertionError("extension replay mismatch")
            state_id = f"RBV_{source_row['anchor_state']}_s{source_row['source_seed']}_p{local:03d}"
            state_path = HERE / "candidate_states" / f"{state_id}.npz"; save_full(state_path, env)
            ref_path = HERE / "audit_refs" / f"{state_id}.npz"
            np.savez_compressed(ref_path, u_flow=source["u_flow"][local], u_safe=source["u_safe"][local], g_raw=source["g_raw"][local], u_exec=source["u_exec"][local], positions_before=positions[local], positions_after=source["positions_after"][local])
        additions.append({
            "state_id": state_id, "category": "RECOVERY", "boundary_candidate": True,
            "recovery_phase": "early_active_validation_extension", "planned_phase_fraction": local / len(actions),
            "actual_phase_fraction": local / len(actions),
            "inter_agent_distance": float(np.linalg.norm(env.positions[0] - env.positions[1])),
            "relative_velocity_norm": float(np.linalg.norm(env.velocities[1] - env.velocities[0])),
            "source_type": "reused_v3_corrected_recovery_trace", "source_path": str(source_path),
            "source_sha256": sha(source_path), "source_trajectory": source_row["source_trajectory"],
            "source_group": source_row["source_group"], "leakage_group": source_row["leakage_group"],
            "source_seed": source_row["source_seed"], "source_rng_id": source_row["source_rng_id"],
            "source_local_step": local, "source_length": len(actions), "absolute_step": env.step_count,
            "anchor_state": source_row["anchor_state"], "anchor_step": source_row["anchor_step"],
            "source_eta": source_row["source_eta"], "anchor_rng_namespace": source_row["anchor_rng_namespace"],
            "rng_namespace": 401000 + index, "state_file": str(state_path.relative_to(HERE)),
            "state_sha256": sha(state_path), "audit_reference_file": str(ref_path.relative_to(HERE)),
            "audit_reference_sha256": sha(ref_path), "step": env.step_count,
            "candidate_since": -1 if env.candidate_since is None else env.candidate_since,
            "stuck_timer": env.stuck_timer, "max_stuck_timer": env.max_stuck_timer,
            "ever_candidate_deadlock": env.ever_candidate_deadlock, "retained_from_v3": False,
            "provisional_split": "validation",
        })
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates + additions))
    arms = [{"arm_id": f"{row['state_id']}__eta_zero", "state_id": row["state_id"], "eta": [0.0, 0.0, 0.0], "seeds": protocol["oracle_seed_default"]} for row in additions]
    for shard in (0, 1):
        (HERE / f"eta_zero_valext_shard{shard}_arms.json").write_text(json.dumps(arms[shard::2], indent=2) + "\n")
    # Adaptive restore lookup must include all candidates during this short add-on.
    (HERE / "state_manifest.jsonl").write_text(manifest_path.read_text())
    print(json.dumps({"added": len(additions), "source_group": "anchor_D2_pair228", "local_steps": [row["source_local_step"] for row in additions], "distances": [row["inter_agent_distance"] for row in additions]}, indent=2))


if __name__ == "__main__":
    main()
