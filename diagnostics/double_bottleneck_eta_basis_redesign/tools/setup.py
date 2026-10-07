#!/usr/bin/env python3
"""Pre-register the frozen P0 versus OrthoFlow3 basis experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
HARD = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
EXPECTED = {
    "checkpoint": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl", "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd"),
    "dataset_manifest": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json", "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56"),
    "macflow": (ROOT / "double_bottleneck/flowbc_4a_agent.py", "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"),
    "environment": (ROOT / "double_bottleneck/environment.py", "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc"),
    "hard_projection": (ROOT / "shared_control/hard_projection.py", "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79"),
    "canonical_p0": (ROOT / "shared_control/diagnostic_corrector.py", "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40"),
    "episode_catalog": (ETA3 / "episode_catalog.json", "ddb1e32646fd771e84fd4cce024e7bcf416f26a3a08c75557f6bb94c8b7cfe5c"),
    "eta_points": (ETA3 / "eta_points.json", "6a0732c5ad35bc2000e61be6c2825034058b30aad09458cf2cf5975cc501f258"),
    "hard_safety_summary": (HARD / "hard_safety_summary.json", "4cba11fc33387675085a6c994c223b7e54c2db0cce0ddabe5d658277a22049da"),
    "p0_full_basin": (ETA3 / "long_run_v2/timeout_basin_matrix.json", "0951e015531505bdbaa6d604c56879784862fa295b608814c1b3824cc1d2c654"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        digest.update(str(item.relative_to(path)).encode() + b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def trace_action_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as trace:
        return len(trace["positions"]) - 1


def main() -> int:
    hashes = {}
    for name, (path, expected) in EXPECTED.items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen mismatch {name}: {actual} != {expected}")
        hashes[name] = actual
    hashes["toy_giveway_source_tree"] = tree_sha(ROOT / "toy_giveway")
    if hashes["toy_giveway_source_tree"] != "10c8ac15a724ef0c80d8c0e4c0de63d0b28fd89a7956c030734e8a43bd8b4a26":
        raise RuntimeError("Toy Give-Way source tree mismatch")

    catalog = json.loads((ETA3 / "episode_catalog.json").read_text())["episodes"]
    targets = [row for row in catalog if row["population"] == "safe_timeout_target"]
    controls = [row for row in catalog if row["population"] == "baseline_success_control"]
    if len(targets) != 61 or len(controls) != 24:
        raise RuntimeError("frozen evaluation population mismatch")

    selection_seed = 2026092701
    rng = np.random.default_rng(selection_seed)
    pilot_targets = [targets[index] for index in sorted(rng.choice(len(targets), 12, replace=False).tolist())]
    pilot_controls = [controls[index] for index in sorted(rng.choice(len(controls), 12, replace=False).tolist())]
    pilot = pilot_targets + pilot_controls
    (OUT / "pilot_catalog.json").write_text(json.dumps({
        "schema": "eta_basis_redesign_pilot_v1",
        "selection_seed": selection_seed,
        "selection": "uniform without replacement over all 61 timeouts and independently over all 24 controls; no P0 label used",
        "episodes": pilot,
    }, indent=2, sort_keys=True) + "\n")

    anchor_seed = 2026092702
    anchor_rng = np.random.default_rng(anchor_seed)
    anchors = []
    for episode in pilot:
        path = HARD / "trajectories/hard_safety" / episode["set"] / f"rollout_{episode['rollout_id']:03d}.npz"
        count = trace_action_count(path)
        for ordinal, step in enumerate(sorted(anchor_rng.choice(count, 4, replace=False).tolist())):
            anchors.append({
                **episode,
                "state_id": f"{episode['episode_id']}|baseline|S{ordinal}",
                "trajectory_class": "baseline_success" if episode["population"] == "baseline_success_control" else "safe_timeout",
                "trajectory_path": str(path.relative_to(ROOT)),
                "source_step": int(step),
                "source_action_count": count,
            })

    eta_metadata = json.loads((ETA3 / "representatives/metadata.json").read_text())["episodes"]
    eta_success = [row for row in eta_metadata if row["traces"]["eta_success"]["terminal"] == "success"]
    episode_lookup = {row["episode_id"]: row for row in catalog}
    for representative in eta_success:
        episode = episode_lookup[representative["episode_id"]]
        path = ROOT / representative["traces"]["eta_success"]["npz"]
        count = trace_action_count(path)
        for ordinal, step in enumerate(sorted(anchor_rng.choice(count, 4, replace=False).tolist())):
            anchors.append({
                **episode,
                "state_id": f"{episode['episode_id']}|eta_success|S{ordinal}",
                "trajectory_class": "successful_eta_corrected",
                "trajectory_path": str(path.relative_to(ROOT)),
                "source_step": int(step),
                "source_action_count": count,
                "eta_theta": representative["traces"]["eta_success"]["theta"],
            })
    if len(anchors) != 108:
        raise RuntimeError(f"unexpected anchor count {len(anchors)}")
    (OUT / "state_anchor_manifest.json").write_text(json.dumps({
        "schema": "eta_basis_redesign_state_anchors_v1",
        "seed": anchor_seed,
        "rule": "exactly four action-state indices uniformly without replacement per stored trajectory; no phase rejection or resampling",
        "calibration_subset": "96 hard-safety baseline anchors only; 12 eta-success anchors are audit-only",
        "anchors": anchors,
    }, indent=2, sort_keys=True) + "\n")

    protocol = {
        "schema": "double_bottleneck_eta_basis_redesign_protocol_v1",
        "registered_at": "2026-09-27T00:00:00+08:00",
        "frozen_hashes": hashes,
        "p0_exact": "g = eta_g*bounded(goal-position) + eta_s*u_safe + eta_r*all_pair_mean_relative_basis",
        "p1_definition": {
            "name": "P1-OrthoFlow3",
            "raw_per_agent": "u_safe - dot(u_safe,B_goal)/(dot(B_goal,B_goal)+epsilon^2)*B_goal",
            "epsilon": "1e-6 * max_speed; smooth Tikhonov denominator, no invented lateral vector",
            "global_scale_rule": "sqrt(mean(||u_safe||^2)/mean(||B_flow_perp_raw||^2)) over all agent rows in the 96 baseline anchors",
            "correction": "eta_g*B_goal + eta_perp*scale*B_flow_perp_raw + eta_r*B_rel",
            "degenerate_threshold": "||scale*B_flow_perp_raw|| <= 1e-6 m/s",
        },
        "basis_audit": {
            "pilot_timeout_trajectories": 12,
            "pilot_success_trajectories": 12,
            "eta_success_trajectories": len(eta_success),
            "anchors_per_trajectory": 4,
            "rank_threshold": "max(1e-3, 0.01*sigma_max)",
            "cosine_zero_threshold": 1e-12,
            "finite_difference_normalized_step": 0.015625,
            "sensitivity_points": ["eta=0", "frozen successful P0 eta where available"],
        },
        "pilot": {
            "selection_seed": selection_seed,
            "timeouts": 12,
            "controls": 12,
            "selection": "uniform without replacement; failure/P0 basin labels excluded from selection",
        },
        "search": {
            "representations": ["P0-3D", "P1-OrthoFlow3"],
            "domain": [[0.5, 1.25], [-0.5, 0.5], [0.0, 0.75]],
            "points": 256,
            "design": "exact frozen common Sobol design sha256 6a0732...",
            "same_episode_seed": True,
            "candidate_selection": "all successful eta if <=8; otherwise top8 by successful count among 8 nearest Sobol neighbors, then distance to eta=0, then eta index",
            "candidate_seeds": list(range(2001, 2017)),
        },
        "local": {
            "trigger": "Q_max >= 0.50",
            "points": 16,
            "normalized_radius": 0.05,
            "sobol_seed": 2026092703,
            "seeds": list(range(3001, 3009)),
        },
        "controls": {"pilot_controls": 12, "seeds": [4001, 4002, 4003, 4004]},
        "full_p1_trigger": {
            "condition_1": "P1 basin existence at least P0 + 3 among 12 pilot timeouts",
            "condition_2": "P1 median Q_max at least P0 median Q_max + 0.15",
            "condition_3": "at least 3 pilot episodes reach Q_max>=0.50 under P1 but not P0",
            "rule": "run full P1 if any condition is true; otherwise stop P1 expansion",
        },
        "expert_span": {
            "states": "same 96 uniformly sampled baseline anchors",
            "expert": "unchanged centralized expert; retain only validated successful continuations",
            "candidate_points": 1024,
            "sobol_seed": 2026092704,
            "optimization": "bounded Powell from best Sobol candidate; max 400 evaluations",
        },
        "decision_options": ["KEEP-P0", "USE-ORTHOFLOW3", "TEST-AGENT6-NEXT", "BASIS-REDESIGN-NEEDED"],
        "prohibitions": ["G_phi", "Agent6 execution", "Pair8", "Temporal6", "safety change", "MACFlow change", "adaptive eta", "targeted states"],
        "artifact_hashes": {
            "pilot_catalog": sha(OUT / "pilot_catalog.json"),
            "state_anchor_manifest": sha(OUT / "state_anchor_manifest.json"),
        },
    }
    (OUT / "PREREGISTRATION.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "schema": "eta_basis_redesign_run_manifest_v1",
        "state": "preregistered",
        "pilot_episodes": 24,
        "state_anchors": len(anchors),
        "frozen_hashes": hashes,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pilot_timeouts": 12, "pilot_controls": 12, "baseline_anchors": 96, "eta_success_anchors": 12, "hashes_verified": len(hashes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
