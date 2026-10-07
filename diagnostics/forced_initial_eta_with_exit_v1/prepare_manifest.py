"""Freeze a new development-only authoritative WIDE cohort before rollout."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/forced_initial_eta_with_exit_v1"
SOURCE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = Path("/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/planning/wide_initial_states_200.npz")
MANIFEST = HERE / "fresh_dev_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
IC_SEED = 2026120101
FLOW_SEED = 2026120102
N = 200


spec = importlib.util.spec_from_file_location("wide_freezer", SOURCE)
wide = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(wide)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() or OVERLAP.exists() or list((HERE / "runs").rglob("episode_*.json")):
        raise RuntimeError("fresh manifest must be frozen exactly once before raw rollout output")
    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as archive:
        historical = np.asarray(archive["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(wide.generate(2026090902, 200), historical):
        raise RuntimeError("authoritative WIDE generator replay mismatch")
    initials = wide.generate(IC_SEED, N)
    if len({row.tobytes() for row in initials}) != N:
        raise RuntimeError("duplicate new IC")
    prior, identities, _, prior_ids, audited_assets = wide.collect_prior()
    distances = np.linalg.norm(
        initials[:, None].astype(np.float64) - prior[None].astype(np.float64), axis=(2, 3)
    )
    seed_values = {value for _, value, _ in identities}
    ids = {f"forced_initial_eta_exit_dev_v1_{index:04d}" for index in range(N)}
    seed_collisions = sorted(seed_values.intersection({IC_SEED, FLOW_SEED}))
    id_collisions = sorted(ids.intersection(prior_ids))
    exact = np.argwhere(distances == 0)
    if len(exact) or seed_collisions or id_collisions:
        raise RuntimeError(("fresh development overlap", exact.tolist(), seed_collisions, id_collisions))
    frozen = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "forced_initial_eta_exit_overlap_v1", "status": "PASS",
        "development_only": True, "episodes": N,
        "exact_initial_state_match_count": 0,
        "minimum_l2_distance_to_prior_initial": float(distances.min()),
        "prior_initial_records_checked": int(len(prior)),
        "prior_assets_checked": len(set(audited_assets)),
        "numeric_seed_values_checked": len(identities),
        "seed_collisions": [], "source_id_collisions": [],
        "ic_root_seed": IC_SEED, "flow_root_seed": FLOW_SEED,
        "no_rejection_or_outcome_filtering": True,
        "completed_before_rollout": True, "frozen_utc": frozen,
    }
    overlap["content_sha256"] = wide.canonical_hash(overlap)
    wide.atomic_json(OVERLAP, overlap)
    manifest = {
        "schema": "forced_initial_eta_exit_fresh_development_manifest_v1",
        "status": "FROZEN_BEFORE_ANY_ROLLOUT", "development_only": True,
        "frozen_utc": frozen, "raw_rollout_records_present_at_freeze": 0,
        "episode_count": N,
        "generator": {
            "implementation": "numpy.default_rng; abs_x U(0.55,1.05); y U(-0.025,0.025); signs [-,+]; float32",
            "ic_root_seed": IC_SEED, "authoritative_replay_bitwise_exact": True,
            "authoritative_asset": str(REFERENCE_ASSET),
            "authoritative_asset_sha256": wide.sha256(REFERENCE_ASSET),
            "script": str(Path(__file__).resolve()),
        },
        "flow": {
            "root_seed": FLOW_SEED,
            "semantics": f"episode_key=fold_in(PRNGKey({FLOW_SEED}),rollout_id); step_key=fold_in(episode_key,absolute_step)",
            "matched_across_all_controllers": True,
        },
        "controllers": ["Safety", "Forced-Eta+Exit", "Learned-Entry+Exit reference", "Direct-g H8 reference"],
        "environment": reference["environment"], "cbf": reference["cbf"],
        "outcome_protocol": reference["outcome_protocol"],
        "horizon_steps": 850, "dt_seconds": 0.05,
        "overlap_audit_path": str(OVERLAP), "overlap_audit_sha256": wide.sha256(OVERLAP),
        "episodes": [{
            "episode_index": index, "source_id": f"forced_initial_eta_exit_dev_v1_{index:04d}",
            "ic_root_seed": IC_SEED, "ic_draw_index": index, "rollout_id": index,
            "flow_root_seed": FLOW_SEED, "initial_positions": value.tolist(),
        } for index, value in enumerate(initials)],
    }
    manifest["content_sha256"] = wide.canonical_hash(manifest)
    wide.atomic_json(MANIFEST, manifest)
    print(json.dumps({"status": "FROZEN", "manifest_sha256": wide.sha256(MANIFEST),
                      "content_sha256": manifest["content_sha256"], "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
