"""Freeze one transparent learned eta point per uniformly sampled basin state."""

from __future__ import annotations

import hashlib, json, os, sys
from pathlib import Path
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
sys.path.insert(0, str(ROOT))
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    output = HERE / "basin_eta_hat_plan.json"
    if output.exists(): raise RuntimeError("refusing to overwrite frozen plan")
    subset = json.loads((HERE / "basin_subset_manifest.json").read_text())
    with np.load(DATA / "samples.npz", allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float64); state_ids = np.asarray(archive["state_id"]); flow_seeds = np.asarray(archive["flow_seed"])
    model = FixedDEtaPredictor(ETA / "best_fixed_d_eta_checkpoint.npz")
    strict_source = json.loads((ETA / "source_manifest.json").read_text())
    strict_rng = {row["state_id"]: int(row["rng_namespace"]) for row in strict_source["states"]}
    arms = []
    for state in subset["selected_states"]:
        indices = np.flatnonzero(state_ids == state["state_id"]); chosen = int(indices[np.argmin(flow_seeds[indices])])
        eta, raw, _ = model.predict(features[chosen:chosen + 1]); category = state["category"]
        seeds = list(range(95710001, 95710065)) if category == "STARTUP" else list(range(95310001, 95310065)) if category == "STRICT_DEADLOCK" else list(range(95210001, 95210065))
        rng_namespace = strict_rng[state["state_id"]] if category == "STRICT_DEADLOCK" else state["rng_namespace"]
        arms.append({
            "arm_id": f"ETA_HAT__{state['state_id']}", "state_id": state["state_id"], "state_file": state["state_file"],
            "state_sha256": state["state_sha256"], "absolute_step": state["absolute_step"], "rng_namespace": rng_namespace,
            "eta": eta[0].tolist(), "eta_normalized_raw": raw[0].tolist(), "eta_clipped": bool(np.any((raw[0] < 0) | (raw[0] > 1))),
            "feature_selection": "smallest flow_seed among the 64 variants", "feature_sample_index": chosen,
            "feature_flow_seed": int(flow_seeds[chosen]), "feature_sha256": hashlib.sha256(features[chosen].tobytes()).hexdigest(), "seeds": seeds,
        })
    plan = {"schema": "direct_eta_basin_eta_hat_plan_v1", "status": "FROZEN_BEFORE_ROLLOUT",
            "subset_manifest_sha256": sha(HERE / "basin_subset_manifest.json"), "checkpoint_sha256": sha(ETA / "best_fixed_d_eta_checkpoint.npz"),
            "semantics": "one eta_hat from the lowest-flow-seed feature variant, fixed across all 64 source-oracle matched futures",
            "arms": arms, "new_continuations": len(arms) * 64,
            "maximum_physical_steps": sum((850 - arm["absolute_step"]) * 64 for arm in arms)}
    plan["content_sha256"] = canonical(plan)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}"); tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n"); os.replace(tmp, output)
    print(json.dumps({"arms": len(arms), "continuations": plan["new_continuations"], "plan_sha256": sha(output)}, indent=2))


if __name__ == "__main__": main()
