"""Pre-outcome support distances for the independently frozen Toy cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
REP = HERE / "toy_replication"


def main() -> None:
    frozen = json.loads((REP / "frozen_proposals.json").read_text())
    historical = json.loads((Path("/home/zhihan/research/Basin_C1/diagnostics/gphi_wide_ic_cadence_v1")
                            / "frozen_benchmark_manifest.json").read_text())
    new_positions = {tuple(np.asarray(s["initial_positions"]).reshape(-1)) for s in frozen["states"]}
    old_positions = {tuple(np.asarray(s["initial_positions"]).reshape(-1)) for s in historical["episodes"]}
    learned = np.load(HERE / "toy_eta_kernel_model.npz")
    ztrain = learned["ztrain"]
    center, scale = learned["center"], learned["scale"]
    eta = np.asarray([[s["eta"][f"sample_{i}"] for i in range(16)]
                      for s in frozen["states"]], np.float64)
    z = (eta.reshape(-1, 3) - center) / scale
    near = np.min(cdist(z, ztrain), axis=1)
    stats = {
        "n_states": len(frozen["states"]),
        "n_proposals": len(z),
        "n_exact_seen_eta": int(np.sum(near < 1e-10)),
        "train_eta_count": len(ztrain),
        "unique_new_initial_states": len(new_positions),
        "exact_initial_state_overlap_with_old_200": len(new_positions & old_positions),
        "unique_new_source_groups": len({s["source_group"] for s in frozen["states"]}),
        "nearest_train_eta_normalized_distance": {
            "min": float(np.min(near)),
            "q25": float(np.quantile(near, 0.25)),
            "median": float(np.median(near)),
            "q75": float(np.quantile(near, 0.75)),
            "max": float(np.max(near)),
        },
        "cohort_rule": frozen["cohort_rule"],
        "label_status": "No outcomes used in this audit",
    }
    (REP / "proposal_support_audit.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
