"""Freeze an outcome-independent Sobol augmentation for Ring basin illustration."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MAIN_ROOT = HERE.parents[1]
POC_ROOT = MAIN_ROOT.parent / "Basin_C1_flow_field_poc_20261004"
sys.path.insert(0, str(POC_ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    benchmark = json.loads((POC_ROOT / "family_study_v1/final_benchmark.json").read_text())
    state = next(s for s in benchmark["states"] if s["uid"] == "family_v1_ring_dev_20261019")
    assert state["scenario"] == "ring_exchange"
    # Fixed before evaluating any outcomes: 64-point scrambled Sobol design.
    sobol_seed = 20261006
    unit = qmc.Sobol(d=3, scramble=True, seed=sobol_seed).random_base2(m=6)
    low = np.asarray([.5, -.5, 0.])
    high = np.asarray([1.25, .5, .75])
    eta = low + unit * (high - low)
    protocol = {
        "schema": "ring_basin_illustration_resample_v1",
        "purpose": "Illustrative 3D visual envelope only; not population inference",
        "state": state,
        "formulations": ["TT", "TF", "FT", "FF"],
        "eta_design": "64-point scrambled Sobol; fixed independently of new outcomes",
        "sobol_seed": sobol_seed,
        "eta_domain_low": low.tolist(),
        "eta_domain_high": high.tolist(),
        "eta_points": eta.tolist(),
        "seeds": list(range(16)),
        "robust_criterion": "at least 15 scientific successes of 16; numerical outcomes unresolved",
        "controller_source": "existing family_study_v1 frozen runtime/controller; no training",
        "random_key_rule": "same frozen family_study_v1 key_for(state UID, seed, physical step) across four formulations",
        "additional_continuations": 64 * 16 * 4,
    }
    encoded = json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = HERE / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("refusing to replace a different frozen protocol")
    path.write_text(encoded)
    with (HERE / "eta_design.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta_index", "eta1", "eta2", "eta3"])
        w.writerows([[i, *row] for i, row in enumerate(eta)])
    manifest = {"protocol_sha256": sha(path), "controller_sha256": sha(POC_ROOT / "validation_stage2/factorial_controller.py"),
                "runtime_sha256": sha(POC_ROOT / "validation_stage2/run_rollouts.py"),
                "runner_sha256": sha(POC_ROOT / "family_study_v1/runner.py"),
                "eta_design_sha256": sha(HERE / "eta_design.csv"), "new_continuations": 4096}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state_uid": state["uid"], "n_eta": len(eta), "continuations": 4096, "protocol_sha256": manifest["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
