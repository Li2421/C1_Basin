"""Reuse exact archived controller-swap evidence and audit present contracts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "orthoflow3_controller_conditioning_probe_v1"
OUT = HERE / "posthoc_cause_audit"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = read(OLD / "protocol.json")
    manifest = read(OLD / "pair_manifest.json")
    db = sqlite3.connect(f"file:{ROOT / 'shared_rollout_db/rollout.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    evidence = []
    for pair in manifest:
        rows = []
        rollouts = []
        for key in ("base", "alternate"):
            rr = db.execute("SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? "
                            "AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE'",
                            (pair["state_uid"], pair["eta_uid"], p[key+"_controller_uid"])).fetchall()
            by_seed = {json.loads(r["seed_key"])["future_index"]: r for r in rr
                       if json.loads(r["seed_key"]).get("future_index") in range(16)}
            assert len(by_seed) == 16
            assert all(not r["numerical_failure"] and not r["collision"] for r in by_seed.values())
            rows.append(np.array([by_seed[k]["success"] for k in range(16)], int))
            rollouts.append([by_seed[k]["rollout_uid"] for k in range(16)])
        a, b = rows
        n10, n01 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        evidence.append({"state_uid": pair["state_uid"], "eta_uid": pair["eta_uid"],
                         "eta": pair["eta"], "kind": pair["kind"],
                         "base_success": int(a.sum()), "alternate_success": int(b.sum()),
                         "paired_seed_p": binomtest(n01, n01+n10).pvalue if n01+n10 else 1.,
                         "base_rollout_uids": rollouts[0], "alternate_rollout_uids": rollouts[1]})
    db.close()
    last = 0.
    for rank, i in enumerate(np.argsort([r["paired_seed_p"] for r in evidence])):
        last = max(last, min(1., evidence[i]["paired_seed_p"]*(len(evidence)-rank)))
        evidence[i]["holm_p"] = float(last)
    current = read(HERE.parent / "physical_state_critic_formal_v1/loso_protocol.json")
    contracts = {}
    for scene, c in current["controllers"].items():
        cp = c["intervention_payload"]
        contracts[scene] = {"controller_uid": c["intervention_uid"],
                            "flow_sha256": cp.get("flow_checkpoint_sha256", cp.get("flow_sha256")),
                            "horizon": cp["horizon"], "dt": cp["dt"],
                            "success_semantics": cp["success_semantics"],
                            "safety_sha_or_rule": cp.get("safety_projection_sha256", cp.get("safety")),
                            "critic_has_controller_input": False}
    out = {"schema": "reused_controller_information_counterexample_v1", "new_rollouts": 0,
           "reused_seed_records": len(evidence)*32, "independent_physical_states": len({r['state_uid'] for r in evidence}),
           "state_eta_pairs": len(evidence),
           "16of16_to_0of16_pairs": sum(r["base_success"] == 16 and r["alternate_success"] == 0 for r in evidence),
           "holm_significant_pairs": sum(r["holm_p"] < .05 for r in evidence),
           "same_current_h_max_error": read(OLD / "summary.json")["same_h_max_abs_error"],
           "physical_only_input_identity": "same physical state UID and eta; deterministic physical-only input cannot distinguish future controller",
           "controller_intervention": p["design"],
           "scope": "Identifiability across an admissible controller family; not attribution of all present four-scene LOSO loss",
           "current_four_scene_contracts": contracts, "pairs": evidence}
    OUT.mkdir(exist_ok=True)
    (OUT / "controller_information.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "current_four_scene_contracts")}, indent=2))


if __name__ == "__main__":
    main()
