"""Read-only comparison on the other task's fully frozen fresh K16 confirmation.

This script refuses to run until proposal freeze and complete seed-level
evidence exist. It never regenerates proposals or launches rollouts.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from offline_baselines import ROOT, OUT, SCENARIOS, dump_json


CONF = ROOT / "diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/confirmation"


def score_eta_only(scenario: str, xyz: np.ndarray) -> np.ndarray:
    frozen = np.load(OUT / f"{scenario}_eta_only_model.npz")
    z = (xyz - frozen["center"]) / frozen["radius"]
    k = np.exp(-cdist(z, frozen["z_train"], "sqeuclidean") / (2 * float(frozen["bandwidth"]) ** 2))
    return np.clip(float(frozen["prior"]) + k @ frozen["alpha"], 0, 1)


def main():
    proposal_file, evidence_file = CONF / "proposals.json", CONF / "seed_evidence.json"
    if not proposal_file.exists() or not evidence_file.exists():
        raise SystemExit("Independent confirmation is not complete; no evaluation performed.")
    proposals = json.loads(proposal_file.read_text())
    evidence = json.loads(evidence_file.read_text())
    assert proposals["frozen_before_outcomes"] is True
    assert not evidence["missing"], "Incomplete confirmation evidence"
    by = {(r["state_uid"], r["candidate"]): r for r in evidence["rows"]}
    detail = []
    for p in proposals["states"]:
        sid, sc = p["state_uid"], p["scenario"]
        assert sc in SCENARIOS
        xyz = np.asarray(p["etas"], float)
        assert xyz.shape == (17, 3)
        ev = [by[(sid, j)] for j in range(17)]
        scores = score_eta_only(sc, xyz)
        etidx = int(np.argmax(scores))
        cridx = int(p["critic_index"])
        oracle = True if any(e["robust"] is True for e in ev) else None if any(e["robust"] is None for e in ev) else False
        detail.append({"scenario": sc, "state_uid": sid, "oracle_b15": oracle,
                       "new_critic_index": cridx, "new_critic_b15": ev[cridx]["robust"],
                       "new_critic_q16": ev[cridx]["Q16"],
                       "eta_only_index": etidx, "eta_only_b15": ev[etidx]["robust"],
                       "eta_only_q16": ev[etidx]["Q16"],
                       "eta_only_score": float(scores[etidx]),
                       "eta_only_numerical_seeds": len(ev[etidx]["numerical_seeds"]),
                       "eta_only_collisions": ev[etidx]["collisions"]})
    with (OUT / "independent_confirmation_eta_only.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
    summary = {}
    for sc in SCENARIOS:
        rr = [r for r in detail if r["scenario"] == sc]
        summary[sc] = {"states": len(rr), "oracle_b15": sum(r["oracle_b15"] is True for r in rr),
                       "new_critic_b15": sum(r["new_critic_b15"] is True for r in rr),
                       "eta_only_b15": sum(r["eta_only_b15"] is True for r in rr),
                       "eta_only_unresolved": sum(r["eta_only_b15"] is None for r in rr),
                       "eta_only_mean_q16": float(np.mean([r["eta_only_q16"] for r in rr if r["eta_only_q16"] is not None])),
                       "eta_only_rescue_vs_new_critic": sum(r["eta_only_b15"] is True and r["new_critic_b15"] is False for r in rr),
                       "eta_only_break_vs_new_critic": sum(r["new_critic_b15"] is True and r["eta_only_b15"] is False for r in rr),
                       "eta_only_selected_collisions": sum(r["eta_only_collisions"] for r in rr),
                       "eta_only_selected_numerical_seeds": sum(r["eta_only_numerical_seeds"] for r in rr)}
    dump_json("independent_confirmation_eta_only.json", {"summary": summary,
              "proposal_source": str(proposal_file), "outcome_source": str(evidence_file),
              "control_freeze": "scenario_eta_only_manifest.json and scenario-specific npz files were finalized before confirmation outcomes",
              "new_rollout": 0})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
