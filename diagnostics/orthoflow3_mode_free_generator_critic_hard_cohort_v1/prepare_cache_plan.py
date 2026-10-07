#!/usr/bin/env python3
"""Global DB preflight plan for frozen difficult WIDE-IC evaluation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_rollout_db.src.rollout_db import eta_identity, uid, canonical
from shared_rollout_db.src.historical_ingest import BASIS_SHA, TOY_FLOW_SHA

OUT = Path(__file__).resolve().parent
RNG = {"name": "wide_ic_current42_future42plus_index_v1", "current_root": 42,
       "future_root_formula": "42+future_index", "rollout_id_fold_in": True,
       "current_t0_latched": True, "physical_step_fold_in": True}
SCENARIO = uid("scn", {"name": "ToyGiveWay", "code": "toy_giveway_frozen_v1"})
CORRECTION_CONFIG = {"scenario": "ToyGiveWay", "flow_sha256": TOY_FLOW_SHA,
    "basis_sha256": BASIS_SHA, "safety": "certified_hard_projection_v1",
    "representation": "P1-OrthoFlow3", "horizon": 850, "dt": 0.05,
    "success_semantics": "frozen_success_deadlock_timeout_v1",
    "conditioning": "true_t0_latched_eta_v1", "rng": RNG}
MAC_CONFIG = {"scenario": "ToyGiveWay", "flow_sha256": TOY_FLOW_SHA,
    "basis_sha256": None, "safety": None, "representation": "MAC_ONLY",
    "horizon": 850, "dt": 0.05, "success_semantics": "window_progress_v2",
    "conditioning": "flow_only_stepwise_v1", "rng": RNG}


def main():
    frozen = json.loads((OUT / "frozen_proposals.json").read_text())
    requests = []
    mapping = []
    for state in frozen["states"]:
        physical = {"initial_positions": state["initial_positions"]}
        content_hash = uid("content", physical).split("_", 1)[1]
        # Ingestor hashes canonical physical state once, then hashes scenario+content.
        import hashlib
        content_hash = hashlib.sha256(canonical(physical).encode()).hexdigest()
        state_uid = uid("state", {"scenario": SCENARIO, "content": content_hash})
        candidates = {"safety": [0.0, 0.0, 0.0], **state["eta"]}
        seen = set()
        for kind, eta in candidates.items():
            eta_uid, eta, _ = eta_identity(eta)
            controller_uid = uid("ctl", CORRECTION_CONFIG)
            mapping.append({"episode_index": state["episode_index"], "kind": kind,
                            "state_uid": state_uid, "eta_uid": eta_uid, "controller_uid": controller_uid})
            key = (eta_uid, controller_uid)
            if key not in seen:
                requests.append({"state_uid": state_uid, "eta_uid": eta_uid, "controller_uid": controller_uid,
                                 "seed_keys": [canonical({"future_index": j}) for j in range(16)]})
                seen.add(key)
        mac_eid, _, _ = eta_identity([0.0, 0.0, 0.0])
        mac_cid = uid("ctl", MAC_CONFIG)
        mapping.append({"episode_index": state["episode_index"], "kind": "mac_only",
                        "state_uid": state_uid, "eta_uid": mac_eid, "controller_uid": mac_cid})
        requests.append({"state_uid": state_uid, "eta_uid": mac_eid, "controller_uid": mac_cid,
                         "seed_keys": [canonical({"future_index": j}) for j in range(16)]})
    (OUT / "planned_rollouts.json").write_text(json.dumps({"requests": requests}, indent=2) + "\n")
    with (OUT / "candidate_cache_keys.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapping[0])); w.writeheader(); w.writerows(mapping)
    (OUT / "evaluation_protocol.json").write_text(json.dumps({"cohort": frozen["cohort"],
        "cohort_sha256": frozen["cohort_sha256"], "generator_training_sha256": frozen["generator_training_sha256"],
        "K": 4, "continuation_indices": list(range(16)), "rng": RNG,
        "correction_controller_config": CORRECTION_CONFIG, "mac_controller_config": MAC_CONFIG,
        "robust_rule": "at least 15 successes of 16 matched continuations", "candidate_count": len(mapping),
        "unique_state_eta_controller_requests": len(requests), "requested_continuations": len(requests) * 16,
        "test_outcomes_used_for_design": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidates": len(mapping), "unique_requests": len(requests), "continuations": len(requests) * 16}))


if __name__ == "__main__": main()
