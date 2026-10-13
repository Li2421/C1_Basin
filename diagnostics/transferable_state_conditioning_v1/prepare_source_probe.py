"""Freeze source-only broad-η probe on the first 16 TRAIN states per scene."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shared_rollout_db.src.rollout_db import canonical, eta_identity, uid

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, FORMAL, SCENES, read, sha

EXPERIMENT = "exp_transferable_q_source_anchor_probe_20261008_v1"


def freeze(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == payload, f"frozen request changed: {path}"
    else:
        path.write_text(payload)


def main():
    split = read(HERE / "frozen_splits.json")
    protocol = read(FORMAL / "loso_protocol.json")
    bank = split["common_anchor_eta"]
    counts = {}
    for scene in SCENES:
        proposals_path = FORMAL / "loso" / scene / "train" / "proposals.json"
        proposals = read(proposals_path)["states"][:16]
        assert len(proposals) == 16
        assert [r["state_uid"] for r in proposals] == split["source_splits"][scene]["train"]["state_uids"][:16]
        controller = protocol["controllers"][scene]["intervention_uid"]
        cases = []; requests = []
        for i, p in enumerate(proposals):
            for j, eta in enumerate(bank):
                eta_uid = eta_identity(eta)[0]
                seed_keys = []
                for future_index in range(4):
                    sk = {"future_index": future_index}
                    if scene == "toy_give_way": sk["rollout_id"] = p["rollout_id"]
                    seed_key = canonical(sk)
                    rollout_uid = uid("roll", dict(state=p["state_uid"], eta=eta_uid,
                                                   controller=controller, seed=seed_key))
                    cases.append(dict(scene=scene, split="train", state_index=i,
                                      state_uid=p["state_uid"], state_alias=p["state_alias"],
                                      eta_index=j, eta=eta, eta_uid=eta_uid,
                                      future_index=future_index, seed_key=seed_key,
                                      controller_uid=controller, rollout_uid=rollout_uid))
                    seed_keys.append(seed_key)
                requests.append(dict(state_uid=p["state_uid"], eta_uid=eta_uid,
                                     controller_uid=controller, seed_keys=seed_keys))
        panel = HERE / "source_anchor_probe" / scene
        freeze(panel / "cases.json", cases)
        freeze(panel / "planned_rollouts.json", dict(requests=requests))
        counts[scene] = {"states": 16, "eta": 16, "seed_slots": len(cases),
                         "source_proposals_sha256": sha(proposals_path),
                         "cases_sha256": sha(panel / "cases.json")}
    freeze(HERE / "source_anchor_probe_counts.json", counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
