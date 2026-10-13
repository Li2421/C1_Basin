"""Freeze natural, label-blind final state and common-η anchor identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "physical_state_critic_formal_v1"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    matched = FORMAL / "matched_protocol.json"
    bank = read(matched)["eta"]
    assert len(bank) == 16 and len({tuple(e) for e in bank}) == 16
    result = {"schema": "transferable_state_conditioning_label_blind_splits_v1",
              "candidate_bank_source_sha256": sha(matched),
              "common_anchor_eta": bank,
              "anchor_discovery_seed_indices": list(range(8)),
              "anchor_evaluation_seed_indices": list(range(8, 16)),
              "historical_targets_role": "DEVELOPMENT only",
              "source_splits": {}, "development": {}, "final_confirmation": {},
              "anchor_subset": {}}
    for scene in SCENES:
        source = {}
        all_uids = set()
        for split in ("train", "validation", "test"):
            path = FORMAL / "loso" / scene / split / "proposals.json"
            rows = read(path)["states"]
            ids = [r["state_uid"] for r in rows]
            assert len(ids) == {"train": 512, "validation": 32, "test": 128}[split]
            assert len(set(ids)) == len(ids) and not (all_uids & set(ids))
            all_uids.update(ids)
            source[split] = {"proposal_sha256": sha(path), "state_uids": ids}
        result["source_splits"][scene] = {k: source[k] for k in ("train", "validation")}
        opened = 48 if scene == "toy_give_way" else 0
        result["development"][scene] = {"reason": "prior prospective Toy labels opened" if opened else "historical targets only",
                                         "formal_test_state_uids": source["test"]["state_uids"][:opened]}
        eligible = source["test"]["state_uids"][opened:]
        result["final_confirmation"][scene] = {"proposal_sha256": source["test"]["proposal_sha256"],
                                                "state_uids": eligible, "count": len(eligible)}
        result["anchor_subset"][scene] = {"rule": "first 64 eligible frozen global TEST states in sampler order",
                                            "state_uids": eligible[:64], "count": 64}
    target = HERE / "frozen_splits.json"
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if target.exists():
        assert target.read_text() == payload, "The final-confirmation split is frozen"
    else:
        target.write_text(payload)
    print(target)


if __name__ == "__main__":
    main()
