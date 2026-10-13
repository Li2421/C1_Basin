"""Plan exact common-bank requests without opening or generating TEST labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shared_rollout_db.src.rollout_db import canonical, eta_identity, uid

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "physical_state_critic_formal_v1"


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == payload, f"frozen artifact changed: {path}"
    else:
        path.write_text(payload)


def main():
    split = read(HERE / "frozen_splits.json")
    protocol = read(FORMAL / "loso_protocol.json")
    eta = split["common_anchor_eta"]
    assert len(eta) == 16
    counts = {}
    for scene, subset in split["anchor_subset"].items():
        proposals = read(FORMAL / "loso" / scene / "test" / "proposals.json")["states"]
        by_uid = {r["state_uid"]: r for r in proposals}
        assert len(by_uid) == 128
        controller = protocol["controllers"][scene]["intervention_uid"]
        requests = []
        cases = []
        for state_index, state_uid in enumerate(subset["state_uids"]):
            row = by_uid[state_uid]
            for eta_index, vector in enumerate(eta):
                eta_uid = eta_identity(vector)[0]
                for future_index in range(16):
                    key = {"future_index": future_index}
                    if scene == "toy_give_way":
                        key["rollout_id"] = row["rollout_id"]
                    seed_key = canonical(key)
                    rollout_uid = uid("roll", dict(state=state_uid, eta=eta_uid,
                                                   controller=controller, seed=seed_key))
                    cases.append(dict(scene=scene, state_index=state_index,
                                      state_uid=state_uid, eta_index=eta_index,
                                      eta=vector, eta_uid=eta_uid,
                                      future_index=future_index, seed_key=seed_key,
                                      controller_uid=controller, rollout_uid=rollout_uid))
                requests.append(dict(state_uid=state_uid, eta_uid=eta_uid,
                                     controller_uid=controller,
                                     seed_keys=[cases[-16+i]["seed_key"] for i in range(16)]))
        panel = HERE / "anchor" / scene
        freeze(panel / "planned_rollouts.json", dict(requests=requests))
        freeze(panel / "cases.json", cases)
        counts[scene] = dict(states=len(subset["state_uids"]), eta=16,
                             slots=len(cases), cases_sha256=hashlib.sha256((panel/"cases.json").read_bytes()).hexdigest())
    freeze(HERE / "anchor_counts.json", counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
