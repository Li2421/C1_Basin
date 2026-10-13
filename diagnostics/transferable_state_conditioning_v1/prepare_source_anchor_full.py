"""Label-blind optional source-only common-bank expansion plan.

This is NOT launched until the small source probe justifies the added labels.
The first 16 TRAIN states overlap the probe by exact UID, so preflight can
reuse their Q4 outcomes after its journal merge.
"""
from __future__ import annotations

import json

from shared_rollout_db.src.rollout_db import canonical, eta_identity, uid

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, FORMAL, SCENES, read, sha

EXPERIMENT = "exp_transferable_q_source_anchor_full_20261008_v1"


def freeze(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(obj,separators=(",",":"),sort_keys=True)+"\n"
    if path.exists():assert path.read_text()==payload,f"frozen source request changed: {path}"
    else:path.write_text(payload)


def main():
    split=read(HERE/"frozen_splits.json")
    protocol=read(FORMAL/"loso_protocol.json")
    bank=split["common_anchor_eta"]
    counts={}
    for scene in SCENES:
        controller=protocol["controllers"][scene]["intervention_uid"]
        for part,nstates,seeds in (("train",128,4),("validation",16,16)):
            proposal_path=FORMAL/"loso"/scene/part/"proposals.json"
            states=read(proposal_path)["states"][:nstates]
            assert [p["state_uid"] for p in states]==split["source_splits"][scene][part]["state_uids"][:nstates]
            cases=[];requests=[]
            for i,p in enumerate(states):
                for j,eta in enumerate(bank):
                    eta_uid=eta_identity(eta)[0]
                    seed_keys=[]
                    for future_index in range(seeds):
                        key={"future_index":future_index}
                        if scene=="toy_give_way":key["rollout_id"]=p["rollout_id"]
                        seed_key=canonical(key)
                        rollout_uid=uid("roll",dict(state=p["state_uid"],eta=eta_uid,
                                                    controller=controller,seed=seed_key))
                        cases.append(dict(scene=scene,split=part,state_index=i,
                                          state_uid=p["state_uid"],state_alias=p["state_alias"],
                                          eta_index=j,eta=eta,eta_uid=eta_uid,
                                          future_index=future_index,seed_key=seed_key,
                                          controller_uid=controller,rollout_uid=rollout_uid))
                        seed_keys.append(seed_key)
                    requests.append(dict(state_uid=p["state_uid"],eta_uid=eta_uid,
                                         controller_uid=controller,seed_keys=seed_keys))
            panel=HERE/"source_anchor_full"/scene/part
            freeze(panel/"cases.json",cases)
            freeze(panel/"planned_rollouts.json",dict(requests=requests))
            counts[f"{scene}/{part}"]={"independent_states":nstates,"eta":16,
                                      "seed_slots":len(cases),"proposal_sha256":sha(proposal_path),
                                      "cases_sha256":sha(panel/"cases.json")}
    freeze(HERE/"source_anchor_full_counts.json",counts)
    print(json.dumps(counts,indent=2))


if __name__=="__main__":main()
