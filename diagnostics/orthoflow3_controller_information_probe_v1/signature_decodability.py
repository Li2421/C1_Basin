"""Cached controller function signature versus H20 summary, source families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .probe import OUT, SRC, SCENES, load_scene, write


def classify(features, split):
    # features[controller, state, channel]; split[state] is frozen source group.
    tr = np.array(split)=="train"; va=np.array(split)=="validation"
    source=features[:,tr,:]; center=source.reshape(-1,features.shape[-1]).mean(0)
    scale=np.maximum(source.reshape(-1,features.shape[-1]).std(0),.05)
    a=(source-center)/scale; b=(features[:,va,:]-center)/scale
    n=a.shape[0]; train=a.reshape(-1,features.shape[-1]); test=b.reshape(-1,features.shape[-1])
    labels=np.repeat(np.arange(n),int(tr.sum())); truth=np.repeat(np.arange(n),int(va.sum()))
    q=np.c_[train,np.ones(len(train))]
    t=np.eye(n)[labels]
    reg=np.diag([10.]*train.shape[-1]+[0.])
    weight=np.linalg.solve(q.T@q+reg,q.T@t)
    pred=(np.c_[test,np.ones(len(test))]@weight).argmax(-1)
    return {"correct":int((pred==truth).sum()),"total":len(truth),"accuracy":float((pred==truth).mean())}


def run():
    rows=[]
    for f in sorted((SRC/"controller_signature/features").glob("*.json")):rows.extend(json.loads(f.read_text()))
    result={}
    for scene in SCENES:
        if not any(r["scene"]==scene for r in rows): continue
        d=load_scene(scene)
        state_ids=sorted({r["state_uid"] for r in d["rows"]})
        split={r["state_uid"]:r["split"] for r in d["rows"]}
        controls=("base","first") if scene=="toy_giveway" else ("base","first","second")
        sig={(r["state_uid"],r["controller"]):r["features"]["mean"] for r in rows if r["scene"]==scene and r["valid"]}
        signature=np.asarray([[sig[(uid,c)] for uid in state_ids] for c in controls],np.float32)
        pair_context=d["context"][:len(controls)]
        compact=np.asarray([[pair_context[j,[i for i,r in enumerate(d["rows"]) if r["state_uid"]==uid]].mean(0)
                             for uid in state_ids] for j in range(len(controls))],np.float32)
        order=[split[uid] for uid in state_ids]
        result[scene]={"states":len(state_ids),"controllers":controls,"H20_full_24D":classify(compact,order),
                       "H20_nominal_10D":classify(compact[:,:,:10],order),
                       "function_signature_24D":classify(signature,order)}
    write(OUT/"function_signature_identity_decodability.json",
          {"outcome_labels_used":False,"new_rollout":0,"source_family_heldout":True,"results":result})
    print(json.dumps(result,indent=2))


if __name__=="__main__":run()
