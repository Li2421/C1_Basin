"""Can H20 response identify the known compatible Flow variant on held-out states?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .probe import OUT, SCENES, load_scene, write


def run():
    result={}
    for scene in SCENES:
        d=load_scene(scene)
        controls=(0,1) if scene=="toy_giveway" else (0,1,2)
        tr=d["split"]=="train"; va=d["split"]=="validation"
        # Candidate-matched controller conditions; never read success labels.
        x=np.concatenate([d["context"][c,tr] for c in controls])
        y=np.concatenate([np.full(int(tr.sum()),c) for c in controls])
        xv=np.concatenate([d["context"][c,va] for c in controls])
        yv=np.concatenate([np.full(int(va.sum()),c) for c in controls])
        means=np.stack([x[y==c].mean(0) for c in controls])
        dist=np.square(xv[:,None,:]-means[None,:,:]).sum(-1)
        centroid=np.asarray(controls)[dist.argmin(1)]
        # Fixed ridge value after source TRAIN standardization. This tests
        # linearly readable identity beyond a naive centroid template.
        reg=10.0
        design=np.c_[x,np.ones(len(x))]
        target=np.stack([y==c for c in controls],-1).astype(float)
        gram=design.T@design+reg*np.diag([1.]*x.shape[1]+[0.])
        weight=np.linalg.solve(gram,design.T@target)
        ridge=np.asarray(controls)[(np.c_[xv,np.ones(len(xv))]@weight).argmax(-1)]
        result[scene]={"TRAIN_states":len({r["state_uid"] for r,t in zip(d["rows"],tr) if t}),
                       "VAL_states":len({r["state_uid"] for r,t in zip(d["rows"],va) if t}),
                       "controllers":controls,"chance":1/len(controls),
                       "centroid_accuracy":float((centroid==yv).mean()),
                       "ridge_accuracy":float((ridge==yv).mean()),
                       "ridge_confusion":[[int(((yv==a)&(ridge==b)).sum()) for b in controls] for a in controls]}
    write(OUT/"context_identity_decodability.json",{"source_family_heldout":True,"outcome_labels_used":False,
           "new_rollout":0,"results":result})
    print(json.dumps(result,indent=2))


if __name__=="__main__":run()
