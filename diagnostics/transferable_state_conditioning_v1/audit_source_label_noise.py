"""Independent 2+2 seed check of Q4 source preference stability.

This is a DEVELOPMENT diagnostic; pair cases share physical states and are
not treated as independent units. The final reversal test uses 8+8 seeds.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, FORMAL, SCENES, read


def main():
    result={"schema":"source_Q4_split_2_2_label_stability_v1",
            "discovery_seeds":[0,1],"evaluation_seeds":[2,3],"scenes":{}}
    a,b=np.triu_indices(16,1)
    for scene in SCENES:
        panel=FORMAL/"loso"/scene/"train"
        cases=read(panel/"cases.json")
        raw=panel/"raw"
        y=np.full((512,16,4),-1,np.int8)
        for c in cases:
            file=raw/f"{c['rollout_uid']}.json"
            if not file.exists():continue
            r=read(file)
            assert r["case"]==c
            y[c["state_index"],c["eta_index"],c["future_index"]]=r["success"]
        complete=np.flatnonzero(np.all(y>=0,axis=(1,2)))
        if not len(complete):
            result["scenes"][scene]={"complete_states":0}
            continue
        d=y[complete,:,:2].mean(2)
        e=y[complete,:,2:].mean(2)
        disc=d[:,a]-d[:,b]
        eval_=e[:,a]-e[:,b]
        valid=(disc!=0)&(eval_!=0)
        state_agreement=np.asarray([np.mean(np.sign(disc[i,valid[i]])==np.sign(eval_[i,valid[i]]))
                                    if valid[i].any() else np.nan for i in range(len(complete))])
        q=y[complete].mean(2)
        result["scenes"][scene]={"complete_states":int(len(complete)),
            "state_indices":complete.tolist(),
            "nonconstant_q4_states":int(np.sum(q.max(1)>q.min(1))),
            "span_ge_half_states":int(np.sum(q.max(1)-q.min(1)>=.5)),
            "discovery_decisive_pair_cases":int(np.sum(disc!=0)),
            "independent_eval_nontie_pair_cases":int(np.sum(valid)),
            "pair_sign_persistence":float(np.mean(np.sign(disc[valid])==np.sign(eval_[valid]))) if valid.any() else None,
            "state_mean_sign_persistence":float(np.nanmean(state_agreement)) if np.isfinite(state_agreement).any() else None,
            "states_with_persistent_pair_cases":int(np.isfinite(state_agreement).sum()),
            "discovery_top1_eval_success":float(np.mean(e[np.arange(len(complete)),d.argmax(1)])),
            "evaluation_oracle_success":float(np.mean(e.max(1))),
            "mean_full_success_eta_per_state":float(np.mean(np.sum(q==1,axis=1)))}
    target=HERE/"source_label_noise_audit.json"
    target.write_text(json.dumps(result,indent=2)+"\n")
    for scene, r in result["scenes"].items():print(scene,{k:v for k,v in r.items() if k!="state_indices"})


if __name__=="__main__":main()
