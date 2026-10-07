"""Fair eta-only control for source+spatial-eta double holdout."""

import json

import numpy as np
from scipy.spatial.distance import cdist

from offline_baselines import OUT, dump_json, kernel_train, kernel_predict
from train_interaction import prepare, SCENARIOS


def main():
    data = prepare()
    out = {}
    for sc in SCENARIOS:
        train, dev, test = (data[sc]["parts"][x] for x in ("train","dev","test"))
        ids = sorted(set(train["eta_uids"]))
        x = np.stack([train["eta"][train["eta_uids"].index(e)] for e in ids])
        eidx = {e:i for i,e in enumerate(ids)}
        sids = sorted(set(train["state_uids"]))
        sidx = {s:i for i,s in enumerate(sids)}
        y = np.full((len(sids),len(ids)),np.nan)
        for j,(s,e) in enumerate(zip(train["state_uids"],train["eta_uids"])):
            y[sidx[s],eidx[e]]=train["y"][j]
        assert np.isfinite(y).mean()>.99
        scans=[]
        for bw in (0.04,0.07,0.1,0.15,0.22,0.33,0.5,0.75):
            for ridge in (0.001,0.01,0.1,1.0):
                prior,alpha=kernel_train(x,y,bw,ridge)
                pp=kernel_predict(x,dev["eta"],prior,alpha,bw)
                nll=float(np.mean(-dev["y"]*np.log(np.clip(pp,1e-5,1-1e-5))-(1-dev["y"])*np.log(np.clip(1-pp,1e-5,1-1e-5))))
                scans.append((nll,bw,ridge))
        dev_nll,bw,ridge=min(scans)
        prior,alpha=kernel_train(x,y,bw,ridge)
        pred=kernel_predict(x,test["eta"],prior,alpha,bw)
        yy=test["y"]
        nll=float(np.mean(-yy*np.log(np.clip(pred,1e-5,1-1e-5))-(1-yy)*np.log(np.clip(1-pred,1e-5,1-1e-5))))
        brier=float(np.mean((yy-pred)**2))
        by={}
        for j,s in enumerate(test["state_uids"]):by.setdefault(s,[]).append(j)
        selected=sum(yy[ix[int(np.argmax(pred[ix]))]] for ix in by.values())
        oracle=sum(np.max(yy[ix]) for ix in by.values())
        out[sc]={"train_states":len(sids),"train_eta":len(ids),"test_states":len(by),
                 "test_eta":len(set(test["eta_uids"])),"bandwidth":bw,"ridge":ridge,
                 "dev_nll":dev_nll,"test_nll":nll,"test_brier":brier,
                 "test_selected_b15_states":int(selected),"test_oracle_b15_states":int(oracle),
                 "new_rollout":0}
    dump_json("spatial_eta_only_result.json",out)
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
