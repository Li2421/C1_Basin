"""Outcome-blind A2 inference for the previously missed, compatible Round11 R6 T2 array."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import fit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R6=ROOT/"diagnostics/controller_late_phase_probe_v1/R6_FEATURES.npz"
TRAIN_R6=ROOT/"diagnostics/ring_controller_population_v1/R6_FEATURES.npz"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert (HERE/"MODEL_FREEZE.receipt.json").exists()
    names, physical, r5=fit.load_external("T2")
    with np.load(R6,allow_pickle=False) as z, np.load(TRAIN_R6,allow_pickle=False) as train:
        source_names=list(z["controller_names"])
        assert set(names)<=set(source_names)
        assert list(z["statistic_names"])==list(train["statistic_names"])
        assert z["delta"].shape[1]==train["delta"].shape[1]==7
        r6=z["delta"][[source_names.index(n) for n in names]].astype(np.float32)
        paired=z["paired_state_count"][[source_names.index(n) for n in names]]
        assert np.isfinite(r6).all() and (paired>=15).all()
    selected=json.loads((HERE/"fits/selected_steps.json").read_text())
    with np.load(HERE/"fits/full_transform.npz",allow_pickle=False) as z:
        transform={k:z[k] for k in z.files}
    plan={"status":"POST_FREEZE_EXTERNAL_FEATURE_AVAILABILITY_CORRECTION","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "cohort":"T2","model":"A2","reason":"Round11 R6_FEATURES.npz contains all 14 T2 original controllers and the identical seven R6 statistics. Initial directory-local audit missed this compatible historical file.",
          "external_T2_labels_already_opened":True,"use_in_T1_hypothesis_gates":False,"model_or_hyperparameter_changed":False,
          "r6_feature_sha256":sha(R6),"training_r6_feature_sha256":sha(TRAIN_R6),"original_model_freeze_receipt_sha256":sha(HERE/"MODEL_FREEZE.receipt.json"),
          "selected_step":selected["A2"],"controller_names":names,"paired_state_count":paired.tolist(),
          "checkpoint_sha256":{str(seed):sha(fit.path_for("full","A2",seed,selected["A2"])) for seed in fit.SEEDS}}
    (HERE/"T2_A2_EXTENSION_PLAN.receipt.json").write_text(json.dumps(plan,indent=2)+"\n")
    outputs={}
    for seed in fit.SEEDS:
        ck=fit.path_for("full","A2",seed,selected["A2"])
        pred=fit.predict("A2",seed,ck,transform,physical,r5,r6)
        assert pred.shape==(14,64,8) and np.isfinite(pred).all()
        path=HERE/"predictions"/f"T2_A2_seed{seed}.npz"
        np.savez_compressed(path,probability=pred,controller_names=names,arms=fit.ARMS)
        outputs[str(path.relative_to(HERE))]=sha(path)
    (HERE/"T2_A2_PREDICTIONS.receipt.json").write_text(json.dumps({"status":"POST_FREEZE_A2_T2_PREDICTIONS","plan_sha256":sha(HERE/"T2_A2_EXTENSION_PLAN.receipt.json"),"files":outputs},indent=2)+"\n")
    print(json.dumps({"status":"T2_A2_PREDICTED","files":list(outputs)}))

if __name__=="__main__":main()
