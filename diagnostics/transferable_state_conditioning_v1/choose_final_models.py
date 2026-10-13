"""Source-only, prespecified concat-vs-residual selection; no TEST labels."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, read, sha
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE
from diagnostics.physical_state_critic_formal_v1.train_unified import select_key

GLOBAL_HERE = HERE
HERE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE)))

def best(panel, kind):
    rows = [read(HERE/"fits"/panel/kind/f"seed{s}"/"complete.json") for s in SEEDS]
    return min(rows,key=lambda r:(*select_key(r["selected"],NEAR_TIE),r["seed"]))


def main():
    split = GLOBAL_HERE/"frozen_splits.json"
    result={"schema":"transferable_q_source_only_final_model_selection_v1",
            "selection_rule":"minimum source-validation selection-regret bucket (1/512), then pairwise ranking, then matched-minus-mismatch source-validation mean success, then NLL; no held-out labels",
            "frozen_splits_sha256":sha(split),"final_outcomes_opened":False,
            "panels":{}}
    for panel in PANELS:
        base_diag=HERE/"fits"/panel/"source_validation_diagnostic.json"
        resid_diag=HERE/"fits"/panel/"residual_source_validation_diagnostic.json"
        baseline=read(base_diag); residual=read(resid_diag)
        models={kind:best(panel,kind) for kind in ("state_concat","state_residual","eta_only_wide")}
        assert all(m["target_labels_used"] is False for m in models.values())
        data_hash=sha(HERE/"source_data"/panel/"trainval.npz")
        assert all(m["trainval_sha256"]==data_hash for m in models.values())
        mismatch={"state_concat":baseline["state_minus_mismatch_mean_success"]["mean"],
                  "state_residual":residual["paired"]["residual_minus_mismatch_mean_success"]["mean"]}
        def model_key(kind):
            score=models[kind]["selected"]
            pair=score["pairwise_accuracy"] if score["pairwise_accuracy"] is not None else -.5
            return (int(np.floor(score["selection_regret"]/NEAR_TIE+1e-9)),
                    -pair,-mismatch[kind],score["validation_nll"],score["step"])
        chosen=min(("state_concat","state_residual"),key=model_key)
        eta=models["eta_only_wide"]
        state=models[chosen]
        assert models["state_residual"]["eta_prior_checkpoint_sha256"]==eta["checkpoint_sha256"]
        state_path=HERE/"fits"/panel/chosen/f"seed{state['seed']}"/"checkpoint.msgpack"
        eta_path=HERE/"fits"/panel/"eta_only_wide"/f"seed{eta['seed']}"/"checkpoint.msgpack"
        assert sha(state_path)==state["checkpoint_sha256"] and sha(eta_path)==eta["checkpoint_sha256"]
        result["panels"][panel]={"state_kind":chosen,
            "state_seed":state["seed"],"state_checkpoint":str(state_path),
            "state_checkpoint_sha256":state["checkpoint_sha256"],
            "eta_seed":eta["seed"],"eta_checkpoint":str(eta_path),
            "eta_checkpoint_sha256":eta["checkpoint_sha256"],
            "source_data_sha256":data_hash,
            "source_validation_concat":models["state_concat"]["selected"],
            "source_validation_residual":models["state_residual"]["selected"],
            "source_validation_eta":eta["selected"],
            "source_validation_concat_mismatch_gain":mismatch["state_concat"],
            "source_validation_residual_mismatch_gain":mismatch["state_residual"],
            "source_validation_diagnostic_sha256":sha(base_diag),
            "residual_validation_diagnostic_sha256":sha(resid_diag)}
    target=HERE/"final_models_frozen.json"
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if target.exists():assert target.read_text()==payload,"Final model freeze is immutable"
    else:target.write_text(payload)
    print(json.dumps({"freeze_sha256":sha(target),
                      "selected":{p:v["state_kind"] for p,v in result["panels"].items()}},indent=2))


if __name__=="__main__":main()
