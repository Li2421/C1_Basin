"""Materialize lambda=0 candidate entries from verified source-balanced artifacts.

No optimizer step or outer-test inference is performed.  Missing lambda=0
candidate checkpoints are symlinks to the exact prior frozen source-balanced
checkpoints.  Their validation predictions are reconstructed with the same
frozen train-only normalization solely so the validation-only selector can
compare the complete pre-registered lambda grid.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "diagnostics" / "gate_source_balancing_v1" / "conditions" / "source_group_balanced"
sys.path.insert(0, str(OUT))
import conditional_invariance_gate_runner as inv  # noqa: E402


def candidate_dir(fold_id: str, seed: int) -> Path:
    return OUT / "candidate_runs" / "lambda_0p00" / fold_id / f"seed_{seed}"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ready = json.loads((OUT / "INVARIANCE_BASELINE_READY.json").read_text())
    if ready.get("status") != "INVARIANCE_BASELINE_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline not verified", ready))
    jax.config.update("jax_enable_x64", True)
    context, folds = inv.load_context_and_folds()
    training = {(row["fold_id"], str(row["training_seed"])): row for row in read_csv(SOURCE / "training_results.csv") if row["training_seed"] != "SEED_MEAN"}
    created, preserved = [], []
    for fold in folds:
        layout = inv.class_cell_layout(fold, context)
        for seed in inv.SEEDS:
            destination = candidate_dir(fold["fold_id"], seed)
            marker = destination / "DONE.json"
            if marker.exists() and json.loads(marker.read_text()).get("status") == "COMPLETE":
                preserved.append(str(destination)); continue
            source_checkpoint = SOURCE / "checkpoints" / fold["fold_id"] / f"seed_{seed}.npz"
            if not source_checkpoint.exists(): raise RuntimeError(("missing verified source-balanced checkpoint", source_checkpoint))
            destination.mkdir(parents=True, exist_ok=True)
            link = destination / "checkpoint.npz"
            if link.exists() or link.is_symlink(): link.unlink()
            link.symlink_to(source_checkpoint)
            with np.load(source_checkpoint, allow_pickle=False) as values:
                params=[]; i=0
                while f"layer_{i}_weight" in values:
                    params.append({"w": jax.numpy.asarray(values[f"layer_{i}_weight"]), "b": jax.numpy.asarray(values[f"layer_{i}_bias"])}); i+=1
                metadata=json.loads(str(values["metadata_json"]))
                mean=np.asarray(values["normalization_mean"],float); scale=np.asarray(values["normalization_scale"],float); binary=np.asarray(values["normalization_binary_mask"],bool)
            if metadata.get("condition") != "SOURCE_GROUP_BALANCED": raise RuntimeError(("source checkpoint not source balanced", source_checkpoint, metadata.get("condition")))
            if not (np.allclose(mean,fold["mean"]) and np.allclose(scale,fold["scale"]) and np.array_equal(binary,fold["binary"])): raise RuntimeError(("source checkpoint normalization mismatch",source_checkpoint))
            ids,p,y=inv.base.cv.aggregate_state(inv.base.cv.predict(params,fold["normalized"][fold["validation_index"]]),context["sample_ids"][fold["validation_index"]],context["label"])
            threshold=float(inv.base.cv.select_threshold(y,p))
            if abs(threshold-float(metadata["validation_selected_threshold"]))>inv.base.TOLERANCE: raise RuntimeError(("source checkpoint threshold reproduction failed",fold["fold_id"],seed))
            validation=[{"fold_id":fold["fold_id"],"heldout_source_group":fold["outer_group"],"lambda_invariance":0.0,"training_seed":seed,"state_id":str(sid),"oracle_label":int(label),"p_gate":float(prob)} for sid,prob,label in zip(ids,p,y)]
            write_csv(destination/"validation_predictions.csv",validation,["fold_id","heldout_source_group","lambda_invariance","training_seed","state_id","oracle_label","p_gate"])
            # No per-epoch baseline history exists in the prior exact run. Do
            # not fabricate it: record transparent lambda=0, penalty-free
            # metadata. Positive-lambda selected histories retain full epochs.
            source_row=training[(fold["fold_id"],str(seed))]
            history=[]
            for cls in (0,1): history.append({"fold_id":fold["fold_id"],"heldout_source_group":fold["outer_group"],"training_seed":seed,"loss_mode":"source_balanced_bce","lambda_invariance":0.0,"epoch":int(source_row["best_epoch"]),"oracle_class":cls,"eligible_training_source_group_count":int(layout["eligible_group_count_by_class"][cls]),"class_included_in_penalty":bool(layout["eligible_class"][cls]),"sampled_source_balanced_BCE":"","conditional_invariance_penalty":0.0,"class_cell_logit_variance":"","total_loss":"","validation_state_BCE":float(source_row["validation_state_BCE"]),"uses_validation_or_outer_test_data":False,"history_provenance":"prior verified source-balanced checkpoint; no per-epoch history reconstructed"})
            inv.base.write_csv(destination/"training_history.csv",history)
            inv.base.write_json(marker,{"status":"COMPLETE","fold_id":fold["fold_id"],"heldout_source_group":fold["outer_group"],"lambda_invariance":0.0,"training_seed":seed,"loss_mode":"source_balanced_bce","best_epoch":int(metadata["best_epoch"]),"epochs_ran":"reused_prior_checkpoint","validation_state_BCE":float(source_row["validation_state_BCE"]),"validation_selected_threshold":threshold,"checkpoint":"checkpoint.npz","validation_predictions":"validation_predictions.csv","history":"training_history.csv","eligible_training_source_groups_by_class":layout["eligible_group_count_by_class"].tolist(),"outer_test_used_for_lambda_selection":False,"finite":bool(np.isfinite(p).all()),"materialized_from":str(source_checkpoint),"optimizer_steps":0})
            created.append(str(destination))
    inv.base.write_json(OUT/"lambda0_reuse_audit.json",{"status":"COMPLETE","preserved_current_lambda0_candidate_entries":len(preserved),"materialized_prior_lambda0_entries":len(created),"source_artifact":str(SOURCE),"baseline_ready":ready,"new_optimizer_steps":0,"outer_test_inference":False,"new_states":0,"new_oracle_rollouts":0})
    print({"status":"COMPLETE","created":len(created),"preserved":len(preserved)},flush=True)


if __name__=="__main__": main()
