"""Commit lambda from validation-only data, then run selected checkpoint OOF inference."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import conditional_invariance_gate_runner as inv  # noqa: E402


CONDITION = "CONDITIONAL_SOURCE_INVARIANCE"


def lambda_tag(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def candidate_dir(fold_id: str, value: float, seed: int) -> Path:
    return OUT / "candidate_runs" / lambda_tag(value) / fold_id / f"seed_{seed}"


def candidate_validation(fold_id: str, value: float, seed: int) -> Path:
    return candidate_dir(fold_id, value, seed) / "validation_predictions.csv"


def checkpoint(fold_id: str, value: float, seed: int) -> Path:
    return candidate_dir(fold_id, value, seed) / "checkpoint.npz"


def load_checkpoint(path: Path, fold: dict):
    with np.load(path, allow_pickle=False) as values:
        params = []
        index = 0
        while f"layer_{index}_weight" in values:
            params.append({"w": jnp.asarray(values[f"layer_{index}_weight"]), "b": jnp.asarray(values[f"layer_{index}_bias"])})
            index += 1
        metadata = json.loads(str(values["metadata_json"]))
        mean = np.asarray(values["normalization_mean"], float)
        scale = np.asarray(values["normalization_scale"], float)
        binary = np.asarray(values["normalization_binary_mask"], bool)
    if metadata.get("condition") != "CONDITIONAL_SOURCE_INVARIANCE_GATE":
        raise RuntimeError(("wrong selected checkpoint", path))
    if not (np.allclose(mean, fold["mean"]) and np.allclose(scale, fold["scale"]) and np.array_equal(binary, fold["binary"])):
        raise RuntimeError(("normalization mismatch", path))
    return params, metadata


def validation_summary(fold: dict, value: float) -> tuple[dict, list[list[dict]]]:
    by_seed = []
    for seed in inv.SEEDS:
        rows = read_csv(candidate_validation(fold["fold_id"], value, seed))
        if not rows or any(row["fold_id"] != fold["fold_id"] or float(row["lambda_invariance"]) != value or int(row["training_seed"]) != seed for row in rows):
            raise RuntimeError(("candidate validation metadata mismatch", fold["fold_id"], value, seed))
        by_seed.append(rows)
    ids = [row["state_id"] for row in by_seed[0]]
    labels = np.asarray([int(row["oracle_label"]) for row in by_seed[0]], int)
    if any([row["state_id"] for row in rows] != ids or [int(row["oracle_label"]) for row in rows] != labels.tolist() for rows in by_seed[1:]):
        raise RuntimeError(("validation seed aggregation mismatch", fold["fold_id"], value))
    p = np.mean([[float(row["p_gate"]) for row in rows] for rows in by_seed], axis=0)
    threshold = float(inv.base.cv.select_threshold(labels, p))
    metrics = inv.base.cv.metrics(labels, p, threshold)
    return ({
        "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "lambda_invariance": value,
        "selection_data": "validation stable states only", "seed_aggregation": "mean probability over 17/23/41",
        "validation_state_count": len(ids), "validation_selected_threshold": threshold,
        **{f"validation_{key}": item for key, item in metrics.items()},
        "validation_state_BCE": float(inv.base.cv.state_bce(labels, p)),
    }, by_seed)


def choose_lambdas(folds: list[dict]) -> tuple[dict[str, float], list[dict]]:
    chosen, all_rows = {}, []
    for fold in folds:
        entries = [validation_summary(fold, value)[0] for value in inv.LAMBDA_GRID]
        # Fixed before OOF: balanced accuracy, then lower worst class error,
        # then validation discrimination/calibration, then lower lambda tie.
        def score(row: dict):
            return (float(row["validation_balanced_accuracy"]), -max(float(row["validation_FPR"]), float(row["validation_FNR"])), float(row["validation_AUROC"]), float(row["validation_AUPRC"]), -float(row["validation_Brier"]), -float(row["lambda_invariance"]))
        best = max(entries, key=score)
        chosen[fold["fold_id"]] = float(best["lambda_invariance"])
        for row in entries:
            row["selected_by_validation_only"] = bool(row is best)
            row["selection_rule"] = "maximize validation BAcc; minimize max(FPR,FNR); maximize AUROC/AUPRC; minimize Brier; lower lambda tie-break"
            row["outer_test_read_during_selection"] = False
            all_rows.append(row)
    return chosen, all_rows


def assemble_selected(fold: dict, context: dict, value: float) -> dict:
    ids = context["sample_ids"]
    labels = context["label"]
    validation, test, models = [], [], {}
    for seed in inv.SEEDS:
        params, metadata = load_checkpoint(checkpoint(fold["fold_id"], value, seed), fold)
        val_ids, val_p, val_y = inv.base.cv.aggregate_state(inv.base.cv.predict(params, fold["normalized"][fold["validation_index"]]), ids[fold["validation_index"]], labels)
        saved = read_csv(candidate_validation(fold["fold_id"], value, seed))
        if saved and ([row["state_id"] for row in saved] != val_ids.tolist() or not np.allclose(np.asarray([float(row["p_gate"]) for row in saved]), val_p, atol=2e-6, rtol=0)):
            raise RuntimeError(("checkpoint validation reproduction failed", fold["fold_id"], value, seed))
        test_ids, test_p, test_y = inv.base.cv.aggregate_state(inv.base.cv.predict(params, fold["normalized"][fold["test_index"]]), ids[fold["test_index"]], labels)
        threshold = float(metadata["validation_selected_threshold"])
        if abs(threshold - float(inv.base.cv.select_threshold(val_y, val_p))) > inv.base.TOLERANCE:
            raise RuntimeError(("validation threshold reproduction failed", fold["fold_id"], value, seed))
        models[seed] = {"params": params, "threshold": threshold}
        validation.append((val_ids, val_p, val_y))
        test.append((test_ids, test_p, test_y))
    if any(not np.array_equal(validation[0][0], item[0]) for item in validation) or any(not np.array_equal(test[0][0], item[0]) for item in test):
        raise RuntimeError(("selected aggregation ID mismatch", fold["fold_id"]))
    val_mean = np.mean([item[1] for item in validation], axis=0)
    return {"fold": fold, "lambda_invariance": value, "models": models, "validation": validation, "test": test, "test_ids": test[0][0], "test_mean": np.mean([item[1] for item in test], axis=0), "ensemble_threshold": float(inv.base.cv.select_threshold(validation[0][2], val_mean))}


def emit_state_rows(result: dict) -> list[dict]:
    rows = []
    for i, seed in enumerate(inv.SEEDS):
        threshold = result["models"][seed]["threshold"]
        threshold_logit = float(inv.base.logit(threshold))
        for state_id, p, y in zip(result["test"][i][0], result["test"][i][1], result["test"][i][2]):
            logit = float(inv.base.logit(p)); predicted = int(p >= threshold)
            rows.append({"condition": CONDITION, "loss_mode": "source_balanced_bce" if result["lambda_invariance"] == 0 else "conditional_source_invariance_bce", "lambda_invariance": result["lambda_invariance"], "fold_id": result["fold"]["fold_id"], "heldout_source_group": result["fold"]["outer_group"], "state_id": str(state_id), "oracle_label": int(y), "p_gate": float(p), "gate_logit": logit, "validation_selected_threshold": threshold, "validation_threshold_logit": threshold_logit, "fold_relative_logit_margin": logit-threshold_logit, "predicted_label": predicted, "correct": bool(predicted == int(y)), "training_seed": seed})
    threshold = result["ensemble_threshold"]; threshold_logit = float(inv.base.logit(threshold))
    for state_id, p, y in zip(result["test_ids"], result["test_mean"], result["test"][0][2]):
        logit = float(inv.base.logit(p)); predicted = int(p >= threshold)
        rows.append({"condition": CONDITION, "loss_mode": "source_balanced_bce" if result["lambda_invariance"] == 0 else "conditional_source_invariance_bce", "lambda_invariance": result["lambda_invariance"], "fold_id": result["fold"]["fold_id"], "heldout_source_group": result["fold"]["outer_group"], "state_id": str(state_id), "oracle_label": int(y), "p_gate": float(p), "gate_logit": logit, "validation_selected_threshold": threshold, "validation_threshold_logit": threshold_logit, "fold_relative_logit_margin": logit-threshold_logit, "predicted_label": predicted, "correct": bool(predicted == int(y)), "training_seed": "SEED_MEAN"})
    return rows


def infer_variants(state_id: str, result: dict, context: dict, role: str) -> list[dict]:
    arrays = context["arrays"]; sample_ids = context["sample_ids"]
    index = np.flatnonzero(sample_ids == state_id); index = index[np.argsort(arrays["flow_seed"][index])]
    if len(index) != 64: raise RuntimeError(("expected 64 saved variants", state_id, len(index)))
    x = result["fold"]["normalized"][index]; rows=[]; per_seed=[]
    for seed in inv.SEEDS:
        model=result["models"][seed]; logits=np.asarray(inv.base.cv.logits(model["params"], jnp.asarray(x,jnp.float32)),float); p=np.asarray(jax.nn.sigmoid(jnp.asarray(logits)),float); per_seed.append(p); t=model["threshold"]; tl=float(inv.base.logit(t))
        for pos, data_index in enumerate(index):
            predicted=int(p[pos]>=t); rows.append({"condition":CONDITION,"role":role,"lambda_invariance":result["lambda_invariance"],"fold_id":result["fold"]["fold_id"],"heldout_source_group":result["fold"]["outer_group"],"state_id":state_id,"oracle_label":int(context["label"][state_id]),"flow_seed":int(arrays["flow_seed"][data_index]),"sample_id":str(arrays["sample_id"][data_index]),"training_seed":seed,"p_gate":float(p[pos]),"gate_logit":float(logits[pos]),"validation_selected_threshold":t,"validation_threshold_logit":tl,"fold_relative_logit_margin":float(logits[pos]-tl),"predicted_label":predicted,"correct":bool(predicted==int(context["label"][state_id]))})
    p=np.mean(per_seed,axis=0); t=result["ensemble_threshold"]; tl=float(inv.base.logit(t))
    for pos, data_index in enumerate(index):
        logit=float(inv.base.logit(p[pos])); predicted=int(p[pos]>=t); rows.append({"condition":CONDITION,"role":role,"lambda_invariance":result["lambda_invariance"],"fold_id":result["fold"]["fold_id"],"heldout_source_group":result["fold"]["outer_group"],"state_id":state_id,"oracle_label":int(context["label"][state_id]),"flow_seed":int(arrays["flow_seed"][data_index]),"sample_id":str(arrays["sample_id"][data_index]),"training_seed":"SEED_MEAN","p_gate":float(p[pos]),"gate_logit":logit,"validation_selected_threshold":t,"validation_threshold_logit":tl,"fold_relative_logit_margin":logit-tl,"predicted_label":predicted,"correct":bool(predicted==int(context["label"][state_id]))})
    return rows


def metric_summary(rows: list[dict]) -> dict:
    if not rows: return {"state_count":0}
    y=np.asarray([int(r["oracle_label"]) for r in rows]); score=np.asarray([float(r["fold_relative_logit_margin"]) for r in rows]); pred=np.asarray([int(r["predicted_label"]) for r in rows]); tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); pos=tp+fn; neg=tn+fp
    return {"score_coordinate":"fold_relative_logit_margin, not raw cross-fold logits","state_count":len(rows),"stable_zero":neg,"stable_nonzero":pos,"balanced_accuracy":((tp/pos)+(tn/neg))/2 if pos and neg else None,"AUROC":float(inv.base.cv.auroc(y,score)) if pos and neg else None,"AUPRC":float(inv.base.cv.auprc(y,score)) if pos and neg else None,"accuracy":float((pred==y).mean()),"FPR":fp/neg if neg else None,"FNR":fn/pos if pos else None,"TP":tp,"TN":tn,"FP":fp,"FN":fn,"mean_zero_margin":float(score[y==0].mean()) if neg else None,"mean_nonzero_margin":float(score[y==1].mean()) if pos else None}


def stable_sets() -> tuple[set[str],set[str],set[str]]:
    source=ROOT/"diagnostics"/"hard_stable_boundary_crossval"; difficult_rows=read_csv(source/"difficult_stable_states.csv"); difficult={r["state_id"] for r in difficult_rows}; robust={r["state_id"] for r in read_csv(source/"repeated_errors.csv") if int(r["oracle_label"])==0}; required={"RBV_Q_pair228_m080_s95401003_p030","RB_Q_pair226_m080_s95400802_p073","RB_Q_pair228_m080_s95401001_p050"}
    if not required <= robust: raise RuntimeError(("robust zero inventory mismatch",required-robust))
    return difficult, robust, {r["state_id"] for r in difficult_rows if int(r["oracle_label"])==1}


def variant_summary(rows: list[dict]) -> list[dict]:
    by=defaultdict(list)
    for row in rows: by[(row["state_id"],str(row["training_seed"]))].append(row)
    output=[]
    for (state,seed),block in sorted(by.items()):
        p=np.asarray([float(r["p_gate"]) for r in block]); m=np.asarray([float(r["fold_relative_logit_margin"]) for r in block]); pred=np.asarray([int(r["predicted_label"]) for r in block]); y=int(block[0]["oracle_label"])
        output.append({"state_id":state,"oracle_label":y,"training_seed":seed,"fold_id":block[0]["fold_id"],"heldout_source_group":block[0]["heldout_source_group"],"lambda_invariance":block[0]["lambda_invariance"],"variant_count":len(block),"correct_variants":int((pred==y).sum()),"wrong_variants":int((pred!=y).sum()),"intervention_fraction":float((pred==1).mean()),"mean_p_gate":float(p.mean()),"std_p_gate":float(p.std()),"mean_fold_relative_margin":float(m.mean()),"std_fold_relative_margin":float(m.std())})
    return output


def selected_histories(folds: list[dict], chosen: dict[str,float]) -> list[dict]:
    output=[]
    for fold in folds:
        value=chosen[fold["fold_id"]]
        for seed in inv.SEEDS:
            path=candidate_dir(fold["fold_id"],value,seed)/"training_history.csv"
            output.extend(read_csv(path))
    return output


def checkpoint_links(folds: list[dict], chosen: dict[str,float]) -> list[dict]:
    root=OUT/"selected_checkpoints"
    if root.exists(): shutil.rmtree(root)
    rows=[]
    for fold in folds:
        value=chosen[fold["fold_id"]]
        for seed in inv.SEEDS:
            target=checkpoint(fold["fold_id"],value,seed); link=root/fold["fold_id"]/f"seed_{seed}.npz"; link.parent.mkdir(parents=True,exist_ok=True); link.symlink_to(target)
            rows.append({"fold_id":fold["fold_id"],"heldout_source_group":fold["outer_group"],"lambda_invariance":value,"training_seed":seed,"checkpoint":str(link),"target":str(target)})
    return rows


def main() -> None:
    started=time.perf_counter(); jax.config.update("jax_enable_x64",True)
    ready=json.loads((OUT/"INVARIANCE_BASELINE_READY.json").read_text())
    if ready.get("status")!="INVARIANCE_BASELINE_READY" or not ready.get("all_checks_passed"): raise RuntimeError(("baseline readiness",ready))
    context,folds=inv.load_context_and_folds(); expected=len(folds)*len(inv.LAMBDA_GRID)*len(inv.SEEDS); done=list((OUT/"candidate_runs").glob("lambda_*/*/seed_*/DONE.json"))
    if len(done)!=expected or any(json.loads(p.read_text()).get("status")!="COMPLETE" for p in done): raise RuntimeError(("candidate training incomplete",len(done),expected))
    # Phase 1: only candidate validation CSVs are read through this commit.
    chosen,selection=choose_lambdas(folds); write_csv(OUT/"hyperparameter_selection.csv",selection); inv.base.write_json(OUT/"selected_lambda_by_fold.json",{"selection_data":"validation stable states only","outer_test_used_for_selection":False,"lambda_invariance_by_fold":chosen})
    # Phase 2: selected checkpoint OOF inference only after selection commit.
    selected={fold["fold_id"]:assemble_selected(fold,context,chosen[fold["fold_id"]]) for fold in folds}; oof=[]
    for result in selected.values(): oof.extend(emit_state_rows(result))
    write_csv(OUT/"oof_predictions.csv",oof)
    difficult,robust_zero,difficult_nonzero=stable_sets(); by_group={r["fold"]["outer_group"]:r for r in selected.values()}; variants=[]
    for state in sorted(difficult):
        role="difficult_stable_nonzero" if context["label"][state] else "difficult_stable_zero"
        if state in robust_zero: role+="|robust_false_positive_zero_reference"
        variants.extend(infer_variants(state,by_group[context["state"][state]["source_group"]],context,role))
    write_csv(OUT/"hard_state_variant_predictions.csv",variants); vsummary=variant_summary(variants); write_csv(OUT/"hard_zero_state_metrics.csv",[r for r in vsummary if r["state_id"] in robust_zero]); write_csv(OUT/"hard_nonzero_state_metrics.csv",[r for r in vsummary if r["state_id"] in difficult_nonzero])
    sm=[r for r in oof if r["training_seed"]=="SEED_MEAN"]; state=context["state"]; allm=metric_summary(sm); recovery=metric_summary([r for r in sm if state[r["state_id"]]["category"]=="RECOVERY"]); diff=metric_summary([r for r in sm if r["state_id"] in difficult])
    for metric,name in ((allm,"all_stable_metrics.json"),(recovery,"recovery_metrics.json"),(diff,"difficult_stable_metrics.json")):
        metric.update({"condition":CONDITION,"selected_lambda_by_fold":chosen}); inv.base.write_json(OUT/name,metric)
    history=selected_histories(folds,chosen); write_csv(OUT/"training_loss_history.csv",history); write_csv(OUT/"conditional_group_class_statistics.csv",history,["fold_id","heldout_source_group","training_seed","loss_mode","lambda_invariance","epoch","oracle_class","eligible_training_source_group_count","class_included_in_penalty","conditional_invariance_penalty","class_cell_logit_variance","uses_validation_or_outer_test_data"]); links=checkpoint_links(folds,chosen); write_csv(OUT/"selected_checkpoint_manifest.csv",links)
    margins=inv.base.calibration_rows(sm,CONDITION); write_csv(OUT/"fold_relative_margin_analysis.csv",margins)
    checks={"status":"CONDITIONAL_INVARIANCE_TRAINING_COMPLETE","candidate_training_count":expected,"candidate_lambda_values_exactly_preregistered":True,"selected_lambda_using_validation_only":True,"outer_test_read_during_lambda_selection":False,"outer_test_excluded_from_training_normalization_validation_threshold_and_regularizer":True,"selected_checkpoint_validation_reproduction":True,"all_selected_outputs_finite":all(np.isfinite(float(r["p_gate"])) and np.isfinite(float(r["gate_logit"])) for r in oof+variants),"frozen_fold_count":len(folds),"feature_schema_changed":False,"oracle_labels_changed":False,"new_states":0,"new_oracle_rollouts":0,"correction_head_trained":False,"closed_loop_run":False}
    inv.base.write_json(OUT/"sanity_checks.json",checks); times=[float(json.loads(p.read_text())["elapsed_s"]) for p in done]; inv.base.write_json(OUT/"runtime_statistics.json",{"started_utc":datetime.now(timezone.utc).isoformat(),"selection_and_selected_evaluation_wall_s":time.perf_counter()-started,"candidate_training_runs":expected,"sum_candidate_task_wall_s":float(sum(times)),"GPU_shards_candidate_training_peak":1,"CPU_threads_per_shard":4,"new_states":0,"new_oracle_rollouts":0}); inv.base.write_json(OUT/"manifest.json",{"condition":CONDITION,"baseline_readiness":ready,"lambda_grid":list(inv.LAMBDA_GRID),"selected_lambda_by_fold":chosen,"selection":"validation only","outputs":["hyperparameter_selection.csv","oof_predictions.csv","hard_state_variant_predictions.csv","training_loss_history.csv","conditional_group_class_statistics.csv","selected_checkpoint_manifest.csv"]})
    print({"status":"CONDITIONAL_INVARIANCE_TRAINING_COMPLETE","selected_lambda_by_fold":chosen,"all_stable":allm,"recovery":recovery,"difficult":diff},flush=True)


if __name__=="__main__": main()
