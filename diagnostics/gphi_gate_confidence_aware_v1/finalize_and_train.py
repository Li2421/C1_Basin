"""Finalize confidence labels and train only a binary gate on stable states."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
PREVIOUS = ROOT / "diagnostics/gphi_gate_feasibility_v1"
ORACLE = ROOT / "diagnostics/oracle_boundary_confidence_audit"
FEATURE_AUDIT = ROOT / "diagnostics/stable_oracle_feature_audit"
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, np.ndarray): return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert)+"\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return center-half, center+half


def p64(p: float) -> float:
    return float(p**64 + 64*(1-p)*p**63)


def sigmoid(x):
    x = np.asarray(x, np.float64)
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum()); neg = len(y)-pos
    if pos == 0 or neg == 0: return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum()-pos*(pos+1)/2)/(pos*neg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0: return float("nan")
    order = np.argsort(-p, kind="stable"); yy = y[order]; pp = p[order]
    tp = fp = 0; previous_recall = 0.; ap = 0.
    for end in np.r_[np.flatnonzero(pp[1:] != pp[:-1])+1, len(y)]:
        start = tp+fp; group_positive = int(np.sum(yy[start:end]))
        tp += group_positive; fp += int(end-start-group_positive)
        recall = tp/pos; precision = tp/max(tp+fp, 1)
        ap += (recall-previous_recall)*precision; previous_recall = recall
    return float(ap)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1)&(y == 1))); tn = int(np.sum((pred == 0)&(y == 0)))
    fp = int(np.sum((pred == 1)&(y == 0))); fn = int(np.sum((pred == 0)&(y == 1)))
    recall = tp/max(tp+fn, 1); specificity = tn/max(tn+fp, 1); precision = tp/max(tp+fp, 1)
    return {
        "state_count": len(y), "stable_zero": int(np.sum(y == 0)), "stable_nonzero": int(np.sum(y == 1)),
        "threshold": threshold, "balanced_accuracy": (recall+specificity)/2,
        "AUROC": auroc(y, p), "AUPRC": auprc(y, p), "accuracy": float(np.mean(pred == y)),
        "precision": precision, "recall": recall, "specificity": specificity,
        "FPR": fp/max(fp+tn, 1), "FNR": fn/max(fn+tp, 1), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Brier": float(np.mean((p-y)**2)), "mean_probability": float(np.mean(p)),
    }


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    unique = np.unique(p)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1]+unique[1:])/2, np.nextafter(unique[-1], np.inf), .5]
    rows = []
    for threshold in np.unique(candidates): rows.append(classification_metrics(y, p, float(threshold)))
    best = max(rows, key=lambda row: (row["balanced_accuracy"], -max(row["FPR"], row["FNR"]), -abs(row["FPR"]-row["FNR"]), -abs(row["threshold"]-.5)))
    return float(best["threshold"]), rows


def aggregate_state(probability: np.ndarray, ids: np.ndarray, labels: dict[str, int]):
    grouped = defaultdict(list)
    for value, state_id in zip(probability, ids): grouped[str(state_id)].append(float(value))
    state_ids = np.asarray(sorted(grouped), dtype=str)
    p = np.asarray([np.mean(grouped[state_id]) for state_id in state_ids])
    y = np.asarray([labels[state_id] for state_id in state_ids], dtype=int)
    return state_ids, p, y


def state_bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1-1e-7)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))


def init_params(key, dimensions: list[int]):
    params = []; keys = jax.random.split(key, len(dimensions)-1)
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        limit = math.sqrt(6/(fan_in+fan_out))
        params.append({"w": jax.random.uniform(layer_key, (fan_in, fan_out), minval=-limit, maxval=limit), "b": jnp.zeros((fan_out,), jnp.float32)})
    return params


def logits(params, x):
    value = x
    for layer in params[:-1]: value = jax.nn.silu(value@layer["w"]+layer["b"])
    return (value@params[-1]["w"]+params[-1]["b"]).reshape(-1)


def predict(params, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.sigmoid(logits(params, jnp.asarray(x, jnp.float32))))


def normalization(features: np.ndarray, schema: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = np.zeros(214, bool)
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            start = int(segment["offset"]); binary[start:start+int(segment["length"])] = True
    mean = features.mean(0); scale = features.std(0); scale[scale < 1e-8] = 1.
    mean[binary] = 0.; scale[binary] = 1.
    return mean, scale, binary


def train_model(name: str, hidden: list[int], seed: int, x_train: np.ndarray, y_train: np.ndarray,
                train_ids: np.ndarray, x_val: np.ndarray, val_ids: np.ndarray, labels: dict[str, int]):
    started = time.monotonic(); params = init_params(jax.random.PRNGKey(seed), [x_train.shape[1], *hidden, 1])
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-5); opt_state = optimizer.init(params)

    @jax.jit
    def step(current, state):
        def loss_fn(candidate):
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits(candidate, jnp.asarray(x_train, jnp.float32)), jnp.asarray(y_train, jnp.float32)))
        loss, grads = jax.value_and_grad(loss_fn)(current); updates, state = optimizer.update(grads, state, current)
        return optax.apply_updates(current, updates), state, loss

    best = params; best_bce = float("inf"); best_epoch = 0; stale = 0; history = []
    for epoch in range(1, 1201):
        params, opt_state, train_loss = step(params, opt_state)
        if epoch == 1 or epoch % 10 == 0:
            train_state = aggregate_state(predict(params, x_train), train_ids, labels)
            val_state = aggregate_state(predict(params, x_val), val_ids, labels)
            val_bce = state_bce(val_state[2], val_state[1])
            history.append({"run": name, "seed": seed, "epoch": epoch, "sample_train_BCE": float(train_loss),
                            "train_state_BCE": state_bce(train_state[2], train_state[1]), "validation_state_BCE": val_bce,
                            "validation_AUROC": auroc(val_state[2], val_state[1]), "elapsed_s": time.monotonic()-started})
            if val_bce < best_bce-1e-6:
                best_bce = val_bce; best_epoch = epoch; best = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params); stale = 0
            else: stale += 1
            if stale >= 35: break
    return best, {"name": name, "hidden": hidden, "seed": seed, "best_epoch": best_epoch,
                  "validation_state_BCE": best_bce, "runtime_s": time.monotonic()-started,
                  "parameter_count": int(sum(np.prod(v.shape) for layer in best for v in layer.values()))}, history


def correlation(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return {"Pearson": None, "Spearman": None}
    return {"Pearson": float(np.corrcoef(a, b)[0, 1]), "Spearman": float(spearmanr(a, b).statistic)}


def main() -> None:
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    source_paths = {
        "dataset_manifest": DATA/"manifest.json", "dataset_samples": DATA/"samples.npz", "dataset_states": DATA/"state_manifest.jsonl",
        "previous_gate_manifest": PREVIOUS/"manifest.json", "oracle_audit_manifest": ORACLE/"manifest.json", "feature_audit_manifest": FEATURE_AUDIT/"manifest.json",
    }
    before = {name: sha(path) for name, path in source_paths.items()}
    state_manifest = {row["state_id"]: row for row in read_jsonl(HERE/"confidence_state_manifest.jsonl")}
    paths = [HERE/"existing_eta_zero_rollouts.jsonl"]
    paths += [HERE/f"raw/to256_shard{shard}/records.jsonl" for shard in range(4)]
    paths += [HERE/f"raw/to512_shard{shard}/records.jsonl" for shard in range(4) if (HERE/f"raw/to512_shard{shard}/records.jsonl").exists()]
    effective = {}; duplicate = 0
    for path in paths:
        for row in read_jsonl(path):
            key = (row["state_id"], int(row["seed"]))
            if key in effective:
                duplicate += 1; old = effective[key]
                if (old["outcome"], int(old["steps"])) != (row["outcome"], int(row["steps"])): raise RuntimeError(("conflicting tuple", key))
                continue
            effective[key] = row
    by_state = defaultdict(list)
    for (state_id, _), row in effective.items(): by_state[state_id].append(row)
    prior_stability = {row["state_id"]: row for row in csv.DictReader((ORACLE/"b63_resampling_stability.csv").open())}
    confidence_rows = []; label = {}; confidence_class = {}; prior_mismatch = []
    for state_id, state in state_manifest.items():
        rows = by_state[state_id]; n = len(rows)
        if n not in (256, 512): raise AssertionError((state_id, n))
        successes = sum(row["outcome"] == "success" for row in rows); estimate = successes/n; probability = p64(estimate)
        original = int(state["original_gate_label"]); reproduce = probability if original == 0 else 1-probability
        stable = reproduce >= .95
        klass = "ORACLE_STABLE_ZERO" if stable and original == 0 else "ORACLE_STABLE_NONZERO" if stable else "ORACLE_AMBIGUOUS"
        lower, upper = wilson(successes, n); outcomes = Counter(row["outcome"] for row in rows)
        record = {**state, "n": n, "successes": successes, "failures": n-successes, "Q0": estimate,
                  "wilson95_lower": lower, "wilson95_upper": upper, "P64_B63": probability,
                  "original_label_reproduction_probability": reproduce, "original_label_flip_probability": 1-reproduce,
                  "oracle_confidence_class": klass, **{f"outcome_{key}": outcomes[key] for key in ("success","deadlock","timeout","collision","execution_error")}}
        confidence_rows.append(record); confidence_class[state_id] = klass
        if stable: label[state_id] = original
        if state_id in prior_stability:
            previous = prior_stability[state_id]
            if (previous["original_label_stable_at_95pct"] == "True") != stable or abs(float(previous["p_hat_Q0"])-estimate) > 1e-15:
                prior_mismatch.append(state_id)
    if prior_mismatch: raise RuntimeError(("prior stability mismatch", prior_mismatch))
    write_csv(HERE/"oracle_confidence_dataset.csv", confidence_rows)
    write_csv(HERE/"stable_train_states.csv", [row for row in confidence_rows if row["oracle_confidence_class"] != "ORACLE_AMBIGUOUS" and row["split"] == "train"])

    split_counts = {}
    for split in ("train","validation","test"):
        rows = [row for row in confidence_rows if row["split"] == split]
        split_counts[split] = {
            "total": len(rows), "stable_zero": sum(row["oracle_confidence_class"] == "ORACLE_STABLE_ZERO" for row in rows),
            "stable_nonzero": sum(row["oracle_confidence_class"] == "ORACLE_STABLE_NONZERO" for row in rows),
            "ambiguous": sum(row["oracle_confidence_class"] == "ORACLE_AMBIGUOUS" for row in rows),
            "confidence_by_category": {klass: dict(Counter(row["category"] for row in rows if row["oracle_confidence_class"] == klass)) for klass in ("ORACLE_STABLE_ZERO","ORACLE_STABLE_NONZERO","ORACLE_AMBIGUOUS")},
        }
    groups = {split: {row["source_group"] for row in confidence_rows if row["split"] == split} for split in ("train","validation","test")}
    overlaps = {"train_validation": sorted(groups["train"]&groups["validation"]), "train_test": sorted(groups["train"]&groups["test"]), "validation_test": sorted(groups["validation"]&groups["test"])}
    write_json(HERE/"split_manifest.json", {"assignment": "Dataset V4 source-group-aware split preserved", "counts": split_counts, "source_group_overlap": overlaps,
                                             "ambiguous_used_for_training": False, "ambiguous_used_for_model_selection": False})
    if any(overlaps.values()): raise RuntimeError(("source group leakage", overlaps))

    with np.load(DATA/"samples.npz", allow_pickle=False) as source: arrays = {key: np.asarray(source[key]) for key in source.files}
    schema = read_json(DATA/"feature_schema.json")
    stable_ids = set(label); ambiguous_ids = set(state_manifest)-stable_ids
    split_ids = {split: sorted(state_id for state_id in stable_ids if state_manifest[state_id]["split"] == split) for split in ("train","validation","test")}
    if any(not split_ids[split] for split in split_ids): raise RuntimeError(("empty stable split", {k:len(v) for k,v in split_ids.items()}))
    sample_stable = np.isin(arrays["state_id"], list(stable_ids)); sample_y = np.asarray([label.get(str(state_id), -1) for state_id in arrays["state_id"]])
    split_index = {split: np.flatnonzero(sample_stable & (arrays["split"] == split)) for split in ("train","validation","test")}
    norm_mean, norm_scale, binary = normalization(arrays["features"][split_index["train"]], schema)
    normalized = ((arrays["features"]-norm_mean)/norm_scale).astype(np.float32)
    write_json(HERE/"normalization.json", {"fit_states": "train ORACLE_STABLE states only", "stable_train_state_count": len(split_ids["train"]),
                                             "mean": norm_mean, "scale": norm_scale, "binary_feature_indices": np.flatnonzero(binary), "epsilon": 1e-8})

    specifications = [("LINEAR", [], 17)]
    specifications += [("MLP_64x64", [64,64], seed) for seed in (17,23,41)]
    specifications += [("MLP_128x128", [128,128], 17)]
    trained = []; histories = []
    for name, hidden, seed in specifications:
        params, summary, history = train_model(name, hidden, seed,
            normalized[split_index["train"]], sample_y[split_index["train"]], arrays["state_id"][split_index["train"]],
            normalized[split_index["validation"]], arrays["state_id"][split_index["validation"]], label)
        state_predictions = {}; metrics_by_split = {}
        for split, index in split_index.items():
            state_predictions[split] = aggregate_state(predict(params, normalized[index]), arrays["state_id"][index], label)
        threshold, threshold_rows = select_threshold(state_predictions["validation"][2], state_predictions["validation"][1])
        for split, (_, probability, yy) in state_predictions.items(): metrics_by_split[split] = classification_metrics(yy, probability, threshold)
        trained.append({"name": name, "hidden": hidden, "seed": seed, "params": params, "summary": summary,
                        "predictions": state_predictions, "threshold": threshold, "threshold_rows": threshold_rows, "metrics": metrics_by_split})
        histories.extend(history)
        print(json.dumps({"model": name, "seed": seed, "epoch": summary["best_epoch"], "val_AUROC": metrics_by_split["validation"]["AUROC"], "val_balanced_accuracy": metrics_by_split["validation"]["balanced_accuracy"]}), flush=True)
    best_by_architecture = []
    for name in ("LINEAR","MLP_64x64","MLP_128x128"):
        candidates = [run for run in trained if run["name"] == name]
        best_by_architecture.append(max(candidates, key=lambda run: (run["metrics"]["validation"]["AUROC"], run["metrics"]["validation"]["balanced_accuracy"], -run["summary"]["validation_state_BCE"])))
    best = max(best_by_architecture, key=lambda run: (run["metrics"]["validation"]["AUROC"], run["metrics"]["validation"]["balanced_accuracy"], -run["summary"]["validation_state_BCE"]))

    # Majority baseline uses state prevalence; model selection and thresholding never see test or ambiguous states.
    train_labels = np.asarray([label[state_id] for state_id in split_ids["train"]]); prior = float(np.mean(train_labels))
    val_labels = np.asarray([label[state_id] for state_id in split_ids["validation"]]); prior_threshold, _ = select_threshold(val_labels, np.full(len(val_labels), prior))
    comparison = []
    for model_name, run in [("MAJORITY", None)]+[(run["name"],run) for run in best_by_architecture]:
        row = {"model": model_name, "seed": "" if run is None else run["seed"], "selected": run is best,
               "parameter_count": 0 if run is None else run["summary"]["parameter_count"],
               "validation_selected_threshold": prior_threshold if run is None else run["threshold"]}
        for split in ("train","validation","test"):
            if run is None:
                yy = np.asarray([label[state_id] for state_id in split_ids[split]]); pp = np.full(len(yy), prior); mm = classification_metrics(yy,pp,prior_threshold)
            else: mm = run["metrics"][split]
            for key in ("balanced_accuracy","AUROC","AUPRC","accuracy","precision","recall","specificity","FPR","FNR","Brier"):
                row[f"{split}_{key}"] = mm[key]
        comparison.append(row)
    write_csv(HERE/"model_comparison.csv", comparison); write_csv(HERE/"training_history.csv", histories)

    checkpoint = {"normalization_mean": norm_mean, "normalization_scale": norm_scale, "normalization_binary_mask": binary,
                  "architecture_json": np.asarray(json.dumps([214,*best["hidden"],1])), "threshold": np.asarray(best["threshold"]),
                  "oracle_label_definition": np.asarray("hard BCE only for original B63 labels with >=0.95 random-64 reproduction probability")}
    for index, layer in enumerate(best["params"]):
        checkpoint[f"layer_{index}_weight"] = np.asarray(layer["w"]); checkpoint[f"layer_{index}_bias"] = np.asarray(layer["b"])
    np.savez_compressed(HERE/"best_gate_checkpoint.npz", **checkpoint)

    best_state = {}
    for split, (ids, probabilities, yy) in best["predictions"].items():
        for state_id, probability in zip(ids, probabilities): best_state[str(state_id)] = float(probability)
    # Predict ambiguous states only after model/threshold selection is frozen.
    for state_id in ambiguous_ids:
        index = np.flatnonzero(arrays["state_id"] == state_id)
        best_state[state_id] = float(np.mean(predict(best["params"], normalized[index])))

    stable_test = best["metrics"]["test"]
    difficult_ids = {row["state_id"] for row in csv.DictReader((ORACLE/"b63_resampling_stability.csv").open()) if row["original_label_stable_at_95pct"] == "True" and ("MATCHED_BOUNDARY" in row["audit_tags"] or "OLD_HARD_ZERO" in row["audit_tags"])}
    difficult_test_ids = sorted(difficult_ids & set(split_ids["test"])); difficult_all_ids = sorted(difficult_ids)
    def subset_metrics(ids: list[str], probabilities: dict[str,float], threshold: float) -> dict:
        yy = np.asarray([label[state_id] for state_id in ids]); pp = np.asarray([probabilities[state_id] for state_id in ids])
        return classification_metrics(yy, pp, threshold)
    difficult_test = subset_metrics(difficult_test_ids, best_state, best["threshold"])
    write_json(HERE/"stable_test_metrics.json", {"model": best["name"], "seed": best["seed"], "threshold": best["threshold"], "metrics": stable_test, "state_ids": split_ids["test"]})
    write_json(HERE/"difficult_stable_metrics.json", {"definition": "prior-audited MATCHED_BOUNDARY or OLD_HARD state with stable original label", "test": difficult_test, "test_state_ids": difficult_test_ids, "all_difficult_stable_count": len(difficult_all_ids)})

    group_rows = [{"scope":"ALL_STABLE_UNSEEN_TEST_GROUPS","source_group":"ALL",**stable_test}, {"scope":"DIFFICULT_STABLE_UNSEEN_TEST_GROUPS","source_group":"ALL",**difficult_test}]
    test_ids_array, test_probability, test_y = best["predictions"]["test"]
    for group in sorted({state_manifest[str(state_id)]["source_group"] for state_id in test_ids_array}):
        mask = np.asarray([state_manifest[str(state_id)]["source_group"] == group for state_id in test_ids_array])
        group_rows.append({"scope":"STABLE_UNSEEN_TEST_GROUP","source_group":group,**classification_metrics(test_y[mask],test_probability[mask],best["threshold"])})
    write_csv(HERE/"source_group_holdout_metrics.csv", group_rows)

    previous_decision = read_json(PREVIOUS/"decision_metrics.json"); previous_threshold = float(previous_decision["threshold"])
    previous_probability = {}
    for row in csv.DictReader((PREVIOUS/"state_metrics.csv").open()):
        if row["model"] == previous_decision["selected_model"]: previous_probability[row["state_id"]] = float(row["p_gate"])
    matched_rows = []
    for state_id in split_ids["test"]:
        yy = label[state_id]; old = previous_probability[state_id]; new = best_state[state_id]
        matched_rows.append({"state_id":state_id,"category":state_manifest[state_id]["category"],"source_group":state_manifest[state_id]["source_group"],
                             "difficult_stable":state_id in difficult_ids,"oracle_gate_label":yy,
                             "previous_p_gate":old,"previous_threshold":previous_threshold,"previous_prediction":int(old>=previous_threshold),"previous_correct":int(old>=previous_threshold)==yy,
                             "new_p_gate":new,"new_threshold":best["threshold"],"new_prediction":int(new>=best["threshold"]),"new_correct":int(new>=best["threshold"])==yy})
    write_csv(HERE/"matched_previous_gate_comparison.csv", matched_rows)
    old_test = subset_metrics(split_ids["test"], previous_probability, previous_threshold)
    old_difficult = subset_metrics(difficult_test_ids, previous_probability, previous_threshold)

    ambiguous_rows = []
    confidence_by_id = {row["state_id"]:row for row in confidence_rows}
    for state_id in sorted(ambiguous_ids):
        row = confidence_by_id[state_id]; probability = best_state[state_id]
        ambiguous_rows.append({"state_id":state_id,"category":row["category"],"split":row["split"],"source_group":row["source_group"],
                               "original_gate_label":row["original_gate_label"],"Q0":row["Q0"],"P64_B63":row["P64_B63"],
                               "oracle_label_reproduction_probability":row["original_label_reproduction_probability"],"oracle_label_flip_probability":row["original_label_flip_probability"],
                               "p_gate_intervene":probability,"gate_confidence_abs_from_half":2*abs(probability-.5),
                               "gate_entropy_nats":float(-(probability*math.log(max(probability,EPS))+(1-probability)*math.log(max(1-probability,EPS))))})
    write_csv(HERE/"ambiguous_analysis.csv", ambiguous_rows)
    amb_p = np.asarray([row["p_gate_intervene"] for row in ambiguous_rows]); amb_q = np.asarray([row["Q0"] for row in ambiguous_rows])
    amb_flip = np.asarray([row["oracle_label_flip_probability"] for row in ambiguous_rows]); amb_stability = 1-amb_flip
    amb_conf = np.asarray([row["gate_confidence_abs_from_half"] for row in ambiguous_rows])
    stable_prob = np.asarray([best_state[state_id] for state_id in stable_ids]); stable_conf = 2*np.abs(stable_prob-.5)
    ambiguity_summary = [{
        "ambiguous_states":len(ambiguous_rows),"p_gate_mean":float(np.mean(amb_p)),"p_gate_median":float(np.median(amb_p)),
        "p_gate_p05":float(np.quantile(amb_p,.05)),"p_gate_p95":float(np.quantile(amb_p,.95)),
        "gate_confidence_mean_ambiguous":float(np.mean(amb_conf)),"gate_confidence_mean_stable":float(np.mean(stable_conf)),
        "p_gate_vs_Q0_Pearson":correlation(amb_p,amb_q)["Pearson"],"p_gate_vs_Q0_Spearman":correlation(amb_p,amb_q)["Spearman"],
        "gate_confidence_vs_oracle_stability_Pearson":correlation(amb_conf,amb_stability)["Pearson"],
        "gate_confidence_vs_oracle_stability_Spearman":correlation(amb_conf,amb_stability)["Spearman"],
        "expected_Q0_direction":"negative: high eta0 success should imply low intervene probability",
    }]
    write_csv(HERE/"ambiguity_probability_analysis.csv", ambiguity_summary)

    # Nested stable-state scaling; state ordering is deterministic and label-balanced, held-out groups unchanged.
    zero_train = sorted((state_id for state_id in split_ids["train"] if label[state_id] == 0), key=lambda value: hashlib.sha256(("scale:"+value).encode()).hexdigest())
    one_train = sorted((state_id for state_id in split_ids["train"] if label[state_id] == 1), key=lambda value: hashlib.sha256(("scale:"+value).encode()).hexdigest())
    scaling_rows = []
    for fraction in (.50,.75,1.0):
        selected = set(zero_train[:max(1,math.ceil(len(zero_train)*fraction))]+one_train[:max(1,math.ceil(len(one_train)*fraction))])
        if fraction == 1.0:
            probability_map = best_state; mm = stable_test; hard_mm = difficult_test; used_seed = best["seed"]
        else:
            train_index = np.flatnonzero(np.isin(arrays["state_id"], list(selected)))
            mean, scale, _ = normalization(arrays["features"][train_index], schema); scaled = ((arrays["features"]-mean)/scale).astype(np.float32)
            params, _, _ = train_model(f"SCALING_{fraction:.2f}", best["hidden"], best["seed"], scaled[train_index], sample_y[train_index], arrays["state_id"][train_index], scaled[split_index["validation"]], arrays["state_id"][split_index["validation"]], label)
            val_ids, val_p, val_y = aggregate_state(predict(params, scaled[split_index["validation"]]), arrays["state_id"][split_index["validation"]], label)
            threshold, _ = select_threshold(val_y,val_p)
            test_ids_s, test_p_s, test_y_s = aggregate_state(predict(params, scaled[split_index["test"]]), arrays["state_id"][split_index["test"]], label)
            mm = classification_metrics(test_y_s,test_p_s,threshold); probability_map = {str(i):float(p) for i,p in zip(test_ids_s,test_p_s)}
            hard_mm = subset_metrics(difficult_test_ids,probability_map,threshold); used_seed = best["seed"]
        scaling_rows.append({"train_fraction":fraction,"stable_train_states":len(selected),"stable_zero_train":sum(label[s] == 0 for s in selected),"stable_nonzero_train":sum(label[s] == 1 for s in selected),
                             "architecture":json.dumps([214,*best["hidden"],1]),"seed":used_seed,"heldout_stable_balanced_accuracy":mm["balanced_accuracy"],"heldout_stable_AUROC":mm["AUROC"],"heldout_stable_FPR":mm["FPR"],"heldout_stable_FNR":mm["FNR"],
                             "difficult_stable_balanced_accuracy":hard_mm["balanced_accuracy"],"difficult_stable_AUROC":hard_mm["AUROC"],"difficult_stable_FPR":hard_mm["FPR"],"difficult_stable_FNR":hard_mm["FNR"]})
    write_csv(HERE/"scaling_analysis.csv", scaling_rows)

    improved_difficult = difficult_test["balanced_accuracy"] >= old_difficult["balanced_accuracy"]+.05 or difficult_test["AUROC"] >= old_difficult["AUROC"]+.05
    scaling_improves = scaling_rows[-1]["difficult_stable_balanced_accuracy"] >= scaling_rows[0]["difficult_stable_balanced_accuracy"]+.05 or scaling_rows[-1]["difficult_stable_AUROC"] >= scaling_rows[0]["difficult_stable_AUROC"]+.05
    strong_global = stable_test["balanced_accuracy"] >= .80 and stable_test["AUROC"] >= .85
    strong_difficult = difficult_test["balanced_accuracy"] >= .75 and difficult_test["AUROC"] >= .75
    ambiguous_less_confident = float(np.mean(amb_conf))+.05 < float(np.mean(stable_conf))
    if improved_difficult and strong_global and strong_difficult and ambiguous_less_confident:
        conclusion = "NOISY_LABELS_WERE_PRIMARY_PROBLEM"
    elif strong_global and (improved_difficult or scaling_improves):
        conclusion = "STABLE_DATA_STILL_LIMITED"
    else:
        conclusion = "STABLE_BOUNDARY_STILL_NOT_LEARNABLE"
    ready = conclusion in ("NOISY_LABELS_WERE_PRIMARY_PROBLEM","STABLE_DATA_STILL_LIMITED") and strong_global and strong_difficult and max(difficult_test["FPR"],difficult_test["FNR"]) <= .25

    manifests256 = [read_json(HERE/f"raw/to256_shard{shard}/manifest.json") for shard in range(4)]
    manifests512 = [read_json(HERE/f"raw/to512_shard{shard}/manifest.json") for shard in range(4) if (HERE/f"raw/to512_shard{shard}/manifest.json").exists()]
    runtime = {"started_utc":started_utc,"finished_utc":datetime.now(timezone.utc).isoformat(),"analysis_and_training_wall_s":time.monotonic()-started,
               "new_continuation_rollouts":sum(x["new_rollouts"] for x in manifests256+manifests512),"new_physical_steps":sum(x["physical_steps"] for x in manifests256+manifests512),
               "to256_parallel_wall_s":max(x["elapsed_s"] for x in manifests256),"to512_parallel_wall_s":max((x["elapsed_s"] for x in manifests512),default=0.),
               "rollout_critical_path_s":max(x["elapsed_s"] for x in manifests256)+max((x["elapsed_s"] for x in manifests512),default=0.),
               "rollout_GPU_shards":4,"training_GPU_shards":1,"allocated_CPU_cores_rollout":12,"allocated_CPU_cores_training":6,
               "JAX_memory_fraction_rollout_per_process":.08,"python":sys.version,"platform":platform.platform()}
    write_json(HERE/"runtime_statistics.json",runtime)
    after = {name:sha(path) for name,path in source_paths.items()}
    sanity = {"passed":before == after and not duplicate and not prior_mismatch and not any(overlaps.values()),"prior_artifacts_unchanged":before == after,
              "duplicate_state_seed_tuples":duplicate,"prior_confidence_results_reproduced_exactly":not prior_mismatch,"source_group_leakage":overlaps,
              "oracle_modified":False,"B63_modified":False,"feature_schema_modified":False,"normalization_fit_on_train_stable_only":True,
              "ambiguous_used_in_BCE":False,"ambiguous_used_for_model_selection":False,"correction_head_trained":False,"learned_closed_loop_run":False,
              "checkpoint_output_dimension":1,"all_checkpoint_arrays_finite":all(np.isfinite(v).all() for v in checkpoint.values() if np.asarray(v).dtype.kind in "fc"),
              "classification":conclusion,"ready_for_gate_plus_correction_pilot":ready}
    write_json(HERE/"sanity_checks.json",sanity)
    if not sanity["passed"]: raise RuntimeError(sanity)

    class_counts = Counter(row["oracle_confidence_class"] for row in confidence_rows)
    category_counts = {klass:dict(Counter(row["category"] for row in confidence_rows if row["oracle_confidence_class"] == klass)) for klass in class_counts}
    decision = {"classification":conclusion,"ready_for_gate_plus_correction_pilot":ready,"confidence_counts":dict(class_counts),"category_counts":category_counts,
                "best_model":best["name"],"best_seed":best["seed"],"architecture":[214,*best["hidden"],1],"threshold":best["threshold"],
                "stable_test":stable_test,"difficult_stable_test":difficult_test,"previous_matched_stable_test":old_test,"previous_matched_difficult_test":old_difficult,
                "ambiguous_summary":ambiguity_summary[0],"scaling":scaling_rows,
                "smallest_next_experiment":"Collect additional generically sampled states from new source groups, confidence-audit eta=0, and repeat the same stable-only split; keep ambiguous states analysis-only." if conclusion == "STABLE_DATA_STILL_LIMITED" else
                  "Audit why stable held-out groups remain inseparable before any correction-head pilot." if conclusion == "STABLE_BOUNDARY_STILL_NOT_LEARNABLE" else
                  "A tiny offline gate-plus-frozen-correction integration check, not a formal closed-loop benchmark."}
    write_json(HERE/"decision_evidence.json",decision)

    report = f"""# Confidence-aware gate retraining

## Decision

**{conclusion}**  
**{'READY' if ready else 'NOT_READY'}_FOR_GATE_PLUS_CORRECTION_PILOT**

Hard BCE used only statistically stable original B_63 labels. Ambiguous states were excluded from normalization, training, validation selection, and primary scoring; they were retained for probability diagnostics.

## Confidence dataset

- States: {len(confidence_rows)} total = {class_counts['ORACLE_STABLE_ZERO']} stable zero / {class_counts['ORACLE_STABLE_NONZERO']} stable nonzero / {class_counts['ORACLE_AMBIGUOUS']} ambiguous.
- Category counts by confidence class: {category_counts}.
- Existing enlarged audit reproduced exactly for all 42 prior states. New continuation rollouts: {runtime['new_continuation_rollouts']}.
- Original Dataset V4 source-group split was preserved with no cross-split group leakage.

## Selected gate

- Architecture/seed: {[214,*best['hidden'],1]} / {best['seed']}.
- Validation-stable-selected threshold: {best['threshold']:.6f}.
- Stable test balanced accuracy / AUROC / AUPRC: {stable_test['balanced_accuracy']:.4f} / {stable_test['AUROC']:.4f} / {stable_test['AUPRC']:.4f}.
- Stable test FPR / FNR / Brier: {stable_test['FPR']:.4f} / {stable_test['FNR']:.4f} / {stable_test['Brier']:.4f}.
- Difficult-stable test balanced accuracy / AUROC / FPR / FNR: {difficult_test['balanced_accuracy']:.4f} / {difficult_test['AUROC']:.4f} / {difficult_test['FPR']:.4f} / {difficult_test['FNR']:.4f} ({len(difficult_test_ids)} states).

## Matched previous-gate comparison

- Previous -> new stable-test balanced accuracy: {old_test['balanced_accuracy']:.4f} -> {stable_test['balanced_accuracy']:.4f}; AUROC: {old_test['AUROC']:.4f} -> {stable_test['AUROC']:.4f}; Brier: {old_test['Brier']:.4f} -> {stable_test['Brier']:.4f}.
- Previous -> new difficult-stable balanced accuracy: {old_difficult['balanced_accuracy']:.4f} -> {difficult_test['balanced_accuracy']:.4f}; AUROC: {old_difficult['AUROC']:.4f} -> {difficult_test['AUROC']:.4f}.

## Ambiguous states and scaling

- Ambiguous p_gate median / 5-95%: {ambiguity_summary[0]['p_gate_median']:.4f} / [{ambiguity_summary[0]['p_gate_p05']:.4f}, {ambiguity_summary[0]['p_gate_p95']:.4f}].
- Mean gate confidence (ambiguous vs stable): {ambiguity_summary[0]['gate_confidence_mean_ambiguous']:.4f} vs {ambiguity_summary[0]['gate_confidence_mean_stable']:.4f}.
- Ambiguous p_gate vs Q0 Pearson/Spearman: {ambiguity_summary[0]['p_gate_vs_Q0_Pearson']:.4f} / {ambiguity_summary[0]['p_gate_vs_Q0_Spearman']:.4f}; the sensible direction is negative.
- Stable training scaling rows are recorded in `scaling_analysis.csv`; difficult-subset improvement from 50% to 100%: {scaling_improves}.

No correction head was trained and no learned closed-loop control was run.
"""
    (HERE/"confidence_aware_gate_report.md").write_text(report)
    required = ["confidence_aware_gate_report.md","oracle_confidence_dataset.csv","stable_train_states.csv","ambiguous_analysis.csv","split_manifest.json","model_comparison.csv","training_history.csv","stable_test_metrics.json","difficult_stable_metrics.json","source_group_holdout_metrics.csv","matched_previous_gate_comparison.csv","ambiguity_probability_analysis.csv","scaling_analysis.csv","best_gate_checkpoint.npz","normalization.json","decision_evidence.json","sanity_checks.json","runtime_statistics.json"]
    write_json(HERE/"manifest.json",{"study":"CONFIDENCE_AWARE_GATE_RETRAINING","classification":conclusion,"ready_for_gate_plus_correction_pilot":ready,
                                      "checkpoint":"best_gate_checkpoint.npz","dataset":str(DATA),"dataset_manifest_sha256":sha(DATA/"manifest.json"),"files_sha256":{name:sha(HERE/name) for name in required}})
    print(json.dumps({"classification":conclusion,"ready":ready,"counts":dict(class_counts),"best":best["name"],"seed":best["seed"],"stable_test":stable_test,"difficult":difficult_test,"runtime":runtime},indent=2),flush=True)


if __name__ == "__main__":
    main()
