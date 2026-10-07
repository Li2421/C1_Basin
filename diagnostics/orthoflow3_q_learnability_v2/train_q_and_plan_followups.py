#!/usr/bin/env python3
"""Consolidate the frozen rollout table, train diagnostic Q, and freeze follow-ups."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import kendalltau, spearmanr


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_q_learnability_v2"
ETA_LOW = np.asarray([0.5, -0.5, 0.0], dtype=np.float64)
ETA_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
ETA_CENTER = (ETA_LOW + ETA_HIGH) / 2
ETA_SCALE = ETA_HIGH - ETA_LOW
SEEDS = (17, 23, 41)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(plan_stem: str) -> list[dict]:
    result = []
    directory = HERE / "raw" / plan_stem
    for path in sorted(directory.glob("shard*.jsonl")):
        if not path.stem.removeprefix("shard").isdigit():
            continue
        result.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    unique = {row["task_id"]: row for row in result}
    if len(unique) != len(result):
        raise RuntimeError((plan_stem, "duplicate task IDs", len(result), len(unique)))
    return list(unique.values())


class QMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(256)(x))
        x = nn.silu(nn.Dense(256)(x))
        return nn.Dense(1)(x).squeeze(-1)


class Logistic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x).squeeze(-1)


def bce(logits, s, n):
    return (n * nn.softplus(logits) - s * logits).sum() / n.sum()


def train(model, x_train, s_train, n_train, x_val, s_val, n_val, seed, *, epochs=2000, patience=120):
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(x_train[:1]))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, x, s, n):
        loss, grads = jax.value_and_grad(lambda p: bce(model.apply(p, x), s, n))(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    best = None; best_epoch = -1; best_val = math.inf; wait = 0; history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            params, state, loss = step(params, state, jnp.asarray(x_train[idx]), jnp.asarray(s_train[idx]), jnp.asarray(n_train[idx]))
            losses.append(float(loss))
        val = float(bce(model.apply(params, jnp.asarray(x_val)), jnp.asarray(s_val), jnp.asarray(n_val)))
        history.append({"seed": seed, "epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": val})
        if val < best_val - 1e-7:
            best_val = val; best_epoch = epoch; best = jax.tree_util.tree_map(lambda value: np.asarray(value), params); wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    return best, best_val, best_epoch, history


def predict(model, params, x):
    return np.asarray(jax.nn.sigmoid(model.apply(params, jnp.asarray(x))), dtype=np.float64)


def balanced_accuracy(y, pred):
    pos = y == 1; neg = ~pos
    return 0.5 * ((pred[pos].mean() if pos.any() else 0.0) + ((~pred[neg]).mean() if neg.any() else 0.0))


def main() -> None:
    started = time.monotonic()
    base = load_records("base_rollout_plan")
    plan = json.loads((HERE / "base_rollout_plan.json").read_text())
    if len(base) != len(plan["tasks"]):
        raise RuntimeError(("incomplete base rollout", len(base), len(plan["tasks"])))
    state_manifest = json.loads((HERE / "eligible_state_manifest.json").read_text())
    states = state_manifest["selected_states"]
    state_by_id = {row["state_id"]: row for row in states}
    h_all = np.asarray(np.load(HERE / "conditioning_features.npz")["features"], dtype=np.float64)

    by_candidate = defaultdict(list)
    for row in base:
        by_candidate[row["candidate_id"]].append(row)
    candidates = []
    for candidate_id, rows in sorted(by_candidate.items()):
        first = rows[0]; counts = Counter(row["outcome"] for row in rows)
        state = state_by_id[first["state_id"]]
        candidates.append({
            "candidate_id": candidate_id, "state_id": first["state_id"], "source_group": first["source_group"],
            "split": first["split"], "h_conditioning_identifier": first["h_conditioning_identifier"],
            "probe_id": first["probe_id"], "eta1": first["eta"][0], "eta2": first["eta"][1], "eta3": first["eta"][2],
            "trial_count": len(rows), "success_count": counts["success"], "safe_deadlock": counts["safe_deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"], "other_numerical": counts["other_numerical"],
            "seed_ids": ";".join(str(row["future_index"]) for row in sorted(rows, key=lambda item: item["future_index"])),
            "feature_index": state["feature_index"],
        })
    with (HERE / "rollout_dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)

    train_rows = [row for row in candidates if row["split"] == "train"]
    histogram = Counter(f"{row['success_count']}/4" for row in train_rows)
    state_diversity = {}
    for state_id in sorted({row["state_id"] for row in train_rows}):
        values = [row["success_count"] for row in train_rows if row["state_id"] == state_id]
        state_diversity[state_id] = {"min_success_count": min(values), "max_success_count": max(values), "has_success_and_failure_eta": min(values) < 4 and max(values) > 0}
    informative_fraction = np.mean([value["has_success_and_failure_eta"] for value in state_diversity.values()])
    prevalence = sum(row["success_count"] for row in train_rows) / sum(row["trial_count"] for row in train_rows)
    dominant_extreme = max(histogram.get("0/4", 0), histogram.get("4/4", 0)) / len(train_rows)
    informative = dominant_extreme < 0.95 and informative_fraction >= 0.10
    info = {
        "candidate_histogram": {key: histogram.get(key, 0) for key in ("0/4", "1/4", "2/4", "3/4", "4/4")},
        "train_candidates": len(train_rows), "train_trial_success_prevalence": prevalence,
        "states_with_both_successful_and_unsuccessful_eta": int(sum(value["has_success_and_failure_eta"] for value in state_diversity.values())),
        "fraction_states_with_both": float(informative_fraction), "dominant_extreme_fraction": dominant_extreme,
        "predeclared_informative_rule": "dominant all-fail/all-success bin <95% and >=10% states have both successful and unsuccessful eta",
        "informative": bool(informative), "per_state": state_diversity,
    }
    write_json(HERE / "data_informativeness.json", info)
    if not informative:
        write_json(HERE / "q_decision.json", {"classification": "Q_DATASET_UNINFORMATIVE", "training_performed": False})
        print(json.dumps(info, indent=2)); return

    train_state_indices = [row["feature_index"] for row in states if row["split"] == "train"]
    h_mean = h_all[train_state_indices].mean(axis=0)
    h_std = h_all[train_state_indices].std(axis=0)
    frozen_std = np.where(h_std < 1e-8, 1.0, h_std)
    normalization = {
        "h_mean": h_mean.tolist(), "h_std": frozen_std.tolist(), "zero_variance_dimensions": np.flatnonzero(h_std < 1e-8).tolist(),
        "eta_center": ETA_CENTER.tolist(), "eta_scale": ETA_SCALE.tolist(),
        "eta_rule": "(eta-domain-center)/(authoritative high-low); explicit eta=0 retained without clipping",
        "fit_on": "TRAIN states only",
    }
    write_json(HERE / "normalization.json", normalization)

    def matrix(rows):
        h = np.stack([(h_all[row["feature_index"]] - h_mean) / frozen_std for row in rows])
        eta = (np.asarray([[row["eta1"], row["eta2"], row["eta3"]] for row in rows]) - ETA_CENTER) / ETA_SCALE
        x = np.concatenate([h, eta], axis=1).astype(np.float32)
        s = np.asarray([row["success_count"] for row in rows], dtype=np.float32)
        n = np.asarray([row["trial_count"] for row in rows], dtype=np.float32)
        return x, s, n
    split_rows = {split: [row for row in candidates if row["split"] == split] for split in ("train", "val", "test")}
    data = {split: matrix(rows) for split, rows in split_rows.items()}
    model = QMLP(); training_history = []; seed_results = []; checkpoints = {}
    train_started = time.monotonic()
    for seed in SEEDS:
        params, val_nll, epoch, history = train(model, *data["train"], *data["val"], seed)
        path = HERE / f"q_seed{seed}.msgpack"; path.write_bytes(serialization.to_bytes(params)); checkpoints[seed] = (params, path)
        training_history.extend(history)
        seed_results.append({"seed": seed, "best_val_nll": val_nll, "best_epoch": epoch, "checkpoint": str(path), "checkpoint_sha256": sha(path)})
    selected = min(seed_results, key=lambda row: (row["best_val_nll"], row["seed"]))
    params, selected_path = checkpoints[selected["seed"]]
    with (HERE / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_history[0])); writer.writeheader(); writer.writerows(training_history)
    write_json(HERE / "training_seed_results.json", {"seeds": seed_results})
    write_json(HERE / "selected_checkpoint.json", selected)
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    write_json(HERE / "model_config.json", {
        "architecture": "217->256->256->1", "activations": "SiLU", "output": "logit/sigmoid",
        "parameter_count": int(parameter_count), "loss": "candidate-level binomial NLL", "optimizer": "Adam lr=1e-3",
        "early_stopping": "VAL NLL only; patience=120; max_epochs=2000", "training_seeds": list(SEEDS),
    })

    # Cheap baselines and held-out metrics are frozen before any follow-up outcomes.
    xtr, str_, ntr = data["train"]; xv, sv, nv = data["val"]; xt, st, nt = data["test"]
    logistic, log_val, log_epoch, _ = train(Logistic(), xtr, str_, ntr, xv, sv, nv, 17, epochs=1200, patience=100)
    ptest = predict(model, params, xt)
    empirical = st / nt
    eps = 1e-9
    test_nll = float(np.sum(-st * np.log(ptest + eps) - (nt-st) * np.log(1-ptest + eps)) / np.sum(nt))
    brier = float(np.sum(nt * (ptest - empirical) ** 2) / np.sum(nt))
    const_nll = float(np.sum(-st*np.log(prevalence+eps)-(nt-st)*np.log(1-prevalence+eps))/np.sum(nt))
    logp = predict(Logistic(), logistic, xt)
    log_nll = float(np.sum(-st*np.log(logp+eps)-(nt-st)*np.log(1-logp+eps))/np.sum(nt))
    calibration = []
    for lo in np.linspace(0, .8, 5):
        mask = (ptest >= lo) & (ptest < lo + .2 if lo < .8 else ptest <= 1)
        if mask.any(): calibration.append({"lower": float(lo), "upper": float(lo+.2), "candidates": int(mask.sum()), "mean_predicted": float(ptest[mask].mean()), "empirical_success": float(np.sum(st[mask])/np.sum(nt[mask]))})
    write_json(HERE / "heldout_probability_metrics.json", {
        "test_binomial_nll_per_trial": test_nll, "test_brier_candidate_q": brier,
        "constant_train_prevalence": prevalence, "constant_test_nll": const_nll,
        "logistic_best_val_nll": log_val, "logistic_best_epoch": log_epoch, "logistic_test_nll": log_nll,
        "calibration_bins": calibration,
    })

    rankings = []; top_rows = []
    offset = 0
    for state in [row for row in states if row["split"] == "test"]:
        rows = [row for row in split_rows["test"] if row["state_id"] == state["state_id"]]
        idx = [split_rows["test"].index(row) for row in rows]
        pred = ptest[idx]; true = np.asarray([row["success_count"]/row["trial_count"] for row in rows])
        rho = spearmanr(pred, true).statistic; tau = kendalltau(pred, true).statistic
        rho = 0.0 if not np.isfinite(rho) else float(rho); tau = 0.0 if not np.isfinite(tau) else float(tau)
        rankings.append({"state_id": state["state_id"], "source_group": state["source_group"], "spearman": rho, "kendall": tau, "true_q_variance": float(np.var(true))})
        choice = int(np.argmax(pred)); best = float(np.max(true)); random_expect = float(np.mean(true))
        top_rows.append({"state_id": state["state_id"], "selected_probe_id": rows[choice]["probe_id"], "predicted_q": float(pred[choice]), "true_q8": float(true[choice]), "best_q8": best, "gap_to_best": best-float(true[choice]), "random_candidate_expectation": random_expect})
    for path, rows in ((HERE/"heldout_statewise_ranking.csv", rankings), (HERE/"top_candidate_quality.csv", top_rows)):
        with path.open("w", newline="") as handle:
            writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # VAL-only zero threshold: predict the screening label success=4/4.
    pval = predict(model, params, xv)
    val_zero_idx = [i for i,row in enumerate(split_rows["val"]) if row["probe_id"] == "zero"]
    yval = np.asarray([split_rows["val"][i]["success_count"] == 4 for i in val_zero_idx])
    thresholds = np.linspace(0, 1, 1001)
    threshold = min(thresholds, key=lambda value: (-balanced_accuracy(yval, pval[val_zero_idx] >= value), value))

    # Freeze zero robust follow-up and gradient cases without using new true outcomes.
    test_states = [row for row in states if row["split"] == "test"]
    zero_tasks = []
    for state in test_states[:8]:
        for future_index in range(8, 64):
            zero_tasks.append({"task_id": f"zero64__{state['state_id']}__f{future_index:02d}", "candidate_id": f"{state['state_id']}__zero", "state_id": state["state_id"], "split": "test", "probe_id": "zero", "eta": [0.,0.,0.], "future_index": future_index})

    def norm_input(state, eta):
        h = ((h_all[state["feature_index"]] - h_mean) / frozen_std).astype(np.float32)
        e = ((np.asarray(eta)-ETA_CENTER)/ETA_SCALE).astype(np.float32)
        return np.concatenate([h,e])
    def q_and_grad(state, eta):
        hpart = jnp.asarray(((h_all[state["feature_index"]]-h_mean)/frozen_std).astype(np.float32))
        en = jnp.asarray(((np.asarray(eta)-ETA_CENTER)/ETA_SCALE).astype(np.float32))
        fn=lambda value: jax.nn.sigmoid(model.apply(params,jnp.concatenate([hpart,value])[None])[0])
        return float(fn(en)), np.asarray(jax.grad(fn)(en),dtype=np.float64)
    eta_by_probe = {row["probe_id"]: np.asarray([row["eta1"],row["eta2"],row["eta3"]]) for row in candidates if row["state_id"] == test_states[0]["state_id"]}
    gradient_cases=[]
    for state in test_states:
        choices=[]
        for probe_id, eta in eta_by_probe.items():
            if probe_id == "zero": continue
            q,g=q_and_grad(state,eta); gn=float(np.linalg.norm(g))
            if gn <= 1e-8: continue
            direction=g/gn; plus=eta + 0.05*direction*ETA_SCALE; minus=eta - 0.05*direction*ETA_SCALE
            if np.all(plus>=ETA_LOW) and np.all(plus<=ETA_HIGH) and np.all(minus>=ETA_LOW) and np.all(minus<=ETA_HIGH):
                choices.append((abs(q-.5),probe_id,eta,q,g,plus,minus))
        if choices:
            _,probe_id,eta,q,g,plus,minus=min(choices,key=lambda item:(item[0],item[1]))
            gradient_cases.append({"case_id": f"grad_{len(gradient_cases):02d}", "state_id":state["state_id"], "base_probe_id":probe_id, "eta_base":eta.tolist(), "eta_plus":plus.tolist(), "eta_minus":minus.tolist(), "predicted_base":q, "gradient_normalized_eta":g.tolist(), "gradient_norm":float(np.linalg.norm(g)), "clipping":False})
        if len(gradient_cases)==6: break
    grad_tasks=[]
    for case in gradient_cases:
        for role,eta,start in (("minus",case["eta_minus"],0),("base",case["eta_base"],8),("plus",case["eta_plus"],0)):
            stop=32
            for future_index in range(start,stop):
                grad_tasks.append({"task_id":f"{case['case_id']}__{role}__f{future_index:02d}","candidate_id":f"{case['case_id']}__{role}","state_id":case["state_id"],"split":"test","probe_id":f"{case['case_id']}__{role}","eta":eta,"future_index":future_index,"gradient_case_id":case["case_id"],"gradient_role":role})
    common_plan={"orthoflow3_sha256":"51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38","future_root_seed":plan["future_root_seed"],"conditioning":plan["conditioning"],"frozen_after_q_before_true_followup_outcomes":True}
    write_json(HERE/"zero_followup_plan.json",{**common_plan,"schema":"orthoflow3_q_v2_zero64_plan","tasks":zero_tasks,"maximum_new_continuations":len(zero_tasks)})
    write_json(HERE/"gradient_followup_plan.json",{**common_plan,"schema":"orthoflow3_q_v2_gradient_plan","tasks":grad_tasks,"maximum_new_continuations":len(grad_tasks)})
    write_json(HERE/"gradient_test_selection.json",{"alpha_normalized_eta":0.05,"gradient_tolerance":1e-8,"selection":"first 6 eligible TEST states; within state frozen probe predicted closest to 0.5","cases":gradient_cases})
    frozen={"checkpoint":str(selected_path),"checkpoint_sha256":sha(selected_path),"normalization_sha256":sha(HERE/'normalization.json'),"zero_feasibility_threshold":float(threshold),"threshold_selected_on":"VAL eta=0 4/4 screening label","gradient_alpha":.05,"gradient_cases":len(gradient_cases),"frozen_before_followup":True}
    write_json(HERE/"frozen_q_manifest.json",frozen)
    write_json(HERE/"pre_followup_summary.json",{
        "train_informativeness":info,"selected_checkpoint":selected,"test_nll":test_nll,"test_brier":brier,
        "ranking_mean_spearman":float(np.mean([r['spearman'] for r in rankings])),"ranking_median_spearman":float(np.median([r['spearman'] for r in rankings])),
        "ranking_mean_kendall":float(np.mean([r['kendall'] for r in rankings])),"ranking_median_kendall":float(np.median([r['kendall'] for r in rankings])),
        "zero_followup_new":len(zero_tasks),"gradient_followup_new":len(grad_tasks),"training_seconds":time.monotonic()-train_started,"script_wall_seconds":time.monotonic()-started,
    })
    print(json.dumps(json.loads((HERE/"pre_followup_summary.json").read_text()),indent=2))


if __name__ == "__main__":
    main()
