#!/usr/bin/env python3
"""Train/evaluate a Toy+DB selector with scenario adapters and a shared mode head.

This script is deliberately rollout-free.  TEST outcome CSVs are opened only
by the ``evaluate`` stage, after all checkpoints and seed selections are frozen.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import rankdata

H = Path(__file__).parent
D = H.parent
TOY = D / "orthoflow3_shared_eta_codebook_v1"
DB = D / "orthoflow3_db_shared_mode_transfer_v1"
SEEDS = (17, 23, 41)
FRACTIONS = (25, 50, 100)
M = 12


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, rows, fields=None):
    path = H / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    path = H / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def metrics(logits, y):
    p = sigmoid(logits)
    eps = 1e-7
    nll = float(np.mean(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
    brier = float(np.mean((p - y) ** 2))
    ranks = []
    for a, b in zip(p, y):
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            ranks.append(float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
    top = np.argmax(p, axis=1)
    return {
        "nll": nll,
        "brier": brier,
        "top1_empirical_Q": float(np.mean(y[np.arange(len(y)), top])),
        "within_state_rank_correlation": float(np.mean(ranks)) if ranks else None,
    }


def selection_metrics(logits, y, b63):
    p = sigmoid(logits)
    top = np.argmax(p, axis=1)
    chosen_q = y[np.arange(len(y)), top]
    chosen_b = b63[np.arange(len(y)), top]
    oracle_q = y.max(axis=1)
    counts = np.bincount(top, minlength=M)
    nz = counts[counts > 0].astype(float)
    entropy = 0.0
    if len(nz) > 1:
        q = nz / nz.sum()
        entropy = float(-np.sum(q * np.log(q)) / np.log(M))
    return {
        **metrics(logits, y),
        "states": len(y),
        "B63_states": int(chosen_b.sum()),
        "B63_fraction": float(chosen_b.mean()),
        "mean_regret": float(np.mean(oracle_q - chosen_q)),
        "max_regret": float(np.max(oracle_q - chosen_q)),
        "selection_entropy": entropy,
        "mode_counts": {str(i): int(v) for i, v in enumerate(counts) if v},
    }


class JointModel(nn.Module):
    """Separate input adapters, shared trunk, shared canonical-mode head."""

    def setup(self):
        self.toy_adapter = nn.Dense(64)
        self.db_adapter = nn.Dense(64)
        self.shared = nn.Dense(64)
        self.head = nn.Dense(M)

    def __call__(self, x_toy, x_db):
        at = nn.silu(self.toy_adapter(x_toy))
        ad = nn.silu(self.db_adapter(x_db))
        zt = nn.silu(self.shared(at))
        zd = nn.silu(self.shared(ad))
        return self.head(zt), self.head(zd), zt, zd


class DBOnlyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(64)(x))
        x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(M)(x)


class ToyOnlyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(64)(x))
        x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(M)(x)


def feature_data(scenario):
    root = TOY if scenario == "toy" else DB
    z = np.load(root / "state_features.npz")
    return {
        "x": z["x"].astype(np.float32),
        "features": z["features"].astype(np.float64),
        "ids": np.asarray(z["state_ids"]).astype(str),
        "splits": np.asarray(z["splits"]).astype(str),
    }


def labels(scenario, split, allow_test=False):
    if split == "test" and not allow_test:
        raise RuntimeError("TEST outcomes are quarantined until evaluate stage")
    root = TOY if scenario == "toy" else DB
    fn = {"train": "train_mode_counts.csv", "val": "val_mode_counts.csv", "test": "test_mode_q64.csv"}[split]
    rows = read_csv(root / fn)
    f = feature_data(scenario)
    ids = f["ids"][f["splits"] == split]
    idx = {s: i for i, s in enumerate(ids)}
    y = np.zeros((len(ids), M), np.float32)
    n = np.zeros((len(ids), M), np.float32)
    b = np.zeros((len(ids), M), bool)
    for r in rows:
        i = idx[r["state_id"]]
        m = int(r["mode_id"])
        y[i, m] = float(r["successes"]) / float(r["trials"])
        n[i, m] = float(r["trials"])
        b[i, m] = str(r.get("B63", "false")).lower() == "true"
    if np.any(n <= 0):
        raise RuntimeError((scenario, split, "incomplete matrix"))
    return ids, y, n, b


def split_x(scenario, split):
    f = feature_data(scenario)
    return f["ids"][f["splits"] == split], f["x"][f["splits"] == split]


def prepare():
    H.mkdir(parents=True, exist_ok=True)
    toy = feature_data("toy")
    db = feature_data("db")
    counts = {}
    for scenario in ("toy", "db"):
        f = toy if scenario == "toy" else db
        counts[scenario] = {s: int(np.sum(f["splits"] == s)) for s in ("train", "val", "test")}
        for split in ("train", "val"):
            ids, y, n, _ = labels(scenario, split)
            assert len(ids) == counts[scenario][split] and y.shape == n.shape == (len(ids), M)

    # DB efficiency subsets: nested, deterministic, outcome-blind source hash.
    split_meta = json.load(open(DB / "db_state_split.json"))
    src = {s["state_id"]: s["source_group"] for s in split_meta["states"] if s["split"] == "train"}
    db_train = list(split_x("db", "train")[0])
    order = sorted(db_train, key=lambda sid: hashlib.sha256(("joint_db_subset_v1|" + src[sid]).encode()).hexdigest())
    subsets = {
        "selection": "Nested SHA256 order of DB TRAIN source_group; outcomes unused",
        "25": order[:16],
        "50": order[:32],
        "100": order,
        "hash_order": order,
    }
    dump("db_train_subsets.json", subsets)

    assets = [
        TOY / "codebook_eta.csv", TOY / "state_features.npz", TOY / "train_mode_counts.csv",
        TOY / "val_mode_counts.csv", TOY / "test_mode_q64.csv", TOY / "selected_model.json",
        Path(json.load(open(TOY / "selected_model.json"))["checkpoint"]),
        DB / "selected_transform.json", DB / "state_features.npz", DB / "train_mode_counts.csv",
        DB / "val_mode_counts.csv", DB / "test_mode_q64.csv", DB / "selected_selector.json",
        Path(json.load(open(DB / "selected_selector.json"))["checkpoint"]),
    ]
    dump("frozen_assets.json", {"files": {str(p): sha(p) for p in assets},
                                "canonical_modes": 12,
                                "transform_frozen": str(DB / "selected_transform.json")})
    audit = {
        "toy": {"dimension": 214, "semantics": "frozen true-t0 214-D h0 state representation",
                "normalization": str(TOY / "normalization.json"), "counts": counts["toy"]},
        "db": {"dimension": 80, "semantics": "flattened observation[4,18] + current raw Flow action[4,2]",
               "normalization": str(DB / "feature_normalization.json"), "counts": counts["db"]},
        "fully_compatible": False,
        "reason": "Dimensions and documented feature semantics differ; direct weight sharing at the input is invalid.",
        "architecture_decision": "Scenario-specific full-input adapters to 64-D; shared 64-D trunk and shared 12-mode head.",
        "features_dropped": 0,
        "test_outcomes_used_for_architecture_or_subsets": False,
    }
    dump("input_schema_audit.json", audit)
    protocol = """# Toy–DB joint selector protocol

- No rollout is generated. Canonical mode IDs, Toy eta, transformed DB eta, and the Toy-to-DB transform are immutable.
- Toy 214-D and DB 80-D inputs have incompatible dimensions and semantics. Each receives a scenario-specific Dense(64)+SiLU adapter. A Dense(64)+SiLU trunk and Dense(12) canonical-mode head are shared.
- Training loss is exactly `0.5 * mean_BCE_Toy + 0.5 * mean_BCE_DB`; each BCE uses the existing binomial success fraction for every state/mode.
- Seeds are 17, 23, 41. Checkpoints are selected on VAL by average scenario NLL, average top-1 empirical Q, worst-scenario NLL, minimum scenario top-1 Q, then lower seed.
- DB TRAIN subsets are nested, outcome-blind SHA256 selections of 16/32/64 states. Toy TRAIN always uses all 128 states.
- TEST outcome files remain unopened until all checkpoint/seed choices are frozen.
"""
    (H / "protocol.md").write_text(protocol)
    dump("working_state.json", {"status": "PREPARED", "completed": ["input_audit", "asset_freeze", "db_subset_freeze"],
                                "next_action": "train joint 25/50/100 and DB-only 25/50"})
    dump("evidence_index.json", {"toy": str(TOY), "db": str(DB), "input_audit": "input_schema_audit.json",
                                 "subsets": "db_train_subsets.json", "frozen_assets": "frozen_assets.json"})
    write_csv("experiment_ledger.csv", [
        {"stage": "freeze_inputs", "status": "COMPLETE", "rollouts": 0, "decision": "Toy214 and DB80 require adapters"},
        {"stage": "freeze_db_subsets", "status": "COMPLETE", "rollouts": 0, "decision": "nested 16/32/64 DB TRAIN states"},
    ])
    print(json.dumps(audit, indent=2))


def val_score(mt, md, seed):
    return (0.5 * (mt["nll"] + md["nll"]),
            -0.5 * (mt["top1_empirical_Q"] + md["top1_empirical_Q"]),
            max(mt["nll"], md["nll"]),
            -min(mt["top1_empirical_Q"], md["top1_empirical_Q"]), seed)


def train_joint(frac, seed):
    tx_ids, tx = split_x("toy", "train")
    tv_ids, tv = split_x("toy", "val")
    dx_ids, dx_all = split_x("db", "train")
    dv_ids, dv = split_x("db", "val")
    _, ty, _, _ = labels("toy", "train")
    _, tvy, _, _ = labels("toy", "val")
    _, dy_all, _, _ = labels("db", "train")
    _, dvy, _, _ = labels("db", "val")
    subset = set(json.load(open(H / "db_train_subsets.json"))[str(frac)])
    keep = np.array([sid in subset for sid in dx_ids])
    dx, dy = dx_all[keep], dy_all[keep]
    assert len(dx) == {25: 16, 50: 32, 100: 64}[frac]

    model = JointModel()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 214)), jnp.zeros((1, 80)))
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    ost = opt.init(params)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(p, o, xt, yt, xd, yd):
        def loss(pp):
            lt, ld, _, _ = model.apply(pp, xt, xd)
            a = jnp.mean(optax.sigmoid_binary_cross_entropy(lt, yt))
            b = jnp.mean(optax.sigmoid_binary_cross_entropy(ld, yd))
            return 0.5 * a + 0.5 * b
        val, grad = jax.value_and_grad(loss)(p)
        updates, o = opt.update(grad, o, p)
        return optax.apply_updates(p, updates), o, val

    best = None
    best_score = None
    best_epoch = -1
    wait = 0
    hist = []
    start = time.time()
    for epoch in range(2400):
        pt = rng.permutation(len(tx))
        pd = rng.permutation(len(dx))
        steps = max(math.ceil(len(tx) / 32), math.ceil(len(dx) / 32))
        for j in range(steps):
            it = pt[(j * 32) % len(pt): min((j * 32) % len(pt) + 32, len(pt))]
            if len(it) < 16:
                it = pt[:32]
            jd = (j * 32) % len(pd)
            idb = pd[jd: min(jd + 32, len(pd))]
            if len(idb) < min(16, len(pd)):
                idb = pd[: min(32, len(pd))]
            params, ost, _ = step(params, ost, jnp.asarray(tx[it]), jnp.asarray(ty[it]),
                                    jnp.asarray(dx[idb]), jnp.asarray(dy[idb]))
        lt, ld, _, _ = model.apply(params, jnp.asarray(tv), jnp.asarray(dv))
        mt, md = metrics(np.asarray(lt), tvy), metrics(np.asarray(ld), dvy)
        score = val_score(mt, md, seed)
        hist.append({"epoch": epoch, "toy_val_nll": mt["nll"], "db_val_nll": md["nll"],
                     "mean_val_nll": score[0], "toy_val_top1_Q": mt["top1_empirical_Q"],
                     "db_val_top1_Q": md["top1_empirical_Q"]})
        if best_score is None or score[:-1] < best_score[:-1]:
            improved_primary = best_score is None or score[0] < best_score[0] - 1e-7
            best_score = score
            best = jax.tree_util.tree_map(np.asarray, params)
            best_epoch = epoch
            wait = 0 if improved_primary else wait + 1
        else:
            wait += 1
        if wait >= 180:
            break
    lt, ld, _, _ = model.apply(best, jnp.asarray(tv), jnp.asarray(dv))
    mt, md = metrics(np.asarray(lt), tvy), metrics(np.asarray(ld), dvy)
    name = f"joint_db{frac}_seed{seed}"
    out = H / name
    out.mkdir(exist_ok=True)
    ck = out / "checkpoint.msgpack"
    ck.write_bytes(serialization.to_bytes(best))
    write_csv(f"{name}/training_history.csv", hist)
    summary = {"model": name, "kind": "joint_adapters_shared_trunk_head", "db_fraction": frac,
               "db_train_states": len(dx), "toy_train_states": len(tx), "seed": seed,
               "best_epoch": best_epoch, "epochs": len(hist), "training_seconds": time.time() - start,
               "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
               "val": {"toy": mt, "db": md, "mean_nll": 0.5 * (mt["nll"] + md["nll"]),
                       "mean_top1_Q": 0.5 * (mt["top1_empirical_Q"] + md["top1_empirical_Q"]),
                       "worst_nll": max(mt["nll"], md["nll"]),
                       "worst_top1_Q": min(mt["top1_empirical_Q"], md["top1_empirical_Q"])}}
    dump(f"{name}/summary.json", summary)
    return summary


def train_dbonly(frac, seed):
    ids, xall = split_x("db", "train")
    _, xv = split_x("db", "val")
    _, yall, _, _ = labels("db", "train")
    _, yv, _, _ = labels("db", "val")
    subset = set(json.load(open(H / "db_train_subsets.json"))[str(frac)])
    keep = np.array([sid in subset for sid in ids])
    x, y = xall[keep], yall[keep]
    model = DBOnlyModel()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 80)))
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(1e-3, weight_decay=1e-4))
    ost = opt.init(params)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(p, o, xb, yb):
        def loss(pp):
            return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp, xb), yb))
        val, grad = jax.value_and_grad(loss)(p)
        updates, o = opt.update(grad, o, p)
        return optax.apply_updates(p, updates), o, val

    best = None
    best_tuple = None
    be = -1
    wait = 0
    hist = []
    start = time.time()
    for epoch in range(2400):
        order = rng.permutation(len(x))
        for j in range(0, len(order), 16):
            ii = order[j:j + 16]
            params, ost, _ = step(params, ost, jnp.asarray(x[ii]), jnp.asarray(y[ii]))
        mv = metrics(np.asarray(model.apply(params, jnp.asarray(xv))), yv)
        score = (mv["nll"], -mv["top1_empirical_Q"], mv["brier"])
        hist.append({"epoch": epoch, "val_nll": mv["nll"], "val_brier": mv["brier"],
                     "val_top1_Q": mv["top1_empirical_Q"]})
        if best_tuple is None or score < best_tuple:
            improved = best_tuple is None or score[0] < best_tuple[0] - 1e-7
            best_tuple = score
            best = jax.tree_util.tree_map(np.asarray, params)
            be = epoch
            wait = 0 if improved else wait + 1
        else:
            wait += 1
        if wait >= 180:
            break
    mv = metrics(np.asarray(model.apply(best, jnp.asarray(xv))), yv)
    name = f"dbonly_db{frac}_seed{seed}"
    out = H / name
    out.mkdir(exist_ok=True)
    ck = out / "checkpoint.msgpack"
    ck.write_bytes(serialization.to_bytes(best))
    write_csv(f"{name}/training_history.csv", hist)
    summary = {"model": name, "kind": "db_only", "db_fraction": frac, "db_train_states": len(x),
               "seed": seed, "best_epoch": be, "epochs": len(hist), "training_seconds": time.time() - start,
               "checkpoint": str(ck), "checkpoint_sha256": sha(ck), "val": mv}
    dump(f"{name}/summary.json", summary)
    return summary


def train_all():
    summaries = []
    for frac in FRACTIONS:
        for seed in SEEDS:
            summaries.append(train_joint(frac, seed))
    for frac in (25, 50):
        for seed in SEEDS:
            summaries.append(train_dbonly(frac, seed))
    selected = {}
    for frac in FRACTIONS:
        q = [x for x in summaries if x["kind"].startswith("joint") and x["db_fraction"] == frac]
        selected[f"joint_{frac}"] = min(q, key=lambda x: (
            x["val"]["mean_nll"], -x["val"]["mean_top1_Q"], x["val"]["worst_nll"],
            -x["val"]["worst_top1_Q"], x["seed"]))
    for frac in (25, 50):
        q = [x for x in summaries if x["kind"] == "db_only" and x["db_fraction"] == frac]
        selected[f"dbonly_{frac}"] = min(q, key=lambda x: (
            x["val"]["nll"], -x["val"]["top1_empirical_Q"], x["val"]["brier"], x["seed"]))
    dump("selected_models.json", selected)
    rows = []
    for x in summaries:
        if x["kind"].startswith("joint"):
            rows.append({"model": x["model"], "kind": x["kind"], "db_fraction": x["db_fraction"],
                         "seed": x["seed"], "best_epoch": x["best_epoch"],
                         "toy_val_nll": x["val"]["toy"]["nll"], "db_val_nll": x["val"]["db"]["nll"],
                         "mean_val_nll": x["val"]["mean_nll"],
                         "toy_val_top1_Q": x["val"]["toy"]["top1_empirical_Q"],
                         "db_val_top1_Q": x["val"]["db"]["top1_empirical_Q"],
                         "training_seconds": x["training_seconds"]})
        else:
            rows.append({"model": x["model"], "kind": x["kind"], "db_fraction": x["db_fraction"],
                         "seed": x["seed"], "best_epoch": x["best_epoch"],
                         "toy_val_nll": "", "db_val_nll": x["val"]["nll"],
                         "mean_val_nll": x["val"]["nll"], "toy_val_top1_Q": "",
                         "db_val_top1_Q": x["val"]["top1_empirical_Q"],
                         "training_seconds": x["training_seconds"]})
    write_csv("training_summary.csv", rows)
    w = json.load(open(H / "working_state.json"))
    w.update(status="TRAINED_AND_FROZEN", completed=w["completed"] + ["joint_training", "db_only_efficiency_baselines", "val_model_selection"],
             next_action="open quarantined TEST matrices and evaluate frozen checkpoints")
    dump("working_state.json", w)
    print(json.dumps({k: {"model": v["model"], "seed": v["seed"]} for k, v in selected.items()}, indent=2))


def apply_joint(summary, toy_x, db_x):
    model = JointModel()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 214)), jnp.zeros((1, 80)))
    params = serialization.from_bytes(template, Path(summary["checkpoint"]).read_bytes())
    lt, ld, zt, zd = model.apply(params, jnp.asarray(toy_x), jnp.asarray(db_x))
    return np.asarray(lt), np.asarray(ld), np.asarray(zt), np.asarray(zd)


def apply_dbonly(summary, x):
    model = DBOnlyModel()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 80)))
    params = serialization.from_bytes(template, Path(summary["checkpoint"]).read_bytes())
    return np.asarray(model.apply(params, jnp.asarray(x)))


def apply_toy_existing(x):
    summary = json.load(open(TOY / "selected_model.json"))
    model = ToyOnlyModel()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 214)))
    params = serialization.from_bytes(template, Path(summary["checkpoint"]).read_bytes())
    return np.asarray(model.apply(params, jnp.asarray(x)))


def apply_db_existing(x):
    summary = json.load(open(DB / "selected_selector.json"))
    model = DBOnlyModel()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 80)))
    params = serialization.from_bytes(template, Path(summary["checkpoint"]).read_bytes())
    return np.asarray(model.apply(params, jnp.asarray(x)))


def evaluate():
    selected = json.load(open(H / "selected_models.json"))
    # TEST outcome matrices are opened for the first time here.
    _, toy_x = split_x("toy", "test")
    _, db_x = split_x("db", "test")
    toy_ids, toy_y, _, toy_b = labels("toy", "test", allow_test=True)
    db_ids, db_y, _, db_b = labels("db", "test", allow_test=True)

    rows = []
    baseline_toy = selection_metrics(apply_toy_existing(toy_x), toy_y, toy_b)
    baseline_db = selection_metrics(apply_db_existing(db_x), db_y, db_b)
    rows.append({"model": "frozen_Toy_only", "scenario": "Toy", "db_fraction": 0, **baseline_toy})
    rows.append({"model": "frozen_DB_only_100", "scenario": "DB", "db_fraction": 100, **baseline_db})
    joint_outputs = {}
    for frac in FRACTIONS:
        lt, ld, _, _ = apply_joint(selected[f"joint_{frac}"], toy_x, db_x)
        joint_outputs[frac] = (lt, ld)
        rows.append({"model": f"joint_DB{frac}", "scenario": "Toy", "db_fraction": frac,
                     **selection_metrics(lt, toy_y, toy_b)})
        rows.append({"model": f"joint_DB{frac}", "scenario": "DB", "db_fraction": frac,
                     **selection_metrics(ld, db_y, db_b)})
    for frac in (25, 50):
        ld = apply_dbonly(selected[f"dbonly_{frac}"], db_x)
        rows.append({"model": f"DB_only_{frac}", "scenario": "DB", "db_fraction": frac,
                     **selection_metrics(ld, db_y, db_b)})
    fields = ["model", "scenario", "db_fraction", "states", "nll", "brier", "top1_empirical_Q",
              "B63_states", "B63_fraction", "mean_regret", "max_regret", "selection_entropy",
              "within_state_rank_correlation", "mode_counts"]
    write_csv("test_metrics.csv", rows, fields)

    # State-level TEST selections for the selected full-data joint model.
    state_rows = []
    for scenario, ids, y, b, logits in (
        ("Toy", toy_ids, toy_y, toy_b, joint_outputs[100][0]),
        ("DB", db_ids, db_y, db_b, joint_outputs[100][1]),
    ):
        p = sigmoid(logits)
        top = np.argmax(p, axis=1)
        for i, sid in enumerate(ids):
            state_rows.append({"scenario": scenario, "state_id": sid, "selected_mode": int(top[i]),
                               "confidence": float(p[i, top[i]]), "selected_Q64": float(y[i, top[i]]),
                               "selected_B63": bool(b[i, top[i]]), "oracle_Q64": float(y[i].max()),
                               "regret": float(y[i].max() - y[i, top[i]])})
    write_csv("joint_test_selections.csv", state_rows)

    # Data-efficiency comparison on the identical frozen DB TEST matrix.
    def get(model):
        return next(r for r in rows if r["model"] == model and r["scenario"] == "DB")
    eff = []
    for frac in (25, 50, 100):
        j = get(f"joint_DB{frac}")
        d = get(f"DB_only_{frac}") if frac < 100 else get("frozen_DB_only_100")
        eff.append({"db_fraction": frac, "db_train_states": {25: 16, 50: 32, 100: 64}[frac],
                    "joint_nll": j["nll"], "dbonly_nll": d["nll"],
                    "joint_top1_Q": j["top1_empirical_Q"], "dbonly_top1_Q": d["top1_empirical_Q"],
                    "joint_B63_fraction": j["B63_fraction"], "dbonly_B63_fraction": d["B63_fraction"],
                    "joint_regret": j["mean_regret"], "dbonly_regret": d["mean_regret"],
                    "nll_transfer_delta": d["nll"] - j["nll"],
                    "top1_transfer_delta": j["top1_empirical_Q"] - d["top1_empirical_Q"]})
    write_csv("data_efficiency.csv", eff)

    # Simple shared-representation diagnostic on TRAIN only.
    _, toy_tx = split_x("toy", "train")
    _, db_tx = split_x("db", "train")
    _, toy_ty, _, _ = labels("toy", "train")
    _, db_ty, _, _ = labels("db", "train")
    _, _, zt, zd = apply_joint(selected["joint_100"], toy_tx, db_tx)
    proto_t, proto_d = [], []
    for m in range(M):
        wt = np.maximum(toy_ty[:, m], 1e-6); wd = np.maximum(db_ty[:, m], 1e-6)
        proto_t.append(np.average(zt, axis=0, weights=wt))
        proto_d.append(np.average(zd, axis=0, weights=wd))
    pt, pd = np.asarray(proto_t), np.asarray(proto_d)
    pt /= np.maximum(np.linalg.norm(pt, axis=1, keepdims=True), 1e-12)
    pd /= np.maximum(np.linalg.norm(pd, axis=1, keepdims=True), 1e-12)
    sim = pt @ pd.T
    diag = np.diag(sim)
    off = sim[~np.eye(M, dtype=bool)]
    prevalence_corr = float(np.corrcoef(toy_ty.mean(0), db_ty.mean(0))[0, 1])
    latent = {"method": "TRAIN-only feasibility-weighted shared-latent prototypes",
              "same_mode_cosine_mean": float(diag.mean()), "same_mode_cosine_median": float(np.median(diag)),
              "cross_mode_cosine_mean": float(off.mean()),
              "same_mode_top1_retrieval_fraction": float(np.mean(np.argmax(sim, axis=1) == np.arange(M))),
              "train_mode_prevalence_correlation": prevalence_corr,
              "interpretation_limit": "Diagnostic only; high mode prevalence and shared-head training can inflate similarity."}
    latent["mode_specific_alignment_supported"] = bool(
        latent["same_mode_cosine_mean"] >= latent["cross_mode_cosine_mean"] + 0.02
        and latent["same_mode_top1_retrieval_fraction"] >= 0.5)
    dump("latent_alignment.json", latent)
    write_csv("latent_mode_similarity.csv",
              [{"toy_mode": i, "db_mode": j, "cosine": float(sim[i, j]), "same_mode": i == j}
               for i in range(M) for j in range(M)])

    full_t = next(r for r in rows if r["model"] == "joint_DB100" and r["scenario"] == "Toy")
    full_d = next(r for r in rows if r["model"] == "joint_DB100" and r["scenario"] == "DB")
    retain_t = min(full_t["top1_empirical_Q"] / max(baseline_toy["top1_empirical_Q"], 1e-12),
                   full_t["B63_fraction"] / max(baseline_toy["B63_fraction"], 1e-12))
    retain_d = min(full_d["top1_empirical_Q"] / max(baseline_db["top1_empirical_Q"], 1e-12),
                   full_d["B63_fraction"] / max(baseline_db["B63_fraction"], 1e-12))
    retention = retain_t >= 0.95 and retain_d >= 0.95
    e25 = eff[0]
    near_full = (e25["joint_top1_Q"] >= 0.95 * baseline_db["top1_empirical_Q"] and
                 e25["joint_B63_fraction"] >= 0.95 * baseline_db["B63_fraction"])
    material_eff = (e25["joint_nll"] <= 0.95 * e25["dbonly_nll"] or
                    e25["joint_top1_Q"] >= e25["dbonly_top1_Q"] + 0.02 or
                    e25["joint_B63_fraction"] >= e25["dbonly_B63_fraction"] + 0.10)
    if retention and near_full and material_eff:
        classification = "JOINT_SELECTOR_STRONGLY_SUPPORTED"
    elif retention:
        classification = "JOINT_SELECTOR_SUPPORTED"
    else:
        classification = "GEOMETRY_SHARED_BUT_SELECTOR_NOT_SHARED"
    positive = {"Toy": full_t["nll"] < baseline_toy["nll"], "DB": full_d["nll"] < baseline_db["nll"]}
    negative = {"Toy": retain_t < 0.95, "DB": retain_d < 0.95}
    decision = {
        "classification": classification,
        "input_compatibility": "INCOMPATIBLE; SCENARIO_SPECIFIC_ADAPTERS_REQUIRED",
        "architecture": "Toy214->64 adapter | DB80->64 adapter; shared 64->64 SiLU ->12 head",
        "selected_joint_model": selected["joint_100"],
        "toy_test": {"joint": full_t, "single_scenario": baseline_toy, "retention_ratio": retain_t},
        "db_test": {"joint": full_d, "single_scenario": baseline_db, "retention_ratio": retain_d},
        "data_efficiency": eff,
        "positive_transfer_by_test_NLL": positive,
        "negative_transfer_below_95pct": negative,
        "joint25_near_dbonly100": near_full,
        "material_25pct_efficiency_gain": material_eff,
        "latent_alignment": latent,
        "answer": "State-to-mode feasibility can be shared through a common trunk/head" if retention else
                  "Only eta-mode geometry is shared under current evidence; feasibility-law sharing failed",
        "new_rollouts": 0,
    }
    dump("final_decision.json", decision)

    report = f"""# Toy–Double-Bottleneck joint selector

Classification: **{classification}**

Toy and DB inputs are not directly compatible: Toy uses 214-D h0, while DB uses 80-D observation+Flow features. The frozen joint model therefore uses scenario-specific 64-D adapters and a shared 64-D trunk plus shared 12-mode head. No feature was dropped and no rollout was generated.

## TEST comparison

| Scenario / model | NLL | Brier | top-1 Q64 | B63 | regret | selection entropy |
|---|---:|---:|---:|---:|---:|---:|
| Toy-only | {baseline_toy['nll']:.4f} | {baseline_toy['brier']:.4f} | {baseline_toy['top1_empirical_Q']:.4f} | {baseline_toy['B63_states']}/32 | {baseline_toy['mean_regret']:.4f} | {baseline_toy['selection_entropy']:.3f} |
| Joint (Toy) | {full_t['nll']:.4f} | {full_t['brier']:.4f} | {full_t['top1_empirical_Q']:.4f} | {full_t['B63_states']}/32 | {full_t['mean_regret']:.4f} | {full_t['selection_entropy']:.3f} |
| DB-only | {baseline_db['nll']:.4f} | {baseline_db['brier']:.4f} | {baseline_db['top1_empirical_Q']:.4f} | {baseline_db['B63_states']}/16 | {baseline_db['mean_regret']:.4f} | {baseline_db['selection_entropy']:.3f} |
| Joint (DB) | {full_d['nll']:.4f} | {full_d['brier']:.4f} | {full_d['top1_empirical_Q']:.4f} | {full_d['B63_states']}/16 | {full_d['mean_regret']:.4f} | {full_d['selection_entropy']:.3f} |

## DB label efficiency

| DB TRAIN fraction | Joint NLL | DB-only NLL | Joint top-1 Q | DB-only top-1 Q | Joint B63 | DB-only B63 |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for e in eff:
        report += (f"| {e['db_fraction']}% ({e['db_train_states']}) | {e['joint_nll']:.4f} | {e['dbonly_nll']:.4f} | "
                   f"{e['joint_top1_Q']:.4f} | {e['dbonly_top1_Q']:.4f} | {e['joint_B63_fraction']:.3f} | {e['dbonly_B63_fraction']:.3f} |\n")
    report += f"""

Toy performance retention: {retain_t:.3f}; DB retention: {retain_d:.3f}. Joint-25% DB is {'near' if near_full else 'not near'} DB-only-100% on the preregistered top-1/B63 criterion. Material 25%-DB efficiency gain over DB-only-25%: {material_eff}.

Shared-latent same-mode prototype cosine is {latent['same_mode_cosine_mean']:.3f} versus cross-mode {latent['cross_mode_cosine_mean']:.3f}; this is diagnostic only.

The prototype diagnostic does **not** independently support mode-specific latent alignment (same-mode top-1 retrieval {latent['same_mode_top1_retrieval_fraction']:.3f}). Predictive/data-efficiency transfer is therefore stronger evidence than latent clustering, and should not be overinterpreted as a scenario-invariant state geometry.

**Answer:** {decision['answer']}. Scenario-specific input adapters remain necessary because the h schemas are different.
"""
    (H / "final_report.md").write_text(report)
    w = json.load(open(H / "working_state.json"))
    final_stages = ["test_evaluation", "data_efficiency", "latent_diagnostic", "final_decision"]
    w.update(status="COMPLETE", completed=list(dict.fromkeys(w["completed"] + final_stages)), next_action="none")
    dump("working_state.json", w)

    # Append ledger without using any outcome to alter prior design choices.
    final_ledger_stages = {"joint_training", "db_only_efficiency", "heldout_test"}
    ledger = [row for row in read_csv(H / "experiment_ledger.csv")
              if row["stage"] not in final_ledger_stages]
    ledger += [
        {"stage": "joint_training", "status": "COMPLETE", "rollouts": 0, "decision": "three seeds at DB 25/50/100"},
        {"stage": "db_only_efficiency", "status": "COMPLETE", "rollouts": 0, "decision": "three seeds at DB 25/50"},
        {"stage": "heldout_test", "status": "COMPLETE", "rollouts": 0, "decision": classification},
    ]
    write_csv("experiment_ledger.csv", ledger, ["stage", "status", "rollouts", "decision"])

    artifact_names = ["protocol.md", "input_schema_audit.json", "frozen_assets.json", "db_train_subsets.json",
                      "training_summary.csv", "selected_models.json", "test_metrics.csv", "joint_test_selections.csv",
                      "data_efficiency.csv", "latent_alignment.json", "latent_mode_similarity.csv",
                      "final_decision.json", "final_report.md", "working_state.json", "experiment_ledger.csv"]
    dump("manifest.json", {"artifacts": {x: sha(H / x) for x in artifact_names}, "new_rollouts": 0,
                           "classification": classification})
    print(json.dumps(decision, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "train", "evaluate"))
    args = ap.parse_args()
    {"prepare": prepare, "train": train_all, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
