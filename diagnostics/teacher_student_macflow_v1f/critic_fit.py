"""Round16 source-only critic fitting and outcome-blind external prediction."""
import os
# Environment configured by round30f core.
# Environment configured by round30f core.
# Environment configured by round30f core.
import hashlib
import json
import sys
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", False)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23

ARMS = ["F10", "F13", "F2", "S200", "S300", "S400", "S500", "S600"]
SEEDS = [17, 23, 41]
STEPS = list(range(500, 12001, 500))
EXT = {
    "T1": (ROOT / "diagnostics/ring_independent_flow_confirmation_v2", "R5_FINAL_FEATURES.npz", "features"),
    "T2": (ROOT / "diagnostics/controller_universal_eta_schedule_v1", None, "R5"),
    "T3": (ROOT / "diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8", "R5_FINAL_FEATURES.npz", "features"),
}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Net(nn.Module):
    dims: tuple
    residual: bool = False

    @nn.compact
    def __call__(self, x):
        for width in self.dims:
            x = nn.silu(nn.Dense(width)(x))
        init = nn.initializers.zeros if self.residual else nn.initializers.lecun_normal()
        x = nn.Dense(1, kernel_init=init, bias_init=nn.initializers.zeros)(x)[..., 0]
        return jnp.tanh(x) if self.residual else x


def net_for(model):
    return Net((16, 16, 16), True) if model in ("A1", "A2") else Net((64, 64, 16))


def source_data():
    folder = ROOT / "diagnostics/ring_controller_population_v1"
    rows = [json.loads(x) for x in (folder / "dataset.jsonl").read_text().splitlines()]
    states = json.loads((folder / "FRESH_STATES.json").read_text())
    state_names = [s["state_uid"] for s in states]
    with np.load(folder / "R5_FEATURES.npz") as z:
        names = list(z["controller_names"]); r5 = z["features"].astype(np.float32)
    with np.load(folder / "R6_FEATURES.npz") as z:
        assert names == list(z["controller_names"])
        r6 = z["delta"].astype(np.float32)
    labels = np.full((len(names), len(states), len(ARMS), 4), -1, np.int8)
    ci = {n: i for i, n in enumerate(names)}; si = {n: i for i, n in enumerate(state_names)}
    ai = {n: i for i, n in enumerate(ARMS)}
    recipe = {}
    for r in rows:
        c, s, a, f = ci[r["controller"]], si[r["state_uid"]], ai[r["arm"]], r["future_seed"]
        assert labels[c, s, a, f] == -1
        labels[c, s, a, f] = -1 if r["unknown"] else r["success"]
        recipe[r["controller"]] = r["recipe"]
    assert len(rows) == labels.size
    y = (labels == 1).sum(-1).astype(np.float32)
    n = (labels != -1).sum(-1).astype(np.float32)
    return dict(names=names, recipes=np.array([recipe[x] for x in names]), states=states,
                physical=physical23(states).astype(np.float32), r5=r5, r6=r6, y=y, n=n,
                labels=labels)


def transform_fit(data, source_idx):
    raw = data["physical"]
    mean = raw.mean(0); scale = np.where(raw.std(0) > 1e-6, raw.std(0), 1)
    r5 = np.delete(data["r5"], np.s_[105:110], axis=1)
    x = r5[source_idx]
    keep = np.isfinite(x).any(0)
    x = x[:, keep]
    m5 = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    clean5 = np.where(np.isfinite(x), x, m5)
    d5 = np.where(clean5.std(0) > 1e-6, clean5.std(0), 1)
    _, _, vt = np.linalg.svd((clean5 - m5) / d5, full_matrices=False)
    components = vt[:2].copy()
    for v in components:
        if v[np.argmax(np.abs(v))] < 0: v *= -1
    r6 = data["r6"][source_idx]
    med = np.nanmedian(np.where(np.isfinite(r6), r6, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0)
    fixed = np.where(np.isfinite(r6), r6, med)
    m6 = fixed.mean(0); d6 = np.where(fixed.std(0) > 1e-6, fixed.std(0), 1)
    return dict(state_mean=mean, state_scale=scale, r5_keep=keep, r5_mean=m5, r5_scale=d5,
                pca=components, r6_median=med, r6_mean=m6, r6_scale=d6)


def features(physical, r5, r6, transform, model):
    s = ((physical - transform["state_mean"]) / transform["state_scale"]).astype(np.float32)
    x5 = np.delete(r5, np.s_[105:110], axis=1)[:, transform["r5_keep"]]
    x5 = np.where(np.isfinite(x5), x5, transform["r5_mean"])
    z5 = ((x5 - transform["r5_mean"]) / transform["r5_scale"]) @ transform["pca"].T
    if r6 is None:
        z6 = None
    else:
        clean = np.where(np.isfinite(r6), r6, transform["r6_median"])
        z6 = ((clean - transform["r6_mean"]) / transform["r6_scale"]).astype(np.float32)
    nc, ns, na = len(r5), len(s), len(ARMS)
    s = np.broadcast_to(s[None, :, None, :], (nc, ns, na, 23))
    onehot = np.broadcast_to(np.eye(na, dtype=np.float32)[None, None, :, :], (nc, ns, na, na))
    pieces = [s]
    if model != "A0":
        pieces.append(np.broadcast_to(z5[:, None, None, :], (nc, ns, na, 2)))
    if model == "A2":
        assert z6 is not None
        pieces.append(np.broadcast_to(z6[:, None, None, :], (nc, ns, na, 7)))
    pieces.append(onehot)
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def fit_one(model, seed, data, source_idx, held_idx, transform, base_path, folder, selected_step=None):
    folder.mkdir(parents=True, exist_ok=True)
    net = net_for(model)
    X = features(data["physical"], data["r5"], data["r6"], transform, model)
    xs = X[source_idx].reshape(-1, X.shape[-1]); xv = X[held_idx].reshape(-1, X.shape[-1])
    y = data["y"][source_idx].reshape(-1); n = data["n"][source_idx].reshape(-1)
    yv = data["y"][held_idx].reshape(-1); nv = data["n"][held_idx].reshape(-1)
    valid = np.flatnonzero(n > 0)
    assert len(valid) and nv.sum() > 0
    if base_path is None:
        bs = np.zeros(len(xs), np.float32); bv = np.zeros(len(xv), np.float32)
    else:
        bnet = net_for("A0")
        bx = features(data["physical"], data["r5"], data["r6"], transform, "A0")
        bpa = serialization.from_bytes(bnet.init(jax.random.PRNGKey(0), jnp.zeros((1, bx.shape[-1]))), Path(base_path).read_bytes())
        bs = np.asarray(bnet.apply(bpa, jnp.asarray(bx[source_idx].reshape(-1, bx.shape[-1]))))
        bv = np.asarray(bnet.apply(bpa, jnp.asarray(bx[held_idx].reshape(-1, bx.shape[-1]))))
    xs, yy, nn, bs = map(jnp.asarray, (xs, y, n, bs))
    xv, yv, nv, bv = map(jnp.asarray, (xv, yv, nv, bv))
    pa = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, X.shape[-1]), jnp.float32))
    opt = optax.chain(optax.clip_by_global_norm(5), optax.adamw(.0008, weight_decay=.01 if model in ("A1", "A2") else .0001))
    state = opt.init(pa)
    @jax.jit
    def block(pa, state, ids):
        def update(carry, idx):
            p, st = carry
            def loss(pp):
                logits = net.apply(pp, xs[idx]) + bs[idx]
                return jnp.mean((yy[idx] * jax.nn.softplus(-logits) + (nn[idx] - yy[idx]) * jax.nn.softplus(logits)) / 4)
            grad = jax.grad(loss)(p)
            delta, st = opt.update(grad, st, p)
            return (optax.apply_updates(p, delta), st), None
        return jax.lax.scan(update, (pa, state), ids)[0]
    @jax.jit
    def val_nll(pa):
        logit = net.apply(pa, xv) + bv
        return jnp.sum(yv * jax.nn.softplus(-logit) + (nv - yv) * jax.nn.softplus(logit)) / jnp.sum(nv)
    rng = np.random.default_rng(seed)
    targets = STEPS if selected_step is None else [selected_step]
    prev = 0; history = []
    for target in targets:
        count = target - prev
        ids = rng.choice(valid, size=(count, 128), replace=True).astype(np.int32)
        pa, state = block(pa, state, jnp.asarray(ids))
        path = folder / f"step{target}.msgpack"
        path.write_bytes(serialization.to_bytes(pa))
        history.append(float(val_nll(pa)))
        prev = target
    write(folder / "fit.json", dict(model=model, seed=seed, source_controllers=[data["names"][i] for i in source_idx],
                                    held_controllers=[data["names"][i] for i in held_idx],
                                    steps=targets, validation_nll=history, checkpoint_sha256=[sha(folder / f"step{s}.msgpack") for s in targets],
                                    base_checkpoint=str(base_path) if base_path else None))
    print(json.dumps(dict(model=model, seed=seed, folder=str(folder.relative_to(ROOT)), final_nll=history[-1])), flush=True)
    return history


def path_for(part, model, seed, step):
    return HERE / "fits" / part / model / f"seed{seed}" / f"step{step}.msgpack"


def cv():
    data = source_data(); recipes = sorted(set(data["recipes"])); assert len(recipes) == 14
    (HERE / "fits").mkdir(exist_ok=True)
    folds = []
    for k in range(7):
        held = [recipes[k], recipes[k + 7]]
        hi = np.flatnonzero(np.isin(data["recipes"], held)); si = np.flatnonzero(~np.isin(data["recipes"], held))
        assert len(hi) == 8 and len(si) == 48
        folds.append((si, hi))
    selected = {}
    for model in ("A0", "A1", "A2", "A3"):
        histories = []
        for k, (si, hi) in enumerate(folds):
            transform = transform_fit(data, si)
            np.savez_compressed(HERE / "fits" / f"fold{k}_transform.npz", **transform)
            for seed in SEEDS:
                base = path_for(f"fold{k}", "A0", seed, selected["A0"]) if model in ("A1", "A2") else None
                histories.append(fit_one(model, seed, data, si, hi, transform, base, HERE / "fits" / f"fold{k}" / model / f"seed{seed}"))
        mean = np.mean(histories, axis=0)
        step = STEPS[int(np.argmin(mean))]
        selected[model] = step
        write(HERE / "fits" / f"{model}_cv_selection.json", dict(model=model, selected_step=step,
              candidate_steps=STEPS, mean_validation_nll=mean.tolist(), fold_seed_nll=histories))
        print("SELECTED", model, step, flush=True)
    write(HERE / "fits" / "selected_steps.json", selected)
    predict_cv()


def predict_cv():
    data = source_data(); recipes = sorted(set(data["recipes"]))
    folds = []
    for k in range(7):
        held = [recipes[k], recipes[k+7]]
        hi = np.flatnonzero(np.isin(data["recipes"], held))
        folds.append(hi)
    selected = json.loads((HERE / "fits" / "selected_steps.json").read_text())
    (HERE / "predictions").mkdir(exist_ok=True)
    for k, hi in enumerate(folds):
        with np.load(HERE / "fits" / f"fold{k}_transform.npz") as z:
            transform = {key: z[key] for key in z.files}
        for model in ("A0", "A1", "A2", "A3"):
            for seed in SEEDS:
                ck = path_for(f"fold{k}", model, seed, selected[model])
                p = predict(model, seed, ck, transform, data["physical"], data["r5"][hi], data["r6"][hi], part=f"fold{k}")
                np.savez_compressed(HERE / "predictions" / f"CV_fold{k}_{model}_seed{seed}.npz",
                                    probability=p, controller_names=[data["names"][i] for i in hi], arms=ARMS)


def load_external(tag):
    folder, ffile, key = EXT[tag]
    if ffile:
        with np.load(folder / ffile, allow_pickle=False) as z:
            names = list(z["controller_names"]); feat = z[key].astype(np.float32)
    else:
        with np.load(ROOT / "diagnostics/controller_behavior_repr_v1/FEATURES.npz", allow_pickle=False) as z:
            names = list(z["controller_names"]); feat = z[key].astype(np.float32)
        with np.load(folder / "LABELS.npz", allow_pickle=False) as z:
            wanted = list(z["controllers"])
        feat = feat[[names.index(n) for n in wanted]]; names = wanted
    states = json.loads((folder / "FRESH_STATES.json").read_text())
    return names, physical23(states).astype(np.float32), feat


def predict(model, seed, checkpoint, transform, physical, r5, r6=None, part="full"):
    x = features(physical, r5, r6, transform, model)
    net = net_for(model)
    pa = serialization.from_bytes(net.init(jax.random.PRNGKey(0), jnp.zeros((1, x.shape[-1]))), checkpoint.read_bytes())
    logits = np.asarray(net.apply(pa, jnp.asarray(x.reshape(-1, x.shape[-1])))).reshape(x.shape[:-1])
    if model in ("A1", "A2"):
        base = path_for(part, "A0", seed, json.loads((HERE / "fits" / "selected_steps.json").read_text())["A0"])
        xb = features(physical, r5, r6, transform, "A0")
        bnet = net_for("A0")
        pb = serialization.from_bytes(bnet.init(jax.random.PRNGKey(0), jnp.zeros((1, xb.shape[-1]))), base.read_bytes())
        logits = logits + np.asarray(bnet.apply(pb, jnp.asarray(xb.reshape(-1, xb.shape[-1])))).reshape(x.shape[:-1])
    return np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))


def full():
    data = source_data(); selected = json.loads((HERE / "fits" / "selected_steps.json").read_text())
    si = np.arange(len(data["names"])); transform = transform_fit(data, si)
    np.savez_compressed(HERE / "fits" / "full_transform.npz", **transform)
    (HERE / "predictions").mkdir(exist_ok=True)
    rates = data["y"].sum(1) / (len(data["states"]) * 4)
    effect = rates[:, ARMS.index("S500")] - rates[:, ARMS.index("F10")]
    x5 = np.delete(data["r5"], np.s_[105:110], axis=1)[:, transform["r5_keep"]]
    x5 = np.where(np.isfinite(x5), x5, transform["r5_mean"])
    x5 = ((x5 - transform["r5_mean"]) / transform["r5_scale"]) @ transform["pca"].T
    xm = x5.mean(0); ym = effect.mean()
    beta = np.linalg.solve((x5-xm).T @ (x5-xm) + np.eye(2), (x5-xm).T @ (effect-ym))
    ge = {}
    for tag in EXT:
        names, _, r5 = load_external(tag)
        xt = np.delete(r5, np.s_[105:110], axis=1)[:, transform["r5_keep"]]
        xt = np.where(np.isfinite(xt), xt, transform["r5_mean"])
        xt = ((xt - transform["r5_mean"]) / transform["r5_scale"]) @ transform["pca"].T
        eh = (xt-xm) @ beta + ym
        ge[tag] = [dict(controller=n, Ehat=float(e), action="F10" if e < .05 else "S500") for n,e in zip(names,eh)]
    write(HERE / "predictions" / "G_E_FROZEN_CHOICES.json", dict(source="round15_only", source_names=data["names"], source_effect=effect.tolist(),
          pca_transform="fits/full_transform.npz", ridge_penalty=1, coefficient=beta.tolist(), xmean=xm.tolist(), ymean=float(ym), external=ge))
    for model in ("A0", "A1", "A2", "A3"):
        for seed in SEEDS:
            base = path_for("full", "A0", seed, selected["A0"]) if model in ("A1", "A2") else None
            fit_one(model, seed, data, si, si, transform, base, HERE / "fits" / "full" / model / f"seed{seed}", selected[model])
            ck = path_for("full", model, seed, selected[model])
            pred = predict(model, seed, ck, transform, data["physical"], data["r5"], data["r6"])
            np.savez_compressed(HERE / "predictions" / f"source_{model}_seed{seed}.npz", probability=pred, controller_names=data["names"], arms=ARMS)
            if model != "A2":
                for tag in EXT:
                    names, physical, r5 = load_external(tag)
                    pred = predict(model, seed, ck, transform, physical, r5)
                    (HERE / "predictions").mkdir(exist_ok=True)
                    np.savez_compressed(HERE / "predictions" / f"{tag}_{model}_seed{seed}.npz", probability=pred, controller_names=names, arms=ARMS)
    write(HERE / "fits" / "full_complete.json", dict(selected_steps=selected, seeds=SEEDS,
          external_labels_read=False, checkpoint_files=[str(x.relative_to(HERE)) for x in (HERE / "fits/full").rglob("*.msgpack")],
          prediction_files=[str(x.relative_to(HERE)) for x in (HERE / "predictions").glob("*.npz")]))


if __name__ == "__main__":
    if sys.argv[1] == "cv": cv()
    elif sys.argv[1] == "predict_cv": predict_cv()
    elif sys.argv[1] == "full": full()
    else: raise SystemExit("usage: fit.py cv|full")
