"""Zero-new-rollout physical-only critic screening on fully cached FF panels.

Training and prediction never open target_truth.json. Evaluation requires all
eight prediction files to exist first. Historical TEST labels remain opened
in previous studies, so this is explicitly exploratory rather than fresh.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.physical_state_critic_formal_v1.train_ring_ff import EtaCritic

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OLD = ROOT / "diagnostics/orthoflow3_loso_root_cause_v1"
VARIANT = os.environ.get("C1_Q_PILOT_VARIANT", "enriched")
assert VARIANT in ("enriched", "strict")
HERE = BASE if VARIANT == "enriched" else BASE / "strict"
SOURCE = OLD / ("joint_enriched" if VARIANT == "enriched" else "joint")
FIRST = ROOT / "diagnostics/orthoflow3_cross_scene_zero_shot_v1"
SCENES = ("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange")
FOLDS = dict(toy="toy_giveway", db="double_bottleneck", four="four_way_intersection", ring="ring_exchange")
PANELS = {**{f"within_{k}": (v,) for k, v in FOLDS.items()},
          **{f"loso_{k}": tuple(s for s in SCENES if s != v) for k, v in FOLDS.items()}}
TARGET = {f"within_{k}": k for k in FOLDS} | {f"loso_{k}": k for k in FOLDS}
SAFETY = dict(db="ctl_fa2fcea368949614555f976c79cadbe634ee2dd2aebc23b01b4d86b1c19cb5bd",
              four="ctl_b7e2fc1c1c575f4e76534621239ee66b68ea321c9accfdf8beb336aebac76318",
              ring="ctl_7f6291a7a2331cd675f227d8c50765cbb1795d3734afcdc09efb7d3b41cfb171")
STEPS = range(200, 2401, 200)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def mask_entities(x):
    x = {k: np.asarray(v).copy() for k, v in x.items()}
    x["agents"][..., 3:7] = 0.  # nominal and bounded Flow
    x["agents"][..., 10:15] = 0.  # policy frame and routing
    x["globals"][...] = 0.  # controller and monitor variables
    assert np.all(x["agents"][..., 3:7] == 0.)
    assert np.all(x["agents"][..., 10:15] == 0.)
    assert np.all(x["globals"] == 0.)
    return x


def source_data(panel):
    sources = PANELS[panel]
    states = read(SOURCE / "source_states.json")
    x = mask_entities(dict(np.load(SOURCE / "source_entities.npz")))
    rows = pq.read_table(SOURCE / "source_pairs.parquet").to_pylist()
    rows = [r for r in rows if r["scenario"] in sources]
    assert all(states[r["state_index"]]["state_uid"] == r["state_uid"] for r in rows)
    train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], np.int32)
    val = np.asarray([i for i, r in enumerate(rows) if r["split"] == "validation"], np.int32)
    assert len(train) and len(val)
    raw_eta = np.asarray([r["eta"] for r in rows], np.float32)
    lo, hi = raw_eta[train].min(0), raw_eta[train].max(0)
    center = (lo + hi) / 2; radius = np.maximum((hi - lo) / 2, 1e-6)
    eta = (raw_eta - center) / radius
    assert np.isfinite(eta).all()
    return rows, x, eta, train, val, center, radius


def validation(logits, rows, val):
    by_scene = defaultdict(lambda: defaultdict(list))
    for logit, i in zip(logits, val):
        r = rows[int(i)]
        by_scene[r["scenario"]][r["state_uid"]].append((float(logit), float(r["q"]), int(r["k"]), int(r["n"])))
    regrets = []; pairwise = []; nll = []
    for scene, states in by_scene.items():
        scene_regret = []; scene_pairwise = []
        for items in states.values():
            z = np.array([r[0] for r in items]); q = np.array([r[1] for r in items])
            scene_regret.append(float(q.max() - q[z.argmax()]))
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    if abs(q[a] - q[b]) >= .25:
                        scene_pairwise.append(float(np.sign(z[a] - z[b]) == np.sign(q[a] - q[b])))
            for a, _, k, n in items:
                nll.append(float(k * np.logaddexp(0, -a) + (n-k) * np.logaddexp(0, a)) / n)
        regrets.append(float(np.mean(scene_regret)))
        if scene_pairwise: pairwise.append(float(np.mean(scene_pairwise)))
    return dict(selection_regret=float(np.mean(regrets)),
                pairwise_accuracy=float(np.mean(pairwise)) if pairwise else None,
                nll=float(np.mean(nll)), validation_states=sum(len(v) for v in by_scene.values()))


def fit(panel, kind):
    assert panel in PANELS and kind in ("state", "eta_only", "eta_only_wide")
    folder = HERE / panel / kind
    if (folder / "fit.json").exists(): return read(folder / "fit.json")
    rows, x_np, eta_np, train, val, center, radius = source_data(panel)
    x = {k: jnp.asarray(v) for k, v in x_np.items()}
    e = jnp.asarray(eta_np)
    k = jnp.asarray(np.asarray([r["k"] for r in rows], np.float32))
    f = jnp.asarray(np.asarray([r["n"]-r["k"] for r in rows], np.float32))
    si = jnp.asarray(np.asarray([r["state_index"] for r in rows], np.int32))
    net = rep.Critic() if kind == "state" else WideEtaOnly() if kind == "eta_only_wide" else EtaCritic()
    def gather(ii): return {a: v[si[ii]] for a, v in x.items()}
    params = (net.init(jax.random.PRNGKey(17), gather(jnp.array([0])), e[:1]) if kind != "eta_only"
              else net.init(jax.random.PRNGKey(17), e[:1]))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
    opt_state = optimizer.init(params)
    @jax.jit
    def block(p, state, draws):
        def step(carry, ii):
            p, state = carry
            def loss(pp):
                z = net.apply(pp, gather(ii), e[ii]) if kind != "eta_only" else net.apply(pp, e[ii])
                return jnp.mean((k[ii] * jax.nn.softplus(-z) + f[ii] * jax.nn.softplus(z)) / (k[ii]+f[ii]))
            _, grad = jax.value_and_grad(loss)(p)
            updates, state = optimizer.update(grad, state, p)
            return (optax.apply_updates(p, updates), state), None
        return jax.lax.scan(step, (p, state), draws)[0]
    @jax.jit
    def predict(p, ii): return net.apply(p, gather(ii), e[ii]) if kind != "eta_only" else net.apply(p, e[ii])
    rng = np.random.default_rng(17)
    pools = {sc: np.array([i for i in train if rows[i]["scenario"] == sc], np.int32) for sc in PANELS[panel]}
    assert all(len(v) for v in pools.values())
    history = []; best = None; best_bytes = None
    for step in STEPS:
        parts = []
        for j, pool in enumerate(pools.values()):
            size = 128 // len(pools) + (j < 128 % len(pools))
            parts.append(pool[rng.integers(len(pool), size=(200, size))])
        draws = np.concatenate(parts, axis=1)
        params, opt_state = block(params, opt_state, jnp.asarray(draws))
        scores = np.concatenate([np.asarray(predict(params, jnp.asarray(val[i:i+256])))
                                 for i in range(0, len(val), 256)])
        metric = validation(scores, rows, val)
        metric["step"] = step; history.append(metric)
        key = (int(np.floor(metric["selection_regret"] / (1/512))),
               -(metric["pairwise_accuracy"] if metric["pairwise_accuracy"] is not None else -.5),
               metric["nll"], step)
        if best is None or key < best[0]: best = (key, metric); best_bytes = serialization.to_bytes(params)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "checkpoint.msgpack").write_bytes(best_bytes)
    write(folder / "history.json", history)
    result = dict(panel=panel, kind=kind, source_scenes=list(PANELS[panel]), source_variant=VARIANT,
                  source_rows=len(rows), train_pairs=len(train), validation_pairs=len(val),
                  train_states=len({rows[i]["state_uid"] for i in train}),
                  validation_states=len({rows[i]["state_uid"] for i in val}),
                  eta_center=center.tolist(), eta_radius=radius.tolist(),
                  selected=best[1], checkpoint_sha256=sha(folder / "checkpoint.msgpack"),
                  source_parquet_sha256=sha(SOURCE / "source_pairs.parquet"),
                  source_entities_sha256=sha(SOURCE / "source_entities.npz"),
                  target_labels_opened=False)
    write(folder / "fit.json", result)
    print(panel, kind, result["selected"], flush=True)
    return result


def target_input(fold):
    path = OLD / "targets" / fold
    return read(path / "manifest.json"), mask_entities(dict(np.load(path / "entities.npz")))


def predict(panel):
    out = HERE / panel / "predictions.npz"
    if out.exists(): return
    fold = TARGET[panel]
    manifest, x_np = target_input(fold)
    assert len(manifest) == {"toy": 48, "db": 24, "four": 24, "ring": 60}[fold]
    ids = [p["state_uid"] for p in manifest]
    source_ids = {s["state_uid"] for s in read(SOURCE / "source_states.json")
                  if s["scenario"] in PANELS[panel]}
    assert not (set(ids) & source_ids), "target state in source TRAIN/VAL"
    x = {k: jnp.asarray(v) for k, v in x_np.items()}
    raw_eta = np.asarray([p["eta"] for p in manifest], np.float32)
    assert raw_eta.shape == (len(ids), 16, 3)
    indices = np.repeat(np.arange(len(ids)), 16)
    scores = {}
    for kind in ("state", "eta_only"):
        info = read(HERE / panel / kind / "fit.json")
        assert info["target_labels_opened"] is False
        ck = HERE / panel / kind / "checkpoint.msgpack"
        assert sha(ck) == info["checkpoint_sha256"]
        e = jnp.asarray((raw_eta - np.asarray(info["eta_center"])) / np.asarray(info["eta_radius"]))
        net = rep.Critic() if kind == "state" else EtaCritic()
        get = lambda ii: {k: v[ii] for k, v in x.items()}
        template = (net.init(jax.random.PRNGKey(17), get(jnp.array([0])), e[:1, :1].reshape(1, 3))
                    if kind == "state" else net.init(jax.random.PRNGKey(17), e[:1, :1].reshape(1, 3)))
        params = serialization.from_bytes(template, ck.read_bytes())
        flat_e = e.reshape(-1, 3)
        values = []
        for a in range(0, len(indices), 256):
            ii = jnp.asarray(indices[a:a+256]); ee = flat_e[a:a+256]
            values.append(np.asarray(net.apply(params, get(ii), ee) if kind == "state"
                                     else net.apply(params, ee)))
        scores[kind] = np.concatenate(values).reshape(len(ids), 16)
        assert np.isfinite(scores[kind]).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state_logits=scores["state"], eta_only_logits=scores["eta_only"],
                        state_choice=np.argmax(scores["state"], axis=1),
                        eta_only_choice=np.argmax(scores["eta_only"], axis=1),
                        state_uids=np.asarray(ids), eta=raw_eta)
    write(HERE / panel / "prediction_audit.json",
          dict(panel=panel, target_labels_opened=False, predictions_sha256=sha(out),
               source_checkpoint_hashes={k: read(HERE/panel/k/"fit.json")["checkpoint_sha256"]
                                        for k in ("state", "eta_only")}))
    print(panel, "predicted", len(ids), flush=True)


def bootstrap(diff):
    a = np.asarray(diff, float); n = len(a); rng = np.random.default_rng(20261008)
    reps = a[rng.integers(n, size=(10000, n))].mean(1)
    return dict(n=n, mean=float(a.mean()), ci95=np.quantile(reps, [.025, .975]).tolist(),
                wins=int((a>0).sum()), losses=int((a<0).sum()), ties=int((a==0).sum()))


def capacity(panel):
    """Eta-only head with the same 32→128→64 head width as the state critic."""
    assert all((HERE/p/"predictions.npz").exists() for p in PANELS)
    fit_info = read(HERE/panel/"eta_only_wide"/"fit.json")
    ck = HERE/panel/"eta_only_wide"/"checkpoint.msgpack"
    assert sha(ck)==fit_info["checkpoint_sha256"]
    fold=TARGET[panel]
    manifest,x_np=target_input(fold)
    ids=[s["state_uid"] for s in manifest]
    old=np.load(HERE/panel/"predictions.npz")
    assert ids==list(old["state_uids"])
    x={k:jnp.asarray(v) for k,v in x_np.items()}
    get=lambda ii:{k:v[ii] for k,v in x.items()}
    raw=np.asarray([s["eta"] for s in manifest],np.float32)
    e=jnp.asarray((raw-np.asarray(fit_info["eta_center"])) / np.asarray(fit_info["eta_radius"]))
    net=WideEtaOnly()
    template=net.init(jax.random.PRNGKey(17),get(jnp.array([0])),e[:1,:1].reshape(1,3))
    params=serialization.from_bytes(template,ck.read_bytes())
    flat=e.reshape(-1,3);state_indices=np.repeat(np.arange(len(ids)),16)
    values=[]
    for start in range(0,len(flat),256):
        ii=jnp.asarray(state_indices[start:start+256])
        values.append(np.asarray(net.apply(params,get(ii),flat[start:start+256])))
    scores=np.concatenate(values).reshape(len(ids),16)
    assert np.isfinite(scores).all()
    choice=scores.argmax(1)
    np.savez_compressed(HERE/panel/"eta_only_wide_predictions.npz",logits=scores,
                        choice=choice,state_uids=np.asarray(ids),eta=raw)
    truth_path=(FIRST if fold=="ring" else OLD/fold)/"target_truth.json"
    truth=read(truth_path)
    assert ids==[s["state_uid"] for s in truth]
    robust=np.asarray([[v is True for v in s["robust"]] for s in truth],bool)
    lower=np.asarray([s["lower"] for s in truth],float)
    ix=np.arange(len(ids));state_choice=old["state_choice"];old_eta=old["eta_only_choice"]
    result=dict(panel=panel,source_variant=VARIANT,test_states=len(ids),
                wide_eta_architecture="eta_encoder32→shared128→shared64; same downstream widths as state critic",
                wide_eta_checkpoint_sha256=sha(ck),
                wide_eta_predictions_sha256=sha(HERE/panel/"eta_only_wide_predictions.npz"),
                B15=dict(Q_state=int(robust[ix,state_choice].sum()),
                         Q_eta_wide=int(robust[ix,choice].sum()),Q_eta_64=int(robust[ix,old_eta].sum())),
                meanQ=dict(Q_state=float(lower[ix,state_choice].mean()),
                           Q_eta_wide=float(lower[ix,choice].mean()),
                           Q_eta_64=float(lower[ix,old_eta].mean())),
                paired_state_minus_wide_B15=bootstrap(robust[ix,state_choice].astype(float)-robust[ix,choice].astype(float)),
                paired_state_minus_wide_meanQ=bootstrap(lower[ix,state_choice]-lower[ix,choice]),
                wide_choice_counts=np.bincount(choice,minlength=16).tolist())
    write(HERE/panel/"capacity_control.json",result)
    print(panel,result["B15"],result["paired_state_minus_wide_B15"],flush=True)


def safety_truth(fold, ids):
    if fold == "toy": return None  # historical second controller is MAC_ONLY, not safety
    db = sqlite3.connect(f"file:{ROOT/'shared_rollout_db/rollout.sqlite'}?mode=ro", uri=True)
    q = ','.join('?' * len(ids)); data = defaultdict(dict)
    for state, seed, success, collision, numerical in db.execute(
        f"SELECT state_uid,seed_key,success,collision,numerical_failure FROM rollout "
        f"WHERE controller_uid=? AND state_uid IN ({q}) AND conflict_quarantined=0",
        [SAFETY[fold], *ids],
    ):
        j = json.loads(seed)["future_index"]
        assert j not in data[state]
        data[state][j] = (success, collision, numerical)
    db.close()
    assert all(set(data[s]) == set(range(16)) for s in ids), "incomplete safety Q16"
    return [dict(success=sum(data[s][j][0] and not data[s][j][2] for j in range(16)),
                 collision=sum(data[s][j][1] for j in range(16)),
                 numerical=sum(data[s][j][2] for j in range(16))) for s in ids]


def evaluate(panel):
    assert all((HERE/p/"predictions.npz").exists() for p in PANELS), "freeze all predictions first"
    fold = TARGET[panel]
    truth_path = (FIRST if fold == "ring" else OLD/fold) / "target_truth.json"
    truth = read(truth_path)
    pred = np.load(HERE/panel/"predictions.npz")
    ids = list(pred["state_uids"])
    assert ids == [s["state_uid"] for s in truth]
    n = len(ids); ii = np.arange(n)
    lower = np.asarray([s["lower"] for s in truth], float)
    upper = np.asarray([s["upper"] for s in truth], float)
    robust = np.asarray([[r is True for r in s["robust"]] for s in truth], bool)
    unknown = np.asarray([[r is None for r in s["robust"]] for s in truth], bool)
    collision = np.asarray([s["collisions"] for s in truth], int)
    choices = {"Q_state": pred["state_choice"], "Q_eta": pred["eta_only_choice"],
               "Oracle": lower.argmax(axis=1)}
    safety = safety_truth(fold, ids)
    methods = {}; selected = {}
    for name, c in choices.items():
        c = np.asarray(c, int); selected[name] = c
        methods[name] = dict(B15_confirmed=int(robust[ii,c].sum()),
                             B15_possible=int((robust[ii,c]|unknown[ii,c]).sum()),
                             mean_success_lower=float(lower[ii,c].mean()),
                             mean_success_upper=float(upper[ii,c].mean()),
                             mean_oracle_regret_lower=float(np.maximum(0,lower.max(1)-upper[ii,c]).mean()),
                             collision_count=int(collision[ii,c].sum()),
                             selection_counts=np.bincount(c,minlength=16).tolist())
    if safety is not None:
        b = np.asarray([s["success"] >= 15 for s in safety], bool)
        q = np.asarray([s["success"] / 16 for s in safety])
        methods["Safety"] = dict(B15_confirmed=int(b.sum()), B15_possible=int(sum(
            s["success"]+s["numerical"]>=15 for s in safety)),
            mean_success_lower=float(q.mean()),
            mean_success_upper=float(np.mean([(s["success"]+s["numerical"])/16 for s in safety])),
            collision_count=int(sum(s["collision"] for s in safety)))
    a = selected["Q_state"]; b = selected["Q_eta"]
    comparisons = dict(state_minus_eta_B15=bootstrap(robust[ii,a].astype(float)-robust[ii,b].astype(float)),
                       state_minus_eta_meanQ=bootstrap(lower[ii,a]-lower[ii,b]))
    if safety is not None:
        bs = np.asarray([s["success"]>=15 for s in safety], float)
        qs = np.asarray([s["success"]/16 for s in safety])
        comparisons["state_minus_safety_B15"] = bootstrap(robust[ii,a].astype(float)-bs)
        comparisons["state_minus_safety_meanQ"] = bootstrap(lower[ii,a]-qs)
    result = dict(panel=panel, fold=fold, sources=list(PANELS[panel]), source_variant=VARIANT,
                  test_states=n,
                  historical_test_labels_previously_opened=True, new_rollouts=0,
                  target_truth_sha256=sha(truth_path),
                  prediction_sha256=sha(HERE/panel/"predictions.npz"),
                  methods=methods, paired=comparisons,
                  safety_limit=("historical MAC_ONLY is not hard Safety; unavailable" if fold=="toy" else None),
                  numerical_unknown_slots=int(sum(sum(s["unknown_seeds"]) for s in truth)))
    write(HERE/panel/"metrics.json", result)
    print(panel, {k:(v["B15_confirmed"],v["B15_possible"]) for k,v in methods.items()},
          comparisons["state_minus_eta_B15"], flush=True)


def shuffle(panel):
    """Frozen-model diagnostic: permute physical states, retain each eta pool."""
    assert all((HERE/p/"predictions.npz").exists() for p in PANELS)
    fold = TARGET[panel]
    truth_path = (FIRST if fold == "ring" else OLD/fold) / "target_truth.json"
    truth = read(truth_path)
    manifest, x_np = target_input(fold)
    pred = np.load(HERE/panel/"predictions.npz")
    ids = list(pred["state_uids"])
    assert ids == [s["state_uid"] for s in truth] == [s["state_uid"] for s in manifest]
    n = len(ids); ix = np.arange(n)
    robust = np.asarray([[v is True for v in s["robust"]] for s in truth], bool)
    lower = np.asarray([s["lower"] for s in truth], float)
    info = read(HERE/panel/"state"/"fit.json")
    ck = HERE/panel/"state"/"checkpoint.msgpack"
    assert sha(ck) == info["checkpoint_sha256"]
    e = jnp.asarray((np.asarray([s["eta"] for s in manifest], np.float32)-
                     np.asarray(info["eta_center"])) / np.asarray(info["eta_radius"]))
    x = {k:jnp.asarray(v) for k,v in x_np.items()}
    net = rep.Critic()
    get = lambda ii: {k:v[ii] for k,v in x.items()}
    template = net.init(jax.random.PRNGKey(17),get(jnp.array([0])),e[:1,:1].reshape(1,3))
    params = serialization.from_bytes(template,ck.read_bytes())
    flat_e = e.reshape(-1,3)
    flat_i = np.repeat(ix,16)
    rng = np.random.default_rng(20261008)
    shuffled=[]
    for replicate in range(32):
        perm = rng.permutation(n)
        while np.any(perm == ix):perm = rng.permutation(n)
        scores=[]
        for a in range(0,len(flat_i),256):
            jj=jnp.asarray(perm[flat_i[a:a+256]])
            scores.append(np.asarray(net.apply(params,get(jj),flat_e[a:a+256])))
        choice=np.concatenate(scores).reshape(n,16).argmax(1)
        shuffled.append(dict(replicate=replicate,choices=choice.tolist(),
                             B15=int(robust[ix,choice].sum()),
                             meanQ=float(lower[ix,choice].mean()),
                             selection_changed=int(np.sum(choice != pred["state_choice"]))))
    observed=pred["state_choice"]
    result=dict(panel=panel,source_variant=VARIANT,states=n,replicates=32,
                observed_B15=int(robust[ix,observed].sum()),
                observed_meanQ=float(lower[ix,observed].mean()),
                shuffled_B15_mean=float(np.mean([r["B15"] for r in shuffled])),
                shuffled_B15_range=[min(r["B15"] for r in shuffled),max(r["B15"] for r in shuffled)],
                shuffled_meanQ_mean=float(np.mean([r["meanQ"] for r in shuffled])),
                mean_selection_changed=float(np.mean([r["selection_changed"] for r in shuffled])),
                details=shuffled)
    write(HERE/panel/"state_shuffle.json",result)
    print(panel,"observed",result["observed_B15"],"shuffle_mean",result["shuffled_B15_mean"],flush=True)


def report():
    rows = [read(HERE/p/"metrics.json") for p in PANELS]
    assert len(rows)==8
    lines = [f"# Cache-first physical-only critic pilot ({VARIANT} source)", "",
             "This is an exploratory, zero-new-rollout result on historical FF execution outcomes.",
             "Both critics used the same cached source labels and K16 target candidates;",
             "the state critic received masked physical entities plus eta, with Flow/controller channels zero.",
             "All eight predictions were frozen before this script opened target truths.",
             "Historical target labels were already opened in earlier studies, and source state coverage",
             "is limited/heterogeneous; do not read this as fresh-state formal confirmation.", "",
             ("The enriched-source variant is ineligible for strict LOSO because some source proposals"
              " may have ancestry in an all-scene generator; its LOSO rows are exploratory only."
              if VARIANT=="enriched" else
              "The strict-source variant excludes all-scene-generator-derived source proposals,"
              " following the archived strict LOSO source rule."), "",
             "| Panel | Train scenes | Test N | Safety | Q(eta) | Q(s,eta) | Oracle | State−eta B15 pp [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m=row["methods"]; d=row["paired"]["state_minus_eta_B15"]
        def fmt(k):return f"{m[k]['B15_confirmed']}/{row['test_states']}" if k in m else "n/a"
        lines.append(f"| {row['panel']} | {', '.join(row['sources'])} | {row['test_states']} | "
                     f"{fmt('Safety')} | {fmt('Q_eta')} | {fmt('Q_state')} | {fmt('Oracle')} | "
                     f"{100*d['mean']:+.1f} [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] |")
    lines += ["", "Toy Safety is n/a: its historical extra controller was MAC_ONLY and has different",
              "success semantics. DB/Four/Ring Safety is hard-safety Q16 from the shared database.",
              "Mean success, oracle regret, collisions, numerical bounds and paired state-level",
              "statistics are in each panel's metrics.json. Full-scale paused evidence remains intact.", ""]
    if all((HERE/p/"state_shuffle.json").exists() for p in PANELS):
        lines += ["| Panel | Observed state B15 | Mean B15 after 32 physical-state derangements |",
                  "|---|---:|---:|"]
        for panel in PANELS:
            s=read(HERE/panel/"state_shuffle.json")
            lines.append(f"| {panel} | {s['observed_B15']}/{s['states']} | {s['shuffled_B15_mean']:.1f}/{s['states']} |")
        lines += ["", "State derangement retains every target eta pool and frozen critic weight; a drop",
                  "indicates the selector uses the physical state rather than only a global eta tendency.", ""]
    if all((HERE/p/"capacity_control.json").exists() for p in PANELS):
        lines += ["| Panel | Q(s,eta) B15 | Wider Q(eta) B15 | State−wider eta pp [95% CI] |",
                  "|---|---:|---:|---:|"]
        for panel in PANELS:
            z=read(HERE/panel/"capacity_control.json")
            d=z["paired_state_minus_wide_B15"]
            lines.append(f"| {panel} | {z['B15']['Q_state']}/{z['test_states']} | "
                         f"{z['B15']['Q_eta_wide']}/{z['test_states']} | "
                         f"{100*d['mean']:+.1f} [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] |")
        lines += ["", "The wider eta-only control uses the same source labels and validation rule",
                  "with 32→128→64 eta/head widths matching the state critic's downstream head.", ""]
    (HERE/"REPORT.md").write_text("\n".join(lines))
    print(HERE/"REPORT.md")


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("action", choices=("fit","predict","evaluate","shuffle","capacity","report"))
    p.add_argument("--panel", choices=tuple(PANELS));p.add_argument("--kind", choices=("state","eta_only","eta_only_wide"))
    a=p.parse_args()
    if a.action=="fit":
        assert a.panel and a.kind
        fit(a.panel,a.kind)
    elif a.action=="predict":
        assert a.panel
        predict(a.panel)
    elif a.action=="evaluate":
        assert a.panel
        evaluate(a.panel)
    elif a.action=="shuffle":
        assert a.panel
        shuffle(a.panel)
    elif a.action=="capacity":
        assert a.panel
        capacity(a.panel)
    else:report()
