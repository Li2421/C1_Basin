#!/usr/bin/env python3
"""Joint, mode-free OrthoFlow3 eta generator and finite-proposal Q critic.

This module consumes only the frozen train/validation basin dataset.  The
generator is trained and selected first; its selected checkpoint is then
frozen before the critic is trained.  No benchmark test state is read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import flax.linen as nn
from flax import serialization, core
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/orthoflow3_basin_dataset_v1"
SCENARIOS = ("double_bottleneck", "four_way_intersection", "ring_exchange")
SEEDS = (17, 23, 41)
K = 4
LOW = np.asarray([0.5, -0.5, 0.0], np.float32)
HIGH = np.asarray([1.25, 0.5, 0.75], np.float32)
CENTER = (LOW + HIGH) / 2
RADIUS = (HIGH - LOW) / 2


def dump(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_int(*parts: Any) -> int:
    text = "\0".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def parse(row: dict[str, Any], key: str) -> Any:
    value = row[key]
    return json.loads(value) if isinstance(value, str) else value


def scalar_environment_fields(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            out.update(scalar_environment_fields(value[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, str) and prefix.endswith(("kind", "schema")):
        out[f"{prefix}={value}"] = 1.0
    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
        for index, item in enumerate(value):
            out[f"{prefix}[{index}]"] = float(item)
    return out


class Generator(nn.Module):
    """One q(eta|h,c), with native-input adapters and a shared trunk."""

    def setup(self):
        self.db_adapter = nn.Dense(96)
        self.four_adapter = nn.Dense(96)
        self.ring_adapter = nn.Dense(96)
        self.context = nn.Dense(32)
        self.shared1 = nn.Dense(128)
        self.shared2 = nn.Dense(64)
        self.out = nn.Dense(6)

    def branch(self, h, c, adapter):
        z = nn.silu(adapter(h))
        ec = nn.silu(self.context(c))
        x = jnp.concatenate((z, ec), axis=-1)
        x = nn.silu(self.shared1(x))
        x = nn.silu(self.shared2(x))
        return self.out(x)

    def db(self, h, c): return self.branch(h, c, self.db_adapter)
    def four(self, h, c): return self.branch(h, c, self.four_adapter)
    def ring(self, h, c): return self.branch(h, c, self.ring_adapter)
    def __call__(self, hd, cd, hf, cf, hr, cr):
        return self.db(hd, cd), self.four(hf, cf), self.ring(hr, cr)


class Critic(nn.Module):
    """Continuous Q(h,c,eta), sharing all post-adapter computation."""

    def setup(self):
        self.db_adapter = nn.Dense(96)
        self.four_adapter = nn.Dense(96)
        self.ring_adapter = nn.Dense(96)
        self.context = nn.Dense(32)
        self.eta_encoder = nn.Dense(32)
        self.shared1 = nn.Dense(128)
        self.shared2 = nn.Dense(64)
        self.out = nn.Dense(1)

    def branch(self, h, c, eta, adapter):
        z = nn.silu(adapter(h))
        ec = nn.silu(self.context(c))
        ee = nn.silu(self.eta_encoder(eta))
        x = jnp.concatenate((z, ec, ee), axis=-1)
        x = nn.silu(self.shared1(x))
        x = nn.silu(self.shared2(x))
        return self.out(x)[..., 0]

    def db(self, h, c, eta): return self.branch(h, c, eta, self.db_adapter)
    def four(self, h, c, eta): return self.branch(h, c, eta, self.four_adapter)
    def ring(self, h, c, eta): return self.branch(h, c, eta, self.ring_adapter)
    def __call__(self, hd, cd, ed, hf, cf, ef, hr, cr, er):
        return self.db(hd, cd, ed), self.four(hf, cf, ef), self.ring(hr, cr, er)


METHOD = {"double_bottleneck": "db", "four_way_intersection": "four", "ring_exchange": "ring"}


def dist_params(raw):
    return raw[..., :3], 0.025 + 0.275 * jax.nn.sigmoid(raw[..., 3:])


def eta_mean(raw):
    return jnp.asarray(CENTER) + jnp.asarray(RADIUS) * jnp.tanh(raw[..., :3])


def eta_from_noise(raw, noise):
    mu, sigma = dist_params(raw)
    return jnp.asarray(CENTER) + jnp.asarray(RADIUS) * jnp.tanh(mu + sigma * noise)


def eta_normalized(eta):
    return (eta - jnp.asarray(CENTER)) / jnp.asarray(RADIUS)


def log_prob(raw, eta):
    u = jnp.clip(eta_normalized(eta), -0.999999, 0.999999)
    z = jnp.arctanh(u)
    mu, sigma = dist_params(raw)
    gaussian = -0.5 * (((z - mu) / sigma) ** 2 + 2 * jnp.log(sigma) + math.log(2 * math.pi))
    jac = jnp.log(jnp.asarray(RADIUS) * (1 - u * u) + 1e-9)
    return jnp.sum(gaussian - jac, axis=-1)


def merge_initialized(module, dims, cdim, critic=False):
    """Initialize every named adapter while retaining one shared parameter set."""
    key = jax.random.PRNGKey(0)
    args = []
    for sc in SCENARIOS:
        args.extend([jnp.zeros((1, dims[sc])), jnp.zeros((1, cdim))])
        if critic:
            args.append(jnp.zeros((1, 3)))
    return module.init(key, *args)


def load_data():
    states = pq.read_table(DATA / "states.parquet").to_pylist()
    labels = pq.read_table(DATA / "eta_labels.parquet").to_pylist()
    state_by_uid = {r["state_uid"]: r for r in states}
    env_maps = [scalar_environment_fields(parse(r, "environment_descriptor")) for r in states]
    env_keys = sorted({key for row in env_maps for key in row})
    scenario_key = {sc: index for index, sc in enumerate(SCENARIOS)}
    prepared = {}
    normalization = {"environment_keys": env_keys, "scenarios": {}}
    for sc in SCENARIOS:
        ss = [r for r in states if r["scenario"] == sc]
        h = np.asarray([parse(r, "conditioning")["flat"] for r in ss], np.float32)
        c = np.zeros((len(ss), len(env_keys) + len(SCENARIOS)), np.float32)
        for i, row in enumerate(ss):
            values = scalar_environment_fields(parse(row, "environment_descriptor"))
            for j, key in enumerate(env_keys): c[i, j] = values.get(key, 0.0)
            c[i, len(env_keys) + scenario_key[sc]] = 1.0
        train = np.asarray([r["split"] == "train" for r in ss])
        hm, hs = h[train].mean(0), h[train].std(0)
        cm, cs = c[train].mean(0), c[train].std(0)
        hs[hs < 1e-6] = 1.0; cs[cs < 1e-6] = 1.0
        normalization["scenarios"][sc] = {"h_mean": hm.tolist(), "h_std": hs.tolist(),
                                                "c_mean": cm.tolist(), "c_std": cs.tolist()}
        index = {row["state_uid"]: i for i, row in enumerate(ss)}
        prepared[sc] = {"states": ss, "h": (h-hm)/hs, "c": (c-cm)/cs, "index": index,
                        "labels": defaultdict(list)}
    ignored = Counter()
    for row in labels:
        if row["state_uid"] not in state_by_uid:
            ignored["unknown_state"] += 1; continue
        if row["numerical_failure_count"] and row["seed_count"] == 0:
            ignored["numerical_uncertified"] += 1; continue
        eta = np.asarray(parse(row, "eta_raw"), np.float32)
        # eta=0 is retained for critic/gating diagnostics but is outside the
        # generator's frozen proposal box and cannot supervise its density.
        item = {**row, "eta": eta, "q": row["success_count"] / max(row["seed_count"], 1),
                "weight": min(int(row["seed_count"]), 16)}
        prepared[row["scenario"]]["labels"][row["state_uid"]].append(item)
    for sc in SCENARIOS:
        prepared[sc]["positive"] = {}
        prepared[sc]["negative"] = {}
        for uid_, rows in prepared[sc]["labels"].items():
            in_box = [r for r in rows if np.all(r["eta"] >= LOW-1e-8) and np.all(r["eta"] <= HIGH+1e-8)]
            prepared[sc]["positive"][uid_] = [r for r in in_box if r["robust_15of16"] is True]
            prepared[sc]["negative"][uid_] = [r for r in in_box if r["robust_15of16"] is False]
    dump("normalization.json", normalization)
    return prepared, normalization, ignored


def split_state_ids(data, sc, split):
    return [r["state_uid"] for r in data[sc]["states"] if r["split"] == split]


def positive_rows(data, sc, uid):
    return data[sc]["positive"].get(uid, [])


def negative_rows(data, sc, uid):
    return data[sc]["negative"].get(uid, [])


def generator_batch(data, sc, split, rng, batch=64):
    ids = [uid for uid in split_state_ids(data, sc, split) if positive_rows(data, sc, uid)]
    chosen = [ids[i] for i in rng.integers(0, len(ids), batch)]
    h, c, ep, en, hasneg = [], [], [], [], []
    for uid in chosen:
        p = positive_rows(data, sc, uid)
        n = negative_rows(data, sc, uid)
        pr = p[int(rng.integers(0, len(p)))]
        nr = n[int(rng.integers(0, len(n)))] if n else pr
        i = data[sc]["index"][uid]
        h.append(data[sc]["h"][i]); c.append(data[sc]["c"][i]); ep.append(pr["eta"]); en.append(nr["eta"]); hasneg.append(bool(n))
    return tuple(np.asarray(x, np.float32) for x in (h, c, ep, en, hasneg))


def generator_validation_loss(model, params, data):
    values = []
    for sc in SCENARIOS:
        ids = [uid for uid in split_state_ids(data, sc, "validation") if positive_rows(data, sc, uid)]
        terms = []
        for uid in ids:
            i = data[sc]["index"][uid]
            raw = model.apply(params, jnp.asarray(data[sc]["h"][i:i+1]),
                              jnp.asarray(data[sc]["c"][i:i+1]), method=getattr(model, METHOD[sc]))
            p = positive_rows(data, sc, uid)
            lp = np.asarray(log_prob(raw.repeat(len(p), 0), jnp.asarray(np.asarray([r["eta"] for r in p]))))
            n = negative_rows(data, sc, uid)
            rank = 0.0
            if n:
                nr = n[stable_int("val-neg", uid) % len(n)]
                pr = p[stable_int("val-pos", uid) % len(p)]
                lpn = float(log_prob(raw, jnp.asarray(nr["eta"])[None])[0])
                lpp = float(log_prob(raw, jnp.asarray(pr["eta"])[None])[0])
                rank = float(np.logaddexp(0, 0.5 + lpn - lpp))
            terms.append(-float(np.mean(lp)) + 0.25*rank)
        values.append(float(np.mean(terms)))
    return float(np.mean(values))


def train_generator(data):
    dims = {sc: data[sc]["h"].shape[1] for sc in SCENARIOS}
    cdim = next(iter(data.values()))["c"].shape[1]
    model = Generator()
    config = {"architecture": "scenario native adapters 96; physical context 32; shared 128-64; diagonal squashed Gaussian",
              "sigma": "0.025+0.275*sigmoid", "eta_domain": [LOW.tolist(), HIGH.tolist()],
              "loss": "scenario/state-balanced robust NLL + 0.25 within-state softplus negative rank margin 0.5",
              "optimizer": "AdamW lr=5e-4 wd=1e-4 clip=5", "steps": 3000,
              "batch_states_per_scenario": 64, "seeds": list(SEEDS), "K": K}
    dump("generator/config.json", config)
    results = []
    for seed in SEEDS:
        params = merge_initialized(model, dims, cdim)
        opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(5e-4, weight_decay=1e-4))
        state = opt.init(params)
        @jax.jit
        def step(p, st, *args):
            def objective(q):
                losses = []
                for k, sc in enumerate(SCENARIOS):
                    h,c,ep,en,hn = args[5*k:5*k+5]
                    raw = model.apply(q, h, c, method=getattr(model, METHOD[sc]))
                    pos = -jnp.mean(log_prob(raw, ep))
                    rank = jnp.sum(hn * jax.nn.softplus(0.5 + log_prob(raw,en)-log_prob(raw,ep))) / jnp.maximum(jnp.sum(hn),1)
                    losses.append(pos + 0.25*rank)
                return jnp.mean(jnp.stack(losses))
            loss, grad = jax.value_and_grad(objective)(p)
            updates, st = opt.update(grad, st, p)
            return optax.apply_updates(p, updates), st, loss
        rngs = {sc: np.random.default_rng(stable_int("generator",seed,sc)) for sc in SCENARIOS}
        best = (float("inf"), None, 0); history=[]; stale=0
        for iteration in range(1,3001):
            args=[]
            for sc in SCENARIOS: args.extend(jnp.asarray(x) for x in generator_batch(data,sc,"train",rngs[sc]))
            params,state,loss=step(params,state,*args)
            if iteration % 100 == 0:
                val=generator_validation_loss(model,params,data)
                history.append({"step":iteration,"train":float(loss),"validation":val})
                if val < best[0]-1e-5: best=(val,serialization.to_bytes(params),iteration); stale=0
                else: stale += 1
                if stale >= 8 and iteration >= 1200: break
        path=OUT/f"generator/seed{seed}/checkpoint.msgpack"; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(best[1])
        dump(f"generator/seed{seed}/history.json",history)
        results.append({"seed":seed,"validation_loss":best[0],"best_step":best[2],"steps":iteration,"checkpoint":str(path),"sha256":sha(path)})
    selected=min(results,key=lambda x:(x["validation_loss"],x["seed"]))
    dump("generator/training.json",{"results":results,"selected":selected})
    template=merge_initialized(model,dims,cdim)
    return model,serialization.from_bytes(template,Path(selected["checkpoint"]).read_bytes()),selected


def nearest_label_metrics(rows, eta):
    usable=[r for r in rows if np.all(r["eta"]>=LOW-1e-8) and np.all(r["eta"]<=HIGH+1e-8)]
    z=np.asarray([(r["eta"]-CENTER)/RADIUS for r in usable]); target=(np.asarray(eta)-CENTER)/RADIUS
    j=int(np.argmin(np.linalg.norm(z-target,axis=1))); row=usable[j]
    return {"q":float(row["q"]),"robust":bool(row["robust_15of16"]),"distance":float(np.linalg.norm(z[j]-target)),"eta_uid":row["eta_uid"]}


def fixed_anchor(data):
    # Training-only shared candidate with maximum state-level robust coverage;
    # ties favor minimum normalized norm and eta UID.
    evidence=defaultdict(set); coordinates={}
    for sc in SCENARIOS:
        for uid in split_state_ids(data,sc,"train"):
            for r in positive_rows(data,sc,uid):
                evidence[r["eta_uid"]].add((sc,uid)); coordinates[r["eta_uid"]]=r["eta"]
    best=min(evidence,key=lambda k:(-len(evidence[k]),float(np.linalg.norm((coordinates[k]-CENTER)/RADIUS)),k))
    return {"eta_uid":best,"eta":coordinates[best].tolist(),"covered_train_states":len(evidence[best])}


def freeze_generator_proposals(model,params,selected,data):
    anchor=fixed_anchor(data); rows=[]; sigma_rows=[]
    for sc in SCENARIOS:
        for uid in split_state_ids(data,sc,"validation"):
            i=data[sc]["index"][uid]; h=data[sc]["h"][i:i+1]; c=data[sc]["c"][i:i+1]
            raw=np.asarray(model.apply(params,jnp.asarray(h),jnp.asarray(c),method=getattr(model,METHOD[sc])))[0]
            mu,sigma=map(np.asarray,dist_params(jnp.asarray(raw)))
            mean=np.asarray(eta_mean(jnp.asarray(raw)))
            rng=np.random.default_rng(stable_int("generator-v1-proposals",selected["seed"],sc,uid))
            samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(K,0),jnp.asarray(rng.standard_normal((K,3)),jnp.float32)))
            labels=data[sc]["labels"][uid]
            proposals=[{"kind":"generator_mean","eta":mean.tolist(),"nearest":nearest_label_metrics(labels,mean)}]
            proposals += [{"kind":f"sample_{k}","eta":samples[k].tolist(),"nearest":nearest_label_metrics(labels,samples[k])} for k in range(K)]
            rows.append({"state_uid":uid,"state_id":data[sc]["states"][i]["state_id"],"scenario":sc,
                         "split":"validation","zero_sufficient":bool(data[sc]["states"][i]["zero_sufficient"]),
                         "generator_mean":mean.tolist(),"samples":samples.tolist(),"sigma":sigma.tolist(),
                         "proposals":proposals,"proposal_seed":stable_int("generator-v1-proposals",selected["seed"],sc,uid)})
            sigma_rows.append((sc,sigma,mean))
    payload={"K":K,"generator_checkpoint":selected,"proposal_seed_rule":"SHA256(generator-v1-proposals|checkpoint seed|scenario|state_uid)",
             "fixed_shared_anchor":anchor,"eta_zero_separate_baseline":True,"states":rows}
    dump("generator/frozen_validation_proposals.json",payload)
    metrics={}
    for sc in SCENARIOS:
        rr=[r for r in rows if r["scenario"]==sc]
        meanq=[r["proposals"][0]["nearest"]["q"] for r in rr]
        meanrob=[r["proposals"][0]["nearest"]["robust"] for r in rr]
        oracleq=[max(p["nearest"]["q"] for p in r["proposals"]) for r in rr]
        oracler=[any(p["nearest"]["robust"] for p in r["proposals"]) for r in rr]
        dist=[p["nearest"]["distance"] for r in rr for p in r["proposals"]]
        sg=np.asarray([r["sigma"] for r in rr]); means=np.asarray([r["generator_mean"] for r in rr])
        metrics[sc]={"states":len(rr),"generator_mean_nearest_label_q":float(np.mean(meanq)),
                     "generator_mean_nearest_robust_rate":float(np.mean(meanrob)),
                     "oracle_best_of_mean_plus_K_nearest_q":float(np.mean(oracleq)),
                     "oracle_robust_coverage_nearest_proxy":float(np.mean(oracler)),
                     "nearest_label_distance_mean":float(np.mean(dist)),"nearest_label_distance_p95":float(np.quantile(dist,.95)),
                     "sigma_mean":sg.mean(0).tolist(),"sigma_median":np.median(sg,0).tolist(),
                     "boundary_saturation_fraction":float(np.mean(np.abs((means-CENTER)/RADIUS)>=.98))}
    dump("generator/validation_label_metrics.json",metrics)
    return payload,metrics


def critic_batch(data,sc,split,rng,batch=96):
    ids=[uid for uid in split_state_ids(data,sc,split) if data[sc]["labels"][uid]]
    chosen=[ids[i] for i in rng.integers(0,len(ids),batch)]
    h=[];c=[];eta=[];y=[];w=[]
    for uid in chosen:
        allrows=[r for r in data[sc]["labels"][uid] if not (r["numerical_failure_count"] and r["seed_count"]==0)]
        pos=[r for r in allrows if r["robust_15of16"] is True]; neg=[r for r in allrows if r["robust_15of16"] is False]
        pool=(pos if rng.random()<.5 and pos else neg if neg else pos)
        r=pool[int(rng.integers(0,len(pool)))]; i=data[sc]["index"][uid]
        h.append(data[sc]["h"][i]);c.append(data[sc]["c"][i]);eta.append((r["eta"]-CENTER)/RADIUS);y.append(r["q"]);w.append(r["weight"])
    return tuple(np.asarray(x,np.float32) for x in (h,c,eta,y,w))


def critic_val_loss(model,params,data):
    vals=[]
    for sc in SCENARIOS:
        rows=[]
        for uid in split_state_ids(data,sc,"validation"): rows.extend(data[sc]["labels"][uid])
        # deterministic state-balanced weight: evidence weight divided by rows in state
        by=Counter(r["state_uid"] for r in rows)
        h=np.asarray([data[sc]["h"][data[sc]["index"][r["state_uid"]]] for r in rows]); c=np.asarray([data[sc]["c"][data[sc]["index"][r["state_uid"]]] for r in rows])
        e=np.asarray([(r["eta"]-CENTER)/RADIUS for r in rows]); y=np.asarray([r["q"] for r in rows]); w=np.asarray([r["weight"]/by[r["state_uid"]] for r in rows])
        logits=np.asarray(model.apply(params,jnp.asarray(h),jnp.asarray(c),jnp.asarray(e),method=getattr(model,METHOD[sc])))
        loss=np.sum(w*(np.logaddexp(0,logits)-y*logits))/np.sum(w);vals.append(loss)
    return float(np.mean(vals))


def train_critic(data):
    dims={sc:data[sc]["h"].shape[1] for sc in SCENARIOS}; cdim=next(iter(data.values()))["c"].shape[1]
    model=Critic(); config={"architecture":"scenario native adapters 96; physical context 32; eta 32; shared 128-64-1",
      "target":"empirical seed success Q","loss":"binomial BCE weighted by min(actual seeds,16)",
      "sampling":"uniform scenario, then state, then class-balanced eta label","optimizer":"AdamW lr=1e-3 wd=1e-4 clip=5",
      "steps":4000,"batch_states_per_scenario":96,"seeds":list(SEEDS),"generator_frozen_first":True}
    dump("critic/config.json",config);results=[]
    for seed in SEEDS:
        params=merge_initialized(model,dims,cdim,critic=True);opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4));state=opt.init(params)
        @jax.jit
        def step(p,st,*args):
            def objective(q):
                losses=[]
                for k,sc in enumerate(SCENARIOS):
                    h,c,e,y,w=args[5*k:5*k+5];logit=model.apply(q,h,c,e,method=getattr(model,METHOD[sc]))
                    losses.append(jnp.sum(w*optax.sigmoid_binary_cross_entropy(logit,y))/jnp.sum(w))
                return jnp.mean(jnp.stack(losses))
            loss,grad=jax.value_and_grad(objective)(p);updates,st=opt.update(grad,st,p);return optax.apply_updates(p,updates),st,loss
        rngs={sc:np.random.default_rng(stable_int("critic",seed,sc)) for sc in SCENARIOS};best=(float("inf"),None,0);history=[];stale=0
        for iteration in range(1,4001):
            args=[]
            for sc in SCENARIOS:args.extend(jnp.asarray(x) for x in critic_batch(data,sc,"train",rngs[sc]))
            params,state,loss=step(params,state,*args)
            if iteration%100==0:
                val=critic_val_loss(model,params,data);history.append({"step":iteration,"train":float(loss),"validation":val})
                if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),iteration);stale=0
                else:stale+=1
                if stale>=10 and iteration>=1500:break
        path=OUT/f"critic/seed{seed}/checkpoint.msgpack";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(best[1]);dump(f"critic/seed{seed}/history.json",history)
        results.append({"seed":seed,"validation_loss":best[0],"best_step":best[2],"steps":iteration,"checkpoint":str(path),"sha256":sha(path)})
    selected=min(results,key=lambda x:(x["validation_loss"],x["seed"]));dump("critic/training.json",{"results":results,"selected":selected})
    template=merge_initialized(model,dims,cdim,critic=True)
    return model,serialization.from_bytes(template,Path(selected["checkpoint"]).read_bytes()),selected


def auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score)
    if y.sum()==0 or (~y).sum()==0:return None
    ranks=rankdata(score);return float((ranks[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))


def evaluate_critic(model,params,data,proposals):
    metrics={};proposal_results=[]
    frozen={(r["scenario"],r["state_uid"]):r for r in proposals["states"]}
    for sc in SCENARIOS:
        rows=[]
        for uid in split_state_ids(data,sc,"validation"):rows.extend(data[sc]["labels"][uid])
        h=np.asarray([data[sc]["h"][data[sc]["index"][r["state_uid"]]] for r in rows]);c=np.asarray([data[sc]["c"][data[sc]["index"][r["state_uid"]]] for r in rows]);e=np.asarray([(r["eta"]-CENTER)/RADIUS for r in rows])
        y=np.asarray([r["q"] for r in rows]);rob=np.asarray([r["robust_15of16"] is True for r in rows]);pred=1/(1+np.exp(-np.asarray(model.apply(params,jnp.asarray(h),jnp.asarray(c),jnp.asarray(e),method=getattr(model,METHOD[sc])))))
        by=defaultdict(list)
        for i,r in enumerate(rows):by[r["state_uid"]].append(i)
        correct=total=0;rhos=[];boundary=[]
        for ix in by.values():
            if len(ix)>2:
                rho=spearmanr(pred[ix],y[ix]).statistic
                if np.isfinite(rho):rhos.append(rho)
            for a in range(len(ix)):
                for b in range(a+1,len(ix)):
                    i,j=ix[a],ix[b]
                    if abs(y[i]-y[j])<.25:continue
                    total+=1;correct+=int((pred[i]-pred[j])*(y[i]-y[j])>0)
            boundary.extend([i for i in ix if .25<y[i]<.9])
        metrics[sc]={"eta_labels":len(rows),"q_mae":float(np.mean(abs(pred-y))),"q_brier":float(np.mean((pred-y)**2)),"q_spearman":float(spearmanr(pred,y).statistic),"robust_auroc":auc(rob,pred),"within_state_spearman":float(np.mean(rhos)),"pairwise_ranking_accuracy_gap025":correct/max(total,1),"ranking_pairs":total,"near_boundary_q_mae":float(np.mean(abs(pred[boundary]-y[boundary]))) if boundary else None,"nonrobust_q_mae":float(np.mean(abs(pred[~rob]-y[~rob]))) if (~rob).any() else None,"mean_predicted_q":float(pred.mean()),"mean_empirical_q":float(y.mean())}
        for uid in split_state_ids(data,sc,"validation"):
            r=frozen[(sc,uid)];etas=np.asarray([r["generator_mean"],*r["samples"]]);i=data[sc]["index"][uid]
            hh=np.repeat(data[sc]["h"][i:i+1],len(etas),0);cc=np.repeat(data[sc]["c"][i:i+1],len(etas),0);ee=(etas-CENTER)/RADIUS
            score=1/(1+np.exp(-np.asarray(model.apply(params,jnp.asarray(hh),jnp.asarray(cc),jnp.asarray(ee),method=getattr(model,METHOD[sc])))))
            truth=np.asarray([p["nearest"]["q"] for p in r["proposals"]]);pick=int(np.argmax(score));oracle=int(np.argmax(truth))
            proposal_results.append({"scenario":sc,"state_uid":uid,"critic_scores":score.tolist(),"critic_index":pick,"oracle_index":oracle,"critic_nearest_q":float(truth[pick]),"oracle_nearest_q":float(truth[oracle]),"regret":float(truth[oracle]-truth[pick]),"critic_nearest_robust":bool(r["proposals"][pick]["nearest"]["robust"]),"oracle_has_robust":bool(any(p["nearest"]["robust"] for p in r["proposals"])),"critic_selected_eta":etas[pick].tolist()})
    for sc in SCENARIOS:
        rr=[r for r in proposal_results if r["scenario"]==sc];cover=[r for r in rr if r["oracle_has_robust"]]
        metrics[sc]["finite_proposal"]={"states":len(rr),"mean_oracle_nearest_q":float(np.mean([r["oracle_nearest_q"] for r in rr])),"mean_critic_selected_nearest_q":float(np.mean([r["critic_nearest_q"] for r in rr])),"critic_gap":float(np.mean([r["regret"] for r in rr])),"robust_selection_rate_on_coverable":float(np.mean([r["critic_nearest_robust"] for r in cover])) if cover else None,"misranking_count_regret_ge025":sum(r["regret"]>=.25 for r in rr)}
    dump("critic/validation_metrics.json",metrics);dump("critic/frozen_validation_selection.json",{"critic_selection_before_rollout":True,"states":proposal_results})
    return metrics,proposal_results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,norm,ignored=load_data()
    audit={"dataset_states_sha256":sha(DATA/"states.parquet"),"dataset_eta_labels_sha256":sha(DATA/"eta_labels.parquet"),"dataset_manifest_sha256":sha(DATA/"manifest.json"),"frozen_test_states_read":0,"mode_labels_used":False,"ignored":dict(ignored),"state_counts":{sc:{sp:len(split_state_ids(data,sc,sp)) for sp in ("train","validation")} for sc in SCENARIOS}}
    dump("input_audit.json",audit)
    gm,gp,gs=train_generator(data)
    frozen,gmetrics=freeze_generator_proposals(gm,gp,gs,data)
    cm,cp,cs=train_critic(data)
    cmetrics,selections=evaluate_critic(cm,cp,data,frozen)
    dump("label_space_summary.json",{"generator":gmetrics,"critic":cmetrics,"generator_checkpoint":gs,"critic_checkpoint":cs,"K":K,"fixed_shared_anchor":frozen["fixed_shared_anchor"]})
    print(json.dumps({"generator":gmetrics,"critic":cmetrics},indent=2))


if __name__=="__main__":main()
