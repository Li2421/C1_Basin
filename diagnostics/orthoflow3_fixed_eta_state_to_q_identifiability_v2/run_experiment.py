#!/usr/bin/env python3
"""Fixed-exact-eta unseen-state identifiability audit; never executes rollouts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.stats import rankdata, spearmanr

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
SRC = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1"
DBPATH = ROOT / "shared_rollout_db/rollout.sqlite"
SOURCES = {
    "Toy": {"root": ROOT/"diagnostics/orthoflow3_shared_eta_codebook_v1", "scenario_name":"ToyGiveWay",
            "controller":"ctl_df736b67f6410260d87812e0a76946af7152d147c9a0a25189d1908a92567b34"},
    "DB": {"root": ROOT/"diagnostics/orthoflow3_db_shared_mode_transfer_v1", "scenario_name":"DoubleBottleneck_4A",
           "controller":"ctl_0ce9b25aa22d23cd3d07a01c61fd72f3c76cec4be3c053a3d4b8fbb6184f543d"},
}
SEEDS = (17, 23, 41)
SCENARIOS = ("Toy", "DB")


def dump(path, obj):
    p = H / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows, fields=None):
    p = H / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["empty"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def safe_float(x):
    if x is None or not np.isfinite(x):
        return None
    return float(x)


def nll_prob(p, y, w=None):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    e = -(y * np.log(p) + (1-y) * np.log(1-p))
    if w is None:
        return float(e.mean()) if len(e) else None
    w = np.asarray(w, float)
    return float(np.sum(w*e)/np.sum(w)) if len(e) and w.sum() else None


def auc(y, score):
    y = np.asarray(y, bool); score = np.asarray(score, float)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    ranks = rankdata(score)
    return float((ranks[y].sum() - pos*(pos+1)/2)/(pos*neg))


class MultiHead(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, h):
        x = nn.silu(nn.Dense(128)(h))
        x = nn.silu(nn.Dense(128)(x))
        x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(self.outputs)(x)


def load_live_database():
    """Aggregate current authoritative DB, resolving feature-state aliases exactly as the prior audit."""
    con=sqlite3.connect(DBPATH);con.row_factory=sqlite3.Row
    all_rows=[];features={};excluded=defaultdict(int)
    for scenario,spec in SOURCES.items():
        z=np.load(spec["root"]/"state_features.npz")
        ids=z["state_ids"].astype(str);splits=z["splits"].astype(str);features["toy" if scenario=="Toy" else "db"]=z["features"].astype(np.float32)
        uid_map={};records=[]
        for fi,(alias,split) in enumerate(zip(ids,splits)):
            cand=con.execute('''SELECT DISTINCT a.state_uid,s.source_group,s.identity_quality
                FROM state_alias a JOIN state s USING(state_uid)
                WHERE a.alias=? AND s.scenario_uid=(SELECT scenario_uid FROM scenario WHERE name=?)
                  AND s.identity_quality IN ('CONDITIONING_EXACT','SOURCE_GROUP_STABLE') ORDER BY a.state_uid''',(alias,spec["scenario_name"])).fetchall()
            exact=[r for r in cand if r["identity_quality"]=="CONDITIONING_EXACT"]
            if not exact:excluded[scenario+"_feature_without_exact_alias"]+=1;continue
            canonical=exact[0];sg=canonical["source_group"]
            compatible=sorted({r["state_uid"] for r in cand if r["source_group"]==sg}) or [canonical["state_uid"]]
            records.append((canonical["state_uid"],alias,sg,split,fi,compatible))
            for uid in compatible:uid_map[uid]=(canonical["state_uid"],alias,sg,split,fi)
        relevant=sorted(uid_map)
        marks=','.join('?'*len(relevant))
        seed_records=defaultdict(dict);meta={}
        sql=f'''SELECT r.state_uid,r.eta_uid,r.seed_key,r.success,r.deadlock,r.timeout,r.collision,e.eta1,e.eta2,e.eta3
                FROM rollout r JOIN eta e USING(eta_uid)
                WHERE r.controller_uid=? AND r.state_uid IN ({marks}) AND r.conflict_quarantined=0
                  AND r.numerical_failure=0 AND r.compatibility_quality='EXACT_REUSE' '''
        for r in con.execute(sql,[spec["controller"],*relevant]):
            canon,alias,sg,split,fi=uid_map[r["state_uid"]];key=(canon,r["eta_uid"]);sk=r["seed_key"]
            val=(int(r["success"]),int(r["deadlock"]),int(r["timeout"]),int(r["collision"]))
            if sk in seed_records[key] and seed_records[key][sk]!=val:seed_records[key][sk]=None
            else:seed_records[key][sk]=val
            meta[key]=(float(r["eta1"]),float(r["eta2"]),float(r["eta3"]),alias,sg,split,fi)
        # Aggregate-only evidence is admitted only if no seed-exact evidence exists. Never merge
        # aggregate rows because their unknown seed sets may overlap.
        aggregate=defaultdict(list)
        sql=f'''SELECT a.aggregate_uid,a.state_uid,a.eta_uid,a.n_trials,a.n_success,a.n_deadlock,a.n_timeout,a.n_collision,
                       a.seed_identities_known,a.evidence_class,e.eta1,e.eta2,e.eta3
                FROM aggregate_evidence a JOIN eta e USING(eta_uid)
                WHERE a.controller_uid=? AND a.state_uid IN ({marks}) AND a.conflict_quarantined=0 AND a.n_trials>0'''
        for r in con.execute(sql,[spec["controller"],*relevant]):
            canon,alias,sg,split,fi=uid_map[r["state_uid"]];key=(canon,r["eta_uid"])
            aggregate[key].append(r);meta.setdefault(key,(float(r["eta1"]),float(r["eta2"]),float(r["eta3"]),alias,sg,split,fi))
        keys=set(seed_records)|set(aggregate)
        for key in keys:
            eta1,eta2,eta3,alias,sg,split,fi=meta[key]
            if key in seed_records:
                seeds=seed_records[key]
                if any(v is None for v in seeds.values()):excluded[scenario+"_cross_alias_seed_conflict"]+=1;continue
                vals=list(seeds.values()); n=len(vals);ns=sum(v[0] for v in vals);nd=sum(v[1] for v in vals);nt=sum(v[2] for v in vals);nc=sum(v[3] for v in vals);klass="SEED_EXACT";known=True
            else:
                bestn=max(int(r["n_trials"]) for r in aggregate[key]);best=[r for r in aggregate[key] if int(r["n_trials"])==bestn]
                outcomes={(int(r["n_success"]),int(r["n_deadlock"] or 0),int(r["n_timeout"] or 0),int(r["n_collision"] or 0)) for r in best}
                if len(outcomes)>1:excluded[scenario+"_aggregate_conflict"]+=1;continue
                r=sorted(best,key=lambda q:q["aggregate_uid"])[0];n=int(r["n_trials"]);ns=int(r["n_success"]);nd=int(r["n_deadlock"] or 0);nt=int(r["n_timeout"] or 0);nc=int(r["n_collision"] or 0);klass="AGGREGATE_"+r["evidence_class"];known=bool(r["seed_identities_known"])
            all_rows.append({"scenario":scenario,"state_uid":key[0],"state_id":alias,"source_group":sg,"state_split":split,"feature_index":fi,
                             "eta_uid":key[1],"eta1":eta1,"eta2":eta2,"eta3":eta3,"controller_uid":spec["controller"],"n_trials":n,"n_success":ns,
                             "n_deadlock":nd,"n_timeout":nt,"n_collision":nc,"seed_identities_known":known,"evidence_class":klass,
                             "empirical_q":ns/n,"n_eff":min(n,16)})
    con.close()
    return all_rows,features,dict(excluded)


def source_split(rows, scenario):
    states = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        s = r["state_uid"]
        rec = states.setdefault(s, {"state_uid": s, "state_id": r["state_id"],
                                    "source_group": r["source_group"], "split": r["state_split"],
                                    "feature_index": int(r["feature_index"])})
        assert rec["split"] == r["state_split"] and rec["source_group"] == r["source_group"]
    groups = defaultdict(set)
    for r in states.values():
        groups[r["split"]].add(r["source_group"])
    overlap = {}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                overlap[f"{a}_{b}"] = len(groups[a] & groups[b])
    return {"scenario": scenario, "method": "frozen source/provenance-group split reused from canonical DB dataset",
            "states": sorted(states.values(), key=lambda x: x["state_uid"]),
            "counts": {s: sum(x["split"] == s for x in states.values()) for s in ("train","val","test")},
            "source_group_counts": {s: len(groups[s]) for s in ("train","val","test")},
            "source_group_overlap": overlap}


def audit_etas(rows, scenario):
    grouped = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario:
            grouped[r["eta_uid"]].append(r)
    audit = []
    for eta, rr in grouped.items():
        first = rr[0]
        z = {"eta_uid": eta, "eta1": first["eta1"], "eta2": first["eta2"], "eta3": first["eta3"],
             "unique_states": len({r["state_uid"] for r in rr}),
             "source_groups": len({r["source_group"] for r in rr})}
        for sp in ("train", "val", "test"):
            qrows = [r for r in rr if r["state_split"] == sp]
            q = np.asarray([r["empirical_q"] for r in qrows], float)
            z[f"{sp}_states"] = len({r["state_uid"] for r in qrows})
            z[f"{sp}_source_groups"] = len({r["source_group"] for r in qrows})
            # Outcomes are recorded for reporting, but panel selection below never reads val/test outcomes.
            z[f"{sp}_q_mean"] = safe_float(q.mean()) if len(q) else None
            z[f"{sp}_q_std"] = safe_float(q.std()) if len(q) else None
            z[f"{sp}_q_min"] = safe_float(q.min()) if len(q) else None
            z[f"{sp}_q_max"] = safe_float(q.max()) if len(q) else None
            z[f"{sp}_b15_prevalence"] = safe_float((q >= 15/16).mean()) if len(q) else None
        tq = np.asarray([r["empirical_q"] for r in rr if r["state_split"] == "train"], float)
        z["train_clear_failure_count"] = int((tq <= .5).sum())
        z["train_robust_count"] = int((tq >= 15/16).sum())
        z["coverage_eligible"] = bool(z["unique_states"] >= 10 and z["train_states"] >= 6 and z["val_states"] >= 2 and z["test_states"] >= 2)
        z["state_dependent_b15"] = bool(z["coverage_eligible"] and z["train_clear_failure_count"] >= 2 and z["train_robust_count"] >= 2 and z["train_q_std"] >= .15 and .1 <= z["train_b15_prevalence"] <= .9)
        z["state_dependent_q_only"] = bool(z["coverage_eligible"] and z["train_q_std"] >= .15 and z["train_q_max"]-z["train_q_min"] >= .4)
        audit.append(z)
    return sorted(audit, key=lambda z:(-z["unique_states"], z["eta_uid"]))


def select_panel(audit, scenario):
    # This function deliberately reads TRAIN outcomes and split coverage only.
    eligible = [z for z in audit if z["coverage_eligible"]]
    if scenario == "Toy":
        primary = [z for z in eligible if z["state_dependent_b15"]]
        primary.sort(key=lambda z:(-z["train_states"], -z["train_q_std"], z["eta_uid"]))
        controls_hi = [z for z in eligible if z["train_b15_prevalence"] >= .9 and not z["state_dependent_b15"]]
        controls_lo = [z for z in eligible if z["train_b15_prevalence"] <= .1 and not z["state_dependent_b15"]]
        controls_hi.sort(key=lambda z:(-z["train_states"], z["eta_uid"]))
        controls_lo.sort(key=lambda z:(-z["train_states"], z["eta_uid"]))
        selected = primary + controls_hi[:2] + controls_lo[:2]
    else:
        # DB has no failure-to-B15 transition. Retain the two Q-varying probes plus all dense controls
        # to quantify regression signal without misrepresenting this as robust identifiability.
        selected = sorted(eligible, key=lambda z:(not z["state_dependent_q_only"], -z["train_q_std"], z["eta_uid"]))
    out = []
    for i,z in enumerate(selected):
        if z["state_dependent_b15"]:
            typ = "A_state_dependent_intermediate"
        elif z["state_dependent_q_only"]:
            typ = "A_q_varying_no_failure_to_robust_transition"
        elif z["train_b15_prevalence"] >= .9:
            typ = "B_mostly_robust_control"
        elif z["train_b15_prevalence"] <= .1:
            typ = "C_mostly_failure_control"
        else:
            typ = "control_intermediate_without_extremes"
        out.append({**z, "head_index": i, "probe_type": typ,
                    "selection_used": "TRAIN outcomes plus VAL/TEST coverage metadata only"})
    return out


def annotate_old_codebook(panel, scenario):
    if scenario == "Toy":
        anchors=[]
        with open(SOURCES["Toy"]["root"]/"codebook_eta.csv") as f:
            for r in csv.DictReader(f):anchors.append((int(r["mode_id"]),np.asarray([float(r["eta1"]),float(r["eta2"]),float(r["eta3"])])))
    else:
        d=json.load(open(SOURCES["DB"]["root"]/"selected_transform.json"));anchors=[(i,np.asarray(x,float)) for i,x in enumerate(d["transform"]["eta"])]
    for z in panel:
        x=np.asarray([z["eta1"],z["eta2"],z["eta3"]]);matches=[m for m,a in anchors if np.max(abs(x-a))<=1e-12]
        z["old_codebook_mode"]=matches[0] if matches else None
    return panel


def make_matrices(rows, panel, scenario, h_store):
    etas = [z["eta_uid"] for z in panel]; emap = {e:i for i,e in enumerate(etas)}
    by_split = {}
    for sp in ("train", "val", "test"):
        rr = [r for r in rows if r["scenario"] == scenario and r["state_split"] == sp and r["eta_uid"] in emap]
        states = sorted({r["state_uid"] for r in rr})
        state_to_feature = {r["state_uid"]: int(r["feature_index"]) for r in rr}
        smap = {s:i for i,s in enumerate(states)}
        y = np.zeros((len(states),len(etas)),np.float32); w = np.zeros_like(y); trials=np.zeros_like(y); mask=np.zeros_like(y,bool)
        rec = {}
        for r in rr:
            i,j=smap[r["state_uid"]],emap[r["eta_uid"]]
            assert not mask[i,j]
            y[i,j]=r["empirical_q"]; w[i,j]=min(r["n_trials"],16); trials[i,j]=r["n_trials"];mask[i,j]=True;rec[(i,j)]=r
        h=np.asarray([h_store[state_to_feature[s]] for s in states],np.float32)
        by_split[sp]={"states":states,"h":h,"y":y,"w":w,"trials":trials,"mask":mask,"records":rec}
    # Authoritative TRAIN-only feature normalization.
    mu=by_split["train"]["h"].mean(0); sd=by_split["train"]["h"].std(0); sd=np.maximum(sd,1e-6)
    for sp in by_split: by_split[sp]["h"]=(by_split[sp]["h"]-mu)/sd
    return by_split,{"mean":mu.tolist(),"std":sd.tolist()}


def matrix_nll(logits, d):
    p=sigmoid(logits); m=d["mask"]; return nll_prob(p[m],d["y"][m],d["w"][m])


def train_model(data, outputs, seed, shuffled=False):
    model=MultiHead(outputs)
    params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,data["train"]["h"].shape[1]),jnp.float32))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4)); state=opt.init(params)
    tr=data["train"]
    h=tr["h"].copy(); y=tr["y"].copy(); w=tr["w"].copy(); mask=tr["mask"].copy()
    if shuffled:
        rng=np.random.default_rng(seed+9000)
        # Per-head shuffling keeps each eta's outcomes/coverage exactly fixed while destroying h pairing.
        ys=np.zeros_like(y); ws=np.zeros_like(w); ms=np.zeros_like(mask)
        for j in range(outputs):
            src=np.where(mask[:,j])[0]; dst=rng.permutation(len(h))[:len(src)]
            ys[dst,j]=y[src,j]; ws[dst,j]=w[src,j]; ms[dst,j]=True
        y,w,mask=ys,ws,ms
    @jax.jit
    def step(p,st,h,y,w,m):
        def loss(q):
            logits=model.apply(q,h)
            e=optax.sigmoid_binary_cross_entropy(logits,y)*w*m
            return jnp.sum(e)/jnp.maximum(jnp.sum(w*m),1)
        val,g=jax.value_and_grad(loss)(p);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,val
    best=None;best_score=1e99;best_step=0;stale=0
    for it in range(1,4001):
        params,state,_=step(params,state,jnp.asarray(h),jnp.asarray(y),jnp.asarray(w),jnp.asarray(mask,dtype=jnp.float32))
        if it%25==0:
            logits=np.asarray(model.apply(params,jnp.asarray(data["val"]["h"])))
            score=matrix_nll(logits,data["val"])
            if score < best_score-1e-6:
                best=jax.tree_util.tree_map(np.asarray,params);best_score=score;best_step=it;stale=0
            else: stale+=1
            if stale>=30:break
    return model,best,{"seed":seed,"shuffled":shuffled,"best_step":best_step,"steps":it,"val_nll":best_score}


def constants(data):
    d=data["train"]; out=[]
    for j in range(d["y"].shape[1]):
        m=d["mask"][:,j]; out.append(float(np.sum(d["w"][m,j]*d["y"][m,j])/np.sum(d["w"][m,j])))
    return np.asarray(out)


def knn_predict(data, split, k):
    tr,te=data["train"],data[split]; pred=np.full_like(te["y"],np.nan,float)
    for j in range(tr["y"].shape[1]):
        ti=np.where(tr["mask"][:,j])[0]; qi=np.where(te["mask"][:,j])[0]
        if not len(ti):continue
        kk=min(k,len(ti)); a=te["h"][qi];b=tr["h"][ti]
        dist=((a[:,None,:]-b[None,:,:])**2).sum(2)
        near=np.argpartition(dist,kk-1,axis=1)[:,:kk]
        pred[qi,j]=tr["y"][ti[near],j].mean(1)
    return pred


def per_eta_metrics(panel,data,predictions):
    rows=[];tr,te=data["train"],data["test"]
    for j,z in enumerate(panel):
        mt=tr["mask"][:,j];me=te["mask"][:,j];yt=tr["y"][mt,j];y=te["y"][me,j];w=te["w"][me,j]
        base={"scenario":z.get("scenario"),"head_index":j,"eta_uid":z["eta_uid"],"eta1":z["eta1"],"eta2":z["eta2"],"eta3":z["eta3"],"probe_type":z["probe_type"],
              "train_states":int(mt.sum()),"test_states":int(me.sum()),"train_q_mean":safe_float(yt.mean()),"train_q_std":safe_float(yt.std()),
              "test_q_mean":safe_float(y.mean()),"test_q_std":safe_float(y.std()),"train_b15_prevalence":safe_float((yt>=15/16).mean()),"test_b15_prevalence":safe_float((y>=15/16).mean())}
        for method,pall in predictions.items():
            p=pall[me,j]
            rho=spearmanr(p,y).statistic if len(y)>1 and np.std(p)>0 and np.std(y)>0 else np.nan
            yy=y>=15/16
            base.update({f"{method}_mae":safe_float(np.mean(abs(p-y))),f"{method}_nll":nll_prob(p,y,w),f"{method}_q_spearman":safe_float(rho),
                         f"{method}_b15_auroc":auc(yy,p),f"{method}_b15_accuracy":safe_float(np.mean((p>=15/16)==yy))})
        base["mlp_delta_mae_vs_constant"]=base["constant_mae"]-base["mlp_mae"]
        base["mlp_delta_nll_vs_constant"]=base["constant_nll"]-base["mlp_nll"]
        base["knn3_delta_mae_vs_constant"]=base["constant_mae"]-base["knn3_mae"]
        base["knn3_delta_nll_vs_constant"]=base["constant_nll"]-base["knn3_nll"]
        rows.append(base)
    return rows


def aggregate_method_metrics(scenario,panel,data,predictions):
    rows=[];te=data["test"]
    groups={"ALL":range(len(panel)),"A_PRIMARY":[i for i,z in enumerate(panel) if z["probe_type"].startswith("A_")],
            "A_B15_TRANSITION":[i for i,z in enumerate(panel) if z["probe_type"]=="A_state_dependent_intermediate"],
            "CONTROLS":[i for i,z in enumerate(panel) if not z["probe_type"].startswith("A_")]}
    for group,js0 in groups.items():
        js=list(js0)
        if not js:continue
        m=te["mask"][:,js]; y=te["y"][:,js][m];w=te["w"][:,js][m]
        for method,pall in predictions.items():
            p=pall[:,js][m];rho=spearmanr(p,y).statistic if len(y)>1 and np.std(p)>0 and np.std(y)>0 else np.nan
            rows.append({"scenario":scenario,"group":group,"method":method,"probes":len(js),"pairs":len(y),"nll":nll_prob(p,y,w),"mae":float(np.mean(abs(p-y))),
                         "q_spearman":safe_float(rho),"b15_accuracy":float(np.mean((p>=15/16)==(y>=15/16))),"b15_auroc":auc(y>=15/16,p)})
    return rows


def pairwise_ordering(scenario,panel,data,predictions):
    rows=[];te=data["test"]
    for j,z in enumerate(panel):
        obs=np.where(te["mask"][:,j])[0];lo=obs[te["y"][obs,j]<=.5];hi=obs[te["y"][obs,j]>=15/16]
        for method,pall in predictions.items():
            vals=[]
            for a in lo:
                for b in hi:
                    d=pall[b,j]-pall[a,j];vals.append(1 if d>1e-12 else .5 if abs(d)<=1e-12 else 0)
            rows.append({"scenario":scenario,"eta_uid":z["eta_uid"],"probe_type":z["probe_type"],"method":method,"failure_states":len(lo),"robust_states":len(hi),
                         "state_pairs":len(vals),"ordering_accuracy":safe_float(np.mean(vals)) if vals else None})
    return rows


def candidate_ranking(scenario,panel,data,predictions):
    rows=[];te=data["test"]
    scopes={"ALL_PANEL":np.arange(len(panel)),
            "A_PRIMARY":np.asarray([j for j,z in enumerate(panel) if z["probe_type"].startswith("A_")],int),
            "OLD_CODEBOOK":np.asarray([j for j,z in enumerate(panel) if z.get("old_codebook_mode") is not None],int)}
    for scope,allowed in scopes.items():
      if not len(allowed):continue
      for method,pall in predictions.items():
        for i,s in enumerate(te["states"]):
            js=allowed[te["mask"][i,allowed]]
            if len(js)<2:continue
            score=pall[i,js];y=te["y"][i,js];pick=js[np.argmax(score)];oracle=js[np.argmax(y)]
            order=js[np.argsort(-score)];has=bool(np.any(y>=15/16));
            rows.append({"scenario":scenario,"scope":scope,"method":method,"state_uid":s,"candidate_count":len(js),"selected_eta_uid":panel[pick]["eta_uid"],"oracle_eta_uid":panel[oracle]["eta_uid"],
                         "selected_true_q":float(te["y"][i,pick]),"oracle_true_q":float(te["y"][i,oracle]),"regret":float(te["y"][i,oracle]-te["y"][i,pick]),
                         "panel_has_b15":has,"selected_b15":bool(te["y"][i,pick]>=15/16),"top3_b15_hit":bool(np.any(te["y"][i,order[:3]]>=15/16)) if has else None})
    return rows


def run_scenario(rows,fs,scenario,audit,panel):
    for z in panel:z["scenario"]=scenario
    key="toy" if scenario=="Toy" else "db"
    data,norm=make_matrices(rows,panel,scenario,fs[key])
    dump(f"normalization_{key}.json",norm)
    c=constants(data); const={sp:np.tile(c,(len(data[sp]["states"]),1)) for sp in data}
    knn={k:{sp:knn_predict(data,sp,k) for sp in ("val","test")} for k in (1,3,5)}
    train_rows=[];models=[];shufs=[]
    for seed in SEEDS:
        model,params,summary=train_model(data,len(panel),seed,False);models.append((model,params,summary));train_rows.append({"scenario":scenario,"model":"normal","outputs":len(panel),**summary})
        out=H/f"{key}_models"/f"seed{seed}";out.mkdir(parents=True,exist_ok=True);(out/"checkpoint.msgpack").write_bytes(serialization.to_bytes(params))
        model2,params2,summary2=train_model(data,len(panel),seed,True);shufs.append((model2,params2,summary2));train_rows.append({"scenario":scenario,"model":"state_shuffled","outputs":len(panel),**summary2})
        out=H/f"{key}_shuffled_models"/f"seed{seed}";out.mkdir(parents=True,exist_ok=True);(out/"checkpoint.msgpack").write_bytes(serialization.to_bytes(params2))
    best=min(models,key=lambda x:(x[2]["val_nll"],x[2]["seed"]));bsh=min(shufs,key=lambda x:(x[2]["val_nll"],x[2]["seed"]))
    pred={"constant":const["test"],"knn1":knn[1]["test"],"knn3":knn[3]["test"],"knn5":knn[5]["test"],
          "mlp":sigmoid(np.asarray(best[0].apply(best[1],jnp.asarray(data["test"]["h"])))),
          "shuffled":sigmoid(np.asarray(bsh[0].apply(bsh[1],jnp.asarray(data["test"]["h"]))))}
    metrics=per_eta_metrics(panel,data,pred)
    aggregate=aggregate_method_metrics(scenario,panel,data,pred)
    ordering=pairwise_ordering(scenario,panel,data,pred)
    ranking=candidate_ranking(scenario,panel,data,pred)
    return {"data":data,"metrics":metrics,"aggregate":aggregate,"ordering":ordering,"ranking":ranking,"training":train_rows,
            "selected":{"normal":best[2],"shuffled":bsh[2]},"pred":pred}


def summary_row(rows, scenario, group, method):
    return next((r for r in rows if r["scenario"]==scenario and r["group"]==group and r["method"]==method),None)


def main():
    started=time.time();H.mkdir(parents=True,exist_ok=True)
    rows,fs,excluded=load_live_database()
    source_pair=H/"canonical_pair_snapshot.parquet"
    pq.write_table(pa.Table.from_pylist(rows),source_pair,compression="zstd")
    # Freeze/check splits and coverage before panel construction.
    for sc,key in (("Toy","toy"),("DB","db")):
        split=source_split(rows,sc); assert all(v==0 for v in split["source_group_overlap"].values());dump(f"state_split_{key}.json",split)
    audits={sc:audit_etas(rows,sc) for sc in SCENARIOS}
    write_csv("coverage_audit_toy.csv",audits["Toy"]);write_csv("coverage_audit_db.csv",audits["DB"])
    panels={sc:annotate_old_codebook(select_panel(audits[sc],sc),sc) for sc in SCENARIOS}
    write_csv("fixed_eta_panel_toy.csv",panels["Toy"]);write_csv("fixed_eta_panel_db.csv",panels["DB"])
    results={sc:run_scenario(rows,fs,sc,audits[sc],panels[sc]) for sc in SCENARIOS}
    write_csv("per_eta_metrics_toy.csv",results["Toy"]["metrics"]);write_csv("per_eta_metrics_db.csv",results["DB"]["metrics"])
    aggregate=sum([results[s]["aggregate"] for s in SCENARIOS],[])
    write_csv("constant_baseline.csv",[r for r in aggregate if r["method"]=="constant"])
    write_csv("knn_baseline.csv",[r for r in aggregate if r["method"].startswith("knn")])
    write_csv("shuffled_control.csv",[r for r in aggregate if r["method"] in ("mlp","shuffled")])
    ordering=sum([results[s]["ordering"] for s in SCENARIOS],[]);write_csv("pairwise_state_ordering.csv",ordering)
    ranking=sum([results[s]["ranking"] for s in SCENARIOS],[]);write_csv("candidate_ranking.csv",ranking)
    training=sum([results[s]["training"] for s in SCENARIOS],[]);write_csv("training_summary.csv",training)
    dump("selected_models.json",{s:results[s]["selected"] for s in SCENARIOS})

    toyA={m:summary_row(aggregate,"Toy","A_PRIMARY",m) for m in ("constant","knn3","mlp","shuffled")}
    dbA={m:summary_row(aggregate,"DB","A_PRIMARY",m) for m in ("constant","knn3","mlp","shuffled")}
    def rank_summary(sc,method,scope="A_PRIMARY"):
        rr=[r for r in ranking if r["scenario"]==sc and r["method"]==method and r["scope"]==scope]
        cov=[r for r in rr if r["panel_has_b15"]]
        return {"states":len(rr),"mean_selected_q":safe_float(np.mean([r["selected_true_q"] for r in rr])),"mean_oracle_q":safe_float(np.mean([r["oracle_true_q"] for r in rr])),
                "mean_regret":safe_float(np.mean([r["regret"] for r in rr])),"b15_selection_rate_when_available":safe_float(np.mean([r["selected_b15"] for r in cov])) if cov else None,
                "top3_b15_hit_rate":safe_float(np.mean([r["top3_b15_hit"] for r in cov])) if cov else None}
    def order_summary(sc,method):
        rr=[r for r in ordering if r["scenario"]==sc and r["method"]==method and r["state_pairs"]]
        n=sum(r["state_pairs"] for r in rr)
        return {"pairs":n,"accuracy":sum(r["ordering_accuracy"]*r["state_pairs"] for r in rr)/n if n else None}

    sanity=[]
    for sc in SCENARIOS:
        for method in ("constant","knn3","mlp","shuffled"):
            z=rank_summary(sc,method,"OLD_CODEBOOK")
            sanity.append({"scenario":sc,"method":method,"exact_old_codebook_modes":sum(x.get("old_codebook_mode") is not None for x in panels[sc]),**z,
                           "interpretation":"Exact-anchor sanity only; universal/high-prevalence anchors excluded from primary A-probe conclusion."})
    write_csv("selector_sanity.csv",sanity)

    toy_dep=sum(z["state_dependent_b15"] for z in audits["Toy"]);db_dep=sum(z["state_dependent_b15"] for z in audits["DB"])
    toy_repeat=sum(z["unique_states"]>=10 for z in audits["Toy"]);db_repeat=sum(z["unique_states"]>=10 for z in audits["DB"])
    toy_signal=(toyA["mlp"] and toyA["constant"] and toyA["mlp"]["nll"]<toyA["constant"]["nll"] and toyA["mlp"]["mae"]<toyA["constant"]["mae"])
    toy_knn=(toyA["knn3"] and toyA["knn3"]["mae"]<toyA["constant"]["mae"])
    toy_shuffle=(toyA["mlp"] and toyA["shuffled"] and toyA["mlp"]["nll"]<toyA["shuffled"]["nll"])
    toy_order=order_summary("Toy","mlp")
    if toy_signal and toy_knn and toy_shuffle and (toy_order["accuracy"] or 0)>.7:
        toy_class="STATE_TO_Q_STRONGLY_LEARNABLE"
    elif toy_signal or toy_knn or (toy_order["accuracy"] or 0)>.6:
        toy_class="STATE_TO_Q_PARTIALLY_LEARNABLE"
    else: toy_class="STATE_TO_Q_NOT_SUPPORTED"
    # DB has zero exact repeated eta containing both clear failure and B15 states; robust identifiability is absent.
    db_class="DATA_NOT_IDENTIFIABLE" if db_dep==0 else ("STATE_TO_Q_PARTIALLY_LEARNABLE" if dbA["mlp"]["nll"]<dbA["constant"]["nll"] else "STATE_TO_Q_NOT_SUPPORTED")
    decision={"task":"ORTHOFLOW3_FIXED_ETA_STATE_TO_Q_IDENTIFIABILITY_V2","new_rollout":0,"database":str(DBPATH),"database_sha256":sha256(DBPATH),"source_pair_table":str(source_pair),"source_pair_table_sha256":sha256(source_pair),"excluded":excluded,
              "split_leakage":{"Toy":json.load(open(H/"state_split_toy.json"))["source_group_overlap"],"DB":json.load(open(H/"state_split_db.json"))["source_group_overlap"]},
              "coverage":{"Toy":{"exact_eta":len(audits["Toy"]),"repeated_ge10":toy_repeat,"coverage_eligible":sum(z["coverage_eligible"] for z in audits["Toy"]),"state_dependent_b15":toy_dep,"panel":len(panels["Toy"])},
                          "DB":{"exact_eta":len(audits["DB"]),"repeated_ge10":db_repeat,"coverage_eligible":sum(z["coverage_eligible"] for z in audits["DB"]),"state_dependent_b15":db_dep,"q_varying_without_b15_transition":sum(z["state_dependent_q_only"] for z in audits["DB"]),"panel":len(panels["DB"])}},
              "primary_metrics":{"Toy":toyA,"DB":dbA},"pairwise_ordering":{"Toy":{m:order_summary("Toy",m) for m in ("constant","knn3","mlp","shuffled")},"DB":{m:order_summary("DB",m) for m in ("constant","knn3","mlp","shuffled")}},
              "candidate_ranking":{"Toy":{m:rank_summary("Toy",m) for m in ("constant","knn3","mlp","shuffled")},"DB":{m:rank_summary("DB",m) for m in ("constant","knn3","mlp","shuffled")}},
              "classification":{"Toy":toy_class,"DB":db_class},"DB_fixed_eta_diagnostic":"DB_FIXED_ETA_NOT_IDENTIFIABLE_FROM_CURRENT_DATA" if db_dep==0 else "IDENTIFIABLE",
              "continuous_critic_failure_interpretation":"If Toy fixed-eta state prediction succeeds, prior continuous failure is primarily attributable to sparse state×eta cross-structure/unseen-eta compositional generalization. DB remains separately limited by absent robust-transition probes."}
    dump("final_decision.json",decision)
    dump("working_state.json",{"status":"complete","completed_stages":["canonical_db_reuse","source_group_split_audit","fixed_probe_audit","panel_freeze","baselines","normal_and_shuffled_training","per_eta_metrics","ranking","final_decision"],"new_rollout":0,"next_action":"none"})
    write_csv("experiment_ledger.csv",[{"stage":"complete","source":"global rollout DB via canonical critic pair table","new_rollouts":0,"reused_pairs":len(rows),"timestamp_unix":time.time()}])
    dump("runtime_statistics.json",{"wall_seconds":time.time()-started,"new_rollout":0,"models_trained":12})
    print(json.dumps({"classification":decision["classification"],"coverage":decision["coverage"],"Toy_A":toyA,"DB_A":dbA,"ranking":decision["candidate_ranking"],"ordering":decision["pairwise_ordering"]},indent=2))


if __name__ == "__main__":
    main()
