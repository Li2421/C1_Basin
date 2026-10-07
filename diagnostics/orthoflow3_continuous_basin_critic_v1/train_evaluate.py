#!/usr/bin/env python3
"""Train continuous Basin critics and evaluate prediction/ranking without rollouts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
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
from scipy.stats import rankdata, spearmanr

H = Path(__file__).parent
SEEDS = (17, 23, 41)
REGIMES = ("A_seen_state_seen_eta", "B_unseen_state_seen_eta", "C_seen_state_unseen_eta", "D_unseen_state_unseen_eta")


def dump(name, obj):
    (H / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(name, data):
    path = H / name; path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(data[0]) if data else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(data)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class JointCritic(nn.Module):
    def setup(self):
        self.toy1 = nn.Dense(128); self.toy2 = nn.Dense(64)
        self.db1 = nn.Dense(128); self.db2 = nn.Dense(64)
        self.eta1 = nn.Dense(32)
        self.c1 = nn.Dense(128); self.c2 = nn.Dense(64); self.out = nn.Dense(1)
    def body(self, z, eta):
        e = nn.silu(self.eta1(eta)); x = jnp.concatenate([z, e], axis=-1)
        return self.out(nn.silu(self.c2(nn.silu(self.c1(x)))))[..., 0]
    def toy(self, h, eta): return self.body(nn.silu(self.toy2(nn.silu(self.toy1(h)))), eta)
    def db(self, h, eta): return self.body(nn.silu(self.db2(nn.silu(self.db1(h)))), eta)
    def __call__(self, ht, et, hd, ed): return self.toy(ht, et), self.db(hd, ed)


class SingleCritic(nn.Module):
    @nn.compact
    def __call__(self, h, eta):
        z = nn.silu(nn.Dense(128)(h)); z = nn.silu(nn.Dense(64)(z))
        e = nn.silu(nn.Dense(32)(eta)); x = jnp.concatenate([z, e], axis=-1)
        x = nn.silu(nn.Dense(128)(x)); x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[..., 0]


class EtaCritic(nn.Module):
    @nn.compact
    def __call__(self, eta):
        x = nn.silu(nn.Dense(32)(eta)); x = nn.silu(nn.Dense(128)(x)); x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[..., 0]


def weighted_nll(logit, y, w):
    return jnp.sum(w * optax.sigmoid_binary_cross_entropy(logit, y)) / jnp.maximum(jnp.sum(w), 1)


def load_data():
    rows = pq.read_table(H / "pair_table.parquet").to_pylist()
    manifest = json.load(open(H / "dataset_manifest.json")); fs = np.load(H / "feature_store.npz")
    eta_c = np.asarray(manifest["eta_normalization"]["center"], np.float32)
    eta_s = np.asarray(manifest["eta_normalization"]["scale"], np.float32)
    arrays = {}
    for scenario, key in (("Toy", "toy"), ("DB", "db")):
        norm = manifest["state_normalization"][scenario]
        h = (fs[key].astype(np.float32) - np.asarray(norm["mean"], np.float32)) / np.asarray(norm["std"], np.float32)
        arrays[scenario] = {"h_store": h}
    for i, r in enumerate(rows): r["row_index"] = i
    return rows, arrays, eta_c, eta_s


def pack(rows, arrays, eta_c, eta_s, select):
    rr = [r for r in rows if select(r)]
    if not rr: return {"rows": [], "h": np.zeros((0, 1), np.float32), "eta": np.zeros((0, 3), np.float32), "y": np.zeros(0, np.float32), "w": np.zeros(0, np.float32)}
    sc = rr[0]["scenario"]
    return {"rows": rr,
            "h": np.asarray([arrays[sc]["h_store"][r["feature_index"]] for r in rr], np.float32),
            "eta": np.asarray([(np.asarray([r["eta1"], r["eta2"], r["eta3"]]) - eta_c) / eta_s for r in rr], np.float32),
            "y": np.asarray([r["empirical_q"] for r in rr], np.float32),
            "w": np.asarray([r["n_eff"] for r in rr], np.float32)}


def np_nll(logits, y, w):
    p = sigmoid(logits); e = 1e-7
    return float(np.sum(w * (-(y * np.log(p + e) + (1-y) * np.log(1-p + e)))) / np.sum(w)) if len(y) else None


def train_all(rows, arrays, eta_c, eta_s):
    train = {s: pack(rows, arrays, eta_c, eta_s, lambda r, s=s: r["scenario"] == s and r["sampled_train"]) for s in ("Toy", "DB")}
    val = {s: pack(rows, arrays, eta_c, eta_s, lambda r, s=s: r["scenario"] == s and r["sampled_val"]) for s in ("Toy", "DB")}
    configs = {"architecture": {"state_adapters": "input->128->64", "eta_encoder": "3->32", "critic": "96->128->64->1", "activation": "SiLU"},
               "optimizer": "AdamW lr=1e-3 weight_decay=1e-4 clip=5", "batch_size_per_scenario": 128,
               "max_steps": 4000, "eval_every": 50, "patience_evals": 20, "seeds": list(SEEDS),
               "loss": "binomial BCE weighted by min(n_trials,16); joint scenarios weighted 0.5/0.5", "new_rollouts": 0}
    dump("train_config.json", configs)
    summaries, saved = [], {}

    def save(model_name, seed, params, summary):
        out = H / model_name / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
        ck = out / "checkpoint.msgpack"; ck.write_bytes(serialization.to_bytes(params))
        summary.update({"model": model_name, "seed": seed, "checkpoint": str(ck), "checkpoint_sha256": sha(ck)})
        dump(str(Path(model_name) / f"seed{seed}" / "summary.json"), summary)
        summaries.append(summary); saved[(model_name, seed)] = params

    # Joint critic.
    for seed in SEEDS:
        model = JointCritic(); params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,214)), jnp.zeros((1,3)), jnp.zeros((1,80)), jnp.zeros((1,3)))
        opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(1e-3, weight_decay=1e-4)); state = opt.init(params)
        @jax.jit
        def step(p, st, ht, et, yt, wt, hd, ed, yd, wd):
            def loss(q):
                lt, ld = model.apply(q, ht, et, hd, ed)
                return .5 * weighted_nll(lt, yt, wt) + .5 * weighted_nll(ld, yd, wd)
            v, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, v
        rt, rd = np.random.default_rng(seed), np.random.default_rng(seed + 1000)
        best, best_score, best_step, stale = None, None, 0, 0
        for stp in range(1, 4001):
            it = rt.integers(0, len(train["Toy"]["y"]), 128); idb = rd.integers(0, len(train["DB"]["y"]), 128)
            params, state, _ = step(params, state, *[jnp.asarray(train["Toy"][k][it]) for k in ("h","eta","y","w")], *[jnp.asarray(train["DB"][k][idb]) for k in ("h","eta","y","w")])
            if stp % 50 == 0:
                lt, ld = model.apply(params, jnp.asarray(val["Toy"]["h"]), jnp.asarray(val["Toy"]["eta"]), jnp.asarray(val["DB"]["h"]), jnp.asarray(val["DB"]["eta"]))
                nt, nd = np_nll(np.asarray(lt), val["Toy"]["y"], val["Toy"]["w"]), np_nll(np.asarray(ld), val["DB"]["y"], val["DB"]["w"])
                score = .5 * nt + .5 * nd
                if best_score is None or score < best_score - 1e-6:
                    best = jax.tree_util.tree_map(np.asarray, params); best_score, best_step, stale = score, stp, 0
                else: stale += 1
                if stale >= 20: break
        save("joint_models", seed, best, {"best_step": best_step, "steps": stp, "val_nll": best_score})

    # Scenario-only critics.
    for scenario, dirname in (("Toy", "toy_only_models"), ("DB", "db_only_models")):
        for seed in SEEDS:
            model = SingleCritic(); dim = train[scenario]["h"].shape[1]
            params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,dim)), jnp.zeros((1,3)))
            opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(1e-3, weight_decay=1e-4)); state = opt.init(params)
            @jax.jit
            def step(p, st, h, e, y, w):
                loss = lambda q: weighted_nll(model.apply(q, h, e), y, w)
                v, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, v
            rng = np.random.default_rng(seed); best, best_score, best_step, stale = None, None, 0, 0
            for stp in range(1, 4001):
                ix = rng.integers(0, len(train[scenario]["y"]), 128)
                params, state, _ = step(params, state, *[jnp.asarray(train[scenario][k][ix]) for k in ("h","eta","y","w")])
                if stp % 50 == 0:
                    logits = np.asarray(model.apply(params, jnp.asarray(val[scenario]["h"]), jnp.asarray(val[scenario]["eta"])))
                    score = np_nll(logits, val[scenario]["y"], val[scenario]["w"])
                    if best_score is None or score < best_score - 1e-6:
                        best = jax.tree_util.tree_map(np.asarray, params); best_score, best_step, stale = score, stp, 0
                    else: stale += 1
                    if stale >= 20: break
            save(dirname, seed, best, {"best_step": best_step, "steps": stp, "val_nll": best_score})

    # Eta-only joint baseline, equally weighted by scenario.
    for seed in SEEDS:
        model = EtaCritic(); params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,3)))
        opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(1e-3, weight_decay=1e-4)); state = opt.init(params)
        @jax.jit
        def step(p, st, et, yt, wt, ed, yd, wd):
            loss = lambda q: .5 * weighted_nll(model.apply(q, et), yt, wt) + .5 * weighted_nll(model.apply(q, ed), yd, wd)
            v, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, v
        rt, rd = np.random.default_rng(seed), np.random.default_rng(seed + 1000); best, best_score, best_step, stale = None, None, 0, 0
        for stp in range(1, 4001):
            it = rt.integers(0, len(train["Toy"]["y"]), 128); idb = rd.integers(0, len(train["DB"]["y"]), 128)
            params, state, _ = step(params, state, jnp.asarray(train["Toy"]["eta"][it]), jnp.asarray(train["Toy"]["y"][it]), jnp.asarray(train["Toy"]["w"][it]), jnp.asarray(train["DB"]["eta"][idb]), jnp.asarray(train["DB"]["y"][idb]), jnp.asarray(train["DB"]["w"][idb]))
            if stp % 50 == 0:
                lt = np.asarray(model.apply(params, jnp.asarray(val["Toy"]["eta"]))); ld = np.asarray(model.apply(params, jnp.asarray(val["DB"]["eta"])))
                score = .5*np_nll(lt,val["Toy"]["y"],val["Toy"]["w"])+.5*np_nll(ld,val["DB"]["y"],val["DB"]["w"])
                if best_score is None or score < best_score - 1e-6:
                    best = jax.tree_util.tree_map(np.asarray, params); best_score, best_step, stale = score, stp, 0
                else: stale += 1
                if stale >= 20: break
        save("eta_only_models", seed, best, {"best_step": best_step, "steps": stp, "val_nll": best_score})
    write_csv("training_summary.csv", summaries)
    selected = {}
    for dirname in ("joint_models", "toy_only_models", "db_only_models", "eta_only_models"):
        x = [r for r in summaries if r["model"] == dirname]
        selected[dirname] = min(x, key=lambda r: (r["val_nll"], r["seed"]))
    dump("selected_models.json", selected)
    return saved, selected


def auc_ap(y, score):
    y = np.asarray(y, bool); score = np.asarray(score)
    if y.sum() == 0 or (~y).sum() == 0: return None, None
    ranks = rankdata(score); auc = (ranks[y].sum() - y.sum()*(y.sum()+1)/2) / (y.sum()*(~y).sum())
    order = np.argsort(-score); ys = y[order]; prec = np.cumsum(ys)/(np.arange(len(ys))+1)
    ap = float(np.sum(prec*ys)/ys.sum())
    return float(auc), ap


def prediction_metrics(logits, data):
    p = sigmoid(logits); y = data["y"]; w = data["w"]
    rho = spearmanr(p, y).statistic if len(y) > 1 and np.std(p) > 0 and np.std(y) > 0 else None
    return {"pairs": len(y), "binomial_nll": np_nll(logits,y,w), "brier": float(np.mean((p-y)**2)),
            "q_mae": float(np.mean(np.abs(p-y))), "q_spearman": None if rho is None or np.isnan(rho) else float(rho),
            "mean_empirical_q": float(y.mean()) if len(y) else None, "mean_predicted_q": float(p.mean()) if len(y) else None}


def b15_metrics(logits, data):
    use = np.asarray([r["n_trials"] >= 16 for r in data["rows"]]); p = sigmoid(logits)[use]
    y = np.asarray([r["b15"] for r in data["rows"]], bool)[use]
    if not len(y): return {"pairs": 0}
    pred = p >= 15/16; tp=np.sum(pred&y); fp=np.sum(pred&~y); fn=np.sum(~pred&y)
    auc, ap = auc_ap(y,p)
    return {"pairs":len(y),"positive":int(y.sum()),"auroc":auc,"auprc":ap,"accuracy":float(np.mean(pred==y)),
            "precision":float(tp/(tp+fp)) if tp+fp else None,"recall":float(tp/(tp+fn)) if tp+fn else None}


def evaluate(rows, arrays, eta_c, eta_s, saved, selected):
    joint = JointCritic(); single = SingleCritic(); eta_model = EtaCritic()
    jp = saved[("joint_models", int(selected["joint_models"]["seed"]))]
    tp = saved[("toy_only_models", int(selected["toy_only_models"]["seed"]))]
    dp = saved[("db_only_models", int(selected["db_only_models"]["seed"]))]
    ep = saved[("eta_only_models", int(selected["eta_only_models"]["seed"]))]
    metrics_rows, b15_rows, calibration = [], [], []

    def logits_for(model_name, scenario, data):
        if model_name == "joint":
            method = joint.toy if scenario == "Toy" else joint.db
            return np.asarray(joint.apply(jp, jnp.asarray(data["h"]), jnp.asarray(data["eta"]), method=method))
        if model_name == "scenario_only":
            return np.asarray(single.apply(tp if scenario=="Toy" else dp, jnp.asarray(data["h"]), jnp.asarray(data["eta"])))
        return np.asarray(eta_model.apply(ep, jnp.asarray(data["eta"])))

    for scenario in ("Toy", "DB"):
        for regime in REGIMES:
            data = pack(rows, arrays, eta_c, eta_s, lambda r,s=scenario,g=regime: r["scenario"]==s and r["regime"]==g and r["sampled_eval"])
            for model_name in ("joint","scenario_only","eta_only"):
                logits = logits_for(model_name, scenario, data); pm = prediction_metrics(logits,data); bm=b15_metrics(logits,data)
                metrics_rows.append({"model":model_name,"scenario":scenario,"regime":regime,**pm})
                b15_rows.append({"model":model_name,"scenario":scenario,"regime":regime,**bm})
                probs=sigmoid(logits); bins=np.minimum((probs*10).astype(int),9)
                for b in range(10):
                    ix=bins==b
                    if ix.any(): calibration.append({"model":model_name,"scenario":scenario,"regime":regime,"bin":b,
                                                      "pairs":int(ix.sum()),"mean_pred":float(probs[ix].mean()),"mean_q":float(data['y'][ix].mean())})
    write_csv("prediction_metrics.csv",metrics_rows);write_csv("b15_metrics.csv",b15_rows);write_csv("calibration.csv",calibration)

    # Finite candidate ranking on held-out TEST states; selection of candidates is outcome-blind farthest coverage.
    ranking=[]
    for scenario in ("Toy","DB"):
        test_rows=[r for r in rows if r["scenario"]==scenario and r["state_split"]=="test" and r["n_trials"]>=16]
        by_state=defaultdict(list)
        for r in test_rows: by_state[r["state_uid"]].append(r)
        for K in (8,16,32):
            state_results=[]
            for state_uid, rr in by_state.items():
                rr = sorted(rr, key=lambda r: r["eta_uid"])
                if len(rr)<K:continue
                z=np.asarray([[(r['eta1']-eta_c[0])/eta_s[0],(r['eta2']-eta_c[1])/eta_s[1],(r['eta3']-eta_c[2])/eta_s[2]] for r in rr])
                chosen=[0];dist=np.linalg.norm(z-z[0],axis=1)
                while len(chosen)<K:
                    j=int(np.argmax(dist));chosen.append(j);dist=np.minimum(dist,np.linalg.norm(z-z[j],axis=1));dist[chosen]=-1
                cand=[rr[j] for j in chosen]; data=pack(cand,arrays,eta_c,eta_s,lambda r:True); pred=logits_for('joint',scenario,data)
                order=np.argsort(-pred); true=np.asarray(data['y']); best=int(np.argmax(true)); selected_i=int(order[0]); oracle=float(true[best]); sel=float(true[selected_i])
                any_b=bool(np.any(true>=15/16)); selected_b=bool(true[selected_i]>=15/16)
                top3_hit=bool(np.max(true[order[:3]]) >= oracle - 1e-12)
                state_results.append({"state_uid":state_uid,"selected_q":sel,"oracle_q":oracle,"regret":oracle-sel,
                                      "any_b15":any_b,"selected_b15":selected_b,"top3_hit":top3_hit,
                                      "unseen_eta_fraction":float(np.mean([r['eta_split']=='test' for r in cand]))})
            if state_results:
                cover=[r for r in state_results if r['any_b15']]
                ranking.append({"scenario":scenario,"K":K,"eligible_states":len(state_results),"mean_selected_q":float(np.mean([r['selected_q'] for r in state_results])),
                                "mean_oracle_q":float(np.mean([r['oracle_q'] for r in state_results])),"mean_regret":float(np.mean([r['regret'] for r in state_results])),
                                "median_regret":float(np.median([r['regret'] for r in state_results])),"b15_coverable_states":len(cover),
                                "b15_selection_rate":float(np.mean([r['selected_b15'] for r in cover])) if cover else None,
                                "top3_oracle_hit_rate":float(np.mean([r['top3_hit'] for r in state_results])),
                                "mean_unseen_eta_fraction":float(np.mean([r['unseen_eta_fraction'] for r in state_results])),
                                "coverage_status":"OK" if len(state_results)>=8 else "DATA_COVERAGE_INSUFFICIENT"})
            else:
                ranking.append({"scenario":scenario,"K":K,"eligible_states":0,"coverage_status":"DATA_COVERAGE_INSUFFICIENT"})
    write_csv("ranking_metrics.csv",ranking)

    # Old 12-anchor selector sanity.
    toy_cb=list(csv.DictReader(open(H.parent/'orthoflow3_shared_eta_codebook_v1'/'codebook_eta.csv')))
    anchors={"Toy":np.asarray([[float(r[f'eta{i}']) for i in (1,2,3)] for r in sorted(toy_cb,key=lambda r:int(r['mode_id']))]),
             "DB":np.asarray(json.load(open(H.parent/'orthoflow3_db_shared_mode_transfer_v1'/'selected_transform.json'))['transform']['eta'])}
    legacy={"Toy":{r['state_id']:int(r['selected_mode']) for r in csv.DictReader(open(H.parent/'orthoflow3_shared_eta_codebook_v1'/'test_selector.csv'))},
            "DB":{r['state_id']:int(r['selected_mode']) for r in csv.DictReader(open(H.parent/'orthoflow3_db_shared_mode_transfer_v1'/'test_results.csv'))}}
    sanity=[]
    for scenario in ("Toy","DB"):
        rr=[r for r in rows if r['scenario']==scenario and r['state_split']=='test' and r['n_trials']>=16]
        by_state=defaultdict(list)
        for r in rr:by_state[r['state_id']].append(r)
        vals=[]
        for sid, sr in by_state.items():
            mode_rows=[]
            for a in anchors[scenario]:
                match=[r for r in sr if np.max(np.abs(np.asarray([r['eta1'],r['eta2'],r['eta3']])-a))<=1e-12]
                if not match:break
                mode_rows.append(match[0])
            if len(mode_rows)!=12 or sid not in legacy[scenario]:continue
            data=pack(mode_rows,arrays,eta_c,eta_s,lambda r:True); pred=logits_for('joint',scenario,data); top=int(np.argmax(pred)); old=legacy[scenario][sid]
            q=np.asarray(data['y']); rho=spearmanr(pred,q).statistic if np.std(q)>0 and np.std(pred)>0 else None
            vals.append({"state_id":sid,"critic_mode":top,"legacy_mode":old,"agreement":top==old,"critic_q":float(q[top]),"legacy_q":float(q[old]),
                         "oracle_q":float(q.max()),"critic_b15":q[top]>=15/16,"legacy_b15":q[old]>=15/16,"rank_spearman":rho})
        sanity.append({"scenario":scenario,"states":len(vals),"mode_agreement":float(np.mean([v['agreement'] for v in vals])) if vals else None,
                       "critic_mean_q":float(np.mean([v['critic_q'] for v in vals])) if vals else None,"legacy_mean_q":float(np.mean([v['legacy_q'] for v in vals])) if vals else None,
                       "oracle_mean_q":float(np.mean([v['oracle_q'] for v in vals])) if vals else None,"critic_b15_rate":float(np.mean([v['critic_b15'] for v in vals])) if vals else None,
                       "legacy_b15_rate":float(np.mean([v['legacy_b15'] for v in vals])) if vals else None,
                       "mean_rank_spearman":float(np.nanmean([v['rank_spearman'] for v in vals if v['rank_spearman'] is not None])) if any(v['rank_spearman'] is not None for v in vals) else None})
    write_csv("selector_sanity.csv",sanity)
    return metrics_rows,b15_rows,ranking,sanity


def main():
    rows,arrays,eta_c,eta_s=load_data();saved,selected=train_all(rows,arrays,eta_c,eta_s)
    metrics,b15,ranking,sanity=evaluate(rows,arrays,eta_c,eta_s,saved,selected)
    dump("working_state.json",{"status":"EVALUATION_COMPLETE","completed":["dataset","joint_training","scenario_baselines","eta_only_baseline","four_regime_evaluation","candidate_ranking","selector_sanity"],"next_action":"scientific decision","new_rollouts":0})
    print(json.dumps({"selected":selected,"D_metrics":[r for r in metrics if r['regime'].startswith('D')],"ranking":ranking,"selector_sanity":sanity},indent=2))


if __name__=="__main__":main()
