#!/usr/bin/env python3
"""Strict falsification tests for Toy->Double-Bottleneck mode transfer."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

H = Path(__file__).parent
D = H.parent
TOY = D / "orthoflow3_shared_eta_codebook_v1"
DB = D / "orthoflow3_db_shared_mode_transfer_v1"
JOINT = D / "orthoflow3_toy_db_joint_selector_v1"
AFF = np.array([.875, 0., .375], float)
SCALE = np.array([.75, 1., .75], float)
M = 12
FOLDS = [[0,3,6,9], [1,4,7,10], [2,5,8,11]]
FAMILIES = ["raw_toy", "translation", "diagonal", "rotation_anisotropic", "regularized_affine"]

def dump(name, obj):
    (H/name).parent.mkdir(parents=True, exist_ok=True)
    (H/name).write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")
def read_csv(path): return list(csv.DictReader(open(path)))
def write_csv(name, rows, fields=None):
    p=H/name; p.parent.mkdir(parents=True, exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ["empty"])
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def horder(x, tag): return hashlib.sha256((tag+"|"+x).encode()).hexdigest()

def kabsch(X,Y):
    x=X-X.mean(0); y=Y-Y.mean(0); U,_,Vt=np.linalg.svd(x.T@y); R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1]*=-1; R=Vt.T@U.T
    s=np.sum((x@R.T)*y)/max(np.sum(x*x),1e-12); A=s*R
    return A, Y.mean(0)-X.mean(0)@A.T

def rot_aniso(X,Y):
    A0,b0=kabsch(X,Y); s0=max(np.linalg.svd(A0,compute_uv=False).mean(),1e-4)
    r0=Rotation.from_matrix(A0/s0).as_rotvec()
    def unpack(q):
        R=Rotation.from_rotvec(q[:3]).as_matrix(); s=np.exp(q[3:6])
        return np.diag(s)@R, q[6:9]
    def fun(q):
        A,b=unpack(q); return (X@A.T+b-Y).ravel()
    q0=np.r_[r0,np.log(np.repeat(s0,3)),b0]
    lo=np.r_[[-math.pi]*3,[-math.log(10)]*3,[-3]*3]
    hi=np.r_[[math.pi]*3,[math.log(10)]*3,[3]*3]
    q=least_squares(fun,q0,bounds=(lo,hi),max_nfev=50000,xtol=1e-12,ftol=1e-12,gtol=1e-12).x
    return unpack(q)

def fit_family(name,X,Y):
    if name=="translation":
        A=np.eye(3); b=(Y-X).mean(0)
    elif name=="diagonal":
        av=[]; bv=[]
        for j in range(3):
            Z=np.c_[X[:,j],np.ones(len(X))]
            a,bj=np.linalg.lstsq(Z,Y[:,j],rcond=None)[0]; av.append(a); bv.append(bj)
        A=np.diag(av); b=np.array(bv)
    elif name=="rotation_anisotropic": A,b=rot_aniso(X,Y)
    elif name=="regularized_affine":
        Aref,bref=rot_aniso(X,Y); xc=X-X.mean(0); yc=Y-Y.mean(0); lam=.25
        A=np.linalg.solve(xc.T@xc+lam*np.eye(3),xc.T@yc+lam*Aref.T).T
        U,s,Vt=np.linalg.svd(A); s=np.maximum(s,s.max()/10.); A=U@np.diag(s)@Vt
        b=Y.mean(0)-X.mean(0)@A.T
    else: raise ValueError(name)
    return A,b

def prepare():
    H.mkdir(parents=True,exist_ok=True); (H/"plans"/"stage_a_q64").mkdir(parents=True,exist_ok=True); (H/"raw").mkdir(exist_ok=True)
    cb=read_csv(TOY/"codebook_eta.csv")
    E=np.array([[float(r[f"eta{i}"]) for i in (1,2,3)] for r in sorted(cb,key=lambda q:int(q["mode_id"]))])
    X=(E-AFF)/SCALE
    cor=read_csv(DB/"mode_correspondence.csv")
    Y=np.array([[float(r[f"db_cluster_z{i}"]) for i in (1,2,3)] for r in sorted(cor,key=lambda q:int(q["mode_id"]))])
    fold_meta=[]; trs=[]; preds=[]
    # The three mode folds are fixed arithmetically and use no outcome statistic.
    for fi,held in enumerate(FOLDS):
        fit=[m for m in range(M) if m not in held]
        fold_meta.append({"fold":fi,"fit_modes":fit,"heldout_modes":held})
        for fam in FAMILIES:
            if fam=="raw_toy": A=np.eye(3); b=np.zeros(3)
            else: A,b=fit_family(fam,X[fit],Y[fit])
            # Same global E_bridge feasibility correction as the original DB transform;
            # it uses only canonical Toy inputs, never held-out DB coordinates/outcomes.
            if fam!="raw_toy":
                all_eta=(X@A.T+b)*SCALE+AFF
                if all_eta[:,2].min()<0: b[2]+=(-all_eta[:,2].min()+1e-6)/SCALE[2]
            fitres=np.linalg.norm(X[fit]@A.T+b-Y[fit],axis=1)
            cond=float(np.linalg.cond(A))
            trs.append({"fold":fi,"family":fam,"fit_modes":";".join(map(str,fit)),"heldout_modes":";".join(map(str,held)),
                        "fit_residual_mean":float(fitres.mean()),"fit_residual_median":float(np.median(fitres)),
                        "fit_residual_max":float(fitres.max()),"condition_number":cond,
                        "A_json":json.dumps(A.tolist(),separators=(",",":")),"b_json":json.dumps(b.tolist(),separators=(",",":"))})
            P=X@A.T+b
            for m in held:
                eta=P[m]*SCALE+AFF
                preds.append({"fold":fi,"family":fam,"mode_id":m,"pred_z1":P[m,0],"pred_z2":P[m,1],"pred_z3":P[m,2],
                              "eta1":eta[0],"eta2":eta[1],"eta3":eta[2],
                              "target_residual":float(np.linalg.norm(P[m]-Y[m])),"condition_number":cond})
    dump("mode_cv_folds.json",{"rule":"mode_id modulo 3; outcome-blind","folds":fold_meta})
    write_csv("leave_mode_out_transforms.csv",trs)

    # Independent robust support distance: transformed anchors that are B63 on at
    # least one independent DB TEST state (TEST outcomes never affect fitting).
    tq=read_csv(DB/"test_mode_q64.csv")
    robust_modes=sorted({int(r["mode_id"]) for r in tq if str(r.get("B63","")).lower()=="true"})
    selected=json.load(open(DB/"selected_transform.json"))["transform"]["eta"]
    Zrob=(np.asarray([selected[m] for m in robust_modes])-AFF)/SCALE
    for r in preds:
        z=np.array([r["pred_z1"],r["pred_z2"],r["pred_z3"]])
        r["nearest_independent_robust_distance"]=float(np.min(np.linalg.norm(Zrob-z,axis=1)))
    write_csv("heldout_mode_predictions.csv",preds)

    split=json.load(open(DB/"db_state_split.json"))
    val=[s for s in split["states"] if s["split"]=="val"]
    panel=sorted(val,key=lambda s:horder(s["source_group"],"transfer_falsification_panel_v1"))[:8]
    dump("stage_a_state_panel.json",{"selection":"first 8 DB VAL states by SHA256(source_group), outcomes unused", "states":panel})
    tasks=[]
    for r in preds:
        for s in panel:
            for k in range(64):
                tasks.append({"probe_id":f"A_f{r['fold']}_{r['family']}_m{r['mode_id']}_{s['state_id']}_{k:02d}",
                              "phase":"stage_a_heldout_q64","state_id":s["state_id"],"controller":f"A_{r['family']}",
                              "family":r["family"],"fold":r["fold"],"mode_id":r["mode_id"],
                              "eta":[r["eta1"],r["eta2"],r["eta3"]],"future_index":k})
    # Deduplicate exact state/eta/seed tasks (none expected across different modes except raw aliases).
    uniq={}
    for t in tasks:
        key=(t["state_id"],tuple(round(x,14) for x in t["eta"]),t["future_index"])
        uniq.setdefault(key,t)
    tasks=list(uniq.values())
    for j in range(6):
        p=H/"plans"/"stage_a_q64"/f"shard{j}.jsonl"
        p.write_text("".join(json.dumps(t,separators=(",",":"))+"\n" for i,t in enumerate(tasks) if i%6==j))
    assets=[]
    for p in [TOY/"codebook_eta.csv",DB/"mode_correspondence.csv",DB/"db_state_split.json",DB/"test_mode_q64.csv",DB/"selected_transform.json",JOINT/"db_train_subsets.json"]:
        assets.append({"path":str(p),"sha256":sha(p)})
    dump("frozen_assets.json",assets)
    dump("working_state.json",{"status":"STAGE_A_Q64_READY","completed":["asset_freeze","mode_folds","heldout_transform_fit","panel_freeze"],
                               "new_q64_eta_state_pairs":len(tasks)//64,"new_continuations_planned":len(tasks),"next_action":"run exact Stage-A panel"})
    dump("hypothesis_status.json",{"A":{"status":"LIVE","question":"leave-mode-out deformation predictive?"},
                                   "B":{"status":"READY_AFTER_A","question":"correct identity beats derangements?"}})
    write_csv("experiment_ledger.csv",[{"stage":"prepare","status":"COMPLETE","new_continuations":0,"decision":"3 fixed folds; 8 outcome-blind DB VAL panel"},
                                        {"stage":"stage_a_plan","status":"READY","new_continuations":len(tasks),"decision":"direct Q64 only; no local search"}])
    print(json.dumps({"folds":fold_meta,"predictions":len(preds),"panel_states":len(panel),"new_continuations":len(tasks)},indent=2))

def aggregate_a():
    preds=read_csv(H/"heldout_mode_predictions.csv")
    rows=[]
    for p in (H/"raw").glob("stage_a_q64_*.jsonl"):
        rows += [json.loads(x) for x in open(p) if x.strip()]
    # Unique scientific key; runner resumes exactly.
    d={(r["state_id"],r["family"],int(r["mode_id"]),int(r["fold"]),int(r["future_index"])):r for r in rows}
    expected=60*8*64
    if len(d)!=expected: raise RuntimeError(f"Stage A incomplete: {len(d)} / {expected}")
    qrows=[]
    for p in preds:
        fam=p["family"]; m=int(p["mode_id"]); fold=int(p["fold"])
        for sid in sorted({k[0] for k in d}):
            rr=[d[(sid,fam,m,fold,k)] for k in range(64)]
            suc=sum(bool(x["success"]) for x in rr); dead=sum(x["outcome"]=="deadlock" for x in rr); tout=sum(x["outcome"]=="timeout" for x in rr)
            coll=sum(bool(x.get("wall_collision")) or bool(x.get("agent_collision")) for x in rr)
            qrows.append({"fold":fold,"family":fam,"mode_id":m,"state_id":sid,"successes":suc,"trials":64,"Q64":suc/64,
                          "B63":suc>=63,"deadlock":dead,"timeout":tout,"collision":coll})
    write_csv("heldout_mode_q64.csv",qrows)
    summary=[]
    by={(int(p["fold"]),p["family"],int(p["mode_id"])):p for p in preds}
    for key,g in itertools.groupby(sorted(qrows,key=lambda r:(int(r["fold"]),r["family"],int(r["mode_id"]))),key=lambda r:(int(r["fold"]),r["family"],int(r["mode_id"]))):
        gg=list(g); p=by[key]
        summary.append({"fold":key[0],"family":key[1],"mode_id":key[2],"panel_B63_states":sum(str(r["B63"]).lower()=="true" for r in gg),
                        "panel_states":8,"B63_prevalence":sum(str(r["B63"]).lower()=="true" for r in gg)/8,
                        "mean_Q64":float(np.mean([float(r["Q64"]) for r in gg])),"target_residual":p["target_residual"],
                        "nearest_independent_robust_distance":p["nearest_independent_robust_distance"],"condition_number":p["condition_number"]})
    write_csv("heldout_mode_summary.csv",summary)
    dump("working_state.json",{"status":"STAGE_A_COMPLETE_B_READY","completed":["asset_freeze","mode_folds","heldout_transform_fit","stage_a_q64"],
                               "new_continuations":expected,"next_action":"train frozen Toy semantic model and DB adapter controls"})
    print(json.dumps({f:{"mean_mode_prevalence":float(np.mean([float(r["B63_prevalence"]) for r in summary if r["family"]==f])),
                         "mean_Q64":float(np.mean([float(r["mean_Q64"]) for r in summary if r["family"]==f]))} for f in FAMILIES},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","aggregate_a","train_b","finalize"]); args=ap.parse_args()
    if args.stage=="prepare": prepare()
    elif args.stage=="aggregate_a": aggregate_a()
    elif args.stage=="train_b": train_b()
    else: finalize()

# Stage-B functions are appended below to keep preparation executable without JAX import overhead.
def train_b():
    import jax, jax.numpy as jnp, optax
    from flax import linen as nn, serialization
    SEEDS=(17,23,41)
    class Semantic(nn.Module):
        @nn.compact
        def __call__(self,x):
            a=nn.silu(nn.Dense(64,name="adapter")(x)); z=nn.silu(nn.Dense(64,name="trunk")(a)); return nn.Dense(12,name="head")(z)
    class Adapted(nn.Module):
        frozen: dict
        @nn.compact
        def __call__(self,x):
            a=nn.silu(nn.Dense(64,name="db_adapter")(x))
            z=nn.silu(nn.Dense(64,name="trunk").apply({"params":self.frozen["trunk"]},a))
            return nn.Dense(12,name="head").apply({"params":self.frozen["head"]},z)
    def feats(root,split):
        z=np.load(root/"state_features.npz"); mask=z["splits"].astype(str)==split
        return z["state_ids"].astype(str)[mask],z["x"].astype(np.float32)[mask]
    def labs(root,split):
        fn={"train":"train_mode_counts.csv","val":"val_mode_counts.csv","test":"test_mode_q64.csv"}[split]
        ids,_=feats(root,split); idx={s:i for i,s in enumerate(ids)}; y=np.zeros((len(ids),12),np.float32)
        for r in read_csv(root/fn): y[idx[r["state_id"]],int(r["mode_id"])]=float(r["successes"])/float(r["trials"])
        return ids,y
    def metr(logits,y,perm=None):
        if perm is None: perm=np.arange(12)
        p=1/(1+np.exp(-np.clip(logits,-40,40))); yy=y[:,perm]
        nll=float(np.mean(-(yy*np.log(np.clip(p,1e-8,1))+(1-yy)*np.log(np.clip(1-p,1e-8,1)))))
        brier=float(np.mean((p-yy)**2)); top=np.argmax(p,axis=1); actual=np.asarray(perm)[top]
        q=y[np.arange(len(y)),actual]; oracle=y.max(1)
        return {"nll":nll,"brier":brier,"mean_selected_Q":float(q.mean()),"B63_states":int(np.sum(q>=63/64)),
                "B63_fraction":float(np.mean(q>=63/64)),"mean_regret":float(np.mean(oracle-q)),"mode_counts":json.dumps(dict(zip(*np.unique(actual,return_counts=True)))).replace("np.int64(","").replace(")","")}
    txid,tx=feats(TOY,"train"); _,ty=labs(TOY,"train"); _,tv=feats(TOY,"val"); _,tvy=labs(TOY,"val")
    toy_summ=[]; toy_params={}
    for seed in SEEDS:
        model=Semantic(); params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)))
        opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4)); os=opt.init(params); rng=np.random.default_rng(seed)
        @jax.jit
        def step(p,o,x,y):
            loss=lambda pp:jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp,x),y))
            v,g=jax.value_and_grad(loss)(p); u,o=opt.update(g,o,p); return optax.apply_updates(p,u),o,v
        best=None; bs=None; be=0; wait=0
        for ep in range(2400):
            for j in range(0,len(tx),32):
                ii=rng.permutation(len(tx))[j:j+32]; params,os,_=step(params,os,jnp.asarray(tx[ii]),jnp.asarray(ty[ii]))
            mv=metr(np.asarray(model.apply(params,jnp.asarray(tv))),tvy); sc=(mv["nll"],-mv["mean_selected_Q"],mv["brier"])
            if bs is None or sc<bs: best=jax.tree_util.tree_map(np.asarray,params); bs=sc; be=ep; wait=0
            else: wait+=1
            if wait>=180: break
        mv=metr(np.asarray(model.apply(best,jnp.asarray(tv))),tvy)
        out=H/"toy_semantic_model"/f"seed{seed}"; out.mkdir(parents=True,exist_ok=True); ck=out/"checkpoint.msgpack"; ck.write_bytes(serialization.to_bytes(best))
        toy_summ.append({"seed":seed,"best_epoch":be,"checkpoint":str(ck),"checkpoint_sha256":sha(ck),**mv}); toy_params[seed]=best
    write_csv("toy_semantic_model/training_summary.csv",toy_summ)
    sel=min(toy_summ,key=lambda r:(float(r["nll"]),-float(r["mean_selected_Q"]),float(r["brier"]),int(r["seed"])))
    frozen=toy_params[int(sel["seed"])]["params"]
    dump("toy_semantic_model/selected.json",sel)
    # Deterministic derangements frozen before DB evaluation.
    rngp=np.random.default_rng(20260930); perms=[]
    while len(perms)<10:
        p=rngp.permutation(12)
        if np.all(p!=np.arange(12)) and p.tolist() not in perms: perms.append(p.tolist())
    dump("permutations.json",{"seed":20260930,"constraint":"derangements, no fixed points","permutations":perms})
    dxid,dxall=feats(DB,"train"); _,dyall=labs(DB,"train"); _,dv=feats(DB,"val"); _,dvy=labs(DB,"val"); _,dt=feats(DB,"test"); _,dty=labs(DB,"test")
    subsets=json.load(open(JOINT/"db_train_subsets.json")); results=[]
    conditions=[("correct",list(range(12)))]+[(f"perm_{i:02d}",p) for i,p in enumerate(perms)]
    for frac in (25,50):
      keep=np.array([sid in set(subsets[str(frac)]) for sid in dxid]); dx=dxall[keep]; dy=dyall[keep]
      for seed in SEEDS:
        for cname,perm in conditions:
          model=Adapted(frozen=frozen); init=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,80))); params={"params":{"db_adapter":init["params"]["db_adapter"]}}
          # Apply frozen trunk/head explicitly; only db_adapter parameters enter optimizer.
          def apply(p,x):
            W=p["params"]["db_adapter"]; a=nn.silu(x@W["kernel"]+W["bias"])
            Wt=frozen["trunk"]; z=nn.silu(a@Wt["kernel"]+Wt["bias"]); Wh=frozen["head"]
            return z@Wh["kernel"]+Wh["bias"]
          opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4)); os=opt.init(params); r=np.random.default_rng(seed)
          pp=np.asarray(perm); target=dy[:,pp]
          @jax.jit
          def step(p,o,x,y):
            loss=lambda q:jnp.mean(optax.sigmoid_binary_cross_entropy(apply(q,x),y))
            v,g=jax.value_and_grad(loss)(p); u,o=opt.update(g,o,p); return optax.apply_updates(p,u),o,v
          # Fixed 1200 epochs for every correct/shuffled condition; no condition-specific early stopping.
          for ep in range(1200):
            order=r.permutation(len(dx))
            for j in range(0,len(dx),16):
              ii=order[j:j+16]; params,os,_=step(params,os,jnp.asarray(dx[ii]),jnp.asarray(target[ii]))
          lv=np.asarray(apply(params,jnp.asarray(dv))); lt=np.asarray(apply(params,jnp.asarray(dt)))
          mv=metr(lv,dvy,perm); mt=metr(lt,dty,perm)
          out=H/("db_correct_adaptation" if cname=="correct" else "db_permutation_controls")/f"db{frac}_{cname}_seed{seed}"; out.mkdir(parents=True,exist_ok=True)
          ck=out/"adapter_checkpoint.msgpack"; ck.write_bytes(serialization.to_bytes(params))
          results.append({"db_fraction":frac,"condition":cname,"permutation":json.dumps(perm,separators=(",",":")),"seed":seed,
                          "train_states":len(dx),"epochs":1200,"checkpoint":str(ck),"checkpoint_sha256":sha(ck),
                          **{"val_"+k:v for k,v in mv.items()},**{"test_"+k:v for k,v in mt.items()}})
    write_csv("permutation_results.csv",results)
    # Aggregate controls: correct across seeds versus per-permutation seed medians and pooled shuffled distribution.
    comp=[]
    for frac in (25,50):
      rr=[r for r in results if int(r["db_fraction"])==frac]; cor=[r for r in rr if r["condition"]=="correct"]; sh=[r for r in rr if r["condition"]!="correct"]
      row={"db_fraction":frac}
      for met in ("test_nll","test_brier","test_mean_selected_Q","test_B63_states","test_mean_regret"):
        cv=np.array([float(r[met]) for r in cor]); sv=np.array([float(r[met]) for r in sh])
        row.update({f"correct_{met}_median":float(np.median(cv)),f"shuffled_{met}_median":float(np.median(sv)),
                    f"shuffled_{met}_q25":float(np.quantile(sv,.25)),f"shuffled_{met}_q75":float(np.quantile(sv,.75)),
                    f"shuffled_{met}_best":float(np.min(sv) if met in ("test_nll","test_brier","test_mean_regret") else np.max(sv)),
                    f"shuffled_{met}_worst":float(np.max(sv) if met in ("test_nll","test_brier","test_mean_regret") else np.min(sv))})
      comp.append(row)
    write_csv("correct_vs_shuffled.csv",comp)
    dump("working_state.json",{"status":"STAGE_B_COMPLETE_FINALIZE_READY","completed":["stage_a","toy_semantic_model","10_derangements_x_3_seeds_x_2_fractions"],"next_action":"final scientific decision"})
    print(json.dumps(comp,indent=2))

def finalize():
    a=read_csv(H/"heldout_mode_summary.csv"); c=read_csv(H/"correct_vs_shuffled.csv"); pr=read_csv(H/"permutation_results.csv")
    fam={}
    for f in FAMILIES:
        z=[r for r in a if r["family"]==f]
        fam[f]={"heldout_modes":len(z),"modes_with_any_B63_support":sum(float(r["B63_prevalence"])>0 for r in z),
                "modes_with_high_B63_support":sum(float(r["B63_prevalence"])>=.5 for r in z),
                "mean_B63_prevalence":float(np.mean([float(r["B63_prevalence"]) for r in z])),
                "median_B63_prevalence":float(np.median([float(r["B63_prevalence"]) for r in z])),
                "mean_Q64":float(np.mean([float(r["mean_Q64"]) for r in z])),
                "median_target_residual":float(np.median([float(r["target_residual"]) for r in z])),
                "median_robust_distance":float(np.median([float(r["nearest_independent_robust_distance"]) for r in z])),
                "max_condition_number":float(np.max([float(r["condition_number"]) for r in z]))}
    primary=fam["rotation_anisotropic"]
    # Predeclared primary is rotation+anisotropic scaling; affine is a flexibility diagnostic.
    robust_fraction=primary["modes_with_high_B63_support"]/12
    residual_gain=fam["raw_toy"]["median_target_residual"]-primary["median_target_residual"]
    if robust_fraction>=.75 and residual_gain>0 and primary["max_condition_number"]<=10: A="PREDICTIVE"
    elif primary["modes_with_any_B63_support"]>=6 or fam["regularized_affine"]["modes_with_any_B63_support"]>=6: A="PARTIAL"
    else: A="OVERFIT"
    # Correct must beat the pooled shuffled median at both fractions on Q/B63/regret;
    # stable means the advantage appears at both 25% and 50%.
    adv=[]
    for r in c:
        adv.append({"frac":int(r["db_fraction"]),"q_gain":float(r["correct_test_mean_selected_Q_median"])-float(r["shuffled_test_mean_selected_Q_median"]),
                    "b63_gain":float(r["correct_test_B63_states_median"])-float(r["shuffled_test_B63_states_median"]),
                    "nll_gain":float(r["shuffled_test_nll_median"])-float(r["correct_test_nll_median"]),
                    "regret_gain":float(r["shuffled_test_mean_regret_median"])-float(r["correct_test_mean_regret_median"])})
    strong=all(x["q_gain"]>=.05 and x["b63_gain"]>=1 and x["regret_gain"]>=.03 for x in adv)
    weak=sum(x["q_gain"]>0 and x["regret_gain"]>0 for x in adv)>=1
    B="SUPPORTED" if strong else ("PARTIAL" if weak else "GENERIC_ONLY")
    joint={("PREDICTIVE","SUPPORTED"):"CROSS_SCENARIO_STRUCTURE_STRONGLY_SUPPORTED",
           ("PREDICTIVE","GENERIC_ONLY"):"GEOMETRY_ONLY_SUPPORTED",
           ("OVERFIT","SUPPORTED"):"LEARNING_ONLY_SUPPORTED"}.get((A,B),"CURRENT_CROSS_SCENARIO_CLAIM_WEAKENED" if A=="OVERFIT" and B=="GENERIC_ONLY" else "CROSS_SCENARIO_STRUCTURE_PARTIALLY_SUPPORTED")
    new_cont=sum(json.load(open(p)).get("new_continuations",0) for p in (H/"raw").glob("stage_a_q64_*_runtime.json"))
    decision={"MODE_DEFORMATION":A,"MODE_SEMANTIC_TRANSFER":B,"joint_classification":joint,"stage_a":fam,"stage_b_advantages":adv,
              "new_continuations":new_cont,"stage_b_new_rollouts":0}
    dump("final_decision.json",decision)
    rows=[{"asset":str(p.relative_to(H)),"sha256":sha(p)} for p in sorted(H.rglob("*")) if p.is_file() and p.name not in ("manifest.json",)]
    dump("manifest.json",{"task":"ORTHOFLOW3_TOY_DB_TRANSFER_FALSIFICATION_V1","artifacts":rows,"frozen_source_assets":json.load(open(H/"frozen_assets.json"))})
    dump("runtime_statistics.json",{"stage_a_new_continuations":new_cont,"stage_b_new_continuations":0,"stage_b_models":len(pr),"mode_folds":3,"heldout_predictions":60})
    report=f"""# OrthoFlow3 Toy→Double-Bottleneck transfer falsification

## Stage A — leave-mode-out deformation

The 12 canonical modes were split deterministically into three folds. Each transform saw eight DB correspondences and predicted four unseen modes without correction or local eta search. Exact Q64 used an outcome-blind eight-state DB validation panel.

| family | modes with ≥50% panel B63 | modes with any panel B63 | mean mode prevalence | mean Q64 | median target residual | max cond. |
|---|---:|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {f} | {v['modes_with_high_B63_support']}/12 | {v['modes_with_any_B63_support']}/12 | {v['mean_B63_prevalence']:.3f} | {v['mean_Q64']:.3f} | {v['median_target_residual']:.3f} | {v['max_condition_number']:.2f} |" for f,v in fam.items())+f"""

Primary geometry conclusion: **MODE_DEFORMATION = {A}**. The preregistered primary family is rotation plus anisotropic scaling and translation; regularized affine is reported as a flexibility diagnostic, not selected post hoc.

## Stage B — mode-ID permutation negative control

A Toy-only semantic trunk/head was frozen. For DB 25% and 50%, only a new DB adapter was trained. Correct identity and ten deterministic derangements used identical subsets, initialization seeds (17/23/41), optimizer, and 1,200 epochs. Evaluation executes the DB eta corresponding to each condition's declared mapping.

| DB fraction | correct Q | shuffled median Q | correct B63 | shuffled median B63 | correct regret | shuffled median regret |
|---:|---:|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {r['db_fraction']}% | {float(r['correct_test_mean_selected_Q_median']):.3f} | {float(r['shuffled_test_mean_selected_Q_median']):.3f} | {float(r['correct_test_B63_states_median']):.1f} | {float(r['shuffled_test_B63_states_median']):.1f} | {float(r['correct_test_mean_regret_median']):.3f} | {float(r['shuffled_test_mean_regret_median']):.3f} |" for r in c)+f"""

Permutation conclusion: **MODE_SEMANTIC_TRANSFER = {B}**.

## Joint conclusion

**{joint}**. Stage A asks whether DB eta geometry is predicted out of correspondence; Stage B separately asks whether Toy output-column semantics help low-data DB adaptation beyond generic multitask regularization. New rollout cost was {new_cont:,} continuations, all confined to the preregistered Stage-A direct-Q64 panel; Stage B generated none.
"""
    (H/"final_report.md").write_text(report)
    dump("working_state.json",{"status":"COMPLETE","completed":["stage_a","stage_b","finalization"],"final_decision":joint,"next_action":None})
    write_csv("experiment_ledger.csv",[{"stage":"prepare","status":"COMPLETE","new_continuations":0,"decision":"fixed folds/panel/protocol"},
                                        {"stage":"stage_a","status":"COMPLETE","new_continuations":new_cont,"decision":A},
                                        {"stage":"stage_b","status":"COMPLETE","new_continuations":0,"decision":B},
                                        {"stage":"finalize","status":"COMPLETE","new_continuations":0,"decision":joint}])
    print(json.dumps(decision,indent=2))

if __name__=="__main__": main()
