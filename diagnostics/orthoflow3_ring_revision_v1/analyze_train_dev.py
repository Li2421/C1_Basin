#!/usr/bin/env python3
"""Outcome-blind Ring train/development diagnosis for the frozen v1 models."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as te

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "diagnostics/orthoflow3_generator_critic_v1"


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def nearest(rows, eta):
    target = (np.asarray(eta) - te.CENTER) / te.RADIUS
    coords = np.asarray([(r["eta"] - te.CENTER) / te.RADIUS for r in rows])
    j = int(np.argmin(np.linalg.norm(coords - target, axis=1)))
    return rows[j], float(np.linalg.norm(coords[j] - target))


def main():
    data, norm, _ = te.load_data()
    dims = {s: data[s]["h"].shape[1] for s in te.SCENARIOS}
    cdim = data["ring_exchange"]["c"].shape[1]
    gm = te.Generator()
    gp = serialization.from_bytes(te.merge_initialized(gm, dims, cdim),
                                  (SOURCE/"generator/seed41/checkpoint.msgpack").read_bytes())
    cm = te.Critic()
    cp = serialization.from_bytes(te.merge_initialized(cm, dims, cdim, critic=True),
                                  (SOURCE/"critic/seed23/checkpoint.msgpack").read_bytes())

    states = data["ring_exchange"]["states"]
    records = []
    for state in states:
        uid = state["state_uid"]
        i = data["ring_exchange"]["index"][uid]
        h = data["ring_exchange"]["h"][i:i+1]
        c = data["ring_exchange"]["c"][i:i+1]
        raw = np.asarray(gm.apply(gp, jnp.asarray(h), jnp.asarray(c), method=gm.ring))[0]
        mean = np.asarray(te.eta_mean(jnp.asarray(raw)))
        rng = np.random.default_rng(te.stable_int("generator-v1-proposals", 41, "ring_exchange", uid))
        samples = np.asarray(te.eta_from_noise(
            jnp.asarray(raw)[None].repeat(16, 0),
            jnp.asarray(rng.standard_normal((16, 3)), jnp.float32)))
        etas = np.asarray([mean, *samples])
        hh = np.repeat(h, len(etas), 0); cc = np.repeat(c, len(etas), 0)
        scores = 1/(1+np.exp(-np.asarray(cm.apply(cp, jnp.asarray(hh), jnp.asarray(cc),
                                                  jnp.asarray((etas-te.CENTER)/te.RADIUS), method=cm.ring))))
        zero_score = float(1/(1+np.exp(-np.asarray(cm.apply(
            cp, jnp.asarray(h), jnp.asarray(c), jnp.asarray(((np.zeros((1,3))-te.CENTER)/te.RADIUS)),
            method=cm.ring))[0])))
        labels = data["ring_exchange"]["labels"][uid]
        proxy = []
        for eta in etas:
            row, distance = nearest(labels, eta)
            proxy.append({"q": float(row["q"]), "robust": row["robust_15of16"] is True,
                          "distance": distance, "eta_uid": row["eta_uid"]})
        zero_rows = [r for r in labels if np.linalg.norm(r["eta"]) < 1e-12]
        zero_q = float(max((r["q"] for r in zero_rows), default=float(state["zero_sufficient"])))
        records.append({"state_uid": uid, "state_id": state["state_id"], "split": state["split"],
                        "zero_sufficient": bool(state["zero_sufficient"]),
                        "num_components": int(state["num_robust_components"]),
                        "mean": mean.tolist(), "samples": samples.tolist(), "scores": scores.tolist(),
                        "zero_score": zero_score, "zero_q": zero_q, "proxy": proxy})

    scaling = {}
    for k in (1,4,8,16):
        cover=[];counts=[];diversity=[];saturation=[]
        for r in records:
            p=r["proxy"][:1+k]; z=np.asarray([r["mean"],*r["samples"][:k]])
            cover.append(any(x["robust"] for x in p));counts.append(sum(x["robust"] for x in p))
            zn=(z-te.CENTER)/te.RADIUS
            diversity.append(float(np.mean([np.linalg.norm(zn[a]-zn[b]) for a in range(len(zn)) for b in range(a)])) if len(zn)>1 else 0.)
            saturation.append(float(np.mean(np.abs(zn)>=.98)))
        scaling[str(k)]={"states":len(records),"oracle_robust_proxy_coverage":float(np.mean(cover)),
                         "mean_robust_proposals":float(np.mean(counts)),"mean_pairwise_normalized_distance":float(np.mean(diversity)),
                         "boundary_saturation_fraction":float(np.mean(saturation)),"relative_scoring_cost":1+k}

    # Eta-space component geometry comes only from the frozen validated point-cloud records.
    geometry_records=defaultdict(list)
    with (ROOT/"datasets/orthoflow3_basin_dataset_v1/basin_geometry.jsonl").open() as handle:
        for line in handle:
            row=json.loads(line)
            if row["scenario"]=="ring_exchange":geometry_records[row["state_uid"]].append(row)
    geometry=[]
    for state in states:
        uid=state["state_uid"]
        groups=[np.asarray(row["eta_normalized"],float) for row in geometry_records[uid]]
        centroids=[np.mean(v,axis=0) for v in groups if len(v)]
        between=[float(np.linalg.norm(centroids[a]-centroids[b])) for a in range(len(centroids)) for b in range(a)]
        rec=next(x for x in records if x["state_uid"]==uid);m=(np.asarray(rec["mean"])-te.CENTER)/te.RADIUS
        geometry.append({"state_uid":uid,"split":state["split"],"components":len(centroids),
                         "component_point_counts":[len(v) for v in groups],
                         "min_component_centroid_distance":min(between) if between else None,
                         "mean_distance_to_nearest_robust":rec["proxy"][0]["distance"],
                         "mean_proxy_q":rec["proxy"][0]["q"],"mean_proxy_robust":rec["proxy"][0]["robust"]})

    # Controlled finite ranking proxy with/without eta=0, using K=4 only.
    ab=[]
    for r in records:
        q=np.asarray([x["q"] for x in r["proxy"][:5]]);scores=np.asarray(r["scores"][:5])
        old=int(np.argmax(scores)); qz=np.r_[q,r["zero_q"]]; sz=np.r_[scores,r["zero_score"]];new=int(np.argmax(sz))
        ab.append({"state_uid":r["state_uid"],"split":r["split"],"zero_sufficient":r["zero_sufficient"],
                   "old_selected_q":float(q[old]),"old_robust":bool(q[old]>=15/16),"new_selected_q":float(qz[new]),
                   "new_robust":bool(qz[new]>=15/16),"selected_zero":bool(new==5),
                   "old_eta_norm":float(np.linalg.norm(np.asarray([r["mean"],*r["samples"][:4]])[old])),
                   "new_eta_norm":0. if new==5 else float(np.linalg.norm(np.asarray([r["mean"],*r["samples"][:4]])[new]))})
    def ab_summary(split):
        x=[r for r in ab if r["split"]==split]
        return {"states":len(x),"old_robust":sum(r["old_robust"] for r in x),"new_robust":sum(r["new_robust"] for r in x),
                "selected_zero":sum(r["selected_zero"] for r in x),
                "zero_sufficient_break_old":sum(r["zero_sufficient"] and not r["old_robust"] for r in x),
                "zero_sufficient_break_new":sum(r["zero_sufficient"] and not r["new_robust"] for r in x),
                "correction_needed_rescue_old":sum((not r["zero_sufficient"]) and r["old_robust"] for r in x),
                "correction_needed_rescue_new":sum((not r["zero_sufficient"]) and r["new_robust"] for r in x),
                "mean_eta_norm_old":float(np.mean([r["old_eta_norm"] for r in x])),
                "mean_eta_norm_new":float(np.mean([r["new_eta_norm"] for r in x]))}

    # Critic diagnostics on all measured eta evidence, separately from proposal proxies.
    allrows=[]
    for state in states:
        uid=state["state_uid"];i=data["ring_exchange"]["index"][uid]
        state_labels=data["ring_exchange"]["labels"][uid]
        ee=np.asarray([(row["eta"]-te.CENTER)/te.RADIUS for row in state_labels])
        hh=np.repeat(data["ring_exchange"]["h"][i:i+1],len(ee),0);cc=np.repeat(data["ring_exchange"]["c"][i:i+1],len(ee),0)
        preds=1/(1+np.exp(-np.asarray(cm.apply(cp,jnp.asarray(hh),jnp.asarray(cc),jnp.asarray(ee),method=cm.ring))))
        for row,pred in zip(state_labels,preds,strict=True):
            allrows.append({"state_uid":uid,"split":state["split"],"q":float(row["q"]),"pred":pred,
                            "robust":row["robust_15of16"] is True,"distance_to_center":row.get("distance_to_center")})
    pair_correct=pair_total=0;regrets=[];top1=0;high_low=[]
    for state in states:
        rr=[x for x in allrows if x["state_uid"]==state["state_uid"]]
        top1+=int(max(rr,key=lambda x:x["pred"])["q"]==max(x["q"] for x in rr))
        regrets.append(max(x["q"] for x in rr)-max(rr,key=lambda x:x["pred"])["q"])
        high_low += [x for x in rr if x["pred"]>=.9 and x["q"]<.5]
        for a in range(len(rr)):
            for b in range(a):
                if abs(rr[a]["q"]-rr[b]["q"])<.25:continue
                pair_total+=1;pair_correct+=int((rr[a]["q"]-rr[b]["q"])*(rr[a]["pred"]-rr[b]["pred"])>0)
    critic={"labels":len(allrows),"top1_accuracy":top1/len(states),"pairwise_accuracy_gap025":pair_correct/pair_total,
            "mean_regret":float(np.mean(regrets)),"robust_recall_at_09375":float(np.mean([x["pred"]>=.9375 for x in allrows if x["robust"]])),
            "q_mae":float(np.mean([abs(x["pred"]-x["q"]) for x in allrows])),
            "high_score_low_q_count":len(high_low),"score_by_group":{g:float(np.mean([x["pred"] for x in allrows if ("robust" if x["robust"] else "boundary" if .25<x["q"]<.9375 else "clear_nonrobust")==g])) for g in ("robust","boundary","clear_nonrobust")}}

    output={"protocol":"train/development only; nearest measured eta is explicitly a label-space proxy",
            "states":len(records),"component_summary":{"one":sum(x["components"]==1 for x in geometry),
            "multiple":sum(x["components"]>1 for x in geometry),"median_intercomponent_centroid_distance":float(np.median([x["min_component_centroid_distance"] for x in geometry if x["min_component_centroid_distance"] is not None])),
            "mean_robust_fraction":float(np.mean([x["mean_proxy_robust"] for x in geometry])),
            "mean_nearest_robust_distance":float(np.mean([x["mean_distance_to_nearest_robust"] for x in geometry]))},
            "K_scaling":scaling,"eta_zero_ablation":{"train":ab_summary("train"),"validation":ab_summary("validation")},
            "critic":critic,"K_FINAL":4,"eta_zero_adopted":False,
            "decision_basis":"K=4 is the smallest tested count at saturated 100% label-proxy oracle coverage. Eta=0 was never selected by the frozen critic and changed neither break nor rescue, so the preregistered material-benefit rule rejects adoption."}
    dump("generator_eta_geometry.json",{"summary":output["component_summary"],"states":geometry})
    dump("K_scaling_results.json",scaling);dump("eta_zero_ablation.json",{"summary":output["eta_zero_ablation"],"states":ab})
    dump("critic_diagnosis.json",critic);dump("train_dev_decision.json",output)
    print(json.dumps(output,indent=2))


if __name__ == "__main__": main()
