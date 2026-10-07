"""Open held-controller outcomes only after every prediction is frozen."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from math import comb

import numpy as np

from shared_rollout_db.src.rollout_db import canonical, connect
from .held_true_t0_benchmark import DEST
from .probe import read, write


def evaluate(stage):
    frozen=read(DEST / "frozen_predictions.json")
    assert frozen["held_outcomes_present_at_score_time"] is False
    all_pairs=read(DEST / "pairs.json")
    assert frozen["pairs"]==[{"state_uid":r["state_uid"],"eta_uid":r["eta_uid"],"stage":r["stage"]} for r in all_pairs]
    controller=read(DEST / "protocol.json")["held_controller_uid"]
    indices=[i for i,r in enumerate(all_pairs) if r["stage"]<=stage]
    records={}
    standard={canonical({"future_index":i}) for i in range(16)}
    with connect(True) as db:
        for i in indices:
            row=all_pairs[i]
            got=db.execute("""SELECT seed_key,success,deadlock,timeout,collision,numerical_failure,
                conflict_quarantined,compatibility_quality FROM rollout
                WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                (row["state_uid"],row["eta_uid"],controller)).fetchall()
            observed=[r for r in got if r["seed_key"] in standard and not r["numerical_failure"]
                      and not r["conflict_quarantined"] and r["compatibility_quality"]=="EXACT_REUSE"]
            numerical=sum(r["seed_key"] in standard and r["numerical_failure"] for r in got)
            success=sum(int(r["success"]) for r in observed)
            failure=len(observed)-success
            records[i]={"state_uid":row["state_uid"],"eta_uid":row["eta_uid"],
                        "observed":len(observed),"success":success,"failure":failure,
                        "numerical":numerical,"Q16":success/16 if len(observed)==16 else None,
                        "Q_lower":success/16,"Q_upper":(16-failure)/16,
                        "B15":success>=15,"non_B15":failure>=2,
                        "collision":sum(int(r["collision"]) for r in observed)}
    if any(r["observed"]+r["numerical"]!=16 for r in records.values()):
        raise RuntimeError("Some planned seeds were not merged")
    groups=defaultdict(list)
    for i in indices:groups[all_pairs[i]["state_uid"]].append(i)
    coverage=[]
    for suid,ids in groups.items():
        b=sum(records[i]["B15"] for i in ids)
        f=sum(records[i]["non_B15"] for i in ids)
        unknown=16-b-f
        status=("SELECTION_ELIGIBLE" if b and f else
                "UNDERDISCRIMINATIVE_ALL_B15" if b==16 else
                "COVERAGE_FAILURE" if not b and not unknown else "UNDERRESOLVED")
        coverage.append({"state_uid":suid,"B15_candidates":b,"non_B15_candidates":f,
                         "unknown_candidates":unknown,"status":status,
                         "hard_1to4":bool(1<=b<=4 and f>=1)})
    c_by={r["state_uid"]:r for r in coverage}
    reversals=[]
    for ua,ub in combinations(groups,2):
        ia,ib=groups[ua],groups[ub]
        for j,k in combinations(range(16),2):
            a0,a1=records[ia[j]]["Q16"],records[ia[k]]["Q16"]
            b0,b1=records[ib[j]]["Q16"],records[ib[k]]["Q16"]
            if None in (a0,a1,b0,b1):continue
            da,db=a0-a1,b0-b1
            if abs(da)>=.25 and abs(db)>=.25 and da*db<0:
                reversals.append((ia[j],ia[k],ib[j],ib[k],int(np.sign(da)),int(np.sign(db))))
    evaluated=[]
    picks={}
    for m in frozen["models"]:
        p=np.asarray(m["probabilities"],float)
        assert len(p)==len(all_pairs)
        key=(m["kind"],m["seed"],m["context"])
        pick={u:max(ids,key=lambda i:(p[i],-i)) for u,ids in groups.items()}
        picks[key]=pick
        eligible=[u for u in groups if c_by[u]["status"]=="SELECTION_ELIGIBLE"]
        s=sum(records[i]["success"] for i in indices)
        f=sum(records[i]["failure"] for i in indices)
        nll=sum(-records[i]["success"]*np.log(np.clip(p[i],1e-8,1))
                -records[i]["failure"]*np.log(np.clip(1-p[i],1e-8,1)) for i in indices)/max(1,s+f)
        evaluated.append({"kind":m["kind"],"seed":m["seed"],"context":m["context"],
            "eligible_states":len(eligible),"B15_selected":sum(records[pick[u]]["B15"] for u in eligible),
            "non_B15_selected":sum(records[pick[u]]["non_B15"] for u in eligible),
            "unknown_selected":sum(not records[pick[u]]["B15"] and not records[pick[u]]["non_B15"] for u in eligible),
            "observed_NLL":float(nll),
            "strong_state_eta_reversals":len(reversals),
            "strong_reversals_both_rankings_correct":sum(
                np.sign(p[aj]-p[ak])==sa and np.sign(p[bj]-p[bk])==sb
                for aj,ak,bj,bk,sa,sb in reversals),
            "strong_reversals_predicted_sign_change":sum(
                (p[aj]-p[ak])*(p[bj]-p[bk])<0 for aj,ak,bj,bk,_,_ in reversals),
            "mean_selected_Q_lower":float(np.mean([records[pick[u]]["Q_lower"] for u in eligible])) if eligible else None,
            "mean_selected_Q_upper":float(np.mean([records[pick[u]]["Q_upper"] for u in eligible])) if eligible else None,
            "severe_false_positive_top1":sum(p[pick[u]]>.9 and records[pick[u]]["Q_upper"]<=.5 for u in eligible),
            "mean_selected_prediction":float(np.mean([p[pick[u]] for u in eligible])) if eligible else None})
    paired=[]
    for seed in (17,23,41):
        a=picks[("eta_only",seed,"none")]
        b=picks[("physical_context",seed,"correct")]
        c=picks[("physical_context",seed,"wrong_base")]
        eligible=[u for u in groups if c_by[u]["status"]=="SELECTION_ELIGIBLE"]
        outcome=lambda choice,u:records[choice[u]]["B15"]
        rescue_eta=sum(outcome(b,u) and not outcome(a,u) for u in eligible)
        break_eta=sum(outcome(a,u) and not outcome(b,u) for u in eligible)
        discordant=rescue_eta+break_eta
        pval=(min(1.,2*sum(comb(discordant,i) for i in range(min(rescue_eta,break_eta)+1))/2**discordant)
              if discordant else 1.)
        paired.append({"seed":seed,"eligible_states":len(eligible),
            "correct_vs_eta_rescue":rescue_eta,
            "correct_vs_eta_break":break_eta,"exact_paired_sign_p":pval,
            "correct_vs_wrong_rescue":sum(outcome(b,u) and not outcome(c,u) for u in eligible),
            "correct_vs_wrong_break":sum(outcome(c,u) and not outcome(b,u) for u in eligible),
            "correct_wrong_top1_changed":sum(b[u]!=c[u] for u in eligible)})
    out={"stage":stage,"states":len(groups),"pairs":len(indices),
         "oracle_B15_states":sum(x["B15_candidates"]>0 for x in coverage),
         "selection_eligible":sum(x["status"]=="SELECTION_ELIGIBLE" for x in coverage),
         "hard_1to4":sum(x["hard_1to4"] for x in coverage),
         "all_B15":sum(x["status"]=="UNDERDISCRIMINATIVE_ALL_B15" for x in coverage),
         "coverage_failure":sum(x["status"]=="COVERAGE_FAILURE" for x in coverage),
         "underresolved":sum(x["status"]=="UNDERRESOLVED" for x in coverage),
         "B15_pairs":sum(records[i]["B15"] for i in indices),
         "non_B15_pairs":sum(records[i]["non_B15"] for i in indices),
         "numerical_seeds":sum(records[i]["numerical"] for i in indices),
         "collisions":sum(records[i]["collision"] for i in indices),
         "strong_state_eta_reversals":len(reversals),
         "coverage":coverage,"models":evaluated,"paired":paired,
         "candidate_outcomes":[{"index":i,**records[i]} for i in indices]}
    out=json.loads(json.dumps(out,allow_nan=False,
                             default=lambda value: value.item() if isinstance(value,np.generic)
                             else TypeError(f"Unserializable {type(value)}")))
    write(DEST / f"evaluation_stage{stage}.json",out)
    for name,rows in (("model_metrics",evaluated),("paired",paired),("state_coverage",coverage)):
        with (DEST / f"{name}_stage{stage}.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k not in ("coverage","models","paired","candidate_outcomes")},indent=2))
    print(json.dumps({"models":out["models"],"paired":out["paired"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",type=int,choices=(1,2),required=True)
    evaluate(parser.parse_args().stage)
