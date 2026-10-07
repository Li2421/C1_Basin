#!/usr/bin/env python3
"""Small, train/dev-only closed-loop validation of frozen generator proposals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from new_benchmark_common.basin_dataset_v1 import (
    DoubleTrainingRuntime,
    TrainingRuntime,
    cached_rows,
    double_runtime_states,
    execute_batch,
    sampled_state_rows,
)

OUT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/orthoflow3_basin_dataset_v1"
SEEDS = tuple(range(16))
COHORT_PER_SCENARIO = 6
SCENARIOS = ("double_bottleneck", "four_way_intersection", "ring_exchange")


def load(path: Path) -> Any: return json.loads(path.read_text())
def dump(name: str, value: Any) -> None:
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable(*parts: str) -> str: return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def frozen_plan() -> dict[str, Any]:
    proposals = load(OUT / "generator/frozen_validation_proposals.json")
    decisions = load(OUT / "critic/frozen_validation_selection.json")
    choice = {(r["scenario"], r["state_uid"]): r for r in decisions["states"]}
    states = []
    for scenario in SCENARIOS:
        eligible = sorted([r for r in proposals["states"] if r["scenario"] == scenario],
                          key=lambda r: stable("closed-loop-v1", scenario, r["state_uid"]))
        for row in eligible[:COHORT_PER_SCENARIO]:
            c = choice[(scenario, row["state_uid"])]
            candidates = [
                {"kind": "B0_hard_safety_eta0", "eta": [0.0, 0.0, 0.0]},
                {"kind": "B1_fixed_shared_anchor", "eta": proposals["fixed_shared_anchor"]["eta"]},
                {"kind": "B2_generator_mean", "eta": row["generator_mean"]},
                *[{"kind": f"sample_{k}", "eta": eta} for k, eta in enumerate(row["samples"])],
            ]
            states.append({**row, "critic_index": c["critic_index"], "critic_scores": c["critic_scores"],
                           "candidates": candidates})
    plan = {"schema": "orthoflow3_generator_critic_v1_closed_loop_plan",
            "selection": f"first {COHORT_PER_SCENARIO} validation states by SHA256(closed-loop-v1|scenario|state_uid)",
            "frozen_test_states": 0, "seeds": list(SEEDS), "states": states}
    dump("closed_loop/plan.json", plan)
    return plan


def runtimes(plan):
    by = defaultdict(list)
    for row in plan["states"]: by[row["scenario"]].append(row)
    result = []
    for scenario in ("four_way_intersection", "ring_exchange"):
        wanted = {r["state_uid"] for r in by[scenario]}
        states = [r for r in sampled_state_rows(scenario) if r["uid"] in wanted]
        result.append((scenario, TrainingRuntime(scenario, states, parent=False), states))
    wanted = {r["state_uid"] for r in by["double_bottleneck"]}
    dbstates = [r for r in double_runtime_states() if r["uid"] in wanted]
    groups = defaultdict(list)
    for row in dbstates: groups[row["controller_uid"]].append(row)
    for controller, states in sorted(groups.items()):
        result.append(("double_bottleneck", DoubleTrainingRuntime(states, controller), states))
    return result


def jobs_for(plan, scenario, runtime_states):
    wanted = {r["uid"]: r for r in runtime_states}
    rows = {r["state_uid"]: r for r in plan["states"] if r["scenario"] == scenario}
    jobs=[]
    for uid, state in wanted.items():
        for candidate in rows[uid]["candidates"]:
            jobs.append({"state": state, "eta": candidate["eta"], "chain": "orthoflow3",
                         "seeds": list(SEEDS), "metadata": {"method": candidate["kind"],
                         "generator_critic_v1": True, "frozen_test": False}})
    return jobs


def preflight(plan):
    detail=[]; totals=Counter()
    for scenario,runtime,states in runtimes(plan):
        for job in jobs_for(plan,scenario,states):
            rows,missing=cached_rows(runtime,job["state"],job["eta"],"orthoflow3",SEEDS)
            detail.append({"scenario":scenario,"state_uid":job["state"]["uid"],"method":job["metadata"]["method"],
                           "eta":job["eta"],"reused":len(rows),"missing_seeds":missing})
            totals["requested"]+=16;totals["reused"]+=len(rows);totals["missing"]+=len(missing)
    out={"database":str(ROOT/"shared_rollout_db/rollout.sqlite"),"summary":dict(totals),"details":detail}
    dump("closed_loop/cache_preflight.json",out);return out


def run(plan, shard, shards):
    results=[]
    for index,(scenario,runtime,states) in enumerate(runtimes(plan)):
        results.append(execute_batch(runtime,f"generator_critic_v1_closed_loop_{scenario}_g{index}",
                                     jobs_for(plan,scenario,states),shard_index=shard,num_shards=shards))
    out={"shard":shard,"shards":shards,"results":results,
         "requested":sum(r["requested"] for r in results),"reused":sum(r["reused"] for r in results),
         "physical":sum(r["physical"] for r in results)}
    dump(f"closed_loop/run_shard{shard}of{shards}.json",out);return out


def finalize(plan):
    records={}; missing=[]; numerical_incomplete=[]
    state_plan={(r["scenario"],r["state_uid"]):r for r in plan["states"]}
    for scenario,runtime,states in runtimes(plan):
        plans={r["state_uid"]:r for r in plan["states"] if r["scenario"]==scenario}
        for state in states:
            row=plans[state["uid"]]
            for candidate in row["candidates"]:
                cached,miss=cached_rows(runtime,state,candidate["eta"],"orthoflow3",SEEDS)
                if miss:
                    from shared_rollout_db.src.rollout_db import connect, canonical, eta_identity
                    keys=[canonical({"future_index":int(seed)}) for seed in miss]
                    with connect(True) as con:
                        qmarks=",".join("?"*len(keys))
                        invalid={r["seed_key"] for r in con.execute(
                            f"""SELECT seed_key FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                            AND numerical_failure=1 AND seed_key IN ({qmarks})""",
                            (state["uid"],eta_identity(candidate["eta"])[0],runtime.controllers["orthoflow3"]["uid"],*keys))}
                    unresolved=[seed for seed,key in zip(miss,keys) if key not in invalid]
                    if unresolved:
                        missing.append({"scenario":scenario,"state_uid":state["uid"],"method":candidate["kind"],"missing":unresolved})
                    else:
                        successes=sum(bool(r["success"]) for r in cached.values()); failures=len(cached)-successes
                        certified=successes>=15 or failures>=2
                        numerical_incomplete.append({"scenario":scenario,"state_uid":state["uid"],"method":candidate["kind"],"numerical_invalid_seeds":miss,"valid_seeds":len(cached),"successes":successes,"failures":failures,"robust_classification_certified":certified})
                        if not certified:
                            missing.append({"scenario":scenario,"state_uid":state["uid"],"method":candidate["kind"],"missing":miss,"reason":"numerical incomplete prevents exact robust classification"})
                records[(scenario,state["uid"],candidate["kind"])]=cached
    if missing:
        dump("closed_loop/missing_postflight.json",missing)
        raise RuntimeError(f"closed-loop evidence incomplete: {len(missing)} tuples")
    dump("closed_loop/numerical_incomplete.json",numerical_incomplete)
    per_state=[]; method_rows=[]
    for (scenario,uid),state in sorted(state_plan.items()):
        q={};robust={};outcomes={}
        for cand in state["candidates"]:
            rr=records[(scenario,uid,cand["kind"])]; k=sum(bool(r["success"]) for r in rr.values()); failures=len(rr)-k
            q[cand["kind"]]=k/16;robust[cand["kind"]]=k>=15 or (len(rr)<16 and k>=15);outcomes[cand["kind"]]=rr
            # Unrun numerical-invalid seeds are not fabricated.  q stores the
            # conservative lower endpoint; q_upper preserves the exact range.
            q[cand["kind"]+"__upper"]=(k+(16-len(rr)))/16
        generated=["B2_generator_mean",*[f"sample_{k}" for k in range(4)]]
        oracle=max(generated,key=lambda x:(q[x],-generated.index(x)))
        critic=generated[int(state["critic_index"])]
        aliases={"B3_random_generator_sample":"sample_0","B4_oracle_best_of_K":oracle,"B5_critic_selected":critic}
        for name,source in aliases.items():q[name]=q[source];q[name+"__upper"]=q[source+"__upper"];robust[name]=robust[source];outcomes[name]=outcomes[source]
        row={"scenario":scenario,"state_uid":uid,"state_id":state["state_id"],"zero_sufficient_dataset":state["zero_sufficient"],
             "critic_choice":critic,"oracle_choice":oracle,"critic_predicted_scores":state["critic_scores"]}
        for name in ("B0_hard_safety_eta0","B1_fixed_shared_anchor","B2_generator_mean","B3_random_generator_sample","B4_oracle_best_of_K","B5_critic_selected"):
            row[f"Q16_lower_{name}"]=q[name];row[f"Q16_upper_{name}"]=q[name+"__upper"];row[f"robust_{name}"]=robust[name]
        per_state.append(row)
        for name in ("B0_hard_safety_eta0","B1_fixed_shared_anchor","B2_generator_mean","B3_random_generator_sample","B4_oracle_best_of_K","B5_critic_selected"):
            counts=Counter(r["outcome"] for r in outcomes[name].values())
            method_rows.append({"scenario":scenario,"state_uid":uid,"method":name,"successes":int(16*q[name]),"valid_seeds":len(outcomes[name]),"Q16_lower":q[name],"Q16_upper":q[name+"__upper"],"robust_15of16":robust[name],"terminal":dict(counts)})
    summary={}
    for scenario in SCENARIOS:
        ss=[r for r in per_state if r["scenario"]==scenario];methods={k[len("Q16_lower_"):] for k in ss[0] if k.startswith("Q16_lower_")};summary[scenario]={}
        for method in sorted(methods):
            base=[r["robust_B0_hard_safety_eta0"] for r in ss];cur=[r[f"robust_{method}"] for r in ss]
            # Paired continuation-level rescue/break is computed from exact seed outcomes.
            rescue=brk=0
            for r in ss:
                source=(r["oracle_choice"] if method=="B4_oracle_best_of_K" else r["critic_choice"] if method=="B5_critic_selected" else "sample_0" if method=="B3_random_generator_sample" else method)
                a=records[(scenario,r["state_uid"],source)];b=records[(scenario,r["state_uid"],"B0_hard_safety_eta0")]
                common=sorted(set(a)&set(b));rescue+=sum(bool(a[k]["success"]) and not bool(b[k]["success"]) for k in common);brk+=sum(not bool(a[k]["success"]) and bool(b[k]["success"]) for k in common)
            summary[scenario][method]={"states":len(ss),"mean_Q16_lower":float(np.mean([r[f"Q16_lower_{method}"] for r in ss])),"mean_Q16_upper":float(np.mean([r[f"Q16_upper_{method}"] for r in ss])),"robust_state_rate":float(np.mean(cur)),"robust_rescue_states":sum(x and not y for x,y in zip(cur,base)),"robust_break_states":sum(y and not x for x,y in zip(cur,base)),"paired_seed_rescue":rescue,"paired_seed_break":brk}
        summary[scenario]["gaps"]={"generator_gap_Q_lower":summary[scenario]["B4_oracle_best_of_K"]["mean_Q16_lower"]-summary[scenario]["B2_generator_mean"]["mean_Q16_lower"],"critic_gap_Q_lower":summary[scenario]["B4_oracle_best_of_K"]["mean_Q16_lower"]-summary[scenario]["B5_critic_selected"]["mean_Q16_lower"],"critic_exploitation_count":sum(r["Q16_lower_B4_oracle_best_of_K"]-r["Q16_lower_B5_critic_selected"]>=.25 for r in ss)}
    dump("closed_loop/per_state.json",per_state);dump("closed_loop/method_evidence.json",method_rows);dump("closed_loop/summary.json",summary)
    post=preflight(plan);dump("closed_loop/cache_postflight.json",post)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("plan","preflight","run","finalize"));parser.add_argument("--shard",type=int,default=0);parser.add_argument("--shards",type=int,default=1);a=parser.parse_args()
    plan=frozen_plan() if a.stage=="plan" else load(OUT/"closed_loop/plan.json")
    value=plan if a.stage=="plan" else preflight(plan) if a.stage=="preflight" else run(plan,a.shard,a.shards) if a.stage=="run" else finalize(plan)
    print(json.dumps(value if a.stage!="plan" else {"states":len(plan["states"])},indent=2))


if __name__=="__main__":main()
