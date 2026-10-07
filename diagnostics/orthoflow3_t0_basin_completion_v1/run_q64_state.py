#!/usr/bin/env python3
"""Complete matched Q64 queries for one frozen true-t0 state."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/zhihan/research/Basin_C1")
PRIOR = ROOT / "diagnostics/orthoflow3_t0_basin_structure_v1"
HERE = ROOT / "diagnostics/orthoflow3_t0_basin_completion_v1"
WRAPPER = PRIOR / "run_t0_anchor.py"
FUTURE_ROOT = 2026092811


def load_wrapper():
    spec=importlib.util.spec_from_file_location("t0_completion_wrapper",WRAPPER)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    states_list=json.load(open(HERE/"frozen_8state_manifest.json"))["attempted_states"]
    state=states_list[args.index];sid=state["state_id"]
    out=HERE/"raw"/sid;out.mkdir(parents=True,exist_ok=True)
    wrapper=load_wrapper();m=wrapper.load_patched()
    m.HERE=out;m.STATE_IDS=[sid];m.QDIR=PRIOR/"synthetic_qdir";m.DDIR=PRIOR/"empty_prior"
    m.QGDIR=ROOT/"diagnostics/orthoflow3_q_guided_direct_eta_v1"
    m.CAP_CONT=500;m.CAP_STEPS=500000
    old=m.load_old_module();old.FUTURE_ROOT=FUTURE_ROOT;old.CAP_CONT=500;old.CAP_STEPS=500000
    # load_old_module needs its original code directory; only the subsequent
    # exact-prior lookup should point at this state's completed t0 run.
    m.OLD=PRIOR/"anchor_runs"/sid
    qstates={x["state_id"]:x for x in json.load(open(PRIOR/"synthetic_qdir/eligible_state_manifest.json"))["selected_states"]}
    states={sid:qstates[sid]};features=np.load(PRIOR/"synthetic_qdir/conditioning_features.npz")["features"]
    _norm,_affine,_scale,lo,hi,_vertices,_tv,_hull,_eq=m.geometry()
    oracle=m.ContinuousOracle(old,states,features,lo,hi).obj
    predictions=list(csv.DictReader(open(HERE/"frozen_controller_predictions.csv")))
    pred={x["controller"]:np.asarray(json.loads(x["eta"]),dtype=float) for x in predictions if x["state_id"]==sid}
    queries=[("ETA_ZERO",np.zeros(3)),("INTERMEDIATE_COMMON",np.array([0.625,0.0,0.375])),
             ("G_LOWJ",pred["G_LOWJ"]),("G_CENTER",pred["G_CENTER"]),("G_MARGIN",pred["G_MARGIN"])]
    results=[]
    for name,eta in queries:
        tasks=old.taskset(sid,eta,64,query_name=name)
        rows=oracle.ensure(tasks,f"completion_{name}")
        outcomes=Counter(x["outcome"] for x in rows);success=[x for x in rows if x["success"]]
        results.append({"state_id":sid,"query":name,"eta":json.dumps(eta.tolist(),separators=(",",":")),
                        "successes":len(success),"trials":64,"Q64":len(success)/64,"B63":len(success)>=63,
                        "strict_deadlock":outcomes.get("safe_deadlock",0),"timeout":outcomes.get("timeout",0),
                        "collision":outcomes.get("collision",0),"other_numerical":outcomes.get("other_numerical",0),
                        "episode_length_mean":statistics.mean(int(x["continuation_steps"]) for x in rows),
                        "successful_J_def_mean":statistics.mean(float(x["J_def"]) for x in success) if success else "",
                        "successful_J_def_median":statistics.median(float(x["J_def"]) for x in success) if success else ""})
    with (out/"q64_results.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={"state_id":sid,"new_continuations":oracle.new,"new_physical_steps":oracle.steps,
             "reused_prior_tuples":len(oracle.used_prior_pilot_keys),"reused_other_tuples":len(oracle.used_source_keys),
             "results":results}
    (out/"runtime.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
