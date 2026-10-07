#!/usr/bin/env python3
"""Finalize fixed local/seed/timing results and protocol status."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[3]; STUDY=ROOT/"diagnostics/double_bottleneck_eta3_full_sobol"; LONG=STUDY/"long_run_v2"


def load(prefix):
    rows=[]
    for path in sorted((LONG/"raw").glob(f"{prefix}_shard*.jsonl")): rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows)!=len({r["job_id"] for r in rows}): raise RuntimeError(f"duplicate {prefix}")
    return rows


def stats(values):
    a=np.asarray(list(values),dtype=float)
    return {"count":int(len(a)),"mean":float(a.mean()) if len(a) else None,"median":float(np.median(a)) if len(a) else None,"p10":float(np.percentile(a,10)) if len(a) else None,"p25":float(np.percentile(a,25)) if len(a) else None,"p75":float(np.percentile(a,75)) if len(a) else None,"p90":float(np.percentile(a,90)) if len(a) else None,"minimum":float(a.min()) if len(a) else None,"maximum":float(a.max()) if len(a) else None}


def level(q): return "High" if q>=.75 else "Medium" if q>=.25 else "Low"


def weighted_nested(rows,name,field=None):
    w=np.asarray([r["episode_steps"] for r in rows],float); v=np.asarray([r["correction"][name][field] if field else r["correction"][name] for r in rows],float); return float(np.dot(w,v)/w.sum())


def projection(rows):
    return {"rollouts":len(rows),"steps":sum(r["episode_steps"] for r in rows),"raw_norm_mean":weighted_nested(rows,"raw_norm","mean"),"executed_norm_mean":weighted_nested(rows,"executable_norm","mean"),"projection_removal_mean":weighted_nested(rows,"projection_removal_norm","mean"),"removal_ratio_mean":weighted_nested(rows,"removal_ratio","mean"),"more_than_half_removed_fraction":weighted_nested(rows,"more_than_half_removed_fraction"),"raw_executable_cosine_mean":weighted_nested(rows,"raw_executable_cosine","mean")}


def main():
    global_rows=load("global"); local=load("local"); seed=load("seed"); eta0=load("eta0")
    manifest=json.loads((LONG/"run_manifest.json").read_text())
    expected=(manifest["local_jobs"],manifest["seed_jobs"],manifest["eta0_jobs"])
    if (len(local),len(seed),len(eta0))!=expected: raise RuntimeError(f"followups incomplete {(len(local),len(seed),len(eta0))} != {expected}")
    reps=json.loads((LONG/"representative_eta.json").read_text())["episodes"]
    local_by=defaultdict(list); seed_by=defaultdict(list)
    for r in local: local_by[r["episode_id"]].append(r)
    for r in seed: seed_by[r["episode_id"]].append(r)
    local_eps=[]; seed_eps=[]
    for rep in reps:
        eid=rep["episode_id"]; lr=local_by[eid]; sr=seed_by[eid]; ql=sum(r["success"] for r in lr)/32; qs=sum(r["success"] for r in sr)/8
        local_eps.append({"episode_id":eid,"eta_rep":rep["eta_rep"],"successes":sum(r["success"] for r in lr),"samples":32,"Q_local":ql,"category":level(ql),"center_success":True,"terminal_reasons":dict(sorted(Counter(r["outcome"] for r in lr).items()))})
        seed_eps.append({"episode_id":eid,"eta_rep":rep["eta_rep"],"successes":sum(r["success"] for r in sr),"samples":8,"Q_seed":qs,"category":level(qs),"terminal_reasons":dict(sorted(Counter(r["outcome"] for r in sr).items()))})
    local_out={"schema":"eta3_long_local_robustness_v1","protocol":{"samples":32,"radius":.05,"seed":930051},"episodes":local_eps,"summary":{"Q_local":stats(r["Q_local"] for r in local_eps),"categories":dict(sorted(Counter(r["category"] for r in local_eps).items())),"collisions":sum(r["wall_collision"] or r["agent_collision"] for r in local)}}
    seed_out={"schema":"eta3_long_seed_robustness_v1","protocol":{"seeds":list(range(1001,1009))},"episodes":seed_eps,"summary":{"Q_seed":stats(r["Q_seed"] for r in seed_eps),"categories":dict(sorted(Counter(r["category"] for r in seed_eps).items())),"collisions":sum(r["wall_collision"] or r["agent_collision"] for r in seed)}}
    global_lookup={(r["episode_id"],int(r["eta_index"])):r for r in global_rows}; zero_lookup={r["episode_id"]:r for r in eta0}
    timing=[]
    for rep in reps:
        eid=rep["episode_id"]; success=global_lookup[(eid,int(rep["eta_rep"]["eta_index"]))]; zero=zero_lookup[eid]
        z=zero["timing"]; s=success["timing"]
        def delta(key): return s[key]-z[key] if s[key] is not None and z[key] is not None else None
        z_goal=z["final_goal_entry_time"] if z["final_goal_entry_time"] is not None else 42.5
        timing.append({"episode_id":eid,"eta_index":rep["eta_rep"]["eta_index"],"first_bottleneck_clearance_delta_seconds":delta("time_first_bottleneck_clears"),"second_bottleneck_clearance_delta_seconds":delta("time_second_bottleneck_clears"),"total_waiting_delta_agent_seconds":s["total_waiting_agent_seconds"]-z["total_waiting_agent_seconds"],"final_goal_entry_delta_censored_seconds":s["final_goal_entry_time"]-z_goal,"completion_time_delta_vs_horizon_seconds":s["completion_time"]-42.5,"eta0_goal_entry_was_censored":z["final_goal_entry_time"] is None,"eta0":z,"eta_rep":s})
    timing_out={"schema":"eta3_long_timing_analysis_v1","sign":"eta_rep minus eta0; negative means earlier/less","goal_entry_censoring":"missing eta0 final entry is censored at 42.5 s","episodes":timing,"aggregate":{key:stats(r[key] for r in timing if r[key] is not None) for key in ("first_bottleneck_clearance_delta_seconds","second_bottleneck_clearance_delta_seconds","total_waiting_delta_agent_seconds","final_goal_entry_delta_censored_seconds","completion_time_delta_vs_horizon_seconds")}}
    projection_out=json.loads((LONG/"projection_statistics.json").read_text()); eta0_projection=projection(eta0); eta0_projection["removal_ratio_mean"]=None; eta0_projection["note"]="raw eta correction is exactly zero; ratio is undefined, while ~1e-11 executed/removal norms are solver tolerance"; projection_out["eta_zero_hard_safety"]=eta0_projection; (LONG/"projection_statistics.json").write_text(json.dumps(projection_out,indent=2,sort_keys=True)+"\n")
    basin=json.loads((LONG/"timeout_basin_matrix.json").read_text())["episodes"]; positive=[r for r in basin if r["basin_exists"]]; negative=[r for r in basin if not r["basin_exists"]]; median=float(np.median([r["rho"] for r in positive]))
    r1=min(positive,key=lambda r:(-r["rho"],r["episode_id"])); r2=min(positive,key=lambda r:(abs(r["rho"]-median),r["episode_id"])); r3=min(positive,key=lambda r:(r["rho"],r["episode_id"])); r4=min(negative,key=lambda r:r["episode_id"]) if negative else None
    selection={"R1":r1["episode_id"],"R2":r2["episode_id"],"R3":r3["episode_id"],"R4":r4["episode_id"] if r4 else None,"rules":"exact fixed protocol","median_positive_rho":median}
    global_summary=json.loads((LONG/"GLOBAL_SUMMARY.json").read_text()); n=global_summary["N_exist"]; ql=float(np.median([r["Q_local"] for r in local_eps])); qs=float(np.median([r["Q_seed"] for r in seed_eps]))
    status="PASS" if n>=31 and ql>=.25 and qs>=.50 else "REVISE" if n>=16 else "REJECT"
    final={"schema":"eta3_long_final_summary_v1","N_exist":n,"timeouts":61,"median_positive_basin_fraction":global_summary["rho_positive"]["median"],"median_Q_local":ql,"median_Q_seed":qs,"median_pairwise_jaccard":json.loads((LONG/"jaccard_overlap.json").read_text())["median"],"best_shared_eta":global_summary["best_shared"],"all_rollouts_collision_free":sum(r["wall_collision"] or r["agent_collision"] for r in global_rows+local+seed+eta0)==0,"tested_rollouts_total":len(global_rows)+len(local)+len(seed)+len(eta0),"status":status,"fixed_rule":{"pass":n>=31 and ql>=.25 and qs>=.50,"existence":n>=31,"local":ql>=.25,"seed":qs>=.50}}
    (LONG/"local_robustness.json").write_text(json.dumps(local_out,indent=2,sort_keys=True)+"\n"); (LONG/"seed_robustness.json").write_text(json.dumps(seed_out,indent=2,sort_keys=True)+"\n"); (LONG/"timing_analysis.json").write_text(json.dumps(timing_out,indent=2,sort_keys=True)+"\n"); (LONG/"representative_selection.json").write_text(json.dumps(selection,indent=2,sort_keys=True)+"\n"); (LONG/"FINAL_SUMMARY.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    manifest.update({"state":"complete_pending_plots_regression","completed_local":len(local),"completed_seed":len(seed),"completed_eta0":len(eta0),"status":status}); (LONG/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(final,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
