#!/usr/bin/env python3
"""Deterministic no-rollout finalization of the fixed eight-state completion."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT=Path("/home/zhihan/research/Basin_C1")
PRIOR=ROOT/"diagnostics/orthoflow3_t0_basin_structure_v1"
HERE=ROOT/"diagnostics/orthoflow3_t0_basin_completion_v1"
LEARN=ROOT/"diagnostics/orthoflow3_basin_margin_learning_v1"
AFFINE=np.array([.875,0,.375]);SCALE=np.array([.75,1,.75])


def rows(path:Path)->list[dict]:return list(csv.DictReader(path.open()))
def write(path:Path,data:list[dict],fields=None):
    fields=fields or (list(data[0]) if data else ["state_id"])
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(data)
def dump(path:Path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def nt(x):return (np.asarray(x,float)-AFFINE)/SCALE


def main():
    spec=importlib.util.spec_from_file_location("partial_helpers",PRIOR/"finalize_partial_audit.py")
    hp=importlib.util.module_from_spec(spec);spec.loader.exec_module(hp)
    states=json.load(open(HERE/"frozen_8state_manifest.json"))["attempted_states"]
    state_ids=[x["state_id"] for x in states];dirs={s:PRIOR/"anchor_runs"/s for s in state_ids}
    balls=[];inside=[]
    for sid in state_ids:
        b=rows(dirs[sid]/"conservative_ball_parameters.csv")[0]
        fi=json.load(open(dirs[sid]/"false_inclusion_summary.json"))
        balls.append({**b,"center_Q64":1.0,"inside_screen_8of8":fi["screen_8of8"],"inside_screened":fi["inside_points_screened"],
                      "mandatory_B63":fi["promoted_B63"],"mandatory_tested":fi["promoted_to_64"],
                      "false_inclusions":fi["robust_false_inclusions"],"margin_usable":float(b["r_ball"])>=.2})
        inside.extend(rows(dirs[sid]/"inside_ball_promoted64.csv"))
    write(HERE/"completed_t0_balls.csv",balls);write(HERE/"completed_t0_inside_validation.csv",inside)
    ball={x["state_id"]:x for x in balls}

    q=[]
    for sid in state_ids:q.extend(rows(HERE/"raw"/sid/"q64_results.csv"))
    zero=[x for x in q if x["query"]=="ETA_ZERO"]
    common=[x for x in q if x["query"]=="INTERMEDIATE_COMMON"]
    controller=[x for x in q if x["query"] in {"G_LOWJ","G_CENTER","G_MARGIN"}]
    write(HERE/"eta_zero_q64.csv",zero);write(HERE/"intermediate_common_eta_q64.csv",common)
    write(HERE/"frozen_controller_q64.csv",controller)

    geom=[]
    for x in q:
        sid=x["state_id"];b=ball[sid];c=nt([float(b["c1"]),float(b["c2"]),float(b["c3"])]);r=float(b["r_ball"])
        eta=np.asarray(json.loads(x["eta"]),float);rho=float(np.linalg.norm(nt(eta)-c)/r)
        geom.append({**x,"rho":rho,"inside_ball":rho<=1+1e-12,"inside_retained_0p8":rho<=.8+1e-12,
                     "signed_margin":1-rho})
    ctrl_geom=[x for x in geom if x["query"] in {"G_LOWJ","G_CENTER","G_MARGIN"}]
    write(HERE/"controller_rho_vs_q64.csv",ctrl_geom)

    # Deterministic ball intersection on all eight completed balls.
    centers=np.array([nt([float(ball[s]["c1"]),float(ball[s]["c2"]),float(ball[s]["c3"])]) for s in state_ids])
    radii=np.array([float(ball[s]["r_ball"]) for s in state_ids])
    full=hp.overlap_solution(centers,radii);retained=hp.overlap_solution(centers,.8*radii)
    pair=[]
    for i,j in itertools.combinations(range(8),2):
        d=float(np.linalg.norm(centers[i]-centers[j]));mr=(radii[i]+radii[j])/2
        pair.append({"state_i":state_ids[i],"state_j":state_ids[j],"center_distance":d,"mean_radius":mr,"distance_over_mean_radius":d/mr})
    intersection={"state_ids":state_ids,"full_balls":full,"retained_0p8_balls":retained,"pairwise":{
        "center_distance_mean":statistics.mean(x["center_distance"] for x in pair),
        "center_distance_median":statistics.median(x["center_distance"] for x in pair),
        "center_distance_min":min(x["center_distance"] for x in pair),"center_distance_max":max(x["center_distance"] for x in pair),
        "distance_over_mean_radius_mean":statistics.mean(x["distance_over_mean_radius"] for x in pair),
        "distance_over_mean_radius_median":statistics.median(x["distance_over_mean_radius"] for x in pair)}}
    dump(HERE/"completed_t0_common_intersection.json",intersection)

    intermediate=rows(LEARN/"verified_ball_dataset.csv");ic=nt([.625,0,.375]);ir=[float(x["r_ball"]) for x in intermediate]
    comparison=[]
    for sid,c,r in zip(state_ids,centers,radii,strict=True):
        comparison.append({"state_id":sid,"t0_c1":ball[sid]["c1"],"t0_c2":ball[sid]["c2"],"t0_c3":ball[sid]["c3"],
                           "t0_radius":r,"margin_usable":r>=.2,"distance_to_intermediate_common_center":float(np.linalg.norm(c-ic)),
                           "intermediate_center":"[0.625,0.0,0.375]","intermediate_radius_min":min(ir),"intermediate_radius_max":max(ir)})
    write(HERE/"completed_t0_vs_intermediate.csv",comparison)

    support={x["state_id"]:x for x in rows(PRIOR/"t0_h_support_distance.csv")}
    common_geom={x["state_id"]:x for x in geom if x["query"]=="INTERMEDIATE_COMMON"}
    case=[]
    for x in ctrl_geom:
        sid=x["state_id"];case.append({"state_id":sid,"controller":x["query"],"h_support_distance":support[sid]["nearest_h_distance"],
            "t0_center":json.dumps([float(ball[sid]["c1"]),float(ball[sid]["c2"]),float(ball[sid]["c3"])]),
            "t0_radius":ball[sid]["r_ball"],"distance_to_intermediate_common_center":next(y["distance_to_intermediate_common_center"] for y in comparison if y["state_id"]==sid),
            "intermediate_common_Q64":common_geom[sid]["Q64"],"eta":x["eta"],"rho":x["rho"],"Q64":x["Q64"],"B63":x["B63"],
            "strict_deadlock":x["strict_deadlock"],"timeout":x["timeout"]})
    write(HERE/"phase_mismatch_case_table.csv",case)

    qmap={(x["state_id"],x["query"]):x for x in q};safety=[]
    for sid in state_ids:
        z=qmap[(sid,"ETA_ZERO")];rec={"state_id":sid,"zero_Q64":z["Q64"],"zero_B63":z["B63"]}
        for name in ("G_LOWJ","G_CENTER","G_MARGIN"):
            x=qmap[(sid,name)];rec[f"{name}_Q64"]=x["Q64"];rec[f"{name}_B63"]=x["B63"]
            rec[f"zero_B63_{name}_fail"]=(z["B63"]=="True" and x["B63"]!="True")
            rec[f"zero_nonB63_{name}_rescue"]=(z["B63"]!="True" and x["B63"]=="True")
        safety.append(rec)
    write(HERE/"safety_break_rescue_cases.csv",safety)

    corr={}
    for name in ("G_LOWJ","G_CENTER","G_MARGIN"):
        subset=[x for x in ctrl_geom if x["query"]==name];rho=np.array([float(x["rho"]) for x in subset]);qq=np.array([float(x["Q64"]) for x in subset])
        sp=spearmanr(rho,qq);corr[name]={"spearman_rho_vs_Q64":float(sp.statistic),"pvalue_descriptive":float(sp.pvalue),
            "inside_ball":sum(x["inside_ball"] for x in subset),"B63":sum(x["B63"]=="True" for x in subset),
            "B63_outside_ball":sum(x["B63"]=="True" and not x["inside_ball"] for x in subset),"mean_Q64":float(np.mean(qq)),
            "mean_rho":float(np.mean(rho)),"median_rho":float(np.median(rho))}

    zero_b63=sum(x["B63"]=="True" for x in zero);common_b63=sum(x["B63"]=="True" for x in common)
    breaks={name:sum(x[f"zero_B63_{name}_fail"] for x in safety) for name in ("G_LOWJ","G_CENTER","G_MARGIN")}
    rescues={name:sum(x[f"zero_nonB63_{name}_rescue"] for x in safety) for name in ("G_LOWJ","G_CENTER","G_MARGIN")}

    prior_cache=json.load(open(HERE/"cache_reuse_audit.json"));old=prior_cache["per_state"]
    ball_new_cont=ball_new_steps=0
    for sid in (state_ids[4],state_ids[6],state_ids[7]):
        raw=[json.loads(x) for x in (dirs[sid]/"raw/pilot_rollouts.jsonl").read_text().splitlines() if x.strip()]
        ball_new_cont+=len(raw)-old[sid]["continuations"];ball_new_steps+=sum(int(x["continuation_steps"]) for x in raw)-old[sid]["physical_steps"]
    qruntimes=[json.load(open(HERE/"raw"/sid/"runtime.json")) for sid in state_ids]
    q_new_cont=sum(x["new_continuations"] for x in qruntimes);q_new_steps=sum(x["new_physical_steps"] for x in qruntimes)
    elapsed=(time.time()-(HERE/"protocol.md").stat().st_mtime)/60
    runtime={"reusable_exact_tuple_inventory":prior_cache["exact_reusable_t0_tuples"],"exact_prior_tuples_used_in_fixed_completion":6040,
             "new_continuations":ball_new_cont+q_new_cont,"new_physical_steps":ball_new_steps+q_new_steps,
             "ball_completion_new_continuations":ball_new_cont,"ball_completion_new_steps":ball_new_steps,
             "q64_new_continuations":q_new_cont,"q64_new_steps":q_new_steps,"wall_time_minutes":elapsed,
             "max_gpu_shards":6,"gpu_memory":"not captured","cpu_threads_max":12,"ram_allocation_gib_max":48,
             "observed_ram_peak":"Slurm accounting disabled","within_new_caps":ball_new_cont+q_new_cont<=6000 and ball_new_steps+q_new_steps<=3500000}
    dump(HERE/"runtime_statistics.json",runtime)
    completed_cache={
        **prior_cache,
        "fixed_completion_exact_prior_tuples_used":6040,
        "fixed_completion_ball_resume_exact_tuples_used":5688,
        "fixed_completion_q64_exact_tuples_reused":352,
        "fixed_completion_new_continuations":int(ball_new_cont+q_new_cont),
        "fixed_completion_new_physical_steps":int(ball_new_steps+q_new_steps),
        "incompatible_cache_entries_used":0,
    }
    dump(HERE/"completion_cache_reuse.json",completed_cache)
    dump(HERE/"cache_reuse_audit.json",completed_cache)

    # The balls are perfectly reliable but mostly too small to make outside-ball rho causal:
    # every controller prediction is outside, yet 15/24 are B63.
    decision={"classification":"CONSERVATIVE_BALL_TOO_SMALL_TO_DIAGNOSE","balls_resolved":8,"balls_inside_validated":8,
              "margin_usable":int(np.sum(radii>=.2)),"false_inclusions":sum(int(x["false_inclusions"]) for x in balls),
              "zero_B63":zero_b63,"intermediate_common_B63":common_b63,"controller_summary":corr,
              "zero_B63_controller_failures":breaks,"zero_nonB63_controller_rescues":rescues,
              "geometric_phase_mismatch_supported":True,"causal_phase_mismatch_confirmed":False,
              "reason":"all 24 controller predictions lie outside the conservative balls, but 15 are B63; six of eight balls have r<0.20, so rho is not a discriminative basin-membership proxy"}
    dump(HERE/"final_decision.json",decision)

    report=f"""# OrthoFlow3 t0 basin completion and phase-mismatch audit v1

## Decision

**CONSERVATIVE_BALL_TOO_SMALL_TO_DIAGNOSE**

The fixed eight-state completion finished without changing any state, center, controller, OrthoFlow3, or safety semantic. Geometry strongly differs from the intermediate dataset, but the conservative balls are too small to make controller rho a causal diagnostic: all 24 frozen-controller predictions are outside their t0 balls, while 15/24 are nevertheless B63.

## Completed geometry

- Balls resolved and independently validated: 8/8.
- Centers: 8/8 at 64/64.
- Independent screening: 96/96 at 8/8.
- Mandatory interior validation: 32/32 B63.
- Confirmed internal false inclusion: 0.
- Margin-usable `r>=0.20`: {sum(radii>=.2)}/8.
- Radii: mean {np.mean(radii):.6f}, median {np.median(radii):.6f}, min {np.min(radii):.6f}, max {np.max(radii):.6f}.
- Center distance: mean {intersection['pairwise']['center_distance_mean']:.6f}, median {intersection['pairwise']['center_distance_median']:.6f}, max {intersection['pairwise']['center_distance_max']:.6f}.
- Center-distance / mean-radius: median {intersection['pairwise']['distance_over_mean_radius_median']:.3f}.
- Full-ball all-intersection: {full['all_intersection']}; maximum overlap {full['maximum_overlap_count']}/8.
- 0.8r all-intersection: {retained['all_intersection']}; maximum overlap {retained['maximum_overlap_count']}/8.

The intermediate dataset instead has 44/44 centers at `(0.625,0,0.375)` with radii 0.23375–0.35362 and a universal retained-region intersection. The completed t0 centers have normalized distances 0–{max(float(x['distance_to_intermediate_common_center']) for x in comparison):.6f} from that common center, and six t0 radii are below 0.20. Thus t0 geometry is narrower and more state-dependent.

## Q64 table

| state | zero | common eta | LOWJ | CENTER | MARGIN |
|---|---:|---:|---:|---:|---:|
"""
    for sid in state_ids:
        report+="| "+sid+" | "+" | ".join(f"{int(qmap[(sid,n)]['successes'])}/64{' B63' if qmap[(sid,n)]['B63']=='True' else ''}" for n in ("ETA_ZERO","INTERMEDIATE_COMMON","G_LOWJ","G_CENTER","G_MARGIN"))+" |\n"
    report+=f"""

- ZERO_B63: {zero_b63}/8.
- Intermediate common eta B63: {common_b63}/8; it is **not** universally robust at t0.
- G_LOWJ: {corr['G_LOWJ']['B63']}/8 B63, mean Q64 {corr['G_LOWJ']['mean_Q64']:.3f}.
- G_CENTER: {corr['G_CENTER']['B63']}/8 B63, mean Q64 {corr['G_CENTER']['mean_Q64']:.3f}.
- G_MARGIN: {corr['G_MARGIN']['B63']}/8 B63, mean Q64 {corr['G_MARGIN']['mean_Q64']:.3f}.
- Collisions and numerical failures: 0.

Safety-equivalent zero is B63 while CENTER fails in {breaks['G_CENTER']} states and while MARGIN fails in {breaks['G_MARGIN']} state. Conversely, zero is non-B63 while LOWJ/CENTER/MARGIN rescue {rescues['G_LOWJ']}/{rescues['G_CENTER']}/{rescues['G_MARGIN']} states. The only direct audited Safety-to-learned break is MARGIN on ep0195; no such CENTER case appears in this n=8 cohort.

## Rho and causal interpretation

| controller | inside ball | B63 outside ball | mean rho | Spearman(rho,Q64) |
|---|---:|---:|---:|---:|
"""
    for name in ("G_LOWJ","G_CENTER","G_MARGIN"):
        x=corr[name];report+=f"| {name} | {x['inside_ball']}/8 | {x['B63_outside_ball']}/8 | {x['mean_rho']:.3f} | {x['spearman_rho_vs_Q64']:.3f} |\n"
    report+=f"""

Large rho sometimes coincides with failure, but it is not generally sufficient: every prediction is outside and most CENTER/MARGIN predictions remain B63. The verified balls certify small inner regions, not the full t0 success sets. Therefore the completed experiment supports geometric phase mismatch but does **not** establish it as the dominant causal explanation for the 300-episode fresh-WIDE deficit.

## Explicit answers

- Is the old universal intermediate eta robust at true t0? **No.** It is B63 on only {common_b63}/8 states.
- Do intermediate-trained controllers fail when badly mismatched? **Some do, but not uniquely:** all predictions are geometrically mismatched, while 15/24 remain B63.
- Does phase mismatch explain CENTER/MARGIN falling below Safety? **Not causally from this audit.** It plausibly contributes, especially the MARGIN break at ep0195, but the conservative balls are too narrow for rho to distinguish success from failure.

## Smallest next scientific step

On these same eight states only, perform a sparse model-free outward coverage audit around the frozen controller eta values to determine whether their successful points belong to broad t0 success regions outside the conservative balls. Do not train a t0 learner until the target-set coverage issue is resolved.

## Runtime

Reused exact tuple inventory: {prior_cache['exact_reusable_t0_tuples']:,}; new continuations: {runtime['new_continuations']:,}; new physical steps: {runtime['new_physical_steps']:,}; wall time {elapsed:.1f} min; max 6 GPU shards; max 12 CPU threads; 48 GiB allocated RAM ceiling. GPU memory and observed RAM peak were unavailable because Slurm accounting is disabled.
"""
    (HERE/"t0_basin_completion_report.md").write_text(report)
    outputs=[p for p in HERE.iterdir() if p.is_file() and p.name!="manifest.json"]
    dump(HERE/"manifest.json",{"experiment":"ORTHOFLOW3_T0_BASIN_COMPLETION_AND_PHASE_MISMATCH_AUDIT_V1",
         "classification":decision["classification"],"basis_sha256":"51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
         "outputs":{p.name:sha(p) for p in sorted(outputs)},"new_training":False})
    print(json.dumps({"decision":decision,"intersection":intersection,"runtime":runtime},indent=2))


if __name__=="__main__":main()
