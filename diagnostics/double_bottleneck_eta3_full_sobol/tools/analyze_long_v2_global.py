#!/usr/bin/env python3
"""Freeze global matrices and deterministic representatives for long v2."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/"diagnostics/double_bottleneck_eta3_full_sobol"
LONG=STUDY/"long_run_v2"
LOW=np.asarray((.5,-.5,0.)); HIGH=np.asarray((1.25,.5,.75)); WIDTH=HIGH-LOW


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a=np.asarray(list(values),dtype=float)
    return {"count":int(len(a)),"mean":float(a.mean()) if len(a) else None,"median":float(np.median(a)) if len(a) else None,"p10":float(np.percentile(a,10)) if len(a) else None,"p90":float(np.percentile(a,90)) if len(a) else None,"minimum":float(a.min()) if len(a) else None,"maximum":float(a.max()) if len(a) else None}


def weighted(rows,key1,key2=None):
    weights=np.asarray([r["episode_steps"] for r in rows],dtype=float)
    vals=np.asarray([r[key1][key2] if key2 else r[key1] for r in rows],dtype=float)
    return float(np.dot(weights,vals)/weights.sum())


def projection(rows):
    return {
        "rollouts":len(rows),"steps":sum(r["episode_steps"] for r in rows),
        "raw_norm_mean":weighted(rows,"correction","raw_norm") if False else weighted_nested(rows,"raw_norm","mean"),
        "raw_norm_median_rollout_weighted":weighted_nested(rows,"raw_norm","median"),
        "executed_norm_mean":weighted_nested(rows,"executable_norm","mean"),
        "projection_removal_mean":weighted_nested(rows,"projection_removal_norm","mean"),
        "removal_ratio_mean":weighted_nested(rows,"removal_ratio","mean"),
        "more_than_half_removed_fraction":weighted_nested_scalar(rows,"more_than_half_removed_fraction"),
        "raw_executable_cosine_mean":weighted_nested(rows,"raw_executable_cosine","mean"),
    }


def weighted_nested(rows,name,field):
    weights=np.asarray([r["episode_steps"] for r in rows],dtype=float); values=np.asarray([r["correction"][name][field] for r in rows],dtype=float)
    return float(np.dot(weights,values)/weights.sum())


def weighted_nested_scalar(rows,name):
    weights=np.asarray([r["episode_steps"] for r in rows],dtype=float); values=np.asarray([r["correction"][name] for r in rows],dtype=float)
    return float(np.dot(weights,values)/weights.sum())


def main():
    paths=sorted((LONG/"raw").glob("global_shard*.jsonl")); rows=[]
    for path in paths: rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids=[r["job_id"] for r in rows]
    if len(rows)!=21760 or len(set(ids))!=21760: raise RuntimeError(f"global incomplete/duplicate {len(rows)}/{len(set(ids))}")
    episodes=json.loads((STUDY/"episode_catalog.json").read_text())["episodes"]
    points=json.loads((STUDY/"eta_points.json").read_text())["points"]
    eids=[e["episode_id"] for e in episodes]; eidx={x:i for i,x in enumerate(eids)}
    matrix=np.zeros((85,256),dtype=bool); lookup={}
    for r in rows: matrix[eidx[r["episode_id"]],int(r["eta_index"])]=r["success"]; lookup[(r["episode_id"],int(r["eta_index"]))]=r
    targets=[e for e in episodes if e["population"]=="safe_timeout_target"]; controls=[e for e in episodes if e["population"]=="baseline_success_control"]
    ti=[eidx[e["episode_id"]] for e in targets]; ci=[eidx[e["episode_id"]] for e in controls]
    tm=matrix[ti]; cm=matrix[ci]; theta=np.asarray([p["theta"] for p in points],dtype=float); unit=(theta-LOW)/WIDTH
    rescue=tm.sum(0); preserve=cm.sum(0)
    representatives=[]; episode_rows=[]
    for row_index,episode in enumerate(targets):
        successful=np.flatnonzero(tm[row_index]); selected=None
        if len(successful):
            scores=[]
            for index in successful:
                distances=np.linalg.norm(unit-unit[index],axis=1); nearest=np.argsort(np.where(np.arange(256)==index,np.inf,distances),kind="stable")[:8]
                neighbor_count=int(tm[row_index,nearest].sum())
                zero_distance=float(np.linalg.norm(theta[index]/WIDTH))
                scores.append((-neighbor_count,zero_distance,int(index),neighbor_count))
            best=min(scores); index=best[2]
            selected={"eta_index":index,"parameter_id":points[index]["parameter_id"],"theta":theta[index].tolist(),"successful_neighbors_among_8":best[3],"normalized_distance_to_eta_zero":best[1]}
            representatives.append({**episode,"eta_rep":selected})
        episode_rows.append({**episode,"successful_eta_count":int(len(successful)),"rho":len(successful)/256.0,"basin_exists":bool(len(successful)),"successful_eta_indices":successful.tolist(),"eta_rep":selected})
    positive=[r for r in episode_rows if r["basin_exists"]]; counts=[r["successful_eta_count"] for r in episode_rows]
    bins={"0":sum(v==0 for v in counts),"1":sum(v==1 for v in counts),"2-4":sum(2<=v<=4 for v in counts),"5-15":sum(5<=v<=15 for v in counts),">15":sum(v>15 for v in counts)}
    overlap=np.zeros((len(positive),len(positive)),dtype=float)
    for i,a in enumerate(positive):
        ai=tm[targets.index(next(e for e in targets if e["episode_id"]==a["episode_id"]))]
        for j,b in enumerate(positive):
            bi=tm[targets.index(next(e for e in targets if e["episode_id"]==b["episode_id"]))]; union=(ai|bi).sum(); overlap[i,j]=(ai&bi).sum()/union if union else 0
    off=overlap[np.triu_indices(len(positive),1)]
    table=[]
    for i,p in enumerate(points): table.append({"eta_index":i,"parameter_id":p["parameter_id"],"theta":p["theta"],"rescue_count":int(rescue[i]),"rescue_fraction":float(rescue[i]/61),"preserve_count":int(preserve[i]),"preserve_fraction":float(preserve[i]/24)})
    with np.load(STUDY/"basin_matrices.npz") as reference:
        reference_match=bool(np.array_equal(tm,reference["timeout_membership"]) and np.array_equal(cm,reference["control_preservation"]))
    if not reference_match: raise RuntimeError("detailed long-run global outcomes differ from frozen initial common-256 matrix")
    summary={"schema":"eta3_long_global_summary_v1","completed":len(rows),"expected":21760,"missing":0,"duplicates":0,"matches_initial_common_256_matrix":reference_match,"collisions":sum(r["wall_collision"] or r["agent_collision"] for r in rows),"outcomes":dict(sorted(Counter(r["outcome"] for r in rows).items())),"N_exist":len(positive),"timeouts":61,"basin_count_bins":bins,"rho_all":stats(r["rho"] for r in episode_rows),"rho_positive":stats(r["rho"] for r in positive),"best_shared":min(table,key=lambda r:(-r["rescue_count"],-r["preserve_count"],r["eta_index"])),"old_comparison":{"old_65":15,"old_A004":15,"seven_point_union":37,"full_256":len(positive)}}
    projection_output={"schema":"eta3_long_projection_v1","successful_eta":projection([r for r in rows if r["success"]]),"failed_eta":projection([r for r in rows if not r["success"]]),"eta_zero":{"raw":0.0,"executed":0.0,"removed":0.0}}
    np.savez_compressed(LONG/"timeout_basin_matrix.npz",membership=tm,episode_ids=np.asarray([e["episode_id"] for e in targets]),eta=theta)
    np.savez_compressed(LONG/"control_preservation_matrix.npz",membership=cm,episode_ids=np.asarray([e["episode_id"] for e in controls]),eta=theta)
    (LONG/"timeout_basin_matrix.json").write_text(json.dumps({"episodes":episode_rows,"eta_points":points},indent=2,sort_keys=True)+"\n")
    (LONG/"representative_eta.json").write_text(json.dumps({"episodes":representatives},indent=2,sort_keys=True)+"\n")
    (LONG/"rescue_preserve_table.json").write_text(json.dumps({"eta":table},indent=2,sort_keys=True)+"\n")
    with (LONG/"rescue_preserve_table.csv").open("w",newline="") as h:
        w=csv.writer(h); w.writerow(("eta_index","parameter_id","eta1","eta2","eta3","rescue_count","rescue_fraction","preserve_count","preserve_fraction")); [w.writerow((r["eta_index"],r["parameter_id"],*r["theta"],r["rescue_count"],r["rescue_fraction"],r["preserve_count"],r["preserve_fraction"])) for r in table]
    jacc={"schema":"eta3_long_jaccard_v1","episode_ids":[r["episode_id"] for r in positive],"matrix":overlap.tolist(),"mean":float(off.mean()) if len(off) else None,"median":float(np.median(off)) if len(off) else None,"zero_fraction":float(np.mean(off==0)) if len(off) else None,"above_0_1_fraction":float(np.mean(off>.1)) if len(off) else None,"above_0_25_fraction":float(np.mean(off>.25)) if len(off) else None}
    (LONG/"jaccard_overlap.json").write_text(json.dumps(jacc,indent=2,sort_keys=True)+"\n")
    (LONG/"projection_statistics.json").write_text(json.dumps(projection_output,indent=2,sort_keys=True)+"\n")
    (LONG/"GLOBAL_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    files=[{"path":str(p.relative_to(ROOT)),"rows":sum(1 for l in p.read_text().splitlines() if l.strip()),"sha256":sha(p)} for p in paths]
    (LONG/"global_rollouts_manifest.json").write_text(json.dumps({"total":len(rows),"files":files},indent=2,sort_keys=True)+"\n")
    # Fixed post-global jobs: 32 cube-Sobol neighbors, exact 8 seeds, and eta=0 timing comparator.
    offsets=(2*qmc.Sobol(3,scramble=True,seed=930051).random_base2(5)-1)*.05
    local_jobs=[]; seed_jobs=[]; zero_jobs=[]
    for rep in representatives:
        center=np.asarray(rep["eta_rep"]["theta"]); center_unit=(center-LOW)/WIDTH
        for k,offset in enumerate(offsets):
            value=LOW+np.clip(center_unit+offset,0,1)*WIDTH
            local_jobs.append({**rep,"stage":"long_local32","representation":"P0-3D","center_eta_index":rep["eta_rep"]["eta_index"],"parameter_id":f"L{k:02d}","sample_type":"fixed_sobol_local_radius_0.05","theta":value.tolist(),"job_id":f"long_local32|{rep['episode_id']}|{k:02d}"})
        for seed in range(1001,1009): seed_jobs.append({**rep,"stage":"long_seed8","representation":"P0-3D","center_eta_index":rep["eta_rep"]["eta_index"],"parameter_id":f"S{seed}","sample_type":"fixed_seed","theta":center.tolist(),"seed":seed,"job_id":f"long_seed8|{rep['episode_id']}|{seed}"})
        zero_jobs.append({**rep,"stage":"long_eta0","representation":"P0-3D","center_eta_index":rep["eta_rep"]["eta_index"],"parameter_id":"ETA0","sample_type":"eta_zero_timing_control","theta":[0.,0.,0.],"job_id":f"long_eta0|{rep['episode_id']}"})
    for name,jobs in (("local",local_jobs),("seed",seed_jobs),("eta0",zero_jobs)): (LONG/f"jobs/{name}.json").write_text(json.dumps({"jobs":jobs},indent=2,sort_keys=True)+"\n")
    manifest=json.loads((LONG/"run_manifest.json").read_text()); manifest.update({"completed_global":21760,"state":"global_complete_followups_registered","positive_episodes":len(positive),"local_jobs":len(local_jobs),"seed_jobs":len(seed_jobs),"eta0_jobs":len(zero_jobs)}); (LONG/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"N_exist":len(positive),"local":len(local_jobs),"seed":len(seed_jobs),"eta0":len(zero_jobs),"best":summary["best_shared"]},sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
