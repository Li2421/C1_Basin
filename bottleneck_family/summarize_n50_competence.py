"""Summarize complete-horizon N=50 safety-pipeline competence traces.

Gate crossing, first goal entry and final goal occupancy remain distinct.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(evaluations: list[Path], output: Path):
    rows = []
    for directory in map(Path,evaluations):
        summary = json.loads((directory/"summary.json").read_text())
        for item in summary.get("rollouts",summary.get("rows",[])):
            trace = directory/"traces"/f"{item['rollout_id']}.npz"
            with np.load(trace,allow_pickle=False) as data:
                x=np.asarray(data["positions"])
                goals=np.asarray(data["goals"])
                meta=json.loads(str(data["metadata_json"].item()))
                wall=np.asarray(data["active_wall_count"]) if "active_wall_count" in data else np.asarray(data["wall_active"])
                pair=np.asarray(data["active_pair_count"]) if "active_pair_count" in data else np.asarray(data["pair_active"])
                correction=np.asarray(data["projection_norm"])
            config=meta["config"]
            tolerance=float(config["goal_tolerance"])
            dt=float(config["dt"])
            distance=np.linalg.norm(x-goals[None],axis=2)
            active=np.linalg.norm(x[0]-goals,axis=1)>tolerance
            if meta.get("mode",meta.get("control_mode","")).startswith("temporal_"):
                active=np.ones(len(goals),dtype=bool)
            mode=item.get("mode",item.get("category",""))
            barrier=float(config["barrier_x"][0])
            direction=np.where(goals[:,0]>barrier,1.,-1.)
            exit_x=float(config["barrier_thickness"])/2 + float(config["agent_radius"])+float(config["wall_radius"])+float(config["wall_collision_margin"])
            progress=direction[None,:]*(x[:,:,0]-barrier)
            crossed=np.any(progress[:,active]>exit_x,axis=0)
            first_crossing=[int(np.flatnonzero(progress[:,agent]>exit_x)[0])
                            if np.any(progress[:,agent]>exit_x) else None
                            for agent in np.flatnonzero(active)]
            crossing_times=[t for t in first_crossing if t is not None]
            reached=np.any(distance[:,active]<=tolerance,axis=0)
            final_at_goal=distance[-1,active]<=tolerance
            time_to_goal=[int(np.flatnonzero(distance[:,agent]<=tolerance)[0])
                          if np.any(distance[:,agent]<=tolerance) else None
                          for agent in np.flatnonzero(active)]
            last=min(2000,len(x)-1)
            pending_at_start=distance[-last-1,active]>tolerance
            window_progress=(distance[-last-1,active]-distance[-1,active])/max(dt*last,1e-12)
            terminal_occupancy=float(np.mean(final_at_goal)) if len(final_at_goal) else 1.
            row=dict(evaluation=str(directory),rollout_id=item["rollout_id"],mode=mode,
                success=bool(item["success"]),termination=item["termination"],
                collision=bool(item["collision"]),steps=int(item["steps"]),
                moving_agents=int(np.sum(active)),crossed_gate=int(np.sum(crossed)),
                ever_reached_goal=int(np.sum(reached)),final_at_goal=int(np.sum(final_at_goal)),
                final_occupancy_fraction=terminal_occupancy,
                final_goal_distance_quantiles=np.quantile(distance[-1,active],[0,.25,.5,.75,1]).tolist(),
                first_goal_time_quantiles=(np.quantile([t for t in time_to_goal if t is not None],
                    [0,.25,.5,.75,1]).tolist() if any(t is not None for t in time_to_goal) else None),
                gate_throughput_per_second=float(np.sum(crossed)/(len(x)-1)/dt),
                gate_crossing_span_seconds=float((max(crossing_times)-min(crossing_times))*dt)
                    if len(crossing_times)>1 else None,
                bottleneck_throughput_per_second=float((len(crossing_times)-1)/
                    max((max(crossing_times)-min(crossing_times))*dt,dt))
                    if len(crossing_times)>1 else None,
                final_window_steps=last,
                final_window_goalward_rate_all=float(np.mean(window_progress)),
                final_window_goalward_rate_pending=float(np.mean(window_progress[pending_at_start]))
                    if np.any(pending_at_start) else None,
                wall_active_fraction=float(np.mean(wall>0)),pair_active_fraction=float(np.mean(pair>0)),
                mean_projection_norm=float(np.mean(correction)),
                min_swept_wall_clearance=item.get("min_swept_wall_clearance"),
                min_swept_agent_clearance=item.get("min_swept_agent_clearance"),
                numerical_error=item.get("numerical_error"))
            rows.append(row)
    grouped={}
    for mode in sorted(set(r["mode"] for r in rows)):
        subset=[r for r in rows if r["mode"]==mode]
        grouped[mode]=dict(runs=len(subset),success=sum(r["success"] for r in subset),
            collision=sum(r["collision"] for r in subset),
            numerical_failure=sum(r["numerical_error"] is not None for r in subset),
            mean_final_occupancy=float(np.mean([r["final_occupancy_fraction"] for r in subset])),
            mean_gate_crossing_fraction=float(np.mean([r["crossed_gate"]/r["moving_agents"] for r in subset])),
            mean_bottleneck_throughput_per_second=float(np.mean([
                r["bottleneck_throughput_per_second"] for r in subset
                if r["bottleneck_throughput_per_second"] is not None]))
                if any(r["bottleneck_throughput_per_second"] is not None for r in subset) else None,
            mean_wall_active_fraction=float(np.mean([r["wall_active_fraction"] for r in subset])),
            mean_pair_active_fraction=float(np.mean([r["pair_active_fraction"] for r in subset])),
            mean_projection_norm=float(np.mean([r["mean_projection_norm"] for r in subset])))
    report=dict(schema="gap1_n50_complete_competence_summary_v1",evaluations=[str(p) for p in evaluations],
                grouped=grouped,rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.evaluation,args.output)["grouped"],indent=2),flush=True)


if __name__=="__main__":
    main()
