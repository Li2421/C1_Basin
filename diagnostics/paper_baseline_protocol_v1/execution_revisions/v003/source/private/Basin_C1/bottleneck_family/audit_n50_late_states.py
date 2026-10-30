"""Compare expert and Flow actions at failed non-opposing post-gate states."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .scenario import Config


def run(traces: list[Path], output: Path, *, anchors=(10000,20000,30000,39000)):
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    rows=[]
    for trace in map(Path,traces):
        with np.load(trace,allow_pickle=False) as data:
            x=np.asarray(data["positions"])
            goals=np.asarray(data["goals"])
            flow=np.asarray(data["u_flow"])
            safe=np.asarray(data["u_safe"])
            projection=np.asarray(data["projection_norm"])
            wall=np.asarray(data["wall_active"])
            pair=np.asarray(data["pair_active"])
            metadata=json.loads(str(data["metadata_json"].item()))
        config=Config(**metadata["config"])
        direction=1 if metadata["category"].endswith("LR") else -1
        for t in anchors:
            if t>=len(flow):
                continue
            env=BottleneckEnv(replace(config,seed=int(metadata["seed"]),split="dev"))
            env.reset(x[t],goals)
            env.velocities=safe[t-1]
            reference=teacher_reference(env)
            teacher_safe=np.asarray(projector(env.snapshot(),reference).velocity)
            delta=goals-x[t]
            distance=np.linalg.norm(delta,axis=1)
            unit=delta/np.maximum(distance[:,None],1e-12)
            pending=(direction*(x[t,:,0]-config.barrier_x[0])>.55)&(distance>config.goal_tolerance)
            if not np.any(pending):
                continue
            radial=lambda velocity:np.sum(velocity[pending]*unit[pending],axis=1)
            row=dict(trace=str(trace),checkpoint=metadata["checkpoint"],
                category=metadata["category"],anchor=t,
                postgate_pending=int(np.sum(pending)),
                postgate_goal_distance_quantiles=np.quantile(distance[pending],[0,.25,.5,.75,1]).tolist(),
                expert_reference_goalward_speed=float(np.mean(radial(reference))),
                expert_projected_goalward_speed=float(np.mean(radial(teacher_safe))),
                flow_goalward_speed=float(np.mean(radial(flow[t]))),
                executed_goalward_speed=float(np.mean(radial(safe[t]))),
                expert_projection_norm=float(np.linalg.norm(teacher_safe-reference)),
                model_projection_norm=float(projection[t]),
                flow_speed=float(np.mean(np.linalg.norm(flow[t,pending],axis=1))),
                executed_speed=float(np.mean(np.linalg.norm(safe[t,pending],axis=1))),
                flow_goal_alignment=float(np.mean(radial(flow[t])/
                    np.maximum(np.linalg.norm(flow[t,pending],axis=1),1e-12))),
                model_wall_active=bool(wall[t]),model_pair_active=bool(pair[t]),
                final_window_wall_active_fraction=float(np.mean(wall[max(0,len(wall)-2000):])),
                final_window_pair_active_fraction=float(np.mean(pair[max(0,len(pair)-2000):])),
                final_window_projection_norm=float(np.mean(projection[max(0,len(projection)-2000):])))
            rows.append(row)
    report=dict(schema="gap1_n50_failed_postgate_state_action_audit_v1",
                anchors=list(anchors),rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.trace,args.output),indent=2),flush=True)


if __name__=="__main__":
    main()
