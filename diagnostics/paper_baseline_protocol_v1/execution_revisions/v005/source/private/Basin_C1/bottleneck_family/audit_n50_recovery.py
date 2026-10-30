"""Check whether the existing one-way expert can recover dense post-gate states.

Sources are previously saved *non-opposing* Flow traces. This diagnostic does
not add the held-out states to training or alter the accepted safety filter.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import eta_identity, uid
from .collect_scale_dagger import state_queue_rollout
from .scenario import Config


def run(trace: Path, dataset: Path, output: Path, *, anchors=(3000, 5000)):
    trace, dataset, output = Path(trace), Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    with np.load(trace, allow_pickle=False) as data:
        positions = np.asarray(data["positions"])
        goals = np.asarray(data["goals"])
        velocities = np.asarray(data["u_safe"])
        metadata = json.loads(str(data["metadata_json"].item()))
    if any(t < 1 or t >= len(positions) for t in anchors):
        raise ValueError("anchors must refer to saved nonterminal states")
    direction = 1 if metadata["category"].endswith("LR") else -1
    if direction != 1:
        raise ValueError("existing state-only recovery teacher is LR; use mirrored replay for RL")
    moving = np.flatnonzero(np.linalg.norm(positions[0]-goals, axis=1) > config.goal_tolerance)
    requests = []
    for t in anchors:
        requests.append(dict(state_uid=uid("state", {
            "physical_fingerprint":config.physical_fingerprint,
            "positions":positions[t].tolist(),"velocities":velocities[t-1].tolist(),
            "goals":goals.tolist(),"source_trace_sha256":hashlib.sha256(trace.read_bytes()).hexdigest(),
            "anchor":t}),eta_uid=eta_identity((0.,0.,0.))[0],
            controller_uid=uid("ctl", {"controller":"state_only_same_direction_queue_teacher_v1",
                                      "safety":HardProjectionConfig().to_dict()}),
            seed_keys=[str(metadata.get("seed",0))]))
    plan = output / "planned_rollouts.json"
    plan.write_text(json.dumps({"requests":requests},indent=2)+"\n")
    cache = preflight(plan)
    (output/"cache_preflight.json").write_text(json.dumps(cache,indent=2)+"\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"] != len(anchors):
        raise RuntimeError("preflight requires cache review/reuse")
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    rows = []
    for t in anchors:
        episode_config = replace(config, split="dev", seed=int(metadata.get("seed",0)))
        result = state_queue_rollout(episode_config, positions[t], velocities[t-1], goals,
                                     moving, projector)
        state = result["states"]
        dist = np.linalg.norm(goals-state[-1],axis=1)
        oriented = direction * (state[:,:,0]-config.barrier_x[0])
        row = dict(anchor=t, success=result["success"], termination=result["termination"],
                   steps=len(result["actions"]), at_goal=int(np.sum(dist<=config.goal_tolerance)),
                   final_mean_goal_distance=float(dist.mean()),
                   postgate_pending_at_anchor=int(np.sum((oriented[0]>.55)&
                                                   (np.linalg.norm(goals-state[0],axis=1)>config.goal_tolerance))),
                   min_swept_clearance=result["min_swept_clearance"])
        rows.append(row)
        np.savez_compressed(output/f"recovery_t{t}.npz",states=state,actions=result["actions"],
                            goals=goals,metadata_json=np.asarray(json.dumps(row)))
        print(json.dumps(row),flush=True)
    report=dict(schema="n50_dense_postgate_recovery_probe_v1",source_trace=str(trace),
                source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),rows=rows)
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--anchors",type=int,nargs="+",default=(3000,5000))
    args=parser.parse_args()
    run(args.trace,args.dataset,args.output,anchors=tuple(args.anchors))


if __name__ == "__main__":
    main()
