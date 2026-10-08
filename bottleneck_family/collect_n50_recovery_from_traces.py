"""Collect N=50 train-only mirrored recoveries from saved non-opposing traces.

The full Flow model traces have already passed rollout preflight. This mode
avoids loading/copying the large original dataset in every recovery worker;
the validated final merge adds the complete expert demos exactly once.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, RecoveryAudit, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_scale_dagger import state_queue_rollout, replay_transform
from .flow_dataset import GapFlowScenario
from .scenario import Config


def collect(source: Path, model_trace_dir: Path, output: Path, *, anchors=(3000,5000,7000),
            terminal_offsets=(), seed=90127, source_label="uniform_recovery"):
    if source_label not in ("uniform_recovery", "late_postgate_recovery", "early_queue_recovery"):
        raise ValueError("invalid recovery source label")
    if any(int(offset) <= 0 for offset in terminal_offsets):
        raise ValueError("terminal offsets must be positive")
    source, model_trace_dir, output = map(Path,(source,model_trace_dir,output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/"manifest.json").read_text())
    config=Config(**manifest["scenario_config"])
    if config.num_agents!=50:
        raise ValueError("N=50 source required")
    candidates=[]
    skipped_anchors=[]
    model_rows=[]
    for path in sorted(model_trace_dir.glob("*.npz")):
        with np.load(path,allow_pickle=False) as data:
            positions=np.asarray(data["positions"])
            velocities=np.asarray(data["velocities"])
            goals=np.asarray(data["goals"])
            meta=json.loads(str(data["metadata_json"].item()))
        if (meta["split"] != "train" or meta["category"] not in (
                "full_LR", "full_RL", "half_LR", "half_RL",
                "five_LR", "five_RL", "parked_half_LR", "parked_half_RL")
                or meta["collision"]):
            raise ValueError(f"model trace is not valid train-only one-way data: {path}")
        if meta["physical_fingerprint"]!=config.physical_fingerprint:
            raise ValueError(f"model trace uses a different physical Gap1 environment: {path}")
        parent=meta["source_rollout_id"]
        source_path=source/"rollouts"/"train"/f"{parent}.npz"
        if not source_path.exists():
            raise ValueError(f"model trace parent absent from train dataset: {parent}")
        with np.load(source_path,allow_pickle=False) as source_data:
            original=json.loads(str(source_data["initial_state_json"].item()))
        if (int(original["episode_seed"])!=int(meta["episode_seed"])
                or not np.allclose(positions[0],original["positions"],atol=0,rtol=0)
                or not np.allclose(goals,original["goals"],atol=0,rtol=0)):
            raise ValueError(f"model trace does not match its source initial state: {path}")
        moving=np.flatnonzero(np.linalg.norm(positions[0]-goals,axis=1)>config.goal_tolerance)
        episode_config=replace(config,seed=int(meta["episode_seed"]),split="train")
        model_rows.append(dict(source=parent,trace=str(path),steps=len(positions)-1,
                               category=meta["category"],
                               checkpoint_sha256=meta["checkpoint_sha256"]))
        selected_anchors = sorted(set(int(anchor) for anchor in anchors) |
                                  {len(positions)-1-int(offset) for offset in terminal_offsets})
        for anchor in selected_anchors:
            if anchor < 0:
                skipped_anchors.append(dict(trace=str(path),anchor=anchor,
                                            reason="terminal_offset_before_start"))
                continue
            if anchor>=len(positions)-1:
                skipped_anchors.append(dict(trace=str(path),anchor=anchor,
                                            reason="model_terminated_before_anchor"))
                continue
            candidates.append((f"{parent}_flow{meta['checkpoint_sha256'][:8]}_recovery_t{anchor:04d}",episode_config,
                               positions[anchor],velocities[anchor],goals,moving,parent,anchor,
                               meta["checkpoint_sha256"],meta["category"]))
    if not candidates:
        raise ValueError("no model traces found")
    requests=[]
    for name,episode_config,p,v,g,moving,parent,anchor,checkpoint_sha,category in candidates:
        requests.append(dict(state_uid=uid("state",{
            "physical_fingerprint":config.physical_fingerprint,
            "positions":p.tolist(),"last_velocity":v.tolist(),"goals":g.tolist(),
            "episode_seed":episode_config.seed,"parent":parent,"anchor":anchor}),
            eta_uid=eta_identity((0.,0.,0.))[0],
            controller_uid=uid("ctl",{"controller":"state_only_same_direction_queue_teacher_v1",
                                      "horizon":config.max_steps}),
            seed_keys=[str(episode_config.seed)]))
    plan=output/"planned_rollouts.json"
    plan.write_text(json.dumps({"requests":requests},indent=2)+"\n")
    cache=preflight(plan)
    (output/"cache_preflight.json").write_text(json.dumps(cache,indent=2)+"\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"]!=len(candidates):
        raise RuntimeError("recovery preflight requires review/reuse")
    writer=DatasetWriter(output/"dataset",GapFlowScenario(config),scenario_config=asdict(config))
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    accepted={"LR":{"trajectories":0,"transitions":0},
              "RL":{"trajectories":0,"transitions":0}}
    failures=[]
    for index,(name,episode_config,p,v,g,moving,parent,anchor,checkpoint_sha,category) in enumerate(candidates):
        # The accepted queue expert is formulated for left-to-right traffic.
        # Reflect a right-to-left *non-opposing* train state into that frame,
        # then replay both directions through the same simulator.
        pp,vv,gg=(np.asarray(value).copy() for value in (p,v,g))
        if category.endswith("_RL"):
            pp[:,0]*=-1
            vv[:,0]*=-1
            gg[:,0]*=-1
        result=state_queue_rollout(episode_config,pp,vv,gg,moving,projector)
        if not result["success"]:
            failures.append(dict(name=name,termination=result["termination"],
                                 category=category,
                                 final_goal_distance=np.linalg.norm(result["states"][-1]-gg,axis=1).tolist()))
            print(json.dumps({"recovery":name,"success":False,"termination":result["termination"]}),flush=True)
            continue
        for mirror in (False,True):
            direction="RL" if mirror else "LR"
            initial,velocity,goals,states,obs,actions,clearance=replay_transform(
                episode_config,pp,vv,gg,result["actions"],mirror=mirror)
            recovery=RecoveryAudit(parent,"train",anchor,None,None,None,seed+index,True)
            writer.add(Trajectory(f"{name}_{direction}","train",source_label,
                {"positions":initial,"velocities":velocity,"goals":goals,
                 "episode_seed":episode_config.seed,"split":"train",
                 "mode":"nonopposing_flow_recovery","direction":direction,
                 "source_model_rollout_id":parent,"source_model_category":category,
                 "source_time":anchor},
                states,obs,actions,
                {"success":True,"terminal_reason":"success","direction":direction,
                 "teacher":"state_only_same_direction_queue_plus_accepted_safety",
                 "source_model_checkpoint_sha256":checkpoint_sha,
                 "min_swept_clearance":clearance},recovery))
            accepted[direction]["trajectories"]+=1
            accepted[direction]["transitions"]+=len(actions)
        print(json.dumps({"recovery":name,"success":True,"steps":len(result["actions"]),
                          "accepted":accepted}),flush=True)
    if accepted["LR"]!=accepted["RL"]:
        raise RuntimeError("directional recovery imbalance")
    report=dict(schema="gap1_n50_recovery_from_saved_train_traces_v1",
        source_dataset=str(source),model_trace_dir=str(model_trace_dir),
        source_label=source_label,absolute_anchors=list(anchors),
        terminal_offsets=list(terminal_offsets),skipped_anchors=skipped_anchors,
        model_trace_files=[str(p) for p in sorted(model_trace_dir.glob("*.npz"))],
        model_trace_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(model_trace_dir.glob("*.npz"))},
        model_rollouts=model_rows,candidate_count=len(candidates),
        accepted=accepted,teacher_failures=failures,preflight=cache["summary"])
    writer.finalize(extra_report=report)
    (output/"collection_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--model-trace-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--anchors",type=int,nargs="+",default=(3000,5000,7000))
    parser.add_argument("--terminal-offsets",type=int,nargs="+",default=())
    parser.add_argument("--seed",type=int,default=90127)
    parser.add_argument("--source-label",choices=("uniform_recovery", "late_postgate_recovery",
                                                 "early_queue_recovery"),
                        default="uniform_recovery")
    args=parser.parse_args()
    print(json.dumps(collect(args.source,args.model_trace_dir,args.output,
                             anchors=tuple(args.anchors),
                             terminal_offsets=tuple(args.terminal_offsets),seed=args.seed,
                             source_label=args.source_label),indent=2),flush=True)


if __name__=="__main__":
    main()
