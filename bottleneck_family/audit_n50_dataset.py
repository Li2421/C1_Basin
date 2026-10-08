"""Stream exact Gap1 demonstration phase and directional counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scenario import Config


PHASES = ("approach", "entry", "crossing", "postgate_far", "near_goal", "settled")


def run(dataset: Path, output: Path, *, seed=846):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset/"manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    offset = config.barrier_thickness/2 + config.agent_radius + config.wall_radius + config.wall_collision_margin + .12
    rng = np.random.default_rng(seed)
    result = {}
    for row in manifest["files"]:
        with np.load(dataset/row["file"],allow_pickle=False) as data:
            initial = json.loads(str(data["initial_state_json"].item()))
            observations = np.asarray(data["observations"][:-1])
            actions = np.asarray(data["actions"])
        direction = initial.get("direction")
        if direction not in ("LR","RL"):
            raise ValueError(f"missing direction in {row['rollout_id']}")
        key = f"{row['split']}_{direction}"
        bucket = result.setdefault(key,dict(trajectories=0,transitions=0,
            independent_seeds=set(), sources={},modes={},phase={phase:dict(trajectories=0,
                task_agent_steps=0, action_active_steps=0,loss_weight_mass=0.,
                sampled_goal_distances=[]) for phase in PHASES},
            max_postgate_pending_per_trajectory=[],
            terminal_goal_occupancy=[],complete_success_trajectories=0))
        bucket["trajectories"] += 1
        bucket["transitions"] += len(actions)
        bucket["independent_seeds"].add(int(initial["episode_seed"]))
        mode=initial.get("mode")
        if mode not in ("solo","five","half","parked_half","full","nonopposing_flow_recovery",
                        "nonopposing_terminal_multi_recovery"):
            raise ValueError(f"unexpected or potentially opposing training mode: {row['rollout_id']}: {mode}")
        mode_bucket=bucket["modes"].setdefault(mode,dict(trajectories=0,transitions=0))
        mode_bucket["trajectories"]+=1
        mode_bucket["transitions"]+=len(actions)
        source = row["source"]
        source_bucket = bucket["sources"].setdefault(source,dict(
            trajectories=0,transitions=0,max_postgate_pending_per_trajectory=[]))
        source_bucket["trajectories"] += 1
        source_bucket["transitions"] += len(actions)
        p = observations[:,:,:2]
        g = p + observations[:,:,4:6]
        distance = np.linalg.norm(observations[:,:,4:6],axis=2)
        initial_active = np.linalg.norm(np.asarray(initial["goals"])-np.asarray(initial["positions"]),axis=1) > config.goal_tolerance
        active_task = initial_active[None]
        progress = (1 if direction=="LR" else -1)*(p[:,:,0]-config.barrier_x[0])
        masks = {
            "approach":active_task & (distance>config.goal_tolerance) & (progress < -offset-.04),
            "entry":active_task & (distance>config.goal_tolerance) & (progress >= -offset-.04) & (progress < -.3),
            "crossing":active_task & (distance>config.goal_tolerance) & (progress >= -.3) & (progress < offset-.04),
            "postgate_far":active_task & (distance>1.) & (progress >= offset-.04),
            "near_goal":active_task & (distance>config.goal_tolerance) & (distance<=1.) & (progress>=offset-.04),
            "settled":active_task & (distance<=config.goal_tolerance),
        }
        moving = np.linalg.norm(actions,axis=2)>.1
        weight = 1.+19.*moving
        if np.any(np.sum(np.stack(list(masks.values())),axis=0) != active_task):
            raise AssertionError(f"nonpartitioned trajectory {row['rollout_id']}")
        pending_postgate = np.sum(active_task & (progress>=offset-.04) &
                                  (distance>config.goal_tolerance),axis=1)
        bucket["max_postgate_pending_per_trajectory"].append(int(np.max(pending_postgate)))
        source_bucket["max_postgate_pending_per_trajectory"].append(int(np.max(pending_postgate)))
        terminal_d = np.linalg.norm(np.asarray(initial["goals"])-
                                    (np.asarray(initial["positions"])+np.sum(actions,axis=0)*config.dt),axis=1)
        # The integrated-action endpoint is checked against the saved goal;
        # rollout files contain the authoritative terminal state as well.
        terminal_at_goal = int(np.sum(terminal_d<=config.goal_tolerance+1e-4))
        bucket["terminal_goal_occupancy"].append(terminal_at_goal)
        bucket["complete_success_trajectories"] += int(terminal_at_goal==config.num_agents)
        for phase,mask in masks.items():
            stats=bucket["phase"][phase]
            count=int(np.sum(mask))
            stats["trajectories"] += int(count>0)
            stats["task_agent_steps"] += count
            stats["action_active_steps"] += int(np.sum(mask & moving))
            stats["loss_weight_mass"] += float(np.sum(weight*mask))
            values=distance[mask]
            if len(values):
                chosen=rng.choice(len(values),size=min(1000,len(values)),replace=False)
                stats["sampled_goal_distances"].extend(values[chosen].tolist())
    for bucket in result.values():
        bucket["independent_seeds"] = len(bucket["independent_seeds"])
        all_motion=sum(p["action_active_steps"] for p in bucket["phase"].values())
        all_weight=sum(p["loss_weight_mass"] for p in bucket["phase"].values())
        for phase in PHASES:
            stats=bucket["phase"][phase]
            stats["active_supervision_fraction"]=stats["action_active_steps"]/max(1,all_motion)
            stats["loss_weight_mass_fraction_proxy"]=stats["loss_weight_mass"]/max(1,all_weight)
            values=np.asarray(stats.pop("sampled_goal_distances"))
            stats["goal_distance_quantiles"]=(np.quantile(values,[0,.1,.5,.9,1]).tolist()
                                              if len(values) else None)
        bucket["max_postgate_pending_quantiles"]=np.quantile(
            bucket.pop("max_postgate_pending_per_trajectory"),[0,.5,1]).tolist()
        bucket["terminal_goal_occupancy_quantiles"]=np.quantile(
            bucket.pop("terminal_goal_occupancy"),[0,.5,1]).tolist()
        for source_bucket in bucket["sources"].values():
            source_bucket["max_postgate_pending_quantiles"]=np.quantile(
                source_bucket.pop("max_postgate_pending_per_trajectory"),[0,.5,1]).tolist()
    report=dict(schema=f"gap1_n{config.num_agents}_dataset_phase_audit_v1",dataset=str(dataset),
                manifest_sha256=__import__("hashlib").sha256((dataset/"manifest.json").read_bytes()).hexdigest(),
                seed=seed,phase_definitions={
                    "entry_start":-offset-.04,"gate_start":-.3,"postgate_start":offset-.04,
                    "near_goal_distance_max":1.,"success_tolerance":config.goal_tolerance,
                    "moving_action_threshold":.1,"loss_weight_mass_proxy":"1+19 I(||a||>0.1), not squared-error loss"},
                results=result)
    for split in ("train","dev"):
        left,right=result.get(f"{split}_LR"),result.get(f"{split}_RL")
        if left is None or right is None:
            continue
        if (left["trajectories"],left["transitions"],left["modes"]) != (
                right["trajectories"],right["transitions"],right["modes"]):
            raise ValueError(f"directional dataset imbalance in {split}")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run(args.dataset,args.output)
    print(json.dumps({key:{"trajectories":value["trajectories"],
                           "transitions":value["transitions"],
                           "independent_seeds":value["independent_seeds"],
                           "max_postgate_pending_quantiles":value["max_postgate_pending_quantiles"]}
                      for key,value in report["results"].items()},indent=2),flush=True)


if __name__=="__main__":
    main()
