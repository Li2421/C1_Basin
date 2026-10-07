"""Development-only forensic audit for a frozen Four-Way Stage-I checkpoint.

This utility never reads a test rollout.  It records physical collision facts,
observation/action alignment, approach-direction reversals, and analysis-only
realized crossing entries.  It does not add a mode feature or alter a policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import Config, FourWayIntersectionEnv
from .expert import CentralizedExpert
from .rollout import crossing_order_signature
from .visualization import save_trajectory_svg


def _load_dev_nominal(dataset: Path):
    manifest=json.loads((dataset / "manifest.json").read_text())
    result=[]
    for row in manifest["files"]:
        if row["split"] != "dev" or row["source"] != "nominal":
            continue
        with np.load(dataset / row["file"], allow_pickle=False) as data:
            initial=json.loads(str(data["initial_state_json"].item()))
            result.append((row["rollout_id"], initial, data["states"].copy(), data["observations"].copy(), data["actions"].copy(), json.loads(str(data["metadata_json"].item()))))
    return result


def _alignment(states, observations, actions, dt, initial_velocity):
    displacement=float(np.max(np.abs(np.diff(states,axis=0)-dt*actions)))
    prior=np.concatenate((np.asarray(initial_velocity,dtype=np.float64)[None],actions[:-1]),axis=0)
    velocity=float(np.max(np.abs(observations[:-1,:,2:4]-prior)))
    geometry=float(np.max(np.abs(observations[:-1,:,:2]-states[:-1])))
    return {"max_dynamics_residual":displacement,"max_prior_velocity_observation_residual":velocity,"max_position_observation_residual":geometry}


def _policy_episode(agent, initial, *, seed, env_config):
    env=FourWayIntersectionEnv(env_config)
    env.reset(np.asarray(initial["positions"]),np.asarray(initial["velocities"]))
    positions=[env.positions.copy()]; actions=[]; infos=[]
    key=jax.random.PRNGKey(seed)
    for t in range(env.config.max_steps):
        u=np.asarray(sample_bounded_actions(agent, env.observation()[None],jax.random.fold_in(key,t))[0],dtype=np.float64)
        # JAX produces a float32 value at the radial bound; subtract only a
        # numerical epsilon before the plant's deliberately strict validator.
        norm=np.linalg.norm(u,axis=1,keepdims=True)
        u=u*np.minimum(1.,(env.config.max_speed-1e-10)/np.maximum(norm,1e-12))
        _,_,done,info=env.step(u)
        positions.append(env.positions.copy()); actions.append(u); infos.append(info)
        if done: break
    return np.asarray(positions),np.asarray(actions),infos


def _physical_mode_diagnostics(positions, actions, initial, *, conflict_radius=.90):
    start=np.asarray(initial["positions"],dtype=np.float64); goal=np.asarray(initial.get("goals",()),dtype=np.float64)
    # Goals are fixed Four-Way geometry and absent from stored initial json.
    if goal.shape != (4,2): goal=FourWayIntersectionEnv().goals
    route=(goal-start); unit=route/np.maximum(np.linalg.norm(route,axis=1,keepdims=True),1e-12)
    projection=np.einsum("tai,ai->ta",actions,unit)
    # A reversal after initial acceleration is evidence of physical indecision,
    # not an invented coordination label.
    reversals=[int(np.sum((projection[1:,a] < -.05)&(projection[:-1,a] > .05))) for a in range(4)]
    entry=[]
    for a in range(4):
        hits=np.flatnonzero(np.linalg.norm(positions[:,a],axis=1) <= conflict_radius)
        entry.append(int(hits[0]) if len(hits) else None)
    order="incomplete" if any(t is None for t in entry) else "->".join("ABCD"[a] for a in np.argsort(entry,kind="stable"))
    return {"first_conflict_entry_steps":entry,"realized_order_if_complete":order,"goal_direction_reversal_counts":reversals,
            "fraction_goal_directed_velocity":np.mean(projection>.05,axis=0).tolist(),"mean_goal_direction_speed":np.mean(projection,axis=0).tolist()}


def run(dataset, checkpoint, output, *, seed=113):
    dataset,checkpoint,output=Path(dataset),Path(checkpoint),Path(output)
    cfg=Config(); agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=cfg.fingerprint)
    cases=_load_dev_nominal(dataset)
    rows=[]; alignments=[]; outer_initial=[]; first_trajectory=None
    for index,(rid,initial,states,obs,expert_actions,metadata) in enumerate(cases):
        alignments.append(_alignment(states,obs,expert_actions,cfg.dt,initial["velocities"]))
        p=np.asarray(initial["positions"]); h=cfg.world_half_extent-cfg.agent_radius
        outer_initial.extend((h-np.abs(p)).reshape(-1).tolist())
        positions,actions,infos=_policy_episode(agent,initial,seed=seed+index,env_config=cfg)
        last=infos[-1]; pair_idx=int(np.argmin(last["swept_agent_distances"]))
        row={"rollout_id":rid,"expert_crossing_order":metadata.get("crossing_order"),"termination":last["termination"],"episode_steps":len(actions),
             "wall_collision":bool(last["wall_collision"]),"agent_collision":bool(last["agent_collision"]),"wall_identity":last.get("nearest_wall_identity"),
             "collision_position":np.asarray(last["positions"]).tolist(),"nearest_wall_clearance":float(last.get("nearest_wall_clearance",np.nan)),
             "min_swept_wall_distance":float(last["min_swept_wall_distance"]),"min_swept_agent_distance":float(last["min_swept_agent_distance"]),
             "closest_pair":list(map(int,env_pair := FourWayIntersectionEnv(cfg).pair_indices[pair_idx])), **_physical_mode_diagnostics(positions,actions,initial)}
        rows.append(row)
        if first_trajectory is None: first_trajectory=(initial,positions,row)
    aggregate={"rollouts":len(rows),"success":float(np.mean([r["termination"]=="success" for r in rows])),"wall_collision":float(np.mean([r["wall_collision"] for r in rows])),"agent_collision":float(np.mean([r["agent_collision"] for r in rows])),"timeout":float(np.mean([r["termination"]=="timeout" for r in rows])),
               "mean_collision_step":float(np.mean([r["episode_steps"] for r in rows])),"wall_id_counts":{name:sum(r["wall_identity"]==name for r in rows) for name in cfg and ("south","east","north","west")},
               "initial_outer_clearance":{"min":float(np.min(outer_initial)),"p05":float(np.quantile(outer_initial,.05)),"median":float(np.median(outer_initial))},
               "alignment":{"max_dynamics_residual":max(x["max_dynamics_residual"] for x in alignments),"max_prior_velocity_observation_residual":max(x["max_prior_velocity_observation_residual"] for x in alignments),"max_position_observation_residual":max(x["max_position_observation_residual"] for x in alignments)}}
    # Expert verification on the same development starts, separately from policy.
    expert_rows=[]
    for rid,initial,*_ in cases:
        env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(initial["positions"]),np.asarray(initial["velocities"]));p=CentralizedExpert().plan(env)
        expert_rows.append({"rollout_id":rid,"success":p.success,"collision":p.collision,"steps":p.episode_steps,"order":p.hypothesis.label,"min_wall_clearance":p.min_wall_clearance})
    payload={"scope":"development only; no test state opened","checkpoint":str(checkpoint),"dataset":str(dataset),"seed":seed,"aggregate":aggregate,"rollouts":rows,"expert_same_initial_states":expert_rows}
    output.mkdir(parents=True,exist_ok=False);(output/"audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True))
    initial,positions,row=first_trajectory; env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(initial["positions"]),np.asarray(initial["velocities"]));save_trajectory_svg(env,positions,output/"representative_policy.svg",terminal_reason=row["termination"],order_signature=row["realized_order_if_complete"])
    return payload


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",default="diagnostics/four_way_intersection_stage1/base_u_v4_dataset");ap.add_argument("--checkpoint",default="diagnostics/four_way_intersection_stage1/base_u_v4_nominal_start_macflow/best.pkl");ap.add_argument("--output",default="diagnostics/four_way_intersection_stage1/v4_nominal_start_dev_audit");ap.add_argument("--seed",type=int,default=113);a=ap.parse_args();print(json.dumps(run(a.dataset,a.checkpoint,a.output,seed=a.seed)["aggregate"],indent=2))
