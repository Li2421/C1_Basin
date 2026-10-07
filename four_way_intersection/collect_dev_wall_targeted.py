"""Create a Four-Way Base-U+W dataset from development wall precursors only.

The input Base-U data remain byte-for-byte preserved in the copied output.
No test rollout is loaded: collision hotspots come exclusively from policy
rollouts started on independently held-out development nominal states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import numpy as np

from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import Config, FourWayIntersectionEnv
from .protocol import FourWayScenario


def _load_initial(dataset: Path, row: dict):
    with np.load(dataset / row["file"], allow_pickle=False) as data:
        return json.loads(str(data["initial_state_json"].item()))


def _write(path, continuation, *, initial_state, rollout_id, audit):
    states=np.asarray(continuation.states,dtype=np.float64); observations=np.asarray(continuation.observations,dtype=np.float32); actions=np.asarray(continuation.actions,dtype=np.float32)
    digest=_digest(states,observations,actions)
    metadata=dict(continuation.metadata); metadata.update(success=True,terminal_reason=continuation.terminal_reason,rollout_id=rollout_id,split="train",source="targeted_wall_obstacle",trajectory_digest=digest)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=states,observations=observations,actions=actions,
                        initial_state_json=np.asarray(json.dumps(_jsonable(initial_state),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(metadata),sort_keys=True)),recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
    return {"file":str(path.relative_to(path.parents[2])),"rollout_id":rollout_id,"split":"train","source":"targeted_wall_obstacle","trajectory_digest":digest,"length":int(len(actions))}


def _bounded(agent, observation, key, limit):
    action=np.asarray(sample_bounded_actions(agent,observation[None],key)[0],dtype=np.float64)
    # Float32 values exactly at a radial bound may be a few ulps above the
    # plant's strict host-side validator.  This is numerical representation,
    # not a safety or coordination intervention.
    norm=np.linalg.norm(action,axis=-1,keepdims=True)
    return action*np.minimum(1.,(limit-1e-8)/np.maximum(norm,1e-12))


def collect(source, output, checkpoint, *, seed=917, jitter_count=3, position_std=.025, velocity_std=.020):
    source,output,checkpoint=Path(source),Path(output),Path(checkpoint)
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((source/"manifest.json").read_text())
    if manifest["counts"]["source"].get("targeted_wall_obstacle",0): raise ValueError("source must be Base-U only")
    config=Config(**manifest["scenario_config"])
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=config.fingerprint)
    scenario=FourWayScenario(config,perturb_position_std=position_std,perturb_velocity_std=velocity_std)
    # This copies an untouched test *file* as part of Base-U+W provenance but
    # never loads it; all subsequent reads are explicitly dev or train.
    shutil.copytree(source,output)
    nominal_dev=[r for r in manifest["files"] if r["split"]=="dev" and r["source"]=="nominal"]
    additions=[]; source_rows=[]; serial=attempted=accepted=invalid=expert_failed=0
    # Dense but still short valid pre-collision window: 0.15--3.0 seconds.
    offsets=tuple(range(3,61,3))
    for case,row in enumerate(nominal_dev):
        initial=_load_initial(source,row); env=FourWayIntersectionEnv(config);env.reset(np.asarray(initial["positions"]),np.asarray(initial["velocities"]))
        snapshots=[env.augmented_state()]; collision=None
        for step in range(config.max_steps):
            action=_bounded(agent,env.observation(),jax.random.fold_in(jax.random.PRNGKey(seed+case),step),config.max_speed)
            _,_,done,info=env.step(action)
            if done: collision=info;break
            snapshots.append(env.augmented_state())
        if not collision or not collision.get("wall_collision"):
            continue  # Agent collisions are deliberately not targeted in this pass.
        used=0; collision_step=int(collision["step"]); wall_name=str(collision.get("nearest_wall_identity","outer_boundary"))
        for offset in offsets:
            time=collision_step-offset
            if not 0 <= time < len(snapshots): continue
            snap=snapshots[time]
            source_state={"positions":np.asarray(snap["positions"],dtype=np.float64),"velocities":np.asarray(snap["last_applied_velocity"],dtype=np.float64),"split":"dev","recovery":True,"source_time":time}
            source_env=FourWayIntersectionEnv(config);source_env.reset(source_state["positions"],source_state["velocities"])
            distance=float(source_env.distances()[0].min())
            for replica in range(jitter_count):
                attempted += 1; perturb_seed=int(np.random.SeedSequence([seed,case,time,replica]).generate_state(1)[0])
                perturbed=scenario.perturb_state(source_state,np.random.default_rng(perturb_seed))
                if not scenario.valid_state(perturbed): invalid += 1;continue
                continuation=scenario.expert(perturbed,np.random.default_rng(perturb_seed))
                if not continuation.success: expert_failed += 1;continue
                audit=RecoveryAudit(source_rollout_id=f"dev_policy_wall_v4_{case:03d}",source_split="dev",source_time=time,collision_type="wall",collision_identity=wall_name,distance_to_collision=distance,perturbation_seed=perturb_seed,expert_success=True)
                rid=f"train_targeted_wall_v4_{serial:06d}"
                additions.append(_write(output/"rollouts"/"train"/f"{rid}.npz",continuation,initial_state=perturbed,rollout_id=rid,audit=audit));serial+=1;accepted+=1;used+=1
        source_rows.append({"source_rollout_id":f"dev_policy_wall_v4_{case:03d}","nominal_rollout_id":row["rollout_id"],"collision_step":collision_step,"collision_identity":wall_name,"accepted":used})
    changed=json.loads((output/"manifest.json").read_text());changed["files"].extend(additions);changed["counts"]["source"]["targeted_wall_obstacle"]=accepted;changed["counts"]["split"]["train"]+=accepted
    report={"policy_checkpoint":str(checkpoint),"source":"development nominal policy rollouts only","destination_split":"train","source_split":"dev","precursor_offsets_steps":list(offsets),"jitter_count_per_precursor":jitter_count,"position_std":position_std,"velocity_std":velocity_std,"attempted":attempted,"accepted":accepted,"invalid_perturbation":invalid,"expert_failed_skipped":expert_failed,"source_rollouts":source_rows,"test_opened":False}
    changed["extra_report"]["targeted_wall_recovery_v4"]=report;(output/"manifest.json").write_text(json.dumps(changed,indent=2,sort_keys=True)+"\n")
    final={**report,"base_source":str(source),"output":str(output),"output_manifest_sha256":hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()};(output/"targeted_wall_recovery_report.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    return final


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("source");ap.add_argument("output");ap.add_argument("checkpoint");ap.add_argument("--seed",type=int,default=917);ap.add_argument("--jitter-count",type=int,default=3);a=ap.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,seed=a.seed,jitter_count=a.jitter_count),indent=2))
