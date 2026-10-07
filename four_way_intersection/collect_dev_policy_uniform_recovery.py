"""Development-policy full-trajectory uniform recovery acquisition for Four-Way.

This is the general local-drift complement to Base-U+W: anchors are uniformly
spaced over every valid, pre-terminal development policy trajectory, rather
than concentrated at collision locations.  Test rollout content is never read.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import jax
import numpy as np
from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import Config, FourWayIntersectionEnv
from .protocol import FourWayScenario


def _initial(root,row):
    with np.load(root/row["file"],allow_pickle=False) as d:return json.loads(str(d["initial_state_json"].item()))


def _bounded(agent,obs,key,limit):
    u=np.asarray(sample_bounded_actions(agent,obs[None],key)[0],dtype=np.float64); n=np.linalg.norm(u,axis=-1,keepdims=True)
    return u*np.minimum(1.,(limit-1e-8)/np.maximum(n,1e-12))


def _write(path,continuation,initial,audit,rid):
    states=np.asarray(continuation.states,dtype=np.float64);obs=np.asarray(continuation.observations,dtype=np.float32);act=np.asarray(continuation.actions,dtype=np.float32);digest=_digest(states,obs,act)
    metadata=dict(continuation.metadata);metadata.update(success=True,terminal_reason=continuation.terminal_reason,rollout_id=rid,split="train",source="uniform_recovery",trajectory_digest=digest)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=states,observations=obs,actions=act,initial_state_json=np.asarray(json.dumps(_jsonable(initial),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(metadata),sort_keys=True)),recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
    return {"file":str(path.relative_to(path.parents[2])),"rollout_id":rid,"split":"train","source":"uniform_recovery","trajectory_digest":digest,"length":len(act)}


def collect(source,output,checkpoint,*,seed=6017,anchors=30,position_std=.035,velocity_std=.025):
    source,output,checkpoint=Path(source),Path(output),Path(checkpoint)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/"manifest.json").read_text());cfg=Config(**manifest["scenario_config"])
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=cfg.fingerprint)
    scenario=FourWayScenario(cfg,perturb_position_std=position_std,perturb_velocity_std=velocity_std)
    # Preservation copy; no test .npz is opened below.
    shutil.copytree(source,output)
    dev=[r for r in manifest["files"] if r["split"]=="dev" and r["source"]=="nominal"]
    rows=[];adds=[];accepted=attempted=invalid=expert_failed=transitions=serial=0
    for case,row in enumerate(dev):
        initial=_initial(source,row);env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(initial["positions"]),np.asarray(initial["velocities"]))
        # Each saved snapshot is physically valid and strictly before terminal
        # collision/timeout transition.  Thus invalid collided states are never
        # used as expert recovery input.
        snaps=[env.augmented_state()]; terminal="timeout"
        for step in range(cfg.max_steps):
            u=_bounded(agent,env.observation(),jax.random.fold_in(jax.random.PRNGKey(seed+case),step),cfg.max_speed)
            _,_,done,info=env.step(u);terminal=str(info["termination"])
            if done:break
            snaps.append(env.augmented_state())
        indices=np.unique(np.linspace(0,len(snaps)-1,min(anchors,len(snaps)),dtype=int)); case_accept=0
        for time in indices:
            snap=snaps[int(time)]; source_state={"positions":np.asarray(snap["positions"],dtype=np.float64),"velocities":np.asarray(snap["last_applied_velocity"],dtype=np.float64),"split":"dev","recovery":True,"source_time":int(time)}
            attempted+=1; perturb_seed=int(np.random.SeedSequence([seed,case,int(time)]).generate_state(1)[0]);perturbed=scenario.perturb_state(source_state,np.random.default_rng(perturb_seed))
            if not scenario.valid_state(perturbed):invalid+=1;continue
            continuation=scenario.expert(perturbed,np.random.default_rng(perturb_seed))
            if not continuation.success:expert_failed+=1;continue
            audit=RecoveryAudit(source_rollout_id=f"dev_policy_uniform_v5_{case:03d}",source_split="dev",source_time=int(time),collision_type=None,collision_identity=None,distance_to_collision=None,perturbation_seed=perturb_seed,expert_success=True)
            rid=f"train_policy_uniform_v6_{serial:06d}";added=_write(output/"rollouts"/"train"/f"{rid}.npz",continuation,perturbed,audit,rid);adds.append(added);serial+=1;accepted+=1;transitions+=int(added["length"]);case_accept+=1
        rows.append({"source_rollout_id":f"dev_policy_uniform_v5_{case:03d}","nominal_rollout_id":row["rollout_id"],"terminal":terminal,"valid_preterminal_states":len(snaps),"uniform_anchor_times":indices.tolist(),"accepted":case_accept})
    changed=json.loads((output/"manifest.json").read_text());changed["files"].extend(adds);changed["counts"]["source"]["uniform_recovery"]+=accepted;changed["counts"]["split"]["train"]+=accepted
    report={"policy_checkpoint":str(checkpoint),"source":"development nominal policy trajectories only; uniformly spaced valid preterminal anchors","destination_split":"train","source_split":"dev","anchors_per_rollout_requested":anchors,"position_std":position_std,"velocity_std":velocity_std,"attempted":attempted,"accepted":accepted,"accepted_transitions":transitions,"invalid_perturbation":invalid,"expert_failed_skipped":expert_failed,"source_rollouts":rows,"test_opened":False}
    changed["extra_report"]["development_policy_uniform_recovery_v6"]=report;(output/"manifest.json").write_text(json.dumps(changed,indent=2,sort_keys=True)+"\n")
    final={**report,"base_source":str(source),"output":str(output),"output_manifest_sha256":hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()};(output/"development_policy_uniform_recovery_report.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n");return final


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("output");p.add_argument("checkpoint");p.add_argument("--seed",type=int,default=6017);p.add_argument("--anchors",type=int,default=30);a=p.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,seed=a.seed,anchors=a.anchors),indent=2))
