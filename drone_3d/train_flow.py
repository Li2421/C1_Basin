"""Train actual shared MACFlow on cached 3-D teacher transitions, on one CPU."""
from . import resources  # noqa: F401, must precede JAX
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from new_benchmark_common.macflow import JointMACFlowAgent, get_config, parameter_count, save_checkpoint
from .pipeline import environment_fingerprint, observation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cached_data(root):
    splits={"train":[],"dev":[]}; sources=[]
    for seed in (17,23,41):
        path=root/f"seed{seed}_route_witness.npz"
        record=json.loads(path.with_suffix(".json").read_text())
        assert sha(path)==record["trace_sha256"] and record["success"]
        with np.load(path,allow_pickle=False) as trace:
            p=trace["positions"]; u=trace["actions"]; g=trace["goals"]
            v=np.concatenate([np.zeros((1,2,3)),u[:-1]])
            obs=np.stack([observation(x,y,g) for x,y in zip(p[:-1],v)])
            # Remove only stationary pre-intervention waiting supervision.
            # Such dwell is a teacher timer, absent from the deployed state.
            keep=(np.arange(len(u))>=record["deadlock_event_step"]) | (np.linalg.norm(u,axis=2).max(axis=1)>.02)
            splits["dev" if seed==41 else "train"].append((obs[keep],u[keep].astype(np.float32)))
            sources.append(dict(seed=seed,split="dev" if seed==41 else "train",path=str(path.resolve()),sha256=sha(path),
                                original_transitions=len(u),retained_transitions=int(keep.sum()),new_rollouts=0))
    return {s:(np.concatenate([v[0] for v in a]),np.concatenate([v[1] for v in a])) for s,a in splits.items()},sources


def train(output, steps=4000, terminal_data=None, width=64):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"best.pkl").exists():
        print("Existing checkpoint retained; read training_summary.json. No retraining."); return
    dataset,sources=cached_data(Path("diagnostics/drone_3d_pilot_v3"))
    if terminal_data is not None:
        for path in sorted(terminal_data.glob("seed*.json")):
            r=json.loads(path.read_text()); assert r["success"] and sha(r["trace_path"])==r["trace_sha256"]
            with np.load(r["trace_path"],allow_pickle=False) as z:
                u=z["actions"].astype(np.float32); velocities=np.concatenate([np.array(r["initial_velocities"])[None],u[:-1]])
                obs=np.stack([observation(p,v,z["goals"]) for p,v in zip(z["positions"][:-1],velocities)])
            # Equalize terminal support with route transitions, before fitting normalization.
            repeat=4
            part=r["split"]; oldx,oldy=dataset[part]
            dataset[part]=(np.concatenate([oldx,np.tile(obs,(repeat,1,1))]),np.concatenate([oldy,np.tile(u,(repeat,1,1))]))
            sources.append(dict(path=str(path.resolve()),sha256=sha(path),split=part,retained_transitions=len(u),training_repeat=repeat,new_rollouts=0))
    x,y=dataset["train"]; dx,dy=dataset["dev"]
    # Floors avoid extreme amplification of dimensions barely varying in 2 traces.
    mean=x.reshape(len(x),-1).mean(0); scale=np.maximum(x.reshape(len(x),-1).std(0),.1)
    cfg=get_config(num_agents=2,obs_dim=12,act_dim=3,agent_order=("A","B"),
                   actor_hidden_dims=(width,width),max_speed=.5,lr=1e-3,
                   obs_mean=tuple(map(float,mean)),obs_scale=tuple(map(float,scale)),
                   act_mean=(0.,)*6,act_scale=(.5,)*6,environment_fingerprint=environment_fingerprint(),
                   endpoint_action_loss_weight=1.)
    protocol=dict(sources=sources,actor="shared official JointMACFlowAgent",config=dict(cfg),steps=steps,batch_size=128,
                  seed=2718,train_source=sha(__file__),simulator_source=sha(Path(__file__).with_name("pilot.py")),
                  observations="p,v,g-p,p_other-p only; no phase/timer/waypoint",test_used=False,
                  history_reuse="All training/validation transitions reused from v3 NPZ; no compatible trained 3D checkpoint existed")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    np.savez_compressed(output/"cached_transitions.npz",train_observations=x,train_actions=y,dev_observations=dx,dev_actions=dy)
    start=time.perf_counter(); start_cpu=time.process_time()
    agent=JointMACFlowAgent.create(2718,jnp.asarray(x[:1]),jnp.asarray(y[:1]),cfg)
    rng=np.random.default_rng(2718); best=float("inf"); logs=[]; best_step=0
    validation={"observations":jnp.asarray(dx),"actions":jnp.asarray(dy)}
    for step in range(1,steps+1):
        indices=rng.integers(len(x),size=128)
        agent,info=agent.update(dict(observations=jnp.asarray(x[indices]),actions=jnp.asarray(y[indices])),step)
        if step%500==0 or step==steps:
            val=float(agent.endpoint_bc_loss(validation,agent.network.params,jax.random.PRNGKey(31415)))
            row=dict(step=step,train_loss=float(info["total_loss"]),dev_endpoint_mse_normalized=val,elapsed_seconds=time.perf_counter()-start)
            logs.append(row); print(json.dumps(row),flush=True)
            if val<best:
                best=val; best_step=step
                save_checkpoint(output/"best.pkl",agent,dict(protocol_sha256=sha(output/"protocol.json"),step=step,dev_endpoint_mse=val))
    save_checkpoint(output/"final.pkl",agent,dict(step=steps,protocol_sha256=sha(output/"protocol.json")))
    summary=dict(parameters=parameter_count(agent),train_transitions=len(x),dev_transitions=len(dx),best_step=best_step,
                 best_dev_endpoint_mse_normalized=best,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-start_cpu,
                 gpu=False,cpu_affinity=sorted(resources.os.sched_getaffinity(0)),new_teacher_rollouts=0,
                 best_sha256=sha(output/"best.pkl"),logs=logs)
    (output/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("diagnostics/drone_3d_pipeline_v1/flow")); p.add_argument("--steps",type=int,default=4000)
    p.add_argument("--terminal-data",type=Path);p.add_argument("--width",type=int,default=64)
    a=p.parse_args(); train(a.output,a.steps,a.terminal_data,a.width)
