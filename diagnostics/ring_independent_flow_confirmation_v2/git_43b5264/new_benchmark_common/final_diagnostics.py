"""Frozen-split Stage-I diagnostics; development is the safe default.

``--split test`` is rejected unless ``--frozen-test`` is supplied.  The guard
is enforced before any rollout archive is opened.
"""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import jax
import numpy as np
from .dataset import DATASET_SCHEMA, TRAJECTORY_SCHEMA, JointTransitionDataset
from .evaluation import _ood_distances, teacher_forced_rmse
from .macflow import load_checkpoint, sample_bounded_actions

@dataclass(frozen=True)
class NominalTrajectory:
    rollout_id: str; initial_state: dict; states: np.ndarray; observations: np.ndarray; actions: np.ndarray; metadata: dict

@dataclass(frozen=True)
class Adapter:
    name: str; make_env: Callable[[],Any]; reset: Callable[[Any,dict],None]
    observation: Callable[[Any],np.ndarray]; step: Callable[[Any,np.ndarray],tuple[bool,dict]]
    snapshot: Callable[[Any],dict]; restore: Callable[[Any,dict],None]
    positions: Callable[[dict],np.ndarray]; mode: Callable[[np.ndarray],str]
    max_speed: float

def load_nominal(dataset: str|Path, split: str='dev', *, frozen_test: bool=False):
    if split not in {'train','dev','test'}: raise ValueError('split must be train, dev or test')
    if split=='test' and not frozen_test: raise PermissionError('test requires explicit --frozen-test')
    root=Path(dataset); raw=(root/'manifest.json').read_bytes(); manifest=json.loads(raw)
    if manifest.get('schema')!=DATASET_SCHEMA or not manifest.get('complete'): raise ValueError('bad manifest')
    rows=[x for x in manifest['files'] if x['split']==split and x['source']=='nominal']
    out=[]; opened=[]
    for row in sorted(rows,key=lambda x:x['rollout_id']):
        rel=Path(row['file'])
        if rel.parts[:2]!=('rollouts',split) or rel.is_absolute() or '..' in rel.parts: raise ValueError('split path violation')
        with np.load(root/rel,allow_pickle=False) as z:
            if str(z['schema'].item())!=TRAJECTORY_SCHEMA: raise ValueError('bad archive')
            meta=json.loads(str(z['metadata_json'].item())); initial=json.loads(str(z['initial_state_json'].item()))
            if meta.get('split')!=split or meta.get('source')!='nominal': raise ValueError('archive split/source mismatch')
            out.append(NominalTrajectory(str(row['rollout_id']),initial,z['states'],z['observations'],z['actions'],meta)); opened.append(str(rel))
    return manifest,tuple(out),{'split':split,'frozen_test':frozen_test,'opened_archives':opened,'opened_test_archives':len(opened) if split=='test' else 0,'manifest_sha256':hashlib.sha256(raw).hexdigest()}

def _bound64(local, maximum):
    n=np.linalg.norm(local,axis=-1,keepdims=True); return local*np.minimum(1.,maximum/np.maximum(n,1e-30))

def _make_adapter(kind, config):
    if kind=='four-world':
        from four_way_intersection.environment import Config, FourWayIntersectionEnv
        from four_way_intersection.rollout import crossing_order_signature
        cfg=Config(**config)
        return Adapter(kind,lambda:FourWayIntersectionEnv(cfg),lambda e,s:e.reset(np.asarray(s['positions']),np.asarray(s['velocities'])),lambda e:e.observation(),lambda e,u:(lambda x:(x[2],x[3]))(e.step(_bound64(u,cfg.max_speed))),lambda e:e.augmented_state(),lambda e,s:e.restore_augmented_state(s),lambda s:np.asarray(s['positions']),crossing_order_signature,cfg.max_speed)
    if kind=='ring-local':
        from ring_exchange.environment import LocalFrameConfig, RingExchangeEnv
        from ring_exchange.local_frame import local_actions_to_world, local_observation
        from ring_exchange.expert import circulation_signature
        cfg=LocalFrameConfig(**config)
        def step(e,u):
            o=e.step(local_actions_to_world(_bound64(u,cfg.max_speed),e.positions)); return o[2],o[3]
        return Adapter(kind,lambda:RingExchangeEnv(cfg),lambda e,s:e.reset(np.asarray(s['positions']),velocities=np.asarray(s['velocities']),goals=np.asarray(s['goals'])),lambda e:local_observation(e.positions,e.velocities,e.goals,cfg),step,lambda e:e.augmented_state(),lambda e,s:e.restore_augmented_state(s),lambda s:np.asarray(s['positions']),circulation_signature,cfg.max_speed)
    raise ValueError('scenario must be four-world or ring-local')

def _reference(adapter, trajectory):
    env=adapter.make_env(); adapter.reset(env,trajectory.initial_state); refs=[adapter.snapshot(env)]
    for action in trajectory.actions:
        done,_=adapter.step(env,np.asarray(action)); refs.append(adapter.snapshot(env))
        if done: break
    return refs

def run(adapter, agent, trajectories, train_observations, train_nominal_t0, *, seed=0):
    rows=[]; all_obs=[]; horizons=(1,5,10,25,50,100); krows=[]
    for index,t in enumerate(trajectories):
        env=adapter.make_env(); adapter.reset(env,t.initial_state); key=jax.random.fold_in(jax.random.PRNGKey(seed),index); path=[adapter.positions(adapter.snapshot(env))]; obs=[]; terminal='timeout'; info={}
        for step in range(100000):
            o=adapter.observation(env); obs.append(o); u=np.asarray(sample_bounded_actions(agent,o[None],jax.random.fold_in(key,step))[0],dtype=np.float64); done,info=adapter.step(env,u); path.append(adapter.positions(adapter.snapshot(env))); terminal=str(info.get('termination','running'))
            if done: break
        all_obs.extend(obs); outcome=env.summary(); rows.append({'rollout_id':t.rollout_id,'termination':terminal,'success':bool(outcome.get('collision_free_success',False)),'wall_collision':bool(outcome.get('wall_collision',False)),'obstacle_collision':bool(outcome.get('obstacle_collision',False)),'agent_collision':bool(outcome.get('agent_collision',False)),'timeout':terminal=='timeout','episode_length':int(outcome.get('episode_steps',len(obs))),'failure_location':np.asarray(path[-1]).tolist(),'mode_signature':adapter.mode(np.asarray(path))})
        refs=_reference(adapter,t)
        for k in horizons:
            if len(refs)<=k: continue
            env=adapter.make_env(); adapter.restore(env,refs[0]); collision=False
            for step in range(k):
                u=np.asarray(sample_bounded_actions(agent,adapter.observation(env)[None],jax.random.fold_in(key,100000+step))[0],dtype=np.float64); done,inf=adapter.step(env,u); collision|=bool(inf.get('wall_collision',False) or inf.get('obstacle_collision',False) or inf.get('agent_collision',False))
                if done: break
            krows.append({'K':k,'divergence':float(np.sqrt(np.mean((adapter.positions(adapter.snapshot(env))-adapter.positions(refs[k]))**2))),'collision':collision})
    distances=_ood_distances(np.asarray(all_obs),train_observations) if all_obs else np.empty(0)
    # The support threshold is a diagnostic calibration, not a training step.
    # Bound its cost deterministically on broad recovery corpora: the reference
    # bank inside ``_ood_distances`` is already capped, and an all-transition
    # probe would otherwise make this quadratic-in-practice computation dwarf
    # the actual closed-loop diagnostic.
    train_probe, train_reference = train_observations[1::2], train_observations[::2]
    if len(train_probe) > 8192:
        train_probe = train_probe[np.linspace(0, len(train_probe) - 1, 8192, dtype=int)]
    rollout_threshold=float(np.quantile(_ood_distances(train_probe,train_reference),.99))
    t0_probe,t0_reference=train_nominal_t0[1::2],train_nominal_t0[::2]
    t0_threshold=float(np.quantile(_ood_distances(t0_probe,t0_reference),.99)); off=0
    for row in rows:
        n=row['episode_length']; d=distances[off:off+n]; start=off; off+=n
        t0=float(_ood_distances(np.asarray(all_obs[start:start+1]),t0_reference)[0]) if n else float('nan')
        row['timestep_0_ood']=bool(n and t0>t0_threshold); row['rollout_ood']=bool(np.any(d>rollout_threshold)); row['timestep_0_nn_distance']=t0 if n else None
    n=max(len(rows),1); rate=lambda x:sum(bool(r[x]) for r in rows)/n; ks={str(k):{'count':len(x:=[q for q in krows if q['K']==k]),'mean_divergence':float(np.mean([q['divergence'] for q in x])) if x else None,'collision_rate':float(np.mean([q['collision'] for q in x])) if x else None} for k in horizons}
    return {'aggregate':{'rollouts':len(rows),'full_task_success':rate('success'),'wall_collision':rate('wall_collision'),'obstacle_collision':rate('obstacle_collision'),'agent_collision':rate('agent_collision'),'timeout':rate('timeout'),'mean_episode_length':float(np.mean([r['episode_length'] for r in rows])) if rows else 0.,'timestep_0_ood':rate('timestep_0_ood'),'rollout_ood':rate('rollout_ood'),'timestep_0_ood_threshold_train_nominal_t0_p99':t0_threshold,'rollout_ood_threshold_train_transitions_p99':rollout_threshold,'k_step':ks},'rollouts':rows,'k_rows':krows}

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--scenario',choices=('four-world','ring-local'),required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=('dev','test'),default='dev'); p.add_argument('--frozen-test',action='store_true'); p.add_argument('--seed',type=int,default=0); a=p.parse_args(argv)
 if a.split=='test' and not a.frozen_test: p.error('--split test requires --frozen-test')
 if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
 manifest,trajs,audit=load_nominal(a.dataset,a.split,frozen_test=a.frozen_test); adapter=_make_adapter(a.scenario,manifest['scenario_config']); agent,meta=load_checkpoint(a.checkpoint,expected_environment_fingerprint=manifest['environment_fingerprint']); train=JointTransitionDataset(a.dataset,'train').observations
 _,train_nominal,train_audit=load_nominal(a.dataset,'train'); result=run(adapter,agent,trajs,train,np.asarray([t.observations[0] for t in train_nominal]),seed=a.seed); obs=np.concatenate([t.observations[:-1] for t in trajs]); acts=np.concatenate([t.actions for t in trajs]); result['teacher_forced']=teacher_forced_rmse(agent,obs,acts,seed=a.seed); result['selection_audit']={**audit,'train_nominal_calibration':train_audit}; result['checkpoint_metadata']=meta; a.output.mkdir(parents=True); (a.output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); [ (a.output/f"{r['rollout_id']}.json").write_text(json.dumps(r,indent=2,sort_keys=True)+'\n') for r in result['rollouts'] ]; print(json.dumps(result['aggregate'],indent=2,sort_keys=True))
if __name__=='__main__': main()
