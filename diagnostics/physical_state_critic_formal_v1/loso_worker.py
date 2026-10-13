"""Frozen original FF rollouts for missing LOSO Q labels, journal-only workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,'/home/zhihan/research/02_C1_Toy_GiveWay')

import jax
import jax.numpy as jnp
import numpy as np

from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import eta_identity
HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_formal_physical_state_critic_loso_20261008_v1'
SCENES=('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    temp.replace(path)


def items(mode):
    assert mode in ('source','test')
    result=[]
    for scene in SCENES:
        for split in (('train','validation') if mode=='source' else ('test',)):
            panel=HERE/'loso'/scene/split
            n=len(read(panel/'missing_cases.json'))
            for start in range(0,n,128):
                result.append(dict(scene=scene,split=split,start=start,end=min(start+128,n)))
    return result


def freeze_items(mode):
    work=items(mode)
    path=HERE/f'loso_{mode}_work_items.json'
    if path.exists():assert read(path)==work
    else:write(path,work)
    print(json.dumps(dict(mode=mode,shards=len(work),slots=sum(r['end']-r['start'] for r in work))))


class ToyRuntime:
    def __init__(self):
        from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
        from shared_control.basis_families import get_basis_family
        from single_integrator.cbf import CBFConfig, barrier_constraints
        from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
        from single_integrator.evaluate import load_policy
        from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.run_rollouts import label
        jax.config.update('jax_enable_x64',True)
        wide=read(HERE.parents[1]/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json')
        self.config=Config(**wide['environment']);self.cbf=CBFConfig(**wide['cbf'])
        self.policy,provenance=load_policy('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
        assert provenance['evaluation_environment']==wide['environment']
        self.sample=jax.jit(lambda obs,key:self.policy.sample_actions(obs[None],seed=key)[0])
        self.basis=get_basis_family('orthoflow3')
        self.builder=StartupAwareFeatureBuilder()
        self.project=project_velocity_with_retry
        self.constraints=barrier_constraints
        self.Env=GiveWayEnv
        self.bound=bounded_nominal
        self.label=label

    def rollout(self,row,eta,future_index,safety):
        env=self.Env(self.config);env.reset(np.asarray(row['physical']['positions'],np.float64))
        initial=jax.random.fold_in(jax.random.PRNGKey(42),int(row['proposal']['rollout_id']))
        future=jax.random.fold_in(jax.random.PRNGKey(42+future_index),int(row['proposal']['rollout_id']))
        error=None;jdef=0.;steps=0
        for step in range(self.config.max_steps):
            try:
                obs=np.asarray(env.observation(),np.float32)
                key=jax.random.fold_in(initial if step==0 else future,step)
                flow=self.bound(np.asarray(self.sample(jnp.asarray(obs),key),np.float64),self.config.max_speed)
                A,lower,_=self.constraints(env.snapshot(),self.cbf)
                safe,_,_,_=self.project(flow,A,lower,self.config.max_speed,self.cbf)
                if step==0:
                    feature,_=self.builder.build(env,{'u_flow':flow,'u_safe':safe},self.config,self.cbf)
                    fingerprint=hashlib.sha256(np.asarray(feature,np.float32).astype(np.float64).tobytes()).hexdigest()
                    assert fingerprint==row['proposal']['conditioning_sha256'], 'Toy frozen true-t0 conditioning mismatch'
                correction=(np.zeros_like(safe) if safety else
                            self.basis.compute(env.positions,env.goals,safe,self.config.max_speed).correction(eta))
                executed,_,_,_=self.project(safe+correction,A,lower,self.config.max_speed,self.cbf)
                if np.min(A@executed.reshape(4)-lower)<-self.cbf.feasibility_tol:
                    raise RuntimeError('Hard-safety projection violated')
                jdef+=self.config.dt*float(np.sum((executed-safe)**2))
                env.step(executed);steps+=1
                if env.done:break
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc),'step':step};break
        outcome,_=self.label(env,error)
        return dict(state_uid=row['proposal']['state_uid'],eta=np.asarray(eta,float).tolist(),
                    future_index=future_index,outcome=outcome,termination=outcome,
                    success=outcome=='success',deadlock=outcome=='deadlock',
                    timeout=outcome=='timeout',collision=outcome=='collision',
                    numerical_failure=outcome=='numerical_failure',episode_length=env.step_count,
                    execution_error=error,J_def=jdef)


class DBRuntime:
    def __init__(self):
        from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
        from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old_db
        from diagnostics.orthoflow3_db_shared_mode_transfer_v1 import run_db
        self.frozen,self.old_db,self.run_db=frozen,old_db,run_db
        self.dataset=old_db.FlowBC4ADataset(next(iter(frozen.DOUBLE_POOLS.values())),'all')
        self.policy,_=old_db.load_checkpoint(old_db.CHECKPOINT,self.dataset.environment_fingerprint)
        for path,expected in old_db.EXPECTED.items():assert old_db.sha(path)==expected

    def rollout(self,row,eta,future_index,safety):
        frozen,run_db=self.frozen,self.run_db
        state=row['proposal'];p=row['physical'];token=state['state_token']
        episode=SimpleNamespace(initial_positions=np.asarray(p['positions'],float),
                                initial_velocities=np.asarray(p['velocities'],float),
                                regime=state['regime'])
        current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(state['current_root']),token),0)
        future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(frozen.DOUBLE_FUTURE_ROOT),token),future_index)
        policy=self.policy
        class Wrapped:
            def __init__(self):self.step=0
            def sample_actions(self,obs,unused):
                key=current if self.step==0 else jax.random.fold_in(future,self.step)
                self.step+=1
                return policy.sample_actions(obs,key)
        job=dict(job_id=f"{state['state_alias']}:{future_index}:{eta_identity(eta)[0]}",
                 stage=EXPERIMENT,representation='P1-OrthoFlow3',theta=np.asarray(eta,float).tolist(),
                 seed=frozen.DOUBLE_FUTURE_ROOT,rollout_id=token,
                 episode_id=state['state_alias'],family_id=row['source']['metadata']['original_sampler_spec']['family_id'])
        raw=run_db.run_one_with_jdef(Wrapped(),self.dataset,episode,job,3.303687238760696)
        valid=bool(raw.get('scientific_outcome_valid',False))
        term=str(raw.get('termination','numerical_failure'))
        collision=bool(raw.get('wall_collision') or raw.get('agent_collision'))
        return dict(state_uid=state['state_uid'],eta=np.asarray(eta,float).tolist(),
                    future_index=future_index,outcome=raw.get('outcome',term),termination=term,
                    success=bool(raw.get('success')) and valid,deadlock='deadlock' in term,
                    timeout=term=='timeout',collision=collision,numerical_failure=not valid,
                    episode_length=int(raw.get('episode_steps',0)),execution_error=None if valid else raw,
                    J_def=raw.get('J_def'),wall_collision=bool(raw.get('wall_collision')),
                    agent_collision=bool(raw.get('agent_collision')))


def make_runtime(scene):
    if scene=='toy_give_way':return ToyRuntime()
    if scene=='double_bottleneck':return DBRuntime()
    from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
    return frozen.FrozenNewRuntime(scene)


def run(index,mode):
    assert mode in ('source','test')
    if mode=='test':
        frozen_models=read(HERE/'loso_models_frozen.json')
        assert frozen_models['test_labels_used'] is False
        assert (HERE/'loso_test_predictions.npz').exists()
    item=read(HERE/f'loso_{mode}_work_items.json')[index]
    scene,split=item['scene'],item['split']
    panel=HERE/'loso'/scene/split
    cases=read(panel/'missing_cases.json')[item['start']:item['end']]
    proposals=read(panel/'proposals.json')['states']
    sources={s['alias']:s for s in read(HERE/f'states_{scene}.json')['states']}
    ctl=read(HERE/'loso_protocol.json')['controllers'][scene]
    runtime=make_runtime(scene)
    new=old=0;records=[]
    for c in cases:
        assert c['scene']==scene and c['split']==split
        proposal=proposals[c['state_index']]
        source=sources[c['state_alias']]
        assert proposal['state_uid']==c['state_uid'] and proposal['state_alias']==source['alias']
        assert eta_identity(c['eta'])[0]==c['eta_uid']
        safety=c['eta_index']==16
        assert c['controller_uid']==ctl['safety_uid' if safety else 'intervention_uid']
        record_path=panel/'raw'/(c['rollout_uid']+'.json')
        if record_path.exists():
            rec=read(record_path);assert rec['case']==c;old+=1
        else:
            row=dict(proposal=proposal,source=source,physical=source['physical'])
            if scene in ('toy_give_way','double_bottleneck'):
                raw=runtime.rollout(row,np.asarray(c['eta'],float),c['future_index'],safety)
            else:
                state=dict(uid=c['state_uid'],alias=c['state_alias'],
                           index=c['state_index'],physical=source['physical'])
                raw=runtime.rollout(state,np.asarray(c['eta'],float),c['future_index'],
                                    'hard_safety_q16' if safety else 'orthoflow3')
            assert raw['state_uid']==c['state_uid'] and raw['future_index']==c['future_index']
            assert eta_identity(raw['eta'])[0]==c['eta_uid']
            if scene not in ('toy_give_way','double_bottleneck'):
                assert raw['controller_uid']==c['controller_uid']
            numerical=int(raw['numerical_failure'])
            success=int(raw['success']);collision=int(raw['collision'])
            assert not(success and (numerical or collision))
            rec=dict(schema='formal_q_loso_seed_v1',experiment_uid=EXPERIMENT,
                     case=c,success=success,deadlock=int(raw['deadlock']),
                     timeout=int(raw['timeout']),collision=collision,
                     numerical_failure=numerical,episode_length=raw['episode_length'],
                     outcome=raw['outcome'],raw=raw,
                     protocol_sha256=sha(HERE/'loso_protocol.json'))
            write(record_path,rec);new+=1
        records.append(rec)
    if records:append_journal(records,EXPERIMENT,f'{mode}_{index:04d}')
    receipt=dict(mode=mode,index=index,scene=scene,split=split,cases=len(cases),new=new,
                 recovered=old,complete=True,protocol_sha256=sha(HERE/'loso_protocol.json'))
    write(HERE/'loso'/'receipts'/f'{mode}_{index:04d}.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze_items','run'))
    p.add_argument('--mode',choices=('source','test'),default='source')
    p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='freeze_items':freeze_items(a.mode)
    else:
        assert a.index is not None
        run(a.index,a.mode)
