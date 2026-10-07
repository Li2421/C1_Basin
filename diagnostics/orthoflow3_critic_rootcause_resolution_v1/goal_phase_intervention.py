"""Stationary goal-regime controller intervention, not a target-selected fix.

Base t0 is unchanged. Later raw Flow is alt unless all agents are within1m
of goals, then smoothly blends toward second, fully second within0.6m.
Safety/basis/RNG are unchanged. Initial H20 is identical by a speed bound;
all0.30m goal-response probes use second. This is a mechanism test, not a
natural-MLP/cross-scene generalization benchmark.
"""
import argparse,copy,hashlib,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .function_support import OUT as SOURCE,read,write
from . import seed_replication as engine
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid,eta_identity,ROOT as DBROOT
from shared_rollout_db.src.cache_writer import append_journal
OUT=SOURCE.parent/'goal_phase_intervention';EXP='exp_orthoflow3_stationary_goal_phase_intervention_v1'


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Interventionalreadyfrozen')
    states=copy.deepcopy([s for s in read(SOURCE/'states.json') if s['split']=='validation'])
    states.sort(key=lambda s:hashlib.sha256(('c1_goal_phase_source_v1\0'+s['source_group']).encode()).hexdigest());states=states[:8]
    lookup={s['uid']:i for i,s in enumerate(states)}
    for i,s in enumerate(states):s['repair_index']=i
    pairs=copy.deepcopy([p for p in read(SOURCE/'pairs.json') if p['state_uid'] in lookup])
    for p in pairs:p.update(state_index=lookup[p['state_uid']],target_seeds=16)
    source=read(SOURCE/'protocol.json');alt,second=source['profiles'][:2]
    assert (alt['name'],second['name'])==('alt','second')
    program=dict(kind='stationary_physical_goal_regime_flow_blend',t0='registered originalbaseflow',future='(1-w)*alt+w*second',
        w='clip((1.0-max_agent_goal_distance)/0.4,0,1)',alt=alt,second=second,runtime_sha256=sha(__file__),
        base_flow_sha256=source['base_flow_sha256'],safety_eta_horizon_success_unchanged=True)
    write(OUT/'controller_program.json',program);ph=sha(OUT/'controller_program.json')
    with connect() as db,transaction(db):
        ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(alt['controller_uid'],)).fetchone();payload=json.loads(ref['config_json'])
        payload.update(flow_checkpoint_sha256=ph,flow_artifact_kind='frozen_controller_program_manifest',conditioning=payload['conditioning']+'+stationary_goal_blend_1m_06m_v1',controller_program=program,controller_program_sha256=ph)
        cid=uid('ctl',payload)
        assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,ref['scenario_uid'],ph,ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profile=dict(name='goal_phase',path=str(OUT/'controller_program.json'),sha256=ph,controller_uid=cid)
        proto={**source,'experiment_uid':EXP,'experiment':'stationary_goal_phase_mechanism','profiles':[profile],'parent_profile':alt,'second_profile':second,'requested_seed_count':256,
            'state_rule':'First8 sourceVALfamilies by SHA256saltc1_goal_phase_source_v1; no outcome/modelscorefiltering',
            'scope':'Mechanistic controller-function intervention, not newnaturalMLP/newscene/neweta benchmark. Allnewlabels remain diagnostic, never used to train frozenmodels.',
            'target_confirmation_labels_used':False,'generator_modified':False,'goal_models_frozen_sha256':sha(SOURCE.parent/'goal_neighborhood_response/models_frozen.json')}
        write(OUT/'protocol.json',proto);write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'stationary_goal_phase_intervention',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'source_only':True,'training_labels_created':False})))
    req=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
    write(OUT/'planned_rollouts.json',{'requests':req});write(OUT/'planned_parent_reuse.json',{'requests':[{**r,'controller_uid':alt['controller_uid']} for r in req]})
    write(OUT/'preexisting_numerical_exclusions.json',{'count':0,'records':[]})
    # No controller can move faster than0.52m/s: all agents stay >1m away
    # throughout the1s initial prefix, including nominal and eta probes.
    initial_min=min(float(np.linalg.norm(np.array(s['physical']['goals'])-s['physical']['positions'],axis=1).min()) for s in states)
    assert initial_min-20*.05*.52>1.
    write(OUT/'input_identity_proof.json',dict(minimum_initial_agent_goal_distance=initial_min,maximum_H20_displacement=.52,
        identical_h_eta_H20=True,goal_probe_max_distance=.30,goal_probe_uses_second_weight=1.,
        note='Stationary future controller differs only in physicalgoalregime, not time orsuccesslabel. Programidentityisnot acriticinput.'))


def configure():engine.OUT=OUT;engine.EXP=EXP


def run(shard):
    import jax
    from new_benchmark_common.basin_dataset_v1 import TrainingRuntime
    from new_benchmark_common.macflow import load_checkpoint
    from new_benchmark_common.safety_eta3 import ScenarioRuntime
    jax.config.update('jax_enable_x64',False);assert jax.default_backend()=='gpu'
    protocol=read(OUT/'protocol.json');program=read(OUT/'controller_program.json');profile=protocol['profiles'][0]
    assert sha(__file__)==program['runtime_sha256'] and sha(profile['path'])==profile['sha256']
    states=read(OUT/'states.json');rt=TrainingRuntime('ring_exchange',states,parent=True);assert rt.checkpoint_sha==protocol['base_flow_sha256'];base=rt.agent
    for p in (protocol['parent_profile'],protocol['second_profile']):assert sha(p['path'])==p['sha256']
    alt,_=load_checkpoint(protocol['parent_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint'])
    second,_=load_checkpoint(protocol['second_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint'])
    track={}
    def flow(env,key):
        t=track['t'];track['t']+=1
        if t==0:
            rt.agent=base;return ScenarioRuntime.flow_world(rt,env,key)
        distance=float(np.linalg.norm(env.goals-env.positions,axis=1).max());w=float(np.clip((1.-distance)/.4,0.,1.))
        rt.agent=alt;a=ScenarioRuntime.flow_world(rt,env,key)
        if w==0.:return a
        track['active']+=1;track['max_weight']=max(track['max_weight'],w)
        if track['first_active'] is None:track['first_active']=t
        rt.agent=second;b=ScenarioRuntime.flow_world(rt,env,key)
        return b if w==1. else (1.-w)*a+w*b
    rt.flow_world=flow
    pre=read(OUT/'cache_preflight.json');missing={(r['state_uid'],r['eta_uid'],sk) for r in pre['details'] for sk in r['missing_seeds']}
    jobs=[(p,k) for p in read(OUT/'pairs.json') for k in range(16) if (p['state_uid'],p['eta_uid'],canonical({'future_index':k})) in missing]
    done=set()
    for path in (DBROOT/'journals'/EXP).glob(f'{native.REVISION}_goalphase_shard{shard}_*.jsonl'):
      for line in path.read_text().splitlines():
        r=json.loads(line)['record'];done.add((r['state_uid'],eta_identity(r['eta'])[0],r['future_index']))
    for i,(p,k) in enumerate(jobs):
        if i%2!=shard or (p['state_uid'],p['eta_uid'],k) in done:continue
        track.update(t=0,active=0,max_weight=0.,first_active=None)
        result=rt.rollout(states[p['state_index']],np.array(p['eta']),k,'orthoflow3')
        result.update(controller_uid=profile['controller_uid'],experiment_uid=EXP,flow_checkpoint_sha256=profile['sha256'],source_split=p['split'],committed_t0_flow_sha256=protocol['base_flow_sha256'],flow_sampler_precision='float32',jax_enable_x64=False,execution_revision=native.REVISION,
            controller_program_sha256=profile['sha256'],goal_blend_active_steps=track['active'],goal_blend_max_weight=track['max_weight'],goal_blend_first_active_step=track['first_active'])
        append_journal([{'schema':'controller_training_repair_v1','record':result}],EXP,f'{native.REVISION}_goalphase_shard{shard}')
        if i%32==shard:print(dict(shard=shard,global_index=i,planned=len(jobs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','merge'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.index)
    else:
        from . import postprocess
        configure();postprocess.run('goal_phase')
