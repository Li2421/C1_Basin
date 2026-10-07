"""Stationary physical-regime counterexample for H20/H80+goal fingerprints.

No time gate, scene ID, outcome-dependent controller selection or safety change.
All current input-query domains use alt exactly; only intermediate task states
can mix in second. This tests a controller-program class, not natural-MLP alias.
"""
import argparse,copy,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .goal_phase_intervention import OUT as PRIOR
from .function_support import read,write,sha
from . import seed_replication as engine
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid,eta_identity,ROOT as DBROOT
from shared_rollout_db.src.cache_writer import append_journal
OUT=PRIOR.parent/'middle_phase_alias';EXP='exp_c1_stationary_middle_phase_context_alias_v1'


def weight(distance):
    return float(np.clip((2.5-distance)/.3,0.,1.)*np.clip((distance-1.0)/.3,0.,1.))


def prepare():
    assert not (OUT/'protocol.json').exists(),'Frozen design exists'
    base=read(PRIOR/'protocol.json');states=read(PRIOR/'states.json');pairs=read(PRIOR/'pairs.json');early=base['parent_profile'];middle=base['second_profile']
    minimum=min(float(np.linalg.norm(np.array(s['physical']['goals'])-s['physical']['positions'],axis=1).min()) for s in states)
    assert minimum-80*.05*.52>2.5 and weight(.3)==0.
    program=dict(kind='stationary_middle_goal_distance_regime_flow_blend',t0='registered originalbaseflow',future='(1-w)*alt+w*second',
        w='clip((2.5-max_agent_goal_distance)/0.3,0,1)*clip((max_agent_goal_distance-1.0)/0.3,0,1)',
        alt=early,second=middle,runtime_sha256=sha(__file__),base_flow_sha256=base['base_flow_sha256'],safety_eta_horizon_success_unchanged=True)
    write(OUT/'controller_program.json',program);ph=sha(OUT/'controller_program.json')
    with connect() as db,transaction(db):
        ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(early['controller_uid'],)).fetchone();payload=json.loads(ref['config_json'])
        payload.update(flow_checkpoint_sha256=ph,flow_artifact_kind='frozen_controller_program_manifest',conditioning=payload['conditioning']+'+stationary_middle_goal_distance_blend_v1',controller_program=program,controller_program_sha256=ph)
        cid=uid('ctl',payload);assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,ref['scenario_uid'],ph,ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profile=dict(name='middle_phase',path=str(OUT/'controller_program.json'),sha256=ph,controller_uid=cid)
        proto={**base,'experiment_uid':EXP,'experiment':'stationary_middle_phase_context_alias','profiles':[profile],
            'parent_profile':early,'second_profile':middle,'requested_seed_count':256,'state_rule':'Reuse8outcome-blindsourceVALfamilies from preregisteredgoal-phasecausaltest; no newoutcomeselection',
            'scope':'Mechanisticstationarycontrollerprogramcounterexample, not claim about alias among naturalMLPcheckpoints.',
            'target_labels_used':False,'new_labels_enter_training':False,'goal_models_frozen_sha256':sha(PRIOR.parent/'goal_neighborhood_response/models_frozen.json')}
        write(OUT/'protocol.json',proto);write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'stationary_middle_phase_context_alias',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'source_only':True,'new_labels_enter_training':False})))
    req=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
    write(OUT/'planned_rollouts.json',{'requests':req});write(OUT/'preexisting_numerical_exclusions.json',dict(count=0,records=[]))
    write(OUT/'input_identity_proof.json',dict(min_initial_agent_goal_distance=minimum,
        max_H80_agent_displacement=80*.05*.52,remaining_distance_lower_bound=minimum-80*.05*.52,
        controller_gate_zero_above=2.5,goal_probes_distance=.3,controller_gate_zero_below=1.0,
        h_equal=True,H20_equal=True,H80_equal=True,agent_response_equal=True,goal_response_equal=True,
        proof='Plant speed bound keeps everyagent >2.5m fromgoal during allinitial80stepquerytrajectories, independentofeta/RNG. Allstaticgoalqueriesare0.30m. On both domains functionw is exactlyzero andno secondFlowcall occurs. Originalt0Flow remainsunchanged.',
        limitation='This is a sufficiencytest over allowedstationarycontrollerprograms, not proof of identicalsummaries among naturalneuralcheckpoints.'))


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
    alt,_=load_checkpoint(protocol['parent_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint']);second,_=load_checkpoint(protocol['second_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint']);track={}
    def flow(env,key):
        t=track['t'];track['t']+=1
        if t==0:rt.agent=base;return ScenarioRuntime.flow_world(rt,env,key)
        distance=float(np.linalg.norm(env.goals-env.positions,axis=1).max());w=weight(distance)
        rt.agent=alt;a=ScenarioRuntime.flow_world(rt,env,key)
        if w==0.:return a
        assert t>80,'Unexpectedearlyactivation invalidates identity proof'
        track['active']+=1;track['max_weight']=max(track['max_weight'],w)
        if track['first_active'] is None:track['first_active']=t
        rt.agent=second;b=ScenarioRuntime.flow_world(rt,env,key);return b if w==1. else (1.-w)*a+w*b
    rt.flow_world=flow
    pre=read(OUT/'cache_preflight.json');missing={(r['state_uid'],r['eta_uid'],sk) for r in pre['details'] for sk in r['missing_seeds']}
    jobs=[(p,k) for p in read(OUT/'pairs.json') for k in range(16) if (p['state_uid'],p['eta_uid'],canonical({'future_index':k})) in missing];done=set()
    for path in (DBROOT/'journals'/EXP).glob(f'{native.REVISION}_middlephase_shard{shard}_*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)['record'];done.add((r['state_uid'],eta_identity(r['eta'])[0],r['future_index']))
    for i,(p,k) in enumerate(jobs):
        if i%2!=shard or (p['state_uid'],p['eta_uid'],k) in done:continue
        track.update(t=0,active=0,max_weight=0.,first_active=None);r=rt.rollout(states[p['state_index']],np.array(p['eta']),k,'orthoflow3')
        r.update(controller_uid=profile['controller_uid'],experiment_uid=EXP,flow_checkpoint_sha256=profile['sha256'],source_split=p['split'],committed_t0_flow_sha256=protocol['base_flow_sha256'],flow_sampler_precision='float32',jax_enable_x64=False,execution_revision=native.REVISION,
            controller_program_sha256=profile['sha256'],middle_active_steps=track['active'],middle_first_active_step=track['first_active'],middle_max_weight=track['max_weight'])
        append_journal([{'schema':'controller_training_repair_v1','record':r}],EXP,f'{native.REVISION}_middlephase_shard{shard}')
        if i%32==shard:print(dict(shard=shard,index=i,requested=len(jobs)),flush=True)


def merge():
    from . import postprocess
    engine.OUT=OUT;engine.EXP=EXP;postprocess.run('middle_phase_alias')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','merge'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='run':run(a.index)
    else:globals()[a.action]()
