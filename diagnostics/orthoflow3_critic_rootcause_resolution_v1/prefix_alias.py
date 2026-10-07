"""Exact H20-observational alias under a frozen delayed controller switch.

This is a posthoc source-only identifiability counterexample, not a natural
unseen-controller benchmark and not a training set. A matches the registered
base-t0/alt-future policy; B is identical for20 actions, then switches to a
second existing frozen policy. Safety, eta basis, horizon and RNG stay fixed.
"""
import argparse,copy,json,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import numpy as np
from . import seed_replication as engine
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid,eta_identity,ROOT as DBROOT
from shared_rollout_db.src.cache_writer import append_journal
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
OUT=Path(__file__).resolve().parent/'prefix_alias';EXP='exp_orthoflow3_rootcause_H20_prefix_alias_v1'
read=engine.read;write=engine.write

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Prefix intervention frozen')
    states=copy.deepcopy(read(engine.OUT/'states.json')[:8]);lookup={s['uid']:i for i,s in enumerate(states)}
    for i,s in enumerate(states):s['repair_index']=i
    pairs=copy.deepcopy([p for p in read(engine.OUT/'pairs.json') if p['state_uid'] in lookup])
    for p in pairs:p.update(state_index=lookup[p['state_uid']],target_seeds=16)
    source=read(engine.OUT/'protocol.json');alt,second=source['profiles'];assert alt['name']=='alt'
    program=dict(kind='frozen_finite_state_controller_program',switch_action_index=20,
        actions_0='committed original base',actions_1_through_19=alt,actions_20_onward=second,
        runtime_sha256=sha(__file__),base_flow_sha256=source['base_flow_sha256'],
        horizon_and_safety_unchanged=True,conditional_on_success=False)
    write(OUT/'controller_program.json',program);program_hash=sha(OUT/'controller_program.json')
    with connect() as db,transaction(db):
        row=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(alt['controller_uid'],)).fetchone();payload=json.loads(row['config_json'])
        payload.update(flow_checkpoint_sha256=program_hash,flow_artifact_kind='frozen_controller_program_manifest',
            conditioning=payload['conditioning']+'+alt_through_action19_then_second_from20_v1',controller_program=program,
            controller_program_sha256=program_hash)
        cid=uid('ctl',payload)
        db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,row['scenario_uid'],program_hash,row['orthoflow3_sha256'],row['safety_config_hash'],row['horizon'],row['dt'],row['success_semantics_version'],payload['conditioning'],row['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profile=dict(name='prefix',path=str(OUT/'controller_program.json'),sha256=program_hash,controller_uid=cid)
        proto={**source,'experiment_uid':EXP,'experiment':'exact_H20_prefix_alias_source_diagnostic','profiles':[profile],
            'requested_seed_count':256,'parent_profile':alt,'second_profile':second,'switch_action_index':20,
            'state_rule':'First8 of earlier outcome-blind SHA256 source-family replication panel; no new outcome or critic score selection',
            'limitation':'Counterexample for frozen controller programs with identical H20 response, not proof that natural stationary v7/v9 assets have identical context',
            'target_labels_used':False,'generator_modified':False,'training_labels_created':False}
        write(OUT/'protocol.json',proto);write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'exact_H20_observational_alias_diagnostic',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'source_only':True,'frozen_delayed_switch':20})))
    requests=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
    reuse=[{**r,'controller_uid':alt['controller_uid']} for r in requests]
    write(OUT/'planned_rollouts.json',{'requests':requests});write(OUT/'planned_parent_reuse.json',{'requests':reuse})
    write(OUT/'preexisting_numerical_exclusions.json',{'count':0,'records':[]})

def configure():engine.OUT=OUT;engine.EXP=EXP

def run(shard):
    import jax
    from new_benchmark_common.basin_dataset_v1 import TrainingRuntime
    from new_benchmark_common.macflow import load_checkpoint
    from new_benchmark_common.safety_eta3 import ScenarioRuntime
    jax.config.update('jax_enable_x64',False);assert jax.default_backend()=='gpu'
    protocol=read(OUT/'protocol.json');program=read(OUT/'controller_program.json');profile=protocol['profiles'][0]
    assert sha(__file__)==program['runtime_sha256'];assert sha(profile['path'])==profile['sha256']
    states=read(OUT/'states.json');rt=TrainingRuntime('ring_exchange',states,parent=True)
    assert rt.checkpoint_sha==protocol['base_flow_sha256']
    base=rt.agent
    for p in (protocol['parent_profile'],protocol['second_profile']):assert sha(p['path'])==p['sha256']
    alt,_=load_checkpoint(protocol['parent_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint'])
    second,_=load_checkpoint(protocol['second_profile']['path'],expected_environment_fingerprint=protocol['environment_fingerprint'])
    count={'t':0}
    def flow(env,key):
        t=count['t'];rt.agent=base if t==0 else alt if t<20 else second;count['t']+=1
        return ScenarioRuntime.flow_world(rt,env,key)
    rt.flow_world=flow
    pre=read(OUT/'cache_preflight.json');missing={(r['state_uid'],r['eta_uid'],sk) for r in pre['details'] for sk in r['missing_seeds']}
    jobs=[(p,k) for p in read(OUT/'pairs.json') for k in range(16) if (p['state_uid'],p['eta_uid'],canonical({'future_index':k})) in missing]
    done=set()
    for file in (DBROOT/'journals'/EXP).glob(f'{native.REVISION}_prefix_shard{shard}_*.jsonl'):
        for line in file.read_text().splitlines():
            r=json.loads(line)['record'];done.add((r['state_uid'],eta_identity(r['eta'])[0],r['future_index']))
    for i,(p,k) in enumerate(jobs):
        if i%2!=shard or (p['state_uid'],p['eta_uid'],k) in done:continue
        count['t']=0;result=rt.rollout(states[p['state_index']],np.array(p['eta']),k,'orthoflow3')
        result.update(controller_uid=profile['controller_uid'],experiment_uid=EXP,flow_checkpoint_sha256=profile['sha256'],
            source_split=p['split'],committed_t0_flow_sha256=protocol['base_flow_sha256'],flow_sampler_precision='float32',jax_enable_x64=False,
            execution_revision=native.REVISION,controller_program_sha256=profile['sha256'],switch_action_index=20)
        append_journal([{'schema':'controller_training_repair_v1','record':result}],EXP,f'{native.REVISION}_prefix_shard{shard}')
        if i%32==shard:print(dict(shard=shard,global_index=i,planned=len(jobs)),flush=True)
    write(OUT/f'prefix_identity_shard{shard}.json',dict(
        same_physical_state=True,same_eta=True,same_seed_namespace=True,same_base_action0=True,
        same_frozen_Flow_actions1_to19=True,first_different_action_index=20,
        H20_includes_actions=list(range(20)),nominal_and_eta_conditioned_H20_identical_by_construction=True,
        note='Prefix equality holds for every physical probe state, eta and RNG seed, not only the cached two context roots. Controller program identity is deliberately NOT input to the current critic.'))

def analyze():
    from scipy.stats import binomtest
    from statsmodels.stats.multitest import multipletests
    p=read(OUT/'protocol.json');rows=[]
    with connect(True) as db:
      for pair in read(OUT/'pairs.json'):
        records=[]
        for c in (p['parent_profile'],p['profiles'][0]):
            rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],c['controller_uid']))}
            records.append([rr[canonical({'future_index':k})] for k in range(16)])
        comparable=[(a,b) for a,b in zip(*records) if not a['numerical_failure'] and not b['numerical_failure']]
        a=np.array([r[0]['success'] for r in comparable]);b=np.array([r[1]['success'] for r in comparable]);rescue=int(((a==0)&(b==1)).sum());brk=int(((a==1)&(b==0)).sum())
        rows.append(dict(state_uid=pair['state_uid'],eta_index=pair['eta_index'],paired_valid=len(a),
            parent_success=sum(r['success'] for r in records[0] if not r['numerical_failure']),
            prefix_success=sum(r['success'] for r in records[1] if not r['numerical_failure']),
            parent_numerical=sum(r['numerical_failure'] for r in records[0]),prefix_numerical=sum(r['numerical_failure'] for r in records[1]),
            paired_Q_delta=float((b-a).mean()),rescue=rescue,breaks=brk,
            McNemar_exact_p=float(binomtest(rescue,rescue+brk,.5).pvalue) if rescue+brk else 1.))
    adjusted=multipletests([r['McNemar_exact_p'] for r in rows],method='holm')[1]
    for r,a in zip(rows,adjusted):r['Holm_p']=float(a)
    csvwrite(OUT/'paired_controller_Q.csv',rows)
    write(OUT/'identifiability_result.json',dict(cells=len(rows),significant_Holm=sum(r['Holm_p']<.05 for r in rows),
        mean_abs_Q_change=float(np.mean([abs(r['paired_Q_delta']) for r in rows])),
        robust_to_complete_failure=sum(r['parent_success']>=15 and r['prefix_success']==0 and r['prefix_numerical']==0 for r in rows),
        identical_H20_inputs=True,source_only_posthoc_controller_program_intervention=True,
        implication='If Q differs, H20 behavior fingerprint cannot be a sufficient statistic for this enlarged controller-program class. This does not by itself attribute natural-v7/v9 model errors to aliasing.',
        new_controller_labels_used_for_model_selection=False))
    print(json.dumps(read(OUT/'identifiability_result.json'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','analyze'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='run':run(a.index)
    else:globals()[a.action]()
