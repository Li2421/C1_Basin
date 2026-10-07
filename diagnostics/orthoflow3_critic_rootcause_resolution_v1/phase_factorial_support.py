"""Crossed stationary early/goal controller programs, cache-first and immutable physics."""
import argparse, copy, json
from pathlib import Path
import numpy as np
from .function_support import OUT as SOURCE, read, write
from . import goal_phase_intervention as runtime
from . import seed_replication as engine
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
from shared_rollout_db.src.rollout_db import connect, transaction, canonical, uid
from shared_rollout_db.src.planner import preflight

OUT=SOURCE.parent/'phase_factorial_support'
RULE=SOURCE.parent/'phase_factorial_protocol.json'
NAMES=('alt_to_second','second_to_alt')


def prepare():
    assert not (OUT/'protocol.json').exists(),'Frozen experiment exists'
    source=read(SOURCE/'protocol.json');old=read(runtime.OUT/'protocol.json')
    states=read(SOURCE/'states.json');pairs=read(SOURCE/'pairs.json')
    initial_min=min(float(np.linalg.norm(np.array(s['physical']['goals'])-s['physical']['positions'],axis=1).min()) for s in states)
    assert initial_min-80*.05*.52>1.
    allreq=[];profiles=[]
    for ci,name in enumerate(NAMES):
        dest=OUT/name;experiment='exp_c1_phase_factorial_'+name+'_v1'
        early,late=(source['profiles'][i] for i in ((0,1) if ci==0 else (1,0)))
        if ci==0:
            profile=copy.deepcopy(old['profiles'][0]);program=read(profile['path'])
        else:
            program=dict(kind='stationary_physical_goal_regime_flow_blend',t0='registered originalbaseflow',future='(1-w)*alt+w*second',
                w='clip((1.0-max_agent_goal_distance)/0.4,0,1)',alt=early,second=late,
                runtime_sha256=sha(runtime.__file__),base_flow_sha256=source['base_flow_sha256'],safety_eta_horizon_success_unchanged=True)
            write(dest/'controller_program.json',program);ph=sha(dest/'controller_program.json')
            with connect() as db,transaction(db):
                ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(early['controller_uid'],)).fetchone();payload=json.loads(ref['config_json'])
                payload.update(flow_checkpoint_sha256=ph,flow_artifact_kind='frozen_controller_program_manifest',conditioning=payload['conditioning']+'+stationary_goal_blend_1m_06m_v1',controller_program=program,controller_program_sha256=ph)
                cid=uid('ctl',payload)
                assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
                db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (cid,ref['scenario_uid'],ph,ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
            profile=dict(name='goal_phase',path=str(dest/'controller_program.json'),sha256=ph,controller_uid=cid)
        # Runtime always checks this file, even when the registered artifact is reused.
        write(dest/'controller_program.json',program)
        assert sha(dest/'controller_program.json')==profile['sha256']
        proto={**source,'experiment_uid':experiment,'experiment':name,'profiles':[profile],
            'parent_profile':early,'second_profile':late,'requested_seed_count':sum(p['target_seeds'] for p in pairs),
            'rule_sha256':sha(RULE),'source_driver_sha256':sha(__file__),'intervention_runtime_sha256':sha(runtime.__file__),
            'target_labels_used':False,'generator_modified':False}
        write(dest/'protocol.json',proto);write(dest/'states.json',states);write(dest/'pairs.json',pairs)
        req=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(p['target_seeds'])]) for p in pairs]
        write(dest/'planned_rollouts.json',{'requests':req});allreq.extend(req)
        with connect() as db,transaction(db):
            db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                (experiment,name,str(dest),sha(dest/'protocol.json'),sha(__file__),canonical({'source_only':True,'target_labels_used':False})))
        profiles.append({**profile,'factorial_name':name,'early_index':ci,'late_index':1-ci})
    write(OUT/'protocol.json',dict(rule=read(RULE),rule_sha256=sha(RULE),profiles=profiles,source_protocol_sha256=sha(SOURCE/'protocol.json'),
        initial_min_goal_distance=initial_min,maximum_H80_displacement=2.08,initial_H20_H80_exact_reuse_proof=True,new_requested=2048))
    write(OUT/'planned_rollouts.json',{'requests':allreq})


def filter_existing():
    totals=dict(requested=0,exact_reuse=0,partial_reuse=0,aggregate_reuse=0,truly_new=0,preexisting_numerical=0)
    for name in NAMES:
        dest=OUT/name;doc=read(dest/'cache_preflight.json');execution=copy.deepcopy(doc);excluded=[]
        with connect(True) as db:
            for item in execution['details']:
                keep=[]
                for sk in item['missing_seeds']:
                    rr=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',(item['state_uid'],item['eta_uid'],item['controller_uid'],sk)).fetchone()
                    if rr:
                        assert rr['numerical_failure'] and not rr['conflict_quarantined'] and rr['compatibility_quality']=='EXACT_REUSE'
                        excluded.append(rr['rollout_uid'])
                    else:keep.append(sk)
                item['missing_seeds']=keep
        execution['summary']['genuinely_missing']-=len(excluded)
        write(dest/'execution_preflight.json',execution);write(dest/'preexisting_numerical_exclusions.json',dict(count=len(excluded),rollout_uids=excluded))
        for a,b in (('requested','total_requested'),('exact_reuse','exact_reusable'),('partial_reuse','partial_reusable'),('aggregate_reuse','aggregate_reusable')):totals[a]+=doc['summary'][b]
        totals['truly_new']+=execution['summary']['genuinely_missing'];totals['preexisting_numerical']+=len(excluded)
    write(OUT/'execution_preflight_summary.json',totals);print(totals)


def run(index):
    name=NAMES[index//2];dest=OUT/name
    assert sha(runtime.__file__)==read(dest/'protocol.json')['intervention_runtime_sha256']
    runtime.OUT=dest;runtime.EXP=read(dest/'protocol.json')['experiment_uid']
    original=runtime.read
    def cached(path):
        return original(dest/'execution_preflight.json') if Path(path)==dest/'cache_preflight.json' else original(path)
    runtime.read=cached
    runtime.run(index%2)


def merge():
    from . import postprocess
    for name in NAMES:
        engine.OUT=OUT/name;engine.EXP=read(engine.OUT/'protocol.json')['experiment_uid'];postprocess.run('phase_factorial')


def dataset():
    source=dict(np.load(SOURCE/'dataset.npz'));pairs=read(SOURCE/'pairs.json');profiles=read(OUT/'protocol.json')['profiles'];ss=[];ff=[];nn=[];keys=[]
    with connect(True) as db:
        for c in profiles:
            s=[];f=[];num=[]
            for p in pairs:
                rows={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
                rr=[rows[canonical({'future_index':k})] for k in range(p['target_seeds'])]
                assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
                valid=[r for r in rr if not r['numerical_failure']];s.append(sum(r['success'] for r in valid));f.append(len(valid)-s[-1]);num.append(len(rr)-len(valid))
                keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rr]))
            ss.append(s);ff.append(f);nn.append(num)
    for key,v in (('success',ss),('failure',ff),('numerical',nn),('standard_success',ss),('standard_failure',ff)):
        source[key]=np.concatenate([source[key],np.array(v,np.float32)])
    for key in ('context','agent_response','valid'):source[key]=np.concatenate([source[key],source[key][[0,1]]])
    source['goal_response']=np.array([np.load(SOURCE.parent/'goal_neighborhood_response'/f'inputs_{c["name"]}.npz')['goal_response'] for c in read(SOURCE/'protocol.json')['profiles']])
    source['goal_response']=np.concatenate([source['goal_response'],source['goal_response'][[1,0]]])
    assert source['valid'].all() and ((source['success']+source['failure'])>0).all()
    assert not (OUT/'dataset.npz').exists(),'Frozen dataset exists'
    np.savez_compressed(OUT/'dataset.npz',**source);write(OUT/'intervention_DB_keys.json',keys)
    write(OUT/'dataset_audit.json',dict(conditions=14,TRAIN_families=64,VAL_families=16,new_intervention_pairs=320,
        valid_intervention_trials=int(np.sum(ss)+np.sum(ff)),intervention_numerical=int(np.sum(nn)),target_labels_used=False,
        input_reuse='H20=earlycontroller exactly byspeedbound;goalstaticresponse=late exactly at0.30m',data_sha256=sha(OUT/'dataset.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','filter','run','merge','dataset'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='run':run(a.index)
    else:globals()[{'filter':'filter_existing'}.get(a.action,a.action)]()
