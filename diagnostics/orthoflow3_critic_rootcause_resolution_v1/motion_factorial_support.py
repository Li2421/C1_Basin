"""Cache-first source supervision for the directly demonstrated input alias."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from .function_support import OUT as SOURCE,read,write,sha
from .phase_factorial_support import OUT as PHASE
from . import goal_motion_alias as runtime
from . import seed_replication as engine
from .goal_velocity_response import OUT as VELOCITY
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
OUT=SOURCE.parent/'motion_factorial_support'
RULE=SOURCE.parent/'motion_factorial_protocol.json'
NAMES=('alt_to_second','second_to_alt')


def prepare():
    assert not (OUT/'protocol.json').exists(),'Frozen source experimentexists'
    source=read(SOURCE/'protocol.json');old=read(runtime.OUT/'protocol.json')
    states=read(SOURCE/'states.json');pairs=read(SOURCE/'pairs.json')
    excluded={s['uid'] for s in read(SOURCE.parent/'goal_motion_confirmation/states.json')}
    assert not excluded & {s['uid'] for s in states}
    minimum=min(float(np.linalg.norm(np.array(s['physical']['goals'])-s['physical']['positions'],axis=1).min()) for s in states)
    assert minimum-80*.05*.52>1.
    allreq=[];profiles=[]
    for ci,name in enumerate(NAMES):
        dest=OUT/name;exp='exp_c1_motion_factorial_'+name+'_v1'
        early,late=(source['profiles'][i] for i in ((0,1) if ci==0 else (1,0)))
        if ci==0:
            profile=copy.deepcopy(old['profiles'][0]);program=read(profile['path'])
        else:
            program=copy.deepcopy(read(runtime.OUT/'controller_program.json'))
            program.update(alt=early,second=late)
            write(dest/'controller_program.json',program);ph=sha(dest/'controller_program.json')
            with connect() as db,transaction(db):
                ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(early['controller_uid'],)).fetchone();payload=json.loads(ref['config_json'])
                payload.update(flow_checkpoint_sha256=ph,flow_artifact_kind='frozen_controller_program_manifest',conditioning=payload['conditioning']+'+stationary_neargoal_motion_blend_v1',controller_program=program,controller_program_sha256=ph)
                cid=uid('ctl',payload)
                assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
                db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (cid,ref['scenario_uid'],ph,ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
            profile=dict(name='goal_motion',path=str(dest/'controller_program.json'),sha256=ph,controller_uid=cid)
        write(dest/'controller_program.json',program)
        assert sha(dest/'controller_program.json')==profile['sha256'] and program['runtime_sha256']==sha(runtime.__file__)
        proto={**source,'experiment_uid':exp,'experiment':name,'profiles':[profile],'parent_profile':early,'second_profile':late,
            'requested_seed_count':sum(p['target_seeds'] for p in pairs),'rule_sha256':sha(RULE),
            'intervention_runtime_sha256':sha(runtime.__file__),'source_driver_sha256':sha(__file__),
            'target_labels_used':False,'generator_modified':False}
        write(dest/'protocol.json',proto);write(dest/'states.json',states);write(dest/'pairs.json',pairs)
        req=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(p['target_seeds'])]) for p in pairs]
        write(dest/'planned_rollouts.json',{'requests':req});allreq+=req
        with connect() as db,transaction(db):
            db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                (exp,'motion_factorial_'+name,str(dest),sha(dest/'protocol.json'),sha(__file__),canonical({'source_only':True,'target_labels_used':False})))
        profiles.append({**profile,'factorial_name':name,'early_index':ci,'late_index':1-ci})
    write(OUT/'protocol.json',dict(rule=read(RULE),rule_sha256=sha(RULE),profiles=profiles,source_protocol_sha256=sha(SOURCE/'protocol.json'),
        initial_min_goal_distance=minimum,H80_max_displacement=2.08,old_inputs_identical_to_early=True,
        moving_goal_inputs_identical_to_late=True,total_requested=sum(len(r['seed_keys']) for r in allreq)))
    write(OUT/'planned_rollouts.json',{'requests':allreq})


def validate():
    from . import phase_factorial_support,state_support
    from shared_rollout_db.src.planner import preflight
    for name in NAMES:
        dest=OUT/name
        write(dest/'cache_preflight.json',preflight(dest/'planned_rollouts.json'))
        state_support.OUT=dest;state_support.validate_plan()
    phase_factorial_support.OUT=OUT
    phase_factorial_support.filter_existing()


def run(index):
    dest=OUT/NAMES[index//6]
    assert sha(runtime.__file__)==read(dest/'protocol.json')['intervention_runtime_sha256']
    runtime.OUT=dest;runtime.EXP=read(dest/'protocol.json')['experiment_uid']
    original=runtime.read
    runtime.read=lambda path:original(dest/'execution_preflight.json') if Path(path)==dest/'cache_preflight.json' else original(path)
    runtime.run(index%6)


def merge():
    from . import postprocess
    for name in NAMES:
        engine.OUT=OUT/name;engine.EXP=read(engine.OUT/'protocol.json')['experiment_uid']
        postprocess.run('motion_factorial')


def dataset():
    assert not (OUT/'dataset.npz').exists()
    d=dict(np.load(PHASE/'dataset.npz'));pairs=read(SOURCE/'pairs.json');profiles=read(OUT/'protocol.json')['profiles']
    ss=[];ff=[];nn=[];keys=[]
    with connect(True) as db:
        for c in profiles:
            s=[];f=[];num=[]
            for p in pairs:
                rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
                rr=[rr[canonical({'future_index':k})] for k in range(p['target_seeds'])]
                assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
                valid=[r for r in rr if not r['numerical_failure']]
                s.append(sum(r['success'] for r in valid));f.append(len(valid)-s[-1]);num.append(len(rr)-len(valid))
                keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rr]))
            ss.append(s);ff.append(f);nn.append(num)
    for key,v in (('success',ss),('failure',ff),('numerical',nn),('standard_success',ss),('standard_failure',ff)):
        d[key]=np.concatenate([d[key],np.array(v,np.float32)])
    for key in ('context','agent_response','valid','goal_response'):
        d[key]=np.concatenate([d[key],d[key][[0,1]]])
    mv=np.array([np.load(VELOCITY/f'inputs_{c["name"]}.npz')['goal_motion_response'] for c in read(SOURCE/'protocol.json')['profiles']])
    d['goal_motion_response']=np.concatenate([mv,mv[[1,0]],mv[[1,0]]])
    assert d['valid'].all() and ((d['success']+d['failure'])>0).all() and d['context'].shape[:2]==(16,160)
    np.savez_compressed(OUT/'dataset.npz',**d)
    write(OUT/'intervention_DB_keys.json',keys)
    write(OUT/'dataset_audit.json',dict(conditions=16,TRAIN_families=64,VAL_families=16,
        new_motion_pairs=320,valid_motion_trials=int(np.sum(ss)+np.sum(ff)),motion_numerical=int(np.sum(nn)),
        target_labels_used=False,independent_confirmation_families_excluded=True,
        feature_reuse='H20/agent/restgoal=early; movinggoal=late, byexactgates onthesephysicalquerydomains',data_sha256=sha(OUT/'dataset.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','validate','run','merge','dataset'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='run':run(a.index)
    else:globals()[a.action]()
