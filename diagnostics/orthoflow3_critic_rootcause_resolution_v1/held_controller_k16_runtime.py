"""Register, measure, execute and merge the precommitted K16 benchmark.

No model adaptation; all success rollouts use the existing audited native
executor and exact global-cache identities. Each task batch is <=4096 seeds.
"""
import argparse, copy, json, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import numpy as np
from .held_controller_k16_v1 import OUT, SEEDS, FAMILY_ROOTS, ROOT, BASE, DATA, read, write, sha
from shared_rollout_db.src.rollout_db import connect, transaction, canonical, uid, eta_identity


def target(index):return OUT/f'target_{SEEDS[index]}'
def batchdir(index,batch):return target(index)/f'batch{batch}'
def exp(index,batch=None):return f'exp_c1_held_controller_k16_{SEEDS[index]}' + ('' if batch is None else f'_batch{batch}')


def freeze_models():
    assert not (OUT/'models_frozen.json').exists()
    cv=ROOT/'source_controller_cv_v2/final_models';docs=[]
    for label,folder,kinds in [('controller_cv',cv,('eta_only','full_context')),('family_cv',DATA,('eta_only','no_context','additive_nominal_context','full_context','full_context_freeze25'))]:
        for r in read(folder/'models_frozen.json')['models']:
            if r['kind'] not in kinds:continue
            p=Path(r['path']);assert sha(p/'checkpoint.msgpack')==r['checkpoint_sha256']
            docs.append(dict(label=label,**r))
    old=ROOT/'raw_bypass_final_models';legacy=[r for r in read(old/'models_frozen.json')['models'] if r['variant'] in ('raw_bypass','eta_only')]
    for r in legacy:assert sha(Path(r['path'])/r['checkpoint'])==r['checkpoint_sha256']
    deps=[Path(__file__),ROOT/'held_controller_k16_evaluate.py',ROOT/'db_transfer_context.py',ROOT/'db_transfer_data.py',ROOT/'db_transfer_train.py',ROOT/'raw_bypass_confirmation.py',BASE/'diagnostics/orthoflow3_controller_training_repair_v1/data.py',BASE/'diagnostics/orthoflow3_controller_intervention_generalization_v1/rich_context.py']
    write(OUT/'models_frozen.json',dict(models=docs,legacy_models=legacy,primary='controller_cv full_context vs controller_cv eta_only; family_cv continuous eta-only retained as stronger-baseline check',normalization_path=str(DATA/'normalization.json'),normalization_sha256=sha(DATA/'normalization.json'),legacy_models_sha256=sha(old/'models_frozen.json'),protocol_sha256=sha(OUT/'protocol.json'),dependencies={str(p):sha(p) for p in deps},target_labels_used=False,models_adapted_to_target=False))
    print(dict(frozen_models=len(docs),legacy_models=len(legacy),new_rollouts=0))


def prepare(index):
    from ring_exchange.environment import Config,sample_instance
    from new_benchmark_common.basin_dataset_v1 import content_hash
    dest=target(index);assert not (dest/'protocol.json').exists()
    guard=read(OUT/'models_frozen.json');rule=read(OUT/'protocol.json');assert guard['protocol_sha256']==sha(OUT/'protocol.json')
    flowdir=OUT/f'flow_seed{SEEDS[index]}';flow=read(flowdir/'frozen_controller.json')
    assert flow['checkpoint_sha256']==sha(flowdir/'best.pkl') and flow['protocol_sha256']==sha(OUT/'protocol.json')
    source=ROOT/'controller_function_support';base=read(source/'protocol.json');ss=read(source/'states.json');panel=read(OUT/'candidate_panel.json')
    cfg=Config(**read(BASE/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
    states=[];pairs=[];sourceids={r['state_uid'] for r in read(DATA/'states.json')}
    with connect() as db,transaction(db):
        sr=db.execute('SELECT * FROM state WHERE state_uid=?',(ss[0]['uid'],)).fetchone()
        cr=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(base['profiles'][0]['controller_uid'],)).fetchone()
        payload=json.loads(cr['config_json']);payload['flow_checkpoint_sha256']=flow['checkpoint_sha256'];cid=uid('ctl',payload)
        assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(cid,cr['scenario_uid'],flow['checkpoint_sha256'],cr['orthoflow3_sha256'],cr['safety_config_hash'],cr['horizon'],cr['dt'],cr['success_semantics_version'],payload['conditioning'],cr['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        for e in panel:
            eu,v,hx=eta_identity(e['eta']);assert eu==e['eta_uid']
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eu,*v,hx))
            rr=db.execute('SELECT canonical_hex FROM eta WHERE eta_uid=?',(eu,)).fetchone();assert rr[0]==hx
        for j in range(rule['families_per_controller']):
            seed=FAMILY_ROOTS[index]+j;instance=sample_instance(seed,'test',cfg)
            physical={k:getattr(instance,k).tolist() for k in ('positions','velocities','goals')};physical['timestep']=0
            ch=content_hash(physical);sid=uid('state',{'scenario':sr['scenario_uid'],'content':ch})
            assert sid not in sourceids and db.execute('SELECT 1 FROM state WHERE state_uid=?',(sid,)).fetchone() is None
            group=f'rootcause_k16:test:{seed}'
            state=dict(uid=sid,index=j,repair_index=j,alias=f'ROOTCAUSE_K16_{seed}',content_hash=ch,physical=physical,source_group=group,rollout_id=group,split='test',source_dataset_split='test',provenance=dict(generator='ring_exchange.environment.sample_instance',initial_seed=seed,split='test',outcome_blind=True,task=exp(index)))
            geometry=json.loads(sr['goals_geometry_json']);geometry['goals']=physical['goals']
            db.execute('INSERT INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)',(sid,sr['scenario_uid'],group,ch,canonical(physical),None,canonical(geometry),canonical(state['provenance']),'CONTENT_EXACT'))
            db.execute('INSERT INTO state_alias VALUES(?,?,?,?)',(sr['scenario_uid'],state['alias'],sid,exp(index)));states.append(state)
            for e in panel:pairs.append(dict(state_uid=sid,state_index=j,eta_uid=e['eta_uid'],eta=e['eta'],eta_index=e['index'],seen_eta=e['seen'],split='test',target_seeds=16))
        profile=dict(name='held',path=str(flowdir/'best.pkl'),sha256=flow['checkpoint_sha256'],controller_uid=cid)
        protocol={**base,'experiment_uid':exp(index),'experiment':'held_controller_K16_independent_confirmation','profiles':[profile],'requested_seed_count':8192,'models_frozen_sha256':sha(OUT/'models_frozen.json'),'confirmation_protocol_sha256':sha(OUT/'protocol.json'),'target_controller_seed':SEEDS[index],'target_labels_used':False,'scope':rule['scope']}
        write(dest/'protocol.json',protocol);write(dest/'states.json',states);write(dest/'pairs.json',pairs)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(exp(index),'held_controller_K16',str(dest),sha(dest/'protocol.json'),sha(__file__),canonical({'parent_protocol':str(OUT/'protocol.json'),'labels_unopened':True})))
        for batch in (0,1):
            bd=batchdir(index,batch);part=[r for r in pairs if 16*batch<=r['state_index']<16*(batch+1)]
            bp={**protocol,'experiment_uid':exp(index,batch),'requested_seed_count':4096,'parent_experiment_uid':exp(index),'stage':batch}
            write(bd/'protocol.json',bp);write(bd/'states.json',states);write(bd/'pairs.json',part)
            req={'requests':[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in part]}
            write(bd/'planned_rollouts.json',req);write(bd/'planned_held.json',req)
            db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(exp(index,batch),'held_controller_K16_batch',str(bd),sha(bd/'protocol.json'),sha(__file__),canonical({'parent_experiment':exp(index),'batch':batch,'rollouts_authorized_only_after_prediction_freeze_and_preflight':True})))
    write(dest/'registration_audit.json',dict(states=len(states),pairs=len(pairs),new_rollouts=0,source_state_overlap=0,controller_never_in_critic_training=True,rule_frozen_before_creation=True))
    print(dict(target=SEEDS[index],states=32,pairs=512,task_rollouts=0),flush=True)


def configure(index,batch=None):
    from . import seed_replication as engine
    engine.OUT=target(index) if batch is None else batchdir(index,batch)
    engine.EXP=exp(index,batch);engine.SHARDS=5
    engine.parent.SHARDS=5;engine.parent.CONTROLLERS=('held',);engine.configure()
    return engine


def physical(index):
    import jax
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    configure(index);jax.config.update('jax_enable_x64',False);native.physical()
    p=read(target(index)/'physical.json');assert len(p)==32
    print(dict(target=SEEDS[index],physical_rows=len(p),new_rollouts=0),flush=True)


def features(index,worker):
    from .db_transfer_context import Measure
    dest=target(index);p=read(dest/'physical.json');pairs=read(dest/'pairs.json')
    profile=read(dest/'protocol.json')['profiles'][0]
    wrong=read(ROOT/'controller_function_support/protocol.json')['profiles'][0]
    # Whole states remain on a worker to reuse nominal and raw Flow queries.
    chosen=[i for i,r in enumerate(pairs) if r['state_index']%5==worker]
    result={};errors=[]
    for label,ctl in [('correct',profile),('wrong_controller',wrong)]:
        assert sha(ctl['path'])==ctl['sha256'];measure=Measure('ring_exchange',ctl['path'])
        arr=[]
        for i in chosen:
            r=pairs[i];ss=p[r['state_index']];assert ss['state_uid']==r['state_uid']
            value,err=measure.one(r['state_uid'],ss['physical'],np.array(r['eta'],float));arr.append(value)
            errors.extend(dict(pair=i,condition=label,**a) for a in err)
        result[label]=np.array(arr)
    dest=dest/'inputs';dest.mkdir(exist_ok=True)
    assert not (dest/f'worker{worker}.npz').exists()
    np.savez_compressed(dest/f'worker{worker}.npz',indices=np.array(chosen),**result)
    write(dest/f'worker{worker}.json',dict(worker=worker,pairs=len(chosen),errors=errors,code_sha256=sha(__file__),context_code_sha256=sha(ROOT/'db_transfer_context.py'),task_rollouts=0,target_labels_used=False))
    print(dict(target=SEEDS[index],worker=worker,measured=len(chosen),invalid_groups=len(errors)),flush=True)


def assemble(index):
    dest=target(index);pairs=read(dest/'pairs.json');result={k:np.zeros((len(pairs),76),np.float32) for k in ('correct','wrong_controller')};seen=[];errors=[]
    for worker in range(5):
        a=np.load(dest/'inputs'/f'worker{worker}.npz');doc=read(dest/'inputs'/f'worker{worker}.json')
        assert doc['code_sha256']==sha(__file__)
        for key in result:result[key][a['indices']]=a[key]
        seen.extend(a['indices'].tolist());errors.extend(doc['errors'])
    assert sorted(seen)==list(range(len(pairs)))
    np.savez_compressed(dest/'context.npz',**result)
    write(dest/'context_audit.json',dict(pairs=len(pairs),errors=errors,all_rows_present=True,new_rollouts=0,target_labels_used=False))


def validate(index,batch):
    from . import state_support
    state_support.OUT=batchdir(index,batch);state_support.validate_plan()
    engine=configure(index,batch);engine.filter_numerical()
    summary=read(batchdir(index,batch)/'cache_preflight.json')['summary']
    print(dict(target=SEEDS[index],batch=batch,**summary),flush=True)


def run(index,batch,worker):
    dest=target(index);guard=read(dest/'prediction_freeze.json')
    assert guard['models_frozen_sha256']==sha(OUT/'models_frozen.json')
    assert guard['predictions_sha256']==sha(dest/'predictions.npz')
    assert guard['runtime_sha256']==sha(__file__)
    for path,digest in read(OUT/'models_frozen.json')['dependencies'].items():assert sha(path)==digest
    assert read(batchdir(index,batch)/'identity_preflight_audit.json')['identity_checks_passed']
    if batch==1:assert read(OUT/'stage1_eligibility.json')['extend_second_stage']
    configure(index,batch).run('held',worker)


def merge(index,batch):
    configure(index,batch).merge()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze_models','prepare','physical','features','assemble','validate','run','merge'));p.add_argument('--index',type=int,default=0);p.add_argument('--worker',type=int,default=0);p.add_argument('--batch',type=int,default=0);a=p.parse_args()
    if a.action=='freeze_models':freeze_models()
    elif a.action in ('features',):features(a.index,a.worker)
    elif a.action in ('validate','merge'):globals()[a.action](a.index,a.batch)
    elif a.action=='run':run(a.index,a.batch,a.worker)
    else:globals()[a.action](a.index)
