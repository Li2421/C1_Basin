"""Twelve source-controller support with matched physical families and eta."""
import argparse,copy,hashlib,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from .controller_diversity import OUT as SOURCE,read,write
from . import seed_replication as engine
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
OUT=SOURCE.parent/'controller_function_support';RULE=SOURCE.parent/'controller_function_support_protocol.json'
EXP='exp_orthoflow3_controller_function_support12_v1'
OLD=('alt','second','extra_a','extra_b');NAMES=tuple(f'function_{i}' for i in range(8));SEEDS=tuple(range(88201,88209))

def flow(index):
    from . import controller_diversity as original
    original.OUT=OUT;original.SEEDS=SEEDS;original.RULE=RULE
    original.flow(index)

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen source design exists')
    source_states=read(SOURCE/'states.json');source_pairs=read(SOURCE/'pairs.json')
    d=np.load(SOURCE/'dataset.npz');pool={r['uid']:r for r in source_states}
    tr=[s for i,s in enumerate(source_states) if d['source'][2*i]!='old46' and s['split']=='train']
    val=[s for s in source_states if s['split']=='validation'];assert len(tr)==160 and len(val)==32
    key=lambda s:hashlib.sha256(('c1_controller_function_support_v1\0'+s['source_group']).encode()).hexdigest()
    selected=sorted(tr,key=key)[:64]+sorted(val,key=key)[:16];states=[];pairs=[];original_indices=[]
    for i,s0 in enumerate(selected):
        s=copy.deepcopy(s0);original_indices.append(s0['index']);s['index']=i;s['repair_index']=i;states.append(s)
        pp=[p for p in source_pairs if p['state_uid']==s['uid']];assert len(pp)==2
        for p0 in pp:
            p=copy.deepcopy(p0);p['state_index']=i;p['dataset_state_index']=i;p['target_seeds']=4 if s['split']=='train' else 16;pairs.append(p)
    base=read(SOURCE/'protocol.json')
    profiles=read(SOURCE.parent/'state_support/protocol.json')['profiles']+base['profiles']
    assert [p['name'] for p in profiles]==list(OLD)
    with connect() as db,transaction(db):
      ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(profiles[0]['controller_uid'],)).fetchone()
      for name,seed in zip(NAMES,SEEDS):
        path=OUT/f'flow_seed{seed}/best.pkl';frozen=read(path.parent/'frozen_controller.json');assert sha(path)==frozen['checkpoint_sha256']
        payload=json.loads(ref['config_json']);payload['flow_checkpoint_sha256']=sha(path);cid=uid('ctl',payload)
        assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,ref['scenario_uid'],sha(path),ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profiles.append(dict(name=name,path=str(path),sha256=sha(path),controller_uid=cid))
      proto={**base,'experiment_uid':EXP,'experiment':'controller_function_support12','profiles':profiles,
          'requested_seed_count':12288,'rule_sha256':sha(RULE),'source_train_families':64,'source_validation_families':16,
          'source_rule':read(RULE)['families'],'target_labels_used':False,'generator_modified':False}
      write(OUT/'protocol.json',proto)
      db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
          (EXP,'controller_function_support12',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'target_labels_used':False})))
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    x=np.load(SOURCE/'frozen_model_entities.npz');np.savez_compressed(OUT/'frozen_model_entities.npz',**{k:x[k][original_indices] for k in x.files})
    srcphys=read(SOURCE/'physical.json');physical=[]
    for i,oldidx in enumerate(original_indices):
        item=copy.deepcopy(srcphys[oldidx]);assert item['state_uid']==states[i]['uid'];physical.append(item)
    write(OUT/'physical.json',physical)
    source_pair_lookup={(p['state_uid'],p['eta_index']):i for i,p in enumerate(source_pairs)}
    ii=np.array([source_pair_lookup[(p['state_uid'],p['eta_index'])] for p in pairs])
    for ci,name in enumerate(OLD):
        np.savez_compressed(OUT/f'inputs_{name}.npz',**{k:d[k][ci,ii] for k in ('context','agent_response','valid')})
    req=[]
    for c in profiles:
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[canonical({'future_index':j}) for j in range(p['target_seeds'])]) for p in pairs]
        write(OUT/f'planned_{c["name"]}.json',{'requests':part});req.extend(part)
    write(OUT/'planned_rollouts.json',{'requests':req});write(OUT/'working_state.json',dict(phase='prepared',new_rollouts=0))
    write(OUT/'source_selection.json',dict(original_state_indices=original_indices,state_uids=[s['uid'] for s in states],outcomes_used=False))

def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.SHARDS=18;engine.parent.SHARDS=18;engine.parent.CONTROLLERS=OLD+NAMES

def features(index):
    from . import support_inputs
    support_inputs.OUT=OUT;support_inputs.features(NAMES[index])

def materialize():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    if (OUT/'dataset.npz').exists():raise FileExistsError('Dataset frozen')
    pairs=read(OUT/'pairs.json');profiles=read(OUT/'protocol.json')['profiles'];s=np.zeros((12,len(pairs)),np.float32);f=s.copy();num=s.copy();keys=[]
    with connect(True) as db:
      for ci,c in enumerate(profiles):
       for i,p in enumerate(pairs):
        rows={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
        rr=[rows[canonical({'future_index':k})] for k in range(p['target_seeds'])]
        assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
        valid=[r for r in rr if not r['numerical_failure']];s[ci,i]=sum(r['success'] for r in valid);f[ci,i]=len(valid)-s[ci,i];num[ci,i]=len(rr)-len(valid)
        keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rr]))
    inputs=[np.load(OUT/f'inputs_{c["name"]}.npz') for c in profiles]
    data=dict(success=s,failure=f,numerical=num,standard_success=s,standard_failure=f,
        eta=np.array([p['eta'] for p in pairs],np.float32),eta_index=np.array([p['eta_index'] for p in pairs]),
        state_index=np.array([p['state_index'] for p in pairs]),split=np.array([p['split'] for p in pairs]))
    for k in ('context','agent_response','valid'):data[k]=np.stack([v[k] for v in inputs])
    assert data['valid'].all() and ((s+f)>0).all()
    np.savez_compressed(OUT/'dataset.npz',**data);write(OUT/'dataset_DB_keys.json',keys)
    write(OUT/'dataset_audit.json',dict(TRAIN_families=64,VAL_families=16,controllers=12,pairs=len(pairs)*12,
        valid_observations=int((s+f).sum()),numerical=int(num.sum()),controller_order=[p['name'] for p in profiles],
        target_labels_used=False,stronger_DB_evidence_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('flow','prepare','features','filter','run','dataset','validate'));p.add_argument('--index',type=int,default=0);p.add_argument('--controller',default='function_0');a=p.parse_args()
    if a.action in ('flow','features'):globals()[a.action](a.index)
    elif a.action=='prepare':prepare()
    elif a.action=='dataset':materialize()
    elif a.action=='validate':
        from . import state_support
        state_support.OUT=OUT;state_support.validate_plan()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run(a.controller,a.index)
