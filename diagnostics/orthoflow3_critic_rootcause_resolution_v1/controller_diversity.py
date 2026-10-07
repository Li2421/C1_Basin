"""Controlled expansion of source controller support, native cache semantics."""
import argparse,copy,json,os,shutil
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from .state_support import OUT as SOURCE,read,write
from . import seed_replication as engine
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
OUT=SOURCE.parent/'controller_diversity';EXP='exp_orthoflow3_rootcause_controller_diversity_v1'
RULE=SOURCE.parent/'controller_diversity_protocol.json'
NAMES=('extra_a','extra_b');SEEDS=(88125,88126)

def flow(index):
    import jax
    from new_benchmark_common.training import train_stage1
    assert jax.default_backend()=='gpu'
    seed=SEEDS[index];dest=OUT/f'flow_seed{seed}'
    assert not (dest/'best.pkl').exists(),'Do not retrain/select source controller by outcomes'
    dataset=SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset'
    result=train_stage1(dataset,dest,seed=seed,steps=100000,batch_size=256,log_interval=250,validation_batches=8,
        early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False)
    write(dest/'frozen_controller.json',dict(seed=seed,checkpoint_sha256=sha(dest/'best.pkl'),training_summary=result,
        expert_manifest_sha256=sha(dataset/'manifest.json'),feasibility_labels_used=False,rule_sha256=sha(RULE)))

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen source-controller design exists')
    states=copy.deepcopy(read(SOURCE/'model_states.json'));pairs=copy.deepcopy(read(SOURCE/'model_pairs.json'))
    for i,s in enumerate(states):s['index']=i;s['repair_index']=i
    for p in pairs:p['state_index']=p['dataset_state_index'];p['target_seeds']=8 if p['split']=='train' else 16
    base=read(SOURCE/'protocol.json');profiles=[]
    with connect() as db,transaction(db):
      ref=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(base['profiles'][0]['controller_uid'],)).fetchone()
      for name,seed in zip(NAMES,SEEDS):
        path=OUT/f'flow_seed{seed}/best.pkl';frozen=read(path.parent/'frozen_controller.json');assert sha(path)==frozen['checkpoint_sha256']
        payload=json.loads(ref['config_json']);payload['flow_checkpoint_sha256']=sha(path);cid=uid('ctl',payload)
        assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,ref['scenario_uid'],sha(path),ref['orthoflow3_sha256'],ref['safety_config_hash'],ref['horizon'],ref['dt'],ref['success_semantics_version'],payload['conditioning'],ref['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profiles.append(dict(name=name,path=str(path),sha256=sha(path),controller_uid=cid))
      proto={**base,'experiment_uid':EXP,'experiment':'controller_diversity_source_labels','profiles':profiles,
          'requested_seed_count':8640,'rule_sha256':sha(RULE),'source_train_families':206,'source_validation_families':32,
          'source_rule':'Reuse exactly old46+new160 TRAIN and32VAL; new controllers88125/88126 fixed before task outcomes',
          'target_labels_used':False,'generator_modified':False}
      write(OUT/'protocol.json',proto)
      db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
          (EXP,'controller_diversity_source_support',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'confirmation_labels_used':False})))
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    shutil.copyfile(SOURCE/'model_entities.npz',OUT/'frozen_model_entities.npz')
    requests=[]
    for c in profiles:
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(p['target_seeds'])]) for p in pairs]
        write(OUT/f'planned_{c["name"]}.json',{'requests':part});requests.extend(part)
    write(OUT/'planned_rollouts.json',{'requests':requests});write(OUT/'working_state.json',dict(phase='prepared',new_rollouts=0))

def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.parent.CONTROLLERS=NAMES;engine.SHARDS=12;engine.parent.SHARDS=12

def physical():
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    configure();engine.configure();native.physical()
    a=np.load(OUT/'entities.npz');b=np.load(OUT/'frozen_model_entities.npz')
    delta={k:float(np.max(abs(a[k]-b[k]))) for k in a.files}
    write(OUT/'state_input_reconstruction_audit.json',dict(max_absolute_difference=delta,preexisting_inputs_preserved_for_training=True))
    assert all(v<2e-6 for v in delta.values()),'Investigate input inconsistency; do not overwrite frozen features'

def features(index):
    from . import support_inputs
    support_inputs.OUT=OUT;support_inputs.features(NAMES[index])

def materialize():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    if (OUT/'dataset.npz').exists():raise FileExistsError('Dataset frozen')
    base=dict(np.load(SOURCE/'dataset.npz'));pairs=read(OUT/'pairs.json');profiles=read(OUT/'protocol.json')['profiles'];keys=[]
    s=np.zeros((2,len(pairs)),np.float32);f=s.copy();num=s.copy()
    with connect(True) as db:
      for ci,c in enumerate(profiles):
       for i,p in enumerate(pairs):
        rows=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid'])).fetchall()
        assert len(rows)==p['target_seeds'] and all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rows)
        assert {r['seed_key'] for r in rows}=={canonical({'future_index':k}) for k in range(p['target_seeds'])}
        valid=[r for r in rows if not r['numerical_failure']];s[ci,i]=sum(r['success'] for r in valid);f[ci,i]=len(valid)-s[ci,i];num[ci,i]=len(rows)-len(valid)
        keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rows]))
    extra=[np.load(OUT/f'inputs_{c}.npz') for c in NAMES]
    for name,a in [('success',s),('failure',f),('numerical',num),('standard_success',s),('standard_failure',f)]:base[name]=np.concatenate([base[name],a],axis=0)
    for name in ('context','agent_response','valid'):base[name]=np.concatenate([base[name],np.stack([v[name] for v in extra])],axis=0)
    assert base['valid'].all(),'Handle invalid response explicitly before training'
    np.savez_compressed(OUT/'dataset.npz',**base)
    write(OUT/'dataset_DB_keys.json',read(SOURCE/'dataset_DB_keys.json')+keys)
    write(OUT/'dataset_audit.json',dict(TRAIN_families=206,VAL_families=32,controllers=4,pairs=1904,
        valid_observations=int((base['success']+base['failure']).sum()),numerical=int(base['numerical'].sum()),
        controller_order=['alt','second',*NAMES],target_labels_used=False,old_labels_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('flow','prepare','physical','features','filter','run','dataset'));p.add_argument('--index',type=int,default=0);p.add_argument('--controller',default='extra_a');a=p.parse_args()
    if a.action=='flow':flow(a.index)
    elif a.action=='features':features(a.index)
    elif a.action in ('prepare','physical'):globals()[a.action]()
    elif a.action=='dataset':materialize()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run(a.controller,a.index)
