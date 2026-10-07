"""New controller88127 and new families; never reuse opened88124 labels."""
import argparse,copy,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from .controller_diversity import OUT as SOURCE,RULE,read,write
from . import seed_replication as engine
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from new_benchmark_common.basin_dataset_v1 import content_hash
OUT=SOURCE.parent/'controller_diversity_confirmation';FLOW=SOURCE/'target_flow_seed88127'
EXP='exp_orthoflow3_rootcause_diversity_confirmation88127_v1'

def flow():
    import jax
    from new_benchmark_common.training import train_stage1
    assert jax.default_backend()=='gpu'
    assert (SOURCE/'models_frozen.json').exists()
    assert not (FLOW/'best.pkl').exists(),'Target controller cannot be selected/retrained by outcome'
    dataset=SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset'
    result=train_stage1(dataset,FLOW,seed=88127,steps=100000,batch_size=256,log_interval=250,validation_batches=8,
        early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False)
    write(FLOW/'frozen_controller.json',dict(seed=88127,checkpoint_sha256=sha(FLOW/'best.pkl'),training_summary=result,
        models_frozen_sha256=sha(SOURCE/'models_frozen.json'),rule_sha256=sha(RULE),target_feasibility_labels_used=False))

def prepare():
    from ring_exchange.environment import Config,sample_instance
    if (OUT/'protocol.json').exists():raise FileExistsError('Target already frozen')
    frozen=read(FLOW/'frozen_controller.json');assert sha(FLOW/'best.pkl')==frozen['checkpoint_sha256']
    assert sha(SOURCE/'models_frozen.json')==frozen['models_frozen_sha256']
    base=read(SOURCE/'protocol.json');source_states=read(SOURCE/'states.json');source_eta=read(SOURCE/'pairs.json')[:2]
    cfg=Config(**read(SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
    states=[];pairs=[]
    with connect() as db,transaction(db):
      sr=db.execute('SELECT * FROM state WHERE state_uid=?',(source_states[0]['uid'],)).fetchone()
      cr=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(base['profiles'][0]['controller_uid'],)).fetchone()
      payload=json.loads(cr['config_json']);payload['flow_checkpoint_sha256']=frozen['checkpoint_sha256'];cid=uid('ctl',payload)
      assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
      db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
          (cid,cr['scenario_uid'],frozen['checkpoint_sha256'],cr['orthoflow3_sha256'],cr['safety_config_hash'],cr['horizon'],cr['dt'],cr['success_semantics_version'],payload['conditioning'],cr['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
      for i in range(64):
        seed=940041000+i;instance=sample_instance(seed,'test',cfg)
        physical={k:getattr(instance,k).tolist() for k in ('positions','velocities','goals')};physical['timestep']=0
        ch=content_hash(physical);sid=uid('state',{'scenario':sr['scenario_uid'],'content':ch})
        assert db.execute('SELECT 1 FROM state WHERE state_uid=?',(sid,)).fetchone() is None,'Exact physical family overlap; stop, do not outcome-filter'
        state=dict(uid=sid,index=i,repair_index=i,alias=f'ROOTCAUSE_DIVERSITY_TEST_{seed}',content_hash=ch,physical=physical,
            source_group=f'rootcause_diversity:test:{seed}',rollout_id=f'rootcause_diversity:test:{seed}',split='test',source_dataset_split='test',
            provenance=dict(generator='ring_exchange.environment.sample_instance',initial_seed=seed,split='test',outcome_blind=True,task=EXP))
        geometry=json.loads(sr['goals_geometry_json']);geometry['goals']=physical['goals']
        db.execute('INSERT INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)',
            (sid,sr['scenario_uid'],state['source_group'],ch,canonical(physical),None,canonical(geometry),canonical(state['provenance']),'CONTENT_EXACT'))
        db.execute('INSERT INTO state_alias VALUES(?,?,?,?)',(sr['scenario_uid'],state['alias'],sid,EXP));states.append(state)
        for e in source_eta:
            p=copy.deepcopy(e);p.update(state_uid=sid,state_index=i,split='test',target_seeds=16);p.pop('dataset_state_index',None);pairs.append(p)
      profile=dict(name='held',path=str(FLOW/'best.pkl'),sha256=frozen['checkpoint_sha256'],controller_uid=cid)
      proto={**base,'experiment_uid':EXP,'experiment':'new_controller88127_new_families_confirmation','profiles':[profile],
          'requested_seed_count':2048,'models_frozen_sha256':frozen['models_frozen_sha256'],
          'rule_sha256':sha(RULE),'scope':'New controller parameters88127 and64 new native Ring families; same2source eta; not cross-scene or unseen-eta.',
          'source_rule':'Native sample_instance940041000+j,test; model-independent two source eta; no outcome filtering',
          'target_labels_used':False}
      write(OUT/'protocol.json',proto)
      db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
          (EXP,'diversity_new_controller_family_confirmation',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'models_frozen_sha256':frozen['models_frozen_sha256']})))
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    req=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
    write(OUT/'planned_rollouts.json',{'requests':req});write(OUT/'planned_held.json',{'requests':req});write(OUT/'working_state.json',dict(phase='prepared_after_model_freeze'))

def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.parent.CONTROLLERS=('held',);engine.SHARDS=18;engine.parent.SHARDS=18

def physical():
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    configure();engine.configure();native.physical()

def features(index):
    # Compute the same physical descriptor for the held controller and each
    # of the four prespecified wrong source controllers. No task labels.
    from .agent_response import instrument
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    import jax
    assert jax.default_backend()=='gpu'
    old=read(SOURCE.parent/'state_support/protocol.json')['profiles']
    extra=read(SOURCE/'protocol.json')['profiles'];held=read(OUT/'protocol.json')['profiles']
    profile=(held+old+extra)[index];target=OUT/f'inputs_{profile["name"]}.npz'
    if target.exists():raise FileExistsError('Response already frozen')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20};rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
    states=read(OUT/'physical.json');pairs=read(OUT/'pairs.json');context=np.zeros((128,24),np.float32);response=np.zeros((128,4,16),np.float32);valid=np.zeros(128,bool);errors=[]
    for i,p in enumerate(pairs):
        state=states[p['state_index']];assert state['state_uid']==p['state_uid'];trace['runs']=[]
        try:
            result=rt.features(state['physical'],p['eta']);assert len(trace['runs'])==4 and all(len(v)==20 for v in trace['runs'])
            means=np.array([np.array(v).mean(0) for v in trace['runs']]);context[i]=result['mean'];response[i]=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],-1)
            assert np.isfinite(context[i]).all() and np.isfinite(response[i]).all();valid[i]=True
        except Exception as e:errors.append(dict(index=i,error=f'{type(e).__name__}: {e}'))
    np.savez_compressed(target,context=context,agent_response=response,valid=valid)
    write(OUT/f'input_audit_{profile["name"]}.json',dict(invalid=errors,models_frozen_sha256=sha(SOURCE/'models_frozen.json'),labels_used=False))

def dataset():
    from . import fresh_controller_test as original
    # Reuse validated exact seed aggregation with an explicit input filename.
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    if (OUT/'dataset.npz').exists():raise FileExistsError('Evidence already frozen')
    profile=read(OUT/'protocol.json')['profiles'][0];pairs=read(OUT/'pairs.json');s=[];f=[];num=[];keys=[]
    with connect(True) as db:
      for p in pairs:
        rows=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],profile['controller_uid'])).fetchall()
        assert len(rows)==16 and {r['seed_key'] for r in rows}=={canonical({'future_index':k}) for k in range(16)}
        assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rows)
        v=[r for r in rows if not r['numerical_failure']];ss=sum(r['success'] for r in v);s.append(ss);f.append(len(v)-ss);num.append(16-len(v))
        keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rows]))
    inp=np.load(OUT/'inputs_held.npz')
    np.savez_compressed(OUT/'dataset.npz',success=np.array(s),failure=np.array(f),numerical=np.array(num),
        context=inp['context'],agent_response=inp['agent_response'],valid=inp['valid'],eta=np.array([p['eta'] for p in pairs],np.float32),
        state_index=np.array([p['state_index'] for p in pairs]),eta_index=np.array([p['eta_index'] for p in pairs]))
    write(OUT/'dataset_DB_keys.json',keys)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('flow','prepare','physical','features','filter','run','dataset'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='features':features(a.index)
    elif a.action in ('flow','prepare','physical','dataset'):globals()[a.action]()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run('held',a.index)
