"""Independent final controller/families for source-function-support scaling."""
import argparse,copy,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .function_support import OUT as SOURCE,RULE,read,write
from . import seed_replication as engine
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from new_benchmark_common.basin_dataset_v1 import content_hash
OUT=SOURCE.parent/'controller_function_confirmation';FLOW=SOURCE/'target_flow_seed88128';EXP='exp_orthoflow3_function_support_confirmation88128_v1'
TARGET_SEED=88128;FAMILY_SEED=940051000

def replica(index):
    global OUT,FLOW,EXP,TARGET_SEED,FAMILY_SEED
    assert index in (0,1)
    TARGET_SEED=88128+index;FAMILY_SEED=940051000+index*10000
    OUT=SOURCE.parent/('controller_function_confirmation' if index==0 else 'controller_function_confirmation_b')
    FLOW=SOURCE/f'target_flow_seed{TARGET_SEED}';EXP=f'exp_orthoflow3_function_support_confirmation{TARGET_SEED}_v1'

def flow():
    import jax
    from new_benchmark_common.training import train_stage1
    assert jax.default_backend()=='gpu'
    assert read(SOURCE/'source_gate.json')['passed'] and (SOURCE/'models_frozen.json').exists()
    assert not (FLOW/'best.pkl').exists()
    dataset=SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset'
    result=train_stage1(dataset,FLOW,seed=TARGET_SEED,steps=100000,batch_size=256,log_interval=250,validation_batches=8,
        early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False)
    write(FLOW/'frozen_controller.json',dict(seed=TARGET_SEED,checkpoint_sha256=sha(FLOW/'best.pkl'),training_summary=result,
        models_frozen_sha256=sha(SOURCE/'models_frozen.json'),rule_sha256=sha(RULE),
        confirmation_protocol_sha256=sha(SOURCE/'independent_confirmation_protocol.json'),target_feasibility_labels_used=False))

def prepare():
    from ring_exchange.environment import Config,sample_instance
    if (OUT/'protocol.json').exists():raise FileExistsError('Target frozen')
    frozen=read(FLOW/'frozen_controller.json');assert sha(FLOW/'best.pkl')==frozen['checkpoint_sha256']
    assert sha(SOURCE/'models_frozen.json')==frozen['models_frozen_sha256']
    base=read(SOURCE/'protocol.json');ss=read(SOURCE/'states.json');etas=read(SOURCE/'pairs.json')[:2]
    cfg=Config(**read(SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config']);states=[];pairs=[]
    with connect() as db,transaction(db):
      sr=db.execute('SELECT * FROM state WHERE state_uid=?',(ss[0]['uid'],)).fetchone()
      cr=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(base['profiles'][0]['controller_uid'],)).fetchone()
      payload=json.loads(cr['config_json']);payload['flow_checkpoint_sha256']=frozen['checkpoint_sha256'];cid=uid('ctl',payload)
      assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
      db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
          (cid,cr['scenario_uid'],frozen['checkpoint_sha256'],cr['orthoflow3_sha256'],cr['safety_config_hash'],cr['horizon'],cr['dt'],cr['success_semantics_version'],payload['conditioning'],cr['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
      for i in range(64):
        seed=FAMILY_SEED+i;instance=sample_instance(seed,'test',cfg)
        physical={k:getattr(instance,k).tolist() for k in ('positions','velocities','goals')};physical['timestep']=0
        ch=content_hash(physical);sid=uid('state',{'scenario':sr['scenario_uid'],'content':ch})
        assert db.execute('SELECT 1 FROM state WHERE state_uid=?',(sid,)).fetchone() is None
        state=dict(uid=sid,index=i,repair_index=i,alias=f'ROOTCAUSE_FUNCTION_TEST_{seed}',content_hash=ch,physical=physical,
            source_group=f'rootcause_function:test:{seed}',rollout_id=f'rootcause_function:test:{seed}',split='test',source_dataset_split='test',
            provenance=dict(generator='ring_exchange.environment.sample_instance',initial_seed=seed,split='test',outcome_blind=True,task=EXP))
        geometry=json.loads(sr['goals_geometry_json']);geometry['goals']=physical['goals']
        db.execute('INSERT INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)',
            (sid,sr['scenario_uid'],state['source_group'],ch,canonical(physical),None,canonical(geometry),canonical(state['provenance']),'CONTENT_EXACT'))
        db.execute('INSERT INTO state_alias VALUES(?,?,?,?)',(sr['scenario_uid'],state['alias'],sid,EXP));states.append(state)
        for e in etas:
            p=copy.deepcopy(e);p.update(state_uid=sid,state_index=i,split='test',target_seeds=16);p.pop('dataset_state_index',None);pairs.append(p)
      profile=dict(name='held',path=str(FLOW/'best.pkl'),sha256=frozen['checkpoint_sha256'],controller_uid=cid)
      proto={**base,'experiment_uid':EXP,'experiment':f'function_support_new_controller{TARGET_SEED}','profiles':[profile],
          'requested_seed_count':2048,'models_frozen_sha256':frozen['models_frozen_sha256'],'rule_sha256':sha(RULE),
          'scope':f'New controller{TARGET_SEED} and64newRingtrue-t0families;2seenexacteta. Notcross-scene or unseeneta.',
          'target_controller_seed':TARGET_SEED,'source_rule':f'Native sample_instance{FAMILY_SEED}+j,test;nooutcomefiltering',
          'confirmation_protocol_sha256':frozen['confirmation_protocol_sha256'],'target_labels_used':False}
      write(OUT/'protocol.json',proto)
      db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
          (EXP,'function_support_confirmation',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'models_frozen_sha256':frozen['models_frozen_sha256']})))
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    requests=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
    write(OUT/'planned_rollouts.json',{'requests':requests});write(OUT/'planned_held.json',{'requests':requests});write(OUT/'working_state.json',dict(phase='prepared_after_model_freeze'))

def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.parent.CONTROLLERS=('held',);engine.SHARDS=18;engine.parent.SHARDS=18

def physical():
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    configure();engine.configure();native.physical()

def features(index):
    import jax
    from .agent_response import instrument
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    assert jax.default_backend()=='gpu'
    profiles=read(OUT/'protocol.json')['profiles']+read(SOURCE/'protocol.json')['profiles'][:4]
    profile=profiles[index];dest=OUT/f'inputs_{profile["name"]}.npz'
    if dest.exists():raise FileExistsError('Inputs frozen')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20};rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
    physical=read(OUT/'physical.json');pairs=read(OUT/'pairs.json');ctx=np.zeros((128,24),np.float32);agent=np.zeros((128,4,16),np.float32);valid=np.zeros(128,bool);errors=[]
    for i,p in enumerate(pairs):
        state=physical[p['state_index']];assert state['state_uid']==p['state_uid'];trace['runs']=[]
        try:
            result=rt.features(state['physical'],p['eta']);assert len(trace['runs'])==4 and all(len(r)==20 for r in trace['runs'])
            means=np.array([np.array(r).mean(0) for r in trace['runs']]);ctx[i]=result['mean'];agent[i]=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],-1)
            assert np.isfinite(ctx[i]).all() and np.isfinite(agent[i]).all();valid[i]=True
        except Exception as e:errors.append(dict(index=i,error=f'{type(e).__name__}: {e}'))
    np.savez_compressed(dest,context=ctx,agent_response=agent,valid=valid)
    write(OUT/f'input_audit_{profile["name"]}.json',dict(invalid=errors,models_frozen_sha256=sha(SOURCE/'models_frozen.json'),labels_used=False))

def dataset():
    from . import diversity_confirmation as original
    original.OUT=OUT;original.SOURCE=SOURCE;original.dataset()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('flow','prepare','physical','features','filter','run','dataset','validate'));p.add_argument('--index',type=int,default=0);p.add_argument('--replicate',type=int,default=0);a=p.parse_args();replica(a.replicate)
    if a.action=='features':features(a.index)
    elif a.action in ('flow','prepare','physical','dataset'):globals()[a.action]()
    elif a.action=='validate':
        from . import state_support
        state_support.OUT=OUT;state_support.validate_plan()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run('held',a.index)
