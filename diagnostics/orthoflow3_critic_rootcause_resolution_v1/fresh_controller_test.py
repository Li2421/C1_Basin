"""New-controller transfer of already frozen source-family critics.

No target task label is available when the Flow asset and critics freeze.
The exact same64 independent state families/two source eta remain fixed.
This is a source-geometry/controller-held-out diagnostic, not cross-scene.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from . import seed_replication as engine
from . import support_confirmation as confirmation
from .fresh_flow import OUT as FLOW,RULE as FLOW_RULE
from .state_support import OUT as SOURCE
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
OUT=SOURCE.parent/'fresh_controller_confirmation';EXP='exp_orthoflow3_rootcause_fresh_controller_seed88124_v1'
read=engine.read;write=engine.write

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Fresh target protocol frozen')
    flow=read(FLOW/'frozen_controller.json');assert flow['target_feasibility_labels_seen'] is False
    assert sha(FLOW/'best.pkl')==flow['checkpoint_sha256'];assert sha(SOURCE/'models_frozen.json')==flow['models_frozen_sha256']
    previous=read(confirmation.OUT/'protocol.json');reference=previous['profiles'][0]
    with connect() as db,transaction(db):
        r=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(reference['controller_uid'],)).fetchone();payload=json.loads(r['config_json'])
        payload['flow_checkpoint_sha256']=flow['checkpoint_sha256'];cid=uid('ctl',payload)
        assert db.execute('SELECT 1 FROM controller_config WHERE controller_uid=?',(cid,)).fetchone() is None
        db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cid,r['scenario_uid'],flow['checkpoint_sha256'],r['orthoflow3_sha256'],r['safety_config_hash'],r['horizon'],r['dt'],r['success_semantics_version'],payload['conditioning'],r['rng_semantics_version'],canonical(payload),'EXACT_PROFILE'))
        profile=dict(name='fresh',path=str(FLOW/'best.pkl'),sha256=flow['checkpoint_sha256'],controller_uid=cid)
        proto={**previous,'experiment_uid':EXP,'experiment':'fresh_controller_seed88124_confirmation','profiles':[profile],
            'requested_seed_count':2048,'models_frozen_sha256':sha(SOURCE/'models_frozen.json'),
            'target_controller_rule_sha256':sha(FLOW_RULE),
            'primary_model':'expanded206_pair_equal/entity_response_mean, seeds17/23/41; source VAL only',
            'primary_baselines':['matched eta_only','correct versus wrong source controller responses'],
            'secondary':'All previously frozen transferable model arms; exclude privileged known-controller-ID models from new-controller use',
            'scope':'New controller parameters, independent source families, same exact two source eta. Not unseen-eta or cross-scene zero-shot.',
            'selection_metrics':'Report all64 families, oracle eligibility, uninformative both-B15 and no-known-B15 separately; no outcome-driven exclusion or re-selection',
            'success_safety_rng_unchanged':True,'target_controller_labels_used_for_training':False}
        write(OUT/'protocol.json',proto)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'fresh_controller_seed88124_frozen_critic_confirmation',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'target_labels_for_training':0})))
    for name in ('states.json','pairs.json','physical.json','entities.npz'):shutil.copyfile(confirmation.OUT/name,OUT/name)
    requests=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=cid,seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in read(OUT/'pairs.json')]
    write(OUT/'planned_rollouts.json',{'requests':requests});write(OUT/'planned_fresh.json',{'requests':requests})
    write(OUT/'working_state.json',dict(phase='prepared',controller_created_after_models_frozen=True))

def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.parent.CONTROLLERS=('fresh',)
    engine.SHARDS=12;engine.parent.SHARDS=12

def features():
    confirmation.OUT=OUT;confirmation.features(0)

def dataset():
    if (OUT/'dataset.npz').exists():raise FileExistsError('Target evidence table frozen')
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    profile=read(OUT/'protocol.json')['profiles'][0];pairs=read(OUT/'pairs.json');s=[];f=[];nums=[];keys=[]
    with connect(True) as db:
      for p in pairs:
        rows=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],profile['controller_uid'])).fetchall()
        assert len(rows)==16 and {r['seed_key'] for r in rows}=={canonical({'future_index':k}) for k in range(16)}
        assert all(r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'] for r in rows)
        valid=[r for r in rows if not r['numerical_failure']];success=sum(r['success'] for r in valid)
        s.append(success);f.append(len(valid)-success);nums.append(16-len(valid))
        keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rows]))
    inputs=np.load(OUT/'inputs_fresh.npz')
    np.savez_compressed(OUT/'dataset.npz',success=np.array(s),failure=np.array(f),numerical=np.array(nums),
        context=inputs['context'],agent_response=inputs['agent_response'],valid=inputs['valid'],
        eta=np.array([p['eta'] for p in pairs],np.float32),state_index=np.array([p['state_index'] for p in pairs]),eta_index=np.array([p['eta_index'] for p in pairs]))
    write(OUT/'dataset_DB_keys.json',keys)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','filter','features','run','dataset'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action in ('prepare','features','dataset'):globals()[a.action]()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run('fresh',a.index)
