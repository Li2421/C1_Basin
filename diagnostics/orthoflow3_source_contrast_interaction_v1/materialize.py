"""Construct exact observed-count source matrix and unchanged H20 inputs."""
from __future__ import annotations
import argparse,json,copy
import numpy as np
from .pipeline import OUT,SRC,read,write,configure,CONTROLLERS,ETA_INDEX
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from shared_rollout_db.src.rollout_db import connect,canonical

SHARDS=2

def physical():
    configure()
    native.physical()
    generated=read(OUT/'physical.json')
    old={r['state_uid']:r for r in read(SRC/'physical.json')}
    for r in generated[:46]:
        q=old[r['state_uid']]
        for k in ('positions','velocities','goals','flow'):
            np.testing.assert_allclose(r['physical'][k],q['physical'][k],atol=1e-6,rtol=0)
    write(OUT/'physical_audit.json',{'old_source_states_replayed_identically':46,
        'new_independent_source_states':len(generated)-46,
        'entity_semantics_unchanged':True})

def features(controller,shard,shards):
    assert controller in CONTROLLERS and shards==SHARDS and 0<=shard<shards
    target=OUT/f'features_{controller}_{shard}of{shards}.json'
    if target.exists():print({'already_complete':str(target)});return
    h20.rc.PROTOCOL={**h20.rc.PROTOCOL,'horizon_steps':20}
    h20.rc.OUT=OUT
    profile=next(p for p in read(OUT/'protocol.json')['profiles'] if p['name']==controller)
    runtime=h20.rc.RichRuntime('ring_exchange',profile['path'])
    assert profile['sha256']==runtime.alt_sha
    old_dataset=np.load(SRC/'variants/large_20/dataset.npz')
    old_index={s['uid']:i for i,s in enumerate(read(SRC/'states.json'))}
    pairs=read(OUT/'pairs.json');physical=read(OUT/'physical.json')
    ci=('base','alt','second').index(controller)
    results=[]
    for i,p in enumerate(pairs):
        if i%shards!=shard:continue
        if p['state_index']<46:
            original=old_index[p['state_uid']]*16+p['eta_index']
            assert old_dataset['valid'][ci,original]
            value=np.asarray(old_dataset['context'][ci,original],float)
            assert np.isfinite(value).all()
            results.append({'pair_index':i,'valid':True,'context':value.tolist(),
                'source':'reused_identical_H20_physical_response'})
        else:
            state=physical[p['state_index']]
            assert state['state_uid']==p['state_uid']
            response=h20.rc.cached(runtime,state,p['eta'])
            results.append({'pair_index':i,'valid':bool(response['valid']),
                'context':response['features']['mean'] if response['valid'] else None,
                'error':response.get('error'),'source':'new_H20_physical_response'})
        if len(results)%16==0:print({'controller':controller,'shard':shard,'done':len(results)},flush=True)
    write(target,results)
    print({'controller':controller,'shard':shard,'complete':len(results),
           'invalid':sum(not r['valid'] for r in results)},flush=True)

def materialize():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    pairs=read(OUT/'pairs.json');states=read(OUT/'states.json');n=len(pairs)
    contexts=np.full((2,n,24),np.nan,np.float32)
    for ci,controller in enumerate(CONTROLLERS):
        found=set()
        for shard in range(SHARDS):
            for r in read(OUT/f'features_{controller}_{shard}of{SHARDS}.json'):
                i=r['pair_index'];assert i not in found;found.add(i)
                if r['valid']:contexts[ci,i]=r['context']
        assert len(found)==n
    success=np.zeros((2,n),np.float32);failure=np.zeros((2,n),np.float32)
    numerical=np.zeros((2,n),np.int16);old_s=np.zeros((2,n),np.float32);old_f=np.zeros((2,n),np.float32)
    source=np.load(SRC/'variants/large_20/dataset.npz')
    old_state={s['uid']:i for i,s in enumerate(read(SRC/'states.json'))}
    profiles={p['name']:p for p in read(OUT/'protocol.json')['profiles']}
    with connect(True) as db:
        for ci,controller in enumerate(CONTROLLERS):
            for i,p in enumerate(pairs):
                raw=db.execute('''SELECT seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality
                    FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?''',
                    (p['state_uid'],p['eta_uid'],profiles[controller]['controller_uid'])).fetchall()
                selected={}
                for r in raw:
                    seed=json.loads(r['seed_key']).get('future_index',-1)
                    if 0<=seed<16:
                        assert seed not in selected
                        assert not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE'
                        selected[seed]=r
                assert len(selected)==16
                good=[r for r in selected.values() if not r['numerical_failure']]
                success[ci,i]=sum(r['success'] for r in good)
                failure[ci,i]=len(good)-success[ci,i]
                numerical[ci,i]=16-len(good)
                assert success[ci,i]+failure[ci,i]>=14
                if p['split']=='train':
                    old_i=old_state[p['state_uid']]*16+p['eta_index']
                    old_c=('base','alt','second').index(controller)
                    old_s[ci,i]=source['success'][old_c,old_i]
                    old_f[ci,i]=source['failure'][old_c,old_i]
                    valid_first=[r for seed,r in selected.items() if seed<4 and not r['numerical_failure']]
                    assert sum(r['success'] for r in valid_first)==old_s[ci,i]
                    assert len(valid_first)-old_s[ci,i]==old_f[ci,i]
    raw_valid=np.isfinite(contexts).all(-1)
    # Input-quality exclusion is state-level and independent of success labels.
    state_input_valid=np.array([raw_valid[:,2*i:2*i+2].all() for i in range(len(states))])
    split=np.array([p['split'] if state_input_valid[p['state_index']] else 'unused' for p in pairs])
    valid=raw_valid & (success+failure>0) & (split[None,:]!='unused')
    contexts=np.where(np.isfinite(contexts),contexts,0.)
    np.savez_compressed(OUT/'dataset.npz',success=success,failure=failure,numerical=numerical,
        original_Q4_success=old_s,original_Q4_failure=old_f,context=contexts,valid=valid,
        eta=np.asarray([p['eta'] for p in pairs],np.float32),
        eta_index=np.asarray([p['eta_index'] for p in pairs],int),
        state_index=np.asarray([p['state_index'] for p in pairs],int),split=split)
    tr=np.flatnonzero(split=='train');val=np.flatnonzero(split=='validation')
    assert len(set(pairs[i]['state_index'] for i in tr))>=40
    write(OUT/'materialize_audit.json',{'source_train_families':len(set(pairs[i]['state_index'] for i in tr)),
        'independent_validation_families':len(set(pairs[i]['state_index'] for i in val)),
        'input_quality_excluded_state_indices':np.flatnonzero(~state_input_valid).tolist(),
        'source_train_Q4_seed_counts_aligned':True,'controller_names':list(CONTROLLERS),
        'exact_eta_indices':list(ETA_INDEX),'numeric_outcomes_not_imputed':int(numerical.sum()),
        'valid_16_seed_pairs':int((numerical==0).sum()),
        'controller_by_split_observed_trials':{
            x:{c:int((success[ci,split==x]+failure[ci,split==x]).sum())
               for ci,c in enumerate(CONTROLLERS)} for x in ('train','validation')}})
    print(read(OUT/'materialize_audit.json'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('physical','features','materialize'))
    ap.add_argument('--controller',choices=CONTROLLERS,default='alt')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=SHARDS)
    ap.add_argument('--index',type=int,default=None)
    a=ap.parse_args()
    if a.index is not None:
        assert a.action=='features' and 0<=a.index<len(CONTROLLERS)*SHARDS
        a.controller=CONTROLLERS[a.index//SHARDS];a.shard=a.index%SHARDS
    {'physical':physical,'features':lambda:features(a.controller,a.shard,a.shards),
      'materialize':materialize}[a.action]()
if __name__=='__main__':main()
