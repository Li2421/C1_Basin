"""Canonical observed-count table: old46 + new160 TRAIN, new32 VAL.

All compatible old Q16/Q64 is retained identically in both data-size arms.
The fresh-seed diagnostic is now historical source evidence, not confirmation
for these models. Independent64-family confirmation remains sealed.
"""
import json
from pathlib import Path
import numpy as np
from .state_support import OUT,read,write
from shared_rollout_db.src.rollout_db import connect

ROOT=OUT.parent.parent
AUDIT=ROOT/'orthoflow3_state_context_learning_audit_v1'
SRC=ROOT/'orthoflow3_source_contrast_interaction_v1'

def materialize():
    if (OUT/'dataset.npz').exists():raise FileExistsError('Support dataset already frozen')
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    old_states=read(AUDIT/'source_states.json');new_states=[s for s in read(OUT/'states.json') if s['split']!='test']
    states=old_states+new_states;assert len(states)==238
    old_pairs=[p for p in read(SRC/'pairs.json') if p['split']=='train']
    new_pairs=read(OUT/'pairs.json');pairs=[]
    for p in old_pairs:pairs.append({**p,'dataset_state_index':p['state_index'],'source':'old46'})
    for p in new_pairs:pairs.append({**p,'dataset_state_index':46+p['state_index'],'source':'new160' if p['split']=='train' else 'new32val'})
    assert len(pairs)==476 and len({(p['state_uid'],p['eta_uid']) for p in pairs})==476
    old=np.load(AUDIT/'source_data_db.npz');indices=np.array([p['state_index']*16+p['eta_index'] for p in old_pairs])
    contexts=np.zeros((2,476,24),np.float32);responses=np.zeros((2,476,4,16),np.float32);valid=np.ones((2,476),bool)
    for ci,c in enumerate(('alt','second')):
        contexts[ci,:92]=old['context'][ci,indices]
        f=np.load(OUT.parent/'agent_response'/f'features_{c}.npz')
        assert np.array_equal(f['indices'],indices);responses[ci,:92]=f['agent_response']
        n=np.load(OUT/f'inputs_{c}.npz');contexts[ci,92:]=n['context'];responses[ci,92:]=n['agent_response'];valid[ci,92:]=n['valid']
    xold=np.load(AUDIT/'source_entities.npz');xnew=np.load(OUT/'entities.npz')
    x={k:np.concatenate([xold[k],xnew[k][:192]],axis=0) for k in xold.files}
    success=np.zeros((2,476),np.float32);failure=np.zeros_like(success);numerical=np.zeros_like(success,dtype=int)
    standard_success=np.zeros_like(success);standard_failure=np.zeros_like(success)
    keys=[];profiles=read(OUT/'protocol.json')['profiles']
    with connect(True) as db:
      for ci,c in enumerate(profiles):
       for i,p in enumerate(pairs):
        records=db.execute('SELECT rollout_uid,seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid'])).fetchall()
        assert records and all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in records)
        assert len({r['seed_key'] for r in records})==len(records)
        for r in records:
            seed=json.loads(r['seed_key'])['future_index']
            if p['source']!='old46':assert seed<p['target_seeds']
            if r['numerical_failure']:numerical[ci,i]+=1
            else:
                success[ci,i]+=r['success'];failure[ci,i]+=1-r['success']
                if seed<16:standard_success[ci,i]+=r['success'];standard_failure[ci,i]+=1-r['success']
        keys.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],
            rollout_uids=[r['rollout_uid'] for r in records],success=int(success[ci,i]),failure=int(failure[ci,i]),numerical=int(numerical[ci,i])))
    valid &= success+failure>0
    split=np.array([p['split'] for p in pairs]);si=np.array([p['dataset_state_index'] for p in pairs]);ei=np.array([p['eta_index'] for p in pairs])
    assert not ({states[i]['source_group'] for i in set(si[split=='train'])}&{states[i]['source_group'] for i in set(si[split=='validation'])})
    np.savez_compressed(OUT/'dataset.npz',success=success,failure=failure,numerical=numerical,
        standard_success=standard_success,standard_failure=standard_failure,context=contexts,agent_response=responses,valid=valid,
        eta=np.array([p['eta'] for p in pairs],np.float32),state_index=si,eta_index=ei,split=split,
        source=np.array([p['source'] for p in pairs]))
    np.savez_compressed(OUT/'model_entities.npz',**x)
    write(OUT/'model_states.json',states);write(OUT/'dataset_DB_keys.json',keys);write(OUT/'model_pairs.json',pairs)
    audit=dict(states=len(states),train_states=206,val_states=32,canonical_controller_pairs=len(keys),
        old_counts='All compatible observed seeds including new replication Q64; identical old46 data in both size arms',
        observed_trials=int((success+failure).sum()),numerical_attempts=int(numerical.sum()),
        invalid_response_controller_pairs=int((~valid).sum()),
        invalid_response_train_pair_indices=np.flatnonzero((split=='train')&~valid.all(0)).tolist(),
        invalid_response_val_pair_indices=np.flatnonzero((split=='validation')&~valid.all(0)).tolist(),
        confirmation_labels_used=False,confirmation_state_features_in_model_data=False,
        training_weight='controller-balanced observed Bernoulli counts, no fake full Q16, no pair duplication')
    write(OUT/'dataset_audit.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':materialize()
