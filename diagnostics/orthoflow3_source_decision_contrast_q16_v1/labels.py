"""Create matched label arms without altering the original frozen dataset."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from .pipeline import OUT,SRC,read,write,sha,ETA_INDEX
from shared_rollout_db.src.rollout_db import connect,canonical

def main():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    src=SRC/'variants/large_20'
    d=dict(np.load(src/'dataset.npz'))
    rows=read(SRC/'pairs.json');selected=read(OUT/'pairs.json')
    ctl=read(OUT/'protocol.json')['selected_controller_uid']
    upgraded_success=d['success'].copy();upgraded_failure=d['failure'].copy()
    upgraded_std_s=d['standard_success'].copy();upgraded_std_f=d['standard_failure'].copy()
    full=np.zeros(len(rows),bool)
    numeric=[];labels=[]
    with connect(True) as db:
        for p in selected:
            i=p['state_index']*16+p['eta_index']
            assert rows[i]['state_uid']==p['state_uid'] and rows[i]['eta_uid']==p['eta_uid']
            assert d['split'][i]=='train' and d['valid'][2,i]
            found={}
            for row in db.execute('''SELECT seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality
                FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?''',
                (p['state_uid'],p['eta_uid'],ctl)):
                k=json.loads(row['seed_key']).get('future_index',-1)
                if 0<=k<16:
                    assert k not in found
                    assert row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined']
                    found[k]=row
            assert len(found)==16
            for k in range(16):
                if found[k]['numerical_failure']:numeric.append({'state_uid':p['state_uid'],'eta_uid':p['eta_uid'],'future_index':k})
            early=[found[k] for k in range(4)]
            early_good=[row for row in early if not row['numerical_failure']]
            assert int(sum(row['success'] for row in early_good))==int(d['success'][2,i])
            assert len(early_good)-int(d['success'][2,i])==int(d['failure'][2,i])
            good=[row for row in found.values() if not row['numerical_failure']]
            s=sum(row['success'] for row in good);f=len(good)-s
            assert s+f>=4
            upgraded_success[2,i]=s;upgraded_failure[2,i]=f
            upgraded_std_s[2,i]=s;upgraded_std_f[2,i]=f
            full[i]=(s+f==16)
            labels.append({'state_uid':p['state_uid'],'state_index':p['state_index'],
                'eta_uid':p['eta_uid'],'eta_index':p['eta_index'],
                'observed_success':int(s),'observed_failure':int(f),
                'full_16_observed':bool(full[i]),
                'B15':'CERTIFIED' if s>=15 else 'REFUTED' if f>=2 else 'UNRESOLVED',
                'initial_Q4_success':int(d['success'][2,i]),'initial_Q4_failure':int(d['failure'][2,i])})
    # A: original frozen counts. B: improved q but exactly four effective
    # trials, holding decision-pair influence fixed. C: original q x4 weight.
    arm_success={k:d['success'].copy() for k in ('original_Q4','Q16_rate_weight4','Q4_weight16')}
    arm_failure={k:d['failure'].copy() for k in arm_success}
    changed=np.array([p['state_index']*16+p['eta_index'] for p in selected],int)
    raw_n=upgraded_success[2,changed]+upgraded_failure[2,changed]
    arm_success['Q16_rate_weight4'][2,changed]=4*upgraded_success[2,changed]/raw_n
    arm_failure['Q16_rate_weight4'][2,changed]=4*upgraded_failure[2,changed]/raw_n
    arm_success['Q4_weight16'][2,changed]=4*d['success'][2,changed]
    arm_failure['Q4_weight16'][2,changed]=4*d['failure'][2,changed]
    for k in arm_success:
        assert np.isfinite(arm_success[k]).all() and np.isfinite(arm_failure[k]).all()
        assert np.all(arm_success[k]>=0) and np.all(arm_failure[k]>=0)
        if k!='Q4_weight16':
            np.testing.assert_allclose(arm_success[k][2,changed]+arm_failure[k][2,changed],4)
    # Same input arrays and eligible pairs in every arm; only supervision differs.
    for k in arm_success:
        np.savez_compressed(OUT/f'labels_{k}.npz',success=arm_success[k],failure=arm_failure[k])
    np.savez_compressed(OUT/'full_Q16_evidence.npz',success=upgraded_std_s,failure=upgraded_std_f,
                        full_16=full,upgraded_indices=changed)
    write(OUT/'upgraded_pair_evidence.json',labels)
    write(OUT/'label_integrity.json',{'old_dataset_sha256':sha(src/'dataset.npz'),
        'all_source_TRAIN_only':True,'pair_count':len(selected),'full_Q16_pairs':int(full[changed].sum()),
        'partially_observed_pairs':int((~full[changed]).sum()),'numeric_seeds':numeric,
        'original_Q4_kept_unchanged':True,'same_inputs_for_all_arms':True,
        'TRAIN_new_label_effective_trial_weight_per_pair':4,
        'all_new_rollouts_in_global_DB':True,
        'label_arms':{'original_Q4':'Frozen original observed count',
                      'Q16_rate_weight4':'New observed success rate; each selected pair keeps weight four',
                      'Q4_weight16':'Original rate; each selected pair weighted as sixteen trials'},
        'no_TARGET_or_VAL_labels_modified':True})
    print({'pairs':len(labels),'full_Q16_pairs':int(full[changed].sum()),
           'partially_observed_pairs':int((~full[changed]).sum()),'numerical':len(numeric)})

if __name__=='__main__':main()
