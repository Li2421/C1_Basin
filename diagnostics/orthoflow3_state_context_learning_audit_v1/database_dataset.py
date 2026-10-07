"""Refresh every canonical training pair from authoritative compatible seeds.

The frozen original snapshot stays intact for causal comparisons. No new rollout.
"""
import json
from collections import Counter
import numpy as np
from shared_rollout_db.src.rollout_db import connect
from .experiment import OUT,SRC,WIDE,read,write,sha

def main():
    if (OUT/'source_data_db.npz').exists():raise FileExistsError('Database-backed dataset already frozen')
    states=read(OUT/'source_states.json');profiles=read(SRC/'protocol.json')['profiles']
    eta_manifest=read(WIDE/'pairs.json')[:16];d=dict(np.load(OUT/'source_data.npz'))
    before=int((d['success']+d['failure']).sum());updates=[];keys=[]
    with connect(True) as db:
      for ci,c in enumerate(profiles):
       cfg=db.execute('SELECT compatibility_quality FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()
       assert cfg[0]=='EXACT_PROFILE'
       for si,state in enumerate(states):
        for ei,e in enumerate(eta_manifest):
            i=si*16+ei
            rows=db.execute('SELECT rollout_uid,seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(state['uid'],e['eta_uid'],c['controller_uid'])).fetchall()
            assert rows and all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rows)
            assert len({r['seed_key'] for r in rows})==len(rows)
            # This matrix has only standard future_index 0..15, so no historical
            # Q32/Q64 evidence can be silently dropped by this exact assertion.
            assert all(0<=json.loads(r['seed_key'])['future_index']<16 for r in rows)
            valid=[r for r in rows if not r['numerical_failure']]
            ss=sum(r['success'] for r in valid);ff=len(valid)-ss
            old_s=float(d['success'][ci,i]);old_f=float(d['failure'][ci,i])
            assert ss>=old_s and ff>=old_f,'Not a pure compatible-seed extension'
            if ss+ff>old_s+old_f:updates.append(dict(controller=c['name'],state_index=si,eta_index=ei,old_trials=old_s+old_f,new_trials=ss+ff))
            d['success'][ci,i]=ss;d['failure'][ci,i]=ff
            d['numerical'][ci,i]=len(rows)-len(valid)
            keys.append(dict(state_uid=state['uid'],eta_uid=e['eta_uid'],controller_uid=c['controller_uid'],
                rollout_uids=[r['rollout_uid'] for r in rows],observed_success=ss,observed_failure=ff,numerical=len(rows)-len(valid)))
    # The primary two-eta test labels must be bit-identical.
    prev=np.load(OUT/'source_data.npz');ix=np.isin(d['eta_index'],[10,15])
    for k in ('success','failure','context','eta'):
        if k=='eta':np.testing.assert_array_equal(prev[k],d[k])
        else:np.testing.assert_array_equal(prev[k][:,ix],d[k][:,ix])
    np.savez_compressed(OUT/'source_data_db.npz',**d)
    write(OUT/'database_dataset_keys.json',keys)
    write(OUT/'database_dataset_audit.json',dict(canonical_pairs=len(keys),source_families=46,
        original_observed_trials=before,current_observed_trials=int((d['success']+d['failure']).sum()),
        previously_omitted_valid_trials=int((d['success']+d['failure']).sum())-before,
        upgraded_pairs=len(updates),updates=updates,numerical_separate=int(d['numerical'].sum()),
        generator_modified=False,NEW_ROLLOUT=0,target_labels_used=False,
        exact_data_hash=sha(OUT/'source_data_db.npz')))
    write(OUT/'database_repair_protocol.json',dict(variants=['wide_db_eta_only','wide_db_full_raw','wide_db_full_scaled'],
        previous_variant_status='Historical-wide snapshot control only; not complete-current-DB training',
        decision_rule='Same source folds, architecture, objective, optimizer, steps; retain every compatible observed seed',
        max_extra_fits=27,target_labels_used=False,NEW_ROLLOUT=0))
    print({k:v for k,v in read(OUT/'database_dataset_audit.json').items() if k!='updates'})

if __name__=='__main__':main()
