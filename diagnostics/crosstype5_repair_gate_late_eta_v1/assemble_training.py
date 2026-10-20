"""Keep all original labels, distinguish unobserved seeds from numerical unknowns."""
from common import *

def main():
    assert (HERE/'training_COMPLETE.receipt.json').exists()
    old=D/'crosstype4_no_repair_schedule_diagnosis_v1/TRAINING_BUNDLE.npz'
    with np.load(old,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(HERE/'TRAINING_INPUTS.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert np.array_equal(data['names'],base['names'])
    assert np.array_equal(data['action'][:,:,:34,:13],base['action'])
    shape=(122,48,44,2);events=('labels','collision','timeout','unknown')
    for k in events:
        data[k]=np.full(shape,-2,np.int8);data[k][:,:,:34,0]=base[k][...,0]
    data['uids']=np.full(shape,'',dtype='<U96');data['uids'][:,:,:34,0]=base['uids'][...,0]
    aliases=read(HERE/'training_ALL_LOCAL_EQUIVALENCE.json');name_index={n:i for i,n in enumerate(data['names'])}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'training_CASES.json'):
            rid=c['rollout_uid'];record=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
            outcome=dict(record) if record is not None else aliases[rid]['outcome']
            # Case controller names are explicit, and seed identity is never guessed.
            ni=name_index[c['base_controller_name']];si=c['state_index'];slot=c['training_slot'];q=c['seed']
            assert q in (0,1) and (q==0 or slot==26)
            idx=(ni,si,slot,q);assert data['labels'][idx]==-2
            unknown=bool(outcome['numerical_failure'])
            data['labels'][idx]=-1 if unknown else int(outcome['success'])
            for key in ('collision','timeout'):data[key][idx]=-1 if unknown else int(outcome[key])
            data['unknown'][idx]=int(unknown);data['uids'][idx]=rid
    assert not np.any(data['labels'][...,0]==-2)
    assert not np.any(data['labels'][:,:,26,1]==-2)
    mask=np.ones(44,bool);mask[26]=False;assert np.all(data['labels'][:,:,mask,1]==-2)
    data['features']=base['features'];data['state_uids']=base['state_uids']
    data['candidate_indices']=(data['pair_index']*32+data['switch_index']*4+data['contains_zero']*2+data['diagonal']).astype(np.int32)
    np.savez_compressed(HERE/'TRAINING_BUNDLE.npz',**data)
    write(HERE/'TRAINING_BUNDLE_SUMMARY.json',dict(utc=utc(),shape=shape,known=int(np.sum(data['labels']>=0)),unknown=int(np.sum(data['labels']==-1)),structural_unobserved=int(np.sum(data['labels']==-2)),original_labels_preserved=True,features_missing_controllers=data['names'][~np.isfinite(data['features']).all(1)].tolist(),metadata_encoding='pair*32+switch*4+contains_zero*2+diagonal'))
    seal('TRAINING_BUNDLE.receipt.json',[HERE/'TRAINING_BUNDLE.npz',HERE/'TRAINING_BUNDLE_SUMMARY.json',HERE/'training_COMPLETE.receipt.json',HERE/'TRAINING_INPUTS.npz',HERE/'assemble_training.py',old],original_execution_labels_preserved=True)
if __name__=='__main__':main()
