"""Extend immutable C5 labels by nine q0 actions; unknown is not unobserved."""
from common import *

def main():
    assert (HERE/'training_COMPLETE.receipt.json').exists();old=D/'crosstype5_repair_gate_late_eta_v1/TRAINING_BUNDLE.npz'
    with np.load(old,allow_pickle=False) as z:prior={k:z[k] for k in z.files}
    with np.load(HERE/'TRAINING_INPUTS.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert np.array_equal(data['action'][:,:,:44,:14],prior['action'])
    for k in ('labels','collision','timeout','unknown'):
        data[k]=np.full((122,48,53,2),-2,np.int8);data[k][:,:,:44]=prior[k]
    data['uids']=np.full((122,48,53,2),'',dtype='<U96');data['uids'][:,:,:44]=prior['uids'];aliases=read(HERE/'training_LOCAL_EQUIVALENCE.json');names={str(n):i for i,n in enumerate(data['names'])}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'training_CASES.json'):
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();r=dict(r) if r is not None else aliases[c['rollout_uid']]['outcome'];idx=(names[c['base_controller_name']],c['state_index'],c['training_slot'],0);unknown=bool(r['numerical_failure'])
            data['labels'][idx]=-1 if unknown else int(r['success']);data['unknown'][idx]=int(unknown)
            for k in ('collision','timeout'):data[k][idx]=-1 if unknown else int(r[k])
            data['uids'][idx]=c['rollout_uid']
    with np.load(HERE/'CONTROLLER_FEATURES_V2.npz',allow_pickle=False) as z:
        assert np.array_equal(z['names'],data['names']);data['features']=z['features'];data['feature_names']=z['feature_names']
    data['state_uids']=prior['state_uids'];theta=data['thresholds'];codes=np.zeros(theta.shape,np.int32)
    for j,t in enumerate((.90,.93,.95,.968,.98,.9681620884830543),1):codes[np.isclose(theta,t,rtol=0,atol=1e-12)]=j
    for j,step in enumerate((200,300,400,500,600),7):codes[(data['action'][...,12]==0)&np.isclose(data['switch_index'],step/700,rtol=0,atol=1e-12)]=j
    data['candidate_indices']=(data['pair_index']*4096+codes*128+data['trigger_indices']*16+(data['action'][...,12]>.5)*8+data['contains_zero']*2+data['diagonal']).astype(np.int32)
    assert not np.any(data['labels'][...,0]==-2)
    for k in ('labels','collision','timeout','unknown','uids'):assert np.array_equal(data[k][:,:,:44],prior[k])
    np.savez_compressed(HERE/'TRAINING_BUNDLE.npz',**data)
    write(HERE/'TRAINING_BUNDLE_SUMMARY.json',dict(utc=utc(),shape=data['labels'].shape,known=int(np.sum(data['labels']>=0)),unknown=int(np.sum(data['labels']==-1)),unobserved=int(np.sum(data['labels']==-2)),old44_all_seeds_preserved=True,feature_dimension=39,action_dimension=19))
    seal('TRAINING_BUNDLE.receipt.json',[HERE/'TRAINING_BUNDLE.npz',HERE/'TRAINING_BUNDLE_SUMMARY.json',HERE/'training_COMPLETE.receipt.json',HERE/'CONTROLLER_FEATURES_V2.receipt.json',old],old_physical_labels_not_rerun=True)

if __name__=='__main__':main()
