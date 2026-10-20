"""Fit outcome-blind R5eta2 PCA on training controllers and freeze all probes."""
from core import *

def main():
    prereg()
    assert not (HERE/'NEW_LABELS_STARTED.receipt.json').exists()
    assert (HERE/'PROBES_FROZEN.receipt.json').exists()
    assert read(HERE/'PROBE_AUDIT.json')['success_rows_created']==0
    assert read(HERE/'STATE_DEDUP_AUDIT.json')['status']=='PASS'
    base=read(HERE/'FINAL_BASE_PROFILES.json')
    with np.load(HERE/'R5_FEATURES.npz',allow_pickle=False) as z:
        names=list(z['controller_names']);r5=z['features'].astype(np.float32)
    with np.load(HERE/'ETA2_FEATURES.npz',allow_pickle=False) as z:
        assert list(z['controller_names'])==names;eta2=z['features'].astype(np.float32)
    assert len(names)==109 and r5.shape==(109,160) and eta2.shape==(109,80)
    assert names[:56]==read(R15/'FINAL_BASE_PROFILES.json')['eligible_controller_order']
    assert names[56:73]==read(R17/'FINAL_BASE_PROFILES.json')['eligible_controller_order']
    assert names[73:]==base['eligible_controller_order']
    test=set(base['test_controller_order'])
    train_idx=np.array([i for i,n in enumerate(names) if n not in test],int)
    assert len(train_idx)==91 and len(test)==18
    r155=np.delete(r5,np.s_[105:110],axis=1)
    full=np.concatenate([r155,eta2],axis=1)
    assert full.shape==(109,235)
    x=full[train_idx];keep=np.isfinite(x).any(0);assert keep.sum()>=2
    x=x[:,keep]
    mean=np.nanmean(np.where(np.isfinite(x),x,np.nan),axis=0)
    fixed=np.where(np.isfinite(x),x,mean)
    scale=np.where(fixed.std(0)>1e-6,fixed.std(0),1)
    _,_,vt=np.linalg.svd((fixed-mean)/scale,full_matrices=False)
    pca=vt[:2].copy()
    for v in pca:
        if v[np.argmax(np.abs(v))]<0:v*=-1
    z2=((np.where(np.isfinite(full[:,keep]),full[:,keep],mean)-mean)/scale)@pca.T
    with np.load(R16/'fits/full_transform.npz',allow_pickle=False) as z:tr={k:z[k] for k in z.files}
    x5=r155[:,tr['r5_keep']]
    z5=((np.where(np.isfinite(x5),x5,tr['r5_mean'])-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T
    assert np.isfinite(z5).all() and np.isfinite(z2).all()
    np.savez_compressed(HERE/'R5ETA2_TRANSFORM.npz',keep=keep,mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32),pca=pca.astype(np.float32),
                        training_indices=train_idx,training_controller_names=np.array([names[i] for i in train_idx]))
    np.savez_compressed(HERE/'FEATURE_MATRIX.npz',controller_names=np.array(names),R5_PCA2=z5.astype(np.float32),
                        R5ETA2_PCA2=z2.astype(np.float32),eta2_raw=eta2,
                        eta2_mean=np.nanmean(np.where(np.isfinite(eta2[train_idx]),eta2[train_idx],np.nan),axis=1))
    files=['PREREGISTRATION.md','PREREGISTRATION.receipt.json',
           'PREREGISTRATION_AMENDMENT.md','PREREGISTRATION_AMENDMENT.receipt.json',
           'TRAINING_PLAN.json','FINAL_BASE_PROFILES.json',
           'STATE_DEDUP_AUDIT.json','TRAIN-new_STATES.json','TEST_STATES.json',
           'PROBES_FROZEN.receipt.json','PROBE_AUDIT.json','R5_FEATURES.npz','ETA2_FEATURES.npz',
           'R5ETA2_TRANSFORM.npz','FEATURE_MATRIX.npz','core.py','probe.py','features.py','states.py']
    write(HERE/'FEATURES_FROZEN.receipt.json',dict(status='FROZEN_BEFORE_ANY_NEW_SUCCESS_LABELS',utc=utc(),
          new_success_labels_generated=False,training_controllers=91,test_controllers_excluded=18,
          files={n:sha(HERE/n) for n in files},external_round16_transform_sha256=sha(R16/'fits/full_transform.npz')))
    features_frozen()
    print(json.dumps(dict(status='FROZEN',controllers=109,eta2_features=80,R5eta2_features=235,pca=2)))

if __name__=='__main__':main()
