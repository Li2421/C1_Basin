"""Freeze all Round17 Q selectors before any new success label is generated."""
from core import *
sys.path.insert(0,str(R16))
import fit as oldfit
import jax

def model_prob(model,seed,checkpoint,transform,physical,r5):
    assert checkpoint.exists()
    return oldfit.predict(model,seed,checkpoint,transform,physical,r5)

def main():
    prereg()
    assert not (HERE/'NEW_LABELS_STARTED.receipt.json').exists()
    assert (HERE/'R5_FROZEN.receipt.json').exists() and (HERE/'A3_ALL/TRAINING_AUDIT.json').exists()
    assert read(HERE/'STATE_DEDUP_AUDIT.json')['status']=='PASS'
    base=read(HERE/'FINAL_BASE_PROFILES.json');names=base['eligible_controller_order']
    assert len(names) in (13,17)
    with np.load(HERE/'R5_FEATURES.npz',allow_pickle=False) as z:
        assert list(z['controller_names'])==names
        r5=z['features'].astype(np.float32)
    states=read(HERE/'FRESH_STATES.json')
    assert len(states)==48
    physical=oldfit.physical23(states).astype(np.float32)
    with np.load(R16/'fits/full_transform.npz',allow_pickle=False) as z:
        r16_transform={k:z[k] for k in z.files}
    with np.load(HERE/'A3_ALL/transform.npz',allow_pickle=False) as z:
        all_transform={k:z[k] for k in z.files}
    selected=read(R16/'fits/selected_steps.json')
    frozen=read(R16/'MODEL_FREEZE.receipt.json')
    assert selected['A0']==9000 and selected['A3']==1000
    assert frozen['status']=='MODEL_RULES_HYPERPARAMETERS_FROZEN_BEFORE_EXTERNAL_OUTCOMES'
    out=HERE/'predictions';out.mkdir(exist_ok=True)
    predictions={};files={}
    for model,tr in [('A0',r16_transform),('A3',r16_transform),('A3_all',all_transform)]:
        fits=[]
        for seed in (17,23,41):
            if model=='A3_all':
                ck=HERE/'A3_ALL'/f'A3_all_seed{seed}_step1000.msgpack';architecture='A3'
            else:
                ck=R16/'fits/full'/model/f'seed{seed}'/f'step{selected[model]}.msgpack';architecture=model
                rel=str(ck.relative_to(R16));assert frozen['files'][rel]==sha(ck)
            p=model_prob(architecture,seed,ck,tr,physical,r5)
            assert p.shape==(len(names),48,8) and np.isfinite(p).all()
            dest=out/f'{model}_seed{seed}.npz'
            np.savez_compressed(dest,probability=p,controller_names=names,state_uids=[s['state_uid'] for s in states],arms=oldfit.ARMS)
            files[str(dest.relative_to(HERE))]=sha(dest)
            fits.append(p)
        predictions[model]=np.stack(fits)
    ge=read(R16/'predictions/G_E_FROZEN_CHOICES.json')
    assert ge['source']=='round15_only' and ge['ridge_penalty']==1
    x5=np.delete(r5,np.s_[105:110],axis=1)[:,r16_transform['r5_keep']]
    x5=np.where(np.isfinite(x5),x5,r16_transform['r5_mean'])
    x5=((x5-r16_transform['r5_mean'])/r16_transform['r5_scale'])@r16_transform['pca'].T
    ehat=(x5-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean']
    source_action=read(R16/'results.json')['source_mean_action']
    assert source_action in oldfit.ARMS
    rows=[]
    for i,name in enumerate(names):
        entry=dict(controller=name,group=base['profiles'][name]['group'],recipe=base['profiles'][name]['recipe'],
                   checkpoint_sha256=base['profiles'][name]['payload']['flow_checkpoint_sha256'],
                   G_E_Ehat=float(ehat[i]),G_E='F10' if ehat[i]<.05 else 'S500',source_mean=source_action)
        for model,p in predictions.items():
            mean=p.mean(0)[i]
            entry[model+'_fixed']=oldfit.ARMS[int(np.argmax(mean.mean(0)))]
            entry[model+'_state']=[oldfit.ARMS[int(k)] for k in np.argmax(mean,axis=1)]
            entry[model+'_predicted_E']=float(np.mean(mean[:,oldfit.ARMS.index('S500')]-mean[:,oldfit.ARMS.index('F10')]))
            entry[model+'_predicted_L']=float(np.mean(mean[:,oldfit.ARMS.index('S500')]-mean[:,oldfit.ARMS.index('F13')]))
        rows.append(entry)
    write(HERE/'FROZEN_CHOICES.json',dict(status='FROZEN_BEFORE_NEW_SUCCESS_LABELS',utc=utc(),
         controller_order=names,state_uids=[s['state_uid'] for s in states],arm_order=oldfit.ARMS,
         choice_rule='mean three fit probabilities across 48 states, then argmax with arm-order tie break',
         state_rule='mean three fit probabilities at each state, then argmax with arm-order tie break',
         source_mean_action=source_action,GE_threshold=.05,controllers=rows))
    files['FROZEN_CHOICES.json']=sha(HERE/'FROZEN_CHOICES.json')
    write(HERE/'CHOICES_FROZEN.receipt.json',dict(status='FROZEN_BEFORE_NEW_SUCCESS_LABELS',utc=utc(),
          files=files,round16_model_freeze_sha256=sha(R16/'MODEL_FREEZE.receipt.json'),
          round16_GE_sha256=sha(R16/'predictions/G_E_FROZEN_CHOICES.json'),
          A3_all_training_audit_sha256=sha(HERE/'A3_ALL/TRAINING_AUDIT.json'),
          R5_frozen_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),
          state_dedup_audit_sha256=sha(HERE/'STATE_DEDUP_AUDIT.json'),new_success_labels_generated=False))
    print(json.dumps(dict(status='FROZEN',controllers=len(rows),predictions=len(files)-1,choices=rows[:2])))

if __name__=='__main__':main()
