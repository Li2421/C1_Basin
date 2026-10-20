"""Freeze C0/C1/C3/C2/A0 predictions and TEST choices before TEST labels."""
from core import *
import jax
import jax.numpy as jnp
from flax import serialization
sys.path.insert(0,str(R16))
import fit as oldfit

MODELS=('C0','C1','C3','C2','A0')
ARMS=oldfit.ARMS

def main():
    train_labels_allowed()
    assert read(HERE/'train_DELIVERY_AUDIT.json')['status']=='PASS'
    assert not (HERE/'TEST_FREEZE.receipt.json').exists()
    assert not (HERE/'test/raw').exists() or not list((HERE/'test/raw').glob('*.json'))
    base=read(HERE/'FINAL_BASE_PROFILES.json');names=base['test_controller_order']
    assert len(names)==18
    states=read(HERE/'TEST_STATES.json');assert len(states)==48
    physical=oldfit.physical23(states).astype(np.float32)
    with np.load(HERE/'FEATURE_MATRIX.npz',allow_pickle=False) as z:
        all_names=list(z['controller_names']);z5=z['R5_PCA2'];z2=z['R5ETA2_PCA2']
    idx=np.array([all_names.index(n) for n in names])
    with np.load(HERE/'R5_FEATURES.npz',allow_pickle=False) as z:
        assert list(z['controller_names'])==all_names;r5=z['features'][idx].astype(np.float32)
    with np.load(R16/'fits/full_transform.npz',allow_pickle=False) as z:r16tr={k:z[k] for k in z.files}
    r16freeze=read(R16/'MODEL_FREEZE.receipt.json')
    assert r16freeze['status']=='MODEL_RULES_HYPERPARAMETERS_FROZEN_BEFORE_EXTERNAL_OUTCOMES'
    out=HERE/'predictions';out.mkdir(exist_ok=True)
    prediction={};files=[]
    for model in MODELS:
        fits=[]
        for seed in (17,23,41):
            if model=='C0':
                ck=R16/'fits/full/A3'/f'seed{seed}'/'step1000.msgpack'
                assert r16freeze['files'][str(ck.relative_to(R16))]==sha(ck)
                p=oldfit.predict('A3',seed,ck,r16tr,physical,r5)
            else:
                audit=read(HERE/'critic'/model/'TRAINING_AUDIT.json')
                assert audit['status']=='FITTED_WITHOUT_TEST_LABELS'
                step=audit['selected_step'];ck=HERE/'critic'/model/f'seed{seed}_step{step}.msgpack'
                tr=HERE/'critic'/model/f'seed{seed}_transform.npz'
                assert any(x['seed']==seed and x['sha256']==sha(ck) and x['transform_sha256']==sha(tr) for x in audit['checkpoints'])
                with np.load(tr,allow_pickle=False) as z:mean=z['state_mean'];scale=z['state_scale']
                s=((physical-mean)/scale).astype(np.float32)
                state=np.broadcast_to(s[None,:,None,:],(18,48,8,23))
                action=np.broadcast_to(np.eye(8,dtype=np.float32)[None,None,:,:],(18,48,8,8))
                if model=='A0':x=np.concatenate([state,action],axis=-1)
                else:
                    z=z5[idx] if model=='C1' else z2[idx]
                    z=np.broadcast_to(z[:,None,None,:],(18,48,8,2))
                    x=np.concatenate([state,z,action],axis=-1)
                net=oldfit.net_for('A3')
                params=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,x.shape[-1]),jnp.float32)),ck.read_bytes())
                p=np.asarray(jax.nn.sigmoid(net.apply(params,jnp.asarray(x.reshape(-1,x.shape[-1]))))).reshape(18,48,8)
                files.extend([str(ck.relative_to(HERE)),str(tr.relative_to(HERE))])
            assert p.shape==(18,48,8) and np.isfinite(p).all()
            dest=out/f'{model}_seed{seed}.npz'
            np.savez_compressed(dest,probability=p,controller_names=names,state_uids=[s['state_uid'] for s in states],arms=ARMS)
            files.append(str(dest.relative_to(HERE)));fits.append(p)
        prediction[model]=np.stack(fits).mean(0)
    rows=[]
    for i,name in enumerate(names):
        row=dict(controller=name,recipe=base['profiles'][name]['recipe'],family=base['profiles'][name]['family'],
                 architecture=base['profiles'][name]['architecture'],seed_offset=base['profiles'][name]['seed_offset'],
                 checkpoint_sha256=base['profiles'][name]['payload']['flow_checkpoint_sha256'])
        for model,p in prediction.items():
            row[model+'_fixed']=ARMS[int(np.argmax(p[i].mean(0)))]
            row[model+'_state']=[ARMS[int(k)] for k in np.argmax(p[i],axis=1)]
            row[model+'_predicted_E']=float(np.mean(p[i,:,ARMS.index('S500')]-p[i,:,ARMS.index('F10')]))
            row[model+'_predicted_L']=float(np.mean(p[i,:,ARMS.index('S500')]-p[i,:,ARMS.index('F13')]))
        rows.append(row)
    write(HERE/'TEST_CHOICES.json',dict(status='FROZEN_BEFORE_TEST_LABELS',utc=utc(),controller_order=names,
          state_uids=[s['state_uid'] for s in states],arm_order=ARMS,
          fixed_rule='mean three fit probabilities over 48 TEST states, then argmax with arm order tie-break',
          state_rule='mean three fit probabilities per state, then argmax with arm order tie-break',controllers=rows))
    files+=['PREREGISTRATION.md','PREREGISTRATION.receipt.json',
            'PREREGISTRATION_AMENDMENT.md','PREREGISTRATION_AMENDMENT.receipt.json',
            'FEATURES_FROZEN.receipt.json',
            'train_LABELS.npz','train_DELIVERY_AUDIT.json','TEST_STATES.json','STATE_DEDUP_AUDIT.json',
            'R5ETA2_TRANSFORM.npz','FEATURE_MATRIX.npz','TEST_CHOICES.json',
            'protocol.json','test_CASES.json','test_OFFICIAL_PREFLIGHT.json',
            'core.py','critic_train.py','freeze_test.py','analyze.py','run.py','design.py','finalize_phase.py']
    for model in ('C1','C3','C2','A0'):
        files.extend([f'critic/{model}/CV_SELECTION.json',f'critic/{model}/TRAINING_AUDIT.json'])
    files=sorted(set(files))
    for n in files:assert (HERE/n).is_file(),n
    write(HERE/'TEST_FREEZE.receipt.json',dict(status='FROZEN_BEFORE_TEST_LABELS',utc=utc(),
          test_success_labels_generated_or_read=False,controllers=18,models=list(MODELS),
          files={n:sha(HERE/n) for n in files},external_round16_model_freeze_sha256=sha(R16/'MODEL_FREEZE.receipt.json')))
    test_labels_allowed()
    print(json.dumps(dict(status='FROZEN_BEFORE_TEST_LABELS',controllers=18,models=list(MODELS))))

if __name__=='__main__':main()
