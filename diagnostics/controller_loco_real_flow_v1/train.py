"""One LOCO fold/representation; three source-only fits, Round 3 optimizer."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import numpy as np
import flax.linen as nn
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23

ARMS=['R0','R3','R4_action','R4_openloop','R3_R4_action']
def read(p):return json.loads(Path(p).read_text())

class Net(nn.Module):
    @nn.compact
    def __call__(self,x):
        x=nn.silu(nn.Dense(64,name='dense1')(x))
        x=nn.silu(nn.Dense(64,name='dense2')(x))
        return nn.Dense(16,name='out')(x)

def descriptor(z,arm):
    out=np.zeros((len(z['R3']),320),np.float32)
    if arm in ('R3','R3_R4_action'):out[:,:256]=z['R3']
    if arm=='R4_action':out[:,:64]=z['R4_action']
    if arm=='R4_openloop':out[:,:64]=z['R4_openloop']
    if arm=='R3_R4_action':out[:,256:320]=z['R4_action']
    return out

def build(fold,arm,stage_b=False):
    p=read(HERE/'protocol.json');names=p['controller_order'];sources=[i for i in range(7) if i!=fold]
    source_names=[names[i] for i in sources]
    if stage_b:source_names+=read(HERE/'stage_b/protocol.json')['controller_order']
    phys=physical23(p['states']);p1=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')
    valphys=physical23(p1['states'])
    train=[];validation=[]
    for name in source_names:
        source_dir=HERE/'static_features' if name in names else HERE/'stage_b/static_features'
        z=np.load(source_dir/f'{name}.npz')
        v=np.load(source_dir/f'{name}_validation.npz')
        expected=p['states'] if name in names else p['states'][:128]
        assert list(z['state_uids'])==[s['state_uid'] for s in expected]
        assert list(v['state_uids'])==[s['state_uid'] for s in p1['states']]
        train.append(np.concatenate((phys[:128],descriptor(z,arm)[:128]),axis=-1))
        validation.append(np.concatenate((valphys,descriptor(v,arm)),axis=-1))
    held_name=names[fold];heldz=np.load(HERE/'static_features'/f'{held_name}.npz')
    held_all=np.concatenate((phys,descriptor(heldz,arm)),axis=-1)
    held=held_all[128:]
    mismatch=[]
    for c in sources:
        z=np.load(HERE/'static_features'/f'{names[c]}.npz')
        mismatch.append(np.concatenate((phys[128:],descriptor(z,arm)[128:]),axis=-1))
    train=np.asarray(train,np.float32);validation=np.asarray(validation,np.float32)
    held=np.asarray(held,np.float32);mismatch=np.asarray(mismatch,np.float32)
    assert train.shape==(len(source_names),128,343) and validation.shape==(len(source_names),24,343)
    assert held.shape==(64,343) and mismatch.shape==(6,64,343)
    assert all(np.isfinite(x).all() for x in (train,validation,held,mismatch))
    return source_names,train,validation,held,mismatch,held_all[:4]

def main(fold,arm,stage_b=False):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    assert arm in ARMS and 0<=fold<7
    jax.config.update('jax_enable_x64',False)
    assert any(d.platform=='gpu' for d in jax.devices()),jax.devices()
    assert read(HERE/'LABEL_AUDIT.json')['status']=='COMPLETE EXACT UID LABEL TENSOR'
    if stage_b:
        assert read(HERE/'stage_b/LABEL_AUDIT.json')['status']=='COMPLETE EXACT UID LABEL TENSOR'
        assert fold in read(HERE/'GATE_RESULTS.json')['necessary_folds']
    folder=(HERE/'stage_b/fits' if stage_b else HERE/'fits')/f'fold{fold}_{arm}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print('already complete',folder);return
    source_names,tr,va,held,mismatch,cal_input=build(fold,arm,stage_b)
    names=read(HERE/'protocol.json')['controller_order']
    orig=np.load(HERE/'LABELS.npz')['outcomes'][:,:128,:,:4]
    extra=np.load(HERE/'stage_b/LABELS.npz')['outcomes'] if stage_b else None
    extra_names=read(HERE/'stage_b/protocol.json')['controller_order'] if stage_b else []
    lab=np.asarray([orig[names.index(n)] if n in names else extra[extra_names.index(n)] for n in source_names])
    S=(lab==1).sum(-1).astype(np.float32);F=(lab==0).sum(-1).astype(np.float32)
    p1=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_LABELS.npz')
    p1names=list(p1['controller_names'])
    SV=np.asarray([p1['Y'][p1names.index(n)].sum(-1) for n in source_names],np.float32)
    FV=np.asarray([(4-p1['Y'][p1names.index(n)]-p1['U'][p1names.index(n)]).sum(-1) for n in source_names],np.float32)
    mean=tr.mean((0,1));std=tr.std((0,1));scale=np.where(std>1e-6,std,1.)
    X=jnp.asarray((tr-mean)/scale);V=jnp.asarray((va-mean)/scale)
    H=jnp.asarray((held-mean)/scale);M=jnp.asarray((mismatch-mean)/scale)
    C=jnp.asarray((cal_input-mean)/scale)
    Sj=jnp.asarray(S);Fj=jnp.asarray(F)
    nsource=len(source_names)
    banks=[np.argwhere(S[i]+F[i]>0).astype(np.int32) for i in range(nsource)]
    assert all(len(b)>0 for b in banks)
    nmean=float(np.mean([S[i,j,k]+F[i,j,k] for i,b in enumerate(banks) for j,k in b]))
    valid=float((SV+FV).sum());assert valid>0
    net=Net();opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
    @jax.jit
    def pred(pa,x):return net.apply(pa,x)
    @jax.jit
    def block(pa,op,cc,ii,jj):
        def step(carry,ids):
            params,state=carry;c,i,j=ids
            def loss(pp):
                z=pred(pp,X[c,i]);chosen=jnp.take_along_axis(z,j[:,None],axis=1)[:,0]
                return jnp.mean(Sj[c,i,j]*jax.nn.softplus(-chosen)+Fj[c,i,j]*jax.nn.softplus(chosen))/nmean
            _,grad=jax.value_and_grad(loss)(params)
            updates,state=opt.update(grad,state,params)
            return (optax.apply_updates(params,updates),state),None
        return jax.lax.scan(step,(pa,op),(cc,ii,jj))[0]
    schedule=[50,100]+list(range(200,12001,200));hists=[];snapshots=[];start=time.time()
    for seed in (17,23,41):
        rng=np.random.default_rng(seed);pa=net.init(jax.random.PRNGKey(seed),X[:1,:1]);op=opt.init(pa)
        prev=0;hist=[];snap=[]
        for target in schedule:
            cc=rng.integers(nsource,size=(target-prev,128));ii=np.empty_like(cc);jj=np.empty_like(cc)
            for code in range(nsource):
                where=cc==code;b=banks[code];draw=b[rng.integers(len(b),size=int(where.sum()))]
                ii[where]=draw[:,0];jj[where]=draw[:,1]
            pa,op=block(pa,op,jnp.asarray(cc),jnp.asarray(ii),jnp.asarray(jj))
            z=np.asarray(pred(pa,V.reshape(-1,343))).reshape(nsource,24,16)
            nll=float((SV*np.logaddexp(0,-z)+FV*np.logaddexp(0,z)).sum()/valid)
            hist.append(nll);snap.append(serialization.to_bytes(pa));prev=target
        hists.append(hist);snapshots.append(snap)
    selected=int(np.argmin(np.mean(hists,axis=0)));step=schedule[selected]
    held_pred=[];wrong_pred=[];val_pred=[];cal_pred=[];paths=[]
    for seed,snap in zip((17,23,41),snapshots):
        path=folder/f'seed{seed}_step{step}.msgpack';path.write_bytes(snap[selected]);paths.append(str(path))
        pa=serialization.from_bytes(pa,snap[selected])
        held_pred.append(np.asarray(pred(pa,H)))
        cal_pred.append(np.asarray(pred(pa,C)))
        wrong_pred.append(np.asarray(pred(pa,M.reshape(-1,343))).reshape(6,64,16))
        val_pred.append(np.asarray(pred(pa,V.reshape(-1,343))).reshape(nsource,24,16))
    np.savez_compressed(folder/'predictions.npz',held=np.asarray(held_pred),mismatched=np.asarray(wrong_pred),source_validation=np.asarray(val_pred),target_calibration=np.asarray(cal_pred))
    np.savez_compressed(folder/'normalization.npz',mean=mean,scale=scale)
    rec=dict(fold=fold,held_controller=names[fold],source_controller_names=source_names,arm=arm,fit_seeds=[17,23,41],
             selected_step=step,validation_nll_history=hists,checkpoints=paths,train_pairs=[len(b) for b in banks],
             validation_states_per_source=24,jax_enable_x64=False,device=[str(d) for d in jax.devices()],
             optimizer='AdamW 0.0008 wd0.0001 clip5',elapsed_seconds=time.time()-start,
             test_labels_read=False,source_only_normalization=True,stage_b=stage_b)
    (folder/'complete.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(fold=fold,arm=arm,selected_step=step,seconds=rec['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('fold',type=int);ap.add_argument('arm',choices=ARMS);ap.add_argument('--stage-b',action='store_true')
    a=ap.parse_args();main(a.fold,a.arm,a.stage_b)
