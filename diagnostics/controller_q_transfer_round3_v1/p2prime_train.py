"""Balanced source-only 16-head critic fits for conditional P2′."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import numpy as np
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23


def read(name):return json.loads((HERE/name).read_text())
def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def build_features(arm):
    protocol=read('P2PRIME_PROTOCOL.json');names=protocol['source_controllers']+protocol['held_controllers'];states=protocol['states']
    phys=physical23(states)
    oldphys=np.load(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz')['physical23'][:512]
    source=[];held=[];mismatch=[]
    for c,name in enumerate(names):
        z=np.load(HERE/'p2prime_features'/f'{name}.npz')
        physical=phys[:288] if c<5 else phys[256:]
        desc=np.zeros((len(physical),320),np.float32)
        more=np.load(HERE/'p3_features'/f'{name}.npz') if arm=='R4_openloop' else None
        if arm=='R1' and c<5:desc[:,c]=1
        if arm in ('R3','R3_R4'):desc[:,:256]=z['R3']
        if arm=='R4_action':desc[:,:64]=z['R4_action']
        if arm=='R4_t0':desc[:,:32]=z['R4_action'][:,:32]
        if arm=='R4_openloop':desc[:,:64]=more['R4_openloop']
        if arm=='R3_R4':desc[:,256:320]=z['R4_action']
        row=np.concatenate((physical,desc),axis=1).astype(np.float32)
        if c<5:source.append(row)
        else:
            held.append(row[32:])
            wrongname='FT' if name=='TF' else 'TF'
            wrong=np.load(HERE/'p2prime_features'/f'{wrongname}.npz')
            wrong_more=np.load(HERE/'p3_features'/f'{wrongname}.npz') if arm=='R4_openloop' else None
            wrong_desc=np.zeros((64,320),np.float32)
            if arm in ('R3','R3_R4'):wrong_desc[:,:256]=wrong['R3'][32:]
            if arm=='R4_action':wrong_desc[:,:64]=wrong['R4_action'][32:]
            if arm=='R4_t0':wrong_desc[:,:32]=wrong['R4_action'][32:,:32]
            if arm=='R4_openloop':wrong_desc[:,:64]=wrong_more['R4_openloop'][32:]
            if arm=='R3_R4':wrong_desc[:,256:320]=wrong['R4_action'][32:]
            mismatch.append(np.concatenate((phys[288:],wrong_desc),axis=1).astype(np.float32))
    oldfeat=np.load(HERE/'p2prime_features'/'FF_historical512.npz')
    oldmore=np.load(HERE/'p3_features'/'FF_historical512.npz') if arm=='R4_openloop' else None
    olddesc=np.zeros((512,320),np.float32)
    if arm=='R1':olddesc[:,0]=1
    if arm in ('R3','R3_R4'):olddesc[:,:256]=oldfeat['R3']
    if arm=='R4_action':olddesc[:,:64]=oldfeat['R4_action']
    if arm=='R4_t0':olddesc[:,:32]=oldfeat['R4_action'][:,:32]
    if arm=='R4_openloop':olddesc[:,:64]=oldmore['R4_openloop']
    if arm=='R3_R4':olddesc[:,256:320]=oldfeat['R4_action']
    source=np.concatenate(source+[np.concatenate((oldphys,olddesc),axis=1).astype(np.float32)])
    held=np.stack(held);mismatch=np.stack(mismatch)
    assert source.shape==(1952,343) and held.shape==mismatch.shape==(2,64,343)
    assert np.isfinite(source).all() and np.isfinite(held).all() and np.isfinite(mismatch).all()
    return source,held,mismatch


def main():
    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from flax import serialization
    assert any(z.platform=='gpu' for z in jax.devices())
    jax.config.update('jax_enable_x64',False)
    assert read('P2PRIME_FEATURE_RECEIPT.json')['status']=='COMPLETED'
    assert read('P2PRIME_LABEL_AUDIT.json')['status']=='COMPLETE EXACT UID LABEL TENSOR'
    lab=np.load(HERE/'P2PRIME_LABELS.npz')['outcomes']
    source_lab=lab[:5,:288,:,:4].reshape(1440,16,4)
    old=np.load(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/source_dataset.npz')
    oldS=old['success'][:512].astype(np.float32);oldF=old['failure'][:512].astype(np.float32)
    S=np.concatenate(((source_lab==1).sum(-1),oldS)).astype(np.float32)
    F=np.concatenate(((source_lab==0).sum(-1),oldF)).astype(np.float32)
    val_by_c={c:np.arange(c*288+256,c*288+288) for c in range(5)}
    train_by_c={c:np.arange(c*288,c*288+256) for c in range(5)}
    train_by_c[0]=np.r_[train_by_c[0],np.arange(1440,1952)]

    class Net(nn.Module):
        @nn.compact
        def __call__(self,x):
            h=nn.silu(nn.Dense(64,name='dense1')(x))
            h=nn.silu(nn.Dense(64,name='dense2')(h))
            return nn.Dense(16,name='out')(h)
    net=Net();opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
    p3_only=os.environ.get('P3_ONLY')=='1'
    if p3_only:
        assert read('P3_FEATURE_RECEIPT.json')['status']=='COMPLETED'
        assert not read('P2PRIME_RESULTS.json')['primary_confirmed']
    arms=('R4_t0','R4_openloop') if p3_only else ('R0','R1','R3','R4_action','R3_R4')
    schedule=[50,100]+list(range(200,12001,200));records=[];started=time.time()
    for count,active in [(3,[0,1,3]),(5,[0,1,2,3,4])]:
        for arm in arms:
            folder=HERE/'p2prime_fits'/f'C{count}_{arm}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                records.append(read(folder/'complete.json'));continue
            source,held,mismatch=build_features(arm)
            tr=np.concatenate([train_by_c[c] for c in active]);val=np.concatenate([val_by_c[c] for c in active])
            mean=source[tr].mean(0);std=source[tr].std(0);scale=np.where(std>1e-6,std,1.)
            x=(source-mean)/scale;hx=(held-mean)/scale;mx=(mismatch-mean)/scale
            X=jnp.asarray(x);H=jnp.asarray(hx);M=jnp.asarray(mx);Sj=jnp.asarray(S);Fj=jnp.asarray(F)
            pair_banks={}
            for c in active:
                pair_banks[c]=np.array([(i,j) for i in train_by_c[c] for j in range(16) if S[i,j]+F[i,j]>0],np.int32)
            valid=np.maximum((S+F)[val].sum(),1)
            nmean=float(np.mean([S[i,j]+F[i,j] for bank in pair_banks.values() for i,j in bank]))

            @jax.jit
            def predict(par,xx):return net.apply(par,xx)

            @jax.jit
            def block(par,optimizer,ii,jj):
                def step(carry,ids):
                    pa,op=carry;i,j=ids
                    def loss(params):
                        z=predict(params,X[i]);chosen=jnp.take_along_axis(z,j[:,None],1)[:,0]
                        return jnp.mean(Sj[i,j]*jax.nn.softplus(-chosen)+Fj[i,j]*jax.nn.softplus(chosen))/nmean
                    _,g=jax.value_and_grad(loss)(pa);up,op=opt.update(g,op,pa)
                    return (optax.apply_updates(pa,up),op),None
                return jax.lax.scan(step,(par,optimizer),(ii,jj))[0]

            hists=[];snapshots=[]
            for seed in (17,23,41):
                rng=np.random.default_rng(seed);pa=net.init(jax.random.PRNGKey(seed),X[:1]);op=opt.init(pa);prev=0;hh=[];buf=[]
                for step in schedule:
                    choices=rng.integers(len(active),size=(step-prev,128));ii=np.empty_like(choices);jj=np.empty_like(choices)
                    for code,c in enumerate(active):
                        mask=choices==code;bank=pair_banks[c]
                        draw=bank[rng.integers(len(bank),size=int(mask.sum()))]
                        ii[mask],jj[mask]=draw[:,0],draw[:,1]
                    pa,op=block(pa,op,jnp.asarray(ii),jnp.asarray(jj))
                    z=np.asarray(predict(pa,X[val]));nll=float((S[val]*np.logaddexp(0,-z)+F[val]*np.logaddexp(0,z)).sum()/valid)
                    hh.append(nll);buf.append(serialization.to_bytes(pa));prev=step
                hists.append(hh);snapshots.append(buf)
            pos=int(np.argmin(np.mean(hists,axis=0)));chosen_step=schedule[pos]
            predicted=[];mismatched=[];source_val=[];cp=[]
            for seed,history in zip((17,23,41),snapshots):
                path=folder/f'seed{seed}_step{chosen_step}.msgpack';path.write_bytes(history[pos]);cp.append(str(path))
                fitted=serialization.from_bytes(pa,history[pos])
                predicted.append(np.asarray(predict(fitted,H.reshape(-1,343))).reshape(2,64,16))
                mismatched.append(np.asarray(predict(fitted,M.reshape(-1,343))).reshape(2,64,16))
                source_val.append(np.asarray(predict(fitted,X[val])))
            np.savez_compressed(folder/'predictions.npz',held=np.stack(predicted),mismatched=np.stack(mismatched),source_val=np.stack(source_val))
            np.savez_compressed(folder/'normalization.npz',mean=mean,scale=scale)
            rec=dict(arm=arm,source_controller_count=count,source_controller_indices=active,fit_seeds=[17,23,41],
                     selected_step=chosen_step,validation_nll_history=hists,checkpoints=cp,
                     train_pairs={str(c):len(pair_banks[c]) for c in active},
                     old_native_ff_supplement_rows=512,new_source_state_rows=256*count,
                     jax_enable_x64=False,elapsed_seconds=time.time()-started)
            write(folder/'complete.json',rec);records.append(rec)
            print('C',count,arm,'step',chosen_step,'elapsed',round(time.time()-started,1),flush=True)
    write(HERE/('P3_TRAINING.json' if p3_only else 'P2PRIME_TRAINING.json'),dict(status='COMPLETE',records=records,new_fits=12 if p3_only else 30,elapsed_seconds=time.time()-started,
                                           device=[str(x) for x in jax.devices()],jax_enable_x64=False,
                                           source_only_training=True))


if __name__=='__main__':main()
