"""Masked source-only fitting. No endpoint labels are imported here."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from util import *
import time
def r0_inputs(folder,states):
    norm=np.load(folder/'normalization.npz');width=len(norm['mean']);raw=np.c_[physical23(states),np.zeros((len(states),width-23),np.float32)]
    return ((raw-norm['mean'])/norm['scale']).astype(np.float32)
def main(index,bed='blend'):
    import jax;import jax.numpy as jnp;import optax;import flax.linen as nn
    from flax import serialization
    jax.config.update('jax_enable_x64',False);assert any(d.platform=='gpu' for d in jax.devices())
    assert read(HERE/'STAGE0_REPRODUCTION.json')['primary_gate_passed']
    tasks=read(HERE/('FIT_TASKS_BLEND.json' if bed=='blend' else 'FIT_TASKS_REAL.json'));t=tasks[index];arm=t['arm'];out=HERE/'fits'/task_name(t);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    start=time.time();data=Data(t);X,S,F,V,H,mean,scale,L=data.build(arm)
    signature=hashlib.sha256(X.tobytes()+L.tobytes()+np.stack((data.SV,data.FV)).tobytes()+arm.encode()).hexdigest()
    shared=HERE/'shared_sparse_fits'/arm if bed=='blend' and t['coverage']=='SPARSE' else out;shared.mkdir(parents=True,exist_ok=True)
    frozen=shared/'MODEL_FROZEN.json';residual=arm.startswith('F2')
    class Residual(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.silu(nn.Dense(16,name='dense1')(x));x=nn.silu(nn.Dense(16,name='dense2')(x))
            return jnp.tanh(nn.Dense(16,name='out',kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x))
    net=Residual() if residual else Net();predict=jax.jit(lambda pa,x:net.apply(pa,x))
    base_train=np.zeros((3,len(data.sources),128,16),np.float32);base_val=np.zeros((3,len(data.valnames),24,16),np.float32);base_held=np.zeros((3,len(data.held),len(data.test_states),16),np.float32)
    if residual:
        base=data.base_folder('R0')
        width=len(np.load(base/'normalization.npz')['mean']);raw_train=np.zeros((len(data.sources),128,width),np.float32)
        for i,n in enumerate(data.sources):
            raw=r0_inputs(base,data.states(n));raw_train[i,:len(raw)]=raw
        base_train=frozen_predict(base,raw_train.reshape(-1,width)).reshape(3,len(data.sources),128,16)
        base_val=frozen_predict(base,np.concatenate([r0_inputs(base,data.val_states) for n in data.valnames])).reshape(3,len(data.valnames),24,16)
        base_held=frozen_predict(base,np.concatenate([r0_inputs(base,data.test_states) for n in data.held])).reshape(3,len(data.held),len(data.test_states),16)
    optimized=False
    if not frozen.exists():
        assert not (bed=='blend' and t['coverage']=='SPARSE') or t['fold']==0,'First source-equivalent SPARSE fit must freeze before other folds'
        optimized=True;NC=len(data.sources);x=jnp.asarray(X);v=jnp.asarray(V);s=jnp.asarray(S);f=jnp.asarray(F)
        banks=[np.argwhere(S[i]+F[i]>0).astype(np.int32) for i in range(NC)];assert all(len(b) for b in banks)
        denom=float(np.mean([S[i,j,k]+F[i,j,k] for i,b in enumerate(banks) for j,k in b]));valid=float((data.SV+data.FV).sum());assert valid>0
        opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.01 if residual else .0001))
        @jax.jit
        def block(pa,op,base,cc,ii,jj):
            def step(carry,ids):
                params,state=carry;c,i,j=ids
                def loss(pp):
                    logits=predict(pp,x[c,i])+base[c,i];z=jnp.take_along_axis(logits,j[:,None],1)[:,0]
                    return jnp.mean(s[c,i,j]*jax.nn.softplus(-z)+f[c,i,j]*jax.nn.softplus(z))/denom
                _,grad=jax.value_and_grad(loss)(params);updates,state=opt.update(grad,state,params)
                return (optax.apply_updates(params,updates),state),None
            return jax.lax.scan(step,(pa,op),(cc,ii,jj))[0]
        schedule=[50,100]+list(range(200,12001,200));histories=[];snapshots=[]
        for fi,seed in enumerate((17,23,41)):
            rng=np.random.default_rng(seed);pa=net.init(jax.random.PRNGKey(seed),x[:1,:1]);op=opt.init(pa);previous=0;hist=[];snaps=[]
            for target in schedule:
                cc=rng.integers(NC,size=(target-previous,128));ii=np.empty_like(cc);jj=np.empty_like(cc)
                for code in range(NC):
                    loc=cc==code;b=banks[code];draw=b[rng.integers(len(b),size=int(loc.sum()))];ii[loc]=draw[:,0];jj[loc]=draw[:,1]
                pa,op=block(pa,op,jnp.asarray(base_train[fi]),jnp.asarray(cc),jnp.asarray(ii),jnp.asarray(jj))
                logits=np.asarray(predict(pa,v.reshape(-1,536))).reshape(len(V),24,16)+base_val[fi]
                hist.append(float((data.SV*np.logaddexp(0,-logits)+data.FV*np.logaddexp(0,logits)).sum()/valid));snaps.append(serialization.to_bytes(pa));previous=target
            histories.append(hist);snapshots.append(snaps)
        selected=int(np.argmin(np.mean(histories,axis=0)));step=schedule[selected];paths=[]
        for seed,ss in zip((17,23,41),snapshots):
            path=shared/f'seed{seed}_step{step}.msgpack';path.write_bytes(ss[selected]);paths.append(str(path))
        np.savez_compressed(shared/'normalization.npz',mean=mean,scale=scale)
        np.savez_compressed(shared/'pca.npz',mean=data.pca_mean,scale=data.pca_scale,components=data.components)
        write(frozen,dict(status='FROZEN',source_signature=signature,arm=arm,checkpoints=paths,checkpoint_sha256=[sha(path) for path in paths],fit_seeds=[17,23,41],selected_step=step,validation_nll_history=histories,source_names=data.sources,jax_enable_x64=False,max_steps=12000,residual_bounded=1. if residual else None,weight_decay=.01 if residual else .0001,network='16/16/16 bounded residual' if residual else '64/64/16 head',source_fit_seconds=time.time()-start,pca_fitted_source_only=True,pca_variance=data.pca_variance,regimes_source_only=data.regimes,base_r0_checkpoint_sha256=[sha(path) for path in read(data.base_folder('R0')/'complete.json')['checkpoints']] if residual else None))
    meta=read(frozen);assert meta['source_signature']==signature
    norm=np.load(shared/'normalization.npz');assert np.array_equal(norm['mean'],mean) and np.array_equal(norm['scale'],scale)
    template=net.init(jax.random.PRNGKey(0),np.zeros((1,536),np.float32));held=[];vals=[]
    for fi,(path,h) in enumerate(zip(meta['checkpoints'],meta['checkpoint_sha256'])):
        assert sha(path)==h;pa=serialization.from_bytes(template,Path(path).read_bytes())
        held.append(np.asarray(predict(pa,jnp.asarray(H.reshape(-1,536)))).reshape(len(data.held),len(data.test_states),16)+base_held[fi])
        vals.append(np.asarray(predict(pa,jnp.asarray(V.reshape(-1,536)))).reshape(len(data.valnames),24,16)+base_val[fi])
    np.savez_compressed(out/'predictions.npz',held=np.array(held),source_validation=np.array(vals),held_controller_names=data.held)
    np.savez_compressed(out/'normalization.npz',mean=mean,scale=scale)
    write(out/'complete.json',dict(status='COMPLETE',task=t,fit_seeds=[17,23,41],checkpoints=meta['checkpoints'],checkpoint_sha256=meta['checkpoint_sha256'],source_signature=signature,optimization_performed=optimized,selected_step=meta['selected_step'],elapsed_seconds=time.time()-start,jax_enable_x64=False,test_labels_read=False,pca_source_only=True,source_only_normalization=True,residual=residual,code_sha256=sha(__file__)))
    print(json.dumps(dict(task=task_name(t),optimized=optimized,elapsed_seconds=time.time()-start)))
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'blend')
