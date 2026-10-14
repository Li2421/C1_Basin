"""Exactly round6 bounded residual optimizer; source-only model selection."""
from core import *
from data import Data,name
import time

def base_inputs(folder,states):
    from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
    norm=np.load(folder/'normalization.npz');width=len(norm['mean']);raw=np.c_[physical23(states),np.zeros((len(states),width-23),np.float32)]
    return ((raw-norm['mean'])/norm['scale']).astype(np.float32)

def frozen_predict(folder,x):
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_loco_real_flow_v1.train import Net
    net=Net();template=net.init(jax.random.PRNGKey(0),np.zeros((1,x.shape[-1]),np.float32));fn=jax.jit(lambda p,x:net.apply(p,x));meta=read(folder/'complete.json')
    return np.array([np.asarray(fn(serialization.from_bytes(template,Path(p).read_bytes()),jnp.asarray(x,np.float32))) for p in meta['checkpoints']])

def main(t):
    gate();import jax,jax.numpy as jnp,optax,flax.linen as nn
    from flax import serialization
    jax.config.update('jax_enable_x64',False)
    assert os.environ.get('SLURM_JOB_ID'),'All fitting is scheduled through Slurm'
    backend=jax.default_backend();assert backend in ('cpu','gpu')
    out=HERE/'fits'/name(t);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    start=time.time();d=Data(t);X,S,F,V,H,mean,scale,L=d.build();sig=hashlib.sha256(X.tobytes()+L.tobytes()+np.stack((d.SV,d.FV)).tobytes()+t['representation'].encode()).hexdigest()
    shared=HERE/'shared_sparse_fits'/(t['representation']+('_imputed' if t['sensitivity'] else '')) if t['bed']=='blend' and t['coverage']=='SPARSE' else out
    shared.mkdir(parents=True,exist_ok=True);frozen=shared/'MODEL_FROZEN.json'
    class Residual(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.silu(nn.Dense(16,name='dense1')(x));x=nn.silu(nn.Dense(16,name='dense2')(x))
            return jnp.tanh(nn.Dense(16,name='out',kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x))
    net=Residual();predict=jax.jit(lambda p,x:net.apply(p,x));base=r0_folder(dict(t,coverage='LOCO' if t['bed']=='real' else t['coverage']));width=len(np.load(base/'normalization.npz')['mean'])
    raw=np.zeros((len(d.sources),128,width),np.float32)
    for i,n in enumerate(d.sources):xx=base_inputs(base,d.states(n));raw[i,:len(xx)]=xx
    bt=frozen_predict(base,raw.reshape(-1,width)).reshape(3,len(d.sources),128,16)
    bv=frozen_predict(base,np.concatenate([base_inputs(base,d.val_states) for n in d.valnames])).reshape(3,len(d.valnames),24,16)
    bh=frozen_predict(base,np.concatenate([base_inputs(base,d.test_states) for n in d.held])).reshape(3,len(d.held),len(d.test_states),16)
    optimized=False
    if not frozen.exists():
        assert t['bed']!='blend' or t['coverage']!='SPARSE' or t['fold']==0
        optimized=True;nc=len(d.sources);x=jnp.asarray(X);v=jnp.asarray(V);s=jnp.asarray(S);f=jnp.asarray(F)
        banks=[np.argwhere(S[i]+F[i]>0).astype(np.int32) for i in range(nc)];assert all(len(b) for b in banks)
        denom=float(np.mean([S[i,j,k]+F[i,j,k] for i,b in enumerate(banks) for j,k in b]));valid=float((d.SV+d.FV).sum());assert valid>0
        opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.01))
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
            rng=np.random.default_rng(seed);pa=net.init(jax.random.PRNGKey(seed),x[:1,:1]);op=opt.init(pa);prev=0;hist=[];snaps=[]
            for target in schedule:
                cc=rng.integers(nc,size=(target-prev,128));ii=np.empty_like(cc);jj=np.empty_like(cc)
                for code in range(nc):
                    loc=cc==code;b=banks[code];draw=b[rng.integers(len(b),size=int(loc.sum()))];ii[loc]=draw[:,0];jj[loc]=draw[:,1]
                pa,op=block(pa,op,jnp.asarray(bt[fi]),jnp.asarray(cc),jnp.asarray(ii),jnp.asarray(jj))
                logits=np.asarray(predict(pa,v.reshape(-1,536))).reshape(len(V),24,16)+bv[fi]
                hist.append(float((d.SV*np.logaddexp(0,-logits)+d.FV*np.logaddexp(0,logits)).sum()/valid));snaps.append(serialization.to_bytes(pa));prev=target
            histories.append(hist);snapshots.append(snaps)
        selected=int(np.argmin(np.mean(histories,axis=0)));step=schedule[selected];paths=[]
        for seed,ss in zip((17,23,41),snapshots):
            path=shared/f'seed{seed}_step{step}.msgpack';path.write_bytes(ss[selected]);paths.append(str(path))
        np.savez_compressed(shared/'normalization.npz',mean=mean,scale=scale);write(shared/'representation.json',d.rep_meta)
        write(frozen,dict(utc=utc(),status='FROZEN',source_signature=sig,task=t,checkpoints=paths,checkpoint_sha256=[sha(p) for p in paths],fit_seeds=[17,23,41],selected_step=step,schedule=schedule,validation_nll_history=histories,source_names=d.sources,validation_names=d.valnames,source_prior_eta=d.src,regimes_source_only=d.regimes,source_fit_seconds=time.time()-start,base_r0_sha256=[sha(p) for p in read(base/'complete.json')['checkpoints']],jax_enable_x64=False,max_steps=12000,residual_bounded=1.,weight_decay=.01,lr=.0008,clip=5,batch_heads=128,network='16/16/16 SiLU bounded residual',test_labels_read=False))
    meta=read(frozen);assert meta['source_signature']==sig
    norm=np.load(shared/'normalization.npz');assert np.array_equal(norm['mean'],mean) and np.array_equal(norm['scale'],scale)
    template=net.init(jax.random.PRNGKey(0),np.zeros((1,536),np.float32));held=[];vals=[]
    for fi,(path,h) in enumerate(zip(meta['checkpoints'],meta['checkpoint_sha256'])):
        assert sha(path)==h;pa=serialization.from_bytes(template,Path(path).read_bytes())
        held.append(np.asarray(predict(pa,jnp.asarray(H.reshape(-1,536)))).reshape(len(d.held),len(d.test_states),16)+bh[fi]);vals.append(np.asarray(predict(pa,jnp.asarray(V.reshape(-1,536)))).reshape(len(d.valnames),24,16)+bv[fi])
    np.savez_compressed(out/'predictions.npz',held=np.array(held),source_validation=np.array(vals),held_controller_names=d.held)
    np.savez_compressed(out/'normalization.npz',mean=mean,scale=scale);write(out/'representation.json',d.rep_meta)
    write(out/'complete.json',dict(status='COMPLETE',utc=utc(),task=t,checkpoints=meta['checkpoints'],checkpoint_sha256=meta['checkpoint_sha256'],fit_seeds=[17,23,41],source_signature=sig,optimization_performed=optimized,selected_step=meta['selected_step'],elapsed_seconds=time.time()-start,test_labels_read=False,jax_enable_x64=False,source_prior_eta=d.src,source_only=True,backend=backend,slurm_job_id=os.environ.get('SLURM_JOB_ID'),code_sha256=sha(__file__)))
    print(json.dumps(dict(task=name(t),optimized=optimized,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
    ts=read(HERE/('TASKS.json' if sys.argv[1]=='task' else 'FIT_TASKS.json'));main(ts[int(sys.argv[2])])
