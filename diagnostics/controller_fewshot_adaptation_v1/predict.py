"""Frozen critic pool inference. No outcome labels or physical stepping."""
from core import *
import time

def main():
    gate();import jax,jax.numpy as jnp,flax.linen as nn
    from flax import serialization
    from diagnostics.controller_loco_real_flow_v1.train import Net
    from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
    jax.config.update('jax_enable_x64',False);assert any(d.platform=='gpu' for d in jax.devices()) and os.environ.get('SLURM_JOB_ID')
    class Residual(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.silu(nn.Dense(16,name='dense1')(x));x=nn.silu(nn.Dense(16,name='dense2')(x))
            return jnp.tanh(nn.Dense(16,name='out',kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x))
    net=Net();residual=Residual();fn=jax.jit(lambda p,x:net.apply(p,x));rf=jax.jit(lambda p,x:residual.apply(p,x))
    p4=read(R4/'protocol.json');pv=read(P1/'P1_EXECUTION_PROTOCOL.json');states=p4['states'][:128]+pv['states'];phys=physical23(states);records=[]
    for fold,n in enumerate(p4['controller_order']):
        start=time.time();out=HERE/'predictions'/f'{fold}.npz';out.parent.mkdir(exist_ok=True)
        base=R4/'fits'/f'fold{fold}_R0';fix=R6/'fits'/f'real_LOCO_fold{fold}_F2_PCA2';norm=np.load(base/'normalization.npz');raw=np.c_[phys,np.zeros((len(states),len(norm['mean'])-23),np.float32)];x=((raw-norm['mean'])/norm['scale']).astype(np.float32)
        template=net.init(jax.random.PRNGKey(0),np.zeros((1,x.shape[-1]),np.float32));m0=read(base/'complete.json');r0=[]
        for path in m0['checkpoints']:r0.append(np.asarray(fn(serialization.from_bytes(template,Path(path).read_bytes()),jnp.asarray(x))))
        r0=np.array(r0);pc=np.load(fix/'pca.npz');sf=np.load(R4/'static_features'/f'{n}.npz');mean=sf['R3'][:128].mean(0).astype(float);co=((mean-pc['mean'])/pc['scale'])@pc['components'][:2].T;pad=np.zeros((len(states),513),np.float32);pad[:,:2]=co;xx=np.c_[phys,pad].astype(np.float32);nnorm=np.load(fix/'normalization.npz');xx=((xx-nnorm['mean'])/nnorm['scale']).astype(np.float32)
        tm=residual.init(jax.random.PRNGKey(0),np.zeros((1,536),np.float32));m2=read(fix/'complete.json');f2=[]
        for i,(path,h) in enumerate(zip(m2['checkpoints'],m2['checkpoint_sha256'])):
            assert sha(path)==h;f2.append(np.asarray(rf(serialization.from_bytes(tm,Path(path).read_bytes()),jnp.asarray(xx)))+r0[i])
        old0=np.load(base/'predictions.npz');old2=np.load(fix/'predictions.npz');held0=old0['held'];held2=old2['held'][:,0]
        np.savez_compressed(out,pool_R0=r0,pool_F2=np.array(f2),held_R0=held0,held_F2=held2,controller_name=n,state_uids=[s['state_uid'] for s in states])
        check_error=float(np.max(abs(r0[:,:4]-old0['target_calibration'])))
        rec=dict(fold=fold,controller=n,source_controllers=[z for z in p4['controller_order'] if z!=n],pool_states=152,pool_logits_shape=list(r0.shape),held_predictions_reused=True,R0_first4_vs_cached_logit_max_error=check_error,checkpoint_files=[dict(path=p,sha256=sha(p)) for p in m0['checkpoints']+m2['checkpoints']],f2_source_pca_co=co,pool_predictions_sha256=sha(out),jax_enable_x64=False,devices=[str(d) for d in jax.devices()],seconds=time.time()-start)
        assert np.isfinite(r0).all() and np.isfinite(f2).all();records.append(rec);print(json.dumps(clean(dict(fold=fold,seconds=rec['seconds'],error=check_error))),flush=True)
    write(HERE/'PREDICTION_RECEIPT.json',dict(utc=utc(),status='COMPLETE',rows=records,checkpoint_training_performed=False,terminal_outcomes_read=False,jax_enable_x64=False))
if __name__=='__main__':main()
