"""GPU-only BC fitting, with sequence state retained across TBPTT windows."""
from common import *
from zoo_models import network,initialize,sequence_examples,infonce
from new_benchmark_common.dataset import JointTransitionDataset
import jax
import jax.numpy as jnp
import optax
import pickle
from flax import serialization
import time
jax.config.update('jax_enable_x64',False)


def padded(dataset,normal):
    lengths=np.asarray([t.length for t in dataset.trajectories],np.int32)
    width=int(lengths.max());n=len(lengths)
    x=np.zeros((n,width,92),np.float32);y=np.zeros((n,width,8),np.float32)
    for i,t in enumerate(dataset.trajectories):
        x[i,:t.length]=t.observations[:-1].reshape(-1,92)
        y[i,:t.length]=t.actions.reshape(-1,8)
    x=(x-np.asarray(normal['obs_mean'],np.float32))/np.asarray(normal['obs_scale'],np.float32)
    y=(y-np.asarray(normal['act_mean'],np.float32))/np.asarray(normal['act_scale'],np.float32)
    return jnp.asarray(x),jnp.asarray(y),jnp.asarray(lengths)


def gaussian_nll(mean,logs,target):
    return jnp.sum(.5*jnp.square((target-mean)*jnp.exp(-logs))+logs+.5*jnp.log(2*jnp.pi),axis=-1)


def train(index):
    prereg()
    gates=read(HERE/'REPRODUCTION_GATE.json')['type_passed']
    assert any(gates[k] for k in ('deterministic','mixture','diffusion','G'))
    assert read(HERE/'ZOO_UNIT_TESTS.json')['passed']
    assert read(HERE/'FRAME_TESTS_NEW.json')['passed']
    assert all(d.platform=='gpu' for d in jax.devices())
    cfg=read(HERE/'TRAINING_CONFIGS.json')[index]
    if cfg['family']=='implicit':
        assert read(HERE/'IMPLICIT_SYNTHETIC_TEST.json')['passed']
    folder=HERE/'policies'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.receipt.json').exists()
    assert sha(cfg['manifest'])==cfg['manifest_sha256']
    assert sha(cfg['flow_checkpoint'])==cfg['flow_checkpoint_sha256']
    original=pickle.loads(Path(cfg['flow_checkpoint']).read_bytes())['config']
    normal={k:np.asarray(original[k],np.float32).tolist() for k in ('obs_mean','obs_scale','act_mean','act_scale')}
    normal['max_speed']=float(original['max_speed'])
    write(folder/'NORMALIZATION.json',normal);write(folder/'CONFIG.json',cfg)
    seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),
         [folder/'CONFIG.json',folder/'NORMALIZATION.json',HERE/'train_bc.py',HERE/'zoo_models.py'],before_training=True)
    trainset=JointTransitionDataset(Path(cfg['manifest']).parent,'train',seed=cfg['seed'])
    dev=JointTransitionDataset(Path(cfg['manifest']).parent,'dev',seed=cfg['validation_seed'])
    family=cfg['family'];net=network(family);key=jax.random.PRNGKey(cfg['seed'])
    pa=initialize(net,family,key);opt=optax.adam(cfg['learning_rate']);state=opt.init(pa)
    rng=np.random.default_rng(cfg['validation_seed']);vi=rng.integers(len(dev.actions),size=2048)
    np.save(folder/'DEV_INDICES.npy',vi)
    devkey=jax.random.PRNGKey(cfg['validation_seed']+18000)
    if family=='recurrent':
        X,Y,lengths=padded(trainset,normal);XV,YV,vl=padded(dev,normal)
        cumulative=np.cumsum(np.asarray(vl));de=np.searchsorted(cumulative,vi,side='right')
        dt=vi-np.r_[0,cumulative[:-1]][de]
        def sequence_loss(p,x,y,mask,hidden):
            def step(h,z):
                xx,yy=z;h,mean,logs=net.apply(p,xx,h)
                return h,gaussian_nll(mean,logs,yy)
            end,losses=jax.lax.scan(step,hidden,(jnp.swapaxes(x,0,1),jnp.swapaxes(y,0,1)))
            losses=losses.T
            return jnp.sum(losses*mask)/jnp.maximum(mask.sum(),1),(end,losses)
        @jax.jit
        def validate(p):
            mask=jnp.arange(XV.shape[1])[None,:]<vl[:,None]
            _,(_,losses)=sequence_loss(p,XV,YV,mask,jnp.zeros((len(vl),256),jnp.float32))
            return losses[jnp.asarray(de),jnp.asarray(dt)].mean()
        streams=jax.random.randint(key,(8,),0,len(lengths));offset=jnp.zeros(8,jnp.int32)
        hidden=jnp.zeros((8,256),jnp.float32)
        @jax.jit
        def block(p,st,streams,offset,hidden,steps):
            def update(carry,step):
                p,st,streams,offset,hidden=carry
                times=offset[:,None]+jnp.arange(32)[None,:]
                mask=times<lengths[streams,None]
                times=jnp.minimum(times,X.shape[1]-1)
                xx=X[streams[:,None],times];yy=Y[streams[:,None],times]
                (_, (h,_)),grad=jax.value_and_grad(sequence_loss,has_aux=True)(p,xx,yy,mask,hidden)
                delta,st=opt.update(grad,st,p);p=optax.apply_updates(p,delta)
                offset=offset+32;done=offset>=lengths[streams]
                next_stream=jax.random.randint(jax.random.fold_in(key,step),(8,),0,len(lengths))
                streams=jnp.where(done,next_stream,streams);offset=jnp.where(done,0,offset)
                hidden=jnp.where(done[:,None],0.,jax.lax.stop_gradient(h))
                return (p,st,streams,offset,hidden),None
            return jax.lax.scan(update,(p,st,streams,offset,hidden),steps)[0]
    else:
        def norm(a,m,s):return (a.reshape(len(a),-1)-np.asarray(normal[m],np.float32))/np.asarray(normal[s],np.float32)
        if family=='chunking':
            xx,yy,mask=sequence_examples(trainset.trajectories,normal,family)
            xv,yv,mv=sequence_examples(dev.trajectories,normal,family)
            M=jnp.asarray(mask);MV=jnp.asarray(mv[vi])
        else:
            xx=norm(trainset.observations,'obs_mean','obs_scale');yy=norm(trainset.actions,'act_mean','act_scale')
            xv=norm(dev.observations,'obs_mean','obs_scale');yv=norm(dev.actions,'act_mean','act_scale')
            M=jnp.ones((len(xx),1));MV=jnp.ones((len(vi),1))
        X,Y,XV,YV=map(jnp.asarray,(xx,yy,xv[vi],yv[vi]))
        def loss(p,x,y,mask,k):
            if family=='chunking':
                error=jnp.mean(jnp.square(net.apply(p,x)-y),axis=-1)
                return jnp.sum(error*mask)/jnp.maximum(mask.sum(),1)
            if family=='cvae':
                prediction,kl=net.apply(p,x,y,k)
                return jnp.mean(jnp.mean(jnp.square(prediction-y),axis=-1)+.5*kl)
            return infonce(net,p,x,y,k,jnp.asarray(normal['act_mean']),jnp.asarray(normal['act_scale']))
        validate=jax.jit(lambda p:loss(p,XV,YV,MV,devkey))
        @jax.jit
        def block(p,st,ids,steps):
            def update(carry,z):
                p,st=carry;idx,step=z
                grad=jax.grad(loss)(p,X[idx],Y[idx],M[idx],jax.random.fold_in(key,step))
                delta,st=opt.update(grad,st,p)
                return (optax.apply_updates(p,delta),st),None
            return jax.lax.scan(update,(p,st),(ids,steps))[0]
    best=float(validate(pa));beststep=0;history=[dict(step=0,dev_loss=best)]
    (folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
    rng=np.random.default_rng(cfg['seed']);before=time.monotonic()
    for target in range(250,cfg['steps']+1,250):
        steps=jnp.arange(target-250,target)
        if family=='recurrent':
            pa,state,streams,offset,hidden=block(pa,state,streams,offset,hidden,steps)
        else:
            ids=rng.integers(len(X),size=(250,256))
            pa,state=block(pa,state,jnp.asarray(ids),steps)
        value=float(validate(pa));assert np.isfinite(value)
        history.append(dict(step=target,dev_loss=value))
        if value<best:
            best=value;beststep=target;(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
        if target%5000==0:
            write(folder/'TRAINING_HISTORY.partial.json',history)
            print(json.dumps(dict(name=cfg['name'],step=target,dev_loss=value,elapsed_seconds=time.monotonic()-before)),flush=True)
    write(folder/'TRAINING_HISTORY.json',history)
    (folder/'final.msgpack').write_bytes(serialization.to_bytes(pa))
    write(folder/'SUMMARY.json',dict(best_step=beststep,best_loss=best,steps=cfg['steps'],
          train_transitions=len(trainset.actions),dev_transitions=len(dev.actions),x64=False,
          elapsed_seconds=time.monotonic()-before))
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),
         [folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'best.msgpack',folder/'TRAINING_HISTORY.json',
          folder/'DEV_INDICES.npy',HERE/'train_bc.py',HERE/'zoo_models.py'],closed_loop_labels_read=False)

if __name__=='__main__':
    train(int(sys.argv[1]))
