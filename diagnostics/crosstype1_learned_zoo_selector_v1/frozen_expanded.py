"""Read frozen round38 weights and transforms, never call its training entry."""
from common import *
import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
jax.config.update('jax_enable_x64',False)


class FrozenNet(nn.Module):
    @nn.compact
    def __call__(self,x):
        for width in (64,64,16):x=nn.silu(nn.Dense(width)(x))
        return nn.Dense(1,kernel_init=nn.initializers.lecun_normal(),bias_init=nn.initializers.zeros)(x)[...,0]


def predict(states,r5):
    folder=D/'blackbox_families_v1';receipt=read(folder/'EXPANDED_MODEL_FROZEN.receipt.json')
    transform=D/'ring_controller_critic_v1/fits/full_transform.npz'
    original=read(D/'ring_controller_critic_v1/MODEL_FREEZE.receipt.json')
    assert sha(transform)==original['files']['fits/full_transform.npz']
    with np.load(transform,allow_pickle=False) as z:tr={k:z[k] for k in z.files}
    physical=physical23(states).astype(np.float32)
    state=((physical-tr['state_mean'])/tr['state_scale']).astype(np.float32)
    x5=np.delete(np.asarray(r5,np.float32),np.s_[105:110],axis=1)[:,tr['r5_keep']]
    x5=np.where(np.isfinite(x5),x5,tr['r5_mean'])
    fingerprint=((x5-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T
    nc,ns=len(r5),len(states)
    x=np.concatenate((np.broadcast_to(state[None,:,None,:],(nc,ns,4,23)),
                      np.broadcast_to(fingerprint[:,None,None,:],(nc,ns,4,2)),
                      np.broadcast_to(np.eye(4,dtype=np.float32),(nc,ns,4,4))),axis=-1).astype(np.float32)
    net=FrozenNet();scores=[]
    for seed in (17,23,41):
        checkpoint=folder/'expanded/full'/str(seed)/'model.msgpack'
        assert sha(checkpoint)==receipt['files'][str(checkpoint)]
        template=net.init(jax.random.PRNGKey(0),jnp.zeros((1,29),jnp.float32))
        parameters=serialization.from_bytes(template,checkpoint.read_bytes())
        probability=jax.nn.sigmoid(net.apply(parameters,jnp.asarray(x.reshape(-1,29))))
        scores.append(np.asarray(probability).reshape(nc,ns,4))
    scores=np.mean(scores,axis=0).mean(1)
    return np.asarray(['F10','F13','S500','RHR'])[scores.argmax(-1)],scores


def test():
    folder=D/'blackbox_families_v1'
    old=read(folder/'FROZEN_CHOICES.json')
    with np.load(folder/'R5_FEATURES.npz',allow_pickle=False) as z:
        names=list(z['controller_names']);r5=z['features']
    choice,scores=predict(read(folder/'EVAL_STATES.json'),r5)
    checks=[dict(controller=name,actual=str(a),reference=old['controllers'][name]['expanded'],
                 passed=str(a)==old['controllers'][name]['expanded']) for name,a in zip(names,choice)]
    write(HERE/'FROZEN_EXPANDED_REPRODUCTION.json',dict(utc=utc(),passed=all(c['passed'] for c in checks),
           checks=checks,physical_steps=0,training_updates=0,task_labels_read=False))
    print('Frozen expanded selector reproduction:',all(c['passed'] for c in checks))

if __name__=='__main__':test()
