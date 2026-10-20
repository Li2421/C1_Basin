"""Original frozen round16 controller-blind, controller-aware, and probe-rule comparators."""
from common import *
from frozen_expanded import FrozenNet
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
from flax import serialization
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
OLD_ARMS=('F10','F13','F2','S200','S300','S400','S500','S600')

def predict(states,r5):
    folder=D/'ring_controller_critic_v1';frozen=read(folder/'MODEL_FREEZE.receipt.json')['files']
    def checked(relative):
        path=folder/relative;assert sha(path)==frozen[relative];return path
    with np.load(checked('fits/full_transform.npz'),allow_pickle=False) as z:tr={k:z[k] for k in z.files}
    selection=read(checked('fits/selected_steps.json'));ge=read(checked('predictions/G_E_FROZEN_CHOICES.json'))
    raw=np.delete(np.asarray(r5,np.float32),np.s_[105:110],axis=1)[:,tr['r5_keep']]
    raw=np.where(np.isfinite(raw),raw,tr['r5_mean']);z5=((raw-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T
    physical=physical23(states).astype(np.float32);state=(physical-tr['state_mean'])/tr['state_scale']
    nc,ns=len(r5),len(states);choices={};scores={}
    for model in ('A0','A3'):
        pieces=[np.broadcast_to(state[None,:,None,:],(nc,ns,8,23))]
        if model=='A3':pieces.append(np.broadcast_to(z5[:,None,None,:],(nc,ns,8,2)))
        pieces.append(np.broadcast_to(np.eye(8,dtype=np.float32),(nc,ns,8,8)))
        x=np.concatenate(pieces,axis=-1).astype(np.float32);net=FrozenNet();values=[]
        for seed in (17,23,41):
            checkpoint=checked(f'fits/full/{model}/seed{seed}/step{selection[model]}.msgpack')
            template=net.init(jax.random.PRNGKey(0),jnp.zeros((1,x.shape[-1]),jnp.float32))
            parameters=serialization.from_bytes(template,checkpoint.read_bytes())
            values.append(np.asarray(jax.nn.sigmoid(net.apply(parameters,jnp.asarray(x.reshape(-1,x.shape[-1]))))).reshape(nc,ns,8))
        scores[model]=np.mean(values,axis=0).mean(1);choices[model]=np.asarray(OLD_ARMS)[scores[model].argmax(-1)]
    effect=(z5-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean']
    choices['G_E']=np.where(effect<.05,'F10','S500');scores['G_E']=effect
    return choices,scores
