"""State/controller/action critic and source-only preprocessing."""
from core import *
import flax.linen as nn
from flax import serialization
from flax.core import unfreeze
import jax,jax.numpy as jnp
from frozen_networks import Critic as OriginalCritic,merge_initialized
jax.config.update('jax_enable_x64',False)

SEEDS=(17,23,41)
MODES=('full','no_controller','no_state','action_only')


class Critic(nn.Module):
    use_state: bool=True
    use_controller: bool=True

    def setup(self):
        if self.use_state:
            self.ring_adapter=nn.Dense(96);self.context=nn.Dense(32)
        self.action_encoder=nn.Dense(32)
        if self.use_controller:self.controller_encoder=nn.Dense(32)
        self.shared1=nn.Dense(128);self.shared2=nn.Dense(64);self.out=nn.Dense(1)

    def __call__(self,h,c,controller,action):
        parts=[]
        if self.use_state:parts.extend([nn.silu(self.ring_adapter(h)),nn.silu(self.context(c))])
        parts.append(nn.silu(self.action_encoder(action)))
        if self.use_controller:parts.append(nn.silu(self.controller_encoder(controller)))
        x=jnp.concatenate(parts,axis=-1)
        x=nn.silu(self.shared1(x));x=nn.silu(self.shared2(x))
        return self.out(x)[...,0]


def network(mode):
    assert mode in MODES
    return Critic(use_state=mode in ('full','no_controller'),use_controller=mode in ('full','no_state'))


def source_transform(data,source_types):
    types=data['types'];mask=np.isin(types,source_types);x=data['features'][mask]
    assert mask.any() and set(types[mask])==set(source_types)
    tt=types[mask];w=np.asarray([1/(len(source_types)*np.sum(tt==t)) for t in tt])
    valid=np.isfinite(x)
    fill=np.sum(np.where(valid,x,0)*w[:,None],0)/np.maximum(np.sum(valid*w[:,None],0),1e-30)
    clean=np.where(valid,x,fill);mean=np.sum(clean*w[:,None],0)
    scale=np.sqrt(np.sum(np.square(clean-mean)*w[:,None],0));scale=np.where(scale>1e-6,scale,1.)
    hm=data['h'].mean(0);hs=data['h'].std(0);hs=np.where(hs>1e-6,hs,1.)
    cm=data['c'].mean(0);cs=data['c'].std(0);cs=np.where(cs>1e-6,cs,1.)
    return dict(source_types=sorted(map(str,source_types)),source_controllers=data['names'][mask].tolist(),
        feature_fill=fill,feature_mean=mean,feature_scale=scale,h_mean=hm,h_scale=hs,c_mean=cm,c_scale=cs)


def transform_arrays(data,transform):
    x=data['features'];C=(np.where(np.isfinite(x),x,transform['feature_fill'])-transform['feature_mean'])/transform['feature_scale']
    return dict(h=np.asarray((data['h']-transform['h_mean'])/transform['h_scale'],np.float32),
        c=np.asarray((data['c']-transform['c_mean'])/transform['c_scale'],np.float32),
        controller=np.asarray(C,np.float32),action=np.asarray(data['action'],np.float32))


def original_parameters():
    norm=read(D/'orthoflow3_generator_critic_v1/normalization.json')
    dims={sc:len(v['h_mean']) for sc,v in norm['scenarios'].items()};cdim=len(norm['environment_keys'])+3
    template=merge_initialized(OriginalCritic(),dims,cdim,critic=True)
    path=D/'orthoflow3_generator_critic_v1/critic/seed23/checkpoint.msgpack'
    assert sha(path)=='f0a2dd358a477590cf434c7803d6ddcc307509157a9698c7dbfaafcf302e02ed'
    return serialization.from_bytes(template,path.read_bytes())['params'],norm['scenarios']['ring_exchange']


def initialize(mode,seed,initialization,transform,original=None):
    net=network(mode)
    params=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,34)),jnp.zeros((1,16)))
    if initialization=='scratch':return params
    assert initialization=='warm'
    old,norm=original if original is not None else original_parameters()
    p=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),unfreeze(params))
    q=p['params']
    def adapted(name,new_mean,new_scale,old_mean,old_scale):
        kernel=np.asarray(old[name]['kernel']);bias=np.asarray(old[name]['bias'])
        ratio=np.asarray(new_scale)/np.asarray(old_scale)
        shift=(np.asarray(new_mean)-np.asarray(old_mean))/np.asarray(old_scale)
        return dict(kernel=(ratio[:,None]*kernel).astype(np.float32),bias=(bias+shift@kernel).astype(np.float32))
    if mode in ('full','no_controller'):
        q['ring_adapter']=adapted('ring_adapter',transform['h_mean'],transform['h_scale'],norm['h_mean'],norm['h_std'])
        q['context']=adapted('context',transform['c_mean'],transform['c_scale'],norm['c_mean'],norm['c_std'])
    q['action_encoder']['kernel'][:3]=np.asarray(old['eta_encoder']['kernel'])
    q['action_encoder']['kernel'][3:]=0.
    q['action_encoder']['bias']=np.asarray(old['eta_encoder']['bias']).copy()
    q['shared1']['kernel'][:]=0.
    if mode in ('no_state','action_only'):
        q['shared1']['kernel'][:32]=np.asarray(old['shared1']['kernel'])[128:160]
        hh=(np.asarray(transform['h_mean'])-norm['h_mean'])/np.asarray(norm['h_std'])
        cc=(np.asarray(transform['c_mean'])-norm['c_mean'])/np.asarray(norm['c_std'])
        z=np.asarray(jax.nn.silu(jnp.asarray(hh@np.asarray(old['ring_adapter']['kernel'])+np.asarray(old['ring_adapter']['bias']))))
        ec=np.asarray(jax.nn.silu(jnp.asarray(cc@np.asarray(old['context']['kernel'])+np.asarray(old['context']['bias']))))
        q['shared1']['bias']=(np.asarray(old['shared1']['bias'])+np.r_[z,ec]@np.asarray(old['shared1']['kernel'])[:128]).astype(np.float32)
    else:
        q['shared1']['kernel'][:160]=np.asarray(old['shared1']['kernel'])
        q['shared1']['bias']=np.asarray(old['shared1']['bias']).copy()
    for name in ('shared2','out'):q[name]=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),old[name])
    return jax.tree_util.tree_map(jnp.asarray,p)


def observations(labels):
    labels=np.asarray(labels)
    assert np.isin(labels,[-2,-1,0,1]).all()
    y=np.sum(labels==1,axis=-1).astype(np.float32)
    n=np.sum(labels>=0,axis=-1).astype(np.float32)
    weight=np.minimum(n,16)
    return y,n,weight


def sampling_weights(data,source_types):
    _,n,_=observations(data['labels']);types=data['types'];mask=np.isin(types,source_types)
    weights=np.zeros_like(n,dtype=float)
    for t in source_types:
        pool=np.flatnonzero(types==t)
        for i in pool:
            valid=n[i]>0;state_valid=valid.any(-1)
            assert state_valid.any()
            weights[i]=valid/(len(source_types)*len(pool)*state_valid.sum()*np.maximum(valid.sum(-1),1)[:,None])
    assert np.isclose(weights.sum(),1.) and np.all(weights[~mask]==0.)
    return weights.reshape(-1)


def candidate_choice(scores):
    scores=np.asarray(scores);choice=np.argmax(scores,axis=-1)
    return np.where(scores[...,12]>=scores.max(-1)-1e-12,12,choice)


def choices(probabilities):
    p=np.asarray(probabilities);mean=p.mean(0);sd=p.std(0)
    primary=candidate_choice(mean);lcb=mean-sd
    conservative=candidate_choice(lcb)
    conservative=np.where(np.take_along_axis(lcb,conservative[...,None],-1)[...,0]<mean[...,12],12,conservative)
    return dict(mean=primary,conservative=conservative)


def metrics(probabilities,labels,types,selected_types):
    p=np.clip(np.asarray(probabilities).mean(0),1e-6,1-1e-6);y,n,weight=observations(labels)
    loss=weight*(-y/np.maximum(n,1)*np.log(p)-(n-y)/np.maximum(n,1)*np.log1p(-p))
    controller_nll=loss.sum((1,2))/np.maximum(weight.sum((1,2)),1)
    empirical=y/np.maximum(n,1)
    true_delta=empirical[...,None]-empirical[...,None,:]
    pred_delta=p[...,None]-p[...,None,:]
    valid=(n[...,None]>0)&(n[...,None,:]>0)&(true_delta>=.25)
    ranking=((pred_delta>0)+.5*(pred_delta==0))*valid
    per_state=ranking.sum((-1,-2))/np.maximum(valid.sum((-1,-2)),1)
    support=valid.any((-1,-2))
    controller_rank=np.sum(per_state*support,axis=1)/np.maximum(support.sum(1),1)
    supported_controller=support.any(1)
    return {str(t):dict(nll=float(controller_nll[types==t].mean()),
        ranking_accuracy=float(controller_rank[(types==t)&supported_controller].mean()) if np.any((types==t)&supported_controller) else None,
        ranking_supported_states=int(support[types==t].sum()),ranking_supported_controllers=int(np.sum((types==t)&supported_controller))) for t in selected_types}
