"""New raw two-eta schedule encoding; original architecture and source scaling."""
from common import *
from legacy_critic_model import Critic,source_transform,transform_arrays,observations,sampling_weights
import jax,jax.numpy as jnp
SEEDS=(17,23,41)
MODES=('full','no_controller')
def network(mode):
    assert mode in MODES
    return Critic(use_state=True,use_controller=mode!='no_controller')
def initialize(mode,seed):return network(mode).init(jax.random.PRNGKey(seed),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,24)),jnp.zeros((1,14)))
def metrics(probabilities,labels,types,selected_types,candidate_indices):
    raw=np.asarray(probabilities).mean(0);p=np.clip(raw,1e-6,1-1e-6);y,n,w=observations(labels);loss=w*(-y/np.maximum(n,1)*np.log(p)-(n-y)/np.maximum(n,1)*np.log1p(-p));nll=loss.sum((1,2))/np.maximum(w.sum((1,2)),1)
    empirical=y/np.maximum(n,1);true=empirical[...,None]-empirical[...,None,:];pred=raw[...,None]-raw[...,None,:];valid=(n[...,None]>0)&(n[...,None,:]>0)&(true>=.25)
    # Packed metadata, never the old dense 9x9x6 grid identity.
    index=np.asarray(candidate_indices);diag=index%2==1
    d1,d2=diag[...,None],diag[...,None,:];pair=index//32;switch=index//4%8
    groups={'all':np.ones_like(valid,bool),'diagonal':d1&d2,'non_diagonal':~d1&~d2,'cross':d1!=d2,'same_pair_different_time':(~d1&~d2)&(pair[...,None]==pair[...,None,:])&(switch[...,None]!=switch[...,None,:])}
    zero=index//2%2==1
    groups['zero_vs_nonzero']=zero[...,None]!=zero[...,None,:]
    rank={}
    for group,mask in groups.items():
        supported=valid&mask;total=supported.sum((-1,-2));correct=(((pred>0)+.5*(pred==0))*supported).sum((-1,-2));state=correct/np.maximum(total,1);count=(total>0).sum(1);controller=(state*(total>0)).sum(1)/np.maximum(count,1)
        rank[group]=(controller,count>0,total)
    result={}
    for t in selected_types:
        mask=types==t;row=dict(nll=float(nll[mask].mean()),ranking={})
        for group,(values,support,pairs) in rank.items():
            supported=mask&support;row['ranking'][group]=dict(accuracy=float(values[supported].mean()) if supported.any() else None,supported_controllers=int(supported.sum()),supported_states=int((pairs[mask]>0).sum()),identifiable_pairs=int(pairs[mask].sum()))
        row['ranking_accuracy']=row['ranking']['all']['accuracy'];result[str(t)]=row
    return result
