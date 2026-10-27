"""Outcome-free model inference and guarded scores for fixed source-only fits."""
from common import *
from networks import Generator,critic,eta_mean,eta_from_noise,stable_int,CENTER,RADIUS
from anchored import predictions,logit,sigmoid,proposal_priors
from descriptors_v2 import target_guard
from flax import serialization
import jax,jax.numpy as jnp,numpy as np

CRITIC=critic('full');GENERATOR=Generator()
critic_apply=jax.jit(lambda p,h,c,ct,e:CRITIC.apply(p,h,c,ct,e))
generator_apply=jax.jit(lambda p,h,c:GENERATOR.apply(p,h,c))

def load(node,mode,seed=None):
    folder=HERE/'folds'/node
    selected=folder/mode/'SELECTED.json'
    row=read(selected)['selected']if seed is None else read(folder/mode/f'seed{seed}'/'FIT.receipt.json')
    assert sha(row['checkpoint'])==row['sha256']
    model=GENERATOR if mode=='generator'else CRITIC
    args=[jnp.zeros((1,6)),jnp.zeros((1,12))]
    if mode!='generator':args.extend([jnp.zeros((1,8)),jnp.zeros((1,3))])
    template=model.init(jax.random.PRNGKey(0),*args)
    return serialization.from_bytes(template,Path(row['checkpoint']).read_bytes()),row

def model_definition(node,mode):
    d=read(HERE/'folds'/node/'definition.json')
    if mode=='raw'and d['descriptor']!='c_raw':d=read(HERE/'folds'/node/'raw_definition.json')
    return d

def normalized_inputs(definition,data,clip=False):
    h=np.asarray(data['h'],float);c=np.asarray(data[definition['descriptor']],float);bounds=definition['ranges'];norm=definition['normalization']
    if clip:h=np.clip(h,bounds['h_lo'],bounds['h_hi']);c=np.clip(c,bounds['c_lo'],bounds['c_hi'])
    return((h-norm['h_mean'])/norm['h_scale']).astype(np.float32),np.repeat(((c-norm['c_mean'])/norm['c_scale'])[None],len(h),axis=0).astype(np.float32)

def raw_scores(params,definition,data,menu,clip=False):
    h,c=normalized_inputs(definition,data,clip);ct=np.asarray(data['controller_one_hot'],np.float32);menu=np.asarray(menu,float)
    C,S,A=len(ct),len(h),len(menu)
    hh=np.broadcast_to(h[None,:,None,:],(C,S,A,6));cc=np.broadcast_to(c[None,:,None,:],(C,S,A,12));xx=np.broadcast_to(ct[:,None,None,:],(C,S,A,8));ee=np.broadcast_to(menu[None,None],(C,S,A,3))
    return np.asarray(critic_apply(params,jnp.asarray(hh),jnp.asarray(cc),jnp.asarray(xx),jnp.asarray((ee-CENTER)/RADIUS,jnp.float32)),float)

def scored(raw,p0,shrinkage,mode,guard_row,state_fallback=True):
    logits=predictions(raw,p0,shrinkage,mode).copy()
    if state_fallback:
        mask=np.asarray(guard_row['states_out_of_range'],bool)
        logits[:,mask,:]=logit(p0)
    return logits,sigmoid(logits).mean((0,1))

def whole_scene(raw,p0,shrinkage,mode,guard_row,state_fallback=True,scene_fallback=True):
    logits,scene_scores=scored(raw,p0,shrinkage,mode,guard_row,state_fallback)
    if scene_fallback and guard_row['scene_out_of_range']:best=int(np.argmax(p0[:49]))
    elif shrinkage==0 and mode!='no_prior':best=int(np.argmax(p0[:49]))
    else:best=int(np.argmax(scene_scores))
    return best,logits,scene_scores

def scene_proposals(params,row,definition,data,name):
    h,c=normalized_inputs(definition,data);raw=np.asarray(generator_apply(params,jnp.asarray(h),jnp.asarray(c)))
    means=np.asarray(eta_mean(jnp.asarray(raw)));etas=[np.median(means,axis=0).astype(float).tolist()];samples=[]
    seed=stable_int('cross-scene-pipeline-v2-proposals',row['seed'],name);rng=np.random.default_rng(seed)
    for i in range(4):
        position=int(rng.integers(len(h)));noise=rng.standard_normal(3)
        eta=np.asarray(eta_from_noise(jnp.asarray(raw[position]),jnp.asarray(noise,jnp.float32)),float).tolist();etas.append(eta)
        samples.append(dict(proposal=i+1,state_position=position,state_index=data['state_indices'][position],noise=noise.tolist(),eta=eta))
    return dict(etas=etas,generator_seed=row['seed'],proposal_seed=seed,state_eta_means=means.tolist(),samples=samples,
        proposal0='Coordinatewise median of state eta_mean outputs; float32 distribution, exact resulting float64 request values.')
