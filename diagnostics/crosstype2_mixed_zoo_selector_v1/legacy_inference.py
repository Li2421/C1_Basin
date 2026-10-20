"""Immutable nine-type all-development critic; original C24/action10 only."""
from core import *
import jax,jax.numpy as jnp
from flax import serialization
import legacy_critic_model as original
from legacy_train_critic import compiled


def predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*predict(model['full'],data)+(1-w)*predict(model['no_controller'],data)
    for key in ('checkpoint','transform','fit_receipt'):assert sha(model[key+'_path'])==model[key+'_sha256']
    transform=read(model['transform_path'])
    for key in ('h_mean','h_scale','c_mean','c_scale'):transform[key]=np.asarray(transform[key],np.float32)
    for key in ('feature_fill','feature_mean','feature_scale'):transform[key]=np.asarray(transform[key],np.float64)
    mode=model['mode'];template=jax.tree.map(lambda *v:jnp.stack(v),*[original.initialize(mode,s,'scratch',transform) for s in (17,23,41)])
    parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes())
    old={**data,'features':data['features'][:,:24],'action':data['action'][:,:13,:10]};arrays=original.transform_arrays(old,transform)
    # Original compiled predictor body and its argument/fusion structure retained.
    return np.asarray(compiled(mode)[2](parameters,{k:jnp.asarray(v) for k,v in arrays.items()}))


def choices(data):
    frozen=read(D/'crosstype1b_generator_critic_pipeline_v1/FROZEN_PIPELINE.json')
    probability=predict(frozen['all_types']['full']['model'],data)
    return original.choices(probability)['mean']
