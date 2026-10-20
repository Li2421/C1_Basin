from common import *
from critic_model import initialize,transform_arrays
from train_critic import compiled
from flax import serialization
import jax,jax.numpy as jnp


def predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*predict(model['full'],data)+(1-w)*predict(model['no_controller'],data)
    for key in ('checkpoint','transform','fit_receipt'):assert sha(model[key+'_path'])==model[key+'_sha256']
    transform=read(model['transform_path'])
    for key in ('h_mean','h_scale','c_mean','c_scale'):transform[key]=np.asarray(transform[key],np.float32)
    for key in ('feature_fill','feature_mean','feature_scale'):transform[key]=np.asarray(transform[key],np.float64)
    mode=model['mode'];template=jax.tree.map(lambda *v:jnp.stack(v),*[initialize(mode,s,'scratch',transform) for s in (17,23,41)])
    parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes())
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    return np.asarray(compiled(mode)[2](parameters,arrays))
