"""Retain exact failure; distinguish host-restored arrays from live GPU arrays."""
from common import *
from new_evaluation_choices import predict,initialize,compiled,action_compiled,transform_arrays
from critic_model import choices
from flax import serialization
import jax,jax.numpy as jnp


def device_predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*device_predict(model['full'],data)+(1-w)*device_predict(model['no_controller'],data)
    transform=read(model['transform_path']);mode=model['mode']
    template=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[initialize(mode,s,'scratch',transform) for s in (17,23,41)])
    parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes())
    parameters=jax.tree_util.tree_map(jnp.asarray,parameters)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    predictor=action_compiled(mode) if mode=='action_only' else compiled(mode)
    return np.asarray(predictor[2](parameters,arrays))


def main():
    with np.load(HERE/'TRAINING_BUNDLE.npz') as z:data={k:z[k] for k in ('names','types','features','h','c','action')}
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES_V2.npz') as z:pf=z['full']
    with np.load(HERE/'MATCHED_NO_STATE_PROBABILITIES_V2B.npz') as z:ps=z['probability']
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz') as z:pc=z['no_controller']
    expected=dict(full=pf,no_state=ps,no_controller=pc);models=read(HERE/'SELECTED_MODELS_V2B.json')['outer'];rows=[]
    for held,group in models.items():
        mask=data['types']==held
        for mode in expected:
            p=predict(group[mode],data)[:,mask];pd=device_predict(group[mode],data)[:,mask];target=expected[mode][:,mask]
            row=dict(type=held,mode=mode,host_restored_max_error=float(np.max(np.abs(p-target))),
                device_restored_max_error=float(np.max(np.abs(pd-target))),host_vs_device_max_error=float(np.max(np.abs(pd-p))),
                host_choice_differences={k:int(np.sum(choices(p)[k]!=choices(target)[k])) for k in ('mean','conservative')},
                device_choice_differences={k:int(np.sum(choices(pd)[k]!=choices(target)[k])) for k in ('mean','conservative')})
            rows.append(row);print(json.dumps(row),flush=True)
    write(HERE/'FROZEN_INFERENCE_DIAGNOSIS.json',dict(utc=utc(),rows=rows,devices=[str(d) for d in jax.devices()],
        device_kinds=[d.device_kind for d in jax.devices()],x64=bool(jax.config.jax_enable_x64),precision='highest'))


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
