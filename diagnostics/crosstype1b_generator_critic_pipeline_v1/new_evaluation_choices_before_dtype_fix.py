"""Frozen highest-precision GPU inference; no rollout labels are read."""
from common import *
from critic_model import transform_arrays,choices,MODES
from action_only_model import initialize
from train_critic import compiled
from train_action_only import compiled as action_compiled
from flax import serialization
import jax,jax.numpy as jnp


def predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        weight=model['full_weight'];assert 0<weight<=1
        return weight*predict(model['full'],data)+(1-weight)*predict(model['no_controller'],data)
    for kind in ('checkpoint','transform','fit_receipt'):
        assert sha(model[kind+'_path'])==model[kind+'_sha256']
    transform=read(model['transform_path']);mode=model['mode']
    params=[initialize(mode,seed,'scratch',transform) for seed in (17,23,41)]
    template=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*params)
    parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes())
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    predictor=action_compiled(mode) if mode=='action_only' else compiled(mode)
    return np.asarray(predictor[2](parameters,arrays))


def main():
    assert not (HERE/'NEW_EVAL_CHOICES.receipt.json').exists()
    for path,digest in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(path)==digest,path
    assert jax.default_backend()=='gpu','Frozen inference uses a Slurm GPU shard'
    assert not (HERE/'new_evaluation_run_SUBMISSION.json').exists()
    frozen=read(HERE/'FROZEN_PIPELINE.json');base=read(HERE/'BASE_PROTOCOL.json')
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data={k:z[k] for k in ('names','types','features')}
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:
        data.update({k:z[k][48:] for k in ('h','c','action','state_uids')})
    assert len(data['h'])==24 and list(data['names'])==base['controller_order']
    nc=len(data['names']);output={k:np.full((nc,24),-1,np.int32) for k in
        ('pipeline','pipeline_conservative','pipeline_scratch','pipeline_warm','no_controller','no_state')}
    probabilities={mode:np.zeros((3,nc,24,13),np.float32) for mode in MODES}
    with jax.default_matmul_precision('highest'):
        for held,group in frozen['outer'].items():
            mask=data['types']==held
            for mode in MODES:
                assert held not in group[mode]['source_types']
                p=predict(group[mode],data);probabilities[mode][:,mask]=p[:,mask];pick=choices(p)
                output['pipeline' if mode=='full' else mode][mask]=pick['mean'][mask]
                if mode=='full':output['pipeline_conservative'][mask]=pick['conservative'][mask]
            for init in ('scratch','warm'):
                p=predict(group['full_'+init],data);output['pipeline_'+init][mask]=choices(p)['mean'][mask]
        p=predict(frozen['all_types']['full']['model'],data)
        probabilities['all_types_reference']=p;output['all_types_reference']=choices(p)['mean']
    # The previous controller-only comparator is already frozen; its choice is
    # independent of state and is copied from the sealed development receipt.
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:
        old=z['frozen_controller_selector'];assert np.all(old==old[:,:1])
        output['frozen_controller_selector']=np.broadcast_to(old[:,:1],(nc,24))
    states=read(HERE/'CANDIDATES.json')['states'][48:]
    output['original_pipeline']=np.broadcast_to(np.asarray([s['original_pipeline_choice'] for s in states])[None],(nc,24))
    output['generator_only']=np.ones((nc,24),np.int32)
    for name,a in (('safety_baseline',10),('S500',11),('Sprog',12),('nominal_only',13)):
        output['all_'+name]=np.full((nc,24),a,np.int32)
    assert all(v.shape==(nc,24) and np.all((v>=0)&(v<14)) for v in output.values())
    np.savez_compressed(HERE/'NEW_EVAL_CHOICES.npz',names=data['names'],types=data['types'],state_uids=data['state_uids'],**output)
    np.savez_compressed(HERE/'NEW_EVAL_PROBABILITIES.npz',names=data['names'],types=data['types'],**probabilities)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/'NEW_EVAL_CHOICES.npz',HERE/'NEW_EVAL_PROBABILITIES.npz',
        HERE/'new_evaluation_choices.py',HERE/'FROZEN_PIPELINE.receipt.json',HERE/'DEVELOPMENT_CHOICES.receipt.json'],
        backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),matmul_precision='highest',
        outcomes_read=False,before_first_new_evaluation_submission=True)
    print('Frozen new-state choices sealed before any new-state rollout',flush=True)


if __name__=='__main__':main()
