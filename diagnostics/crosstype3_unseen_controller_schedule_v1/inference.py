"""Typed, highest-precision frozen inference for new and legacy comparators."""
from common import *
from critic_model import transform_arrays,initialize
from train_critic import compiled
from flax import serialization
import jax,jax.numpy as jnp
def restore_transform(model):
    for kind in ('checkpoint','transform','fit_receipt'):assert sha(model[kind+'_path'])==model[kind+'_sha256']
    t=read(model['transform_path'])
    for key in ('h_mean','h_scale','c_mean','c_scale'):t[key]=np.asarray(t[key],np.float32)
    for key in ('feature_fill','feature_mean','feature_scale'):t[key]=np.asarray(t[key],np.float64)
    return t
def predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*predict(model['full'],data)+(1-w)*predict(model['no_controller'],data)
    t=restore_transform(model);template=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[initialize(model['mode'],s) for s in (17,23,41)]);parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes());arrays=transform_arrays(data,t)
    if arrays['action'].ndim==3:arrays['action']=np.broadcast_to(arrays['action'][None],(len(data['names']),*arrays['action'].shape))
    result=np.asarray(compiled(model['mode'])[2](parameters,{k:jnp.asarray(v) for k,v in arrays.items()}));assert np.isfinite(result).all() and np.all((result>=0)&(result<=1));assert result.shape==(3,*arrays['action'].shape[:3]);return result
def legacy_predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*legacy_predict(model['full'],data)+(1-w)*legacy_predict(model['no_controller'],data)
    from legacy_critic_model import initialize as legacy_initialize,transform_arrays as legacy_arrays
    from legacy_train_critic import compiled as legacy_compiled
    t=restore_transform(model);template=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[legacy_initialize(model['mode'],s,'scratch',t) for s in (17,23,41)]);parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes());arrays={k:jnp.asarray(v) for k,v in legacy_arrays(data,t).items()};result=np.asarray(legacy_compiled(model['mode'])[2](parameters,arrays));assert np.isfinite(result).all() and np.all((result>=0)&(result<=1));return result
def legacy_actions(etas,base):
    from frozen_networks import CENTER,RADIUS
    actions=[]
    for e in etas[:5]:
        actions.extend([dict(name='legacy_fixed',eta_a=list(e),eta_b=list(e),switch_mode='fixed',switch_step=700,safety_baseline=False),dict(name='legacy_progress',eta_a=list(e),eta_b=base['eta'][10],switch_mode='progress',switch_step=700,safety_baseline=False)])
    actions.extend([dict(name='safety_baseline',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',switch_step=700,safety_baseline=True),dict(name='legacy_step500',eta_a=base['eta'][13],eta_b=base['eta'][10],switch_mode='step500',switch_step=500,safety_baseline=False),dict(name='legacy_progress_anchor',eta_a=base['eta'][13],eta_b=base['eta'][10],switch_mode='progress',switch_step=700,safety_baseline=False)])
    vectors=[]
    for a in actions:vectors.append([*((np.asarray(a['eta_a'])-CENTER)/RADIUS),*((np.asarray(a['eta_b'])-CENTER)/RADIUS),*[float(a['switch_mode']==m) for m in ('fixed','progress','step500')],float(a['safety_baseline'])])
    return actions,np.asarray(vectors,np.float32)
def test():
    assert jax.default_backend()=='gpu'
    from train_critic import load_data
    data=load_data();selected=read(HERE/'SELECTED_MODELS.json');checks=[]
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:expected=z['full'];expected_noC=z['no_controller']
    with jax.default_matmul_precision('highest'):
        for held,models in selected['outer'].items():
            mask=data['types']==held
            for key,truth in [('full',expected),('no_controller',expected_noC)]:
                p=predict(models[key],data);equal=p[:,mask].tobytes()==truth[:,mask].tobytes();checks.append(dict(type=held,model=key,bitwise=equal,max_abs_error=float(np.max(abs(p[:,mask]-truth[:,mask])))))
    OLD=D/'crosstype1b_generator_critic_pipeline_v1'
    with np.load(OLD/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:legacy={k:z[k] for k in ('names','types','features')}
    with np.load(OLD/'STATE_INPUTS.npz',allow_pickle=False) as z:legacy.update({k:z[k][48:] for k in ('h','c','action')});old_all_actions=z['action']
    old_candidates=read(OLD/'CANDIDATES.json')['states'];base=read(OLD/'BASE_PROTOCOL.json');rebuilt=np.asarray([legacy_actions(s['proposals'],base)[1] for s in old_candidates],np.float32);assert rebuilt.tobytes()==old_all_actions.tobytes(),'Legacy action encoding must be exactly reconstructed'
    with np.load(OLD/'NEW_EVAL_PROBABILITIES.npz',allow_pickle=False) as z:expected_legacy=z['full']
    old_models=read(OLD/'FROZEN_PIPELINE.json');legacy_checks=[]
    with jax.default_matmul_precision('highest'):
        for held,group in old_models['outer'].items():
            mask=legacy['types']==held;p=legacy_predict(group['full'],legacy);legacy_checks.append(dict(type=held,bitwise=p[:,mask].tobytes()==expected_legacy[:,mask].tobytes(),max_abs_error=float(np.max(abs(p[:,mask]-expected_legacy[:,mask])))))
    from design import action_case
    old_states=read(OLD/'NEW_EVAL_STATES.json');old_cases=read(OLD/'new_evaluation_CASES.json');identity_checks=0
    for c in old_cases:
        if c['phase_state_index']!=0 or c['seed']!=0:continue
        actions,_=legacy_actions(old_candidates[48]['proposals'],base);a=actions[c['candidate_index']] if c['candidate_index']<13 else dict(name='nominal_only',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',switch_step=700);current,_=action_case(base,c['base_controller_name'],old_states[0],a,0,0,c['candidate_index']);assert current['rollout_uid']==c['rollout_uid'] and current['controller_uid']==c['controller_uid'];identity_checks+=1
    assert identity_checks==62*14
    result=dict(utc=utc(),passed=all(r['bitwise'] for r in checks+legacy_checks),new_model_checks=checks,original_cross_type1b_checks=legacy_checks,legacy_all72_action_encodings_bitwise=True,legacy_exact_uid_reconstructions=identity_checks,x64=False,matmul_precision='highest',new_eval_labels_read=False)
    write(HERE/'FROZEN_INFERENCE_TESTS.json',result);seal('FROZEN_INFERENCE_TESTS.receipt.json',[HERE/'FROZEN_INFERENCE_TESTS.json',HERE/'inference.py',HERE/'SELECTED_MODELS.json'])
    assert result['passed'],'Frozen/legacy model restore must be bitwise; do not relax the gate'
if __name__=='__main__':test()
