"""Meaningful dimension/gradient/no-controller independence gates before fitting."""
from common import *
from critic_model import initialize,network,source_transform,transform_arrays,sampling_weights
from train_critic import load_data
from menu_c6_v2 import menu,pools,physical_key
import jax,jax.numpy as jnp

def main():
    data=load_data();source=sorted(set(data['types']))[:-1];t=source_transform(data,source);arrays=transform_arrays(data,t);checks={};a=jnp.asarray(arrays['action'][0,0,:2]);h=jnp.asarray(arrays['h'][:2]);c=jnp.asarray(arrays['c'][:2]);C=jnp.asarray(arrays['controller'][:2]);p=initialize('no_controller',17);net=network('no_controller')
    q=net.apply(p,h,c,C,a);q2=net.apply(p,h,c,C+1000,a);checks['no_controller_exact_independence']=np.asarray(q).tobytes()==np.asarray(q2).tobytes()
    for mode in ('full','no_controller'):
        model=network(mode);pp=initialize(mode,17);g=jax.grad(lambda v:jnp.sum(model.apply(v,h,c,C,a)))(pp);checks[mode+'_finite_gradient']=all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(g))
    w=sampling_weights(data,source).reshape(122,48,53);checks['held_out_gradient_weight_zero']=bool(np.all(w[~np.isin(data['types'],source)]==0));checks['source_preprocessing_only']=set(t['source_controllers'])==set(data['names'][np.isin(data['types'],source)])
    rows=menu(data['proposals'][0]);pool=pools(rows);checks['canonical1313_unique']=len(rows)==len({physical_key(a) for a in rows})==1313;checks['first_layer_has_no_trigger']=all(a['trigger']=='none' for a,m in zip(rows,pool['first']) if m);checks['online_only_trigger_or_safety']=all(a['trigger']!='none' or not np.any(a['eta_a']) for a,m in zip(rows,pool['online']) if m)
    with np.load(D/'crosstype5_repair_gate_late_eta_v1/TRAINING_BUNDLE.npz',allow_pickle=False) as z:checks['all_old_labels_retained']=np.array_equal(data['labels'][:,:,:44],z['labels']);checks['all_old_action14_retained']=np.array_equal(data['action'][:,:,:44,:14],z['action'])
    result=dict(utc=utc(),passed=all(checks.values()),checks=checks,architecture='C5 unchanged except action19 and feature39 input widths',x64=False)
    write(HERE/'MODEL_TESTS.json',result);seal('MODEL_TESTS.receipt.json',[HERE/'MODEL_TESTS.json',HERE/'test_models_c6.py',HERE/'critic_model.py',HERE/'TRAINING_BUNDLE.receipt.json']);assert result['passed'],result
    print('Model gates passed',flush=True)

if __name__=='__main__':main()
