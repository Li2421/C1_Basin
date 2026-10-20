"""Meaningful preprocessing, missing-label, warm-function and ablation tests."""
from common import *
from critic_model import *
from frozen_networks import Critic as FrozenCritic,merge_initialized,CENTER,RADIUS


def main():
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data.update(h=z['h'][:48],c=z['c'][:48],action=z['action'][:48])
    rng=np.random.default_rng(17);data['labels']=rng.integers(0,2,size=(62,48,13,2),dtype=np.int8)
    data['labels'][:1,:1,:3]=np.asarray([[-2,-2],[-1,-1],[1,-1]])
    y,n,w=observations(data['labels']);assert y[0,0,:3].tolist()==[0,0,1] and n[0,0,:3].tolist()==[0,0,1]
    source=sorted(set(data['types'])-{'Flow'});transform=source_transform(data,source)
    alternate={**data,'features':data['features'].copy()};alternate['features'][data['types']=='Flow']=12345.
    other=source_transform(alternate,source)
    for k in ('feature_fill','feature_mean','feature_scale','h_mean','h_scale','c_mean','c_scale'):assert np.array_equal(transform[k],other[k])
    probability=sampling_weights(data,source).reshape(62,48,13)
    assert np.all(probability[data['types']=='Flow']==0.)
    for t in source:assert np.isclose(probability[data['types']==t].sum(),1/len(source))
    arrays=transform_arrays(data,transform);assert all(np.isfinite(v).all() for v in arrays.values())
    original,norm=original_parameters();oldmodel=FrozenCritic()
    old_h=(data['h'][:8]-norm['h_mean'])/np.asarray(norm['h_std'])
    old_c=(data['c'][:8]-norm['c_mean'])/np.asarray(norm['c_std'])
    a=arrays['action'][:8,0]
    expected=np.asarray(oldmodel.apply({'params':original},jnp.asarray(old_h),jnp.asarray(old_c),jnp.asarray(a[:,:3]),method=oldmodel.ring))
    errors={}
    for mode in MODES:
        net=network(mode);pa=initialize(mode,17,'warm',transform,(original,norm))
        h=jnp.asarray(arrays['h'][:8]);c=jnp.asarray(arrays['c'][:8]);C=jnp.asarray(arrays['controller'][:8]);aa=jnp.asarray(a)
        logits=np.asarray(net.apply(pa,h,c,C,aa))
        if mode!='no_state':
            error=float(np.max(np.abs(np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))-np.asarray(jax.nn.sigmoid(jnp.asarray(expected))))))
            errors[mode]=error;assert error<2e-6,(mode,error)
        if mode=='no_controller':assert np.array_equal(logits,np.asarray(net.apply(pa,h,c,C+123.,aa)))
        if mode=='no_state':assert np.array_equal(logits,np.asarray(net.apply(pa,h+123.,c-123.,C,aa)))
        restored=serialization.from_bytes(pa,serialization.to_bytes(pa))
        assert np.array_equal(logits,np.asarray(net.apply(restored,h,c,C,aa)))
    tied=np.ones((3,2,4,13),np.float32)*.8
    assert np.all(choices(tied)['mean']==12) and np.all(choices(tied)['conservative']==12)
    write(HERE/'CRITIC_MODEL_TESTS.json',dict(utc=utc(),passed=True,source_only_normalization=True,
        missing_and_unknown_not_failures=True,type_equal_sampling=True,warm_start_probability_max_error=errors,
        ablated_input_invariance=True,serialization_exact=True,tie_prefers_progress=True,new_rollouts=0,real_training_labels_read=False))
    seal('CRITIC_MODEL_TESTS.receipt.json',[HERE/'CRITIC_MODEL_TESTS.json',HERE/'test_critic_model.py',HERE/'critic_model.py'])
    print('Critic model tests passed',errors,flush=True)


if __name__=='__main__':main()
