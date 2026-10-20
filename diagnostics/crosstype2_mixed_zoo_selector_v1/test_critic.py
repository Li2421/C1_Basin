"""Synthetic-only tests: no held-type outcome can enter a fitted parameter."""
from common import *
from critic_model import *
from train_critic import compiled
from frozen_networks import Critic as FrozenCritic


def fixture():
    rng=np.random.default_rng(202610182301)
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'][:48];c=z['c'][:48];action=z['action'][:48]
    return dict(names=np.asarray([f'synthetic{i}' for i in range(12)]),types=np.repeat(['A','B','C'],4),features=rng.normal(size=(12,34)),h=h,c=c,action=action,labels=rng.integers(0,2,(12,48,16,2),dtype=np.int8))


def main():
    assert jax.default_backend()=='gpu'
    data=fixture();data['features'][0,:]=np.nan;data['labels'][0,0,:3]=[[-2,-2],[-1,-1],[1,-1]]
    y,n,w=observations(data['labels']);assert y[0,0,:3].tolist()==[0,0,1] and n[0,0,:3].tolist()==[0,0,1]
    source=['B','C'];transform=source_transform(data,source)
    alternate={**data,'features':data['features'].copy()};alternate['features'][data['types']=='A']=12345.
    other=source_transform(alternate,source)
    for key in ('feature_fill','feature_mean','feature_scale','h_mean','h_scale','c_mean','c_scale'):assert np.array_equal(transform[key],other[key])
    weights=sampling_weights(data,source);assert np.all(weights.reshape(12,48,16)[:4]==0)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()};assert all(np.isfinite(v).all() for v in arrays.values())
    rng=np.random.default_rng(41);ids=np.stack([rng.choice(len(weights),size=(2,256),p=weights) for _ in SEEDS],axis=1).astype(np.int32)
    Y,N,_=observations(data['labels']);perturbed=data['labels'].copy();perturbed[:4]=1-np.maximum(perturbed[:4],0);Y2,N2,_=observations(perturbed)
    errors={};original,norm=original_parameters();oldmodel=FrozenCritic()
    oldh=(data['h'][:8]-norm['h_mean'])/np.asarray(norm['h_std']);oldc=(data['c'][:8]-norm['c_mean'])/np.asarray(norm['c_std'])
    aa=arrays['action'][:8,0];expected=np.asarray(jax.nn.sigmoid(oldmodel.apply({'params':original},jnp.asarray(oldh),jnp.asarray(oldc),aa[:,:3],method=oldmodel.ring)))
    for mode in MODES:
        net=network(mode);warm=initialize(mode,17,'warm',transform,(original,norm));hh=arrays['h'][:8];cc=arrays['c'][:8];C=arrays['controller'][:8]
        logits=np.asarray(net.apply(warm,hh,cc,C,aa))
        if mode in ('full','no_controller'):
            errors[mode]=float(np.max(np.abs(np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))-expected)));assert errors[mode]<2e-6
        if mode in ('no_controller','action_only'):assert np.array_equal(logits,np.asarray(net.apply(warm,hh,cc,C+100.,aa)))
        if mode in ('no_state','action_only'):assert np.array_equal(logits,np.asarray(net.apply(warm,hh+100.,cc-100.,C,aa)))
        restored=serialization.from_bytes(warm,serialization.to_bytes(warm));assert np.array_equal(logits,np.asarray(net.apply(restored,hh,cc,C,aa)))
        params=[initialize(mode,seed,'scratch',transform) for seed in SEEDS];parameters=jax.tree.map(lambda *a:jnp.stack(a),*params)
        opt,block,predict=compiled(mode);states=jax.vmap(opt.init)(parameters)
        result=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(Y.reshape(-1)),jnp.asarray(N.reshape(-1)))
        changed=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(Y2.reshape(-1)),jnp.asarray(N2.reshape(-1)))
        for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(changed)):assert np.array_equal(np.asarray(a),np.asarray(b))
        probability=np.asarray(predict(result[0][0],arrays));assert probability.shape==(3,12,48,16) and np.isfinite(probability).all()
    tied=np.ones((3,12,48,16),np.float32)*.8;assert np.all(choices(tied)['mean']==12) and np.all(choices(tied)['conservative']==12)
    write(HERE/'CRITIC_MODEL_TESTS.json',dict(utc=utc(),passed=True,warm_start_probability_max_error=errors,source_only_preprocessing=True,ablated_input_invariance=True,serialization_bitwise=True,ties_prefer_progress=True,real_labels_used=False))
    seal('CRITIC_MODEL_TESTS.receipt.json',[HERE/'CRITIC_MODEL_TESTS.json',HERE/'test_critic.py',HERE/'critic_model.py'])
    write(HERE/'CRITIC_TRAINING_TESTS.json',dict(utc=utc(),passed=True,modes=MODES,held_type_label_perturbation_parameters_optimizer_losses_bitwise_equal=True,three_seed_probability_shape=[3,12,48,16],real_labels_used=False))
    seal('CRITIC_TRAINING_TESTS.receipt.json',[HERE/'CRITIC_TRAINING_TESTS.json',HERE/'test_critic.py',HERE/'train_critic.py'])
    print('All four critic modes passed synthetic GPU source-isolation, warm-function and serialization tests',errors)


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
