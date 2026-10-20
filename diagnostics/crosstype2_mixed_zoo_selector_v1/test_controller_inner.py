from common import *
from test_critic import fixture
from critic_model import *
from train_critic import compiled
import jax,jax.numpy as jnp


def main():
    assert jax.default_backend()=='gpu'
    data=fixture();held=data['names'][0];data['types']=data['types'].astype('U160');data['types'][0]='__held_controller__'+held
    source=['A'];transform=source_transform(data,source);assert held not in transform['source_controllers']
    changed={**data,'features':data['features'].copy()};changed['features'][0]=10000
    other=source_transform(changed,source)
    for k in ('feature_fill','feature_mean','feature_scale'):assert np.array_equal(transform[k],other[k])
    weights=sampling_weights(data,source).reshape(12,48,16);assert weights[0].sum()==0 and weights[4:].sum()==0 and np.isclose(weights[1:4].sum(),1.)
    rng=np.random.default_rng(202610182401);indices=np.stack([rng.choice(weights.size,size=(2,256),p=weights.reshape(-1)) for _ in SEEDS],axis=1).astype(np.int32)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()};y,n,_=observations(data['labels']);labels=data['labels'].copy();labels[[0,*range(4,12)]]=1-labels[[0,*range(4,12)]];yy,nn,_=observations(labels)
    for mode in ('full','no_controller'):
        parameters=jax.tree.map(lambda *v:jnp.stack(v),*[initialize(mode,s,'scratch',transform) for s in SEEDS]);opt,block,_=compiled(mode);states=jax.vmap(opt.init)(parameters)
        a=block(parameters,states,jnp.asarray(indices),arrays,jnp.asarray(y.reshape(-1)),jnp.asarray(n.reshape(-1)))
        b=block(parameters,states,jnp.asarray(indices),arrays,jnp.asarray(yy.reshape(-1)),jnp.asarray(nn.reshape(-1)))
        assert all(np.array_equal(np.asarray(v),np.asarray(w)) for v,w in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
    write(HERE/'CONTROLLER_INNER_TESTS.json',dict(utc=utc(),passed=True,user_authorized_m1=True,held_controller_and_outer_types_excluded_from_sampling=True,held_controller_features_excluded_from_preprocessing=True,held_labels_do_not_change_parameters_optimizer_or_losses=True,real_labels_used=False))
    seal('CONTROLLER_INNER_TESTS.receipt.json',[HERE/'CONTROLLER_INNER_TESTS.json',HERE/'test_controller_inner.py',HERE/'train_critic.py',HERE/'single_source_selection.py'])
    print('Single-source controller-inner isolation tests passed')


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
