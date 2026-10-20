"""GPU input invariance, warm equivalence and held-label isolation."""
from common import *
from action_only_model import *
from train_action_only import compiled


def main():
    assert jax.default_backend()=='gpu'
    with np.load(HERE/'CONTROLLER_FEATURES.npz') as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'STATE_INPUTS.npz') as z:data.update({k:z[k][:48] for k in ('h','c','action')})
    source=sorted(set(data['types'])-{'Flow'});transform=source_transform(data,source)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    rng=np.random.default_rng(17);data['labels']=rng.integers(0,2,(62,48,13,2))
    for init in ('scratch','warm'):
        p=initialize('action_only',17,init,transform);net=network('action_only')
        args=(arrays['h'][:8],arrays['c'][:8],arrays['controller'][:8],arrays['action'][:8,0])
        result=np.asarray(net.apply(p,*args))
        changed=(args[0]+100,args[1]-100,args[2]*0+123,args[3])
        assert np.array_equal(result,np.asarray(net.apply(p,*changed)))
        if init=='warm':
            old=base_initialize('no_state',17,'warm',transform)
            expected=np.asarray(base_network('no_state').apply(old,*args))
            assert np.max(np.abs(result-expected))<2e-6
    parameters=jax.tree_util.tree_map(lambda *v:jnp.stack(v),*[initialize('action_only',s,'scratch',transform) for s in SEEDS])
    opt,block,predict=compiled('action_only');states=jax.vmap(opt.init)(parameters)
    weights=sampling_weights(data,source);ids=np.stack([rng.choice(len(weights),(2,256),p=weights) for s in SEEDS],axis=1)
    y,n,_=observations(data['labels']);changed=data['labels'].copy();mask=data['types']=='Flow';changed[mask]=1-changed[mask]
    y2,n2,_=observations(changed)
    a=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(y.reshape(-1)),jnp.asarray(n.reshape(-1)))
    b=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(y2.reshape(-1)),jnp.asarray(n2.reshape(-1)))
    assert all(np.array_equal(np.asarray(x),np.asarray(z)) for x,z in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
    write(HERE/'ACTION_ONLY_TESTS.json',dict(utc=utc(),passed=True,state_and_controller_invariance=True,
        warm_matches_original_no_state_initialization=True,held_labels_do_not_change_optimizer=True,real_labels_used=False))
    seal('ACTION_ONLY_TESTS.receipt.json',[HERE/'ACTION_ONLY_TESTS.json',HERE/'action_only_model.py',HERE/'train_action_only.py',HERE/'test_action_only.py'])
    print('Matched action-only branch GPU tests passed',flush=True)


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
