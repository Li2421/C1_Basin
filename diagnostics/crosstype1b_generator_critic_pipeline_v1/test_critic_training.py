"""GPU optimizer and held-label perturbation test using synthetic outcomes."""
from common import *
from critic_model import *
from train_critic import compiled


def main():
    assert jax.default_backend()=='gpu'
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data.update(h=z['h'][:48],c=z['c'][:48],action=z['action'][:48])
    rng=np.random.default_rng(41);data['labels']=rng.integers(0,2,(62,48,13,2),dtype=np.int8)
    source=sorted(set(data['types'])-{'Flow'});transform=source_transform(data,source)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    weights=sampling_weights(data,source);ids=np.stack([rng.choice(len(weights),size=(2,256),p=weights) for _ in SEEDS],axis=1).astype(np.int32)
    Y,N,_=observations(data['labels']);perturbed=data['labels'].copy();mask=data['types']=='Flow';perturbed[mask]=1-perturbed[mask]
    Y2,N2,_=observations(perturbed);assert np.array_equal(N,N2)
    checked=[]
    for mode in MODES:
        params=[initialize(mode,seed,'scratch',transform) for seed in SEEDS]
        parameters=jax.tree_util.tree_map(lambda *a:jnp.stack(a),*params)
        opt,block,predict=compiled(mode);states=jax.vmap(opt.init)(parameters)
        (p1,s1),loss1=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(Y.reshape(-1)),jnp.asarray(N.reshape(-1)))
        (p2,s2),loss2=block(parameters,states,jnp.asarray(ids),arrays,jnp.asarray(Y2.reshape(-1)),jnp.asarray(N2.reshape(-1)))
        for a,b in zip(jax.tree_util.tree_leaves((p1,s1,loss1)),jax.tree_util.tree_leaves((p2,s2,loss2))):assert np.array_equal(np.asarray(a),np.asarray(b))
        probability=np.asarray(predict(p1,arrays));assert probability.shape==(3,62,48,13)
        assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
        assert np.isfinite(np.asarray(loss1)).all()
        checked.append(mode)
    write(HERE/'CRITIC_TRAINING_TESTS.json',dict(utc=utc(),passed=True,modes=checked,
        held_type_labels_cannot_change_parameters_optimizer_or_losses=True,three_independent_seed_optimizers=True,
        expected_probability_shape=[3,62,48,13],real_outcome_training=False,x64=False,matmul_precision='highest'))
    seal('CRITIC_TRAINING_TESTS.receipt.json',[HERE/'CRITIC_TRAINING_TESTS.json',HERE/'test_critic_training.py',HERE/'train_critic.py'])
    print('GPU training/held-label perturbation tests passed for all three critics',flush=True)


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
