"""Analytic encoding/ranking checks and actual source-isolated gradient test."""
from common import *
from critic_model import *
from train_critic import compiled
from select_models import summarize_inner,choose_sampled
from candidates import catalog,encode,draw
from flax import serialization
import jax,jax.numpy as jnp,copy
def main():
    rng=np.random.default_rng(12);types=np.asarray(['a','a','b','b','c','c']);names=np.asarray([f'c{i}' for i in range(6)]);proposals=rng.uniform([.5,-.5,0],[1.25,.5,.75],(48,8,3));encoded=np.asarray([[encode(a) for a in catalog(p)] for p in proposals],np.float32);indices=np.asarray([[draw(str(n),f's{s}') for s in range(48)] for n in names]);action=encoded[np.arange(48)[None,:,None],indices];labels=rng.integers(0,2,size=(6,48,26,1),dtype=np.int8);labels[0,0,0,0]=-1
    data=dict(names=names,types=types,features=rng.normal(size=(6,24)),h=rng.normal(size=(48,100)).astype(np.float32),c=rng.normal(size=(48,33)).astype(np.float32),action=action,labels=labels,candidate_indices=indices);source=['a','b'];tr=source_transform(data,source);other=copy.deepcopy(data);other['features'][types=='c']+=1e6;other['labels'][types=='c']=1-other['labels'][types=='c'];tr2=source_transform(other,source)
    for key in tr:assert np.array_equal(tr[key],tr2[key])
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,tr).items()};weights=sampling_weights(data,source);weights2=sampling_weights(other,source);assert np.array_equal(weights,weights2) and np.all(weights.reshape(6,48,26)[types=='c']==0)
    params=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[initialize('full',s) for s in SEEDS]);opt,block,predict=compiled('full');states=jax.vmap(opt.init)(params);ids=np.stack([rng.choice(len(weights),size=(3,16),p=weights) for _ in SEEDS],1).astype(np.int32)
    y,n,_=observations(data['labels']);y2,n2,_=observations(other['labels'])
    with jax.default_matmul_precision('highest'):
        (p1,_),_=block(params,states,jnp.asarray(ids),arrays,jnp.asarray(y.reshape(-1)),jnp.asarray(n.reshape(-1)));(p2,_),_=block(params,states,jnp.asarray(ids),arrays,jnp.asarray(y2.reshape(-1)),jnp.asarray(n2.reshape(-1)));assert all(np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(p1),jax.tree_util.tree_leaves(p2)))
        typed=copy.deepcopy(tr)
        for k in ('h_mean','h_scale','c_mean','c_scale'):typed[k]=np.asarray(json.loads(json.dumps(np.asarray(tr[k]).tolist())),np.float32)
        for k in ('feature_fill','feature_mean','feature_scale'):typed[k]=np.asarray(json.loads(json.dumps(np.asarray(tr[k]).tolist())),np.float64)
        restored=serialization.from_bytes(params,serialization.to_bytes(p1));pred=predict(p1,arrays);again=predict(restored,{k:jnp.asarray(v) for k,v in transform_arrays(data,typed).items()});assert np.array_equal(pred,again)
    histories={t:{s:dict(validation={t:dict(nll=2/s,ranking_accuracy=.7),'outer':dict(nll=-100,ranking_accuracy=1)}) for s in (100,200)} for t in source};selection=summarize_inner(source,histories,[100,200]);histories2=copy.deepcopy(histories)
    for h in histories2.values():
        for r in h.values():r['validation']['outer']['nll']=1e9
    assert selection==summarize_inner(source,histories2,[100,200]) and selection['selected']['step']==200
    test_labels=np.zeros((1,1,3,1),np.int8);test_labels[0,0,0,0]=1;test_p=np.asarray([[[[.9,.1,.2]]]]);test_indices=np.asarray([[[3,8,9]]]);m=metrics(test_p,test_labels,np.asarray(['a']),['a'],test_indices)['a'];assert m['ranking']['cross']['accuracy']==1 and m['ranking']['same_pair_different_time']['accuracy'] is None
    choice=choose_sampled(test_p,test_indices);assert choice['self_schedule'][0,0]==0 and choice['fixed500'][0,0]==0
    sharp=metrics(np.asarray([[[[.9999999,.9999998]]]],np.float32),np.asarray([[[[1],[0]]]],np.int8),np.asarray(['a']),['a'],np.asarray([[[3,8]]]))['a'];assert sharp['ranking_accuracy']==1.,'NLL clipping must not introduce false ranking ties'
    write(HERE/'CRITIC_TESTS.json',dict(utc=utc(),passed=True,source_feature_perturbation_invariant=True,source_gradient_outer_labels_bitwise_invariant=True,source_checkpoint_outer_metric_invariant=True,checkpoint_and_typed_transform_inference_bitwise=True,ranking_categories_analytic=True,unknown_excluded_from_loss=True,backend=jax.default_backend(),x64=False))
    seal('CRITIC_TESTS.receipt.json',[HERE/'CRITIC_TESTS.json',HERE/'test_critic.py',HERE/'train_critic.py',HERE/'critic_model.py',HERE/'select_models.py'])
    print('Critic source-isolation, inference and analytic ranking tests passed',flush=True)
if __name__=='__main__':main()
