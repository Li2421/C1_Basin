"""Refit at the source-controller-CV step; never choose on a target."""
import argparse, hashlib, os, time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .db_transfer_data import OUT as DATA, ROOT, SCENES, read, write, sha
from .db_transfer_train import model, groups, SEEDS
from .source_controller_cv_v2 import OUT as CV, KINDS
OUT=CV/'final_models'


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    kind=KINDS[index//3];seed=SEEDS[index%3]
    selected=read(CV/'selection.json')['selection'][kind]['held'];steps=selected['step']
    dest=OUT/kind/f'seed{seed}';assert not (dest/'complete.json').exists()
    d=np.load(DATA/'training_arrays.npz');x=dict(np.load(DATA/'source_entities.npz'));rows=read(DATA/'pairs.json');g=groups(rows)
    si,e,c,s,f=[d[k] for k in ('state_index','eta','context','success','failure')]
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    initial=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    flat=traverse_util.flatten_dict(p);mask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in flat})
    def make_step(warm):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff,ww):
            def loss(p0):
                z=m.apply(p0,xx,ee,cc);return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))*ww)
            val,gr=jax.value_and_grad(loss)(pp)
            if warm:gr=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),gr,mask)
            u,oo=opt.update(gr,oo,pp)
            if warm:u=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),u,mask)
            return optax.apply_updates(pp,u),oo,val
        return step
    warm,step=make_step(True),make_step(False)
    gg=[(g[sc,pool,'train'],1/float((s+f)[g[sc,pool,'train']].mean())) for sc in SCENES for pool in ('historical_wide','intervention')]
    rng=np.random.default_rng(seed);order=hashlib.sha256();prefix=None;start=time.perf_counter()
    for it in range(1,steps+1):
        ii=np.concatenate([rng.choice(ii,48) for ii,_ in gg]);ww=np.concatenate([np.full(48,w,np.float32) for _,w in gg]);order.update(ii.tobytes())
        p,o,loss=(warm if it<=25 else step)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii]),jnp.asarray(s[ii]),jnp.asarray(f[ii]),jnp.asarray(ww))
        if it==100:prefix=order.hexdigest()
    dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoint.msgpack').write_bytes(serialization.to_bytes(p))
    write(dest/'complete.json',dict(kind=kind,seed=seed,steps=steps,initial_sha256=initial,batch_order_sha256=order.hexdigest(),first100_batch_sha256=prefix,seconds=time.perf_counter()-start,backend=jax.default_backend(),checkpoint_sha256=sha(dest/'checkpoint.msgpack'),selection_sha256=sha(CV/'selection.json'),normalization_sha256=sha(DATA/'normalization.json'),training_arrays_sha256=sha(DATA/'training_arrays.npz'),model_code_sha256=sha(ROOT/'db_transfer_train.py'),code_sha256=sha(__file__),new_rollouts=0,target_used=False))
    print(dict(kind=kind,seed=seed,steps=steps),flush=True)


def freeze():
    docs=[]
    for kind in KINDS:
        for seed in SEEDS:
            p=OUT/kind/f'seed{seed}';d=read(p/'complete.json')
            assert d['checkpoint_sha256']==sha(p/'checkpoint.msgpack') and d['code_sha256']==sha(__file__)
            docs.append({**d,'path':str(p)})
    for seed in SEEDS:assert len({d['first100_batch_sha256'] for d in docs if d['seed']==seed})==1
    assert not (OUT/'models_frozen.json').exists()
    write(OUT/'models_frozen.json',dict(models=docs,normalization_path=str(DATA/'normalization.json'),source_selection_sha256=sha(CV/'selection.json'),target_used=False,new_rollouts=0))
    print(dict(frozen_models=len(docs),new_rollouts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','freeze'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.action=='train' else freeze()
