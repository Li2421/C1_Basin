"""Identical TT/FF fits; neither final TEST counts nor scores choose checkpoints."""
from __future__ import annotations
import argparse,hashlib,itertools,os,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .design import ROOT,MAIN,SCENES,read,write,freeze,sha
from .dataset import CHAINS
from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train import model
KINDS=('eta_only','full_context');SEEDS=(17,23,41)
RUNS=list(itertools.product(SCENES,CHAINS,KINDS,SEEDS))


def inputs():
    m=read(ROOT/'dataset_frozen.json')
    assert m['dataset_sha256']==sha(ROOT/'dataset.npz') and m['entities_sha256']==sha(ROOT/'entities.npz')
    assert m['model_code_sha256']==sha(MAIN/'diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_transfer_train.py')
    d=np.load(ROOT/'dataset.npz');x=dict(np.load(ROOT/'entities.npz'));return d,x,m


def folder(scene,chain,kind,seed):return ROOT/'models'/scene/chain/kind/f'seed{seed}'


def run(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    scene,chain,kind,seed=RUNS[index];ci=CHAINS.index(chain);d,x,m=inputs();dest=folder(scene,chain,kind,seed)
    assert not (dest/'complete.json').exists(),'Completed fit is frozen'
    tr=np.flatnonzero((d['scene']==scene)&(d['split']=='train'));va=np.flatnonzero((d['scene']==scene)&(d['split']=='validation'))
    pairs=np.array([(i,j) for i in tr for j in range(16) if d['success'][ci,i,j]+d['failure'][ci,i,j]>0])
    assert len(pairs)>0;vp=np.array([(i,j) for i in va for j in range(16)])
    net=model(kind);gather=lambda ii:{k:jnp.asarray(v[ii]) for k,v in x.items()}
    pp=net.init(jax.random.PRNGKey(seed),gather(tr[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    initial_sha=hashlib.sha256(serialization.to_bytes(pp)).hexdigest()
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));opt=optimizer.init(pp)
    warmmask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in traverse_util.flatten_dict(pp)})
    nmean=np.mean([d['success'][ci,i,j]+d['failure'][ci,i,j] for i,j in pairs])
    def make_step(warm):
        @jax.jit
        def step(p,o,xx,ee,cc,ss,ff):
            def loss(q):
                z=net.apply(q,xx,ee,cc);return jnp.mean(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/nmean
            val,grad=jax.value_and_grad(loss)(p)
            if warm:grad=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),grad,warmmask)
            upd,o=optimizer.update(grad,o,p)
            if warm:upd=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),upd,warmmask)
            return optax.apply_updates(p,upd),o,val
        return step
    step,warm=make_step(False),make_step(True)
    predict=jax.jit(lambda p,xx,ee,cc:net.apply(p,xx,ee,cc))
    def evaluate(params):
        ii,jj=vp.T;z=np.asarray(predict(params,gather(ii),jnp.asarray(d['eta'][ii,jj]),jnp.asarray(d['context'][ci,ii,jj])))
        ss=d['success'][ci,ii,jj];ff=d['failure'][ci,ii,jj]
        nll=float(np.sum(ss*np.logaddexp(0,-z)+ff*np.logaddexp(0,z))/np.sum(ss+ff))
        return nll,z
    rng=np.random.default_rng(seed);order=hashlib.sha256();history=[];dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    for it in range(1,2001):
        draw=rng.integers(len(pairs),size=128);order.update(draw.tobytes());ii,jj=pairs[draw].T
        pp,opt,loss=(warm if it<=25 else step)(pp,opt,gather(ii),jnp.asarray(d['eta'][ii,jj]),jnp.asarray(d['context'][ci,ii,jj]),
            jnp.asarray(d['success'][ci,ii,jj]),jnp.asarray(d['failure'][ci,ii,jj]))
        if it%100:continue
        nll,z=evaluate(pp);(dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(pp))
        np.savez_compressed(dest/f'val_step{it}.npz',indices=vp,logits=z)
        history.append(dict(step=it,VAL_NLL=nll,train_loss=float(loss)))
    write(dest/'history.json',history)
    write(dest/'complete.json',dict(scene=scene,chain=chain,kind=kind,seed=seed,steps=2000,batch_size=128,eval_every=100,
        seconds=time.perf_counter()-start,backend=jax.default_backend(),initial_sha256=initial_sha,batch_draw_sha256=order.hexdigest(),
        pair_order_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),train_pair_count=len(pairs),train_nmean=float(nmean),
        dataset_sha256=m['dataset_sha256'],train_code_sha256=sha(__file__),test_labels_used=False))
    print(dict(scene=scene,chain=chain,kind=kind,seed=seed,completed=True,seconds=time.perf_counter()-start),flush=True)


def freeze_models():
    d,x,m=inputs();runs=[]
    for scene,chain,kind in itertools.product(SCENES,CHAINS,KINDS):
        histories=[read(folder(scene,chain,kind,seed)/'history.json') for seed in SEEDS]
        scores=np.mean([[r['VAL_NLL'] for r in h] for h in histories],axis=0);best=int(np.argmin(scores));step=histories[0][best]['step']
        for seed,h in zip(SEEDS,histories):
            dest=folder(scene,chain,kind,seed);doc=read(dest/'complete.json')
            assert doc['train_code_sha256']==sha(__file__) and doc['dataset_sha256']==m['dataset_sha256']
            path=dest/f'step{step}.msgpack'
            runs.append(dict(**doc,selected_step=step,VAL_NLL=h[best]['VAL_NLL'],checkpoint=str(path),checkpoint_sha256=sha(path)))
    for scene,kind,seed in itertools.product(SCENES,KINDS,SEEDS):
        rr=[r for r in runs if (r['scene'],r['kind'],r['seed'])==(scene,kind,seed)]
        for key in ('initial_sha256','batch_draw_sha256','pair_order_sha256','train_pair_count','train_nmean'):assert len({r[key] for r in rr})==1,(scene,kind,seed,key)
    freeze(ROOT/'models_frozen.json',dict(runs=runs,selection='Same mean three-seed source VAL observed-count NLL rule, no TEST outcomes',
        dataset_sha256=m['dataset_sha256'],normalization_sha256=m['normalization_sha256'],train_code_sha256=sha(__file__),
        same_TT_FF_initialization_and_minibatch_draws=True,shared_hyperparameters=True,new_task_rollouts=0))
    print(dict(models_frozen=len(runs)),flush=True)


def predict_test():
    import jax,jax.numpy as jnp
    from flax import serialization
    d,x,m=inputs();frozen=read(ROOT/'models_frozen.json');out={};rows=[]
    p=read(ROOT/'protocol.json')
    for run0 in frozen['runs']:
        scene,chain,kind,seed=[run0[k] for k in ('scene','chain','kind','seed')];ci=CHAINS.index(chain)
        idx=np.flatnonzero((d['scene']==scene)&(d['split']=='test'));va=np.flatnonzero((d['scene']==scene)&(d['split']=='validation'))
        pp=serialization.msgpack_restore(__import__('pathlib').Path(run0['checkpoint']).read_bytes());net=model(kind)
        assert sha(run0['checkpoint'])==run0['checkpoint_sha256']
        func=jax.jit(lambda xx,e,c:net.apply(pp,xx,e,c))
        key=f'{scene}__{chain}__{kind}__{seed}'
        for split,indices in (('test',idx),('validation',va)):
            # A predetermined cyclic derangement for each scene/split. Same eta,
            # same chain, other state; no model/outcome dependence.
            perm=sorted(indices,key=lambda i:hashlib.sha256(('tt_ff_shuffle_v1|'+p['states'][i]['state_uid']).encode()).hexdigest())
            replace=dict(zip(perm,np.roll(perm,-1)));other=np.array([replace[i] for i in indices])
            for condition in ('correct','state_shuffle','context_shuffle','state_context_shuffle','eta_shuffle'):
                si=other if condition in ('state_shuffle','state_context_shuffle') else indices
                cs=other if condition in ('context_shuffle','state_context_shuffle') else indices
                ej=np.roll(np.arange(16),1) if condition=='eta_shuffle' else np.arange(16)
                ii=np.repeat(si,16);eta=d['eta'][indices][:,ej].reshape(-1,3);context=d['context'][ci,cs].reshape(-1,76)
                pred=np.asarray(func({k:jnp.asarray(v[ii]) for k,v in x.items()},jnp.asarray(eta),jnp.asarray(context))).reshape(len(indices),16)
                out[f'{key}__{split}__{condition}']=pred
            out[f'{key}__{split}__indices']=indices
        rows.append(dict(key=key,checkpoint_sha256=run0['checkpoint_sha256'],scene=scene,chain=chain,kind=kind,seed=seed))
    global_rules=[]
    for scene,chain in itertools.product(SCENES,CHAINS):
        ci=CHAINS.index(chain);tr=np.flatnonzero((d['scene']==scene)&(d['split']=='train'))
        ss=d['success'][ci,tr];ff=d['failure'][ci,tr]
        # A frozen non-neural global-eta control checks that a weak neural
        # eta-only fit cannot masquerade as state-conditioning benefit.
        q=np.divide(ss,ss+ff,out=np.full_like(ss,np.nan),where=(ss+ff)>0)
        prior=np.nanmean(q,0);assert np.isfinite(prior).all()
        z=np.log(np.clip(prior,1e-6,1-1e-6)/np.clip(1-prior,1e-6,1-1e-6))
        for split in ('test','validation'):
            idx=np.flatnonzero((d['scene']==scene)&(d['split']==split));key=f'{scene}__{chain}__global_train_eta__0__{split}'
            out[key+'__correct']=np.broadcast_to(z,(len(idx),16));out[key+'__indices']=idx
        global_rules.append(dict(scene=scene,chain=chain,TRAIN_Q_prior=prior.tolist(),selected_eta_index=int(np.argmax(z)),test_labels_used=False))
    np.savez_compressed(ROOT/'frozen_predictions.npz',**out)
    freeze(ROOT/'test_predictions_frozen.json',dict(runs=rows,predictions_sha256=sha(ROOT/'frozen_predictions.npz'),
        model_freeze_sha256=sha(ROOT/'models_frozen.json'),test_outcomes_read=False,global_eta_controls=global_rules,
        shuffle_rule='hash-ordered cyclic state derangement, per scene/split; eta roll1'))
    print(dict(test_predictions_frozen=True,models=len(rows),test_outcomes_read=False),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('train','freeze','predict'));a.add_argument('--index',type=int);q=a.parse_args()
    run(q.index) if q.action=='train' else freeze_models() if q.action=='freeze' else predict_test()
