"""Matched pure-NLL source supervision controls, never target-selected."""
import argparse
import collections
import hashlib
import itertools
import os
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from scipy.special import expit
from .db_transfer_data import ROOT, SCENES, read, write, sha
from .db_transfer_train import model, SEEDS
from .source_joint_support_inputs import OUT, ARMS

CONFIGS=[(arm,kind) for arm in ARMS for kind in ('eta_only','full_context')]+[('crossed_controller','additive_nominal_context')]
STEPS=tuple(range(100,2001,100))


def load():
    rows=read(OUT/'pairs.json');d=np.load(OUT/'arrays.npz');idx=np.load(OUT/'indices.npz')
    x=dict(np.load(OUT/'entities.npz'))
    return rows,d,idx,x


def nll(z,rows,ids):
    s=np.array([rows[i]['s'] for i in ids]);f=np.array([rows[i]['f'] for i in ids])
    return float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum())


def balance(z,rows,ids):
    parts=[]
    for sc in SCENES:
        for pool in ('historical_wide','intervention'):
            loc=np.array([j for j,i in enumerate(ids) if rows[i]['scenario']==sc and rows[i]['pool']==pool])
            if len(loc):parts.append(nll(z[loc],rows,ids[loc]))
    return float(np.mean(parts))


def measures(z,rows,ids,interaction=False):
    rr=[rows[i] for i in ids];q=np.array([r['s']/(r['s']+r['f']) for r in rr]);p=expit(z)
    groups=collections.defaultdict(list)
    for j,r in enumerate(rr):groups[r['controller_uid'],r['state_uid']].append(j)
    selected=[];eligible=0;top2=top3=0;reversals=collections.defaultdict(list)
    for (ctl,_),ll in groups.items():
        if len(ll)<2:continue
        good=any(rr[j]['B15'] for j in ll);eligible+=good
        order=sorted(ll,key=lambda j:-z[j]);selected.append(order[0])
        if good:
            top2+=any(rr[j]['B15'] for j in order[:2]);top3+=any(rr[j]['B15'] for j in order[:3])
        if interaction:
            for a,b in itertools.combinations(sorted(ll,key=lambda j:rr[j]['eta_uid']),2):
                direction=1 if rr[a]['B15'] and rr[b]['f16']>=8 else -1 if rr[b]['B15'] and rr[a]['f16']>=8 else 0
                reversals[ctl,rr[a]['eta_uid'],rr[b]['eta_uid']].append((q[a]-q[b],p[a]-p[b],direction))
    selected=np.array(selected,dtype=int)
    yes=sum(rr[j]['B15'] for j in selected);no=sum(rr[j]['nonB15'] for j in selected)
    result=dict(NLL=nll(z,rows,ids),MAE=float(abs(p-q).mean()),B15=int(yes),nonB15=int(no),
        unresolved=len(selected)-yes-no,cases=len(selected),oracle_B15=int(eligible),
        selected_Q=float(q[selected].mean()),selected_p=float(p[selected].mean()),
        severe_FP=int(sum(p[j]>.9 and rr[j]['f16']>=8 for j in selected)),
        top2_B15=top2/eligible if eligible else None,top3_B15=top3/eligible if eligible else None)
    if interaction:
        accuracy=[];centered=[]
        for vals in reversals.values():
            a=np.array(vals)
            if len(a)>=3:centered.extend(a[:,:2]-a[:,:2].mean(0))
            plus=a[:,2]==1;minus=a[:,2]==-1
            if plus.any() and minus.any():
                accuracy.append(.5*(np.mean((a[plus,1]>0)+.5*(a[plus,1]==0))+np.mean((a[minus,1]<0)+.5*(a[minus,1]==0))))
        a=np.array(centered)
        skill=float(1-np.mean((a[:,0]-a[:,1])**2)/np.mean(a[:,0]**2)) if len(a) and np.mean(a[:,0]**2)>1e-10 else None
        result.update(strong_reversal_eta_pairs=len(accuracy),reversal_balanced_accuracy=float(np.mean(accuracy)) if accuracy else None,centered_state_contrast_skill=skill)
    return result


def groups(rows,d,idx,arm):
    s,f=d['success'],d['failure'];result=[]
    for sc in SCENES:
        for pool in ('historical_wide','intervention'):
            ii=np.array([i for i in idx['fit'] if rows[i]['scenario']==sc and rows[i]['pool']==pool]);assert len(ii)
            if sc=='ring_exchange' and pool=='intervention':
                jj=idx[arm];result.extend([(ii,24,1/float((s+f)[ii].mean())),(jj,24,1/float((s+f)[jj].mean()))])
            else:result.append((ii,48,1/float((s+f)[ii].mean())))
    return result


def run(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    arm,kind=CONFIGS[index//3];seed=SEEDS[index%3]
    doc=read(OUT/'input_complete.json');assert doc['arrays_sha256']==sha(OUT/'arrays.npz')
    protocol=read(OUT/'protocol.json');assert protocol['model_code_sha256']==sha(ROOT/'db_transfer_train.py')
    rows,d,idx,x=load();si,e,c,s,f=[d[k] for k in ('state_index','eta','context','success','failure')]
    dest=OUT/'models'/arm/kind/f'seed{seed}';assert not (dest/'complete.json').exists(),'Frozen completed fit'
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    initial=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    mask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in traverse_util.flatten_dict(p)})
    def step_fn(warm):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff,ww):
            def loss(p0):
                z=m.apply(p0,xx,ee,cc);return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))*ww)
            v,gr=jax.value_and_grad(loss)(pp)
            if warm:gr=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),gr,mask)
            update,oo=opt.update(gr,oo,pp)
            if warm:update=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),update,mask)
            return optax.apply_updates(pp,update),oo,v
        return step
    warm,step=step_fn(True),step_fn(False)
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    def score(pp,ids):
        zz=[]
        for start in range(0,len(ids),256):
            ii=ids[start:start+256];n=len(ii);ii=np.pad(ii,(0,256-n),mode='edge')
            zz.extend(np.asarray(predict(pp,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii])))[:n])
        return np.array(zz)
    gg=groups(rows,d,idx,arm);rng=np.random.default_rng(seed);order=hashlib.sha256();uniform=hashlib.sha256();history=[];allpred={};start=time.perf_counter()
    dest.mkdir(parents=True,exist_ok=True)
    for it in range(1,2001):
        draws=[]
        for ids,num,w in gg:
            u=rng.random(num);uniform.update(u.tobytes());draws.append(ids[(u*len(ids)).astype(int)])
        ii=np.concatenate(draws);ww=np.concatenate([np.full(num,w,np.float32) for _,num,w in gg]);order.update(ii.tobytes())
        p,o,loss=(warm if it<=25 else step)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii]),jnp.asarray(s[ii]),jnp.asarray(f[ii]),jnp.asarray(ww))
        if it not in STEPS:continue
        zz={name:score(p,idx[name]) for name in ('seen','augheld','augseen')}
        metrics={name:measures(z,rows,idx[name]) for name,z in zz.items()}
        criterion=.5*(balance(zz['seen'],rows,idx['seen'])+metrics['augheld']['NLL'])
        history.append(dict(step=it,train_objective=float(loss),selection_NLL=criterion,**metrics))
        for name,z in zz.items():allpred[f'{name}_step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
    np.savez_compressed(dest/'predictions.npz',**allpred)
    write(dest/'history.json',history)
    write(dest/'complete.json',dict(arm=arm,kind=kind,seed=seed,initial_sha256=initial,uniform_order_sha256=uniform.hexdigest(),batch_order_sha256=order.hexdigest(),
       training_code_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),dataset_sha256=doc['arrays_sha256'],seconds=time.perf_counter()-start,
       backend=jax.default_backend(),new_rollouts=0,target_used=False))
    print(dict(arm=arm,kind=kind,seed=seed,seconds=time.perf_counter()-start),flush=True)


def freeze():
    rows,d,idx,x=load();models=[];results=[]
    for seed in SEEDS:
        docs=[read(OUT/'models'/a/k/f'seed{seed}'/'complete.json') for a,k in CONFIGS]
        assert len({r['uniform_order_sha256'] for r in docs})==1
        for kind in ('eta_only','full_context'):
            assert len({r['initial_sha256'] for r in docs if r['kind']==kind})==1
    for arm,kind in CONFIGS:
        runs=[read(OUT/'models'/arm/kind/f'seed{seed}'/'history.json') for seed in SEEDS]
        at=min(range(len(STEPS)),key=lambda i:np.mean([r[i]['selection_NLL'] for r in runs]));step=STEPS[at]
        for seed in SEEDS:
            dest=OUT/'models'/arm/kind/f'seed{seed}';complete=read(dest/'complete.json');assert complete['training_code_sha256']==sha(__file__)
            z=np.load(dest/'predictions.npz')
            mm={name:measures(z[f'{name}_step{step}'],rows,idx[name],True) for name in ('seen','augheld','augseen')}
            results.append(dict(arm=arm,kind=kind,seed=seed,step=step,**mm))
            models.append(dict(arm=arm,kind=kind,seed=seed,step=step,checkpoint=str(dest/f'step{step}.msgpack'),checkpoint_sha256=sha(dest/f'step{step}.msgpack')))
    assert not (OUT/'models_frozen.json').exists(),'Freeze is immutable'
    write(OUT/'models_frozen.json',dict(models=models,source_only=True,target_used=False,new_rollouts=0,
        normalization_sha256=sha(OUT/'normalization.json'),training_code_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json')))
    write(OUT/'source_results.json',results)
    print([{k:r[k] for k in ('arm','kind','seed','step','augheld')} for r in results],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','freeze']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    run(a.index) if a.action=='run' else freeze()
