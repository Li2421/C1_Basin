"""Round16 A3 architecture and optimizer; five root-recipe CV folds."""
from core import *
import jax
import jax.numpy as jnp
import optax
from flax import serialization
sys.path.insert(0,str(R16))
import fit as oldfit

ARMS=oldfit.ARMS
MODELS=('C1','C3','C2','A0')
STEPS=list(range(500,12001,500))
SEEDS=(17,23,41)

def cohort_rows():
    train_labels_allowed()
    assert read(HERE/'train_DELIVERY_AUDIT.json')['status']=='PASS'
    assert not (HERE/'test/raw').exists() or not list((HERE/'test/raw').glob('*.json'))
    with np.load(HERE/'FEATURE_MATRIX.npz',allow_pickle=False) as z:
        names=list(z['controller_names']);z5=z['R5_PCA2'];z2=z['R5ETA2_PCA2']
    fmap={n:(z5[i],z2[i]) for i,n in enumerate(names)}
    rows=[]
    src=oldfit.source_data()
    assert len(src['names'])==56
    for i,n in enumerate(src['names']):
        rows.append(dict(name=n,recipe=str(src['recipes'][i]),physical=src['physical'],y=src['y'][i],n=src['n'][i],z5=fmap[n][0],z2=fmap[n][1],cohort='R15'))
    r17base=read(R17/'FINAL_BASE_PROFILES.json')
    with np.load(R17/'LABELS.npz',allow_pickle=False) as z:
        names17=list(z['controller_names']);lab=z['Y'].astype(np.int8)
    assert names17==r17base['eligible_controller_order'] and lab.shape==(17,48,8,4)
    physical17=oldfit.physical23(read(R17/'FRESH_STATES.json')).astype(np.float32)
    for i,n in enumerate(names17):
        profile=r17base['profiles'][n]
        recipe='base_u_v7_local_plus_drift_dataset' if profile['group']=='N1' else profile['recipe']
        rows.append(dict(name=n,recipe=recipe,physical=physical17,y=(lab[i]==1).sum(-1).astype(np.float32),
                         n=(lab[i]!=-1).sum(-1).astype(np.float32),z5=fmap[n][0],z2=fmap[n][1],cohort='R17'))
    base=read(HERE/'FINAL_BASE_PROFILES.json')
    with np.load(HERE/'train_LABELS.npz',allow_pickle=False) as z:
        namesnew=list(z['controller_names']);lab=z['Y'].astype(np.int8)
    assert namesnew==base['train_controller_order'] and lab.shape==(18,32,8,4)
    physicalnew=oldfit.physical23(read(HERE/'TRAIN-new_STATES.json')).astype(np.float32)
    for i,n in enumerate(namesnew):
        rows.append(dict(name=n,recipe=base['profiles'][n]['recipe'],physical=physicalnew,
                         y=(lab[i]==1).sum(-1).astype(np.float32),n=(lab[i]!=-1).sum(-1).astype(np.float32),
                         z5=fmap[n][0],z2=fmap[n][1],cohort='TRAIN-new'))
    assert len(rows)==91 and len({x['name'] for x in rows})==91
    return rows

def model_rows(rows,model):
    assert model in MODELS
    return [r for r in rows if model!='C3' or r['cohort']!='TRAIN-new']

def fold_roots(rows):
    roots=sorted({r['recipe'] for r in rows})
    assert len(roots)==15
    return [roots[j::5] for j in range(5)]

def state_transform(rows):
    raw=np.concatenate([r['physical'] for r in rows])
    mean=raw.mean(0);scale=np.where(raw.std(0)>1e-6,raw.std(0),1)
    return mean.astype(np.float32),scale.astype(np.float32)

def cells(rows,model,mean,scale):
    xs=[];ys=[];ns=[]
    eye=np.eye(8,dtype=np.float32)
    for r in rows:
        state=((r['physical']-mean)/scale).astype(np.float32)
        state=np.repeat(state[:,None,:],8,axis=1)
        action=np.broadcast_to(eye[None,:,:],(len(state),8,8))
        if model=='A0':x=np.concatenate([state,action],axis=-1)
        else:
            z=r['z5'] if model=='C1' else r['z2']
            z=np.broadcast_to(z[None,None,:],(len(state),8,2))
            x=np.concatenate([state,z,action],axis=-1)
        mask=r['n']>0
        xs.append(x[mask]);ys.append(r['y'][mask]);ns.append(r['n'][mask])
    X=np.concatenate(xs).astype(np.float32);Y=np.concatenate(ys).astype(np.float32);N=np.concatenate(ns).astype(np.float32)
    assert X.shape[1]==(31 if model=='A0' else 33) and np.all(N>0)
    return X,Y,N

def train_once(model,seed,training,validation,targets,out):
    mean,scale=state_transform(training)
    X,Y,N=cells(training,model,mean,scale)
    x=jnp.asarray(X);y=jnp.asarray(Y);n=jnp.asarray(N)
    if validation:
        XV,YV,NV=cells(validation,model,mean,scale)
        xv=jnp.asarray(XV);yv=jnp.asarray(YV);nv=jnp.asarray(NV)
    net=oldfit.net_for('A3')
    opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
    params=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,X.shape[1]),jnp.float32))
    state=opt.init(params)
    @jax.jit
    def block(p,s,ids):
        def update(carry,idx):
            pp,ss=carry
            def loss(q):
                logits=net.apply(q,x[idx])
                return jnp.mean((y[idx]*jax.nn.softplus(-logits)+(n[idx]-y[idx])*jax.nn.softplus(logits))/4)
            grads=jax.grad(loss)(pp);delta,ss=opt.update(grads,ss,pp)
            return (optax.apply_updates(pp,delta),ss),None
        return jax.lax.scan(update,(p,s),ids)[0]
    @jax.jit
    def nll_val(p):
        logits=net.apply(p,xv)
        return jnp.sum(yv*jax.nn.softplus(-logits)+(nv-yv)*jax.nn.softplus(logits))/jnp.sum(nv)
    rng=np.random.default_rng(seed);previous=0;history=[]
    for target in targets:
        ids=rng.integers(len(X),size=(target-previous,128),dtype=np.int32)
        params,state=block(params,state,jnp.asarray(ids));previous=target
        if validation:history.append(float(nll_val(params)))
    if out:
        out.mkdir(parents=True,exist_ok=True)
        checkpoint=out/f'seed{seed}_step{targets[-1]}.msgpack'
        checkpoint.write_bytes(serialization.to_bytes(params))
        np.savez_compressed(out/f'seed{seed}_transform.npz',state_mean=mean,state_scale=scale)
    return history

def main(model):
    assert model in MODELS
    assert not jax.config.x64_enabled
    assert any(str(device).startswith('cuda') for device in jax.devices()),jax.devices()
    rows=model_rows(cohort_rows(),model)
    folds=fold_roots(rows)
    histories=[];fold_records=[]
    for k,held in enumerate(folds):
        training=[r for r in rows if r['recipe'] not in held]
        validation=[r for r in rows if r['recipe'] in held]
        assert training and validation and set(r['recipe'] for r in training).isdisjoint(held)
        for seed in SEEDS:
            h=train_once(model,seed,training,validation,STEPS,None)
            histories.append(h);fold_records.append(dict(fold=k,held_roots=held,seed=seed,validation_nll=h,
                                                        train_controllers=len(training),held_controllers=len(validation)))
            print(json.dumps(dict(model=model,fold=k,seed=seed,last_nll=h[-1])),flush=True)
    average=np.mean(histories,axis=0);step=STEPS[int(np.argmin(average))]
    out=HERE/'critic'/model;out.mkdir(parents=True,exist_ok=True)
    write(out/'CV_SELECTION.json',dict(status='SELECTED_WITHOUT_TEST_LABELS',utc=utc(),model=model,
          controllers=len(rows),roots=sorted({r['recipe'] for r in rows}),folds=folds,
          candidate_steps=STEPS,mean_validation_nll=average.tolist(),selected_step=step,
          fold_seed_records=fold_records,selection_rule='minimum mean held-out known-label NLL; earliest step on tie'))
    checkpoints=[]
    for seed in SEEDS:
        train_once(model,seed,rows,None,[step],out)
        ck=out/f'seed{seed}_step{step}.msgpack';tr=out/f'seed{seed}_transform.npz'
        checkpoints.append(dict(seed=seed,checkpoint=str(ck.relative_to(HERE)),sha256=sha(ck),transform_sha256=sha(tr)))
        print(json.dumps(dict(model=model,full_fit_seed=seed,selected_step=step)),flush=True)
    write(out/'TRAINING_AUDIT.json',dict(status='FITTED_WITHOUT_TEST_LABELS',utc=utc(),model=model,
          source_controllers=len(rows),selected_step=step,cv_selection_sha256=sha(out/'CV_SELECTION.json'),
          checkpoints=checkpoints,training_labels_sha256=sha(HERE/'train_LABELS.npz'),
          round15_labels_sha256=sha(R15/'dataset.jsonl'),round17_labels_sha256=sha(R17/'LABELS.npz'),
          feature_matrix_sha256=sha(HERE/'FEATURE_MATRIX.npz'),test_labels_read=False))

if __name__=='__main__':main(sys.argv[1])
