"""Source-only causal coverage control; one fixed architecture and optimizer."""
import argparse,time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax
from diagnostics.controller_q_diversity_v1.train import Critic
from .study import HERE,SCENES,read,write,freeze,sha

FOLDS={'joint':None,'held_toy':'toy_give_way','held_db':'double_bottleneck',
       'held_four':'four_way_intersection','held_ring':'ring_exchange'}
CONDITIONS=('matrix_only','native_eta_prior','native_actual')
JOBS=[dict(fold=f,condition=c,seed=s) for f in FOLDS for c in CONDITIONS for s in (17,23,41)]


def metrics(q,d,ix,z):
    s,f,u=(d[k][ix] for k in ('success','failure','unknown'));valid=s+f>0;y=s/np.maximum(s+f,1)
    matrix=d['kind'][ix]=='matrix';mse={}
    for name,keep in [('matrix',matrix),('native',~matrix)]:
        g=valid&keep[:,None];mse[name]=float(np.mean(((q-y)**2)[g]))
    contrasts=[]
    for sid in np.unique(d['state_uid'][ix][matrix]):
        ii=np.flatnonzero((d['state_uid'][ix]==sid)&matrix);assert len(ii)==5
        a,b=np.triu_indices(5,1);a=ii[a];b=ii[b];g=valid[a]&valid[b]
        contrasts.extend((((q[a]-q[b])-(y[a]-y[b]))**2)[g].tolist())
    cmse=float(np.mean(contrasts));eligible=s+f+u>0;lo=s/np.maximum(s+f+u,1)
    chosen=np.argmax(np.where(eligible,q,-np.inf),-1)
    regret=float(np.mean(np.max(np.where(eligible,lo,-np.inf),-1)-lo[np.arange(len(ix)),chosen]))
    nll=float(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/np.sum(s+f))
    return dict(score=.5*(mse['matrix']+mse['native'])+.5*cmse,matrix_Q_MSE=mse['matrix'],native_Q_MSE=mse['native'],
        matrix_controller_contrast_MSE=cmse,selection_regret=regret,nll=nll)


def fit(index):
    job=JOBS[index];p=read(HERE/'protocol.json');audit=read(HERE/'dataset_audit.json');assert sha(HERE/'dataset.npz')==audit['dataset_sha256']
    d=np.load(HERE/'dataset.npz');held=FOLDS[job['fold']]
    source=d['scene']!=held if held is not None else np.ones(len(d['scene']),bool)
    tr=(d['split']=='train')&source;va=np.flatnonzero((d['split']=='validation')&source)
    out=HERE/'fits'/job['fold']/job['condition']/f"seed{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    freeze(out/'job.json',dict(**job,dataset_sha256=audit['dataset_sha256'],held_labels_used=False))
    # All arms use the same SOURCE input normalization; no target statistics.
    mask=d['entity_agent_mask'].astype(bool);v=d['feature'][tr][mask[tr]]
    mean=v.mean(0);scale=np.maximum(v.std(0),.05)
    freeze(out/'normalization.json',dict(mean=mean.tolist(),scale=scale.tolist(),source_train_only=True,clip=10.))
    c=jnp.asarray(np.clip((d['feature']-mean)/scale,-10,10)*mask[...,None],jnp.float32)
    x={k[7:]:jnp.asarray(d[k],jnp.float32) for k in d.files if k.startswith('entity_')};e=jnp.asarray(d['eta'],jnp.float32)
    s=d['success'].astype(np.float32);f=d['failure'].astype(np.float32);eligible=s+f>0
    if job['condition']=='native_eta_prior':
        # Average probability over physical states, not over rollout seeds.
        # Exact eta matching preserves the old candidate-generation prior.
        coverage={}
        for sc in SCENES:
            ii=np.flatnonzero(tr&(d['scene']==sc)&(d['kind']!='matrix'))
            if not len(ii):continue
            groups={}
            for i in ii:
                for j in np.flatnonzero(eligible[i]):groups.setdefault(d['eta'][i,j].tobytes(),[]).append((i,j))
            coverage[sc]=[len(v) for v in groups.values()]
            for values in groups.values():
                rr,jj=np.asarray(values).T;n=s[rr,jj]+f[rr,jj];prior=np.mean(s[rr,jj]/n)
                s[rr,jj]=prior*n;f[rr,jj]=(1-prior)*n
        freeze(out/'prior_coverage.json',coverage)
    s,f=jnp.asarray(s),jnp.asarray(f)
    net=Critic('state_field_compact_bilinear');gather=lambda ii:{k:v[ii] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(job['seed']),gather(jnp.array([0])),c[:1],e[:1,0])
    optm=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(p['learning_rate'],weight_decay=p['weight_decay']));opt=optm.init(params)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            pp,oo=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(par):
                z=net.apply(par,gather(ii),c[ii],e[ii,jj]);ss=s[ii,jj];ff=f[ii,jj]
                return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1))
            loss0,grad=jax.value_and_grad(loss)(pp);updates,oo=optm.update(grad,oo,pp);pp=optax.apply_updates(pp,updates)
            return (pp,oo),loss0
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def predict(params,ii):
        return jax.vmap(lambda j:net.apply(params,gather(ii),c[ii],e[ii,j]),out_axes=1)(jnp.arange(16))
    def infer(par,ii):return np.concatenate([np.asarray(predict(par,jnp.asarray(ii[k:k+32]))) for k in range(0,len(ii),32)])
    pools={}
    for sc in SCENES:
        for name in np.unique(d['controller']):
            keep=tr&(d['scene']==sc)&(d['controller']==name)
            if job['condition']=='matrix_only':keep&=d['kind']=='matrix'
            rr,jj=np.where(eligible&keep[:,None])
            if len(rr):pools[sc,name]=np.column_stack((rr,jj))
    scenes=sorted({k[0] for k in pools});batch=p['batch_size']//len(scenes)
    rng=np.random.default_rng(job['seed']);history=[];best=None;t0=time.monotonic()
    for step in range(p['validation_every'],p['steps']+1,p['validation_every']):
        chunks=[]
        for sc in scenes:
            entries=[(name,pool) for (scene,name),pool in pools.items() if scene==sc]
            weights=np.array([.2 if name=='native_flow' else .8/5 for name,_ in entries]) if len(entries)==6 else np.ones(len(entries))/len(entries)
            which=rng.choice(len(entries),size=(p['validation_every'],batch),p=weights)
            draws=np.empty(which.shape+(2,),np.int32)
            for j,(_,pool) in enumerate(entries):
                selected=which==j;draws[selected]=pool[rng.integers(len(pool),size=selected.sum())]
            chunks.append(draws)
        (params,opt),loss=block(params,opt,jnp.asarray(np.concatenate(chunks,1)))
        z=infer(params,va).astype(np.float64);q=1/(1+np.exp(-np.clip(z,-100,100)))
        score=metrics(q,d,va,z);row=dict(step=step,training_nll=float(np.mean(loss)),**score);history.append(row)
        key=(score['score'],score['selection_regret'],score['nll'],step)
        if best is None or key<best:
            best=key;best_bytes=serialization.to_bytes(params);chosen=row
        if step%1000==0:
            write(out/'history.json',history)
            print(dict(**job,step=step,score=score['score'],best_step=chosen['step'],seconds=round(time.monotonic()-t0,1)),flush=True)
    (out/'checkpoint.msgpack').write_bytes(best_bytes)
    best_params=serialization.from_bytes(params,best_bytes)
    z=infer(best_params,np.arange(len(d['scene']))).astype(np.float64)
    np.savez_compressed(out/'predictions.npz',logits=z,probability=1/(1+np.exp(-np.clip(z,-100,100))))
    write(out/'history.json',history);write(out/'complete.json',dict(**job,selected=chosen,seconds=time.monotonic()-t0,source_states=len(np.unique(d['state_uid'][tr]))))


def review():
    from diagnostics.controller_operator_repair_v1.review import corr,center,gain_bounds
    d=np.load(HERE/'dataset.npz');rows=[];comparisons=[];saved={};boots={}
    for fold,held in FOLDS.items():
        pred={c:np.mean([np.load(HERE/'fits'/fold/c/f'seed{s}'/'predictions.npz')['probability'] for s in (17,23,41)],0) for c in CONDITIONS}
        for sc in SCENES:
            # Matrix panels have identical controllers/candidates and natural,
            # outcome-blind physical states; native panels are separately saved.
            ix=np.flatnonzero((d['scene']==sc)&(d['split']=='validation')&(d['kind']=='matrix'))
            ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//5;kk=[0,5,10,15];shape=(n,5,4)
            s,f,u=(d[k][ix][:,kk].reshape(shape) for k in ('success','failure','unknown'));assert np.all(s+f+u==4)
            lo=s/4;hi=(s+u)/4;y=s/np.maximum(s+f,1);valid=s+f>0
            pp={c:p[ix][:,kk].reshape(shape) for c,p in pred.items()};boot=np.random.default_rng(2026101201).integers(n,size=(20000,n))
            def record(name,base,model):
                delta=(((base-y)**2-(model-y)**2)*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1)
                lower,upper=gain_bounds(base,model,lo,hi);b=delta[boot].mean(1);bl=lower[boot].mean(1);bh=upper[boot].mean(1)
                label=f'{fold}_{sc}_{name}';boots[label]=b;boots[label+'_low']=bl;boots[label+'_high']=bh
                comparisons.append(dict(fold=fold,scene=sc,comparison=name,N=n,Q_MSE_improvement=float(delta.mean()),
                    CI95=np.quantile(b,[.025,.975]).tolist(),censor_robust_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))]))
            for condition,q in pp.items():
                a,b=np.triu_indices(5,1);good=valid[:,a]&valid[:,b]
                rows.append(dict(fold=fold,scene=sc,condition=condition,panel='held_scene_development' if held==sc else 'source_validation',N=n,
                    Q_MSE=float(np.mean(((q-y)**2)[valid])),controller_contrast_corr=corr((q[:,a]-q[:,b])[good],(y[:,a]-y[:,b])[good]),
                    state_effect_corr=corr(center(q,[0])[valid],center(y,[0])[valid]),eta_centered_corr=corr(center(q,[-1])[valid],center(y,[-1])[valid]),
                    selected_verified_success=float(np.take_along_axis(lo,q.argmax(-1)[...,None],-1).mean()),numerical_unknown=int(u.sum())))
                saved[f'{fold}_{sc}_{condition}']=q
            record('actual_vs_matrix_only',pp['matrix_only'],pp['native_actual'])
            record('actual_vs_native_eta_prior',pp['native_eta_prior'],pp['native_actual'])
            saved[f'{fold}_{sc}_success']=s;saved[f'{fold}_{sc}_unknown']=u
        np.savez_compressed(HERE/f'native_predictions_{fold}.npz',row_indices=np.flatnonzero(d['kind']!='matrix'),
            **{k:v[d['kind']!='matrix'] for k,v in pred.items()})
    np.savez_compressed(HERE/'development_predictions.npz',**saved);np.savez_compressed(HERE/'development_bootstrap.npz',**boots)
    write(HERE/'development_results.json',dict(rows=rows,comparisons=comparisons,new_rollouts=0,
        scope='DEVELOPMENT coverage intervention. Native labels preserve historical execution; no formal t0/K16/B15 confirmation claim.'))
    for r in rows:
        if r['panel']=='held_scene_development':print({k:r[k] for k in ('scene','condition','Q_MSE','controller_contrast_corr','state_effect_corr')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.stage=='fit':fit(a.index)
    else:review()
