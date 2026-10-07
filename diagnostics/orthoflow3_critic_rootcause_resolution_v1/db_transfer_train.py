"""One preregistered DB-held-out fold; source-only fitting and selection."""
import argparse,collections,hashlib,os,time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .db_transfer_data import OUT, ROOT, BASE, SCENES, read, write, sha

KINDS=('eta_only','no_context','additive_nominal_context','full_context','full_context_freeze25')
SEEDS=(17,23,41)
NOMINAL=[*range(10),*range(24,32),*[i for k in (40,56) for j in range(4) for i in (k+4*j,k+4*j+1)],72,73,74,75]

def model(kind):
    import jax.numpy as jnp
    import flax.linen as nn
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    class Critic(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            e=nn.silu(nn.Dense(32,name='eta_encoder')(eta))
            if kind=='eta_only':
                z=nn.silu(nn.Dense(128,name='eta1')(e));z=nn.silu(nn.Dense(64,name='eta2')(z))
                return nn.Dense(1,name='eta_out')(z)[...,0]
            if kind=='no_context':context=jnp.zeros_like(context)
            if kind=='additive_nominal_context':
                h=rep.Encoder(name='physical_encoder')(x)
                a=nn.silu(nn.Dense(128,name='a1')(jnp.concatenate((h,context[:,NOMINAL]),-1)))
                a=nn.silu(nn.Dense(64,name='a2')(a));a=nn.Dense(1,name='a_out')(a)[...,0]
                b=nn.silu(nn.Dense(128,name='b1')(e));b=nn.silu(nn.Dense(64,name='b2')(b))
                return a+nn.Dense(1,name='b_out')(b)[...,0]
            response=jnp.broadcast_to(context[:,None,24:40],(*x['agents'].shape[:2],16))
            xx={**x,'agents':jnp.concatenate((x['agents'],response),-1)}
            h=rep.Encoder(name='physical_encoder')(xx)
            context_global=jnp.concatenate((context[:,:24],context[:,40:]),-1)
            c=nn.silu(nn.Dense(32,name='context_encoder')(context_global))
            a=nn.Dense(128,name='trunk1')(jnp.concatenate((h,e,c),-1))
            a+=nn.Dense(128,use_bias=False,kernel_init=nn.initializers.zeros,name='raw_context_skip')(context)
            z=nn.silu(a);z=nn.silu(nn.Dense(64,name='trunk2')(z))
            return nn.Dense(1,name='out')(z)[...,0]
    return Critic()

def normalized():
    rows=read(OUT/'pairs.json');x=dict(np.load(OUT/'source_entities.npz'));cc=np.load(OUT/'source_context.npz')['context'];eta=np.array([r['eta'] for r in rows],np.float32)
    tr=np.array([r['split']=='train' for r in rows]);weights=np.zeros(len(rows))
    for sc in SCENES:
        idx=np.array([r['scenario']==sc for r in rows])&tr;weights[idx]=1/idx.sum()/len(SCENES)
    ec=(weights[:,None]*eta).sum(0);es=np.maximum(np.sqrt((weights[:,None]*(eta-ec)**2).sum(0)),.1)
    center=np.zeros(76);scale=np.ones(76)
    for a,b,flag,floor in ((0,24,73,.05),(24,40,73,.01),(40,56,74,.05),(56,72,75,.05)):
        w=weights*(cc[:,flag]>0);w/=w.sum();center[a:b]=(w[:,None]*cc[:,a:b]).sum(0)
        scale[a:b]=np.maximum(np.sqrt((w[:,None]*(cc[:,a:b]-center[a:b])**2).sum(0)),floor)
    c=(cc-center)/scale
    for a,b,flag in ((0,40,73),(40,56,74),(56,72,75)):c[:,a:b]*=cc[:,flag,None]>0
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=center.tolist(),context_scale=scale.tolist(),physical='fixed physical units, not target fitted',source_only=True)
    return rows,x,((eta-ec)/es).astype(np.float32),c.astype(np.float32),norm

def groups(rows):
    return {(sc,pool,split):np.array([i for i,r in enumerate(rows) if (r['scenario'],r['pool'],r['split'])==(sc,pool,split)]) for sc in SCENES for pool in ('historical_wide','intervention') for split in ('train','validation')}

def val_score(z,rows,idx):
    s=np.array([r['s'] for r in rows]);f=np.array([r['f'] for r in rows]);result=[]
    for sc in SCENES:
        for pool in ('historical_wide','intervention'):
            ii=np.flatnonzero([rows[i]['scenario']==sc and rows[i]['pool']==pool for i in idx])
            if len(ii):result.append(float((s[idx[ii]]*np.logaddexp(0,-z[ii])+f[idx[ii]]*np.logaddexp(0,z[ii])).sum()/(s[idx[ii]]+f[idx[ii]]).sum()))
    return float(np.mean(result)),result

def data_audit_materialize():
    from .db_transfer_context import materialize
    materialize();rows,x,e,c,norm=normalized();g=groups(rows)
    assert all(len(v)>0 for v in g.values())
    for sc in SCENES:
        ix=np.flatnonzero([r['scenario']==sc for r in rows])
        assert np.min(c[ix,73:76].mean(0))>.95,(sc,'Too many invalid physical measurements; diagnose before fitting')
    write(OUT/'normalization.json',norm)
    np.savez_compressed(OUT/'training_arrays.npz',eta=e,context=c,state_index=np.array([r['state_index'] for r in rows]),success=np.array([r['s'] for r in rows],np.float32),failure=np.array([r['f'] for r in rows],np.float32))
    # Define permutation controls without model predictions or outcomes.
    n=len(rows);controls={k:np.arange(n) for k in ('wrong_controller','joint_state_context_shuffle','eta_shuffle')}
    for name,keys in [('wrong_controller',('scenario','state_uid','eta_uid')),('joint_state_context_shuffle',('scenario','controller_uid','eta_uid')),('eta_shuffle',('scenario','state_uid','controller_uid'))]:
        d=collections.defaultdict(list)
        for i,r in enumerate(rows):
            if r['split']=='validation':d[tuple(r[k] for k in keys)].append(i)
        for indices in d.values():
            indices=sorted(indices,key=lambda i:hashlib.sha256((name+'|'+str(i)).encode()).hexdigest())
            for a,b in zip(indices,np.roll(indices,-1)):controls[name][a]=b
    np.savez_compressed(OUT/'source_controls.npz',**controls)
    write(OUT/'fit_manifest.json',dict(dataset_sha256=sha(OUT/'training_arrays.npz'),source_rows_sha256=sha(OUT/'pairs.json'),normalization_sha256=sha(OUT/'normalization.json'),context_code_sha256=sha(ROOT/'db_transfer_context.py'),train_code_sha256=sha(__file__),new_rollouts=0,target_used=False,groups={str(k):len(v) for k,v in g.items()}))
    print(dict(materialized=True,pairs=len(rows),new_rollouts=0),flush=True)

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    kind=KINDS[index//3];seed=SEEDS[index%3];dest=OUT/'models'/kind/f'seed{seed}'
    assert not (dest/'complete.json').exists(),'Completed fit is frozen'
    manifest=read(OUT/'fit_manifest.json');assert manifest['train_code_sha256']==sha(__file__)
    rows=read(OUT/'pairs.json');d=np.load(OUT/'training_arrays.npz');x=dict(np.load(OUT/'source_entities.npz'));g=groups(rows)
    si,e,c,s,f=[d[k] for k in ('state_index','eta','context','success','failure')]
    val=np.flatnonzero([r['split']=='validation' for r in rows]);model0=model(kind)
    pp=model0.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    init_sha=hashlib.sha256(serialization.to_bytes(pp)).hexdigest()
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));os0=opt.init(pp)
    flat=traverse_util.flatten_dict(pp);mask=traverse_util.unflatten_dict({k:k[-2] in ('trunk1','trunk2','out','raw_context_skip') for k in flat})
    warmmask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in flat})
    def make_step(freeze,warm=False):
        @jax.jit
        def step(p,o,xx,ee,cc,ss,ff,ww):
            def loss(p0):
                z=model0.apply(p0,xx,ee,cc);return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))*ww)
            v,gr=jax.value_and_grad(loss)(p)
            mm=warmmask if warm else mask
            if freeze or warm:gr=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),gr,mm)
            updates,o=opt.update(gr,o,p)
            if freeze or warm:updates=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),updates,mm)
            return optax.apply_updates(p,updates),o,v
        return step
    step,fixed,warm=make_step(False),make_step(True),make_step(False,True)
    predict=jax.jit(lambda p,xx,ee,cc:model0.apply(p,xx,ee,cc))
    def predictions(p,indices=val,condition='correct'):
        sm=si[indices];em=indices;cm=indices
        if condition!='correct':
            controls=np.load(OUT/'source_controls.npz')
            if condition=='state_shuffle':sm=si[controls['joint_state_context_shuffle'][indices]]
            elif condition=='joint_state_context_shuffle':cm=controls[condition][indices];sm=si[cm]
            elif condition=='wrong_controller':cm=controls[condition][indices]
            elif condition=='eta_shuffle':em=controls[condition][indices]
        arr=[]
        for start in range(0,len(indices),256):
            ss=slice(start,start+256);n=len(sm[ss]);pad=lambda a:np.pad(a,((0,256-len(a)),*((0,0) for _ in a.shape[1:])),mode='edge')
            xx={k:pad(v[sm[ss]]) for k,v in x.items()}
            arr.append(np.asarray(predict(p,xx,pad(e[em[ss]]),pad(c[cm[ss]])))[:n])
        return np.concatenate(arr)
    rng=np.random.default_rng(seed);order=hashlib.sha256();history=[];best=(np.inf,None,None,None);start=time.perf_counter()
    gg=[(g[sc,pool,'train'],1/float((s+f)[g[sc,pool,'train']].mean())) for sc in SCENES for pool in ('historical_wide','intervention')]
    for it in range(1,2001):
        draw=[rng.choice(ii,48) for ii,_ in gg];ii=np.concatenate(draw);ww=np.concatenate([np.full(48,w,np.float32) for _,w in gg]);order.update(ii.tobytes())
        apply=warm if it<=25 else fixed if kind=='full_context_freeze25' else step
        pp,os0,loss=apply(pp,os0,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii]),jnp.asarray(s[ii]),jnp.asarray(f[ii]),jnp.asarray(ww))
        if it%100:continue
        z=predictions(pp);score,detail=val_score(z,rows,val);history.append(dict(step=it,VAL_NLL=score,components=detail,train_loss=float(loss)))
        if score<best[0]:best=(score,serialization.to_bytes(pp),it,z)
    dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoint.msgpack').write_bytes(best[1]);chosen=serialization.msgpack_restore(best[1])
    zz={'correct':best[3],**{name:predictions(chosen,condition=name) for name in ('wrong_controller','state_shuffle','joint_state_context_shuffle','eta_shuffle')}}
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,**zz)
    write(dest/'history.json',history);write(dest/'complete.json',dict(kind=kind,seed=seed,steps=2000,best_step=best[2],source_VAL_NLL=best[0],seconds=time.perf_counter()-start,backend=jax.default_backend(),initial_sha256=init_sha,batch_order_sha256=order.hexdigest(),checkpoint_sha256=sha(dest/'checkpoint.msgpack'),dataset_sha256=manifest['dataset_sha256'],code_sha256=sha(__file__),target_labels_used=False,new_rollouts=0))
    print(dict(kind=kind,seed=seed,VAL_NLL=best[0],step=best[2],seconds=time.perf_counter()-start),flush=True)

def freeze():
    models=[]
    for kind in KINDS:
        for seed in SEEDS:
            p=OUT/'models'/kind/f'seed{seed}';r=read(p/'complete.json');assert r['code_sha256']==sha(__file__)
            assert r['checkpoint_sha256']==sha(p/'checkpoint.msgpack');models.append({**r,'path':str(p)})
    for seed in SEEDS:assert len({r['batch_order_sha256'] for r in models if r['seed']==seed})==1
    selected=min(('full_context','full_context_freeze25'),key=lambda k:np.mean([r['source_VAL_NLL'] for r in models if r['kind']==k]))
    assert not (OUT/'models_frozen.json').exists()
    write(OUT/'models_frozen.json',dict(models=models,source_selected_full=selected,criterion='Mean three-seed source VAL NLL; target remains unopened',normalization_sha256=sha(OUT/'normalization.json'),data_sha256=sha(OUT/'training_arrays.npz'),new_rollouts=0,target_labels_used=False))
    source_diagnostic()
    print(dict(models=len(models),selected=selected),flush=True)

def source_diagnostic():
    from scipy.special import expit
    rows=read(OUT/'pairs.json');docs=read(OUT/'models_frozen.json')['models'];result=[]
    for run in docs:
        d=np.load(__import__('pathlib').Path(run['path'])/'validation_predictions.npz');idx=d['indices'];correct=expit(d['correct'])
        group=collections.defaultdict(list)
        for loc,i in enumerate(idx):
            r=rows[i];group[r['scenario'],r['controller_uid'],r['state_uid']].append(loc)
        for condition in ('correct','wrong_controller','state_shuffle','joint_state_context_shuffle','eta_shuffle'):
            z=d[condition];prob=expit(z);score,parts=val_score(z,rows,idx);good=bad=unknown=eligible=changed=0
            for locs in group.values():
                if len(locs)<2:continue
                oracle=any(rows[idx[j]]['B15'] for j in locs)
                if not oracle:continue
                eligible+=1;j=locs[np.argmax(prob[locs])];old=locs[np.argmax(correct[locs])];r=rows[idx[j]]
                good+=bool(r['B15']);bad+=bool(r['nonB15']);unknown+=not(r['B15'] or r['nonB15']);changed+=j!=old
            result.append(dict(kind=run['kind'],seed=run['seed'],condition=condition,NLL=score,components=parts,oracle_eligible=eligible,selected_B15=good,nonB15=bad,unresolved=unknown,top1_changed=changed,mean_abs_probability_change=float(abs(prob-correct).mean())))
    write(OUT/'source_input_use.json',result)

def unit():
    import jax,jax.numpy as jnp
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    x=dict(np.load(OUT/'source_entities.npz'));xx=gather(x,np.arange(4));e=jnp.array([[0.,0.,0.],[.1,.4,.2],[-.2,.1,.3],[.7,-.1,.1]])
    c=jnp.arange(304,dtype=jnp.float32).reshape(4,76)/304;checks=[]
    for kind in KINDS:
        m=model(kind);p=m.init(jax.random.PRNGKey(17),xx,e,c);z=np.array(m.apply(p,xx,e,c))
        assert z.shape==(4,) and np.isfinite(z).all()
        if kind=='no_context':np.testing.assert_array_equal(z,np.array(m.apply(p,xx,e,c+100)))
        if kind=='eta_only':np.testing.assert_array_equal(z,np.array(m.apply(p,{k:jnp.zeros_like(v) for k,v in xx.items()},e,c+100)))
        if kind=='additive_nominal_context':
            # A genuine additive control cannot reverse eta preference across h.
            x0=gather(x,[0,1]);c0=c[:2];ee=jnp.broadcast_to(e[0],(2,3));ef=jnp.broadcast_to(e[1],(2,3))
            delta=np.array(m.apply(p,x0,ee,c0)-m.apply(p,x0,ef,c0));np.testing.assert_allclose(delta[0],delta[1],atol=1e-6)
            cc=c.at[:,[10,11,12,13,14,15,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38,39,42,43,46,47,50,51,54,55,58,59,62,63,66,67,70,71]].add(999)
            np.testing.assert_array_equal(z,np.array(m.apply(p,xx,e,cc)))
        checks.append(dict(kind=kind,passed=True))
    write(OUT/'model_unit_tests.json',checks);print(checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['materialize','train','freeze','unit','source_diagnostic']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.action=='train' else (data_audit_materialize() if a.action=='materialize' else globals()[a.action]())
