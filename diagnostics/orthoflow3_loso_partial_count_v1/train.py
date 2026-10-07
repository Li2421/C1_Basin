"""Source-only shared/eta-only critics; likelihood and matched-weight control."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,time,json
import numpy as np
import pyarrow.parquet as pq
import jax,jax.numpy as jnp,optax
from flax import serialization
from scipy.special import gammaln,logsumexp
from .data import OUT,ROOT,FOLDS,SCENES,VARIANTS,load,sha,dump,baseline
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for,gather
jax.config.update('jax_default_matmul_precision','highest')
COUNTS=np.arange(15,dtype=np.float32)
LOGCOMB=(gammaln(17)-gammaln(COUNTS+1)-gammaln(17-COUNTS)).astype(np.float32)

def data(fold,variant):
    manifest=load(OUT/fold/variant/'dataset_manifest.json');sources=manifest['sources']
    rr=pq.read_table(OUT/'all_pairs.parquet',filters=[('scenario','in',sources)]).to_pylist()
    rows=[r for r in rr if (variant!='full_continuation' or r['old_full_member']) and (variant!='negative_binary' or r['old_full_member'] or r['non_b15_confirmed'])]
    assert len(rows)==len(manifest['row_indices']) and not any(r['scenario']==FOLDS[fold] for r in rows)
    x=dict(np.load(OUT/'entities.npz'));norm=load(OUT/fold/'normalization.json')
    e=(np.array([r['eta'] for r in rows],np.float32)-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
    s=np.array([r['s'] for r in rows],np.float32);f=np.array([r['f'] for r in rows],np.float32);n=s+f
    binary=np.array([variant=='negative_binary' and r['partial_record'] for r in rows])
    exposure=np.where(binary,1.,n)
    groups={sc:{sp:np.array([i for i,r in enumerate(rows) if r['scenario']==sc and r['split']==sp]) for sp in ('train','validation')} for sc in sources}
    mean={sc:float(exposure[g['train']].mean()) for sc,g in groups.items()}
    inv=np.array([1/mean[r['scenario']] for r in rows],np.float32)
    return rows,x,e,s,f,binary,inv,groups

def per_pair_loss(z,s,f,binary):
    lp=jax.nn.log_sigmoid(z);lf=jax.nn.log_sigmoid(-z)
    sequential=-s*lp-f*lf
    # Exact nonB15 event probability. Stable even when sigmoid rounds to one.
    negative=-jax.scipy.special.logsumexp(jnp.asarray(LOGCOMB)+jnp.asarray(COUNTS)*lp[:,None]+(16-jnp.asarray(COUNTS))*lf[:,None],axis=1)
    return jnp.where(binary,negative,sequential)

def train(fold,variant,kind,seed):
    d=OUT/fold/variant/kind/f'seed{seed}'
    if (d/'training.json').exists():return
    assert not (OUT/'target_predictions.json').exists(),'Frozen target evaluation already started'
    rows,x,e,s,f,binary,inv,groups=data(fold,variant);si=np.array([r['state_index'] for r in rows]);m=model_for(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,[si[0]]),jnp.zeros((1,3)));opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4));state=opt.init(p)
    @jax.jit
    def step(p,state,bx,be,bs,bf,bb,bw):
        def loss(par):return jnp.mean(per_pair_loss(m.apply(par,bx,be),bs,bf,bb)*bw)
        loss,g=jax.value_and_grad(loss)(p);u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,loss
    pred=jax.jit(lambda p,bx,be:m.apply(p,bx,be))
    rng=np.random.default_rng(seed);history=[];best=(float('inf'),None,0);stale=0;t=time.time()
    for it in range(1,4001):
        ix=np.concatenate([rng.choice(g['train'],96) for g in groups.values()])
        p,state,loss=step(p,state,gather(x,si[ix]),jnp.asarray(e[ix]),jnp.asarray(s[ix]),jnp.asarray(f[ix]),jnp.asarray(binary[ix]),jnp.asarray(inv[ix]))
        if it%100:continue
        val={}
        for sc,g in groups.items():
            ix=g['validation'];z=np.concatenate([np.asarray(pred(p,gather(x,si[jj]),jnp.asarray(e[jj]))) for jj in (ix[i:i+256] for i in range(0,len(ix),256))])
            ll=np.logaddexp(0,-z)*s[ix]+np.logaddexp(0,z)*f[ix]
            if variant=='negative_binary':
                lp=-np.logaddexp(0,-z);lf=-np.logaddexp(0,z);neg=-logsumexp(LOGCOMB+COUNTS*lp[:,None]+(16-COUNTS)*lf[:,None],axis=1);ll=np.where(binary[ix],neg,ll)
            denominator=np.where(binary[ix],1.,s[ix]+f[ix]).sum();value=float(ll.sum()/denominator)
            val[sc]={'observed_likelihood_per_exposure':value,'observed_pairs':len(ix),'exposure':float(denominator)}
        score=float(np.mean([r['observed_likelihood_per_exposure'] for r in val.values()]));history.append({'step':it,'train_objective':float(loss),'source_val':score,'scenes':val})
        if score<best[0]-1e-5:best=(score,serialization.to_bytes(p),it);stale=0
        else:stale+=1
        if it%500==0:print(json.dumps({'fold':fold,'variant':variant,'kind':kind,'seed':seed,'step':it,'source_val':score,'seconds':time.time()-t}),flush=True)
        if stale>=10 and it>=1500:break
    d.mkdir(parents=True,exist_ok=True);(d/'checkpoint.msgpack').write_bytes(best[1]);prefix=f'{fold}/{variant}/{kind}/seed{seed}'
    dump(prefix+'/history.json',history);dump(prefix+'/training.json',{'fold':fold,'variant':variant,'kind':kind,'seed':seed,'validation_nll':best[0],'best_step':best[2],'steps':it,'seconds':time.time()-t,'checkpoint':str(d/'checkpoint.msgpack'),'sha256':sha(d/'checkpoint.msgpack'),'backend':jax.default_backend(),'dataset_sha256':sha(OUT/fold/variant/'dataset_manifest.json'),'normalization_sha256':sha(OUT/fold/'normalization.json'),'target_scene':FOLDS[fold],'target_labels_used':False,'training_source_scenes':list(groups)})

def run(fold,variant):
    assert load(OUT/'likelihood_unit_tests.json')['passed']
    for kind in ('shared','eta_only'):
        for seed in (17,23,41):train(fold,variant,kind,seed)
    result={}
    for kind in ('shared','eta_only'):
        rr=[load(OUT/fold/variant/kind/f'seed{s}/training.json') for s in (17,23,41)];result[kind]={'runs':rr,'selected':min(rr,key=lambda r:(r['validation_nll'],r['seed']))}
    dump(f'{fold}/{variant}/models_frozen.json',{**result,'selection':'source-only mean scene likelihood','target_labels_used':False})
    dump('working_state.json',{'stage':'training','completed_fold':fold,'completed_variant':variant,'new_rollout':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',choices=FOLDS,required=True);ap.add_argument('--variant',choices=VARIANTS,required=True);a=ap.parse_args();run(a.fold,a.variant)
