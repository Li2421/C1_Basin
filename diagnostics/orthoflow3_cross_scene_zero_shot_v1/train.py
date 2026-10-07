"""Clean source-only pure-NLL critics; no target data imports."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import jax,jax.numpy as jnp,optax
jax.config.update('jax_default_matmul_precision','highest')
import flax.linen as nn
from flax import serialization
from scipy.special import expit
from scipy.stats import spearmanr
from .build_source import ROOT,OUT,SOURCES,load,dump,sha
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

class EtaOnly(nn.Module):
    @nn.compact
    def __call__(self,x,e):
        z=nn.silu(nn.Dense(32,name='eta_encoder')(e))
        z=nn.silu(nn.Dense(128,name='shared1')(z));z=nn.silu(nn.Dense(64,name='shared2')(z))
        return nn.Dense(1,name='out')(z)[...,0]

def model_for(kind):return rep.Critic() if kind=='shared' else EtaOnly()
def data():
    rows=pq.read_table(OUT/'source_pairs.parquet').to_pylist()
    x=dict(np.load(OUT/'source_entities.npz'));norm=load(OUT/'normalization.json')
    e=(np.asarray([r['eta'] for r in rows],np.float32)-np.asarray(norm['eta_center'],np.float32))/np.asarray(norm['eta_radius'],np.float32)
    return rows,x,e,np.asarray([r['q'] for r in rows],np.float32),np.asarray([r['state_index'] for r in rows])
def gather(x,ix):return {k:jnp.asarray(v[ix]) for k,v in x.items()}
def metrics(p,y):
    p=np.clip(p,1e-7,1-1e-7)
    return {'nll':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),'mae':float(np.mean(abs(p-y))),
            'brier':float(np.mean((p-y)**2)),'spearman':float(spearmanr(p,y).statistic) if np.std(y)>0 and np.std(p)>0 else None}

def train(kind,seed):
    assert load(OUT/'structural_tests.json')['passed']
    assert not (OUT/'target_predictions.json').exists(),'No training after target predictions'
    rows,x,e,y,si=data();m=model_for(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,[0]),jnp.zeros((1,3)))
    opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4));state=opt.init(p)
    @jax.jit
    def step(p,state,bx,be,by):
        def loss(par):return jnp.mean(optax.sigmoid_binary_cross_entropy(m.apply(par,bx,be),by))
        value,grad=jax.value_and_grad(loss)(p);up,state=opt.update(grad,state,p)
        return optax.apply_updates(p,up),state,value
    predict=jax.jit(lambda p,bx,be:jax.nn.sigmoid(m.apply(p,bx,be)))
    groups={sc:{sp:np.array([i for i,r in enumerate(rows) if r['scenario']==sc and r['split']==sp]) for sp in ('train','validation')} for sc in SOURCES}
    rng=np.random.default_rng(seed);history=[];best=(float('inf'),None,0);stale=0;t=time.time()
    for it in range(1,4001):
        ix=np.concatenate([rng.choice(groups[sc]['train'],96) for sc in SOURCES])
        p,state,loss=step(p,state,gather(x,si[ix]),jnp.asarray(e[ix]),jnp.asarray(y[ix]))
        if it%100:continue
        val={}
        for sc in SOURCES:
            ix=groups[sc]['validation'];pred=[]
            for start in range(0,len(ix),256):
                ii=ix[start:start+256];pred.extend(np.asarray(predict(p,gather(x,si[ii]),jnp.asarray(e[ii]))).tolist())
            val[sc]=metrics(np.asarray(pred),y[ix])
        score=float(np.mean([v['nll'] for v in val.values()]));history.append({'step':it,'train':float(loss),'validation':score,'by_scene':val})
        if score<best[0]-1e-5:best=(score,serialization.to_bytes(p),it);stale=0
        else:stale+=1
        if it%500==0:print(json.dumps({'kind':kind,'seed':seed,'step':it,'val':score,'seconds':time.time()-t}),flush=True)
        if stale>=10 and it>=1500:break
    d=OUT/kind/f'seed{seed}';d.mkdir(parents=True,exist_ok=True);(d/'checkpoint.msgpack').write_bytes(best[1])
    dump(f'{kind}/seed{seed}/history.json',history)
    result={'kind':kind,'seed':seed,'best_step':best[2],'validation_nll':best[0],'steps':it,'seconds':time.time()-t,
      'checkpoint':str(d/'checkpoint.msgpack'),'sha256':sha(d/'checkpoint.msgpack'),'backend':jax.default_backend(),
      'train_data_sha256':sha(OUT/'source_pairs.parquet'),'normalization_sha256':sha(OUT/'normalization.json'),'target_labels_used':False}
    dump(f'{kind}/seed{seed}/training.json',result);return result

def structural():
    states=load(OUT/'source_states.json');rows,x,e,y,si=data()
    model=rep.Critic();p=model.init(jax.random.PRNGKey(17),gather(x,[0]),jnp.zeros((1,3)))
    checks=[];cpu=jax.devices('cpu')[0];gpu=jax.devices()[0]
    for sc in SOURCES:
        for r in [r for r in states if r['scenario']==sc][:3]:
            scene=r['physical'];n=len(scene['positions']);perm=list(range(n-1,-1,-1))
            a=rep.batch([rep.entities(scene)]);b=rep.batch([rep.entities(rep.transform(scene,.7,(.3,-.8),perm,list(range(len(scene['obstacles'])-1,-1,-1))))])
            pp=jax.device_put(p,cpu);ea=jnp.array([[.2,-.1,.4]])
            ca=np.asarray(model.apply(pp,jax.device_put(a,cpu),jax.device_put(ea,cpu)))
            cb=np.asarray(model.apply(pp,jax.device_put(b,cpu),jax.device_put(ea,cpu)))
            ga=np.asarray(model.apply(jax.device_put(p,gpu),jax.device_put(a,gpu),jax.device_put(ea,gpu)))
            assert np.max(abs(ca-cb))<2e-5,(sc,'passive',ca,cb)
            assert np.max(abs(ca-ga))<2e-5,(sc,'cpu_gpu',ca,ga)
            checks.append({'scenario':sc,'state_uid':r['state_uid'],'passive_error':float(np.max(abs(ca-cb))),'cpu_device_error':float(np.max(abs(ca-ga)))})
    # Exact control semantics, including Toy N=2 and four-agent sources.
    from shared_control.basis_families import get_basis_family,ORTHOFLOW3_SCALE
    from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import basis_terms
    errors=[]
    for r in states[:8]+[s for s in states if s['scenario']=='toy_giveway'][:8]:
        s=r['physical'];p0=np.asarray(s['positions']);g=np.asarray(s['goals']);u=np.asarray(s['flow']);v=s['max_speed']
        aa=get_basis_family('orthoflow3').compute(p0,g,u,v).values;bb=basis_terms('P1-OrthoFlow3',p0,g,u,v,ORTHOFLOW3_SCALE)
        errors.append(max(float(np.max(abs(a-b))) for a,b in zip(aa,bb)))
    assert max(errors)<1e-12
    # Same item alone and mixed/padded: detects accidental entity-mask leakage.
    small=[rep.entities(next(r['physical'] for r in states if r['scenario']==s)) for s in SOURCES]
    mixed=np.asarray(model.apply(p,rep.batch(small),jnp.zeros((3,3))))
    alone=np.array([float(model.apply(p,rep.batch([s]),jnp.zeros((1,3)))[0]) for s in small])
    assert max(abs(mixed-alone))<2e-5
    result={'passed':True,'checks':checks,'device':str(gpu),'cpu_gpu_checked':gpu.platform=='gpu','mixed_padding_error':float(max(abs(mixed-alone))),'basis_equivalence_max_error':max(errors),'target_examples_used':0}
    dump('structural_tests.json',result);print(json.dumps(result));return result

def freeze():
    result={}
    for kind in ('shared','eta_only'):
        rr=[load(OUT/kind/f'seed{s}/training.json') for s in (17,23,41)]
        result[kind]={'runs':rr,'selected':min(rr,key=lambda x:(x['validation_nll'],x['seed']))}
    result.update(protocol_sha256=sha(OUT/'protocol.json'),source_manifest_sha256=sha(OUT/'source_data_manifest.json'),normalization_sha256=sha(OUT/'normalization.json'),target_labels_used=False)
    dump('models_frozen.json',result)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['structural','shared','eta_only','freeze']);ap.add_argument('--seed',type=int,default=17);a=ap.parse_args()
    if a.action=='structural':structural()
    elif a.action=='freeze':freeze()
    else:print(json.dumps(train(a.action,a.seed)))
