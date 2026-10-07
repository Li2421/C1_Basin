"""Matched source data-size intervention; confirmation never opened here."""
import argparse,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import spearmanr
from .state_support import OUT,read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite,sha

KINDS=('eta_only','aggregate_context','entity_response_mean','entity_response_attached')
SIZES=('old46','expanded206');SEEDS=(17,23,41)
STEPS=(5,10,25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500)

def model(kind):
    import jax.numpy as jnp
    import flax.linen as nn
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    if kind=='native_state_controller':
        class NativeModel(nn.Module):
            @nn.compact
            def __call__(self,x,eta,context):
                h=nn.silu(nn.Dense(128,name='native1')(x['native']))
                h=nn.silu(nn.Dense(64,name='native2')(h))
                e=nn.silu(nn.Dense(32,name='eta')(eta))
                c=nn.silu(nn.Dense(16,name='known_controller')(context[:,88:90]))
                z=nn.silu(nn.Dense(128,name='trunk1')(jnp.concatenate([h,e,c],-1)))
                z=nn.silu(nn.Dense(64,name='trunk2')(z))
                return nn.Dense(1,name='out')(z)[...,0]
        return NativeModel()
    class Model(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            response=context[:,24:].reshape(-1,x['agents'].shape[1],16)
            if kind in ('eta_only','aggregate_context'):response=jnp.zeros_like(response)
            elif kind=='entity_response_mean':response=jnp.broadcast_to(response.mean(1,keepdims=True),response.shape)
            xx={**x,'agents':jnp.concatenate([x['agents'],response],axis=-1)}
            return Critic(kind!='eta_only',kind!='eta_only',False,name='core')(xx,eta,context[:,:24],jnp.zeros((len(eta),3)))
    return Model()

def load(size,kind=None):
    d=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(OUT/'model_entities.npz'))
    shared_valid=d['valid'].all(0)
    fit=np.flatnonzero((d['split']=='train')&shared_valid&((d['source']=='old46') if size=='old46' else True))
    val=np.flatnonzero(d['split']=='validation')
    assert shared_valid[val].all(),'No silently dropped invalid VAL response: implement declared fallback before evaluation'
    ec=d['eta'][fit].mean(0);es=np.maximum(d['eta'][fit].std(0),.1)
    cc=d['context'][:,fit].mean((0,1));cs=np.maximum(d['context'][:,fit].std((0,1)),.05)
    ac=d['agent_response'][:,fit].mean((0,1,2));asc=np.maximum(d['agent_response'][:,fit].std((0,1,2)),.01)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),
        agent_center=ac.tolist(),agent_scale=asc.tolist(),fit_pairs=len(fit),fit_states=len(set(d['state_index'][fit])))
    d['eta']=(d['eta']-ec)/es
    context=(d['context']-cc)/cs;response=(d['agent_response']-ac)/asc
    d['input_context']=np.concatenate([context,response.reshape(2,len(d['eta']),-1)],axis=-1).astype(np.float32)
    if kind=='native_state_controller':
        from ring_exchange.local_frame import local_observation
        from ring_exchange.environment import Config
        cfg=Config(**read(OUT.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
        states=read(OUT/'model_states.json')
        native=np.array([local_observation(*[np.array(s['physical'][k]) for k in ('positions','velocities','goals')],cfg).reshape(-1) for s in states])
        fs=np.unique(d['state_index'][fit]);nc=native[fs].mean(0);ns=np.maximum(native[fs].std(0),.05)
        x['native']=((native-nc)/ns).astype(np.float32);norm.update(native_center=nc.tolist(),native_scale=ns.tolist())
        cid=np.broadcast_to(np.eye(2,dtype=np.float32)[:,None,:],(2,len(d['eta']),2))
        d['input_context']=np.concatenate([d['input_context'],cid],axis=-1)
    return d,x,fit,val,norm

def metrics(z,d,indices):
    s=d['success'][:,indices];f=d['failure'][:,indices];n=s+f;q=s/n;p=expit(z)
    met=dict(NLL=float(np.mean((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum(1)/n.sum(1))),
        MAE=float(abs(p-q).mean()),Spearman=float(spearmanr(p.ravel(),q.ravel()).statistic),pairs=int(q.size))
    ss=d['standard_success'][:,indices];ff=d['standard_failure'][:,indices];rows=[]
    for c in range(2):
      for state in sorted(set(d['state_index'][indices])):
        pos=np.flatnonzero(d['state_index'][indices]==state)
        if len(pos)!=2:continue
        assert np.array_equal(d['eta_index'][indices[pos]],[10,15]);a,b=pos
        choice=int(pos[np.argmax(z[c,pos])]);good=ss[c,pos]>=15;bad=ff[c,pos]>=2
        low=ss[c,pos]/16;high=(16-ff[c,pos])/16
        strong=1 if low[0]-high[1]>=.25 else -1 if high[0]-low[1]<=-.25 else 0
        rows.append(dict(controller=c,state_index=int(state),eta_index=int(d['eta_index'][indices[choice]]),
            predicted=float(p[c,choice]),Q=float(q[c,choice]),oracle_Q=float(q[c,pos].max()),
            B15=bool(ss[c,choice]>=15),non_B15=bool(ff[c,choice]>=2),unresolved=bool(ss[c,choice]<15 and ff[c,choice]<2),
            oracle_B15=bool(good.any()),severe=bool(p[c,choice]>.9 and (16-ff[c,choice])/16<=.5),
            true_delta=float(q[c,a]-q[c,b]),pred_delta=float(p[c,a]-p[c,b]),strong_sign=strong,
            strong_correct=bool(np.sign(z[c,a]-z[c,b])==strong) if strong else None))
    met.update(cases=len(rows),B15=sum(r['B15'] for r in rows),unresolved=sum(r['unresolved'] for r in rows),
        oracle_B15=sum(r['oracle_B15'] for r in rows),selected_Q=float(np.mean([r['Q'] for r in rows])),
        regret=float(np.mean([r['oracle_Q']-r['Q'] for r in rows])),severe=sum(r['severe'] for r in rows),
        strong=sum(bool(r['strong_sign']) for r in rows),strong_correct=sum(r['strong_correct'] is True for r in rows))
    return met,rows

def train(index,pair_equal=False):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    from diagnostics.orthoflow3_controller_training_repair_v1.train import controller_balanced_nll
    assert jax.default_backend()=='gpu'
    if index>=24:size=SIZES[(index-24)//3];kind='native_state_controller';seed=SEEDS[(index-24)%3]
    else:size=SIZES[index//12];kind=KINDS[(index%12)//3];seed=SEEDS[index%3]
    run_size=size+'_pair_equal' if pair_equal else size
    dest=OUT/'models'/run_size/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    d,x,fit,val,norm=load(size,kind);si=d['state_index'];eta=d['eta'];context=d['input_context']
    m=model(kind);params=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,context.shape[-1])))
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));optim=optimizer.init(params)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            logits=m.apply(pp,xx,e,c).reshape(2,-1)
            if pair_equal:
                n=s+f;q=s/jnp.maximum(n,1)
                return jnp.mean(q*jax.nn.softplus(-logits)+(1-q)*jax.nn.softplus(logits))
            return controller_balanced_nll(logits,s,f)
        v,g=jax.value_and_grad(loss)(p);updates,o=optimizer.update(g,o,p);return optax.apply_updates(p,updates),o,v
    pred=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
    def predict(p,ii,condition='correct'):
        states=sorted(set(si[ii]));mp={s:states[(j+1)%len(states)] for j,s in enumerate(states)}
        lookup={(int(si[i]),int(d['eta_index'][i])):int(i) for i in ii}
        shifted=np.array([lookup[(mp[int(si[i])],int(d['eta_index'][i]))] for i in ii])
        ss=np.array([mp[int(s)] for s in si[ii]]) if condition in ('state_shuffle','joint_state_context_shuffle') else si[ii]
        cxidx=shifted if condition in ('context_state_shuffle','joint_state_context_shuffle') else ii
        scores=[]
        for c in range(2):
            cc=1-c if condition=='wrong_controller' else c;values=[]
            for start in range(0,len(ii),128):
                sl=slice(start,start+128)
                values.extend(np.asarray(pred(p,gather(x,ss[sl]),jnp.asarray(eta[ii[sl]]),jnp.asarray(context[cc,cxidx[sl]]))).tolist())
            scores.append(values)
        return np.array(scores)
    rng=np.random.default_rng(seed);best=(float('inf'),None,0);history=[]
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,2);ci=np.repeat(np.arange(2),32)
        params,optim,loss=step(params,optim,gather(x,si[ii]),jnp.asarray(eta[ii]),jnp.asarray(context[ci,ii]),
            jnp.asarray(d['success'][:,draw]),jnp.asarray(d['failure'][:,draw]))
        if it not in STEPS:continue
        fm,_=metrics(predict(params,fit),d,fit);vm,_=metrics(predict(params,val),d,val)
        history.append(dict(step=it,train=fm,validation=vm))
        if vm['NLL']<best[0]-1e-5:best=(vm['NLL'],serialization.to_bytes(params),it)
    dest.mkdir(parents=True,exist_ok=True);(dest/'best.msgpack').write_bytes(best[1]);(dest/'last.msgpack').write_bytes(serialization.to_bytes(params))
    pp=serialization.from_bytes(params,best[1]);rows=[];decisions=[];predictions={}
    for condition in ('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller'):
        z=predict(pp,val,condition);met,dec=metrics(z,d,val);predictions[condition]=z
        rows.append(dict(size=run_size,kind=kind,seed=seed,condition=condition,split='source_validation',**met))
        decisions.extend(dict(size=run_size,kind=kind,seed=seed,condition=condition,**r) for r in dec)
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,**predictions)
    csvwrite(dest/'metrics.csv',rows);csvwrite(dest/'decisions.csv',decisions)
    write(dest/'normalization.json',norm);write(dest/'history.json',history)
    write(dest/'complete.json',dict(size=run_size,kind=kind,seed=seed,best_step=best[2],best_source_VAL_NLL=best[0],
        weighting='pair_equal' if pair_equal else 'continuation_equivalent',
        protocol_hash=sha(OUT/'protocol.json'),dataset_hash=sha(OUT/'dataset.npz'),code_hash=sha(__file__),
        confirmation_labels_used=False,target_LOSO_labels_used=False,parameter_count=int(sum(a.size for a in jax.tree.leaves(params)))))
    print(json.dumps(rows[0]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--pair-equal',action='store_true');a=p.parse_args();train(a.index,a.pair_equal)
