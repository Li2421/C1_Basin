"""Matched-label-budget causal intervention on environment/controller coverage.

DEVELOPMENT ONLY. Adding target labels is an identification experiment, never
reported as zero-shot LOSO. Frozen architecture, representation and optimizer.
Only the source/target location and alignment of 512 added outcomes differ.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from diagnostics.controller_q_diversity_v1 import train
from diagnostics.controller_q_diversity_v1.experiment import write,freeze,sha

HERE=Path(__file__).resolve().parent/'crossed_support'
OLD=HERE.parent.parent/'controller_q_diversity_v1'
ARMS=('source','more_source','crossed','crossed_wrong_controller','crossed_wrong_eta')
SCENE={'toy':'toy_give_way','ring':'ring_exchange'}

def prepare():
    HERE.mkdir(exist_ok=True)
    d=np.load(OLD/'data/source_augmented128/dataset.npz')
    final=np.load(OLD/'data/final/dataset.npz')
    # Rows may have equal length to eta by coincidence: use explicit row keys.
    rowkeys=[k for k in d.files if k!='eta']
    for fold,scene in SCENE.items():
        source=(d['split']=='train')&(d['scene']==scene)&(d['state_index']<64)&(d['controller']!='flow')
        more=(d['split']=='train')&(d['scene']==scene)&(d['state_index']>=64)&(d['state_index']<72)&np.isin(d['controller'],['analytic0','analytic1','analytic2','analytic3'])
        cross=(d['split']=='train')&(d['scene']!=scene)&(d['state_index']<8)&np.isin(d['controller'],['analytic0','analytic1','analytic2','analytic3'])
        assert source.sum()==512 and more.sum()==cross.sum()==32
        val=(d['split']=='validation')&(d['controller']!='flow')
        for arm in ARMS:
            keep=source|(more if arm=='more_source' else cross if arm.startswith('crossed') else False)|val
            out={k:d[k][keep].copy() for k in rowkeys};out['eta']=d['eta']
            out['split']=np.where(out['split']=='train','train','validation')
            if arm.startswith('crossed_wrong'):
                added=(out['split']=='train')&(out['scene']!=scene)
                for sid in np.unique(out['state_uid'][added]):
                    ii=np.flatnonzero(added&(out['state_uid']==sid))
                    ii=ii[np.argsort(out['controller'][ii])]
                    assert len(ii)==4
                    idx=int(out['state_index'][ii[0]])
                    for k in ('success','failure','unknown','seed_success','seed_unknown','seed_collision'):
                        if arm=='crossed_wrong_controller':out[k][ii]=out[k][np.roll(ii,1+idx%3)]
                        else:out[k][ii]=np.roll(out[k][ii],1+idx%15,axis=1)
            path=HERE/'data'/fold/arm/'dataset.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():np.savez_compressed(path,**out)
            freeze(path.parent/'manifest.json',dict(parent_sha256=sha(OLD/'data/source_augmented128/dataset.npz'),
                dataset_sha256=sha(path),added_source_states=8 if arm=='more_source' else 0,
                added_target_states=8 if arm.startswith('crossed') else 0,
                added_rollouts=0,source_train_states=64,source_train_controllers=8,
                additional_rows=int(more.sum()) if arm!='source' else 0,
                label_control=arm,not_LOSO=True))
    jobs=[dict(fold=f,arm=a,seed=s) for f in SCENE for a in ARMS for s in (17,23,41)]
    freeze(HERE/'jobs.json',jobs)
    freeze(HERE/'protocol.json',dict(hypothesis='Training lacks crossed scene x controller x eta response support, rather than merely too few physical states or unavailable optimizer fit.',
        arms=ARMS,source='64natural states x8analytic controllers x16eta Q1; no nativeFlow',
        addition='8natural states x4original controllers x16eta Q1. New-source and new-target additions matched in label count and weight.',
        controller_control='On each added target state, cyclically permute the four entire controller outcome tensors. Preserves state/eta marginal, destroys correct controller association.',
        eta_control='Cyclically permute eta outcomes independently by state, same permutation for its controllers. Preserves controller/state mean success.',
        network='Existing state_contact_bilinear, no representation or objective change.',
        optimization='Same6000steps,120cells perstep, AdamW8e-4,decay1e-4. Uniform valid source/additional cells. Fixed last-step checkpoint, no target checkpoint selection.',
        normalization='Original source-only TRAIN fitted in all five arms; target inputs do not refit normalization.',
        evaluation='All original16VAL states per scene with seen0..3/unseen4..5controllers; opened32-state confirmation with controllers6..7. All nowDEVELOPMENT.',
        seeds=[17,23,41],bootstrap_replicates=20000,bootstrap_unit='physical state',
        zero_new_rollouts=True,zero_shot_claim_for_crossed_arms=False,
        literature=['https://proceedings.mlr.press/v119/raileanu20a.html','https://www.jmlr.org/papers/v25/23-0802.html']))

def fit(index):
    job=json.loads((HERE/'jobs.json').read_text())[index];fold,arm,seed=job['fold'],job['arm'],job['seed']
    dest=HERE/'fits'/fold/arm/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    d=np.load(HERE/'data'/fold/arm/'dataset.npz');source=np.load(HERE/'data'/fold/'source'/'dataset.npz')
    norm=train.normalization(source,'state_contact_bilinear');write(dest/'normalization.json',norm)
    x,c,e=train.inputs(d,norm);s=jnp.asarray(d['success'],jnp.float32);f=jnp.asarray(d['failure'],jnp.float32)
    ti=np.flatnonzero(d['split']=='train');ii,jj=np.nonzero((d['success'][ti]+d['failure'][ti])>0);pool=np.stack((ti[ii],jj),-1)
    net=train.Critic('state_contact_bilinear');params=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
    optim=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(8e-4,weight_decay=1e-4));opt=optim.init(params)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            p,o=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):
                z=net.apply(pp,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
                return jnp.mean(s[ii,jj]*jax.nn.softplus(-z)+f[ii,jj]*jax.nn.softplus(z))
            value,g=jax.value_and_grad(loss)(p);update,o=optim.update(g,o,p)
            return (optax.apply_updates(p,update),o),value
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def pred(pp,xx,cc,ee):return jax.nn.sigmoid(net.apply(pp,xx,cc,ee))
    def predict(data):
        xx,cc,ee=train.inputs(data,norm);ii=np.repeat(np.arange(len(cc)),16);jj=np.tile(np.arange(16),len(cc));result=[]
        for start in range(0,len(ii),256):
            inds=ii[start:start+256];etas=jj[start:start+256]
            result.extend(np.asarray(pred(params,{k:v[inds] for k,v in xx.items()},cc[inds],ee[etas])))
        return np.asarray(result).reshape(len(cc),16)
    rng=np.random.default_rng(seed);hist=[];t0=time.monotonic()
    for step in range(200,6001,200):
        draws=pool[rng.integers(len(pool),size=(200,120))]
        (params,opt),loss=block(params,opt,jnp.asarray(draws));hist.append(dict(step=step,training_nll=float(jnp.mean(loss))))
        if step%1000==0:write(dest/'history.json',hist)
    (dest/'checkpoint.msgpack').write_bytes(serialization.to_bytes(params))
    final=np.load(OLD/'data/final/dataset.npz')
    np.savez_compressed(dest/'predictions.npz',source=predict(d),opened_confirmation=predict(final))
    write(dest/'history.json',hist)
    write(dest/'complete.json',dict(**job,steps=6000,checkpoint_rule='fixed final step; no selection',
        seconds=time.monotonic()-t0,dataset_sha256=sha(HERE/'data'/fold/arm/'dataset.npz'),
        checkpoint_sha256=sha(dest/'checkpoint.msgpack'),code_sha256=sha(__file__),not_LOSO=True))
    print(json.dumps(dict(**job,seconds=time.monotonic()-t0)),flush=True)

def review():
    rows=[];comparisons=[];boots={}
    final=np.load(OLD/'data/final/dataset.npz')
    for fold,source_scene in SCENE.items():
        allpred={a:{panel:[] for panel in ('source','opened_confirmation')} for a in ARMS}
        datasets={}
        for arm in ARMS:
            datasets[arm]=np.load(HERE/'data'/fold/arm/'dataset.npz')
            for seed in (17,23,41):
                path=HERE/'fits'/fold/arm/f'seed{seed}'
                assert (path/'complete.json').exists()
                p=np.load(path/'predictions.npz')
                for panel in allpred[arm]:allpred[arm][panel].append(p[panel])
        panels=[('source',sc,seen) for sc in SCENE.values() for seen in (True,False)]+[('opened_confirmation',sc,False) for sc in SCENE.values()]
        for panel,scene,seen in panels:
            errors={};cmse={};predstore={};labels=None;boot=None
            for arm in ARMS:
                d=final if panel=='opened_confirmation' else datasets[arm]
                mask=(d['scene']==scene)
                if panel=='source':mask&=(d['split']=='validation')&np.isin(d['controller'],[f'analytic{k}' for k in (range(4) if seen else (4,5))])
                ix=np.flatnonzero(mask);ix=np.asarray(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
                q=np.mean(allpred[arm][panel],0)[ix];ss=d['success'][ix];ff=d['failure'][ix];uu=d['unknown'][ix];known=ss+ff>0
                truth=ss/np.maximum(ss+ff,1);ids=np.unique(d['state_uid'][ix]);n=len(ids);nc=len(ix)//n
                q=q.reshape(n,nc,16);truth=truth.reshape(n,nc,16);known=known.reshape(n,nc,16)
                errors[arm]=np.sum((q-truth)**2*known,axis=(1,2))/np.maximum(known.sum((1,2)),1)
                aa,bb=np.triu_indices(nc,1);valid=known[:,aa]&known[:,bb]
                cmse[arm]=np.sum(((q[:,aa]-q[:,bb])-(truth[:,aa]-truth[:,bb]))**2*valid,axis=(1,2))/np.maximum(valid.sum((1,2)),1)
                predstore[arm]=q
                y=truth- truth.mean(-1,keepdims=True);yp=q-q.mean(-1,keepdims=True)
                rho=float(np.corrcoef(y[known],yp[known])[0,1]) if yp[known].std()>1e-12 and y[known].std()>1e-12 else None
                rows.append(dict(fold=fold,scene=scene,panel=panel,seen_controller=seen,arm=arm,N=n,
                    mse=float(errors[arm].mean()),controller_contrast_mse=float(cmse[arm].mean()),eta_centered_correlation=rho,
                    train_nll_final=float(np.mean([json.loads((HERE/'fits'/fold/arm/f'seed{s}'/'history.json').read_text())[-1]['training_nll'] for s in (17,23,41)]))))
            boot=np.random.default_rng(2026101102).integers(n,size=(20000,n))
            for a,b in [('source','more_source'),('more_source','crossed'),('crossed_wrong_controller','crossed'),('crossed_wrong_eta','crossed')]:
                for label,values in [('Q_MSE',errors),('controller_contrast_MSE',cmse)]:
                    diff=values[a]-values[b];dist=diff[boot].mean(1)
                    key=f'{fold}_{scene}_{panel}_{seen}_{a}_to_{b}_{label}';boots[key]=dist
                    comparisons.append(dict(fold=fold,scene=scene,panel=panel,seen_controller=seen,baseline=a,intervention=b,metric=label,N=n,improvement=float(diff.mean()),CI95=np.quantile(dist,[.025,.975]).tolist()))
    np.savez_compressed(HERE/'bootstrap.npz',**boots)
    write(HERE/'results.json',dict(rows=rows,comparisons=comparisons,not_LOSO=True,new_rollouts=0))
    print('Crossed-support causal comparison saved; no final transfer claim',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='fit':fit(a.index)
    else:globals()[a.action]()
