"""Matched source-family crossfit of preprocessing/support repairs.

The previously opened 16-family panel and all LOSO targets are excluded.
This is a post-hoc source diagnostic, not an independent confirmation claim.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SRC=ROOT/'diagnostics/orthoflow3_source_contrast_interaction_v1'
WIDE=ROOT/'diagnostics/orthoflow3_controller_state_residual_factorial_v1'
KINDS=('eta_only','full_raw','full_state_scaled','full_scaled','wide_eta_only','wide_full_scaled',
       'wide_db_eta_only','wide_db_full_raw','wide_db_full_scaled')
SEEDS=(17,23,41)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('protocol frozen')
    old=read(SRC/'states.json');source=[r for r in old if r['split']=='train']
    assert len(source)==46 and len({r['source_group'] for r in source})==46
    order=sorted(range(46),key=lambda i:hashlib.sha256(('c1_state_context_audit_source_crossfit_v1\0'+source[i]['source_group']).encode()).hexdigest())
    folds=[]
    for fold in range(3):
        test=order[fold::3];rest=[s for s in order if s not in test]
        inner=rest[:6];fit=rest[6:]
        assert not(set(fit)&set(inner) or set(fit)&set(test) or set(inner)&set(test))
        folds.append(dict(fold=fold,fit=fit,inner=inner,test=test))
    d=np.load(SRC/'dataset.npz');wide=np.load(WIDE/'variants/large_20/dataset.npz')
    oldstates=read(WIDE/'states.json');omap={s['uid']:i for i,s in enumerate(oldstates)}
    indices=np.array([omap[s['uid']]*16+e for s in source for e in range(16)])
    entity=dict(np.load(SRC/'entities.npz'))
    oldentity=dict(np.load(WIDE/'entities.npz'))
    for key in entity:
        np.testing.assert_allclose(entity[key][:46],oldentity[key][[omap[s['uid']] for s in source]],atol=1e-6,rtol=0)
    arrays={k:wide[k][1:3,indices].copy() for k in ('success','failure','numerical','context','valid')}
    arrays.update(eta=wide['eta'][indices],eta_index=np.tile(np.arange(16),46),state_index=np.repeat(np.arange(46),16))
    upgraded=[]
    for i in np.flatnonzero(d['split']=='train'):
        j=int(d['state_index'][i])*16+int(d['eta_index'][i])
        np.testing.assert_allclose(arrays['eta'][j],d['eta'][i],rtol=0,atol=0)
        np.testing.assert_allclose(arrays['context'][:,j],d['context'][:,i],rtol=0,atol=1e-6)
        for key in ('success','failure','numerical','valid'):arrays[key][:,j]=d[key][:,i]
        upgraded.append(j)
    assert len(upgraded)==92 and arrays['valid'].all() and np.isfinite(arrays['context']).all()
    assert len(set((source[s]['uid'],tuple(e)) for s,e in zip(arrays['state_index'],arrays['eta'])))==736
    np.savez_compressed(OUT/'source_data.npz',**arrays)
    np.savez_compressed(OUT/'source_entities.npz',**{k:v[:46] for k,v in entity.items()})
    write(OUT/'source_states.json',source)
    write(OUT/'protocol.json',dict(
        kind='posthoc_source_only_diagnostic',new_rollout=0,
        source_families=46,controllers=['alt','second'],wide_eta_count=16,
        narrow_eta_indices=[10,15],folds=folds,seeds=SEEDS,variants=KINDS,
        fit_steps=1500,learning_rate=.0008,weight_decay=.0001,batch_pairs=32,
        checkpoint='source inner-family observed-count NLL only; first check step 5',
        evaluation_steps=[5,10,25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500],
        source_pair_upgrade='replace existing Q4 counts with available Q16 counts, never append duplicates',
        new_labels_used=False,opened_independent_panel_used=False,LOSO_target_labels_used=False,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [SRC/'dataset.npz',SRC/'entities.npz',WIDE/'variants/large_20/dataset.npz']},
        state_scaling='shared masked channels, FIT-only mean/std floor .05',
        context_scaling='FIT-only per-channel std; zero constant (<1e-6) channels; std floor 1e-4',
        diagnostic_limit='Eta pair originally selected using these source labels; crossfit is diagnostic, not fresh confirmation'))
    write(OUT/'dataset_manifest.json',dict(states=46,pairs_per_controller=736,
        unique_controller_state_eta_pairs=1472,upgraded_controller_pairs=184,
        observed_continuations=int((arrays['success']+arrays['failure']).sum()),
        numerical_excluded=int(arrays['numerical'].sum()),new_rollout=0,
        all_other_compatible_source_Q4_evidence_retained=True))
    print(read(OUT/'dataset_manifest.json'))

def load(kind,fold):
    from diagnostics.orthoflow3_controller_training_repair_v1.state_normalization import normalize_entities
    data_path=OUT/('source_data_db.npz' if kind.startswith('wide_db_') else 'source_data.npz')
    d=dict(np.load(data_path));x=dict(np.load(OUT/'source_entities.npz'))
    protocol=read(OUT/'protocol.json');sp=protocol['folds'][fold]
    active=np.ones(len(d['eta']),bool) if kind.startswith('wide_') else np.isin(d['eta_index'],[10,15])
    fi=np.flatnonzero(np.isin(d['state_index'],sp['fit'])&active)
    iv=np.flatnonzero(np.isin(d['state_index'],sp['inner'])&np.isin(d['eta_index'],[10,15]))
    te=np.flatnonzero(np.isin(d['state_index'],sp['test'])&np.isin(d['eta_index'],[10,15]))
    ec=d['eta'][fi].mean(0);es=np.maximum(d['eta'][fi].std(0),.1)
    cc=d['context'][:,fi].mean((0,1));cs=np.maximum(d['context'][:,fi].std((0,1)),.05)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist())
    if kind in ('full_state_scaled','full_scaled','wide_full_scaled','wide_db_full_scaled'):
        x,norm['state']=normalize_entities(x,sp['fit'])
    if kind in ('full_scaled','wide_full_scaled','wide_db_full_scaled'):
        std=d['context'][:,fi].astype(np.float64).std((0,1))
        cs=np.maximum(std,1e-4);constant=std<1e-6
        norm.update(context_scale=cs.tolist(),context_constant_mask=constant.tolist())
    d['eta']=((d['eta']-ec)/es).astype(np.float32)
    d['context']=((d['context']-cc)/cs).astype(np.float32)
    if 'context_constant_mask' in norm:d['context'][...,norm['context_constant_mask']]=0
    return d,x,fi,iv,te,norm

def metrics(logits,d,indices):
    from scipy.special import expit
    s=d['success'][:,indices];f=d['failure'][:,indices];n=s+f;q=s/n;p=expit(logits)
    out=dict(NLL=float(np.mean(np.sum(s*np.logaddexp(0,-logits)+f*np.logaddexp(0,logits),1)/n.sum(1))),
        MAE=float(abs(p-q).mean()),n_pairs=int(n.size))
    decisions=[]
    for c in range(2):
      for state in np.unique(d['state_index'][indices]):
        pos=np.flatnonzero(d['state_index'][indices]==state);ii=indices[pos]
        if not np.array_equal(d['eta_index'][ii],[10,15]):continue
        selected=int(pos[np.argmax(logits[c,pos])]);a,b=pos
        lower=s[c,pos]/16;upper=(16-f[c,pos])/16
        sign=1 if lower[0]-upper[1]>=.25 else -1 if upper[0]-lower[1]<=-.25 else 0
        decisions.append(dict(controller=c,state=int(state),selected_eta=int(d['eta_index'][indices[selected]]),
            selected_B15=bool(s[c,selected]>=15),oracle_B15=bool((s[c,pos]>=15).any()),
            selected_Q=float(q[c,selected]),oracle_Q=float(q[c,pos].max()),
            severe=bool(p[c,selected]>.9 and upper[np.where(pos==selected)[0][0]]<=.5),
            true_delta=float(q[c,a]-q[c,b]),pred_delta=float(p[c,a]-p[c,b]),
            strong_sign=sign,correct_strong=bool(np.sign(logits[c,a]-logits[c,b])==sign) if sign else None))
    out.update(cases=len(decisions),B15=sum(r['selected_B15'] for r in decisions),oracle_B15=sum(r['oracle_B15'] for r in decisions),
        selected_Q=float(np.mean([r['selected_Q'] for r in decisions])) if decisions else None,
        regret=float(np.mean([r['oracle_Q']-r['selected_Q'] for r in decisions])) if decisions else None,
        strong=sum(bool(r['strong_sign']) for r in decisions),
        strong_correct=sum(r['correct_strong'] is True for r in decisions),severe=sum(r['severe'] for r in decisions))
    return out,decisions

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    from diagnostics.orthoflow3_controller_training_repair_v1.train import controller_balanced_nll
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu','Submit GPU training through Slurm'
    kind=KINDS[index//9];fold=(index%9)//3;seed=SEEDS[index%3]
    dest=OUT/'models'/kind/f'fold{fold}_seed{seed}'
    if (dest/'complete.json').exists():return
    protocol=read(OUT/'protocol.json');d,x,fit,inner,test,norm=load(kind,fold)
    is_eta=kind.endswith('eta_only');model=Critic(not is_eta,not is_eta,False)
    si=d['state_index'];ee=d['eta'];cx=d['context']
    params=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));os=opt.init(params)
    @jax.jit
    def step(p,os,xx,e,c,s,f):
        def loss(pp):return controller_balanced_nll(model.apply(pp,xx,e,c,jnp.zeros((len(e),3))).reshape(2,-1),s,f)
        val,grad=jax.value_and_grad(loss)(p);updates,os=opt.update(grad,os,p)
        return optax.apply_updates(p,updates),os,val
    pred=jax.jit(lambda p,xx,e,c:model.apply(p,xx,e,c,jnp.zeros((len(e),3))))
    def predict(p,indices,condition='correct'):
        states=sorted(set(si[indices]));mapping={s:states[(j+1)%len(states)] for j,s in enumerate(states)}
        look={(int(si[i]),int(d['eta_index'][i])):i for i in indices}
        ci=np.asarray([look[(mapping[int(si[i])],int(d['eta_index'][i]))] for i in indices])
        out=[]
        for c in range(2):
            cc=1-c if condition=='wrong_controller' else c
            context_indices=ci if condition in ('context_state_shuffle','joint_state_context_shuffle') else indices
            ss=np.asarray([mapping[int(s)] for s in si[indices]]) if condition in ('state_shuffle','joint_state_context_shuffle') else si[indices]
            v=[]
            for start in range(0,len(indices),128):
                sl=slice(start,start+128)
                v.extend(np.asarray(pred(p,gather(x,ss[sl]),jnp.asarray(ee[indices[sl]]),jnp.asarray(cx[cc,context_indices[sl]]))).tolist())
            out.append(v)
        return np.asarray(out)
    rng=np.random.default_rng(seed);history=[];best=(float('inf'),None,0);best_legacy=(float('inf'),None,0)
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,2);ci=np.repeat(np.arange(2),32)
        params,os,loss=step(params,os,gather(x,si[ii]),jnp.asarray(ee[ii]),jnp.asarray(cx[ci,ii]),jnp.asarray(d['success'][:,draw]),jnp.asarray(d['failure'][:,draw]))
        if it not in protocol['evaluation_steps']:continue
        fz=predict(params,fit);vz=predict(params,inner)
        fm,_=metrics(fz,d,fit);vm,_=metrics(vz,d,inner)
        history.append(dict(step=it,train=fm,inner=vm))
        if vm['NLL']<best[0]-1e-5:best=(vm['NLL'],serialization.to_bytes(params),it)
        if it%100==0 and vm['NLL']<best_legacy[0]-1e-5:best_legacy=(vm['NLL'],serialization.to_bytes(params),it)
    final=serialization.to_bytes(params);dest.mkdir(parents=True,exist_ok=True)
    allmetrics=[];allrows=[];predictions={}
    for ck,blob in [('best',best[1]),('legacy_cadence',best_legacy[1]),('last',final)]:
        p=serialization.from_bytes(params,blob);(dest/f'{ck}.msgpack').write_bytes(blob)
        conditions=('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller') if ck=='best' else ('correct',)
        for condition in conditions:
            z=predict(p,test,condition);met,rows=metrics(z,d,test)
            allmetrics.append(dict(kind=kind,fold=fold,seed=seed,checkpoint=ck,condition=condition,**met))
            allrows.extend(dict(kind=kind,fold=fold,seed=seed,checkpoint=ck,condition=condition,**r) for r in rows)
            predictions[f'{ck}_{condition}']=z
    np.savez_compressed(dest/'predictions.npz',indices=test,**predictions)
    csvwrite(dest/'metrics.csv',allmetrics);csvwrite(dest/'decisions.csv',allrows)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(kind=kind,fold=fold,seed=seed,best_step=best[2],legacy_best_step=best_legacy[2],
        best_inner_NLL=best[0],test_used_for_selection=False,new_rollout=0,protocol_sha256=sha(OUT/'protocol.json'),
        training_code_sha256=sha(__file__),data_sha256=sha(OUT/('source_data_db.npz' if kind.startswith('wide_db_') else 'source_data.npz'))))
    print(json.dumps(allmetrics[0]),flush=True)

def summarize():
    rows=[];decisions=[];selection=[]
    for kind in KINDS:
      for fold in range(3):
       for seed in SEEDS:
        p=OUT/'models'/kind/f'fold{fold}_seed{seed}'
        selection.append(read(p/'complete.json'))
        rows.extend(list(csv.DictReader((p/'metrics.csv').open())))
        decisions.extend(list(csv.DictReader((p/'decisions.csv').open())))
    csvwrite(OUT/'crossfit_metrics.csv',rows);csvwrite(OUT/'crossfit_decisions.csv',decisions)
    output=[]
    for kind in KINDS:
      for seed in SEEDS:
       for ck in ('best','legacy_cadence','last'):
        conditions=('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller') if ck=='best' else ('correct',)
        for condition in conditions:
            r=[v for v in rows if v['kind']==kind and int(v['seed'])==seed and v['checkpoint']==ck and v['condition']==condition]
            assert len(r)==3
            weights=np.array([int(v['n_pairs']) for v in r]);cases=np.array([int(v['cases']) for v in r])
            result=dict(kind=kind,seed=seed,checkpoint=ck,condition=condition,
                NLL=float(np.average([float(v['NLL']) for v in r],weights=weights)),
                MAE=float(np.average([float(v['MAE']) for v in r],weights=weights)),
                B15=sum(int(v['B15']) for v in r),oracle_B15=sum(int(v['oracle_B15']) for v in r),cases=int(cases.sum()),
                regret=float(np.average([float(v['regret']) for v in r],weights=cases)),
                strong=sum(int(v['strong']) for v in r),strong_correct=sum(int(v['strong_correct']) for v in r))
            output.append(result)
    csvwrite(OUT/'pooled_metrics.csv',output)
    write(OUT/'checkpoint_selection.json',selection)
    print(json.dumps([r for r in output if r['checkpoint']=='best' and r['condition']=='correct'],indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('prepare','train','summarize'));a.add_argument('--index',type=int,default=0);z=a.parse_args()
    if z.action=='prepare':prepare()
    elif z.action=='train':train(z.index)
    else:summarize()
