"""Focused Ring FF state-only diagnosis. No controller-derived predictor input.

The physical controller that generated success labels stays fixed. Its ID,
weights, queried flow, short trajectory and goal response are all excluded.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
import json
import hashlib
import csv
import time
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=HERE.parent/'orthoflow3_tt_ff_matched_learnability_v1'
OUT=HERE/'state_only'
SEEDS=(17,23,41)
ARMS=('entity_state_eta','local_state_eta','local_state_16heads','local_state_16heads_shuffled')
STEPS=tuple(range(100,2001,100))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def prepare():
    from ring_exchange.local_frame import local_observation
    OUT.mkdir(exist_ok=True)
    p=read(SOURCE/'protocol.json');d=dict(np.load(SOURCE/'dataset.npz'))
    ix=np.flatnonzero(d['scene']=='ring_exchange')
    indices={split:np.flatnonzero((d['scene']=='ring_exchange')&(d['split']==split)) for split in ('train','validation','test')}
    cfg=SimpleNamespace(**p['environments']['ring_exchange']['config']['config'])
    local=np.zeros((len(d['scene']),92),np.float32)
    for i in ix:
        s=p['states'][int(i)]['physical']
        assert s['timestep']==0
        assert np.min(np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=1))>1e-5
        local[i]=local_observation(s['positions'],s['velocities'],s['goals'],cfg).ravel()
    tr=indices['train']
    center=local[tr].mean(0);scale=local[tr].std(0)
    active=scale>1e-6
    scale=np.maximum(scale,.01)
    normalized=(local-center)/scale
    normalized[:,~active]=0
    entities=dict(np.load(SOURCE/'entities.npz'))
    # agents = distance, velocity[2], nominal flow[2], bounded flow[2], ...
    entities['agents']=entities['agents'].copy();entities['agents'][...,3:7]=0
    assert np.all(entities['agents'][...,3:7]==0)
    # No response/context array is included in this predictor input artifact.
    np.savez_compressed(OUT/'inputs.npz',local=normalized,local_raw=local,eta=d['eta'],
                        local_center=center,local_scale=scale,local_active=active,**{'entity_'+k:v for k,v in entities.items()})
    protocol=dict(scope='Small single-cohort cached-data Ring FF mechanism diagnosis',
        fixed_label_controller=p['controllers']['ring_exchange']['FF']['controller_uid'],
        arms={
            'entity_state_eta':'Original full network with all76 context channels zero AND four nominal-flow agent channels zero',
            'local_state_eta':'64x64 SiLU MLP on TRAIN-standardized pure physical local observation92 + eta3',
            'local_state_16heads':'Same64x64 physical MLP,16 outputs tied to the fixed eta bank; no eta embedding',
            'local_state_16heads_shuffled':'Same16-output network, one whole-state TRAIN input permutation shared across all eta; physical-label pairing destroyed'},
        pure_state='positions, velocities, goals and fixed geometry; radial/tangential coordinates preserve native agent order; no identifiers',
        excluded=['controller ID','controller weights','nominal flow','bounded nominal flow','H20 response','goal-response fingerprints'],
        splits={k:v.tolist() for k,v in indices.items()},
        samples='32 independent TRAIN states x16 eta, original common-valid Q4;8 VAL and16 already-opened TEST states with Q16',
        seeds=list(SEEDS),steps=list(STEPS),batch_size=128,optimizer='AdamW lr0.0008 wd0.0001 clip5',
        selections=['Original mean3-seed VAL count NLL','Predefined mean3-seed VAL centered-by-eta probability MSE'],
        primary_metrics=['Same-eta cross-state Q-change sign accuracy for empirical gaps>=0.25',
                         'Correlation and R2 after removing each eta mean','State x eta interaction correlation',
                         'Two-state eta reversal accuracy'],
        negative_control='One fixed TRAIN state permutation, seed2026100721, never shuffles eta columns or held-out labels',
        reporting='All checkpoints and source-only selections; no TEST-selected winner; retrospective mechanism evidence',
        validation_limit='Only8 VAL families; comparing already-opened TEST cannot establish independent confirmation',
        new_rollouts=0,source_dataset_sha256=sha(SOURCE/'dataset.npz'),input_sha256=sha(OUT/'inputs.npz'),
        source_entity_sha256=sha(SOURCE/'entities.npz'),script_sha256=sha(__file__))
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')['input_sha256']==protocol['input_sha256']
    else:write(OUT/'protocol.json',protocol)
    print(json.dumps(dict(states=len(ix),input_active_dimensions=int(active.sum()),splits={k:len(v) for k,v in indices.items()},arms=list(ARMS))))


def state_effect(p,s,f):
    """Remove eta means on valid cells; never fit a calibration to these labels."""
    n=s+f;mask=n>0
    q=np.divide(s,n,out=np.zeros_like(s,dtype=float),where=mask)
    counts=mask.sum(0)
    qm=np.divide((q*mask).sum(0),counts,out=np.zeros(q.shape[1]),where=counts>0)
    pm=np.divide((p*mask).sum(0),counts,out=np.zeros(q.shape[1]),where=counts>0)
    qr=(q-qm)[mask];pr=(p-pm)[mask]
    mse=float(np.mean((pr-qr)**2));var=float(np.mean(qr**2))
    rho=float(np.corrcoef(qr,pr)[0,1]) if np.std(qr)>1e-9 and np.std(pr)>1e-9 else None
    return dict(state_effect_MSE=mse,state_effect_R2=1-mse/var if var else None,state_effect_rho=rho,
                true_state_effect_sd=float(np.std(qr)),predicted_state_effect_sd=float(np.std(pr)))


def model(arm):
    import flax.linen as nn
    import jax.numpy as jnp
    if arm=='entity_state_eta':
        from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train import model as source_model
        return source_model('full_context')
    class PhysicalMLP(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            v=jnp.concatenate((x,eta),-1) if arm=='local_state_eta' else x
            h=nn.silu(nn.Dense(64,name='dense1')(v));h=nn.silu(nn.Dense(64,name='dense2')(h))
            y=nn.Dense(1 if arm=='local_state_eta' else 16,name='out')(h)
            return y[...,0] if arm=='local_state_eta' else y
    return PhysicalMLP()


def fit():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization,traverse_util
    from scipy.special import expit
    p=read(OUT/'protocol.json');inputs=dict(np.load(OUT/'inputs.npz'));d=dict(np.load(SOURCE/'dataset.npz'))
    assert sha(SOURCE/'dataset.npz')==p['source_dataset_sha256'] and sha(OUT/'inputs.npz')==p['input_sha256']
    indices={k:np.array(v) for k,v in p['splits'].items()};tr,va=indices['train'],indices['validation']
    entity={k.removeprefix('entity_'):v for k,v in inputs.items() if k.startswith('entity_')}
    donor=np.arange(len(d['scene']));donor[tr]=np.random.default_rng(2026100721).permutation(tr)
    write(OUT/'negative_control_permutation.json',dict(train=tr.tolist(),donor=donor[tr].tolist()))
    pairs=np.array([(i,j) for i in tr for j in range(16) if d['success'][1,i,j]+d['failure'][1,i,j]>0])
    nmean=float(np.mean([d['success'][1,i,j]+d['failure'][1,i,j] for i,j in pairs]))
    originals=read(SOURCE/'models_frozen.json')['runs'];completed=[]
    for arm in ARMS:
        net=model(arm)
        head=arm.startswith('local_state_16heads')
        def gather(ii,jj,training=False):
            si=donor[ii] if training and arm.endswith('_shuffled') else ii
            xx={k:jnp.asarray(v[si]) for k,v in entity.items()} if arm=='entity_state_eta' else jnp.asarray(inputs['local'][si])
            return xx,jnp.asarray(inputs['eta'][ii,jj]),jnp.zeros((len(ii),76),jnp.float32)
        def predict(p,xx,ee,cc,jj):
            z=net.apply(p,xx,ee,cc)
            return jnp.take_along_axis(z,jj[:,None],1)[:,0] if head else z
        infer=jax.jit(predict)
        optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001))
        mask=None
        def make_step(warm):
            @jax.jit
            def train_step(pp,oo,xx,ee,cc,jj,ss,ff):
                def objective(params):
                    z=predict(params,xx,ee,cc,jj)
                    return jnp.mean(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/nmean
                loss,gr=jax.value_and_grad(objective)(pp)
                if warm:gr=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),gr,mask)
                up,oo=optimizer.update(gr,oo,pp)
                if warm:up=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),up,mask)
                return optax.apply_updates(pp,up),oo,loss
            return train_step
        step,warm=make_step(False),make_step(True)
        def predict_split(pp,split,training=False):
            ix=indices[split];ii,jj=np.repeat(ix,16),np.tile(np.arange(16),len(ix))
            z=[]
            for a in range(0,len(ii),128):
                sl=slice(a,a+128)
                z.append(np.asarray(infer(pp,*gather(ii[sl],jj[sl],training),jnp.asarray(jj[sl]))))
            return np.concatenate(z).reshape(len(ix),16)
        for seed in SEEDS:
            dest=OUT/'fits'/arm/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                completed.append(read(dest/'complete.json'));continue
            started=time.monotonic()
            # Dummy inputs match old shapes exactly for the original entity model.
            xx,ee,cc=gather(tr[:1],np.array([0]))
            pp=net.init(jax.random.PRNGKey(seed),xx,jnp.zeros_like(ee),cc)
            initial=hashlib.sha256(serialization.to_bytes(pp)).hexdigest()
            old=next(r for r in originals if (r['scene'],r['chain'],r['kind'],r['seed'])==('ring_exchange','FF','full_context',seed))
            if arm=='entity_state_eta':assert initial==old['initial_sha256']
            mask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in traverse_util.flatten_dict(pp)})
            oo=optimizer.init(pp);rng=np.random.default_rng(seed);drawhash=hashlib.sha256();history=[]
            for it in range(1,2001):
                draw=rng.integers(len(pairs),size=128);drawhash.update(draw.tobytes());ii,jj=pairs[draw].T
                pp,oo,loss=(warm if it<=25 and arm=='entity_state_eta' else step)(pp,oo,*gather(ii,jj,True),jnp.asarray(jj),
                   jnp.asarray(d['success'][1,ii,jj]),jnp.asarray(d['failure'][1,ii,jj]))
                if it%100:continue
                pred={sp:predict_split(pp,sp,training=(sp=='train')) for sp in ('train','validation')}
                if arm.endswith('_shuffled'):pred['train_correct_input']=predict_split(pp,'train',False)
                scores={}
                for sp in ('train','validation'):
                    s,f=d['success'][1,indices[sp]],d['failure'][1,indices[sp]];z=pred[sp]
                    scores[sp]=dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum()),**state_effect(expit(z),s,f))
                history.append(dict(step=it,**scores))
                (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(pp))
                np.savez_compressed(dest/f'step{it}_source.npz',**pred)
            assert drawhash.hexdigest()==old['batch_draw_sha256']
            write(dest/'history.json',history)
            done=dict(arm=arm,seed=seed,backend=jax.default_backend(),seconds=time.monotonic()-started,
                      initial_sha256=initial,batch_draw_sha256=drawhash.hexdigest(),pair_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),
                      parameter_count=sum(int(x.size) for x in jax.tree.leaves(pp)),protocol_sha256=sha(OUT/'protocol.json'),
                      training_script_sha256=sha(__file__),test_labels_read=False)
            write(dest/'complete.json',done);completed.append(done)
            print('complete',arm,seed,round(done['seconds'],1),'last',history[-1],flush=True)
    selection={}
    for arm in ARMS:
        hist=[read(OUT/'fits'/arm/f'seed{s}'/'history.json') for s in SEEDS]
        selection[arm]={}
        for key in ('NLL','state_effect_MSE'):
            vals=np.mean([[r['validation'][key] for r in hh] for hh in hist],0)
            selection[arm][key]=dict(step=STEPS[int(np.argmin(vals))],mean_VAL_by_step=dict(zip(map(str,STEPS),map(float,vals))))
    write(OUT/'selection.json',dict(arms=selection,selected_on='VAL only',test_labels_read=False))
    write(OUT/'training_complete.json',dict(runs=completed,new_rollouts=0,selection_sha256=sha(OUT/'selection.json')))


def export():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    p=read(OUT/'protocol.json');data=dict(np.load(OUT/'inputs.npz'))
    ix=np.array(p['splits']['test']);ii,jj=np.repeat(ix,16),np.tile(np.arange(16),len(ix))
    entities={k.removeprefix('entity_'):v for k,v in data.items() if k.startswith('entity_')}
    results={};ck=[]
    assert (OUT/'selection.json').exists()
    for arm in ARMS:
        net=model(arm);fun=jax.jit(net.apply);heads=arm.startswith('local_state_16heads')
        for seed in SEEDS:
            for step in STEPS:
                path=OUT/'fits'/arm/f'seed{seed}'/f'step{step}.msgpack';pp=serialization.msgpack_restore(path.read_bytes())
                z=[]
                for start in range(0,len(ii),128):
                    sl=slice(start,start+128);si,ej=ii[sl],jj[sl]
                    xx={k:jnp.asarray(v[si]) for k,v in entities.items()} if arm=='entity_state_eta' else jnp.asarray(data['local'][si])
                    value=np.asarray(fun(pp,xx,jnp.asarray(data['eta'][si,ej]),jnp.zeros((len(si),76),jnp.float32)))
                    if heads:value=value[np.arange(len(si)),ej]
                    z.append(value)
                results[f'{arm}__{seed}__{step}']=np.concatenate(z).reshape(len(ix),16)
                ck.append(dict(arm=arm,seed=seed,step=step,sha256=sha(path)))
            print('exported',arm,seed,flush=True)
    np.savez_compressed(OUT/'test_logits.npz',**results)
    write(OUT/'export_manifest.json',dict(checkpoints=ck,selection_sha256=sha(OUT/'selection.json'),script_sha256=sha(__file__),test_labels_read=False))


def metrics(z,s,f,u,k):
    from scipy.special import expit
    from diagnostics.state_nuisance_rootcause_v1.state_signal_path import assess,reversals
    pred=expit(np.asarray(z,float));m=assess(z,s,f,u,k,reversals(s,f,u,k))
    m.update(state_effect(pred,s,f))
    lo,hi=s/k,(s+u)/k
    i,j=np.triu_indices(len(s),1)
    signpos=lo[i]-hi[j]>=.25;signneg=lo[j]-hi[i]>=.25
    eligible=signpos|signneg;diff=z[i]-z[j]
    correct=(diff>0)*signpos+(diff<0)*signneg+.5*(diff==0)*eligible
    m.update(same_eta_state_change_cases=int(eligible.sum()),
             same_eta_state_change_accuracy=float(correct[eligible].mean()) if eligible.any() else None)
    return m


def analyze():
    from diagnostics.state_nuisance_rootcause_v1.analyze import labels
    p=read(OUT/'protocol.json');d=np.load(SOURCE/'dataset.npz');t=np.load(SOURCE/'test_truth.npz')
    y={sp:labels('ring_exchange','FF',sp,d,t) for sp in ('train','validation','test')}
    test=np.load(OUT/'test_logits.npz');old=np.load(HERE/'replayed_logits.npz');oldtraj=np.load(HERE/'checkpoint_overfitting/test_logits.npz')
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            for step in STEPS:
                source=np.load(OUT/'fits'/arm/f'seed{seed}'/f'step{step}_source.npz')
                for split in ('train','validation','test'):
                    z=test[f'{arm}__{seed}__{step}'] if split=='test' else source[split]
                    yy=y[split];k=4 if split=='train' else 16
                    m=metrics(z,yy['s'],yy['f'],yy['u'],k)
                    m['NLL_common']=float((yy['cs']*np.logaddexp(0,-z)+yy['cf']*np.logaddexp(0,z)).sum()/(yy['cs']+yy['cf']).sum())
                    rows.append(dict(arm=arm,seed=seed,step=step,split=split,**m))
    for kind in ('eta_only','full_context'):
        for seed in SEEDS:
            for split in ('train','validation','test'):
                z=old[f'ring_exchange__FF__{kind}__{seed}__{split}'];yy=y[split]
                rows.append(dict(arm='original_'+kind,seed=seed,step='selected',split=split,**metrics(z,yy['s'],yy['f'],yy['u'],4 if split=='train' else 16)))
    for seed in SEEDS:
        for step in (100,500,2000):
            for split in ('train','validation','test'):
                z=oldtraj[f'full_replay__{seed}__{step}'] if split=='test' else np.load(HERE/'fits/full_replay'/f'seed{seed}'/f'step{step}_source.npz')[split]
                yy=y[split];rows.append(dict(arm='original_full_trajectory',seed=seed,step=step,split=split,**metrics(z,yy['s'],yy['f'],yy['u'],4 if split=='train' else 16)))
    csvout(OUT/'metrics.csv',rows)
    chosen=read(OUT/'selection.json')['arms'];summary={}
    keys=('NLL','state_effect_rho','state_effect_R2','same_eta_state_change_accuracy','interaction_rho','reversal_both_correct')
    for arm in ARMS:
        summary[arm]={}
        for rule,step in [('VAL_NLL',chosen[arm]['NLL']['step']),('VAL_state_effect',chosen[arm]['state_effect_MSE']['step']),('last',2000)]:
            summary[arm][rule]=dict(step=step)
            for split in ('train','validation','test'):
                rr=[r for r in rows if r['arm']==arm and r['step']==step and r['split']==split]
                summary[arm][rule][split]={k:float(np.mean([r[k] for r in rr])) if all(r.get(k) is not None for r in rr) else None for k in keys}
                if split=='test':summary[arm][rule][split]['B15_by_seed']=[r['B15'] for r in rr]
    write(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','fit','export','analyze'))
    globals()[parser.parse_args().action]()
