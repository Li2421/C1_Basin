"""Bounded FF-only transfer study. Existing TEST is a regression panel, not fresh confirmation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,hashlib,json,itertools,time,csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'orthoflow3_tt_ff_matched_learnability_v1'
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
SCENES=('toy_give_way','ring_exchange');SEEDS=(17,23,41)
ALPHAS=(0.,.25,.5,.75,1.)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def preregister():
    assert not (ROOT/'protocol.json').exists()
    files=[OLD/k for k in ('dataset.npz','entities.npz','frozen_predictions.npz','models_frozen.json','normalization.json','protocol.json','test_truth.npz','old_paradigm_repair_audit.csv')]
    files += [FIELD/'family_study_v2'/k for k in ('decision.json','factorial_summary.csv','analysis.json')]
    write(ROOT/'protocol.json',dict(scope='Small post-hoc FF transfer/regression study; previously opened TEST is not independent confirmation',
        new_rollouts=0,generator_modified=False,eta_lifting_modified=False,TT_modified=False,
        branches=['no_h: same Full architecture, zero physical continuous h channels; preserve entity masks/counts and actual FF response context',
                  'prior shrinkage: p=(1-alpha)*p_eta+alpha*p_conditional; alpha chosen on VAL only, same alpha across seeds'],
        controls=['frozen eta-only','frozen full-context','context shuffle','three-seed probability ensemble (secondary)'],
        seeds=SEEDS,steps=2000,batch_size=128,eval_cadence=100,optimizer='AdamW lr8e-4 wd1e-4 clip5; same minibatch stream/init as frozen full',
        checkpoint_selection='minimum three-seed mean VAL observed-count NLL, same as original',
        shrinkage_alphas=ALPHAS,branch_selection='VAL NLL only; no TEST-specific choice; alpha0 is prior fallback NOT state-learning success',
        stop='6 small fits + VAL-only shrinkage + one regression evaluation; no iterative test tuning or new rollout',
        source_hashes={str(p):sha(p) for p in files},code_sha256=sha(__file__)))

def data():
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.train import inputs
    d,x,m=inputs()
    # Keep valid-entity masks, but remove all measured physical channels.
    # Agent count is constant within each scene. FF context still carries physical state response.
    xx={k:(v if k.endswith('_mask') else np.zeros_like(v)) for k,v in x.items()}
    return d,xx,m

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train import model
    protocol=read(ROOT/'protocol.json');assert protocol['code_sha256']==sha(__file__)
    scene,seed=list(itertools.product(SCENES,SEEDS))[index];d,x,m=data();ci=1
    dest=ROOT/'models'/scene/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    tr=np.flatnonzero((d['scene']==scene)&(d['split']=='train'));va=np.flatnonzero((d['scene']==scene)&(d['split']=='validation'))
    pairs=np.array([(i,j) for i in tr for j in range(16) if d['success'][ci,i,j]+d['failure'][ci,i,j]>0]);vp=np.array([(i,j) for i in va for j in range(16)])
    net=model('full_context');gather=lambda ii:{k:jnp.asarray(v[ii]) for k,v in x.items()}
    pp=net.init(jax.random.PRNGKey(seed),gather(tr[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    initsha=hashlib.sha256(serialization.to_bytes(pp)).hexdigest()
    reference=read(OLD/'models'/scene/'FF'/'full_context'/f'seed{seed}'/'complete.json')
    assert initsha==reference['initial_sha256']
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));opt=optimizer.init(pp)
    mask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in traverse_util.flatten_dict(pp)})
    nmean=np.mean([d['success'][ci,i,j]+d['failure'][ci,i,j] for i,j in pairs])
    def make_step(warm):
        @jax.jit
        def step(p,o,xx,ee,cc,ss,ff):
            def loss(q):
                z=net.apply(q,xx,ee,cc);return jnp.mean(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/nmean
            val,grad=jax.value_and_grad(loss)(p)
            if warm:grad=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),grad,mask)
            upd,o=optimizer.update(grad,o,p)
            if warm:upd=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),upd,mask)
            return optax.apply_updates(p,upd),o,val
        return step
    step,warm=make_step(False),make_step(True);predict=jax.jit(lambda p,xx,ee,cc:net.apply(p,xx,ee,cc))
    rng=np.random.default_rng(seed);order=hashlib.sha256();hist=[];start=time.perf_counter()
    for it in range(1,2001):
        draw=rng.integers(len(pairs),size=128);order.update(draw.tobytes());ii,jj=pairs[draw].T
        pp,opt,loss=(warm if it<=25 else step)(pp,opt,gather(ii),jnp.asarray(d['eta'][ii,jj]),jnp.asarray(d['context'][ci,ii,jj]),jnp.asarray(d['success'][ci,ii,jj]),jnp.asarray(d['failure'][ci,ii,jj]))
        if it%100:continue
        ii,jj=vp.T;z=np.asarray(predict(pp,gather(ii),jnp.asarray(d['eta'][ii,jj]),jnp.asarray(d['context'][ci,ii,jj])))
        s=d['success'][ci,ii,jj];f=d['failure'][ci,ii,jj];nll=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum())
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(pp));np.savez_compressed(dest/f'val_step{it}.npz',logits=z.reshape(len(va),16),indices=va)
        hist.append(dict(step=it,VAL_NLL=nll,train_loss=float(loss)))
    assert order.hexdigest()==reference['batch_draw_sha256']
    assert hashlib.sha256(pairs.tobytes()).hexdigest()==reference['pair_order_sha256']
    write(dest/'history.json',hist);write(dest/'complete.json',dict(scene=scene,seed=seed,seconds=time.perf_counter()-start,backend=jax.default_backend(),
        matched_init=True,matched_pair_order=True,matched_batch_draws=True,steps=2000,code_sha256=sha(__file__),TEST_labels_read=False))
    print(json.dumps(dict(scene=scene,seed=seed,completed=True)),flush=True)

def finalize():
    import jax,jax.numpy as jnp
    from flax import serialization
    from scipy.special import expit,logit
    from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train import model
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.evaluate import point_metrics,reversals,interval
    assert not (ROOT/'final_decision.json').exists()
    d,x,m=data();old=np.load(OLD/'frozen_predictions.npz');net=model('full_context');pred={};selection={};alpha_scores=[]
    for scene in SCENES:
        paths=[ROOT/'models'/scene/f'seed{seed}' for seed in SEEDS]
        hh=[read(p/'history.json') for p in paths];best=int(np.argmin(np.mean([[r['VAL_NLL'] for r in h] for h in hh],0)));step=hh[0][best]['step']
        selection[scene]={'no_h_checkpoint_step':step,'checkpoint_hashes':{str(seed):sha(p/f'step{step}.msgpack') for seed,p in zip(SEEDS,paths)}}
        for seed,p in zip(SEEDS,paths):
            assert read(p/'complete.json')['code_sha256']==sha(__file__)
            pp=serialization.msgpack_restore((p/f'step{step}.msgpack').read_bytes());fn=jax.jit(lambda xx,ee,cc:net.apply(pp,xx,ee,cc))
            for split in ('validation','test'):
                idx=np.flatnonzero((d['scene']==scene)&(d['split']==split));protocol=read(OLD/'protocol.json')
                perm=sorted(idx,key=lambda i:hashlib.sha256(('tt_ff_shuffle_v1|'+protocol['states'][i]['state_uid']).encode()).hexdigest())
                replace=dict(zip(perm,np.roll(perm,-1)));other=np.array([replace[i] for i in idx])
                for condition in ('correct','context_shuffle'):
                    cs=idx if condition=='correct' else other;ii=np.repeat(idx,16)
                    z=np.asarray(fn({k:jnp.asarray(v[ii]) for k,v in x.items()},jnp.asarray(d['eta'][idx].reshape(-1,3)),jnp.asarray(d['context'][1,cs].reshape(-1,76)))).reshape(len(idx),16)
                    pred[scene,'no_h',seed,split,condition]=z
                    for kind,oldkind in (('eta_only','eta_only'),('full','full_context')):
                        pred[scene,kind,seed,split,condition]=old[f'{scene}__FF__{oldkind}__{seed}__{split}__{condition}']
        va=np.flatnonzero((d['scene']==scene)&(d['split']=='validation'));s=d['success'][1,va];f=d['failure'][1,va]
        def vnll(p):
            p=np.clip(p,1e-7,1-1e-7);return float(-(s*np.log(p)+f*np.log1p(-p)).sum()/(s+f).sum())
        for branch in ('full','no_h'):
            scores=[]
            for alpha in ALPHAS:
                score=np.mean([vnll((1-alpha)*expit(pred[scene,'eta_only',seed,'validation','correct'])+alpha*expit(pred[scene,branch,seed,'validation','correct'])) for seed in SEEDS]);scores.append(score)
                alpha_scores.append(dict(scene=scene,branch=branch,alpha=alpha,VAL_NLL=float(score)))
            alpha=ALPHAS[int(np.argmin(scores))];selection[scene][branch+'_alpha']=alpha
            for seed,split,condition in itertools.product(SEEDS,('validation','test'),('correct','context_shuffle')):
                p=(1-alpha)*expit(pred[scene,'eta_only',seed,split,condition])+alpha*expit(pred[scene,branch,seed,split,condition]);pred[scene,branch+'_shrunk',seed,split,condition]=logit(np.clip(p,1e-7,1-1e-7))
        kinds=('eta_only','full','no_h','full_shrunk','no_h_shrunk')
        scores={kind:float(np.mean([vnll(expit(pred[scene,kind,seed,'validation','correct'])) for seed in SEEDS])) for kind in kinds}
        selection[scene]['VAL_scores']=scores;selection[scene]['VAL_selected']=min(kinds,key=lambda k:scores[k])
    # All checkpoints/alphas/branch choices are frozen before opening labels for this evaluation.
    write(ROOT/'selection_frozen.json',dict(selection=selection,TEST_labels_used=False,scope='post-hoc regression; cohort previously examined',new_rollouts=0))
    np.savez_compressed(ROOT/'predictions_frozen.npz',**{'__'.join(map(str,k)):v for k,v in pred.items()});csvout(ROOT/'alpha_validation.csv',alpha_scores)
    truth=np.load(OLD/'test_truth.npz');o=truth['outcomes'][1];ids=truth['indices'];s=((o[...,0]==1)&(o[...,1]==0)).sum(-1).astype(float);u=o[...,1].sum(-1).astype(float);f=16-u-s
    metrics=[];paired=[];perstate=[];table=[]
    for scene in SCENES:
        idx=np.flatnonzero((d['scene']==scene)&(d['split']=='test'));ix=np.array([np.flatnonzero(ids==i)[0] for i in idx]);hits={}
        for kind in ('eta_only','full','no_h','full_shrunk','no_h_shrunk'):
            for seed,condition in itertools.product(SEEDS,('correct','context_shuffle')):
                z=pred[scene,kind,seed,'test',condition];mm,aa=point_metrics(z,s[ix],f[ix],u[ix],s[ix],f[ix]);rr=reversals(s[ix],f[ix],z)
                metrics.append(dict(scene=scene,kind=kind,seed=seed,condition=condition,**mm,**rr))
                if condition=='correct':
                    hits[kind,seed]=aa
                    for j,i in enumerate(idx):perstate.append(dict(scene=scene,kind=kind,seed=seed,state_index=int(i),selected_eta=int(aa['selected'][j]),B15=bool(aa['hit'][j]),unknown=bool(aa['unknown'][j])))
            zz=np.mean([expit(pred[scene,kind,seed,'test','correct']) for seed in SEEDS],0);mm,_=point_metrics(logit(np.clip(zz,1e-7,1-1e-7)),s[ix],f[ix],u[ix],s[ix],f[ix])
            metrics.append(dict(scene=scene,kind=kind,seed='ensemble',condition='correct_secondary',**mm))
            mm=[r for r in metrics if r['scene']==scene and r['kind']==kind and r['condition']=='correct'];row=dict(scene=scene,kind=kind,B15_by_seed=[r['B15'] for r in mm],oracle=mm[0]['oracle_B15'])
            for k in ('NLL','MAE','Spearman','selected_Q_observed','Q_regret','severe_FP','stable_reversal_accuracy'):row[k+'_mean']=float(np.mean([r[k] for r in mm]))
            table.append(row)
            for base in ('eta_only','full'):
                delta=[];resc=[];br=[]
                for seed in SEEDS:
                    if (base,seed) not in hits:break
                    a,b=hits[kind,seed],hits[base,seed]
                    delta.append(a['hit'].astype(float)-b['hit']);resc.append(int((a['hit']&b['bad']).sum()));br.append(int((a['bad']&b['hit']).sum()))
                if len(delta)==3:paired.append(dict(scene=scene,kind=kind,baseline=base,rescue_by_seed=resc,break_by_seed=br,delta_B15=float(np.mean(delta)),state_bootstrap95=interval(np.mean(delta,0))))
    csvout(ROOT/'metrics.csv',metrics);csvout(ROOT/'summary.csv',table);csvout(ROOT/'per_state.csv',perstate);write(ROOT/'paired.json',paired)
    write(ROOT/'final_decision.json',dict(completed=True,scope='Small FF post-hoc regression, NOT fresh confirmation/LOSO/unseen eta',new_rollouts=0,selection=selection,table=table,paired=paired,
        stop_requested_by_user='Small validation completed; hand off for review. No automatic expansion.',
        no_B15_headroom_vs_eta_only=True,warning='alpha0 or eta-only selection is robust-prior fallback, not evidence of learning state/context interaction',
        ablation_evidence=str(FIELD/'family_study_v2/decision.json')))
    print(json.dumps(dict(completed=True,selection=selection,table=table),allow_nan=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preregister','train','finalize'));p.add_argument('--index',type=int);a=p.parse_args()
    train(a.index) if a.action=='train' else globals()[a.action]()
