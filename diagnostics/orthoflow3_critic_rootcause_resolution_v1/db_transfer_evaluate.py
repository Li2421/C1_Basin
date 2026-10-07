"""Identical frozen DB K16 pools. Models/scores frozen before cached labels."""
import argparse,csv,json,os,sqlite3,subprocess,sys,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest,spearmanr
from .db_transfer_data import OUT, ROOT, BASE, TARGET, read, write, sha

def csvwrite(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def inputs(worker,workers=5):
    from .db_transfer_context import Measure
    assert (OUT/'models_frozen.json').exists(), 'Source selection must precede target input measurement'
    manifest=read(OUT/'target_input_manifest.json');phys=read(TARGET/'physical.json')
    wrong=read(BASE/'diagnostics/orthoflow3_controller_intervention_generalization_v1/balanced_expansion/protocol.json')['profiles']['double_bottleneck']['alternate_path']
    right=BASE/'diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl'
    c=sqlite3.connect(f'file:{BASE}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
    uid={r['controller_uid'] for r in manifest};assert len(uid)==1
    fhash=c.execute('SELECT flow_checkpoint_sha256 FROM controller_config WHERE controller_uid=?',(next(iter(uid)),)).fetchone()[0];c.close();assert fhash==sha(right)
    ii=np.array([i for i in range(len(manifest)*16) if i%workers==worker]);values={};errors=[];timings={}
    for condition,path in (('correct',right),('wrong_controller',wrong)):
        measure=Measure('double_bottleneck',path);out=[];tt=[]
        for i in ii:
            s,k=divmod(int(i),16);start=time.perf_counter();v,err=measure.one(manifest[s]['state_uid'],phys[s],np.array(manifest[s]['eta'][k]))
            out.append(v);tt.append(time.perf_counter()-start);errors.extend(dict(index=int(i),condition=condition,**e) for e in err)
        values[condition]=np.array(out);timings[condition]=tt
    dest=OUT/'target_inputs';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'worker{worker}.npz',indices=ii,**values)
    write(dest/f'worker{worker}.json',dict(worker=worker,errors=errors,timings=timings,models_sha256=sha(OUT/'models_frozen.json'),context_code_sha256=sha(ROOT/'db_transfer_context.py'),outcomes_read=False,new_rollouts=0))
    print(dict(worker=worker,pairs=len(ii),errors=len(errors)),flush=True)

def predict():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    from .db_transfer_train import model
    assert not (OUT/'target_predictions.npz').exists(), 'Frozen predictions cannot be overwritten'
    frozen=read(OUT/'models_frozen.json');norm=read(OUT/'normalization.json');m=read(OUT/'target_input_manifest.json');n=len(m)*16
    raw={k:np.zeros((n,76),np.float32) for k in ('correct','wrong_controller')};saw=[]
    for w in range(5):
        d=np.load(OUT/'target_inputs'/f'worker{w}.npz');doc=read(OUT/'target_inputs'/f'worker{w}.json')
        assert doc['models_sha256']==sha(OUT/'models_frozen.json') and doc['context_code_sha256']==sha(ROOT/'db_transfer_context.py')
        saw.extend(d['indices'].tolist())
        for k in raw:raw[k][d['indices']]=d[k]
    assert sorted(saw)==list(range(n))
    context={}
    for k,v in raw.items():
        cc=(v-np.array(norm['context_center']))/norm['context_scale']
        for a,b,f in ((0,40,73),(40,56,74),(56,72,75)):cc[:,a:b]*=v[:,f,None]>0
        context[k]=cc.astype(np.float32)
    physical=read(TARGET/'physical.json');x=rep.batch([rep.entities(p) for p in physical])
    oldx=np.load(TARGET/'entities.npz')
    for k in x:np.testing.assert_allclose(x[k],oldx[k],rtol=0,atol=2e-6)
    eta=np.array([r['eta'] for r in m],np.float32).reshape(-1,3);e=((eta-norm['eta_center'])/norm['eta_scale']).astype(np.float32);si=np.repeat(np.arange(len(m)),16)
    # Secondary state tensor shuffle leaves actual C(h,eta) fixed; report that scope.
    shifted=np.roll(np.arange(len(m)),-1);scores={};latency=[]
    for run in frozen['models']:
        kind,seed=run['kind'],run['seed'];p=serialization.msgpack_restore((__import__('pathlib').Path(run['path'])/'checkpoint.msgpack').read_bytes());mod=model(kind)
        fn=jax.jit(lambda xx,ee,cc:mod.apply(p,xx,ee,cc))
        for condition in ('correct','wrong_controller','state_shuffle'):
            indices=shifted[si] if condition=='state_shuffle' else si
            cc=context['wrong_controller' if condition=='wrong_controller' else 'correct'];z=[]
            for i in range(0,n,128):z.append(np.asarray(fn(gather(x,indices[i:i+128]),jnp.asarray(e[i:i+128]),jnp.asarray(cc[i:i+128]))))
            scores[f'{kind}__seed{seed}__{condition}']=expit(np.concatenate(z)).reshape(-1,16)
    for kind in sorted({r['kind'] for r in frozen['models']}):
        for condition in ('correct','wrong_controller','state_shuffle'):scores[f'{kind}__ensemble__{condition}']=np.mean([scores[f'{kind}__seed{s}__{condition}'] for s in (17,23,41)],0)
    np.savez_compressed(OUT/'target_predictions.npz',**scores)
    np.savez_compressed(OUT/'target_context.npz',**raw)
    write(OUT/'prediction_freeze.json',dict(models_sha256=sha(OUT/'models_frozen.json'),predictions_sha256=sha(OUT/'target_predictions.npz'),target_inputs_sha256=sha(OUT/'target_input_manifest.json'),normalization_sha256=sha(OUT/'normalization.json'),target_outcomes_opened_in_this_round=False,code_sha256=sha(__file__),new_rollouts=0))
    print(dict(predictions_frozen=True,states=len(m),models=len(frozen['models']),new_rollouts=0),flush=True)

def evaluate():
    from shared_rollout_db.src.rollout_db import eta_identity,canonical
    freeze=read(OUT/'prediction_freeze.json');assert freeze['predictions_sha256']==sha(OUT/'target_predictions.npz')
    m=read(OUT/'target_input_manifest.json');req=[]
    for r in m:
        for eta in r['eta']:req.append(dict(state_uid=r['state_uid'],eta_uid=eta_identity(eta)[0],controller_uid=r['controller_uid'],seed_keys=[canonical({'future_index':j}) for j in range(16)]))
    write(OUT/'planned_cached_evaluation.json',dict(requests=req))
    subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(OUT/'planned_cached_evaluation.json'),'--output',str(OUT/'cache_preflight.json')],cwd=BASE,check=True,capture_output=True)
    con=sqlite3.connect(f'file:{BASE}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
    truth=[]
    for r in req:
        found=[dict(a) for a in con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(r['state_uid'],r['eta_uid'],r['controller_uid']))]
        by={a['seed_key']:a for a in found};rr=[by[k] for k in r['seed_keys'] if k in by]
        assert all(a['compatibility_quality']=='EXACT_REUSE' and not a['conflict_quarantined'] for a in rr)
        valid=[a for a in rr if not a['numerical_failure']];s=sum(a['success'] for a in valid);f=len(valid)-s;u=16-len(valid)
        truth.append(dict(**r,success=s,failure=f,unknown=u,B15=True if s>=15 else False if f>=2 else None,lower=s/16,upper=(s+u)/16,numerical=sum(a['numerical_failure'] for a in rr),collision=sum(a['collision'] for a in rr),rollout_uids=[a['rollout_uid'] for a in rr]))
    con.close();write(OUT/'cached_truth.json',truth)
    shape=(len(m),16);lo=np.array([r['lower'] for r in truth]).reshape(shape);hi=np.array([r['upper'] for r in truth]).reshape(shape);good=np.array([r['B15'] is True for r in truth]).reshape(shape);unknown=np.array([r['B15'] is None for r in truth]).reshape(shape);oracle=good.any(1)
    scores=dict(np.load(OUT/'target_predictions.npz'))
    historical=read(BASE/'diagnostics/orthoflow3_loso_partial_count_v1/target_predictions.json')['folds']['db']
    assert historical['state_uids']==[r['state_uid'] for r in m]
    for name in ('partial_count_shared','partial_count_eta_only','old_shared','old_source_selected_eta'):scores['historical__'+name]=np.asarray(historical['scores'][name])
    # Target-supervised references are copied only after all source scores freeze.
    old=read(BASE/'diagnostics/orthoflow3_loso_root_cause_v1/db/target_predictions.json')
    by={r['state_uid']:r for r in old['rows']}
    for name in ('target_reference','target_only','target_supervised'):
        if name in next(iter(by.values()))['scores']:scores['target_supervised_reference']=np.array([by[r['state_uid']]['scores'][name] for r in m]);break
    rows=[];selected={};picks={};tails=[]
    for name,p in scores.items():
        pick=p.argmax(1);idx=np.arange(len(m));g=good[idx,pick];u=unknown[idx,pick];l=lo[idx,pick];h=hi[idx,pick];pred=p[idx,pick]
        selected[name]=g;picks[name]=pick;valid=hi==lo;clip=np.clip(p,1e-7,1-1e-7)
        top=np.argsort(-p,axis=1)
        rows.append(dict(method=name,states=len(m),B15=int(g.sum()),unresolved=int(u.sum()),oracle_B15=int(oracle.sum()),gap=int(oracle.sum()-g.sum()),selection_if_available=float(g[oracle].mean()),selected_Q_lower=float(l.mean()),selected_Q_upper=float(h.mean()),selected_prediction=float(pred.mean()),regret_lower=float(np.maximum(0,lo.max(1)-h).mean()),regret_upper=float((hi.max(1)-l).mean()),severe_false_positive=int(((pred>.9)&(h<=.5)).sum()),p95_precision=float(g[pred>.95].mean()) if (pred>.95).any() else None,NLL=float((-lo[valid]*np.log(clip[valid])-(1-lo[valid])*np.log1p(-clip[valid])).mean()),MAE=float(abs(p[valid]-lo[valid]).mean()),Spearman=float(spearmanr(p[valid],lo[valid]).statistic),top2_B15=int(good[np.arange(len(m))[:,None],top[:,:2]].any(1).sum()),top3_B15=int(good[np.arange(len(m))[:,None],top[:,:3]].any(1).sum())))
        for a,b in ((0,.5),(.5,.9),(.9,.95),(.95,1.00001)):
            mask=(p>=a)&(p<b);tails.append(dict(method=name,min=a,max=min(b,1),count=int(mask.sum()),pred=float(p[mask].mean()) if mask.any() else None,Q_lower=float(lo[mask].mean()) if mask.any() else None,Q_upper=float(hi[mask].mean()) if mask.any() else None,B15_precision=float(good[mask].mean()) if mask.any() else None))
    comparisons=[];rng=np.random.default_rng(2026100527)
    for kind in ('full_context','full_context_freeze25'):
        for seed in (*['seed'+str(s) for s in (17,23,41)],'ensemble'):
            a=f'{kind}__{seed}__correct'
            for b in (f'eta_only__{seed}__correct',f'no_context__{seed}__correct',f'additive_nominal_context__{seed}__correct',f'{kind}__{seed}__wrong_controller'):
                diff=selected[a].astype(int)-selected[b].astype(int);r=int((diff>0).sum());br=int((diff<0).sum());boot=diff[rng.integers(0,len(m),(20000,len(m)))].mean(1)
                comparisons.append(dict(main=a,control=b,rescue=r,breaks=br,net=r-br,CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975)),paired_p=float(binomtest(r,r+br,.5).pvalue) if r+br else 1,top1_changed=int((picks[a]!=picks[b]).sum())))
    csvwrite(OUT/'target_metrics.csv',rows);csvwrite(OUT/'paired_comparisons.csv',comparisons);csvwrite(OUT/'upper_tail.csv',tails)
    write(OUT/'evaluation_audit.json',dict(states=len(m),candidate_pairs=len(truth),oracle_B15=int(oracle.sum()),numerical=sum(r['numerical'] for r in truth),unknown_seed_count=sum(r['unknown'] for r in truth),collisions=sum(r['collision'] for r in truth),source_selected_kind=read(OUT/'models_frozen.json')['source_selected_full'],new_rollouts=0,target_labels_used_for_selection=False,prediction_freeze_sha256=sha(OUT/'prediction_freeze.json'),scope='Source-only label-excluded DB LOSO on reused fixed target; not a new independent confirmation'))
    show=[r for r in rows if r['method'].endswith('__correct') or r['method'].startswith('historical')]
    print(json.dumps([{k:r[k] for k in ('method','B15','unresolved','NLL','selected_Q_lower','severe_false_positive')} for r in show]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','predict','evaluate']);p.add_argument('--worker',type=int,default=0);a=p.parse_args()
    inputs(a.worker) if a.action=='inputs' else globals()[a.action]()
