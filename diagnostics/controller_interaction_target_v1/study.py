"""A single diagnostic: can the existing input/network learn isolated S x eta?

No new controller, state, feature query, rollout, or architecture search.
"""
import argparse,time
from pathlib import Path
import numpy as np
from diagnostics.controller_cached_union_v1.study import HERE as BASE, FOLDS, SEEDS, read,write,freeze,sha

HERE=Path(__file__).resolve().parent
ARMS=('full_q_mse','interaction_mse')
JOBS=[dict(fold=f,arm=a,seed=s) for f in FOLDS for a in ARMS for s in SEEDS]


def project(y,mask):
    """Weighted two-way additive projection on observed cells only.

    Residual has zero observed row and column means. A complete rectangular
    panel reduces exactly to y - state_mean - eta_mean + grand_mean.
    """
    mask=np.asarray(mask,bool);y=np.asarray(y,float)
    a=np.zeros(len(y));b=np.zeros(y.shape[1]);nr=np.maximum(mask.sum(1),1);nc=np.maximum(mask.sum(0),1)
    for iteration in range(1000):
        aa=np.sum(np.where(mask,y-b,0),1)/nr
        bb=np.sum(np.where(mask,y-aa[:,None],0),0)/nc
        if max(np.max(abs(aa-a)),np.max(abs(bb-b)))<1e-12:
            a,b=aa,bb;break
        a,b=aa,bb
    else:raise RuntimeError('Additive projection did not converge')
    out=np.where(mask,y-a[:,None]-b,0)
    assert np.max(abs(out.sum(0)/nc))<1e-9
    assert np.max(abs(out.sum(1)/nr))<1e-9
    return out


def prepare():
    HERE.mkdir(exist_ok=True)
    assert read(BASE/'completion.json')['fits']==60
    reused=['orthoflow3_controller_state_residual_factorial_v1/train_models.py',
        'orthoflow3_controller_intervention_generalization_v1/train_triplet_probe.py',
        'orthoflow3_source_contrast_interaction_v1/train_models.py',
        'controller_q_diversity_v1/train.py']
    freeze(HERE/'protocol.json',dict(
        schema='isolated_state_eta_target_v1',question='Is failure to learn transferable interaction explained by competition with state difficulty and controller/eta marginal effects in the full-Q objective?',
        history={str(BASE.parent/p):sha(BASE.parent/p) for p in reused},
        history_difference='Old H20 residual/additive and certified controller-order losses exist and are not rerun. No direct source-TRAIN two-way-demeaned target regression on the current actualFF compact-field cache union was found. Stage17 checkpoints/labels/features are all reused.',
        base_protocol_sha256=sha(BASE/'protocol.json'),dataset_sha256=sha(BASE/'dataset.npz'),
        source_inputs='Exactly Stage17 native-slot physical entities and compact64 controller input. No extra context, sceneID, program parameter, or simulator query.',
        arms=ARMS,folds=FOLDS,seeds=SEEDS,
        target='For each SOURCE TRAIN scene/controller, regress observed empirical success onto state and eta additive effects with equal observed-cell weights; interaction target is the residual. All numeric-unknown cells remain unknown; fit ratios use only observed outcomes, exactly as the preceding BCE experiment.',
        controls='full_q_mse uses the uncentered empirical success on identical cells, draws, scalar-output architecture, initialization, optimizer and checkpoint rule. Old slot BCE, fair eta-only and zero-interaction predictors are reused. This isolates target projection from the change of BCE to regression.',
        model='Same Critic(state_field_compact) and exact parameter shapes. Raw scalar output trained by squared error. For interaction_mse only, selection score = frozen source-only state-erased Stage17 prediction + predicted interaction. It is an uncalibrated diagnostic score, not a validated probability or deployed critic.',
        selection='PRIMARY checkpoint: SOURCE VAL per(scene,controller) interaction residual MSE; then pessimistic selection regret, negative pairwise ranking, full-score MSE, earliest step. Same rule for both new arms. Test never selects anything. This is an interaction-learnability diagnostic, not a replacement formal benchmark checkpoint rule.',
        train=read(BASE/'protocol.json')['training'],normalization='Reuse exact source-TRAIN Stage17 slot normalization per fold/seed.',
        evaluation='Every four-scene held panel and joint sourceVAL. Primary mechanism uses raw-network interaction prediction, zero-interaction and same-network full-Q control; independent seed halves and all-C-pair statistics retained. Secondary score selection reuses fixed-policy, mismatch, reversal, collision and censor audits. Previously opened Q16 also DEVELOPMENT; no positive final claim without untouched confirmation.',
        stop_rule='If isolated target also fails in LOSO, do not tune its weight/size/lr: reject mean-effect competition as a sufficient explanation and inspect the remaining input/support problem. If it helps, require Q/selection/mismatch and fresh confirmation before any claim of repair.',
        new_rollouts=0,new_static_queries=0,
        literature=[dict(url='https://proceedings.mlr.press/v48/wangf16.html',
            relevance='Separating state value and action advantage motivates a controlled decomposition.',
            boundary='That paper does not establish our double-centered target or cross-scene controller transfer. This is our falsifiable diagnostic, not a reproduced success claim.')],
        code_sha256=sha(__file__)))
    d=dict(np.load(BASE/'dataset.npz'));mask=d['success']+d['failure']>0
    y=d['success']/np.maximum(d['success']+d['failure'],1);res=np.zeros_like(y,dtype=float);groups=[]
    train=d['split']=='train'
    for sc in np.unique(d['scene'][train]):
        for ctl in np.unique(d['controller'][train&(d['scene']==sc)]):
            ix=np.flatnonzero(train&(d['scene']==sc)&(d['controller']==ctl));m=mask[ix]
            res[ix]=project(y[ix],m)
            groups.append(dict(scene=str(sc),controller=str(ctl),N=len(ix),observed_cells=int(m.sum()),
                Q_second_moment=float(np.mean(y[ix][m]**2)),interaction_second_moment=float(np.mean(res[ix][m]**2))))
    np.savez_compressed(HERE/'training_targets.npz',full_q=y,interaction=res,known=mask,train=train)
    write(HERE/'target_audit.json',dict(groups=groups,target_sha256=sha(HERE/'training_targets.npz'),
        validation_in_training_projection=False,other_scenes_in_any_group=False,new_rollouts=0))
    print(dict(fits=len(JOBS),new_rollouts=0,groups=len(groups)),flush=True)


def evaluate(raw,score,d,ix):
    s,f,u=[d[k][ix] for k in ('success','failure','unknown')]
    count=s+f+u;observed=count>0;known=s+f>0
    lo=s/np.maximum(count,1);hi=(s+u)/np.maximum(count,1);y=s/np.maximum(s+f,1)
    chosen=np.where(observed,score,-np.inf).argmax(-1);rr=np.arange(len(ix))
    regret=np.where(observed,hi-lo[rr,chosen,None],-np.inf);regret[rr,chosen]=0
    regret=np.maximum(regret.max(-1),0)
    a,b=np.triu_indices(16,1);pos=(lo[:,a]-hi[:,b]>=.25)&observed[:,a]&observed[:,b]
    neg=(lo[:,b]-hi[:,a]>=.25)&observed[:,a]&observed[:,b];strong=pos|neg
    delta=score[:,a]-score[:,b];acc=np.where(abs(delta)<1e-7,.5,np.where(pos,delta>0,delta<0))
    groups=[];scenes=[]
    for sc in np.unique(d['scene'][ix]):
        gg=[]
        for ctl in np.unique(d['controller'][ix][d['scene'][ix]==sc]):
            j=np.flatnonzero((d['scene'][ix]==sc)&(d['controller'][ix]==ctl));m=known[j]
            truth=project(y[j],m);pred=project(raw[j],m)
            g=dict(scene=str(sc),controller=str(ctl),N=len(j),
                interaction_MSE=float(np.mean((truth-pred)[m]**2)),interaction_skill=float(np.mean((2*truth*pred-pred**2)[m])),
                regret=float(regret[j].mean()),ranking=float(acc[j][strong[j]].mean()) if strong[j].any() else .5,
                Q_MSE=float(np.mean((score[j]-y[j])[m]**2)),selected_success=float(lo[j,chosen[j]].mean()))
            gg.append(g);groups.append(g)
        scenes.append({k:float(np.mean([g[k] for g in gg])) for k in ('interaction_MSE','interaction_skill','regret','ranking','Q_MSE')})
    return dict(**{k:float(np.mean([g[k] for g in scenes])) for k in scenes[0]},groups=groups)


def fit(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic
    job=JOBS[index];p=read(HERE/'protocol.json');h=p['train']
    assert sha(__file__)==p['code_sha256'] and sha(BASE/'dataset.npz')==p['dataset_sha256']
    assert sha(HERE/'training_targets.npz')==read(HERE/'target_audit.json')['target_sha256']
    d=dict(np.load(BASE/'dataset.npz'));t=dict(np.load(HERE/'training_targets.npz'))
    held=FOLDS[job['fold']];source=np.ones(len(d['scene']),bool) if held is None else d['scene']!=held
    tr=source&(d['split']=='train');va=np.flatnonzero(source&(d['split']=='validation'))
    assert not (set(d['physical_hash'][tr])&set(d['physical_hash'][va]))
    if held:assert held not in d['scene'][tr] and held not in d['scene'][va]
    out=HERE/'fits'/job['fold']/job['arm']/f"seed{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    parent=BASE/'fits'/job['fold']/'union_native_slots'/f"seed{job['seed']}"
    norm=read(parent/'normalization.json');freeze(out/'normalization.json',norm)
    freeze(out/'job.json',dict(**job,protocol_sha256=sha(HERE/'protocol.json'),
        source_train_physical_hashes=sorted(set(d['physical_hash'][tr])),source_validation_physical_hashes=sorted(set(d['physical_hash'][va])),held_labels_used=False))
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    c=jnp.asarray(np.clip((d['feature']-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10)*d['entity_agent_mask'][...,None],jnp.float32)
    e=jnp.asarray((d['eta']-np.asarray(norm['eta_center']))/np.asarray(norm['eta_radius']),jnp.float32)
    target=jnp.asarray(t['interaction' if job['arm']=='interaction_mse' else 'full_q'],jnp.float32)
    prior=np.load(BASE/'fits'/job['fold']/'union_slots_state_erased'/f"seed{job['seed']}"/'predictions.npz')['probability']
    net=Critic('state_field_compact');gather=lambda ii:{k:v[ii] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(job['seed']),gather(jnp.array([0])),c[:1],e[:1])
    optim=optax.chain(optax.clip_by_global_norm(h['clip_grad']),optax.adamw(h['lr'],weight_decay=h['weight_decay']));opt=optim.init(params)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            par,state=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):return jnp.mean((net.apply(pp,gather(ii),c[ii],e[jj])-target[ii,jj])**2)
            value,grad=jax.value_and_grad(loss)(par);updates,state=optim.update(grad,state,par)
            return (optax.apply_updates(par,updates),state),value
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def predict(par,ii,jj):return net.apply(par,gather(ii),c[ii],e[jj])
    def infer(par,rows):
        ii=np.repeat(rows,16);jj=np.tile(np.arange(16),len(rows));parts=[]
        for start in range(0,len(ii),256):parts.append(np.asarray(predict(par,jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
        return np.concatenate(parts).reshape(-1,16)
    pools={}
    for sc in np.unique(d['scene'][tr]):
        pools[sc]=[]
        for ctl in np.unique(d['controller'][tr&(d['scene']==sc)]):
            rows=[]
            for i in np.flatnonzero(tr&(d['scene']==sc)&(d['controller']==ctl)):
                js=np.flatnonzero(t['known'][i]);assert len(js);rows.append((int(i),js))
            pools[sc].append(rows)
    rng=np.random.default_rng(job['seed']);best=None;history=[];t0=time.monotonic()
    for step in range(h['validation_every'],h['steps']+1,h['validation_every']):
        chunks=[]
        for sc,groups in pools.items():
            shape=(h['validation_every'],h['batch_per_scene']);which=rng.integers(len(groups),size=shape);draw=np.empty(shape+(2,),np.int32)
            for group_index,group in enumerate(groups):
                picked=np.argwhere(which==group_index);state_choice=rng.integers(len(group),size=len(picked))
                for choice in np.unique(state_choice):
                    places=picked[state_choice==choice];i,js=group[int(choice)];eta_choice=js[rng.integers(len(js),size=len(places))]
                    draw[places[:,0],places[:,1],0]=i;draw[places[:,0],places[:,1],1]=eta_choice
            chunks.append(draw)
        (params,opt),loss=block(params,opt,jnp.asarray(np.concatenate(chunks,1)))
        raw=infer(params,va);score=raw+prior[va] if job['arm']=='interaction_mse' else raw
        m=evaluate(raw,score,d,va);row=dict(step=step,training_MSE=float(np.mean(loss)),**m);history.append(row)
        key=(m['interaction_MSE'],m['regret'],-m['ranking'],m['Q_MSE'],step)
        if best is None or key<best:best=key;best_bytes=serialization.to_bytes(params);selected=row
        if step%1000==0:
            write(out/'history.json',history);print(dict(**job,step=step,best_step=selected['step'],interaction_MSE=m['interaction_MSE'],seconds=time.monotonic()-t0),flush=True)
    chosen=serialization.from_bytes(params,best_bytes);(out/'checkpoint.msgpack').write_bytes(best_bytes)
    raw=infer(chosen,np.arange(len(d['scene'])));score=raw+prior if job['arm']=='interaction_mse' else raw
    np.savez_compressed(out/'predictions.npz',probability=score,raw_network_output=raw,state_uid=d['state_uid'],
        probability_key_is_uncalibrated_score=np.array(True))
    write(out/'history.json',history)
    write(out/'complete.json',dict(**job,selected=selected,
        source_train_metrics=evaluate(raw[np.flatnonzero(tr)],score[np.flatnonzero(tr)],d,np.flatnonzero(tr)),
        checkpoint_sha256=sha(out/'checkpoint.msgpack'),protocol_sha256=sha(HERE/'protocol.json'),seconds=time.monotonic()-t0,
        raw_predictions_are_calibrated_probability=False,held_scene_labels_used_for_fit_or_selection=False,new_rollouts=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--index',type=int);a=ap.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
