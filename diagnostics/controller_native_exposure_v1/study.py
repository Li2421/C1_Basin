"""Matched native-branch exposure/interference audit. No new rollout labels."""
import argparse,time
from pathlib import Path
import numpy as np
from diagnostics.controller_cached_union_v1.study import HERE as BASE,SEEDS,read,write,freeze,sha

HERE=Path(__file__).resolve().parent
OLD=BASE.parent/'controller_entity_coverage_check_v1'
ARMS=('joint_replay','isolated_sparse_field','isolated_sparse_zero','isolated_dense_field','isolated_dense_zero')
JOBS=[dict(arm=a,seed=s) for a in ARMS for s in SEEDS]


def prepare():
    HERE.mkdir(exist_ok=True)
    assert read(BASE.parent/'controller_interaction_target_v1/completion.json')['fits']==30
    d=dict(np.load(BASE/'dataset.npz'));old=dict(np.load(OLD/'dataset.npz'))
    tr=d['split']=='train';native=(d['scene']=='ring_exchange')&(d['controller']=='native_flow')&(d['source']=='native_ring512')
    rows=np.flatnonzero(native);oi=d['source_row'][rows]
    for k in old:
        if k.startswith('entity_'):
            a=d[k][rows];b=old[k][oi]
            if k=='entity_obstacles':a=a[:,:,:b.shape[2]]
            if k=='entity_obstacle_mask':a=a[:,:b.shape[1]]
            np.testing.assert_array_equal(a,b)
    freeze(HERE/'protocol.json',dict(
        schema='native_branch_exposure_interference_v1',new_rollouts=0,
        question='Why did mixed-controller training fail to retain the already learnable native Ring state interaction: diluted native exposure, interaction with other tasks, or the added state-dependent controller descriptor?',
        trigger='Stage17 nativeRing TRAIN Q error stays near its mean-only predictor; expected exposures/cell 4.626 versus187.5 in the previous successful standalone512 fit. This is a confound, not yet a demonstrated cause of LOSO failure.',
        history_audit='Historical index and controller scripts checked for matched native-only exposure, negative-transfer or gradient-conflict tests; none found. Existing native physical-only512/slot fits are reused anchors, not rerun. Old/current feature channel shapes differ, so their exposure comparison alone is not causal.',
        prior_completed=['controller_cached_union_v1','controller_interaction_target_v1','controller_entity_symmetry_probe_v1'],
        dataset_sha256=sha(BASE/'dataset.npz'),old_native_sha256=sha(OLD/'dataset.npz'),
        physical_compatibility='All native544 TRAIN/VAL physical entity arrays match the previous dataset exactly after removing masked padding.',
        arms=ARMS,seeds=SEEDS,
        input='Same current Stage17 physical/API-slot encoding, compact64 controller features, scalar architecture, initialization and frozen joint SOURCE TRAIN normalization. Zero arms zero exactly the64 controller channels, retaining same parameter shapes and physical slots.',
        sparse='Reconstruct exact Stage17 joint draws. Delete non-native loss terms while retaining every native example at the same optimizer step. Loss denominator remains480; padded native batches use mask. 6000steps. No new states/labels.',
        dense='Native512 only,12000steps x128 independently drawn observed cells, matching the old exposure budget. Same sourceVAL checkpoint criterion and current model shape.',
        joint_replay='Three necessary replays retain native-VAL-selected and final checkpoints that Stage17 did not save. Original global-selected checkpoint must reproduce byte-for-byte, or report a replay-integrity blocker before causal interpretation. This is not a repeated unchanged scientific comparison.',
        checkpoint='All new arms use native SOURCE VAL32 only: regret, negative ranking, state_eta_MSE, NLL, earliest. Final-step also retained. No TEST model choice. Replay additionally tracks original source-global choice for hash verification.',
        evaluation='Old opened nativeTEST64 (original16/fresh-origin48 strata) plus sourceTRAIN/VAL. Paired physical-state bootstrap20000, wrong-state multiple derangements, all16fixed policies, independent8/8 reversals and censor/collision reporting. This single-controller diagnostic cannot establish controller generalization or LOSO.',
        feature_gap='64 old nativeTEST states have no cachedcompact64 descriptor; reuse all labels/physicalinputs and query only those static features, with Environment.step forbidden. No success label affects queries.',
        stop_rule='Complete this factorial and interpret exposure versus task-mixture versus descriptor. Do not silently treat a local native repair as the root cause or repair of all LOSO failures.',
        literature=[dict(url='https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html',
            use='Motivates separating multi-task interference from per-task learning. No PCGrad intervention is being introduced and interference is not presumed.')],
        code_sha256=sha(__file__)))
    pools={}
    for sc in np.unique(d['scene'][tr]):
        pools[sc]=[]
        for ctl in np.unique(d['controller'][tr&(d['scene']==sc)]):
            group=[]
            for i in np.flatnonzero(tr&(d['scene']==sc)&(d['controller']==ctl)):
                js=np.flatnonzero(d['success'][i]+d['failure'][i]>0);assert len(js);group.append((int(i),js))
            pools[sc].append(group)
    for seed in SEEDS:
        rng=np.random.default_rng(seed);blocks=[]
        for _ in range(30):
            chunks=[]
            for sc,groups in pools.items():
                shape=(200,120);which=rng.integers(len(groups),size=shape);draw=np.empty(shape+(2,),np.int32)
                for gi,group in enumerate(groups):
                    places=np.argwhere(which==gi);choices=rng.integers(len(group),size=len(places))
                    for choice in np.unique(choices):
                        where=places[choices==choice];i,js=group[int(choice)];jj=js[rng.integers(len(js),size=len(where))]
                        draw[where[:,0],where[:,1],0]=i;draw[where[:,0],where[:,1],1]=jj
                chunks.append(draw)
            blocks.append(np.concatenate(chunks,1))
        all_draws=np.concatenate(blocks);keep=native[all_draws[:,:,0]];size=int(keep.sum(1).max())
        sparse=np.zeros((6000,size,2),np.int32);mask=np.zeros((6000,size),np.float32)
        for i in range(6000):
            a=all_draws[i][keep[i]];sparse[i,:len(a)]=a;sparse[i,len(a):]=a[:1] if len(a) else [np.flatnonzero(native&tr)[0],0];mask[i,:len(a)]=1
        np.savez_compressed(HERE/f'draws_seed{seed}.npz',joint=all_draws,sparse=sparse,mask=mask)
        write(HERE/f'draw_audit_seed{seed}.json',dict(native_draws=int(mask.sum()),zero_native_steps=int((mask.sum(1)==0).sum()),
            observed_native_pairs=int(((d['success']+d['failure']>0)&(native&tr)[:,None]).sum()),sha256=sha(HERE/f'draws_seed{seed}.npz')))
    write(HERE/'input_compatibility.json',dict(native_physical_arrays_exact=True,train_states=int((native&tr).sum()),validation_states=int((native&(d['split']=='validation')).sum()),old_test_states=64,new_rollouts=0))
    print(dict(fits=len(JOBS),new_rollouts=0,new_static_test_descriptors=64),flush=True)


def fit(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic
    from diagnostics.controller_cached_union_v1.learn import metrics
    job=JOBS[index];p=read(HERE/'protocol.json');assert sha(__file__)==p['code_sha256'] and sha(BASE/'dataset.npz')==p['dataset_sha256']
    d=dict(np.load(BASE/'dataset.npz'));native=(d['scene']=='ring_exchange')&(d['controller']=='native_flow')&(d['source']=='native_ring512')
    tr=np.flatnonzero(native&(d['split']=='train'));va=np.flatnonzero(native&(d['split']=='validation'));global_va=np.flatnonzero(d['split']=='validation')
    out=HERE/'fits'/job['arm']/f"seed{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    parent=BASE/'fits/joint/union_native_slots'/f"seed{job['seed']}";norm=read(parent/'normalization.json');freeze(out/'normalization.json',norm)
    freeze(out/'job.json',dict(**job,protocol_sha256=sha(HERE/'protocol.json'),source_train_states=d['state_uid'][tr].tolist(),source_val_states=d['state_uid'][va].tolist(),test_used=False))
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')};x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    feature_dtype=d['feature'].dtype
    c=jnp.asarray(np.clip((d['feature']-np.asarray(norm['mean'],dtype=feature_dtype))/np.asarray(norm['scale'],dtype=feature_dtype),-10,10)*d['entity_agent_mask'][...,None],jnp.float32)
    if job['arm'].endswith('_zero'):c=jnp.zeros_like(c)
    e=jnp.asarray((d['eta']-np.asarray(norm['eta_center']))/np.asarray(norm['eta_radius']),jnp.float32)
    s,f=[jnp.asarray(d[k],jnp.float32) for k in ('success','failure')]
    net=Critic('state_field_compact');gather=lambda ii:{k:v[ii] for k,v in x.items()}
    par=net.init(jax.random.PRNGKey(job['seed']),gather(jnp.array([0])),c[:1],e[:1])
    optim=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));opt=optim.init(par)
    sparse='sparse' in job['arm'];dense='dense' in job['arm'];steps=12000 if dense else 6000
    @jax.jit
    def block(par,opt,draws,masks):
        def step(carry,dm):
            params,state=carry;draw,mask=dm;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):
                z=net.apply(pp,gather(ii),c[ii],e[jj]);ss,ff=s[ii,jj],f[ii,jj]
                values=(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1)
                return jnp.sum(values*mask)/480 if sparse else jnp.mean(values)
            value,grad=jax.value_and_grad(loss)(params);updates,state=optim.update(grad,state,params)
            return (optax.apply_updates(params,updates),state),value
        return jax.lax.scan(step,(par,opt),(draws,masks))
    @jax.jit
    def predict(par,ii,jj):return net.apply(par,gather(ii),c[ii],e[jj])
    def infer(par,rows):
        ii=np.repeat(rows,16);jj=np.tile(np.arange(16),len(rows));a=[]
        for start in range(0,len(ii),256):a.append(np.asarray(predict(par,jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
        return np.concatenate(a).reshape(-1,16)
    rng=np.random.default_rng(job['seed']);rr,jj=np.where(d['success'][tr]+d['failure'][tr]>0);pool=np.column_stack((tr[rr],jj))
    draws=dict(np.load(HERE/f"draws_seed{job['seed']}.npz"));best=None;global_best=None;hist=[];t0=time.monotonic()
    for step in range(200,steps+1,200):
        if dense:dd=pool[rng.integers(len(pool),size=(200,128))];mm=np.ones(dd.shape[:2],np.float32)
        else:
            dd=draws['sparse' if sparse else 'joint'][step-200:step];mm=draws['mask'][step-200:step] if sparse else np.ones(dd.shape[:2],np.float32)
        (par,opt),loss=block(par,opt,jnp.asarray(dd),jnp.asarray(mm))
        z=infer(par,va);m=metrics(z,d,va);row=dict(step=step,training_nll=float(np.mean(loss)),**m);hist.append(row)
        key=(m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)
        if best is None or key<best:best=key;chosen=row;best_bytes=serialization.to_bytes(par)
        if job['arm']=='joint_replay':
            gm=metrics(infer(par,global_va),d,global_va);gkey=(gm['regret'],-gm['ranking'],gm['state_eta_MSE'],gm['nll'],step)
            if global_best is None or gkey<global_best:global_best=gkey;global_bytes=serialization.to_bytes(par);global_step=step
        if step%1000==0:write(out/'history.json',hist);print(dict(**job,step=step,best=chosen['step'],Q_MSE=m['Q_MSE'],seconds=time.monotonic()-t0),flush=True)
    (out/'checkpoint.msgpack').write_bytes(best_bytes);(out/'final_checkpoint.msgpack').write_bytes(serialization.to_bytes(par))
    replay=None
    if job['arm']=='joint_replay':
        (out/'original_global_checkpoint.msgpack').write_bytes(global_bytes);replay=sha(out/'original_global_checkpoint.msgpack')==sha(parent/'checkpoint.msgpack')
        write(out/'replay_integrity.json',dict(byte_identical=replay,global_step=global_step,old_step=read(parent/'complete.json')['selected']['step']))
        assert replay,'Original Stage17 checkpoint replay is not byte-identical; causal comparison blocked until resolved'
    selected=serialization.from_bytes(par,best_bytes);ix=np.flatnonzero(native)
    zs=infer(selected,ix);zf=infer(par,ix)
    np.savez_compressed(out/'native_train_val_predictions.npz',state_uid=d['state_uid'][ix],union_rows=ix,
        probability=1/(1+np.exp(-np.clip(zs,-80,80))),final_probability=1/(1+np.exp(-np.clip(zf,-80,80))))
    write(out/'history.json',hist);write(out/'complete.json',dict(**job,selected=chosen,source_train=metrics(infer(selected,tr),d,tr),seconds=time.monotonic()-t0,
        checkpoint_sha256=sha(out/'checkpoint.msgpack'),final_checkpoint_sha256=sha(out/'final_checkpoint.msgpack'),replay_byte_identical=replay,new_rollouts=0,test_used_for_selection=False))


def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    old=np.load(OLD/'dataset.npz');states=read(BASE.parent/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')['states']
    rt=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    def forbidden(*args,**kwargs):raise RuntimeError('Physical Env.step forbidden')
    rt.env_cls.step=forbidden
    selected=[s for s,k in zip(states,old['split']) if k=='test'];count=0
    for s in selected[index::workers]:
        path=HERE/'features'/s['state_uid']/'flow.npz'
        if path.exists():continue
        path.parent.mkdir(parents=True,exist_ok=True);nominal=controller_context(rt,s);field=field_context(rt,s)
        np.savez_compressed(path,nominal=nominal,field=field);count+=1
    write(HERE/'feature_jobs'/f'{index}.json',dict(created=count,physical_steps=0,new_rollouts=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit','features']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='fit':fit(a.index)
    else:features(a.index,a.workers)
