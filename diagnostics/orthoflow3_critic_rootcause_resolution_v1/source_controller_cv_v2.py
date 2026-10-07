"""Same model/data: source-family versus source-controller validation unit.

No target records or target predictions are used by this experiment.
"""
import argparse, collections, hashlib, os, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from .db_transfer_data import OUT as DATA, ROOT, SCENES, read, write, sha
from .db_transfer_train import model, SEEDS
from .controller_validation import folds

OUT = ROOT / 'source_controller_cv_v2'
KINDS = ('eta_only', 'full_context')
STEPS = tuple(range(100, 2001, 100))


def dependencies(controller):
    program = controller['config'].get('controller_program')
    if not program:
        return {controller['sha256']}
    result = set()
    def visit(obj):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                result.add(obj['sha256'])
            for value in obj.values(): visit(value)
        elif isinstance(obj, list):
            for value in obj: visit(value)
    visit(program)
    assert result, 'Unrecognized controller program: must audit parents'
    return result


def normalized(rows, fit):
    raw = np.load(DATA / 'source_context.npz')['context']
    eta = np.array([r['eta'] for r in rows], np.float32)
    w = np.zeros(len(rows))
    for scene in SCENES:
        idx = np.array([i for i in fit if rows[i]['scenario'] == scene])
        assert len(idx)
        w[idx] = 1 / len(idx) / len(SCENES)
    ec = (w[:, None] * eta).sum(0)
    es = np.maximum(np.sqrt((w[:, None] * (eta - ec)**2).sum(0)), .1)
    cc, cs = np.zeros(76), np.ones(76)
    for a, b, flag, floor in ((0,24,73,.05),(24,40,73,.01),(40,56,74,.05),(56,72,75,.05)):
        ww = w * (raw[:, flag] > 0); ww /= ww.sum()
        cc[a:b] = (ww[:, None] * raw[:, a:b]).sum(0)
        cs[a:b] = np.maximum(np.sqrt((ww[:, None] * (raw[:, a:b]-cc[a:b])**2).sum(0)), floor)
    c = (raw-cc)/cs
    for a,b,flag in ((0,40,73),(40,56,74),(56,72,75)):
        c[:,a:b] *= raw[:,flag,None] > 0
    norm = dict(eta_center=ec.tolist(), eta_scale=es.tolist(), context_center=cc.tolist(), context_scale=cs.tolist(), fitting_rows=fit.tolist(), source_only=True)
    return ((eta-ec)/es).astype(np.float32), c.astype(np.float32), norm


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Protocol frozen'
    rows = read(DATA/'pairs.json'); controllers = read(DATA/'controllers.json')
    profiles = read(ROOT/'controller_function_support/protocol.json')['profiles']
    docs = []
    for fold, held in enumerate(folds()):
        hashes = {profiles[j]['sha256'] for j in held}
        excluded = {uid for uid, ctl in controllers.items() if dependencies(ctl) & hashes}
        native = {uid for uid in excluded if not controllers[uid]['config'].get('controller_program')}
        fit = np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['controller_uid'] not in excluded])
        seen = np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r['controller_uid'] not in excluded])
        heldval = np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r['controller_uid'] in native])
        assert len(fit) and len(seen) and len(heldval)
        assert not ({rows[i]['family'] for i in fit} & {rows[i]['family'] for i in np.r_[seen,heldval]})
        assert all(not (dependencies(controllers[rows[i]['controller_uid']]) & hashes) for i in fit)
        e,c,norm = normalized(rows,fit); dest=OUT/f'fold{fold}'
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'arrays.npz',eta=e,context=c,fit=fit,seen=seen,held=heldval)
        write(dest/'normalization.json',norm)
        docs.append(dict(fold=fold,held_profile_indices=held,held_hashes=sorted(hashes),excluded_controller_uids=sorted(excluded),native_validation_controller_uids=sorted(native),fit_pairs=len(fit),seen_val_pairs=len(seen),held_val_pairs=len(heldval),held_val_families=len({rows[i]['family'] for i in heldval}),normalization_sha256=sha(dest/'normalization.json')))
    write(OUT/'protocol.json',dict(hypothesis='Family-only early stopping can reward controller memorization. Hold future Flow functions and independent source families out simultaneously; keep architecture, optimizer, data universe and pure observed-count NLL fixed.',folds=docs,kinds=KINDS,seeds=SEEDS,steps=STEPS,selection='mean source-held-controller continuation NLL across three folds and seeds; one common step per model kind',seen_selection_control='same trajectories selected by source seen-controller family VAL',normalization='only fitting controllers and TRAIN families, scene balanced',held_parent_policy='exclude every controller program containing any held future Flow, regardless of controller UID',new_rollouts=0,target_used=False,model_code_sha256=sha(ROOT/'db_transfer_train.py'),code_sha256=sha(__file__),rows_sha256=sha(DATA/'pairs.json')))
    print(docs,flush=True)


def metrics(z, rows, idx):
    from scipy.special import expit
    s=np.array([rows[i]['s'] for i in idx]); f=np.array([rows[i]['f'] for i in idx]); n=s+f
    q=s/n; p=expit(z)
    groups=collections.defaultdict(list); ctl=collections.defaultdict(list)
    for loc,i in enumerate(idx):
        r=rows[i]; groups[r['controller_uid'],r['state_uid']].append(loc); ctl[r['controller_uid']].append(loc)
    nll=s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)
    b15=unknown=oracle=cases=0; contrast=[]
    for locs in groups.values():
        if len(locs)<2: continue
        cases+=1; oracle+=any(rows[idx[j]]['B15'] for j in locs)
        j=locs[np.argmax(z[locs])]; r=rows[idx[j]]
        b15+=bool(r['B15']); unknown+=not(r['B15'] or r['nonB15'])
        if len(locs)==2:
            locs=sorted(locs,key=lambda j:rows[idx[j]]['eta_uid'])
            a,b=locs; contrast.append((rows[idx[a]]['controller_uid'],q[a]-q[b],p[a]-p[b]))
    centered=[]
    for uid in sorted({r[0] for r in contrast}):
        a=np.array([r[1:] for r in contrast if r[0]==uid]); centered.extend(a-a.mean(0))
    a=np.array(centered)
    skill=float(1-np.mean((a[:,1]-a[:,0])**2)/np.mean(a[:,0]**2)) if len(a) and np.mean(a[:,0]**2)>1e-10 else None
    return dict(NLL=float(nll.sum()/n.sum()),controller_equal_NLL=float(np.mean([nll[j].sum()/n[j].sum() for j in ctl.values()])),MAE=float(abs(p-q).mean()),B15=int(b15),unresolved=int(unknown),oracle_B15=int(oracle),cases=cases,centered_state_contrast_skill=skill)


def run(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    fold=index//6;kind=KINDS[index%6//3];seed=SEEDS[index%3]
    protocol=read(OUT/'protocol.json')
    assert protocol['code_sha256']==sha(__file__) and protocol['model_code_sha256']==sha(ROOT/'db_transfer_train.py')
    dest=OUT/f'fold{fold}'/kind/f'seed{seed}'
    assert not (dest/'complete.json').exists(), 'Completed fit is frozen'
    rows=read(DATA/'pairs.json'); d=np.load(OUT/f'fold{fold}'/'arrays.npz')
    e,c=d['eta'],d['context'];fit,seen,held=d['fit'],d['seen'],d['held']
    x=dict(np.load(DATA/'source_entities.npz'));si=np.array([r['state_index'] for r in rows])
    s=np.array([r['s'] for r in rows],np.float32); f=np.array([r['f'] for r in rows],np.float32)
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
    initial=sha_bytes(serialization.to_bytes(p)); opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    flat=traverse_util.flatten_dict(p);warmmask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in flat})
    def step_fn(warm):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff,ww):
            def loss(p0):
                z=m.apply(p0,xx,ee,cc);return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))*ww)
            loss,gr=jax.value_and_grad(loss)(pp)
            if warm: gr=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),gr,warmmask)
            u,oo=opt.update(gr,oo,pp)
            if warm: u=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),u,warmmask)
            return optax.apply_updates(pp,u),oo,loss
        return step
    warm,step=step_fn(True),step_fn(False)
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    def score(pp,idx):
        scores=[]
        for start in range(0,len(idx),256):
            ii=idx[start:start+256];num=len(ii);ii=np.pad(ii,(0,256-num),mode='edge')
            scores.extend(np.asarray(predict(pp,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii])))[:num])
        return np.array(scores)
    groups=[]
    for scene in SCENES:
        for pool in ('historical_wide','intervention'):
            ii=np.array([i for i in fit if rows[i]['scenario']==scene and rows[i]['pool']==pool]);assert len(ii)
            groups.append((ii,1/float((s+f)[ii].mean())))
    dest.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);order=hashlib.sha256();history=[];scores={};start=time.perf_counter()
    for it in range(1,2001):
        ii=np.concatenate([rng.choice(ii,48) for ii,_ in groups]);ww=np.concatenate([np.full(48,w,np.float32) for _,w in groups]);order.update(ii.tobytes())
        p,o,loss=(warm if it<=25 else step)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ii]),jnp.asarray(s[ii]),jnp.asarray(f[ii]),jnp.asarray(ww))
        if it not in STEPS:continue
        zz=score(p,held);ss=score(p,seen)
        history.append(dict(step=it,train_objective=float(loss),held=metrics(zz,rows,held),seen=metrics(ss,rows,seen)))
        scores[f'held_step{it}']=zz;scores[f'seen_step{it}']=ss
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
    np.savez_compressed(dest/'predictions.npz',held_indices=held,seen_indices=seen,**scores)
    write(dest/'history.json',history)
    write(dest/'complete.json',dict(kind=kind,seed=seed,fold=fold,initial_sha256=initial,batch_order_sha256=order.hexdigest(),seconds=time.perf_counter()-start,backend=jax.default_backend(),code_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),new_rollouts=0,target_labels_used=False))
    print(dict(kind=kind,seed=seed,fold=fold,seconds=time.perf_counter()-start),flush=True)


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


def summarize():
    result=[];selection={}
    for fold in range(3):
        for seed in SEEDS:
            done=[read(OUT/f'fold{fold}'/kind/f'seed{seed}'/'complete.json') for kind in KINDS]
            assert len({d['batch_order_sha256'] for d in done})==1
    for kind in KINDS:
        allruns=[read(OUT/f'fold{fold}'/kind/f'seed{seed}'/'history.json') for fold in range(3) for seed in SEEDS]
        for j,step in enumerate(STEPS):
            for scope in ('seen','held'):
                rr=[a[j][scope] for a in allruns]
                result.append(dict(kind=kind,step=step,scope=scope,NLL=float(np.mean([r['controller_equal_NLL'] for r in rr])),B15_by_seed=[sum(rr[f*3+s]['B15'] for f in range(3)) for s in range(3)],oracle_per_seed=sum(rr[f*3]['oracle_B15'] for f in range(3)),cases_per_seed=sum(rr[f*3]['cases'] for f in range(3)),mean_state_contrast_skill=float(np.mean([r['centered_state_contrast_skill'] for r in rr if r['centered_state_contrast_skill'] is not None]))))
        selection[kind]={scope:min([r for r in result if r['kind']==kind and r['scope']==scope],key=lambda r:r['NLL']) for scope in ('seen','held')}
    write(OUT/'trajectories.json',result)
    write(OUT/'selection.json',dict(selection=selection,new_rollouts=0,target_used=False,scope='Source model selection, not independent confirmation; same-controller and unseen-controller validation contrasted on matched trajectories'))
    print(selection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    run(a.index) if a.action=='run' else globals()[a.action]()
