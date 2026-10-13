"""Frozen pure-physical critics and independent-state learning curves."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from diagnostics.state_nuisance_rootcause_v1.real_state_expansion import OUT, SOURCE, read, write, freeze, sha
from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import state_effect, model

ARMS = ('local92_eta', 'physical23_eta', 'physical23_heads')
SUPPLEMENTARY = ('physical25_heads',)
SEEDS = (17, 23, 41)


def csvout(path, rows):
    with Path(path).open('w') as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def features():
    from ring_exchange.local_frame import vectors_to_local, local_observation
    from types import SimpleNamespace
    p = read(OUT / 'protocol.json')
    cfg = SimpleNamespace(**p['environment']['config']['config'])
    raw92, raw23, orientation, errors, full_errors = [], [], [], [], []
    wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
    for s in p['states']:
        v = s['physical']; assert v['timestep'] == 0
        pos, vel, goal = [np.asarray(v[k]) for k in ('positions', 'velocities', 'goals')]
        obs = local_observation(pos, vel, goal, cfg).ravel()
        theta = np.arctan2(pos[:, 1], pos[:, 0]); gt = np.arctan2(goal[:, 1], goal[:, 0])
        orientation.append([np.cos(theta[0]),np.sin(theta[0])])
        rp, rg = np.linalg.norm(pos, axis=1), np.linalg.norm(goal, axis=1)
        rel = wrap(theta[1:] - theta[0] - np.arange(1,4)*np.pi/2)
        gd = wrap(gt-theta-np.pi); vl = vectors_to_local(vel, pos)
        raw92.append(obs); raw23.append(np.r_[rp, rel, rg, gd, vl.ravel()])
        th = np.r_[0., rel + np.arange(1,4)*np.pi/2]
        radial = np.c_[np.cos(th), np.sin(th)]; tangent = np.c_[-np.sin(th), np.cos(th)]
        pp = rp[:,None]*radial; gg = rg[:,None]*np.c_[np.cos(th+np.pi+gd), np.sin(th+np.pi+gd)]
        vv = vl[:,:1]*radial + vl[:,1:]*tangent
        err = float(np.max(abs(obs-local_observation(pp,vv,gg,cfg).ravel())))
        assert err < 2e-6; errors.append(err)
        rotation=np.array([[np.cos(theta[0]),-np.sin(theta[0])],[np.sin(theta[0]),np.cos(theta[0])]])
        full_error=max(float(np.max(abs(a@rotation.T-b))) for a,b in ((pp,pos),(vv,vel),(gg,goal)))
        assert full_error<1e-12;full_errors.append(full_error)
    raw92 = np.asarray(raw92, np.float32); raw23 = np.asarray(raw23, np.float32)
    old92 = np.load(HERE/'state_only/inputs.npz')
    old23 = np.load(HERE/'state_only/minimal_physical/inputs.npz')
    ix = read(HERE/'state_only/protocol.json')['splits']['train']
    np.testing.assert_array_equal(raw92[:32], old92['local_raw'][ix])
    np.testing.assert_array_equal(raw23[:32], old23['raw'][ix])
    local = (raw92-old92['local_center'])/old92['local_scale']
    local[:,~old92['local_active']] = 0
    physical = (raw23-old23['center'])/old23['scale']
    orientation=np.asarray(orientation,np.float32)
    orientation_center=orientation[:32].mean(0);orientation_scale=np.maximum(orientation[:32].std(0),.01)
    physical25=np.c_[physical,(orientation-orientation_center)/orientation_scale]
    serialized_errors=[]
    for i,state in enumerate(p['states']):
        values=(physical25[i,:23]*old23['scale']+old23['center']).astype(float)
        direction=(physical25[i,23:]*orientation_scale+orientation_center).astype(float)
        angle=np.arctan2(direction[1],direction[0])
        theta=angle+np.r_[0.,values[4:7]+np.arange(1,4)*np.pi/2]
        radial=np.c_[np.cos(theta),np.sin(theta)];tangent=np.c_[-np.sin(theta),np.cos(theta)]
        pp=values[:4,None]*radial
        gg=values[7:11,None]*np.c_[np.cos(theta+np.pi+values[11:15]),np.sin(theta+np.pi+values[11:15])]
        vl=values[15:].reshape(4,2);vv=vl[:,:1]*radial+vl[:,1:]*tangent
        error=max(float(np.max(abs(a-np.asarray(state['physical'][key])))) for a,key in ((pp,'positions'),(vv,'velocities'),(gg,'goals')))
        assert error<2e-6;serialized_errors.append(error)
    eta = np.load(SOURCE/'dataset.npz')['eta'][ix[0]]
    np.savez_compressed(OUT/'physical_inputs.npz', local92=local, physical23=physical, physical25=physical25,
        raw_local92=raw92, raw_physical23=raw23, eta=eta,
        raw_orientation=orientation,orientation_center=orientation_center,orientation_scale=orientation_scale,
        local_center=old92['local_center'], local_scale=old92['local_scale'],
        physical_center=old23['center'], physical_scale=old23['scale'])
    audit = dict(states=len(p['states']), raw23_reconstructs_native92_max_error=max(errors),
        physical25_ideal_reconstruction_max_error=max(full_errors),
        physical25_serialized_float32_reconstruction_max_error=max(serialized_errors),
        original32_feature_parity=True, constant92_dimensions=int((~old92['local_active']).sum()),
        normalization='frozen original32 TRAIN statistics, reused identically at every prefix',
        controller_inputs=False, feature_extraction_queries_controller=False,
        identifiers_in_model=False, inputs_sha256=sha(OUT/'physical_inputs.npz'))
    write(OUT/'feature_audit.json', audit)
    print(json.dumps(audit), flush=True)


def fit():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    p = read(OUT/'protocol.json'); inp = dict(np.load(OUT/'physical_inputs.npz'))
    data = dict(np.load(OUT/'source_dataset.npz')); cfg = p['training']
    code_sources={}
    archive=OUT/'code_snapshots';archive.mkdir(exist_ok=True)
    for path in (Path(__file__).resolve(),HERE/'state_only_experiment.py',HERE/'real_state_expansion.py'):
        digest=sha(path);target=archive/(path.stem+'__'+digest[:16]+'.py')
        if target.exists():assert sha(target)==digest
        else:target.write_bytes(path.read_bytes())
        code_sources[str(path)]=dict(sha256=digest,archived_path=str(target))
    producer_hash=code_sources[str(Path(__file__).resolve())]['sha256']
    assert np.all(data['present'][data['split']=='train',:,:4])
    assert np.all(data['present'][data['split']=='validation'])
    assert not data['present'][data['split']=='test'].any(), 'Fresh TEST labels entered fitting data'
    val = np.flatnonzero(data['split']=='validation')
    ss, ff = data['success'].astype(np.float32), data['failure'].astype(np.float32)
    sj, fj, etaj = jnp.asarray(ss), jnp.asarray(ff), jnp.asarray(inp['eta'])
    every, maximum = cfg['checkpoints_every'], cfg['max_steps']
    checkpoint_schedule=read(OUT/'checkpoint_schedule_amendment.json')['checkpoints']
    assert checkpoint_schedule==sorted(set([50,100]+list(range(every,maximum+1,every))))
    completed = []
    configurations = [(a, n) for a in ARMS for n in p['train_prefixes']] + [('physical23_heads_shuffled',512)] + [('physical25_heads',n) for n in (32,512)]
    for arm in ARMS + ('physical23_heads_shuffled',) + SUPPLEMENTARY:
        heads = 'heads' in arm
        net = model('local_state_16heads' if heads else 'local_state_eta')
        x = inp['local92' if arm.startswith('local92') else 'physical25' if arm.startswith('physical25') else 'physical23']
        xj = jnp.asarray(x)
        opt = optax.chain(optax.clip_by_global_norm(cfg['clip']), optax.adamw(cfg['lr'], weight_decay=cfg['weight_decay']))
        @jax.jit
        def block(params, optim, all_ii, all_jj, donor, nmean):
            def one(carry, indices):
                pp, oo = carry; ii, jj = indices
                def loss(pp):
                    zz = net.apply(pp, xj[donor[ii]], etaj[jj], jnp.zeros((len(ii),76), jnp.float32))
                    z = jnp.take_along_axis(zz,jj[:,None],1)[:,0] if heads else zz
                    return jnp.mean(sj[ii,jj]*jax.nn.softplus(-z)+fj[ii,jj]*jax.nn.softplus(z))/nmean
                loss0, grad = jax.value_and_grad(loss)(pp)
                up, oo = opt.update(grad,oo,pp)
                return (optax.apply_updates(pp,up),oo),loss0
            return jax.lax.scan(one,(params,optim),(all_ii,all_jj))
        @jax.jit
        def predict(pp, xx):
            if heads:
                return net.apply(pp, xx, jnp.zeros((len(xx),3)), jnp.zeros((len(xx),76)))
            ii = jnp.repeat(jnp.arange(len(xx)),16); jj = jnp.tile(jnp.arange(16),len(xx))
            return net.apply(pp,xx[ii],etaj[jj],jnp.zeros((len(ii),76))).reshape(len(xx),16)
        for _, n in [v for v in configurations if v[0]==arm]:
            folder = OUT/'fits'/f'{arm}_n{n}'
            folder.mkdir(parents=True, exist_ok=True)
            donor = np.arange(len(x))
            if arm.endswith('_shuffled'):
                donor[:n] = np.random.default_rng(202610071031).permutation(n)
            donorj = jnp.asarray(donor)
            pairs = np.array([(i,j) for i in range(n) for j in range(16) if ss[i,j]+ff[i,j]>0])
            nmean = float(np.mean(ss[pairs[:,0],pairs[:,1]]+ff[pairs[:,0],pairs[:,1]]))
            for seed in SEEDS:
                dest = folder/f'seed{seed}'; dest.mkdir(exist_ok=True)
                if (dest/'complete.json').exists():
                    result=read(dest/'complete.json')
                    assert result['data_sha256']==sha(OUT/'source_dataset.npz')
                    assert result['inputs_sha256']==sha(OUT/'physical_inputs.npz')
                    completed.append(result); continue
                start=time.monotonic(); rng=np.random.default_rng(seed)
                pp=net.init(jax.random.PRNGKey(seed),xj[:1],jnp.zeros((1,3)),jnp.zeros((1,76)))
                initial=hashlib.sha256(serialization.to_bytes(pp)).hexdigest(); oo=opt.init(pp)
                history=[];last_step=0
                for step0 in checkpoint_schedule:
                    draw=rng.integers(len(pairs),size=(step0-last_step,cfg['batch_size']))
                    ii,jj=pairs[draw].transpose(2,0,1)
                    (pp,oo),losses=block(pp,oo,jnp.asarray(ii),jnp.asarray(jj),donorj,jnp.asarray(nmean))
                    ztr=np.asarray(predict(pp,xj[donor[:n]])); zv=np.asarray(predict(pp,xj[val]))
                    def nll(z,s,f): return float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum())
                    history.append(dict(step=step0,train_NLL=nll(ztr,ss[:n],ff[:n]),
                        validation_NLL=nll(zv,ss[val],ff[val]),
                        last_batch_NLL=float(np.asarray(losses)[-1])))
                    (dest/f'step{step0}.msgpack').write_bytes(serialization.to_bytes(pp))
                    np.savez_compressed(dest/f'step{step0}_source.npz',train=ztr,validation=zv)
                    last_step=step0
                result=dict(arm=arm,n=n,seed=seed,initial_sha256=initial,history=history,
                    elapsed_seconds=time.monotonic()-start, data_sha256=sha(OUT/'source_dataset.npz'),
                    inputs_sha256=sha(OUT/'physical_inputs.npz'),controller_inputs=False)
                result['producer_code_sha256']=producer_hash
                write(dest/'complete.json',result);completed.append(result)
                print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    selections={}
    for arm,n in configurations:
        rr=[r for r in completed if r['arm']==arm and r['n']==n];assert len(rr)==3
        mean=np.mean([[h['validation_NLL'] for h in r['history']] for r in rr],0)
        chosen=int(np.argmin(mean));step=rr[0]['history'][chosen]['step']
        selections[f'{arm}_n{n}']=dict(arm=arm,n=n,selected_step=step,mean_VAL_NLL=mean.tolist(),
            checkpoint_steps=[h['step'] for h in rr[0]['history']],
            evaluation_checkpoints=[dict(seed=r['seed'],step=st,
                path=str(OUT/'fits'/f'{arm}_n{n}'/f'seed{r["seed"]}'/f'step{st}.msgpack'),
                sha256=sha(OUT/'fits'/f'{arm}_n{n}'/f'seed{r["seed"]}'/f'step{st}.msgpack'))
                for r in rr for st in sorted({step,2000,maximum})],
            models=[dict(seed=r['seed'],path=str(OUT/'fits'/f'{arm}_n{n}'/f'seed{r["seed"]}'/f'step{step}.msgpack'),
                sha256=sha(OUT/'fits'/f'{arm}_n{n}'/f'seed{r["seed"]}'/f'step{step}.msgpack')) for r in rr])
    # Every independent TEST choice is now fixed, including secondary checkpoints.
    freeze(OUT/'models_frozen.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        source_data_sha256=sha(OUT/'source_dataset.npz'),inputs_sha256=sha(OUT/'physical_inputs.npz'),
        orientation_amendment_sha256=sha(OUT/'orientation_control_amendment.json'),
        checkpoint_schedule_amendment_sha256=sha(OUT/'checkpoint_schedule_amendment.json'),
        fitting_code_sha256=producer_hash,code_sources=code_sources,selections=selections,
        software=dict(python=sys.version,numpy=np.__version__,jax=jax.__version__,optax=optax.__version__,devices=[str(d) for d in jax.devices()]),
        selection='mean3-seed32-state validation count NLL only',
        secondary_checkpoints=[2000,maximum],test_labels_opened=False,
        eta_only_baseline='per-eta TRAIN empirical success rate, clipped1e-6 only for log scoring'))
    print(json.dumps(dict(models_frozen=True,configurations=len(selections),fits=len(completed))),flush=True)


def calculate(z,s,f,k=16,include_pairs=True):
    from scipy.special import expit
    z=np.asarray(z,float);s=np.asarray(s,float);f=np.asarray(f,float)
    valid=s+f>0;u=k-s-f;pp=expit(z)
    m=dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum()),**state_effect(pp,s,f))
    qq=np.divide(s,s+f,out=np.zeros_like(s),where=valid)
    # Residual after both per-eta and per-state means, on complete cells.
    q0=qq-qq.mean(0,keepdims=True)-qq.mean(1,keepdims=True)+qq.mean()
    p0=pp-pp.mean(0,keepdims=True)-pp.mean(1,keepdims=True)+pp.mean()
    m['interaction_rho']=float(np.corrcoef(q0[valid],p0[valid])[0,1]) if p0[valid].std()>1e-9 else None
    if include_pairs:
        i,j=np.triu_indices(len(s),1)
        pos=s[i]/k-(s[j]+u[j])/k>=.25;neg=s[j]/k-(s[i]+u[i])/k>=.25
        eligible=pos|neg;diff=z[i]-z[j]
        correct=(diff>0)*pos+(diff<0)*neg+.5*(diff==0)*eligible
        m['same_eta_state_change_accuracy']=float(correct[eligible].mean()) if eligible.any() else None
        m['state_pairs_eta_cases']=int(eligible.sum())
        a,b=np.triu_indices(s.shape[1],1);ii,jj=i[:,None],j[:,None];aa,bb=a[None,:],b[None,:]
        lo,hi=s/k,(s+u)/k
        forward=(lo[ii,aa]-hi[ii,bb]>=.25)&(lo[jj,bb]-hi[jj,aa]>=.25)
        reverse=(lo[ii,bb]-hi[ii,aa]>=.25)&(lo[jj,aa]-hi[jj,bb]>=.25)
        reversals=forward|reverse;d1=z[ii,aa]-z[ii,bb];d2=z[jj,bb]-z[jj,aa]
        both=((d1>0)&(d2>0)&forward)|((d1<0)&(d2<0)&reverse)
        m['eligible_reversal_cases']=int(reversals.sum())
        m['reversal_both_correct']=float(both[reversals].mean()) if reversals.any() else None
    choice=z.argmax(1);row=np.arange(len(s));sc,fc=s[row,choice],f[row,choice]
    m.update(top1_success_lower=float(np.mean(sc/k)),top1_success_upper=float(np.mean((sc+k-sc-fc)/k)),
             B15=int((sc>=15).sum()) if k==16 else None,
             B15_unknown=int(((sc<15)&(fc<2)).sum()) if k==16 else None)
    return m


def evaluate():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from scipy.special import logit
    frozen=read(OUT/'models_frozen.json');data=dict(np.load(OUT/'test_dataset.npz'))
    inp=dict(np.load(OUT/'physical_inputs.npz'))
    assert frozen['inputs_sha256']==sha(OUT/'physical_inputs.npz')
    ss,ff=data['success'],data['failure'];va=np.flatnonzero(data['split']=='validation')
    indices=dict(validation=va,
        original_test=np.flatnonzero((data['split']=='test')&(data['origin']=='original')),
        fresh_test=np.flatnonzero((data['split']=='test')&(data['origin']=='fresh')))
    rows=[];logits={}
    for name,sel in frozen['selections'].items():
        arm,n=sel['arm'],sel['n'];heads='heads' in arm
        net=model('local_state_16heads' if heads else 'local_state_eta')
        x=inp['local92' if arm.startswith('local92') else 'physical25' if arm.startswith('physical25') else 'physical23']
        template=net.init(jax.random.PRNGKey(17),jnp.asarray(x[:1]),jnp.zeros((1,3)),jnp.zeros((1,76)))
        @jax.jit
        def predict(pp,xx):
            if heads:return net.apply(pp,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76)))
            ii=jnp.repeat(jnp.arange(len(xx)),16);jj=jnp.tile(jnp.arange(16),len(xx))
            return net.apply(pp,xx[ii],jnp.asarray(inp['eta'])[jj],jnp.zeros((len(ii),76))).reshape(len(xx),16)
        for step0 in sorted(set([sel['selected_step']]+frozen['secondary_checkpoints'])):
            for seed in SEEDS:
                path=OUT/'fits'/name/f'seed{seed}'/f'step{step0}.msgpack'
                exact=next(m for m in sel['evaluation_checkpoints'] if m['seed']==seed and m['step']==step0)
                assert sha(path)==exact['sha256']
                if step0==sel['selected_step']:
                    expected=next(m for m in sel['models'] if m['seed']==seed)['sha256'];assert sha(path)==expected
                pp=serialization.from_bytes(template,path.read_bytes())
                for split,ix in indices.items():
                    z=np.asarray(predict(pp,jnp.asarray(x[ix])))
                    logits[f'{name}__{seed}__{step0}__{split}']=z
                    rows.append(dict(arm=arm,n=n,seed=seed,step=step0,selection='VAL_NLL' if step0==sel['selected_step'] else 'fixed_step',
                        split=split,states=len(ix),**calculate(z,ss[ix],ff[ix])))
                ztr=np.load(OUT/'fits'/name/f'seed{seed}'/f'step{step0}_source.npz')['train']
                rows.append(dict(arm=arm,n=n,seed=seed,step=step0,selection='VAL_NLL' if step0==sel['selected_step'] else 'fixed_step',
                    split='train',states=n,**calculate(ztr,ss[:n],ff[:n],4,include_pairs=False)))
        print('evaluated',name,flush=True)
    for n in read(OUT/'protocol.json')['train_prefixes']:
        rate=ss[:n].sum(0)/(ss[:n]+ff[:n]).sum(0)
        for split,ix in indices.items():
            z=np.broadcast_to(logit(np.clip(rate,1e-6,1-1e-6)),(len(ix),16))
            logits[f'eta_only_n{n}__{split}']=z
            rows.append(dict(arm='eta_only',n=n,seed='analytic',step=0,selection='TRAIN_prior',split=split,
                        states=len(ix),**calculate(z,ss[ix],ff[ix])))
    np.savez_compressed(OUT/'evaluation_logits.npz',**logits)
    csvout(OUT/'metrics.csv',rows)
    write(OUT/'evaluation_audit.json',dict(fresh_test_states=len(indices['fresh_test']),original_test_states=len(indices['original_test']),
        model_seeds=list(SEEDS),test_data_sha256=sha(OUT/'test_dataset.npz'),frozen_models_sha256=sha(OUT/'models_frozen.json'),
        post_TEST_model_selection=False,rows=len(rows)))


def analyze():
    from scipy.special import expit
    frozen=read(OUT/'models_frozen.json');data=np.load(OUT/'test_dataset.npz');z=np.load(OUT/'evaluation_logits.npz')
    ss,ff=data['success'],data['failure'];ix=np.flatnonzero((data['split']=='test')&(data['origin']=='fresh'))
    s,f=ss[ix],ff[ix]
    with (OUT/'metrics.csv').open() as h:rows=list(csv.DictReader(h))
    keys=('NLL','state_effect_rho','state_effect_R2','interaction_rho','same_eta_state_change_accuracy','reversal_both_correct','top1_success_lower','B15')
    summary=[]
    for name,sel in frozen['selections'].items():
        for kind,st in [('VAL_NLL',sel['selected_step']),('fixed2000',2000),('last12000',12000)]:
            for split in ('train','validation','original_test','fresh_test'):
                rr=[r for r in rows if r['arm']==sel['arm'] and int(r['n'])==sel['n'] and int(r['step'])==st and r['split']==split]
                assert len(rr)==3
                summary.append(dict(arm=sel['arm'],n=sel['n'],rule=kind,step=st,split=split,
                    **{k:float(np.mean([float(r[k]) for r in rr])) if all(r.get(k) for r in rr) else None for k in keys}))
    summary.extend(dict(arm=r['arm'],n=int(r['n']),rule='TRAIN_prior',step=0,split=r['split'],
                        **{k:float(r[k]) if r.get(k) else None for k in keys}) for r in rows if r['arm']=='eta_only')
    csvout(OUT/'learning_curve.csv',summary)
    rng=np.random.default_rng(202610071049);draws=rng.integers(len(ix),size=(1000,len(ix)))
    ci=[]
    for arm in ARMS:
        for n in (32,64,128,256,512):
            name=f'{arm}_n{n}';st=frozen['selections'][name]['selected_step']
            pred=np.stack([expit(z[f'{name}__{seed}__{st}__fresh_test']) for seed in SEEDS])
            refname=f'{arm}_n32';refst=frozen['selections'][refname]['selected_step']
            ref=np.stack([expit(z[f'{refname}__{seed}__{refst}__fresh_test']) for seed in SEEDS])
            previous_n=max(32,n//2);previous_name=f'{arm}_n{previous_n}'
            previous_step=frozen['selections'][previous_name]['selected_step']
            previous=np.stack([expit(z[f'{previous_name}__{seed}__{previous_step}__fresh_test']) for seed in SEEDS])
            zz=np.stack([z[f'{name}__{seed}__{st}__fresh_test'] for seed in SEEDS])
            prior=z[f'eta_only_n{n}__fresh_test']
            loss=(s[None]*np.logaddexp(0,-zz)+f[None]*np.logaddexp(0,zz)).mean(0)
            priorloss=s*np.logaddexp(0,-prior)+f*np.logaddexp(0,prior)
            br=[];bd=[];bn=[];bp=[]
            for d in draws:
                a=np.mean([state_effect(v[d],s[d],f[d])['state_effect_rho'] for v in pred])
                b=np.mean([state_effect(v[d],s[d],f[d])['state_effect_rho'] for v in ref])
                pv=np.mean([state_effect(v[d],s[d],f[d])['state_effect_rho'] for v in previous])
                br.append(a);bd.append(a-b);bn.append(float((loss[d]-priorloss[d]).sum()/(s[d]+f[d]).sum()))
                bp.append(a-pv)
            ci.append(dict(arm=arm,n=n,step=st,physical_test_states=len(ix),bootstrap_replicates=len(draws),
                state_rho_95=np.quantile(br,[.025,.975]).tolist(),
                delta_state_rho_vs32_95=np.quantile(bd,[.025,.975]).tolist(),
                previous_n=previous_n,delta_state_rho_vs_previous_size_95=np.quantile(bp,[.025,.975]).tolist(),
                delta_NLL_vs_eta_only_95=np.quantile(bn,[.025,.975]).tolist()))
    write(OUT/'physical_state_bootstrap.json',dict(method='paired resampling of48 fresh physical states; average the3 fits within each draw',rows=ci))
    # Independent seeds estimate label repeatability on the untouched state set.
    outcomes=data['outcomes'][ix];halves=[]
    for a,b in ((0,8),(8,16)):
        valid=outcomes[:,:,a:b,1]==0
        suc=((outcomes[:,:,a:b,0]==1)&valid).sum(-1)
        halves.append((suc,valid.sum(-1)-suc))
    s0,f0=halves[0];s1,f1=halves[1]
    q0=np.divide(s0,s0+f0,out=np.zeros_like(s0,dtype=float),where=s0+f0>0)
    shared=(s0+f0>0)&(s1+f1>0)
    reliability=state_effect(q0,np.where(shared,s1,0),np.where(shared,f1,0))
    write(OUT/'fresh_test_label_repeatability.json',dict(seed_split='0:8 versus8:16',**reliability))
    # Direct data correspondence, independent of any learned critic.
    inputs=np.load(OUT/'physical_inputs.npz');nearest=[]
    qtest=np.divide(s,s+f,out=np.zeros_like(s,dtype=float),where=s+f>0)
    qtrain=np.divide(ss[:512],ss[:512]+ff[:512],out=np.full_like(ss[:512],np.nan,dtype=float),where=ss[:512]+ff[:512]>0)
    for representation in ('physical23','physical25','local92'):
        x=inputs[representation]
        for n in (32,64,128,256,512):
            distance=((x[ix,None,:]-x[None,:n,:])**2).mean(-1)
            donor=distance.argmin(1)
            shared=np.isfinite(qtrain[donor])&(s+f>0)
            score=state_effect(np.nan_to_num(qtrain[donor]),np.where(shared,s,0),np.where(shared,f,0))
            near_rmse=float(np.sqrt(np.mean((qtrain[donor]-qtest)[shared]**2)))
            all_diff=qtrain[None,:n,:]-qtest[:,None,:]
            all_diff=np.where((s+f>0)[:,None,:],all_diff,np.nan)
            random_rmse=float(np.sqrt(np.nanmean(all_diff**2)))
            nearest.append(dict(representation=representation,n=n,nearest_state_rho=score['state_effect_rho'],
                nearest_Q_RMSE=near_rmse,random_train_Q_RMSE=random_rmse,nearest_distance=float(np.sqrt(distance.min(1)).mean())))
    source_outcomes=data['outcomes'][:512];source_half=[]
    for a,b in ((0,2),(2,4)):
        valid=source_outcomes[:,:,a:b,1]==0
        sc=((source_outcomes[:,:,a:b,0]==1)&valid).sum(-1)
        source_half.append((sc,valid.sum(-1)-sc))
    sa,fa=source_half[0];sb,fb=source_half[1]
    qa=np.divide(sa,sa+fa,out=np.zeros_like(sa,dtype=float),where=sa+fa>0)
    shared=(sa+fa>0)&(sb+fb>0)
    source_repeat=state_effect(qa,np.where(shared,sb,0),np.where(shared,fb,0))
    eta_prior=ss[:512].sum(0)/(ss[:512]+ff[:512]).sum(0);best_fixed=int(eta_prior.argmax())
    headroom=dict(fixed_eta_chosen_by_TRAIN512=best_fixed,fresh_test_states=len(ix),
        fixed_eta_confirmed_B15=int((s[:,best_fixed]>=15).sum()),
        any_eta_confirmed_B15=int((s>=15).any(1).sum()),
        fixed_eta_mean_success_lower=float((s[:,best_fixed]/16).mean()),
        empirical_oracle_mean_success_lower=float((s/16).max(1).mean()),
        interpretation='Descriptive finite16-seed oracle on the fixed bank; not a policy selected from TEST')
    write(OUT/'data_only_correspondence.json',dict(nearest_neighbors=nearest,train512_two_seed_halves=source_repeat,eta_selection_headroom=headroom,
        caveat='Nearest-neighbor results depend on this physical metric and noisyQ4 labels; not a proof of global learnability or unlearnability.'))
    complete_cells=s+f==16;sensitivity=[]
    for name,sel in frozen['selections'].items():
        st=sel['selected_step'];rr=[]
        for seed in SEEDS:
            zz=z[f'{name}__{seed}__{st}__fresh_test']
            mm=calculate(zz,np.where(complete_cells,s,0),np.where(complete_cells,f,0),include_pairs=False)
            rr.append(mm)
        sensitivity.append(dict(arm=sel['arm'],n=sel['n'],complete_seed_cells=int(complete_cells.sum()),
            state_effect_rho=float(np.mean([r['state_effect_rho'] for r in rr])),NLL=float(np.mean([r['NLL'] for r in rr]))))
    write(OUT/'complete_seed_cell_sensitivity.json',dict(rule='Exclude an entire TEST state-eta cell if any seed is numerical unknown',rows=sensitivity))
    write(OUT/'research_results.json',dict(results=summary,bootstrap=ci,label_repeatability=reliability,
        research_computation_complete=True,formal_batch_delivery_complete=False,
        video_gate='Run mandatory four-scene generator; absent documented deadlock cases are a delivery gap, never synthetic substitutes'))
    print(json.dumps([r for r in summary if r['split']=='fresh_test' and r['rule'] in ('VAL_NLL','TRAIN_prior')],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('features','fit','evaluate','analyze'))
    globals()[ap.parse_args().action]()
