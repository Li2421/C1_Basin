"""Mechanism controls for the focused controller-free physical-state audit."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import OUT,HERE,SOURCE,SEEDS,read,write,sha,csvout,state_effect,metrics,model


def synthetic():
    """Known physical function, same state splits and4/16-trial supervision."""
    import jax
    import jax.numpy as jnp
    import optax
    from scipy.special import expit,logit
    from flax import serialization
    dest=OUT/'synthetic_control';dest.mkdir(exist_ok=True)
    p=read(OUT/'protocol.json');d=np.load(SOURCE/'dataset.npz');inputs=np.load(OUT/'inputs.npz')
    ix={k:np.array(v) for k,v in p['splits'].items()};tr,va=ix['train'],ix['validation']
    x=inputs['local'];eta=inputs['eta'][tr[0]]
    prior=(d['success'][1,tr].sum(0)+1)/(d['success'][1,tr].sum(0)+d['failure'][1,tr].sum(0)+2)
    # Four goal-displacement/tangent coordinates and two radii, all physical.
    a=x[:,5]-x[:,51];b=x[:,28]-x[:,74];c=x[:,0]-x[:,46]
    raw=(a[:,None]*eta[None,:,0]+b[:,None]*eta[None,:,1]+.5*c[:,None]*eta[None,:,2])
    scale=np.maximum(raw[tr].std(0),.1);effect=(raw-raw[tr].mean(0))/scale
    true_p=expit(logit(np.clip(prior,.02,.98))[None,:]+4*effect)
    actual=np.load(HERE/'state_signal_path/exact_outcomes.npz')['outcomes']
    valid=(actual[0,...,1]==0)&(actual[1,...,1]==0)
    uniforms=np.random.default_rng(2026100722).random(valid.shape)
    binary=uniforms<true_p[...,None]
    ss=np.zeros_like(d['success'][1]);ff=np.zeros_like(ss)
    for split,idx in ix.items():
        k=4 if split=='train' else 16
        vv=valid[idx,:,:k];ss[idx]=(binary[idx,:,:k]*vv).sum(-1);ff[idx]=vv.sum(-1)-ss[idx]
    write(dest/'protocol.json',dict(type='Synthetic pipeline positive control, not real rollout evidence',
        formula='logit(TRAIN eta prior)+4*standardized[(x5-x51)*eta0+(x28-x74)*eta1+0.5*(x0-x46)*eta2]',
        inputs='Exactly same pure physical local92 input as real state16head experiment',
        targets='Independent Bernoulli samples:4 TRAIN,16 VAL/TEST; same numerical validity masks; known conditional probabilities saved',
        normalization='TRAIN only',seeds=list(SEEDS),steps=2000,selection='Mean3-seed VAL NLL',
        purpose='Check that the same input/training pipeline generalizes a known state-dependent function with finite-trial labels',
        no_physical_evidence=True,source_protocol_sha256=sha(OUT/'protocol.json')))
    np.savez_compressed(dest/'targets.npz',success=ss,failure=ff,true_probability=true_p)
    net=model('local_state_16heads');optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001))
    nmean=float((ss[tr]+ff[tr]).mean());pairs=np.array([(i,j) for i in tr for j in range(16)])
    @jax.jit
    def step(pp,oo,xx,jj,s,f):
        def loss(p):
            z=net.apply(p,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76)))
            z=jnp.take_along_axis(z,jj[:,None],1)[:,0]
            return jnp.mean(s*jax.nn.softplus(-z)+f*jax.nn.softplus(z))/nmean
        value,grad=jax.value_and_grad(loss)(pp);up,oo=optimizer.update(grad,oo,pp)
        return optax.apply_updates(pp,up),oo,value
    predict=jax.jit(lambda pp,xx:net.apply(pp,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76))))
    results={};histories=[];initial=[]
    for seed in SEEDS:
        pp=net.init(jax.random.PRNGKey(seed),jnp.asarray(x[tr[:1]]),jnp.zeros((1,3)),jnp.zeros((1,76)))
        ih=__import__('hashlib').sha256(serialization.to_bytes(pp)).hexdigest()
        original=read(OUT/'fits/local_state_16heads'/f'seed{seed}'/'complete.json')
        assert ih==original['initial_sha256'];initial.append(ih)
        oo=optimizer.init(pp);rng=np.random.default_rng(seed);history=[]
        for it in range(1,2001):
            draw=rng.integers(len(pairs),size=128);ii,jj=pairs[draw].T
            pp,oo,_=step(pp,oo,jnp.asarray(x[ii]),jnp.asarray(jj),jnp.asarray(ss[ii,jj]),jnp.asarray(ff[ii,jj]))
            if it%100:continue
            entry=dict(step=it)
            for split,idx in ix.items():
                z=np.asarray(predict(pp,jnp.asarray(x[idx])));results[f'{seed}__{it}__{split}']=z
                entry[split]=float((ss[idx]*np.logaddexp(0,-z)+ff[idx]*np.logaddexp(0,z)).sum()/(ss[idx]+ff[idx]).sum())
            history.append(entry)
        histories.append(history);print('synthetic completed',seed,history[-1],flush=True)
    val=np.mean([[r['validation'] for r in h] for h in histories],0);selected=histories[0][int(np.argmin(val))]['step']
    rows=[]
    for seed in SEEDS:
        for step0 in sorted({selected,2000}):
            for split,idx in ix.items():
                k=4 if split=='train' else 16;z=results[f'{seed}__{step0}__{split}'];uu=k-ss[idx]-ff[idx]
                m=metrics(z,ss[idx],ff[idx],uu,k)
                m.update({'known_'+key:value for key,value in state_effect(expit(z),true_p[idx],1-true_p[idx]).items()})
                rows.append(dict(seed=seed,step=step0,split=split,**m))
    csvout(dest/'metrics.csv',rows);np.savez_compressed(dest/'logits.npz',**results)
    write(dest/'summary.json',dict(selected_step=selected,mean_VAL_NLL=val.tolist(),matching_initial_hashes=initial,
          source_hard_label_fraction=float(((ss[tr]==0)|(ff[tr]==0)).mean()),
          test={str(st):{key:float(np.mean([r[key] for r in rows if r['step']==st and r['split']=='test']))
                 for key in ('NLL','state_effect_rho','known_state_effect_rho','same_eta_state_change_accuracy','reversal_both_correct')}
                for st in sorted({selected,2000})}))


def kernel():
    """Low-capacity smoothness controls; analytic ridge solutions, VAL selection."""
    from scipy.special import logit
    from scipy.spatial.distance import cdist
    from diagnostics.state_nuisance_rootcause_v1.analyze import labels
    dest=OUT/'kernel_control';dest.mkdir(exist_ok=True)
    p=read(OUT/'protocol.json');d=np.load(SOURCE/'dataset.npz');truth=np.load(SOURCE/'test_truth.npz')
    inp=np.load(OUT/'inputs.npz');x=inp['local'][:,inp['local_active']].astype(float)
    ix={sp:np.array(v) for sp,v in p['splits'].items()};tr=ix['train']
    yl={sp:labels('ring_exchange','FF',sp,d,truth) for sp in ix}
    alphas=[.0001,.001,.01,.1,1.,10.,100.];kernels=['linear','rbf_half','rbf_one','rbf_two']
    write(dest/'protocol.json',dict(kind='Analytic regularized regressions on same pure physical inputs',
          kernels=kernels,alphas=alphas,selection='Mean VAL NLL or VAL centered-by-eta MSE, no TEST selection',
          target='Q4 empirical means with per-cell valid-count weights; TRAIN eta mean intercept',
          prior='Only TRAIN source cells',limitations='Metric-specific local/linear smoothness test; not proof of unlearnability'))
    distance={sp:cdist(x[idx],x[tr],'sqeuclidean')/x.shape[1] for sp,idx in ix.items()}
    med=float(np.median(distance['train'][np.triu_indices(len(tr),1)]))
    s,f=yl['train']['cs'],yl['train']['cf'];n=s+f;qq=s/n;prior=s.sum(0)/n.sum(0)
    preds={};validation=[];history=[]
    for kind in kernels:
        if kind=='linear':
            kk={sp:x[idx]@x[tr].T/x.shape[1] for sp,idx in ix.items()}
        else:
            width={'rbf_half':.5,'rbf_one':1.,'rbf_two':2.}[kind]
            kk={sp:np.exp(-dd/(2*med*width**2)) for sp,dd in distance.items()}
        for alpha in alphas:
            w=[]
            for j in range(16):
                weights=n[:,j]/4
                w.append(np.linalg.solve(kk['train']+alpha*np.diag(1/weights),qq[:,j]-prior[j]))
            w=np.stack(w,1)
            name=f'{kind}__{alpha}'
            for split in ix:
                pp=np.clip(prior+kk[split]@w,1e-6,1-1e-6)
                preds[name+'__'+split]=logit(pp)
            z=preds[name+'__validation'];yy=yl['validation'];m=metrics(z,yy['s'],yy['f'],yy['u'],16)
            validation.append(dict(name=name,**m))
    selected={rule:min(validation,key=lambda r:r[rule])['name'] for rule in ('NLL','state_effect_MSE')}
    write(dest/'selection.json',dict(selected=selected,VAL_scores=validation))
    for rule,name in selected.items():
        for split in ix:
            yy=yl[split];z=preds[name+'__'+split]
            history.append(dict(selection_rule=rule,model=name,split=split,**metrics(z,yy['s'],yy['f'],yy['u'],4 if split=='train' else 16)))
    csvout(dest/'metrics.csv',history);np.savez_compressed(dest/'predictions.npz',**preds)
    # Pure data nearest-neighbor consistency, no learned representation.
    s_tr,f_tr=yl['train']['s'],yl['train']['f'];qtr=s_tr/(s_tr+f_tr)
    neighbors=[]
    for split in ('validation','test'):
        yy=yl[split];q=yy['s']/np.maximum(yy['s']+yy['f'],1)
        nn=distance[split].argmin(1);all_dist=[];near=[]
        for i in range(len(q)):
            for j in range(len(qtr)):
                valid=(yy['u'][i]==0)&(yl['train']['u'][j]==0)
                diff=q[i,valid]-qtr[j,valid];rmse=float(np.sqrt(np.mean(diff**2)))
                all_dist.append(rmse)
                if j==nn[i]:near.append(rmse)
        neighbors.append(dict(split=split,nearest_train_indices=tr[nn].tolist(),
             nearest_Q_RMSE=float(np.mean(near)),random_train_Q_RMSE=float(np.mean(all_dist)),
             nearest_physical_distance=float(np.sqrt(distance[split].min(1)).mean()),
             random_physical_distance=float(np.sqrt(distance[split]).mean())))
    write(dest/'neighbor_consistency.json',dict(rows=neighbors,TRAIN_median_squared_distance=med))
    print(json.dumps(dict(selected=selected,metrics=history,neighbors=neighbors),indent=2))


def fit_heads(name,x,ss,ff,indices,known=None):
    """Identical16-head learner for focused input/label interventions."""
    import jax
    import jax.numpy as jnp
    import optax
    from scipy.special import expit
    from flax import serialization
    dest=OUT/name;dest.mkdir(exist_ok=True)
    tr=indices['train'];pairs=np.array([(i,j) for i in tr for j in range(16) if ss[i,j]+ff[i,j]>0])
    nmean=float(np.mean([ss[i,j]+ff[i,j] for i,j in pairs]));net=model('local_state_16heads')
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001))
    @jax.jit
    def step(pp,oo,xx,jj,s,f):
        def loss(p):
            z=net.apply(p,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76)))
            z=jnp.take_along_axis(z,jj[:,None],1)[:,0]
            return jnp.mean(s*jax.nn.softplus(-z)+f*jax.nn.softplus(z))/nmean
        value,grad=jax.value_and_grad(loss)(pp);up,oo=opt.update(grad,oo,pp)
        return optax.apply_updates(pp,up),oo,value
    predict=jax.jit(lambda pp,xx:net.apply(pp,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76))))
    predictions={};histories=[];initial=[];rows=[]
    for seed in SEEDS:
        pp=net.init(jax.random.PRNGKey(seed),jnp.asarray(x[tr[:1]]),jnp.zeros((1,3)),jnp.zeros((1,76)))
        initial.append(__import__('hashlib').sha256(serialization.to_bytes(pp)).hexdigest())
        oo=opt.init(pp);rng=np.random.default_rng(seed);history=[]
        for it in range(1,2001):
            ii,jj=pairs[rng.integers(len(pairs),size=128)].T
            pp,oo,_=step(pp,oo,jnp.asarray(x[ii]),jnp.asarray(jj),jnp.asarray(ss[ii,jj]),jnp.asarray(ff[ii,jj]))
            if it%100:continue
            entry=dict(step=it)
            for split,idx in indices.items():
                z=np.asarray(predict(pp,jnp.asarray(x[idx])));predictions[f'{seed}__{it}__{split}']=z
                if split!='test':
                    entry[split]=float((ss[idx]*np.logaddexp(0,-z)+ff[idx]*np.logaddexp(0,z)).sum()/(ss[idx]+ff[idx]).sum())
            history.append(entry)
            (dest/f'seed{seed}_step{it}.msgpack').write_bytes(serialization.to_bytes(pp))
        histories.append(history);print('completed',name,seed,history[-1],flush=True)
    val=np.mean([[r['validation'] for r in h] for h in histories],0);selected=histories[0][int(np.argmin(val))]['step']
    for seed in SEEDS:
        for st in sorted({100,500,selected,2000}):
            for split,idx in indices.items():
                z=predictions[f'{seed}__{st}__{split}'];k=4 if split=='train' else 16
                # Counts may be scaled expected counts in noise-free diagnostics.
                m=metrics(z,ss[idx],ff[idx],np.maximum(k-ss[idx]-ff[idx],0),k)
                if known is not None:m.update({'known_'+key:v for key,v in state_effect(expit(z),known[idx],1-known[idx]).items()})
                rows.append(dict(seed=seed,step=st,split=split,**m))
    csvout(dest/'metrics.csv',rows);np.savez_compressed(dest/'logits.npz',**predictions)
    write(dest/'summary.json',dict(selected_step=selected,mean_VAL_NLL=val.tolist(),initial_hashes=initial,
       states={k:len(v) for k,v in indices.items()},input_dimension=x.shape[1],
       results={str(st):{sp:{key:float(np.mean([r[key] for r in rows if r['step']==st and r['split']==sp]))
                            for key in ('NLL','state_effect_rho','state_effect_R2','same_eta_state_change_accuracy','reversal_both_correct')+
                                       (('known_state_effect_rho',) if known is not None else ())}
                        for sp in indices} for st in sorted({100,500,selected,2000})}))


def minimal():
    from ring_exchange.local_frame import vectors_to_local,local_observation
    from types import SimpleNamespace
    p=read(OUT/'protocol.json');phys=read(SOURCE/'protocol.json');d=np.load(SOURCE/'dataset.npz')
    indices={k:np.array(v) for k,v in p['splits'].items()};allix=np.concatenate(list(indices.values()))
    cfg=SimpleNamespace(**phys['environments']['ring_exchange']['config']['config'])
    x=np.zeros((112,23),np.float32);errors=[]
    wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
    for i in allix:
        state=phys['states'][int(i)]['physical'];pos,vel,goal=[np.array(state[k]) for k in ('positions','velocities','goals')]
        theta=np.arctan2(pos[:,1],pos[:,0]);gt=np.arctan2(goal[:,1],goal[:,0])
        rp,rg=np.linalg.norm(pos,axis=1),np.linalg.norm(goal,axis=1)
        rel=wrap(theta[1:]-theta[0]-np.arange(1,4)*np.pi/2)
        gd=wrap(gt-theta-np.pi);vl=vectors_to_local(vel,pos)
        x[i]=np.r_[rp,rel,rg,gd,vl.ravel()]
        # Reconstruct every native physical observation, up to irrelevant global rotation.
        th=np.r_[0.,rel+np.arange(1,4)*np.pi/2]
        radial=np.c_[np.cos(th),np.sin(th)];tangent=np.c_[-np.sin(th),np.cos(th)]
        pp=rp[:,None]*radial;gg=rg[:,None]*np.c_[np.cos(th+np.pi+gd),np.sin(th+np.pi+gd)]
        vv=vl[:,:1]*radial+vl[:,1:]*tangent
        err=float(abs(local_observation(pos,vel,goal,cfg)-local_observation(pp,vv,gg,cfg)).max());assert err<2e-6
        errors.append(err)
    tr=indices['train'];mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),.01);xx=(x-mean)/scale
    ss=d['success'][1].copy();ff=d['failure'][1].copy()
    from diagnostics.state_nuisance_rootcause_v1.analyze import labels
    truth=np.load(SOURCE/'test_truth.npz');yy=labels('ring_exchange','FF','test',d,truth)
    ss[indices['test']],ff[indices['test']]=yy['cs'],yy['cf']
    dest=OUT/'minimal_physical';dest.mkdir(exist_ok=True)
    write(dest/'protocol.json',dict(amendment='After original controller-free models failed, isolate feature redundancy and initial velocity inputs',
          minimal23='4 start radii +3 relative start angles +4 goal radii +4 goal angular offsets +8 local initial velocities',
          minimal15='Same physical geometry, omit8 initial velocity coordinates',
          proof='23-dimensional representation reconstructs all92 native local physical observations',
          maximum_reconstruction_error=max(errors),normalization='TRAIN only, featurewise',
          control='Same16-head64x64 model,32/8/16state splits,3seeds,batches and2000steps; no controller responses',
          no_claim='15-dimensional variant may omit genuinely relevant velocities; improvement alone would not prove velocities irrelevant'))
    np.savez_compressed(dest/'inputs.npz',raw=x,normalized=xx,center=mean,scale=scale)
    fit_heads('minimal23',xx,ss,ff,indices)
    fit_heads('minimal15',xx[:,:15],ss,ff,indices)


def synthetic_identification():
    """Separate sample count, nuisance dimensions, and finite-trial noise in known function."""
    from scipy.special import expit,logit
    from ring_exchange.environment import sample_instance,LocalFrameConfig
    from ring_exchange.local_frame import local_observation
    p=read(OUT/'protocol.json');inputs=np.load(OUT/'inputs.npz');targets=np.load(OUT/'synthetic_control/targets.npz')
    ix={k:np.array(v) for k,v in p['splits'].items()};tr=ix['train'];x=inputs['local']
    d=np.load(SOURCE/'dataset.npz');prior=(d['success'][1,tr].sum(0)+1)/(d['success'][1,tr].sum(0)+d['failure'][1,tr].sum(0)+2)
    eta=inputs['eta'][tr[0]]
    basis=lambda a:np.c_[a[:,5]-a[:,51],a[:,28]-a[:,74],.5*(a[:,0]-a[:,46])]
    design=basis(x);raw=design@eta.T;mu=raw[tr].mean(0);sig=np.maximum(raw[tr].std(0),.1)
    truth=lambda xx:expit(logit(np.clip(prior,.02,.98))[None,:]+4*(basis(xx)@eta.T-mu)/sig)
    np.testing.assert_allclose(truth(x),targets['true_probability'],atol=1e-6)
    extras=[]
    cfg=LocalFrameConfig(**read(SOURCE/'protocol.json')['environments']['ring_exchange']['config']['config'])
    for seed in range(945790000,945790224):
        st=sample_instance(seed,'development',cfg)
        extras.append(local_observation(st.positions,st.velocities,st.goals,cfg).ravel())
    ex=(np.asarray(extras)-inputs['local_center'])/inputs['local_scale'];ex[:,~inputs['local_active']]=0
    expanded=np.concatenate((x,ex));true=truth(expanded)
    sy=targets['success'];fy=targets['failure'];new=(np.random.default_rng(2026100723).random((224,16,4))<true[112:,:,None]).sum(-1)
    sy_exp=np.concatenate((sy,new));fy_exp=np.concatenate((fy,4-new));ix_exp={**ix,'train':np.r_[tr,np.arange(112,336)]}
    min_mean=design[tr].mean(0);min_scale=np.maximum(design[tr].std(0),.01);small=(design-min_mean)/min_scale
    dest=OUT/'synthetic_identification';dest.mkdir(exist_ok=True)
    write(dest/'protocol.json',dict(type='Known-function identification test, not physical rollout evidence',
          trigger='Original synthetic state16head fixture also generalized poorly with32 states and92 inputs',
          controls=['Keep32states and all92 inputs; replace sampled labels with exact conditional probability',
                    'Keep original32states and sampled labels; provide only3 known sufficient physical coordinates',
                    'Keep92 inputs and sampled4-trial labels; expand independent synthetic TRAIN states32to256'],
          fixed='Same underlying physical-state function, held-out physical states,16-head model,3 seeds and2000updates',
          note='Large synthetic improvement would identify finite-data/representation limits of the learner, not prove the real-Q function has the same sufficient coordinates',
          new_physical_rollouts=0))
    np.savez_compressed(dest/'expanded.npz',input=expanded,true_probability=true,success=sy_exp,failure=fy_exp,small=small)
    n=sy+fy
    fit_heads('synthetic_exact_probability',x,n*targets['true_probability'],n*(1-targets['true_probability']),ix,targets['true_probability'])
    fit_heads('synthetic_sufficient3',small,sy,fy,ix,targets['true_probability'])
    fit_heads('synthetic_256states',expanded,sy_exp,fy_exp,ix_exp,true)


def fixed_eta():
    """No controller, eta input, or multi-task sharing: one Q(s) target."""
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from scipy.special import expit
    from scipy.optimize import minimize
    dest=OUT/'fixed_eta';dest.mkdir(exist_ok=True)
    p=read(OUT/'protocol.json');d=np.load(SOURCE/'dataset.npz');ix={k:np.array(v) for k,v in p['splits'].items()}
    x=np.load(OUT/'minimal_physical/inputs.npz')['normalized'];tr=ix['train']
    outcomes=np.load(HERE/'state_signal_path/exact_outcomes.npz')['outcomes'][1]
    covariance=[]
    for j in range(16):
        o=outcomes[tr,j,:4];ok=(o[...,1]==0).all(1)
        a,b=o[ok,:2,0].mean(1),o[ok,2:,0].mean(1)
        covariance.append(float(np.mean((a-a.mean())*(b-b.mean()))))
    eta=int(np.argmax(covariance))
    write(dest/'protocol.json',dict(eta_index=eta,eta_value=read(SOURCE/'protocol.json')['eta'][eta],
          choice='Largest TRAIN cross-half2+2-seed state-effect covariance; deterministic rule, no TEST selection',
          covariance_by_eta=covariance,input='23 pure physical coordinates, no eta or controller input',
          models=['Regularized linear logistic, VAL choice of penalty','One-hidden-layer16-unit SiLU network,3 seeds,2000steps'],
          selections='VAL count NLL; last checkpoint also reported',test='Previously opened16-family retrospective panel'))
    targets={}
    for split,idx in ix.items():
        k=4 if split=='train' else 16;o=outcomes[idx,eta,:k];valid=o[...,1]==0
        s=((o[...,0]==1)&valid).sum(1);f=((o[...,0]==0)&valid).sum(1)
        # Use the original matched training counts, as in all primary fits.
        if split!='test':s,f=d['success'][1,idx,eta],d['failure'][1,idx,eta]
        targets[split]=(s,f)
    def assess(z,split):
        s,f=targets[split];q=s/(s+f);prob=expit(z);i,j=np.triu_indices(len(s),1)
        gap=q[i]-q[j];valid=abs(gap)>=.25;diff=z[i]-z[j]
        return dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum()),
             state_rho=float(np.corrcoef(prob,q)[0,1]) if prob.std()>1e-10 else None,
             state_direction_accuracy=float(((gap[valid]*diff[valid]>0)+.5*(diff[valid]==0)).mean()),
             state_pair_cases=int(valid.sum()))
    # Each low-capacity fit is convex; no neural shared-trunk explanation applies.
    xt=np.c_[x[tr],np.ones(len(tr))];s,f=targets['train'];val=[];linear={}
    for penalty in (.0001,.001,.01,.1,1.,10.):
        def objective(w):
            z=xt@w;loss=(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum()+penalty/2*np.sum(w[:-1]**2)
            gradient=xt.T@((s+f)*expit(z)-s)/(s+f).sum();gradient[:-1]+=penalty*w[:-1]
            return loss,gradient
        opt=minimize(objective,np.zeros(24),jac=True,method='L-BFGS-B',options={'maxiter':2000,'gtol':1e-9,'ftol':1e-12})
        assert opt.success,opt.message
        linear[penalty]={sp:np.c_[x[idx],np.ones(len(idx))]@opt.x for sp,idx in ix.items()}
        val.append(dict(penalty=penalty,**assess(linear[penalty]['validation'],'validation')))
    chosen=min(val,key=lambda r:r['NLL'])['penalty'];results=[]
    for split,z in linear[chosen].items():results.append(dict(model='linear',seed=0,step='VAL-selected',split=split,**assess(z,split)))
    class Small(nn.Module):
        @nn.compact
        def __call__(self,xx):return nn.Dense(1)(nn.silu(nn.Dense(16)(xx)))[:,0]
    net=Small();optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001))
    @jax.jit
    def step(pp,oo,xx,ss,ff):
        def loss(params):
            z=net.apply(params,xx);return jnp.mean(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/float((s+f).mean())
        ll,g=jax.value_and_grad(loss)(pp);up,oo=optimizer.update(g,oo,pp)
        return optax.apply_updates(pp,up),oo
    predict=jax.jit(net.apply);predictions={};valhist=[]
    for seed in SEEDS:
        pp=net.init(jax.random.PRNGKey(seed),jnp.asarray(x[tr[:1]]));oo=optimizer.init(pp);rng=np.random.default_rng(seed);hh=[]
        for it in range(1,2001):
            draw=rng.integers(len(tr),size=128)
            pp,oo=step(pp,oo,jnp.asarray(x[tr[draw]]),jnp.asarray(s[draw]),jnp.asarray(f[draw]))
            if it%100:continue
            for split,idx in ix.items():predictions[f'{seed}__{it}__{split}']=np.asarray(predict(pp,jnp.asarray(x[idx])))
            hh.append(assess(predictions[f'{seed}__{it}__validation'],'validation')['NLL'])
        valhist.append(hh)
    chosen_step=(int(np.argmin(np.mean(valhist,0)))+1)*100
    for seed in SEEDS:
        for st in sorted({chosen_step,2000}):
            for split in ix:results.append(dict(model='small_MLP',seed=seed,step=st,split=split,**assess(predictions[f'{seed}__{st}__{split}'],split)))
    csvout(dest/'metrics.csv',results);np.savez_compressed(dest/'logits.npz',**predictions)
    write(dest/'summary.json',dict(eta_index=eta,linear_penalty=chosen,linear_validation_grid=val,MLP_selected_step=chosen_step,
          results=results,independent_train_states=len(tr),independent_test_states=len(ix['test'])))
    print(json.dumps(dict(eta=eta,linear_penalty=chosen,MLP_step=chosen_step,results=results),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('synthetic','kernel','minimal','synthetic_identification','fixed_eta'))
    globals()[parser.parse_args().action]()
