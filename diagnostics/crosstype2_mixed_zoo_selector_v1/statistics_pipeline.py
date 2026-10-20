"""Paired type/controller/state bootstrap and seed-cross-fitted state oracle."""
import numpy as np

BASE=12
FIELDS=('success_lower','success_upper','collision','timeout','unknown','gain','gain_lower','gain_upper',
        'regret','regret_lower','regret_upper','in_sample_regret','in_sample_regret_lower','in_sample_regret_upper')


def best(scores):
    out=np.argmax(scores,axis=-1)
    return np.where(scores[...,BASE]>=scores.max(-1)-1e-12,BASE,out)


def gather(values,choice):return np.take_along_axis(values,choice[...,None,:],axis=-2)[...,0,:]


def oracle_choices(lower):
    # batch, controller, state, candidate, seed; nominal is excluded.
    values=lower[...,:16,:];b,n,s,_,q=values.shape
    inside=np.repeat(best(values.mean(-1))[...,None],q,axis=-1)
    controller=np.broadcast_to(best(values.mean((2,4)))[:,:,None,None],(b,n,s,q))
    out=dict(state_in_sample_oracle=inside,controller_oracle=controller)
    if q==2:
        out['state_crossfit_oracle']=np.stack((best(values[...,1]),best(values[...,0])),axis=-1)
        choice=np.stack((best(values[...,1].mean(2)),best(values[...,0].mean(2))),axis=-1)
        out['controller_crossfit_oracle']=np.broadcast_to(choice[:,:,None,:],(b,n,s,q))
    return out


def equivalence(uids):
    n,s,a,q=uids.shape;out=np.broadcast_to(np.arange(a,dtype=np.int8)[None,None,:,None],(n,s,a,q)).copy()
    for j in range(a):
        for k in range(j):out[:,:,j]=np.where(uids[:,:,j]==uids[:,:,k],np.minimum(out[:,:,j],out[:,:,k]),out[:,:,j])
    return out


def sample_indices(types,states,rng,batch,type_equal):
    n=len(types)
    if type_equal:
        groups=np.unique(types);pools=[np.flatnonzero(types==t) for t in groups];maximum=max(map(len,pools))
        table=np.zeros((len(groups),maximum),int);counts=np.asarray(list(map(len,pools)))
        for i,pool in enumerate(pools):table[i,:len(pool)]=pool
        chosen=rng.integers(len(groups),size=(batch,len(groups)));sizes=counts[chosen]
        offsets=(rng.random((batch,len(groups),maximum))*sizes[...,None]).astype(int)
        cc=table[chosen[...,None],offsets];active=np.arange(maximum)<sizes[...,None]
        weights=active/(len(groups)*sizes[...,None]);cc=cc.reshape(batch,-1);weights=weights.reshape(batch,-1)
    else:cc=rng.integers(n,size=(batch,n));weights=np.full((batch,n),1/n)
    ss=rng.integers(states,size=(*cc.shape,states))
    return cc,ss,weights


def point_indices(types,states,type_equal):
    n=len(types);cc=np.arange(n)[None];ss=np.broadcast_to(np.arange(states),(1,n,states))
    w=np.asarray([1/(len(set(types))*np.sum(types==t)) for t in types]) if type_equal else np.full(n,1/n)
    return cc,ss,w[None]


def draw(data,fixed,cc,ss,w,pairs,details=False):
    labels=data['labels'][cc[...,None],ss];lo=(labels==1).astype(float);hi=(labels!=0).astype(float)
    eq=data['uid_class'][cc[...,None],ss];q=labels.shape[-1]
    methods={name:np.repeat(choice[cc[...,None],ss][...,None],q,axis=-1) for name,choice in fixed.items()}
    methods.update(oracle_choices(lo));names=list(methods)
    values={name:dict(lo=gather(lo,c),hi=gather(hi,c),eq=gather(eq,c),choice=c) for name,c in methods.items()}
    base=values['all_Sprog'];inside=values['state_in_sample_oracle'];oracle=values.get('state_crossfit_oracle',inside)
    def mean(v):return np.sum(v.mean((-1,-2))*w,axis=1)
    def delta(a,b):
        same=a['eq']==b['eq']
        return (mean(a['lo']-b['lo']),mean(np.where(same,0.,a['lo']-b['hi'])),mean(np.where(same,0.,a['hi']-b['lo'])))
    metrics=[]
    for name in names:
        value=values[name];choice=methods[name]
        events=[gather(data[e][cc[...,None],ss],choice) for e in ('collision','timeout','unknown')]
        metrics.append(np.stack([mean(value['lo']),mean(value['hi']),*[mean(e) for e in events],
            *delta(value,base),*delta(oracle,value),*delta(inside,value)],axis=-1))
    contrasts={f'{a}__minus__{b}':np.stack(delta(values[a],values[b]),axis=-1) for a,b in pairs if a in values and b in values}
    if all(k in values for k in ('pipeline','all_RHR','source_fixed')):
        fixed_deltas=[np.stack(delta(values['pipeline'],values[b]),axis=-1) for b in ('all_Sprog','all_RHR','source_fixed')]
        contrasts['pipeline__minus__best_fixed']=np.minimum.reduce(fixed_deltas)
    return names,np.stack(metrics,axis=1),contrasts,values if details else None


def evaluate_group(data,fixed,type_equal=True,replicates=5000):
    types=data['types'];ns=data['labels'].shape[1];nc=len(types);q=data['labels'].shape[-1]
    assert not np.any(data['labels']==-2),'Missing observations cannot be evaluated as unknown'
    fixed={k:np.broadcast_to(v[:,None],(nc,ns)) if np.asarray(v).ndim==1 else np.asarray(v) for k,v in fixed.items()}
    assert 'all_Sprog' in fixed and 'all_safety_baseline' in fixed
    assert all(np.shape(v)==(nc,ns) for v in fixed.values())
    pairs=[('pipeline',x) for x in ('all_Sprog','all_S500','all_safety_baseline','no_controller','no_state','original_pipeline','old_pipeline','all_RHR','source_fixed','source_mean_fixed','few_2','few_4','few_8')]
    pairs.extend((oracle,baseline) for oracle in ('state_in_sample_oracle','controller_oracle','state_crossfit_oracle','controller_crossfit_oracle') for baseline in ('all_Sprog','source_fixed'))
    cc,ss,w=point_indices(types,ns,type_equal);names,point,contrasts,detail=draw(data,fixed,cc,ss,w,pairs,True)
    rng=np.random.default_rng(202610182201);samples=[];paired={k:[] for k in contrasts}
    for start in range(0,replicates,32):
        cc,ss,w=sample_indices(types,ns,rng,min(32,replicates-start),type_equal)
        order,v,contrast,_=draw(data,fixed,cc,ss,w,pairs);assert order==names;samples.append(v)
        for k in paired:paired[k].append(contrast[k])
    samples=np.concatenate(samples);ci=np.quantile(samples,[.025,.975],axis=0)
    report={};safety=detail['all_safety_baseline']
    for i,name in enumerate(names):
        value=point[0,i]*100;interval=ci[:,i]*100;actual=detail[name]
        same=actual['eq']==safety['eq']
        harm=np.where(same,0.,safety['hi']-actual['lo'])[0].mean((1,2))
        certain=np.where(same,0.,safety['lo']-actual['hi'])[0].mean((1,2))
        oracle=detail.get('state_crossfit_oracle')
        maximum=float(((oracle['lo']-actual['lo'])[0].mean((1,2))*100).max()) if oracle else None
        in_max=float(((detail['state_in_sample_oracle']['lo']-actual['lo'])[0].mean((1,2))*100).max())
        report[name]=dict(success_lower_pct=float(value[0]),success_upper_pct=float(value[1]),
            success_CI95_pct=[float(interval[0,0]),float(interval[1,1])],
            collision_pct=float(value[2]),collision_CI95_pct=interval[:,2].tolist(),
            timeout_pct=float(value[3]),timeout_CI95_pct=interval[:,3].tolist(),
            unknown_pct=float(value[4]),unknown_CI95_pct=interval[:,4].tolist(),
            gain_vs_progress_pp=float(value[5]),gain_unknown_bounds_pp=value[6:8].tolist(),
            gain_CI95_pp=[float(interval[0,6]),float(interval[1,7])],
            state_crossfit_regret_pp=float(value[8]) if q==2 else None,
            state_crossfit_regret_unknown_bounds_pp=value[9:11].tolist() if q==2 else None,
            state_crossfit_regret_CI95_pp=[float(interval[0,9]),float(interval[1,10])] if q==2 else None,
            max_controller_crossfit_regret_pp=maximum,
            state_in_sample_regret_pp=float(value[11]),state_in_sample_regret_unknown_bounds_pp=value[12:14].tolist(),
            state_in_sample_regret_CI95_pp=[float(interval[0,12]),float(interval[1,13])],max_controller_in_sample_regret_pp=in_max,
            harm_possible=int(np.sum(harm>=.1-1e-12)),harm_certain=int(np.sum(certain>=.1-1e-12)),
            disasters_vs_F10=None,disaster_unavailable_reason='F10 is not one of this round\'s sixteen requested candidates')
    pair_report={}
    for k,p in contrasts.items():
        values=np.concatenate(paired[k]);value=p[0]*100
        pair_report[k]=dict(gain_pp=float(value[0]),unknown_bounds_pp=value[1:].tolist(),
            CI95_pp=[float(np.quantile(values[:,1],.025)*100),float(np.quantile(values[:,2],.975)*100)])
    return dict(methods=report,paired_comparisons=pair_report,controllers=nc,types=sorted(set(types.tolist())),
        replicates=replicates,type_equal=type_equal,state_crossfit_oracle_available=q==2,
        oracle_reselected_in_every_bootstrap=True,seeds_clustered=True,unknown_preserved=True,shared_uid_unknowns_cancelled=True)


def evaluate(data,fixed,replicates=5000):
    data=dict(data)
    data['uid_class']=equivalence(data['uids']) if 'uids' in data else np.broadcast_to(np.arange(data['labels'].shape[2],dtype=np.int8)[None,None,:,None],data['labels'].shape)
    nc=len(data['types']);reports={}
    for group in [*sorted(set(data['types'])),'combined']:
        mask=np.ones(nc,bool) if group=='combined' else data['types']==group
        subset={k:v[mask] for k,v in data.items() if k in ('labels','collision','timeout','unknown','uids','uid_class','names','types')}
        methods={k:np.asarray(v)[mask] for k,v in fixed.items()}
        reports[str(group)]={'type_equal':evaluate_group(subset,methods,True,replicates)}
        if group=='combined':reports[str(group)]['controller_equal']=evaluate_group(subset,methods,False,replicates)
    return reports
