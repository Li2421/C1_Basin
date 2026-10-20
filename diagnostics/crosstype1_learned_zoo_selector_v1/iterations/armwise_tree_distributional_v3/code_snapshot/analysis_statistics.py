"""Paired hierarchical bootstrap with oracle selection repeated in every draw.

Input arrays are already gated by the development/confirmatory assembler.
Original state indices determine parity even after duplicated bootstrap draws.
No fitting, controller inference, simulation, or evidence I/O occurs here.
"""
import numpy as np
from selector import ARMS,BASELINE,best_arm

DYNAMIC=('crossfit_oracle','in_sample_oracle','type_best_fixed_oracle','nominal_only')
METRICS=('success_lower','success_upper','collision','timeout','unknown',
         'gain_point','gain_lower','gain_upper','regret_point','regret_lower','regret_upper',
         'in_sample_regret_point','in_sample_regret_lower','in_sample_regret_upper')


def gather_arm(values,choice):
    return np.take_along_axis(values,choice[...,None,None],axis=-2)[...,0,:]


def oracle_choices(lower,state_indices,controller_types,controller_weights):
    # lower: bootstrap draw, sampled controller, sampled state, arm, seed.
    parity=state_indices%2;choices=[]
    for fold in (0,1):
        mask=parity!=fold
        count=mask.sum(-1)
        means=np.sum(lower*mask[...,None,None],axis=(2,4))/np.maximum(count[...,None]*lower.shape[-1],1)
        choices.append(np.where(count>0,best_arm(means),BASELINE))
    cross=np.where(parity==0,choices[0][...,None],choices[1][...,None])
    inside=np.broadcast_to(best_arm(lower.mean((2,4)))[...,None],parity.shape)
    pertype=np.empty(parity.shape,int)
    for t in np.unique(controller_types):
        mask=controller_types==t
        weighted=mask*controller_weights
        count=weighted.sum(1)
        means=np.sum(lower*weighted[:,:,None,None,None],axis=(1,2,4))/np.maximum(count[:,None]*lower.shape[2]*lower.shape[-1],1e-30)
        choice=best_arm(means)
        pertype=np.where(mask[:,:,None],choice[:,None,None],pertype)
    return dict(crossfit_oracle=cross,in_sample_oracle=inside,type_best_fixed_oracle=pertype)


def sample_indices(types,states,rng,batch,type_equal):
    n=len(types)
    if type_equal:
        groups=np.unique(types);pools=[np.flatnonzero(types==t) for t in groups]
        maximum=max(map(len,pools));table=np.zeros((len(groups),maximum),int)
        counts=np.asarray(list(map(len,pools)))
        for t,pool in enumerate(pools):table[t,:len(pool)]=pool
        sampled_types=rng.integers(len(groups),size=(batch,len(groups)))
        sizes=counts[sampled_types]
        offsets=(rng.random((batch,len(groups),maximum))*sizes[...,None]).astype(int)
        cc=table[sampled_types[...,None],offsets]
        active=np.arange(maximum)<sizes[...,None]
        weights=active/(len(groups)*sizes[...,None])
        cc=cc.reshape(batch,-1);weights=weights.reshape(batch,-1)
    else:
        cc=rng.integers(n,size=(batch,n));weights=np.full((batch,n),1/n)
    ss=rng.integers(states,size=(*cc.shape,states))
    return cc,ss,weights


def point_indices(types,states,type_equal):
    cc=np.arange(len(types))[None,:];ss=np.broadcast_to(np.arange(states),(1,len(types),states))
    weights=(np.array([1/(len(np.unique(types))*np.sum(types==t)) for t in types]) if type_equal
             else np.full(len(types),1/len(types)))
    return cc,ss,weights[None,:]


def draw(data,fixed,cc,ss,weight):
    labels=data['labels'][cc[...,None],ss]
    lo=(labels==1).astype(float);hi=(labels!=0).astype(float)
    methods={name:choice[cc[...,None],ss] for name,choice in fixed.items()}
    methods.update(oracle_choices(lo,ss,data['types'][cc],weight))
    methods['nominal_only']=np.full(ss.shape,-1,int)
    oracle=methods['crossfit_oracle'];oracle_lo=gather_arm(lo,oracle);oracle_hi=gather_arm(hi,oracle)
    inside=methods['in_sample_oracle'];inside_lo=gather_arm(lo,inside);inside_hi=gather_arm(hi,inside)
    baseline_lo=lo[...,BASELINE,:];baseline_hi=hi[...,BASELINE,:]
    values=[];details={}
    def mean(x):return np.sum(x.mean((-1,-2))*weight,axis=1)
    for name,choice in methods.items():
        if name=='nominal_only':
            nominal=data['nominal'][cc[...,None],ss]
            low=(nominal==1).astype(float);high=(nominal!=0).astype(float)
            event=[data['nominal_'+e][cc[...,None],ss] for e in ('collision','timeout','unknown')]
        else:
            low=gather_arm(lo,choice);high=gather_arm(hi,choice)
            event=[gather_arm(data[e][cc[...,None],ss],choice) for e in ('collision','timeout','unknown')]
        same=(choice==BASELINE)[...,None];same_oracle=(choice==oracle)[...,None];same_inside=(choice==inside)[...,None]
        values.append(np.stack([mean(low),mean(high),*[mean(x) for x in event],mean(low-baseline_lo),
            mean(np.where(same,0.,low-baseline_hi)),mean(np.where(same,0.,high-baseline_lo)),
            mean(oracle_lo-low),mean(np.where(same_oracle,0.,oracle_lo-high)),
            mean(np.where(same_oracle,0.,oracle_hi-low)),mean(inside_lo-low),
            mean(np.where(same_inside,0.,inside_lo-high)),mean(np.where(same_inside,0.,inside_hi-low))],axis=-1))
        details[name]=dict(low=low,high=high,choice=choice,oracle_lo=oracle_lo,inside_lo=inside_lo)
    return list(methods),np.stack(values,axis=1),details


def evaluate_group(data,fixed,type_equal=True,replicates=5000,seed=202610180501):
    types=np.asarray(data['types']);ns=data['labels'].shape[1];nc=len(types)
    fixed={name:(np.broadcast_to(np.asarray(v)[:,None],(nc,ns)) if np.asarray(v).ndim==1 else np.asarray(v))
           for name,v in fixed.items()}
    assert not set(fixed)&set(DYNAMIC)
    cc,ss,w=point_indices(types,ns,type_equal);names,point,detail=draw(data,fixed,cc,ss,w)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,64):
        cc,ss,w=sample_indices(types,ns,rng,min(64,replicates-start),type_equal)
        actual,sampled,_=draw(data,fixed,cc,ss,w);assert actual==names;samples.append(sampled)
    samples=np.concatenate(samples);interval=np.quantile(samples,[.025,.975],axis=0)
    report={}
    safety_lo=(data['labels'][:,:,0,:]==1).astype(float);safety_hi=(data['labels'][:,:,0,:]!=0).astype(float)
    fixed10_lo=(data['labels'][:,:,1,:]==1).astype(float);fixed10_hi=(data['labels'][:,:,1,:]!=0).astype(float)
    for m,name in enumerate(names):
        values=point[0,m]*100;ci=interval[:,m]*100;item=detail[name]
        low,high,choice=item['low'][0],item['high'][0],item['choice'][0]
        possible_harm=np.where((choice==0)[...,None],0.,safety_hi-low).mean((1,2))
        certain_harm=np.where((choice==0)[...,None],0.,safety_lo-high).mean((1,2))
        possible_disaster=np.where((choice==1)[...,None],0.,fixed10_hi-low).mean((1,2))
        certain_disaster=np.where((choice==1)[...,None],0.,fixed10_lo-high).mean((1,2))
        controller_regret=(item['oracle_lo'][0]-low).mean((1,2))*100
        report[name]=dict(success_lower_pct=float(values[0]),success_upper_pct=float(values[1]),
            success_CI95_pct=[float(ci[0,0]),float(ci[1,1])],
            success_lower_CI95_pct=ci[:,0].tolist(),success_upper_CI95_pct=ci[:,1].tolist(),
            collision_pct=float(values[2]),collision_CI95_pct=ci[:,2].tolist(),
            timeout_pct=float(values[3]),timeout_CI95_pct=ci[:,3].tolist(),
            unknown_pct=float(values[4]),unknown_CI95_pct=ci[:,4].tolist(),
            gain_vs_schedule_pp=float(values[5]),gain_unknown_bounds_pp=values[6:8].tolist(),
            gain_CI95_pp=[float(ci[0,6]),float(ci[1,7])],
            crossfit_regret_pp=float(values[8]),crossfit_regret_unknown_bounds_pp=values[9:11].tolist(),
            crossfit_regret_CI95_pp=[float(ci[0,9]),float(ci[1,10])],
            max_controller_regret_pp=float(controller_regret.max()),
            in_sample_regret_pp=float(values[11]),in_sample_regret_unknown_bounds_pp=values[12:14].tolist(),
            in_sample_regret_CI95_pp=[float(ci[0,12]),float(ci[1,13])],
            max_controller_in_sample_regret_pp=float((item['inside_lo'][0]-low).mean((1,2)).max()*100),
            harm_possible=int(np.sum(possible_harm>=.1-1e-12)),harm_certain=int(np.sum(certain_harm>=.1-1e-12)),
            disaster_possible=int(np.sum(possible_disaster>=.1-1e-12)),disaster_certain=int(np.sum(certain_disaster>=.1-1e-12)))
    return dict(methods=report,controllers=nc,types=sorted(set(types.tolist())),replicates=replicates,
                type_equal=type_equal,paired=True,seed=seed,oracle_reselected_within_bootstrap=True,
                state_parity='original state index, unchanged by resampling',seed_replicates_clustered=True,
                oracle_selection_target='lower success labels; upper success bounds retain that lower-label decision')


def evaluate(data,fixed,replicates=5000):
    reports={};types=np.asarray(data['types']);nc=len(types)
    for group in [*sorted(set(types)),'combined']:
        mask=np.ones(nc,bool) if group=='combined' else types==group
        subset={k:v[mask] for k,v in data.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==nc
                and k not in ('arms','feature_names','state_uids')}
        all_choices={k:np.asarray(v)[mask] for k,v in fixed.items()}
        choices={k:v for k,v in all_choices.items() if np.all((v>=0)&(v<len(ARMS)))}
        unavailable={k:'At least one controller selected an arm outside the frozen menu; no substitution or partial-cohort aggregate'
                     for k in all_choices if k not in choices}
        reports[str(group)]={'type_equal':evaluate_group(subset,choices,True,replicates)}
        if group=='combined':reports[str(group)]['controller_equal']=evaluate_group(subset,choices,False,replicates)
        for record in reports[str(group)].values():record['unavailable_methods']=unavailable
    return reports
