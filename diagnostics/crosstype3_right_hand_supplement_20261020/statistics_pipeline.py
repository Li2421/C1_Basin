"""Paired clustered bootstrap for a partially evaluated schedule menu.

Sampling follows cross-type1b statistics_pipeline.py (SHA in history inventory).
Padding is missing evidence, never a zero-success rollout. Oracle is only the
actually evaluated action set and excludes nominal execution.
"""
import numpy as np
def gather(x,choice):return np.take_along_axis(x,choice[...,None,:],axis=-2)[...,0,:]
def equivalence(uids):
    out=np.broadcast_to(np.arange(uids.shape[2],dtype=np.int16)[None,None,:,None],uids.shape).copy()
    for a in range(uids.shape[2]):
        for b in range(a):out[:,:,a]=np.where((uids[:,:,a]==uids[:,:,b])&(uids[:,:,a]!=''),np.minimum(out[:,:,a],out[:,:,b]),out[:,:,a])
    return out
def sample_indices(types,states,rng,batch,type_equal,fixed_type_strata=False):
    n=len(types)
    if type_equal:
        groups=np.unique(types);pools=[np.flatnonzero(types==t) for t in groups];maximum=max(map(len,pools));table=np.zeros((len(groups),maximum),int);counts=np.asarray(list(map(len,pools)))
        for i,pool in enumerate(pools):table[i,:len(pool)]=pool
        chosen=np.broadcast_to(np.arange(len(groups)),(batch,len(groups))) if fixed_type_strata else rng.integers(len(groups),size=(batch,len(groups)));sizes=counts[chosen];offsets=(rng.random((batch,len(groups),maximum))*sizes[...,None]).astype(int);cc=table[chosen[...,None],offsets];active=np.arange(maximum)<sizes[...,None];weights=active/(len(groups)*sizes[...,None]);cc=cc.reshape(batch,-1);weights=weights.reshape(batch,-1)
    else:cc=rng.integers(n,size=(batch,n));weights=np.full((batch,n),1/n)
    return cc,rng.integers(states,size=(*cc.shape,states)),weights
def point_indices(types,states,type_equal):
    n=len(types);cc=np.arange(n)[None];ss=np.broadcast_to(np.arange(states),(1,n,states));weights=np.asarray([1/(len(set(types))*np.sum(types==t)) for t in types]) if type_equal else np.full(n,1/n)
    return cc,ss,weights[None]
def oracle_choices(labels,mask):
    lo=(labels==1).astype(float);mask=np.broadcast_to(mask[...,None],labels.shape)&(labels!=-2);assert np.all(mask.any(-2))
    mean=np.where(mask,lo,-np.inf).mean(-1);inside=np.argmax(mean,-1);q=labels.shape[-1];out=dict(evaluated_state_in_sample_oracle=np.repeat(inside[...,None],q,-1))
    if q==2:out['evaluated_state_crossfit_oracle']=np.stack([np.argmax(np.where(mask[...,1],lo[...,1],-np.inf),-1),np.argmax(np.where(mask[...,0],lo[...,0],-np.inf),-1)],-1)
    return out
FIELDS=('success_lower','success_upper','collision','timeout','unknown','gain_safety','gain_safety_lower','gain_safety_upper','regret','regret_lower','regret_upper','in_sample_regret','in_sample_regret_lower','in_sample_regret_upper','non_diagonal')
def draw(data,methods,cc,ss,w,details=False):
    labels=data['labels'][cc[...,None],ss];lo=(labels==1).astype(float);hi=((labels!=0)&(labels!=-2)).astype(float);eq=data['uid_class'][cc[...,None],ss];mask=data['oracle_mask'][cc[...,None],ss];q=labels.shape[-1]
    choices={k:np.repeat(v[cc[...,None],ss][...,None],q,-1) for k,v in methods.items()};choices.update(oracle_choices(labels,mask));values={k:dict(lo=gather(lo,a),hi=gather(hi,a),eq=gather(eq,a),choice=a) for k,a in choices.items()}
    def mean(x):return np.sum(x.mean((-1,-2))*w,1)
    def delta(a,b):
        same=a['eq']==b['eq'];return np.stack([mean(a['lo']-b['lo']),mean(np.where(same,0,a['lo']-b['hi'])),mean(np.where(same,0,a['hi']-b['lo']))],-1)
    baseline=values['all_safety_baseline'];inside=values['evaluated_state_in_sample_oracle'];oracle=values.get('evaluated_state_crossfit_oracle',inside);names=list(values);metrics=[]
    for name in names:
        value=values[name];a=choices[name];events=[gather(data[k][cc[...,None],ss],a) for k in ('collision','timeout','unknown')];non_diag=gather(np.broadcast_to(data['non_diagonal'][cc[...,None],ss][...,None],labels.shape),a)
        metrics.append(np.stack([mean(value['lo']),mean(value['hi']),*[mean(e) for e in events],*np.moveaxis(delta(value,baseline),-1,0),*np.moveaxis(delta(oracle,value),-1,0),*np.moveaxis(delta(inside,value),-1,0),mean(non_diag)],-1))
    pairs={}
    requested=[('pipeline',k) for k in methods if k!='pipeline']+[('self_schedule','fixed500')]
    for a,b in requested:
        if a in values and b in values:pairs[a+'__minus__'+b]=delta(values[a],values[b])
    if 'all_S500' in values and 'all_Sprog' in values:pairs['pipeline__minus__best_human_schedule']=np.minimum(delta(values['pipeline'],values['all_S500']),delta(values['pipeline'],values['all_Sprog']))
    return names,np.stack(metrics,1),pairs,values if details else None
def evaluate_group(data,methods,type_equal=True,replicates=5000,fixed_type_strata=False):
    types=data['types'];nc=len(types);ns=data['labels'].shape[1];q=data['labels'].shape[-1]
    assert 'all_safety_baseline' in methods and 'pipeline' in methods
    for method,indices in methods.items():
        assert indices.shape==(nc,ns)
        assert not np.any(data['labels'][np.arange(nc)[:,None],np.arange(ns)[None],indices]==-2),(method,'Missing evidence cannot be scored')
    cc,ss,w=point_indices(types,ns,type_equal);names,point,paired,detail=draw(data,methods,cc,ss,w,True);rng=np.random.default_rng(202610193201);samples=[];contrasts={k:[] for k in paired}
    for start in range(0,replicates,32):
        cc,ss,w=sample_indices(types,ns,rng,min(32,replicates-start),type_equal,fixed_type_strata);order,value,p,_=draw(data,methods,cc,ss,w);assert order==names;samples.append(value)
        for k in contrasts:contrasts[k].append(p[k])
    samples=np.concatenate(samples);ci=np.quantile(samples,[.025,.975],axis=0)*100;point=point[0]*100;report={};safety=detail['all_safety_baseline'];oracle=detail.get('evaluated_state_crossfit_oracle',detail['evaluated_state_in_sample_oracle'])
    for i,name in enumerate(names):
        value=point[i];actual=detail[name];same=actual['eq']==safety['eq'];possible=np.where(same,0,safety['hi']-actual['lo'])[0].mean((1,2));certain=np.where(same,0,safety['lo']-actual['hi'])[0].mean((1,2));maximum=(oracle['lo']-actual['lo'])[0].mean((1,2)).max()*100
        report[name]=dict(success_lower_pct=float(value[0]),success_upper_pct=float(value[1]),success_CI95_pct=[float(ci[0,i,0]),float(ci[1,i,1])],collision_pct=float(value[2]),timeout_pct=float(value[3]),unknown_pct=float(value[4]),gain_vs_safety_pp=float(value[5]),gain_vs_safety_unknown_bounds_pp=value[6:8].tolist(),gain_vs_safety_CI95_pp=[float(ci[0,i,6]),float(ci[1,i,7])],evaluated_action_crossfit_regret_pp=float(value[8]) if q==2 else None,evaluated_action_crossfit_regret_unknown_bounds_pp=value[9:11].tolist() if q==2 else None,evaluated_action_crossfit_regret_CI95_pp=[float(ci[0,i,9]),float(ci[1,i,10])] if q==2 else None,evaluated_action_in_sample_regret_pp=float(value[11]),evaluated_action_in_sample_regret_unknown_bounds_pp=value[12:14].tolist(),evaluated_action_in_sample_regret_CI95_pp=[float(ci[0,i,12]),float(ci[1,i,13])],max_controller_regret_pp=float(maximum),harm_possible=int(np.sum(possible>=.1-1e-12)),harm_certain=int(np.sum(certain>=.1-1e-12)),selected_non_diagonal_fraction=float(value[14]/100),disasters_vs_F10=None,full_menu_oracle_available=False,controller_fixed_oracle_available=False)
        if 'all_F10' in detail:
            default=detail['all_F10'];equal=actual['eq']==default['eq'];possible_default=np.where(equal,0,default['hi']-actual['lo'])[0].mean((1,2));certain_default=np.where(equal,0,default['lo']-actual['hi'])[0].mean((1,2))
            report[name].update(disasters_vs_F10=int(np.sum(possible_default>=.1-1e-12)),disasters_certain_vs_F10=int(np.sum(certain_default>=.1-1e-12)))
    comparisons={}
    for k,p in paired.items():
        value=p[0]*100;draws=np.concatenate(contrasts[k])*100;comparisons[k]=dict(gain_pp=float(value[0]),unknown_bounds_pp=value[1:].tolist(),CI95_pp=[float(np.quantile(draws[:,1],.025)),float(np.quantile(draws[:,2],.975))])
    return dict(methods=report,paired_comparisons=comparisons,controllers=nc,types=sorted(set(types.tolist())),type_equal=type_equal,fixed_type_strata=fixed_type_strata,bootstrap_seed=202610193201,replicates=replicates,seeds_clustered=True,oracle_scope='Only actually evaluated non-nominal actions, not full320 grid',oracle_reselected_every_bootstrap=True,best_human_schedule_reselected_every_bootstrap=True,controller_fixed_oracle_reason='Sparse changing evaluated-action sets do not identify a common complete fixed menu',unknown_preserved=True,shared_uid_unknowns_cancelled=True)
