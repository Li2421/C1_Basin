"""Complete chosen methods remain estimable when historical references are absent."""
import numpy as np
from statistics_pipeline import point_indices,sample_indices
def evaluate(data,methods,type_equal=True,replicates=5000):
    labels=data['labels'];nc,ns=labels.shape[:2];q=labels.shape[-1];chosen={};missing={}
    for key,index in methods.items():
        raw=labels[np.arange(nc)[:,None],np.arange(ns)[None],index]
        if np.any(raw==-2):missing[key]='Incomplete exact cached coverage; no imputation';continue
        chosen[key]=dict(lo=(raw==1).astype(float),hi=(raw!=0).astype(float),uid=data['uids'][np.arange(nc)[:,None],np.arange(ns)[None],index],**{k:data[k][np.arange(nc)[:,None],np.arange(ns)[None],index] for k in ('collision','timeout','unknown')})
    keys=list(chosen)
    def draw(cc,ss,w):
        def mean(x):return (x[cc[...,None],ss].mean((-1,-2))*w).sum(1)
        rates=np.stack([np.stack([mean(chosen[key][field]) for field in ('lo','hi','collision','timeout','unknown')],-1) for key in keys],1);pairs={}
        def delta(a,b):
            aa,bb=chosen[a],chosen[b];same=aa['uid']==bb['uid']
            return np.stack([mean(aa['lo']-bb['lo']),mean(np.where(same,0,aa['lo']-bb['hi'])),mean(np.where(same,0,aa['hi']-bb['lo']))],-1)
        if 'pipeline' in chosen:
            for key in chosen:
                if key!='pipeline':pairs['pipeline__minus__'+key]=delta('pipeline',key)
        if 'self_schedule' in chosen and 'fixed500' in chosen:pairs['self_schedule__minus__fixed500']=delta('self_schedule','fixed500')
        if all(k in chosen for k in ('pipeline','all_S500','all_Sprog')):pairs['pipeline__minus__best_human_schedule']=np.minimum(delta('pipeline','all_S500'),delta('pipeline','all_Sprog'))
        return rates,pairs
    point,pairs=draw(*point_indices(data['types'],ns,type_equal));rng=np.random.default_rng(202610193201);samples=[];contrasts={k:[] for k in pairs}
    for start in range(0,replicates,32):
        value,p=draw(*sample_indices(data['types'],ns,rng,min(32,replicates-start),type_equal));samples.append(value)
        for k in contrasts:contrasts[k].append(p[k])
    samples=np.concatenate(samples);report={}
    for i,key in enumerate(keys):
        value=point[0,i]*100;ci=[float(np.quantile(samples[:,i,0],.025)*100),float(np.quantile(samples[:,i,1],.975)*100)]
        report[key]=dict(success_lower_pct=float(value[0]),success_upper_pct=float(value[1]),success_CI95_pct=ci,collision_pct=float(value[2]),timeout_pct=float(value[3]),unknown_pct=float(value[4]),evaluated_action_crossfit_regret_pp=None,evaluated_action_in_sample_regret_pp=None,max_controller_regret_pp=None,harm_possible=None,harm_certain=None,selected_non_diagonal_fraction=None,reason='Paired safety/oracle metrics unavailable where exact historical reference coverage is incomplete')
    comparisons={}
    for k,p in pairs.items():
        bs=np.concatenate(contrasts[k])*100;comparisons[k]=dict(gain_pp=float(p[0,0]*100),unknown_bounds_pp=(p[0,1:]*100).tolist(),CI95_pp=[float(np.quantile(bs[:,1],.025)),float(np.quantile(bs[:,2],.975))])
    return dict(methods=report,paired_comparisons=comparisons,unavailable_methods=missing,controllers=nc,types=sorted(set(data['types'].tolist())),type_equal=type_equal,replicates=replicates,bootstrap_seed=202610193201,unknown_preserved=True,missing_not_failure=True)
