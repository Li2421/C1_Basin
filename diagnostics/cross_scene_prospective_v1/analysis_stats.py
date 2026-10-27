"""Prespecified prospective inference; this file is frozen before any test rollout."""
from common import *
import numpy as np
from scipy.stats import wilcoxon,binomtest

B=5000
SEED=2026102513
METHODS=('knn_dedup','ridge_dedup','knn_full','ridge_full','global_fixed','eta13','safety')

def interval(a):return np.percentile(a,[2.5,97.5]).tolist()

def bootstrap(tables,names,seed=SEED):
    rng=np.random.default_rng(seed);draws=np.empty((B,len(METHODS)));upper=np.empty_like(draws)
    for b in range(B):
        vals=[];ups=[]
        for i in rng.integers(len(names),size=len(names)):
            row=tables[names[i]];y=np.asarray(row['method_y'],float);u=np.asarray(row['method_unknown'],float)
            C,S,M=y.shape;ct=rng.integers(C,size=C);st=rng.integers(S,size=(C,S))
            vals.append(y[ct[:,None],st].mean((0,1)))
            ups.append(np.minimum(1,y[ct[:,None],st]+u[ct[:,None],st]).mean((0,1)))
        draws[b]=np.mean(vals,axis=0);upper[b]=np.mean(ups,axis=0)
    rates=np.mean([np.asarray(tables[n]['method_y']).mean((0,1))for n in names],axis=0)
    urates=np.mean([np.minimum(1,np.asarray(tables[n]['method_y'])+tables[n]['method_unknown']).mean((0,1))for n in names],axis=0)
    return dict(scene_names=names,replicates=B,seed=seed,
        rates={m:dict(rate=float(rates[i]),ci=interval(draws[:,i]),unknown_success_upper=float(urates[i]),upper_ci=interval(upper[:,i]))for i,m in enumerate(METHODS)},
        comparisons={m:dict(estimate=float(rates[i]-rates[4]),ci=interval(draws[:,i]-draws[:,4]),
            unknown_success_upper_difference=float(urates[i]-urates[4]))for i,m in enumerate(METHODS[:4])})

def robust(values):
    d=np.asarray(values,float);nz=d[np.abs(d)>1e-14];positive=int((nz>0).sum())
    return dict(median=float(np.median(d)),trimmed_mean_one_per_tail=float(np.sort(d)[1:-1].mean()),
        wins=positive,losses=int((nz<0).sum()),ties=int(len(d)-len(nz)),
        wilcoxon_two_sided_p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)if len(nz)else 1.,
        sign_test_two_sided_p=float(binomtest(positive,len(nz),.5,alternative='two-sided').pvalue)if len(nz)else 1.)

def main():
    frozen=read(HERE/'MODEL_FROZEN.receipt.json')
    for p,h in frozen['files'].items():assert sha(p)==h,p
    tables=read(HERE/'TEST_LABEL_TABLES.json');names=list(tables)
    result=bootstrap(tables,names);result['strata']={
        'bottleneck':bootstrap(tables,[n for n in names if n.startswith('xsp_')],SEED+1),
        'crowded14':bootstrap(tables,[n for n in names if n=='lit_crowded_random_n14'],SEED+2)}
    per={}
    for n,t in tables.items():
        y=np.asarray(t['method_y']);rates=y.mean((0,1));q=np.asarray(t['y']).mean((0,1))
        per[n]=dict(rates=dict(zip(METHODS,rates.tolist())),evaluated_posthoc_best=t['action_keys'][int(q.argmax())],
            evaluated_posthoc_best_rate=float(q.max()),controllers=t['controllers'],states=t['state_indices'])
    result['per_scene']=per
    result['robust']={m:robust([per[n]['rates'][m]-per[n]['rates']['global_fixed']for n in names])for m in METHODS[:4]}
    choices=read(HERE/'FROZEN_CHOICES.json')
    result['different_knn_global_scenes']=sum(choices[n]['knn_dedup']!=choices[n]['global_fixed']for n in names)
    result['primary']=dict(result['comparisons']['knn_dedup'],
        verdict='PASSED'if len(names)==12 and result['comparisons']['knn_dedup']['ci'][0]>0 else 'NOT_PASSED',
        requires_twelve_eligible_scenes=True,actual_scene_count=len(names))
    result['scope']='在本修正族、本控制器集与本安全层下，对这12个新场景；瓶颈族修正对全局修正，不外推未测拓扑。'
    result['bootstrap']='Paired scene -> controller type -> state; unknown=failure, fixed-policy unknown-success upper separately; no refitting.'
    write(HERE/'STATISTICS.json',result)

if __name__=='__main__':main()
