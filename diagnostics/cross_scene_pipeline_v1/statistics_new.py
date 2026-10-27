"""Prespecified paired hierarchical estimates; fixed prospective fold choices."""
from common import *
from results import result
import numpy as np
from scipy.stats import wilcoxon,binomtest

METHODS=('pipeline','global_fixed','strong_global','knn','critic49','generator_mean','no_controller49','scene_only49','eta13','safety','in_sample49')
PAIRS={'pipeline_minus_global_fixed':('pipeline','global_fixed'),'pipeline_minus_strong_global':('pipeline','strong_global'),
    'pipeline_minus_knn':('pipeline','knn'),'critic49_minus_global_fixed':('critic49','global_fixed'),
    'generator_mean_minus_global_fixed':('generator_mean','global_fixed'),
    'critic49_minus_no_controller49':('critic49','no_controller49'),'critic49_minus_scene_only49':('critic49','scene_only49')}
B=5000
SEED=2026102517

def assemble():
    old=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/LABEL_TABLES.json');out={}
    refs=read(HERE/'main_REFERENCES.json');index={(r['fold'],r['scene'],r['controller'],r['state_index'],r['method']):r for r in refs}
    for fold in read(HERE/'FOLDS.json'):
        group=fold['kind']if fold['kind']!='lofo'else fold['name'];target=out.setdefault(group,{})
        for name,ch in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
            source=old[name];sy=np.array(source['y'],float);su=np.array(source['unknown'],float)
            assert source['controllers']==ch['controllers']and source['state_indices']==ch['state_indices']
            C,S,A=sy.shape;methods=METHODS if fold['generator']else ('critic49','strong_global');y=np.zeros((C,S,len(methods)));u=np.zeros_like(y)
            best=int(sy.mean((0,1)).argmax());keys=[a['key']for a in actions()]
            for mi,method in enumerate(methods):
                if method in ('pipeline','generator_mean'):
                    for ci,ctl in enumerate(ch['controllers']):
                        for si,st in enumerate(ch['state_indices']):
                            r=index[(fold['name'],name,ctl,st,method)]
                            if r['source']=='historical_49_label':
                                a=r['historical_action'];y[ci,si,mi]=sy[ci,si,a];u[ci,si,mi]=su[ci,si,a]
                            else:
                                raw=result('main',r['rollout_uid']);y[ci,si,mi]=int(raw['success']);u[ci,si,mi]=int(raw['numerical_failure'])
                else:
                    selected=np.full((C,S),best)if method=='in_sample49'else np.full((C,S),keys.index(ch[method]))if isinstance(ch[method],str)else np.array(ch[method])
                    y[:,:,mi]=np.take_along_axis(sy,selected[:,:,None],axis=2)[:,:,0];u[:,:,mi]=np.take_along_axis(su,selected[:,:,None],axis=2)[:,:,0]
            target[name]=dict(fold=fold['name'],controllers=ch['controllers'],state_indices=ch['state_indices'],methods=methods,
                y=y.tolist(),unknown=u.tolist(),same_pool_global_candidate=ch['global_fixed'],strong_global_candidate=ch['strong_global'],
                in_sample_best_candidate=keys[best],in_sample_best_scope='Fixed candidate from the historical49 only; optimistic descriptive reference.')
    write(HERE/'EVALUATION_LABEL_TABLES.json',out);return out

def robust(d):
    d=np.asarray(d,float);nz=d[np.abs(d)>1e-14];w=int((nz>0).sum());l=int((nz<0).sum())
    return dict(median=float(np.median(d)),trimmed_mean_drop_one_each_tail=float(np.sort(d)[1:-1].mean()),wins=w,losses=l,ties=int(len(d)-len(nz)),
        wilcoxon_two_sided_p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)if len(nz)else 1.,
        sign_test_two_sided_p=float(binomtest(w,len(nz),.5,alternative='two-sided').pvalue)if len(nz)else 1.)

def estimate(tables,pairs,seed):
    names=list(tables);methods=list(tables[names[0]]['methods']);mi={m:i for i,m in enumerate(methods)}
    arrays=[np.array(tables[n]['y'],float)for n in names];upper=[np.minimum(1,np.array(tables[n]['y'])+tables[n]['unknown'])for n in names]
    rates=np.array([a.mean((0,1))for a in arrays]);urates=np.array([a.mean((0,1))for a in upper])
    rng=np.random.default_rng(seed);samples=np.empty((B,len(methods)));ups=np.empty_like(samples)
    for b in range(B):
        vals=[];uv=[]
        for j in rng.integers(len(names),size=len(names)):
            a=arrays[j];C,S,M=a.shape;ct=rng.integers(C,size=C);st=rng.integers(S,size=(C,S))
            vals.append(a[ct[:,None],st].mean((0,1)));uv.append(upper[j][ct[:,None],st].mean((0,1)))
        samples[b]=np.mean(vals,axis=0);ups[b]=np.mean(uv,axis=0)
    result=dict(scene_names=names,replicates=B,seed=seed,
        rates={m:dict(rate=float(rates[:,i].mean()),ci=np.percentile(samples[:,i],[2.5,97.5]).tolist(),
            unknown_success_upper=float(urates[:,i].mean()),upper_ci=np.percentile(ups[:,i],[2.5,97.5]).tolist())for i,m in enumerate(methods)},
        per_scene={n:dict(rates=dict(zip(methods,rates[i].tolist())),unknown_success_upper=dict(zip(methods,urates[i].tolist())),
            controllers=tables[n]['controllers'],state_indices=tables[n]['state_indices'],same_pool_global_candidate=tables[n]['same_pool_global_candidate'],
            strong_global_candidate=tables[n]['strong_global_candidate'])for i,n in enumerate(names)},comparisons={})
    for key,(a,b)in pairs.items():
        i,j=mi[a],mi[b];d=rates[:,i]-rates[:,j]
        result['comparisons'][key]=dict(estimate=float(d.mean()),ci=np.percentile(samples[:,i]-samples[:,j],[2.5,97.5]).tolist(),
            per_scene=dict(zip(names,d.tolist())),both_policies_unknown_as_success_difference=float((urates[:,i]-urates[:,j]).mean()),robust=robust(d))
    return result

def diversity():
    rows={}
    for f in read(HERE/'FOLDS.json'):
        if f['kind']!='loso':continue
        for name,ch in read(HERE/'folds'/f['name']/'CHOICES.json').items():
            chosen=np.array(ch['pipeline']);C,S=chosen.shape;identity=np.empty((C,S),object)
            for c in range(C):
                for s in range(S):identity[c,s]=etahex(ch['candidate_etas'][s][chosen[c,s]])
            ii,jj=np.triu_indices(S,1);ci,cj=np.triu_indices(C,1)
            rows[name]=dict(generated_proposal_fraction=float((chosen>=49).mean()),
                distinct_eta_across_states_controller_fraction=float(np.mean([len(set(identity[c]))>1 for c in range(C)])),
                different_eta_state_pair_fraction=float(np.mean(identity[:,ii]!=identity[:,jj])),
                distinct_eta_across_types_state_fraction=float(np.mean([len(set(identity[:,s]))>1 for s in range(S)])),
                different_eta_type_pair_fraction=float(np.mean(identity[ci]!=identity[cj]))if len(ci)else None,
                different_candidate_index_state_pair_fraction=float(np.mean(chosen[:,ii]!=chosen[:,jj])))
    return dict(per_scene=rows,macro={key:float(np.mean([r[key]for r in rows.values()if r[key]is not None]))for key in next(iter(rows.values()))})

def main():
    for p,h in read(HERE/'STATISTICS_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    data=assemble();main=estimate(data['loso'],PAIRS,SEED)
    key='pipeline_minus_global_fixed';primary=dict(main['comparisons'][key],verdict='PASSED'if main['comparisons'][key]['ci'][0]>0 else 'NOT_PASSED')
    assert len(data['loso'])==11
    lofo={k:estimate(data[k],PAIRS,SEED+i+1)for i,k in enumerate(('lofo__bottleneck','lofo__literature'))}
    full=estimate(data['full18'],{'critic49_minus_previous_global':('critic49','strong_global')},SEED+3)
    write(HERE/'STATISTICS.json',dict(primary=primary,loso=main,lofo=lofo,full18=full,diversity=diversity(),
        scope='在本修正族、本控制器集与本安全层下，对这11个去重场景；去重是在看过18场景标签之后预先确定。',
        bootstrap='5000 paired draws: scenes -> controller types -> states; frozen fold fits/choices, no refitting; unknown=failure and fixed-policy success upper separately.'))
    print('PRIMARY',primary['verdict'],primary['estimate'],primary['ci'],flush=True)

if __name__=='__main__':main()
