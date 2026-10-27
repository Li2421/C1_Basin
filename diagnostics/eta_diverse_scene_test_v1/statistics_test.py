"""Frozen25 exhaustive coverage and paired half-split statistics; no models."""
from common import *
from design import NEW_NAMES,OLD_NAMES,CONTROL_IDS,ETA_KEYS,menu
from stages import read_result
from itertools import combinations
import numpy as np
import math

SEED=2026102611;P=200;B=5000

def combos(k):return np.asarray(list(combinations(range(25),k)),int)

def best_set(q,k):
    sets=combos(k);scores=np.max(q[:,sets],axis=-1).mean(0);i=int(np.argmax(scores))
    return sets[i],float(scores[i])

def prepare_statistics():
    old=read(HERE/'OLD_LABEL_TABLES.json');q=np.asarray([np.asarray(old[n]['y']).mean((0,1))for n in OLD_NAMES]);fixed={}
    common_denominator=math.lcm(*(3*len(old[n]['state_indices'])for n in OLD_NAMES))
    exact=np.asarray([np.rint(np.asarray(old[n]['y']).sum((0,1))).astype(np.int64)*(common_denominator//(3*len(old[n]['state_indices'])))for n in OLD_NAMES])
    for k in(1,2,3):
        selected,_=best_set(exact,k);score=float(q[:,selected].max(-1).mean());fixed[str(k)]=dict(indices=selected.tolist(),candidates=[menu()[i]for i in selected],existing_scene_mean_oracle_success=score,
            existing_scene_rates={n:float(q[j,selected].max())for j,n in enumerate(OLD_NAMES)},combinations_checked=len(combos(k)))
    write(HERE/'EXISTING_BEST_SETS.json',fixed)
    rng=np.random.default_rng(SEED);splits={}
    for n in list(NEW_NAMES)+list(OLD_NAMES):
        S=12 if n in NEW_NAMES else len(old[n]['state_indices']);order=np.argsort(rng.random((P,S)),axis=1,kind='stable')
        splits[n]=dict(train=order[:,:S//2].tolist(),score=order[:,S//2:].tolist(),states=S)
    write(HERE/'HALF_SPLITS.json',dict(seed=SEED,replicates=P,algorithm='Independent uniform random-key permutations, canonical scene order; same scoring half for every method in a scene.',scenes=splits,
        bootstrap_rng_state=rng.bit_generator.state))
    print('Old-only exact best sets and 200 split lists frozen.',flush=True)

def new_labels():
    chosen=read(HERE/'SELECTED_CONTROLLERS.json');scenes=read(HERE/'SCENES.json');out={};missing=[];lookup={}
    for c in read(HERE/'main_CASES.json'):lookup[(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],c['requested_action'])]=c
    for scene in scenes:
        n=scene['name']
        if not chosen[n]['test_applicable']:continue
        y=np.zeros((3,12,25));u=np.zeros_like(y);complete=np.ones_like(y,bool);ids=[]
        for ci,ctl in enumerate(CONTROL_IDS):
            for si,state in enumerate(scene['states']):
                for ai,key in enumerate(ETA_KEYS):
                    c=lookup[(n,ctl,state['index'],key)]
                    try:r=read_result('main',c)
                    except FileNotFoundError:complete[ci,si,ai]=False;missing.append(c['rollout_uid']);continue
                    y[ci,si,ai]=r['success'];u[ci,si,ai]=r['numerical_failure'];ids.append(c['rollout_uid'])
        out[n]=dict(controllers=CONTROL_IDS,state_indices=[s['index']for s in scene['states']],eta_keys=ETA_KEYS,y=y.tolist(),unknown=u.tolist(),complete=bool(complete.all()),completed_cells=int(complete.sum()),rollout_uids=ids)
    write(HERE/'NEW_LABEL_TABLES.json',out);write(HERE/'UNCOMPLETED_REQUESTS.json',dict(rollout_uids=missing,count=len(missing)))
    return out

def split_scores(row,split):
    counts=np.rint(np.asarray(row['y']).sum(0)).astype(np.int16);upper=np.asarray(row['y'])+np.asarray(row['unknown']);train=counts[np.asarray(split['train'])].sum(1,dtype=np.int16);test=counts[np.asarray(split['score'])].sum(1,dtype=np.int16)
    denom=3*(len(row['state_indices'])//2);best=train.argmax(1);own=test[np.arange(P),best]
    upper_scores=upper.sum(0)[np.asarray(split['score'])].sum(1)
    return dict(train=train,score=test,selected=best,own=own,denominator=denom,own_rate=float(own.sum(dtype=np.int64)/(P*denom)),
        own_unknown_success_upper=float(upper_scores[np.arange(P),best].sum()/(P*denom)),choice_counts=np.bincount(best,minlength=25).tolist())

def bootstrap_gaps(counts,fixed,rng):
    S,A=counts.shape;half=S//2;denom=3*half*P;primary=np.empty((B,3));secondary=np.empty_like(primary)
    for start in range(0,B,100):
        count=min(100,B-start);ids=rng.integers(S,size=(count,S));sample=counts[ids]
        order=np.argsort(rng.random((count,P,S)),axis=-1,kind='stable')[:,:,:half]
        train=sample[np.arange(count)[:,None,None],order,:].sum(2,dtype=np.int16);score=sample.sum(1,dtype=np.int16)[:,None,:]-train
        best=train.argmax(-1);own=np.take_along_axis(score,best[:,:,None],axis=-1)[:,:,0]
        for k in(1,2,3):
            indices=np.asarray(fixed[str(k)]['indices']);pool=score[:,:,indices].max(-1);selected=indices[train[:,:,indices].argmax(-1)];selected_score=np.take_along_axis(score,selected[:,:,None],axis=-1)[:,:,0]
            primary[start:start+count,k-1]=(own-pool).sum(1,dtype=np.int64)/denom
            secondary[start:start+count,k-1]=(own-selected_score).sum(1,dtype=np.int64)/denom
    return primary,secondary

def coverage(names,points):
    available=[n for n in names if n in points];out={};qtrain=np.stack([points[n]['train']/points[n]['denominator']for n in available]);qscore=np.stack([points[n]['score']/points[n]['denominator']for n in available])
    own=np.mean([points[n]['own_rate']for n in available]);common_denominator=math.lcm(*(points[n]['denominator']for n in available))
    exact_train=np.stack([points[n]['train'].astype(np.int64)*(common_denominator//points[n]['denominator'])for n in available])
    for k in(1,2,3):
        sets=combos(k);selected_sets=[];oracle_rates=[];secondary_rates=[]
        for rep in range(P):
            values=np.max(exact_train[:,rep,sets],axis=-1).sum(0);chosen=sets[int(np.argmax(values))];selected_sets.append(chosen.tolist())
            oracle_rates.append(float(qscore[:,rep,chosen].max(-1).mean()))
            local=qtrain[:,rep,chosen].argmax(-1);selected=chosen[local];secondary_rates.append(float(qscore[np.arange(len(available)),rep,selected].mean()))
        pool=float(np.mean(oracle_rates));secondary=float(np.mean(secondary_rates));out[str(k)]=dict(own_best_crossvalidated_rate=float(own),pool_oracle_scoring_rate=pool,coverage_gap=float(own-pool),
            train_selected_pool_rate=secondary,train_selected_pool_gap=float(own-secondary),selected_sets_by_split=selected_sets)
    return dict(requested_scenes=names,available_scenes=available,missing_scenes=[n for n in names if n not in available],full_requested_group_available=len(available)==len(names),
        results=out,results_scope='Complete requested group'if len(available)==len(names)else'Available-scene subset only; full requested group cannot be calculated.')

def main():
    for p,h in read(HERE/'STATISTICS_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    new=new_labels();old=read(HERE/'OLD_LABEL_TABLES.json');rows={**old,**{n:r for n,r in new.items()if r['complete']}};fixed=read(HERE/'EXISTING_BEST_SETS.json');split=read(HERE/'HALF_SPLITS.json');points={n:split_scores(row,split['scenes'][n])for n,row in rows.items()}
    rng=np.random.default_rng(SEED);rng.bit_generator.state=split['bootstrap_rng_state'];new_results={};draws={}
    statuses=read(HERE/'CAPABILITY_TABLE.json')['scene_status']
    for n in NEW_NAMES:
        if n not in rows:
            new_results[n]=dict(status=statuses[n]['status']if not statuses[n]['test_applicable']else'主标签未完成',brings_new_need=None,judgable=False);continue
        row=rows[n];y=np.asarray(row['y']);u=np.asarray(row['unknown']);q=y.mean((0,1));qu=np.minimum(1,y+u).mean((0,1));order=np.lexsort((np.arange(25),-q));best=int(order[0]);p=points[n]
        pri,sec=bootstrap_gaps(np.rint(y.sum(0)).astype(np.int16),fixed,rng);draws[n+'__primary']=pri;draws[n+'__secondary']=sec;gaps={}
        for k in(1,2,3):
            indices=np.asarray(fixed[str(k)]['indices']);pool=p['score'][:,indices].max(1);local=indices[p['train'][:,indices].argmax(1)];secondary=p['score'][np.arange(P),local];d=P*p['denominator']
            gaps[str(k)]=dict(gap=float((p['own']-pool).sum(dtype=np.int64)/d),ci=np.percentile(pri[:,k-1],[2.5,97.5]).tolist(),own_best_crossvalidated_rate=p['own_rate'],pool_oracle_scoring_rate=float(pool.sum(dtype=np.int64)/d),
                train_selected_pool_gap=float((p['own']-secondary).sum(dtype=np.int64)/d),train_selected_pool_ci=np.percentile(sec[:,k-1],[2.5,97.5]).tolist(),
                pool_candidate_full_state_rates=[dict(key=ETA_KEYS[a],eta=menu()[a]['eta_a'],success_rate=float(q[a]),unknown_success_upper=float(qu[a]))for a in indices])
        judgable=.20<=float(q[best])<=.95;need=judgable and gaps['3']['gap']>=.10
        new_results[n]=dict(status='带来新需求'if need else'未达到新需求条件'if judgable else'本测试无法判断',judgable=judgable,brings_new_need=bool(need)if judgable else None,
            in_sample_best_rate=float(q[best]),in_sample_best_candidate=menu()[best],top3=[dict(candidate=menu()[a],rate=float(q[a]),unknown_success_upper=float(qu[a]))for a in order[:3]],
            candidate_rates=[dict(candidate=menu()[a],rate=float(q[a]),unknown_success_upper=float(qu[a]))for a in range(25)],own_best_crossvalidated_rate=p['own_rate'],own_best_unknown_success_upper=p['own_unknown_success_upper'],cv_choice_counts=p['choice_counts'],gaps=gaps,
            per_controller_best={ctl:dict(candidate=menu()[int(np.argmax(y[c].mean(0)))],rate=float(y[c].mean(0).max()))for c,ctl in enumerate(CONTROL_IDS)})
    np.savez_compressed(HERE/'BOOTSTRAP_GAPS.npz',**draws)
    eligible=[n for n in NEW_NAMES if n in points];cross={};cross_names=eligible+['xsc_swap_n6']
    for source in cross_names:
        cross[source]={}
        for target in cross_names:
            if source==target:continue
            a=points[source]['selected'];t=points[target];borrowed=t['score'][np.arange(P),a]
            cross[source][target]=dict(loss=float((t['own']-borrowed).sum(dtype=np.int64)/(P*t['denominator'])),target_own_best_rate=t['own_rate'],borrowed_rate=float(borrowed.sum(dtype=np.int64)/(P*t['denominator'])))
    groups=dict(new_plus_base=coverage(list(NEW_NAMES)+['xsc_swap_n6'],points),new_plus_existing11=coverage(list(NEW_NAMES)+list(OLD_NAMES),points))
    needs=[n for n,r in new_results.items()if r['brings_new_need']is True];pairs=[]
    for i,a in enumerate(needs):
        for b in needs[i+1:]:
            ab=cross[a][b]['loss'];ba=cross[b][a]['loss']
            if ab>=.10 and ba>=.10:pairs.append(dict(a=a,b=b,loss_a_to_b=ab,loss_b_to_a=ba))
    passed=len(needs)>=2 and bool(pairs)
    write(HERE/'STATISTICS.json',dict(new_scenes=new_results,existing_best_sets=fixed,cross_use_losses=cross,coverage=groups,bootstrap=dict(seed=SEED,replicates=B,inner_half_splits=P,rng_continues_after_frozen_point_splits=True),
        primary_pool_scoring='Oracle maximum on scoring half; user confirmed.',secondary_pool_scoring='Within-pool choice on training half, score on scoring half.',unknown_policy='Unknown=failure; candidate and fixed selected-own-strategy success upper reported without silently changing choices.'))
    write(HERE/'TEST_DECISION.json',dict(utc=utc(),meets_expansion_condition=passed,status='符合需求，值得扩展'if passed else'本轮未达到值得扩展条件',new_scenes=new_results,new_need_scenes=needs,qualifying_mutual_pairs=pairs,
        thresholds=dict(best_rate_min=.20,best_rate_max=.95,new_gap3_min=.10,both_cross_use_losses_min=.10,minimum_new_need_scenes=2),
        coverage_gap3={k:dict(full_requested_group_available=v['full_requested_group_available'],gap=v['results']['3']['coverage_gap']if v['full_requested_group_available']else None,
            available_subset_gap=v['results']['3']['coverage_gap'],missing_scenes=v['missing_scenes'])for k,v in groups.items()},
        no_additional_rollouts=True,no_second_stage=True,no_model_training=True,all_thresholds_use_unrounded_json=True))
    print('TEST_DECISION',passed,{n:(r['status'],r.get('gaps',{}).get('3',{}).get('gap'))for n,r in new_results.items()},flush=True)
if __name__=='__main__':main()
